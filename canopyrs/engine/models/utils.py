import random
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch.optim.lr_scheduler import StepLR

from canopyrs.engine.utils import object_id_column_name


def collate_fn_trivial(image_batch):
    return image_batch


def collate_fn_segmentation(batch):
    if type(batch[0][0]) is np.ndarray:
        data = np.array([item[0] for item in batch])
        data = torch.tensor(data, dtype=torch.float32)
    else:
        data = torch.tensor([item[0] for item in batch], dtype=torch.float32)

    for item in batch:
        item[1]['labels'] = [-1 if x is None else x for x in item[1]['labels']]

    labels = [{'masks': torch.tensor(np.array(item[1]['masks']), dtype=torch.int8),
               'labels': torch.tensor(np.array(item[1]['labels']), dtype=torch.int8),
               'area': torch.tensor(np.array(item[1]['area']).astype(np.int32), dtype=torch.float32),
               'iscrowd': torch.tensor(np.array(item[1]['iscrowd']), dtype=torch.bool),
               'image_id': torch.tensor(np.array(item[1]['image_id']), dtype=torch.int16)} for item in batch]

    if 'labels_polygons' in batch[0][1]:
        for i, item in enumerate(batch):
            labels[i]['labels_polygons'] = item[1]['labels_polygons']

    return data, labels


def collate_fn_detection(batch):
    if type(batch[0][0]) is np.ndarray:
        data = np.array([item[0] for item in batch])
        data = torch.tensor(data, dtype=torch.float32)
    else:
        data = torch.tensor([item[0] for item in batch], dtype=torch.float32)

    # For detection, we set all labels to 1, we don't care about the object class in our case
    for item in batch:
        item[1]['labels'] = [1 for _ in item[1]['labels']]

    labels = [{'boxes': torch.tensor(np.array(item[1]['boxes']), dtype=torch.float32),
               'labels': torch.tensor(np.array(item[1]['labels']), dtype=torch.long)} for item in batch]

    return data, labels


def collate_fn_infer_image_box(data_batch):
    image_batch = [data[0] for data in data_batch]
    boxes_batch = [np.array(data[1]['boxes']) for data in data_batch]
    boxes_object_ids = [data[1]['other_attributes'][object_id_column_name] for data in data_batch]
    return image_batch, boxes_batch, boxes_object_ids


def collate_fn_infer_image_masks(data_batch):
    image_batch = [data[0] for data in data_batch]
    masks_batch = [np.array(data[1]['masks']) for data in data_batch]
    masks_object_ids = [data[1]['other_attributes'][object_id_column_name] for data in data_batch]
    return image_batch, masks_batch, masks_object_ids


def collate_fn_images(batch):
    """
    Pad all images in the batch to (C, H_max, W_max) and stack.
    Works for lists of np.ndarray or torch.Tensor.
    """

    batch = [img if isinstance(img, np.ndarray) else img.numpy() for img in batch]

    C = batch[0].shape[0]
    H_max = max(img.shape[1] for img in batch)
    W_max = max(img.shape[2] for img in batch)

    stacked = np.zeros((len(batch), C, H_max, W_max), dtype=batch[0].dtype)

    for i, img in enumerate(batch):
        c, h, w = img.shape
        stacked[i, :, :h, :w] = img  # top-left pad

    final_tensor = torch.from_numpy(stacked).float()
    return final_tensor


def resolve_hf_checkpoint_path(checkpoint_path):
    """Resolve a checkpoint path that may point at a Hugging Face URL.

    If the path is a Hugging Face 'resolve' URL, download the file and return
    the local path. Otherwise return the path unchanged (as a Path).
    """
    checkpoint_path = Path(checkpoint_path)
    if 'huggingface.co' in checkpoint_path.parts:
        if "huggingface.co" not in checkpoint_path.as_posix():
            raise ValueError("The provided Path does not contain a valid Hugging Face URL.")
        # Strip everything up to and including 'huggingface.co/'
        path = Path(str(checkpoint_path).replace("\\", "/").split("huggingface.co/")[-1])
        if "resolve" not in path.parts:
            raise ValueError("The provided Path is not in the expected Hugging Face format.")
        repo_id = "/".join(path.parts[:2])
        filename = path.name
        checkpoint_path = Path(hf_hub_download(repo_id=repo_id, filename=filename))
    return checkpoint_path


def try_rename_state_dict_keys_with_model(checkpoint_state_dict_path):
    """Fallback when load_state_dict fails on key mismatch: normalize common
    prefixes by stripping 'model.'/'module.' or adding 'model.'.
    """
    checkpoint = torch.load(checkpoint_state_dict_path, weights_only=True)
    if "model" in checkpoint.keys():
        # Case where other attributes are stored in the checkpoint
        checkpoint = checkpoint["model"]
    new_state_dict = OrderedDict()
    if all(s.startswith("model.") for s in checkpoint.keys()):
        for key, value in checkpoint.items():
            new_state_dict[key[6:]] = value
    elif all(s.startswith("module.") for s in checkpoint.keys()):
        for key, value in checkpoint.items():
            new_state_dict[key[7:]] = value
    else:
        for key, value in checkpoint.items():
            new_state_dict['model.' + key] = value
    return new_state_dict


def load_state_dict_with_key_repair(model, checkpoint_path, weights_only=False, verbose=True):
    """Load a checkpoint into ``model``, repairing key prefixes on mismatch.

    Resolves Hugging Face URLs, then loads the state dict. If the keys don't
    match the model (RuntimeError), retries once after normalizing common key
    prefixes ('model.' / 'module.'). No-op if ``checkpoint_path`` is falsy.
    """
    if not checkpoint_path:
        return
    checkpoint_path = resolve_hf_checkpoint_path(checkpoint_path)
    try:
        model.load_state_dict(torch.load(checkpoint_path, weights_only=weights_only))
        if verbose:
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
    except RuntimeError:
        if verbose:
            print("Error loading checkpoint, will try to rename state dict keys.")
        state_dict = try_rename_state_dict_keys_with_model(checkpoint_path)
        model.load_state_dict(state_dict)
        if verbose:
            print("Succeeded to load checkpoint by modifying keys!")


def _download_checkpoint_from_hf_url(url, verbose=True):
    """Download a checkpoint from a full Hugging Face 'resolve' URL
    (https://huggingface.co/{repo_id}/resolve/{revision}/{filename}),
    honoring the revision. Returns the local Path, or None on failure.
    """
    pattern = r"https?://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+)"
    match = re.match(pattern, url)
    if not match:
        if verbose:
            print(f"  Could not parse HuggingFace URL: {url}")
        return None

    repo_id, revision, filename = match.group(1), match.group(2), match.group(3)
    if verbose:
        print(f"\n{'='*60}")
        print("Downloading checkpoint from HuggingFace:")
        print(f"  Repo: {repo_id}")
        print(f"  Revision: {revision}")
        print(f"  File: {filename}")
    try:
        local_path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
        if verbose:
            print(f"  Downloaded to: {local_path}")
        return Path(local_path)
    except Exception as e:
        if verbose:
            print(f"  Download failed: {e}")
        return None


def load_finetuned_checkpoint(model, checkpoint_path, strict=False, verbose=True):
    """Load a fine-tuned checkpoint into ``model``.

    - No-op if ``checkpoint_path`` is falsy (no fine-tuned weights requested;
      the model keeps its base pretrained weights).
    - Accepts a local path or a full Hugging Face 'resolve' URL.
    - Unwraps a 'model_state_dict' wrapper (full training checkpoint) if present.
    - Loads with the given ``strict`` flag.

    Raises if a checkpoint path is given but cannot be resolved or downloaded:
    a requested checkpoint that fails to load is an error, not a silent
    fallback to the base model.
    """
    if not checkpoint_path:
        return
    checkpoint_path_str = str(checkpoint_path)

    if checkpoint_path_str.startswith("https://huggingface.co/") or \
            checkpoint_path_str.startswith("http://huggingface.co/"):
        local_path = _download_checkpoint_from_hf_url(checkpoint_path_str, verbose=verbose)
        if local_path is None:
            raise ValueError(
                f"Failed to resolve/download checkpoint from Hugging Face URL: {checkpoint_path_str}"
            )
    else:
        local_path = Path(checkpoint_path_str)
        if not local_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path_str}")

    if verbose:
        print(f"\n{'='*60}")
        print("Loading fine-tuned checkpoint:")
        print(f"  Path: {local_path}")

    state_dict = torch.load(local_path, map_location='cpu')
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        model_state_dict = state_dict['model_state_dict']
        if verbose:
            print("  Checkpoint type: Full training checkpoint")
            if 'epoch' in state_dict:
                print(f"  Epoch: {state_dict['epoch']}")
    else:
        model_state_dict = state_dict
        if verbose:
            print("  Checkpoint type: Model weights only")

    model.load_state_dict(model_state_dict, strict=strict)
    if verbose:
        print("✓ Fine-tuned weights loaded successfully!")
        print(f"{'='*60}\n")


def set_all_seeds(seed: int):
    """
    Set random seeds for Python, NumPy, and PyTorch.
    Avoids forcing deterministic behavior to maintain performance.

    Args:
        seed (int): The seed value to use.
    """
    # Python's built-in random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Torch: for CPU and GPU operations
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print(f"Random seeds set to {seed}.")


class WarmupStepLR:
    def __init__(self, optimizer, step_size, gamma=0.1, warmup_steps=10, base_lr=1e-6):
        self.step_size = step_size
        self.gamma = gamma
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.max_lr = optimizer.param_groups[0]['lr']
        self.optimizer = optimizer
        self.scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        self.current_step = 0
        self.step()

    def step(self):
        # Warm-up phase
        if self.current_step < self.warmup_steps + 1:
            lr = self.base_lr + (self.max_lr - self.base_lr) * self.current_step / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        # StepLR phase
        else:
            self.scheduler.step()
        self.current_step += 1

    def get_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
