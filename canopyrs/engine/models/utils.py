import random

import numpy as np
import torch
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
