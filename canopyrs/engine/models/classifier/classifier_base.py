from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Tuple, Dict

import numpy as np
import torch
from geodataset.dataset import InstanceSegmentationLabeledRasterCocoDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from canopyrs.engine.config_parsers import ClassifierConfig
from huggingface_hub import hf_hub_download

from torchmetrics import F1Score


class ClassifierWrapperBase(ABC):
    """Base class for all classifier model wrappers"""

    def __init__(self, config: ClassifierConfig):
        """
        Initialize the classifier wrapper.

        Args:
            config: Configuration object containing model parameters
        """
        self.config = config
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = None

    @abstractmethod
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform forward pass through the model.

        Args:
            images: Batch of images (N, C, H, W)

        Returns:
            Dict containing 'scores' and 'labels' tensors
        """
        pass

    def _infer(self, data_loader: DataLoader) -> Tuple[List[Dict[str, torch.Tensor]], List]:
        """
        Run inference on a full dataset.

        Args:
            data_loader: DataLoader yielding batches of images

        Returns:
            List of prediction dictionaries
        """
        self.model.eval()
        all_predictions = []
        all_object_ids = []

        with torch.no_grad():
            data_loader_with_progress = tqdm(data_loader,
                                             desc="Inferring classifier...",
                                             leave=True)
            for batch in data_loader_with_progress:
                # Handle different types of batch outputs from the dataloader
                polygon_ids_batch = None
                if isinstance(batch, tuple) and len(batch) == 3:
                    # (images, labels_gt, polygon_ids)
                    # GT labels are ignored during inference
                    images, _, polygon_ids_batch = batch
                    if isinstance(polygon_ids_batch[0], list):  # Flatten if needed
                        polygon_ids_batch = [item[0] for item in polygon_ids_batch]
                elif isinstance(batch, tuple) and len(batch) == 2:
                    # (images, labels_gt) or (images, polygon_ids)
                    images, second_item = batch
                    if all(isinstance(item, (int, str, type(None))) for item in second_item): # Check if it looks like IDs
                        polygon_ids_batch = second_item
                    # else: it's labels_gt, polygon_ids_batch remains None
                else:
                    images = batch  # Just images

                if type(images) is not torch.Tensor:
                    # Convert images to tensor if they are not already
                    if isinstance(images, list):
                        images = torch.tensor(np.array(images), dtype=torch.float32)
                    else:
                        images = torch.stack(images)

                images = images.to(self.device)

                outputs = self.forward(images)
                all_predictions.extend(outputs)

                all_object_ids.extend(polygon_ids_batch)

        return all_predictions, all_object_ids

    def infer(self, infer_ds: InstanceSegmentationLabeledRasterCocoDataset, collate_fn_classification):
        """
        Run inference on a dataset and return predictions along with object IDs when available.

        Args:
            infer_ds: The dataset to run inference on
            collate_fn_classification: Collate function for batching

        Returns:
            A tuple of (tiles_paths, class_scores, class_predictions) or
            (tiles_paths, class_scores, class_predictions, object_ids_from_dl) if object IDs are available
        """

        infer_dl = DataLoader(infer_ds, batch_size=self.config.batch_size, shuffle=False,
                              collate_fn=collate_fn_classification,
                              num_workers=3, persistent_workers=True)

        predictions, object_ids_from_dl = self._infer(infer_dl)

        # Process results
        class_scores = [result['scores'].cpu().numpy().tolist()
                        for result in predictions]
        class_predictions = [result['labels'].cpu().item()
                             for result in predictions]

        # Extract tile paths - use the correct attribute based on dataset implementation
        if hasattr(infer_ds, 'tiles') and isinstance(infer_ds.tiles, dict):
            tiles_paths = [infer_ds.tiles[i]['path'] for i in range(len(infer_ds.tiles))]
        elif hasattr(infer_ds, 'tile_paths'):
            tiles_paths = infer_ds.tile_paths
        else:
            raise AttributeError("Dataset does not have recognized tile paths attribute")

        return tiles_paths, class_scores, class_predictions, object_ids_from_dl


class TorchTrainerClassifierWrapperBase(ClassifierWrapperBase):
    """Extended base class for classifiers with training capability"""

    def __init__(self, config: ClassifierConfig):
        """Initialize the trainable classifier wrapper"""
        super().__init__(config)

        # Add metrics for evaluation
        # TODO: add f1, accuracy, recall, precision

    def load_checkpoint(self, checkpoint_path):
        """
        Load model weights from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file
        """
        if not checkpoint_path:
            return

        checkpoint_path = Path(checkpoint_path)
        if 'huggingface.co' in checkpoint_path.parts:   # TODO merge this with the one in DetectorWrapperBase as a single function
            # Handle HuggingFace model loading
            if "huggingface.co" not in checkpoint_path.as_posix():
                raise ValueError("The provided Path does not contain a valid Hugging Face URL.")

            path = Path(str(checkpoint_path).replace("\\", "/").split("huggingface.co/")[-1])
            if "resolve" not in path.parts:
                raise ValueError("The provided Path is not in the expected Hugging Face format.")

            repo_id = "/".join(path.parts[:2])
            filename = path.name
            checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)

        # Load the checkpoint
        try:
            checkpoint = torch.load(checkpoint_path)
            # if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            #     self.model.load_state_dict(checkpoint['model_state_dict'])
            # elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            #     self.model.load_state_dict(checkpoint['model'])
            # else:
            self.model.load_state_dict(checkpoint)
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
        except RuntimeError:
            print("Error loading checkpoint, will try to rename state dict keys.")
            state_dict = try_rename_state_dict_keys_with_model(checkpoint_path)
            self.model.load_state_dict(state_dict)
            print("Succeed to load checkpoint by modifying keys!")

    def _evaluate(self, data_loader, epoch=None):
        """Run evaluation on validation data"""
        # TODO: include all metrics with torchmetrics
        # F1, accuracy, recall, precision
        self.model.eval()
        f1_metric = F1Score(task="multiclass",
                            num_classes=self.num_classes,
                            average='micro',
                            multidim_average="global").to(self.config.device)
        all_preds = []
        all_targets = []

        with torch.no_grad():
            desc = f"Epoch {epoch + 1} (scoring)" if epoch is not None else "Scoring"
            for images, targets in tqdm(data_loader, desc=desc, leave=True):
                # Move data to device
                if isinstance(images, list):
                    images = [img.to(self.device) for img in images]
                else:
                    images = images.to(self.device)

                targets = targets.to(self.device)

                # Run forward pass
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)

                all_preds.append(predicted.cpu())
                all_targets.append(targets.cpu())

        # Calculate F1 score
        all_preds = torch.stack(all_preds)
        all_targets = torch.stack(all_targets)
        f1_score = f1_metric(all_preds, all_targets)

        return {"f1": f1_score}, all_preds, all_targets


def try_rename_state_dict_keys_with_model(checkpoint_state_dict_path):
    # Structure the OrderedDict keys to match requirements
    checkpoint = torch.load(checkpoint_state_dict_path, weights_only=True)
    if "model" in checkpoint.keys():
        # Case where other attributes are stored in the checkpoint
        checkpoint = checkpoint["model"]
    # Create a new OrderedDict with the keys prefixed with "model."
    new_state_dict = OrderedDict()
    if all(s.startswith("model.") for s in checkpoint.keys()):
        # try removing the 'model.' prefix
        for key, value in checkpoint.items():
            new_key = key[6:]
            new_state_dict[new_key] = value
    elif all(s.startswith("module.") for s in checkpoint.keys()):
        # try removing the 'model.' prefix
        for key, value in checkpoint.items():
            new_key = key[7:]
            new_state_dict[new_key] = value
    else:
        # try adding the 'model.' prefix
        for key, value in checkpoint.items():
            new_key = 'model.' + key  # Prefix "model." to each key
            new_state_dict[new_key] = value
    return new_state_dict
