from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

import numpy as np
import torch
from geodataset.dataset import ClassificationLabeledRasterCocoDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from canopyrs.engine.config_parsers import ClassifierConfig
from canopyrs.engine.models.utils import load_state_dict_with_key_repair

# Only used by the commented-out training scaffolding (see _evaluate below).
# from torchmetrics import F1Score


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

    def infer(self, infer_ds: ClassificationLabeledRasterCocoDataset, collate_fn_classification):
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

    def infer_v2(self, loader):
        """v2 inference: consume a loader yielding ``(object_ids, images)`` batches and return aligned
        ``(object_ids, class_predictions, class_scores)`` — one predicted class index and one full
        per-class score list per tile. Reuses ``forward``; builds no DataLoader of its own."""
        self.model.eval()
        object_ids, class_predictions, class_scores = [], [], []
        with torch.no_grad():
            for batch_ids, images in tqdm(loader, desc="Inferring classifier...", leave=True):
                images = torch.stack([img for img in images]).to(self.device)
                for out in self.forward(images):
                    class_scores.append(out['scores'].cpu().numpy().tolist())
                    class_predictions.append(out['labels'].cpu().item())
                object_ids.extend(batch_ids)
        return object_ids, class_predictions, class_scores

    def load_checkpoint(self, checkpoint_path):
        """Load model weights from a checkpoint file.

        Resolves Hugging Face URLs and falls back to key-renaming on a key
        mismatch. See ``models.utils.load_state_dict_with_key_repair``.

        Args:
            checkpoint_path: Path to the checkpoint file
        """
        load_state_dict_with_key_repair(self.model, checkpoint_path)
