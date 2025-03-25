from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from engine.config_parsers import ClassifierConfig
from geodataset.dataset import ClassificationLabeledRasterCocoDataset
from huggingface_hub import hf_hub_download

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

    def _infer(self, data_loader: DataLoader) -> List[Dict[str, torch.Tensor]]:
        """
        Run inference on a full dataset.

        Args:
            data_loader: DataLoader yielding batches of images

        Returns:
            List of prediction dictionaries
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            data_loader_with_progress = tqdm(data_loader,
                                            desc="Inferring classifier...",
                                            leave=True)
            for images in data_loader_with_progress:
                if isinstance(images, list):
                    images = [img.to(self.device) for img in images]
                else:
                    images = images.to(self.device)

                outputs = self.forward(images)
                predictions.extend(outputs)

        return predictions

    def infer(self, infer_ds: ClassificationLabeledRasterCocoDataset, collate_fn: callable) -> Tuple[List[Path], List[List[float]], List[List[int]]]:
        """
        Run inference on a dataset and return class scores and predictions.

        Args:
            infer_ds: Dataset to perform inference on
            collate_fn: Function to collate dataset items into batches

        Returns:
            Tuple of (tile_paths, class_scores, class_predictions)
        """
        infer_dl = DataLoader(infer_ds, batch_size=self.config.batch_size, shuffle=False,
                             collate_fn=collate_fn,
                             num_workers=3, persistent_workers=True)

        results = self._infer(infer_dl)

        # Process results
        class_scores = [result['scores'].cpu().numpy().tolist() for result in results]
        class_predictions = [result['labels'].cpu().numpy().tolist() for result in results]
        tiles_paths = infer_ds.tile_paths

        return tiles_paths, class_scores, class_predictions

class TorchTrainerClassifierWrapperBase(ClassifierWrapperBase):
    """Extended base class for classifiers with training capability"""

    def __init__(self, config: ClassifierConfig):
        """Initialize the trainable classifier wrapper"""
        super().__init__(config)

        # Add metrics for evaluation
        # TODO: add f1, recall, precision
        self.accuracy_metric = torch.nn.functional.accuracy

    def load_checkpoint(self, checkpoint_path):
        """
        Load model weights from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file
        """
        if not checkpoint_path:
            return

        checkpoint_path = Path(checkpoint_path)
        if 'huggingface.co' in checkpoint_path.parts:
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
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    def _evaluate(self, data_loader, epoch=None):
        """Run evaluation on validation data"""
        # TODO: include all metrics with torchmetrics
        # F1, accuracy, recall, precision
        self.model.eval()
        total_correct = 0
        total_samples = 0
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
                
                # Accumulate metrics
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)
                
                all_preds.append(predicted.cpu())
                all_targets.append(targets.cpu())
        
        # Calculate accuracy
        accuracy = total_correct / total_samples
        
        # Combine all predictions and targets
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        return {"accuracy": accuracy}, all_preds, all_targets
