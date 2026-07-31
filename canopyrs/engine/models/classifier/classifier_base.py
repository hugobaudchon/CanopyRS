from abc import ABC, abstractmethod
from typing import Dict

import torch

from canopyrs.engine.config_parsers import ClassifierConfig
from canopyrs.engine.loader import InferTimer
from canopyrs.engine.models.utils import load_state_dict_with_key_repair


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

    def infer(self, loader):
        """Consume a ``tile_loader``, iterated as ``(object_ids, images)`` batches, and return aligned
        ``(object_ids, class_predictions, class_scores)`` — one predicted class index and one full
        per-class score list per tile. Reuses ``forward``; builds no DataLoader of its own."""
        self.model.eval()
        object_ids, class_predictions, class_scores = [], [], []
        timer = InferTimer("Inferring classifier...")
        with torch.no_grad():
            for batch_ids, images in timer.batches(loader):
                images = torch.stack([img for img in images]).to(self.device)
                timer.mark("prep")
                outputs = self.forward(images)
                timer.mark("gpu")
                for out in outputs:
                    class_scores.append(out['scores'].cpu().numpy().tolist())
                    class_predictions.append(out['labels'].cpu().item())
                object_ids.extend(batch_ids)
                timer.mark("post")
        timer.report()
        return object_ids, class_predictions, class_scores

    def load_checkpoint(self, checkpoint_path):
        """Load model weights from a checkpoint file.

        Resolves Hugging Face URLs and falls back to key-renaming on a key
        mismatch. See ``models.utils.load_state_dict_with_key_repair``.

        Args:
            checkpoint_path: Path to the checkpoint file
        """
        load_state_dict_with_key_repair(self.model, checkpoint_path)
