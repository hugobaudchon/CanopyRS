import torch
import torch.nn as nn
from typing import Dict, List, Union
import torchvision.models as models

from canopyrs.engine.config_parsers import ClassifierConfig
from canopyrs.engine.models.classifier.classifier_base import TorchTrainerClassifierWrapperBase
from canopyrs.engine.models.registry import CLASSIFIER_REGISTRY


@CLASSIFIER_REGISTRY.register('swin')
class SwinClassifier(TorchTrainerClassifierWrapperBase):
    """Swin Transformer-based classifier implementation using PyTorch's native models"""

    # Mapping of architecture names to model functions
    MODEL_MAPPING = {
        # Swin V1 models
        'swin-tiny': models.swin_t,
        'swin-small': models.swin_s,
        'swin-base': models.swin_b,
        # Swin V2 models
        'swin-v2-tiny': models.swin_v2_t,
        'swin-v2-small': models.swin_v2_s,
        'swin-v2-base': models.swin_v2_b,
        'swin-v2-b': models.swin_v2_b,  # Alias for convenience
    }

    # Mapping of architecture to weights enum
    WEIGHTS_MAPPING = {
        # Swin V1 models
        'swin-tiny': models.Swin_T_Weights.DEFAULT,
        'swin-small': models.Swin_S_Weights.DEFAULT,
        'swin-base': models.Swin_B_Weights.DEFAULT,
        # Swin V2 models
        'swin-v2-tiny': models.Swin_V2_T_Weights.DEFAULT,
        'swin-v2-small': models.Swin_V2_S_Weights.DEFAULT,
        'swin-v2-base': models.Swin_V2_B_Weights.DEFAULT,
        'swin-v2-b': models.Swin_V2_B_Weights.DEFAULT,  # Alias
    }

    def __init__(self, config: ClassifierConfig):
        super().__init__(config)

        arch_key = self.config.architecture.lower()

        # Find the matching model based on architecture
        model_fn = None
        for key in self.MODEL_MAPPING:
            if key in arch_key:
                model_fn = self.MODEL_MAPPING[key]
                weights_enum = self.WEIGHTS_MAPPING[key] if self.config.pretrained else None
                break

        if model_fn is None:
            raise ValueError(f"Unsupported Swin architecture: {self.config.architecture}. "
                             f"Supported variants: {list(self.MODEL_MAPPING.keys())}")

        print(f"Loading Swin Transformer model: {arch_key}")

        # Initialize the model with pretrained weights if specified
        self.base_model = model_fn(weights=weights_enum)

        # Get the classifier head structure
        if hasattr(self.base_model, 'heads'):
            # Swin V2 models have a "heads" attribute with head layers
            num_features = self.base_model.heads.head.in_features
            self.base_model.heads.head = nn.Linear(num_features, self.config.num_classes)
        else:
            # Swin V1 models have a "head" attribute
            num_features = self.base_model.head.in_features
            self.base_model.head = nn.Linear(num_features, self.config.num_classes)

        # Freeze layers if specified
        if self.config.freeze_layers > 0:
            # The Swin model structure has features.0, features.1, etc. as blocks
            modules_to_freeze = min(self.config.freeze_layers, len(self.base_model.features))

            for i in range(modules_to_freeze):
                for param in self.base_model.features[i].parameters():
                    param.requires_grad = False

        self.model = self.base_model
        self.model.to(self.device)

        # Load checkpoint if provided
        if self.config.checkpoint_path:
            self.load_checkpoint(self.config.checkpoint_path)

    def forward(self, images: Union[torch.Tensor, List[torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """
        Run inference on batch of images.

        Args:
            images: Batch of images as tensor or list of tensors

        Returns:
            List of dictionaries with 'scores' (probabilities) and 'labels' (predicted classes)
        """
        # Handle different input formats
        if isinstance(images, list):
            if len(images) == 1:
                images = images[0]
            else:
                images = torch.stack(images)

        # Forward pass through the model
        outputs = self.model(images)

        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Get predicted classes
        values, indices = torch.max(probabilities, 1)

        # Create individual prediction dictionaries for each image
        result = []
        for i in range(len(indices)):
            result.append({
                'scores': probabilities[i],
                'labels': indices[i]
            })

        return result
