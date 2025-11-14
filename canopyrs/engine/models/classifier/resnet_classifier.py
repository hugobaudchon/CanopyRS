import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Union

from canopyrs.engine.config_parsers import ClassifierConfig
from canopyrs.engine.models.classifier.classifier_base import TorchTrainerClassifierWrapperBase
from canopyrs.engine.models.registry import CLASSIFIER_REGISTRY


@CLASSIFIER_REGISTRY.register('resnet')
class ResNetClassifier(TorchTrainerClassifierWrapperBase):
    """ResNet-based classifier implementation"""

    def __init__(self, config: ClassifierConfig):
        super().__init__(config)

        # Define model based on the specified architecture
        if self.config.architecture == "resnet18":
            base_model = models.resnet18(weights="IMAGENET1K_V2" if self.config.pretrained else None)
        elif self.config.architecture == "resnet34":
            base_model = models.resnet34(weights="IMAGENET1K_V2" if self.config.pretrained else None)
        elif self.config.architecture == "resnet50":
            base_model = models.resnet50(weights="IMAGENET1K_V2" if self.config.pretrained else None)
        elif self.config.architecture == "resnet101":
            base_model = models.resnet101(weights="IMAGENET1K_V2" if self.config.pretrained else None)
        elif self.config.architecture == "resnet152":
            base_model = models.resnet152(weights="IMAGENET1K_V2" if self.config.pretrained else None)
        else:
            raise ValueError(f"Unsupported ResNet architecture: {self.config.architecture}")

        if self.config.freeze_layers > 0:
            for i, child in enumerate(base_model.children()):
                if i < self.config.freeze_layers:
                    for param in child.parameters():
                        param.requires_grad = False

        # Replace the final fully connected layer
        num_features = base_model.fc.in_features
        base_model.fc = nn.Linear(num_features, self.config.num_classes)

        self.model = base_model
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

        # Forward pass
        outputs = self.model(images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Get predicted classes and convert to list of dictionaries
        values, indices = torch.max(probabilities, 1)

        # Create individual prediction dictionaries for each image
        result = []
        for i in range(len(indices)):
            result.append({
                'scores': probabilities[i],
                'labels': indices[i]
            })

        return result
