import timm
import torch
import torch.nn as nn
from typing import Dict, List, Union

from canopyrs.engine.config_parsers import ClassifierConfig
from canopyrs.engine.models.classifier.classifier_base import ClassifierWrapperBase
from canopyrs.engine.models.registry import CLASSIFIER_REGISTRY


def _has_cls_token(backbone):
    return hasattr(backbone, "cls_token")


def _get_blocks(backbone):
    return getattr(backbone, "blocks", None)


def extract_cls_features(backbone, x, n_cls_layers=2):
    """CLS token from the last n_cls_layers blocks, concatenated → (B, n*D)."""
    if n_cls_layers == 1:
        out = backbone.forward_features(x)       # (B, T, D)
        return out[:, 0] if _has_cls_token(backbone) else out.mean(dim=1)

    blocks = _get_blocks(backbone)
    captured = {}
    handles = [
        blocks[-(i + 1)].register_forward_hook(
            lambda m, inp, out, i=i: captured.__setitem__(i, out)
        )
        for i in range(n_cls_layers)
    ]
    try:
        backbone.forward_features(x)             # runs the blocks; hooks capture outputs
        has_cls = _has_cls_token(backbone)
        parts = []
        for i in range(n_cls_layers - 1, -1, -1):   # oldest → newest block
            f = captured[i]                          # (B, T, D)
            parts.append(f[:, 0] if has_cls else f.mean(dim=1))
        return torch.cat(parts, dim=-1)              # (B, n_cls_layers * D)
    finally:
        for h in handles:
            h.remove()


class ViTTimmClassifier(nn.Module):
    """timm ViT backbone (e.g. DINOv3 ViT-S) with a multi-layer CLS-token head."""

    def __init__(self, num_classes=14, n_cls_layers=2,
                 backbone_name="vit_small_patch16_dinov3.lvd1689m",
                 pretrained=False):
        super().__init__()
        # dynamic_img_size=True is REQUIRED — base model is 224/patch16, you run at 512
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                           num_classes=0, dynamic_img_size=True)
        self.n_cls_layers = n_cls_layers
        d = self.backbone.embed_dim                  # 384 for ViT-S
        self.head = nn.Sequential(                   # ← the classifier on top
            nn.LayerNorm(n_cls_layers * d),          # 768
            nn.Linear(n_cls_layers * d, num_classes) # 768 → 14
        )

    def forward(self, x):
        feats = extract_cls_features(self.backbone, x, self.n_cls_layers)
        return self.head(feats)


@CLASSIFIER_REGISTRY.register('vit_timm')
class ViTTimmClassifierWrapper(ClassifierWrapperBase):
    """CanopyRS wrapper around ViTTimmClassifier (timm-based ViT classifier).

    `config.architecture` is passed straight through as the timm backbone name
    (e.g. 'vit_small_patch16_dinov3.lvd1689m').
    """

    def __init__(self, config: ClassifierConfig):
        super().__init__(config)

        self.model = ViTTimmClassifier(
            num_classes=config.num_classes,
            n_cls_layers=config.n_cls_layers,
            backbone_name=config.architecture,
            # When a finetuned checkpoint is provided, don't fetch timm weights
            # (they'd just be overwritten). Set pretrained: false in YAML to skip.
            pretrained=config.pretrained and not config.checkpoint_path,
        )
        self.model.to(self.device)

        # Inputs arrive in 0..1 (the loader/dataset divides by 255); the backbone was trained on
        # ImageNet-normalized images, so we must apply the same mean/std here — otherwise the model
        # sees a distribution it never trained on and predictions are garbage.
        self.norm_mean = torch.tensor(config.norm_mean, device=self.device).view(1, 3, 1, 1)
        self.norm_std = torch.tensor(config.norm_std, device=self.device).view(1, 3, 1, 1)

        # Load checkpoint if provided
        if config.checkpoint_path:
            self.load_checkpoint(config.checkpoint_path)

    def forward(self, images: Union[torch.Tensor, List[torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """
        Run inference on a batch of images.

        Args:
            images: Batch of images as tensor or list of tensors

        Returns:
            List of dictionaries with 'scores' (probabilities) and 'labels' (predicted classes)
        """
        # Handle different input formats
        if isinstance(images, list):
            images = images[0] if len(images) == 1 else torch.stack(images)

        # 0..1 -> ImageNet-normalized, matching training (see __init__).
        images = images.to(self.device).float()
        images = (images - self.norm_mean) / self.norm_std

        # Forward pass
        outputs = self.model(images)
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
