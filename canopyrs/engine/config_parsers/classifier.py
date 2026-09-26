from pydantic import Field
from typing import Optional, List

from canopyrs.engine.config_parsers.base import BaseConfig

class ClassifierConfig(BaseConfig):
    """Configuration for classifier models"""
    model: str = Field(..., description="Model name (resnet, swin)")
    architecture: str = Field(..., description="Model architecture variant (resnet50, swin-base, etc)")
    num_classes: int = Field(..., description="Number of output classes")
    pretrained: bool = Field(True, description="Whether to use pretrained weights")
    checkpoint_path: Optional[str] = Field(None, description="Path to model checkpoint")
    freeze_layers: int = Field(0, description="Number of layers to freeze")
    batch_size: int = Field(8, description="Batch size for inference")
    class_names: Optional[List[str]] = Field(None, description="Names of output classes")
    norm_mean: List[float] = Field(
        [0.485, 0.456, 0.406],
        description=(
            "Per-channel (RGB) mean subtracted from the 0..1 image before the backbone. "
            "Must match training: forest-ssl uses ImageNet stats (the default)."
        ),
    )
    norm_std: List[float] = Field(
        [0.229, 0.224, 0.225],
        description="Per-channel (RGB) std the 0..1 image is divided by (after norm_mean). ImageNet default.",
    )
    n_cls_layers: int = Field(
        1,
        description=(
            "ViT/timm classifiers only: number of final transformer blocks "
            "whose CLS tokens are concatenated to form the head input "
            "(head dim = n_cls_layers * embed_dim). Must match the value "
            "used at training time for the checkpoint to load. Ignored by "
            "torchvision-based classifiers (resnet, swin)."
        ),
    )
