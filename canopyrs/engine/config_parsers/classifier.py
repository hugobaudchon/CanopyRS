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
