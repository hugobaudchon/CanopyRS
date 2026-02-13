from pydantic import Field, validator
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
    categories_config_path: Optional[str] = Field(
        None,
        description="Path to canonical categories JSON (benchmark only)",
    )
    pipeline_outputs_root: Optional[str] = Field(
        None,
        description=(
            "Root folder of existing pipeline outputs"
            " (benchmark only)"
        ),
    )
    tilerizer_config_path: Optional[str] = Field(
        None,
        description=(
            "Path to tilerizer YAML config for re-tilerizing GT"
            " before classification (benchmark only)"
        ),
    )
    input_imagery: Optional[str] = Field(
        None,
        description=(
            "Path to raw raster imagery, required when"
            " tilerizer_config_path is set (benchmark only)"
        ),
    )

    @validator('model')
    def validate_model(cls, v):
        valid_models = ['resnet', 'swin']
        if v not in valid_models:
            raise ValueError(f"Model must be one of {valid_models}")
        return v

    @validator('architecture')
    def validate_architecture(cls, v, values):
        model = values.get('model')
        if model == 'resnet':
            valid_archs = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
            if v not in valid_archs:
                raise ValueError(f"For ResNet, architecture must be one of {valid_archs}")
        elif model == 'swin':
            valid_types = ['tiny', 'small', 'base', 'large']
            if not any(t in v for t in valid_types):
                raise ValueError(f"For Swin, architecture must contain one of {valid_types}")
        return v
