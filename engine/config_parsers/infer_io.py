from typing import Optional, Dict, Union

from pydantic import Field

from engine.config_parsers.base import BaseConfig


class InferIOConfig(BaseConfig):
    input_imagery: str
    output_folder: str

    input_gpkg: Optional[str] = None
    input_coco: Optional[str] = None

    ground_truth_gpkg: Optional[str] = None

    aoi_config: str = 'generate'
    aoi_type: Optional[str] = 'band'
    aoi: Union[str, dict] = Field(default_factory=lambda: {
            'percentage': 1.0,
            'position': 1
    })
