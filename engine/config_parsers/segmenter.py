from typing import Optional

from engine.config_parsers.base import BaseConfig


class SegmenterConfig(BaseConfig):
    # General model definition
    model: str = 'sam2'
    backbone: Optional[str] = 'l'
    checkpoint_path: str = None
    pp_n_workers: int = 8
    pp_down_scale_masks_px: Optional[int] = 512
    pp_simplify_tolerance: float = 0.0
    pp_remove_rings: bool = True
    pp_remove_small_geoms: float = 50
    box_padding_percentage: float = 0.00
    box_batch_size: int = 250
