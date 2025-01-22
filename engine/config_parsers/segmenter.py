from engine.config_parsers.base import BaseConfig


class SegmenterConfig(BaseConfig):
    # General model definition
    model: str = 'sam2'
    backbone: str = 'l'
    checkpoint_path: str = None
    pp_simplify_tolerance: float = 1.0
    pp_remove_rings: bool = True
    pp_remove_small_geoms: float = 50
    box_padding_percentage: float = 0.00
    n_postprocess_workers: int = 4
    box_batch_size: int = 250
