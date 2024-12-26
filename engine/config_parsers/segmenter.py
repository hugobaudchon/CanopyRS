from engine.config_parsers.base import BaseConfig


class SegmenterConfig(BaseConfig):
    # General model definition
    model: str = 'sam2'
    backbone: str = 'l'
    checkpoint_path: str = None
    simplify_tolerance: float = 1.0
    box_padding_percentage: float = 0.00
    n_postprocess_workers: int = 4
    box_batch_size: int = 250
