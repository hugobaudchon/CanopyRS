from pydantic import Field

from engine.config_parsers.base import BaseConfig


class TilerizerConfig(BaseConfig):
    tile_type: str = 'tile'
    tile_size: int = 1024
    tile_overlap: float = 0.5
    ground_resolution: float = None
    scale_factor: float = None
    use_variable_tile_size: bool = False
    variable_tile_size_pixel_buffer: int = 5
    ignore_black_white_alpha_tiles_threshold: float = 0.75
    coco_n_workers: int = 5

    ignore_tiles_without_labels: bool = True    # impacts inference and evaluation!
    min_intersection_ratio: float = 0.4     # impacts evaluation

    main_label_category_column_name: str = None
    other_labels_attributes_column_names: list = Field(default_factory=list)

    use_variable_tile_size: bool = False
    variable_tile_size_pixel_buffer: int or None = None
