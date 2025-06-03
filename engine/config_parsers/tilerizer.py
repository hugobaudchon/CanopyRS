from pydantic import Field, validator
from typing import Optional, List

from engine.config_parsers.base import BaseConfig
from engine.utils import object_id_column_name

class TilerizerConfig(BaseConfig):
    tile_type: str = 'tile'
    tile_size: int = 1024
    tile_overlap: float = 0.5
    ground_resolution: Optional[float] = None
    scale_factor: Optional[float] = None
    use_variable_tile_size: bool = False
    variable_tile_size_pixel_buffer: int = 5
    ignore_black_white_alpha_tiles_threshold: float = 0.75
    coco_n_workers: int = 5

    ignore_tiles_without_labels: bool = True    # impacts inference and evaluation!
    min_intersection_ratio: float = 0.4     # impacts evaluation

    main_label_category_column_name: Optional[str] = None
    other_labels_attributes_column_names: list = Field(default_factory=list)

    use_variable_tile_size: bool = False
    variable_tile_size_pixel_buffer: int or None = None

    persistent_object_id_col: str = object_id_column_name

    @validator('other_labels_attributes_column_names', always=True)
    def ensure_persistent_id_in_other_attributes(cls, v, values):
        persistent_id_col = values.get('persistent_object_id_col')
        if persistent_id_col and persistent_id_col not in v:
            # Automatically add it if not present, or raise an error
            # print(f"Warning: '{persistent_id_col}' was not in other_labels_attributes_column_names. Adding it.")
            # v.append(persistent_id_col)
            # OR, more strictly:
            raise ValueError(
                f"The 'persistent_object_id_col' ('{persistent_id_col}') must be included in "
                f"'other_labels_attributes_column_names' if you want it in the COCO output."
            )
        return v
