from typing import Any, Optional

from canopyrs.engine.config_parsers.detector import DetectorConfig

class SegmenterConfig(DetectorConfig):
    # General model definition
    model: str = 'sam2'
    architecture: Optional[str] = 'l'
    checkpoint_path: Optional[str] = None
    image_batch_size: int = 1
    box_batch_size: Optional[int] = 250

    pp_n_workers: int = 8
    pp_down_scale_masks_px: Optional[int] = 512
    pp_simplify_tolerance: float = 0.0
    pp_remove_rings: bool = True
    pp_remove_small_geoms: float = 50

    box_padding_percentage: float = 0.00

    def model_post_init(self, __context: Any) -> None:                                  # TODO I should probably remove image_batch_size and rename it batch_size. But will have to adapt SAM configs etc..
        super().model_post_init(__context)
        if 'batch_size' in self.model_fields_set and self.batch_size is not None:
            self.image_batch_size = self.batch_size
