from typing import Optional

from engine.config_parsers.base import BaseConfig


class RubiscoDbWriterConfig(BaseConfig):
    model_id_detector: int
    model_id_classifier: Optional[int] = None
    config_folder_path: str = "config/"  # Default path to CanopyRS/config/
    enabled: bool = False  # Disabled by default