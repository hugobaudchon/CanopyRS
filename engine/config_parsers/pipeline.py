from typing import List

from engine.config_parsers.base import BaseConfig


class PipelineConfig(BaseConfig):
    components_configs: List[dict[str, str]]


