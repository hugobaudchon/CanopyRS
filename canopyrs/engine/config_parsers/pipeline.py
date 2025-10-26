from pathlib import Path
from typing import List

import yaml

from canopyrs.engine.config_parsers.tilerizer import TilerizerConfig
from canopyrs.engine.config_parsers.detector import DetectorConfig
from canopyrs.engine.config_parsers.aggregator import AggregatorConfig
from canopyrs.engine.config_parsers.segmenter import SegmenterConfig
from canopyrs.engine.config_parsers.classifier import ClassifierConfig

from canopyrs.engine.config_parsers.base import BaseConfig, get_config_path


class PipelineConfig(BaseConfig):
    components_configs: List[tuple[str, BaseConfig]]

    @classmethod
    def from_yaml(cls, path: str or Path) -> 'PipelineConfig':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        components_configs = []
        for component_config in data['components_configs']:
            component_type = list(component_config.keys())[0]
            config_data = list(component_config.values())[0]

            if component_type == 'tilerizer':
                component_cls = TilerizerConfig
            elif component_type == 'detector':
                component_cls = DetectorConfig
            elif component_type == 'aggregator':
                component_cls = AggregatorConfig
            elif component_type == 'segmenter':
                component_cls = SegmenterConfig
            elif component_type == 'classifier':
                component_cls = ClassifierConfig
            else:
                raise ValueError(f'Invalid component {component_config}')

            if isinstance(config_data, str):
                config_name = list(component_config.values())[0]
                config_path = get_config_path(config_name)
                component_config = component_cls.from_yaml(config_path)
            elif isinstance(config_data, dict):
                component_config = component_cls(**config_data)
            else:
                raise ValueError(f'Invalid config data for component type {component_type}: {config_data}')

            components_configs.append((component_type, component_config))

        return cls(components_configs=components_configs)
