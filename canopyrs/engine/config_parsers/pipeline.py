from pathlib import Path
from typing import List

import yaml

from canopyrs.engine.config_parsers.tilerizer import TilerizerConfig
from canopyrs.engine.config_parsers.detector import DetectorConfig
from canopyrs.engine.config_parsers.aggregator import AggregatorConfig
from canopyrs.engine.config_parsers.segmenter import SegmenterConfig
from canopyrs.engine.config_parsers.classifier import ClassifierConfig

from canopyrs.engine.config_parsers.base import BaseConfig, get_config_path

CONFIG_CLASS_BY_KIND = {
    'tilerizer': TilerizerConfig,
    'detector': DetectorConfig,
    'aggregator': AggregatorConfig,
    'segmenter': SegmenterConfig,
    'classifier': ClassifierConfig,
}


class PipelineConfig(BaseConfig):
    components_configs: List[tuple[str, BaseConfig]]

    @classmethod
    def from_yaml(cls, path: str or Path) -> 'PipelineConfig':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        components_configs = []
        for step in data['components_configs']:
            (component_type, config_data), = step.items()   # one {kind: config} mapping per step
            if component_type not in CONFIG_CLASS_BY_KIND:
                raise ValueError(f'Invalid component {step}')
            component_cls = CONFIG_CLASS_BY_KIND[component_type]

            if isinstance(config_data, str):
                component_config = component_cls.from_yaml(get_config_path(config_data))
            elif isinstance(config_data, dict):
                component_config = component_cls(**config_data)
            else:
                raise ValueError(f'Invalid config data for component type {component_type}: {config_data}')

            components_configs.append((component_type, component_config))

        return cls(components_configs=components_configs)
