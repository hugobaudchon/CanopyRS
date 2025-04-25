from pathlib import Path
from typing import List

import yaml

from engine.config_parsers.tilerizer import TilerizerConfig
from engine.config_parsers.detector import DetectorConfig
from engine.config_parsers.aggregator import AggregatorConfig
from engine.config_parsers.segmenter import SegmenterConfig
from engine.config_parsers.classifier import ClassifierConfig
from engine.config_parsers.rubisco_db_writer import RubiscoDbWriterConfig

from engine.config_parsers.base import BaseConfig, get_config_path


class PipelineConfig(BaseConfig):
    components_configs: List[tuple[str, BaseConfig]]

    @classmethod
    def from_yaml(cls, path: str or Path) -> 'PipelineConfig':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        components_configs = []
        for component_config in data['components_configs']:
            component_type = list(component_config.keys())[0]
            config_name = list(component_config.values())[0]
            config_path = get_config_path(config_name)

            if component_type == 'tilerizer':
                component_config = TilerizerConfig.from_yaml(config_path)
            elif component_type == 'detector':
                component_config = DetectorConfig.from_yaml(config_path)
            elif component_type == 'aggregator':
                component_config = AggregatorConfig.from_yaml(config_path)
            elif component_type == 'segmenter':
                component_config = SegmenterConfig.from_yaml(config_path)
            elif component_type == 'classifier':
                component_config = ClassifierConfig.from_yaml(config_path)
            elif component_type == 'rubisco_db_writer':
                component_config = RubiscoDbWriterConfig.from_yaml(config_path)
            else:
                raise ValueError(f'Invalid component {component_config}')

            components_configs.append((component_type, component_config))

        return cls(components_configs=components_configs)
