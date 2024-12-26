from pathlib import Path

import geopandas as gpd

from config.default import default_path
from engine.components.aggregator import AggregatorComponent
from engine.components.detector import DetectorComponent
from engine.components.segmenter import SegmenterComponent
from engine.components.tilerizer import TilerizerComponent

from engine.config_parsers import (AggregatorConfig, ClassifierConfig, ClustererConfig, DetectorConfig,
                                   EmbedderConfig, InferIOConfig, PipelineConfig, SegmenterConfig, TilerizerConfig)
from engine.data_state import DataState


class Pipeline:
    def __init__(self, io_config: InferIOConfig, config: PipelineConfig):
        self.io_config = io_config
        self.config = config
        # self.validate_component_order()

        self.data_state = DataState(
            imagery_path=self.io_config.input_imagery,
            parent_output_path=self.io_config.output_folder,
            coco_paths={'infer': self.io_config.input_coco},
            results_gdf=gpd.read_file(self.io_config.input_gpkg) if self.io_config.input_gpkg else None
        )

    def run(self):
        for component_id, component_config in enumerate(self.config.components_configs):
            component = self._get_component(component_config, component_id)
            self.data_state = component.run(self.data_state)

    def _get_component(self, component_config, component_id):
        component_type = list(component_config.keys())[0]
        config_description = list(component_config.values())[0]
        if config_description.startswith('default'):
            config_path = Path(default_path) / f'{config_description.split("/")[1]}.yaml'
        else:
            config_path = config_description

        if component_type == 'tilerizer':
            component_config = TilerizerConfig.from_yaml(config_path)
            return TilerizerComponent(component_config, self.io_config.output_folder, component_id)
        elif component_type == 'detector':
            component_config = DetectorConfig.from_yaml(config_path)
            return DetectorComponent(component_config, self.io_config.output_folder, component_id)
        elif component_type == 'aggregator':
            component_config = AggregatorConfig.from_yaml(config_path)
            return AggregatorComponent(component_config, self.io_config.output_folder, component_id)
        elif component_type == 'segmenter':
            component_config = SegmenterConfig.from_yaml(config_path)
            return SegmenterComponent(component_config, self.io_config.output_folder, component_id)
        # elif isinstance(component_config, EmbedderConfig):
        #     return build_embedder()
        # elif isinstance(component_config, ClassifierConfig):
        #     return build_classifier()
        # elif isinstance(component_config, ClustererConfig):
        #     return build_clusterer()
        else:
            raise ValueError(f'Invalid component {component_config}')
