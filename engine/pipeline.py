from pathlib import Path

import geopandas as gpd

from engine.components.aggregator import AggregatorComponent
from engine.components.detector import DetectorComponent
from engine.components.evaluator import EvaluatorComponent
from engine.components.segmenter import SegmenterComponent
from engine.components.tilerizer import TilerizerComponent

from engine.config_parsers import PipelineConfig, InferIOConfig
from engine.data_state import DataState
from engine.utils import parse_tilerizer_aoi_config, infer_aoi_name, ground_truth_aoi_name, green_print, \
    clean_side_processes


class Pipeline:
    def __init__(self, io_config: InferIOConfig, config: PipelineConfig):
        self.io_config = io_config
        self.config = config
        self._validate_components_order()

        # Initialize data state from the io (input/output) config
        self.data_state = DataState(
            imagery_path=self.io_config.input_imagery,
            parent_output_path=self.io_config.output_folder,
            tiles_path=self.io_config.tiles_path,
            infer_coco_path=self.io_config.input_coco,
            infer_gdf=gpd.read_file(self.io_config.input_gpkg)
                      if self.io_config.input_gpkg else None,
            ground_truth_gdf=gpd.read_file(self.io_config.ground_truth_gpkg)
                             if self.io_config.ground_truth_gpkg else None,
        )

        # Initialize AOI configuration (Area of Interest, used by the Tilerizer)
        self.infer_aois_config = parse_tilerizer_aoi_config(
            aoi_config=self.io_config.aoi_config,
            aoi_type=self.io_config.aoi_type,
            aois={infer_aoi_name: self.io_config.aoi}
        )
        self.ground_truth_aoi_config = parse_tilerizer_aoi_config(
            aoi_config=self.io_config.aoi_config,
            aoi_type=self.io_config.aoi_type,
            aois={ground_truth_aoi_name: self.io_config.aoi}
        )

        green_print("Pipeline initialized")

    def __call__(self):
        # Run each component in the pipeline, sequentially
        for component_id, component_config_tuple in enumerate(self.config.components_configs):
            component = self._get_component(component_config_tuple, component_id)
            self.data_state = component.run(self.data_state)

        # Final cleanup of side processes (COCO files generation...) at the end of the pipeline
        clean_side_processes(self.data_state)

        green_print("Pipeline finished")

    def _get_component(self, component_config_tuple, component_id):
        component_type, component_config = component_config_tuple

        if component_type == 'tilerizer':
            return TilerizerComponent(component_config, self.io_config.output_folder, component_id,
                                      self.infer_aois_config, self.ground_truth_aoi_config)
        elif component_type == 'detector':
            return DetectorComponent(component_config, self.io_config.output_folder, component_id)
        elif component_type == 'aggregator':
            return AggregatorComponent(component_config, self.io_config.output_folder, component_id)
        elif component_type == 'segmenter':
            return SegmenterComponent(component_config, self.io_config.output_folder, component_id)
        elif component_type == 'evaluator':
            clean_side_processes(self.data_state)    # making sure COCO files are generated before starting evaluation
            return EvaluatorComponent(component_config, self.io_config.output_folder, component_id)
        # elif isinstance(component_config, EmbedderConfig):
        #     return build_embedder()
        # elif isinstance(component_config, ClassifierConfig):
        #     return build_classifier()
        # elif isinstance(component_config, ClustererConfig):
        #     return build_clusterer()
        else:
            raise ValueError(f'Invalid component {component_config}')

    def _validate_components_order(self):
        pass    # TODO
