from pathlib import Path

import geopandas as gpd

from engine.components.aggregator import AggregatorComponent
from engine.components.detector import DetectorComponent
from engine.components.evaluator import EvaluatorComponent
from engine.components.segmenter import SegmenterComponent
from engine.components.tilerizer import TilerizerComponent

from engine.config_parsers import (PipelineConfig, InferIOConfig, TilerizerConfig,
                                   DetectorConfig, AggregatorConfig, SegmenterConfig)
from engine.config_parsers.base import get_config_path
from engine.config_parsers.evaluator import EvaluatorConfig
from engine.data_state import DataState
from engine.utils import parse_tilerizer_aoi_config, infer_aoi_name, ground_truth_aoi_name


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
            tiles_names=list([tile.name for  tile in Path(self.io_config.tiles_path).rglob('*.tif')])
                        if self.io_config.tiles_path else None,
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

    def run(self, get_data_state=False):
        # Run each component in the pipeline, sequentially
        for component_id, component_config in enumerate(self.config.components_configs):
            component = self._get_component(component_config, component_id)
            self.data_state = component.run(self.data_state)

        # Final cleanup of side processes (COCO files generation...) at the end of the pipeline
        self._clean_side_processes()

        # Register any files that might have been missed during async processing
        self._register_known_component_files()

        if get_data_state == True:
            return self.data_state

    def _get_component(self, component_config, component_id):
        component_type = list(component_config.keys())[0]
        config_name = list(component_config.values())[0]
        config_path = get_config_path(config_name)

        if component_type == 'tilerizer':
            component_config = TilerizerConfig.from_yaml(config_path)
            return TilerizerComponent(component_config, self.io_config.output_folder, component_id,
                                      self.infer_aois_config, self.ground_truth_aoi_config)
        elif component_type == 'detector':
            component_config = DetectorConfig.from_yaml(config_path)
            return DetectorComponent(component_config, self.io_config.output_folder, component_id)
        elif component_type == 'aggregator':
            component_config = AggregatorConfig.from_yaml(config_path)
            return AggregatorComponent(component_config, self.io_config.output_folder, component_id)
        elif component_type == 'segmenter':
            component_config = SegmenterConfig.from_yaml(config_path)
            return SegmenterComponent(component_config, self.io_config.output_folder, component_id)
        elif component_type == 'evaluator':
            component_config = EvaluatorConfig.from_yaml(config_path)
            self._clean_side_processes()    # making sure COCO files are generated before starting evaluation
            return EvaluatorComponent(component_config, self.io_config.output_folder, component_id)
        # elif isinstance(component_config, EmbedderConfig):
        #     return build_embedder()
        # elif isinstance(component_config, ClassifierConfig):
        #     return build_classifier()
        # elif isinstance(component_config, ClustererConfig):
        #     return build_clusterer()
        else:
            raise ValueError(f'Invalid component {component_config}')

    def _clean_side_processes(self):
        for side_process in self.data_state.side_processes:
            if isinstance(side_process, tuple):
                attribute_name = side_process[0]
                future_or_result = side_process[1]
                
                # Check if this is a Future object with a .result() method
                if hasattr(future_or_result, 'result'):
                    result = future_or_result.result()
                else:
                    result = future_or_result  # It's already a result
                
                # Update the data_state attribute
                if attribute_name:
                    setattr(self.data_state, attribute_name, result)
                
                # If there's registration info, register the output file
                if len(side_process) > 2 and isinstance(side_process[2], dict):
                    reg_info = side_process[2]
                    
                    # If an expected_path was provided, use it
                    if 'expected_path' in reg_info:
                        file_path = Path(reg_info['expected_path'])
                    # Otherwise try to get a path from the result
                    elif isinstance(result, (str, Path)):
                        file_path = Path(result)
                    else:
                        file_path = None
                        
                    if file_path:
                        # Register the component folder first
                        self.data_state.register_component_folder(
                            reg_info['component_name'], 
                            reg_info['component_id'],
                            file_path.parent
                        )
                        # Then register the file
                        self.data_state.register_output_file(
                            reg_info['component_name'],
                            reg_info['component_id'],
                            reg_info['file_type'],
                            file_path
                        )
        
        # Clear processed side processes
        self.data_state.side_processes = []
        
        return self.data_state  # Return the updated data_state

    def _register_known_component_files(self):
        """Register any output files that might have been missed during async processing"""
        for component_id, component_config in enumerate(self.config.components_configs):
            component_type = list(component_config.keys())[0]
            component_path = Path(self.io_config.output_folder) / f"{component_id}_{component_type}"
            
            # Register the component folder if it exists
            if component_path.exists():
                self.data_state.register_component_folder(component_type, component_id, component_path)
                
                # Register COCO files
                for coco_file in component_path.glob("*.json"):
                    # Skip non-COCO JSON files if needed
                    if "_coco_" in coco_file.name:
                        self.data_state.register_output_file(
                            component_type, component_id, 'coco', coco_file
                        )
                
                # Register GeoPackage files
                for gpkg_file in component_path.glob("*.gpkg"):
                    file_type = 'pre_aggregated_gpkg' if 'notaggregated' in gpkg_file.name else 'gpkg'
                    self.data_state.register_output_file(
                        component_type, component_id, file_type, gpkg_file
                    )

    def _validate_components_order(self):
        pass    # TODO
