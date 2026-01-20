from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import geopandas as gpd

from canopyrs.engine.components.aggregator import AggregatorComponent
from canopyrs.engine.components.detector import DetectorComponent
from canopyrs.engine.components.segmenter import SegmenterComponent
from canopyrs.engine.components.tilerizer import TilerizerComponent
from canopyrs.engine.components.classifier import ClassifierComponent

from canopyrs.engine.config_parsers import PipelineConfig, InferIOConfig
from canopyrs.engine.data_state import DataState
from canopyrs.engine.utils import parse_tilerizer_aoi_config, infer_aoi_name, green_print, get_component_folder_name, \
    object_id_column_name, tile_path_column_name


class Pipeline:
    def __init__(self, io_config: InferIOConfig, config: PipelineConfig):
        self.io_config = io_config
        self.config = config
        self._validate_components_order()

        self.background_executor = ProcessPoolExecutor(max_workers=1)

        # Initialize data state from the io (input/output) config
        self.data_state = DataState(
            imagery_path=self.io_config.input_imagery,
            parent_output_path=self.io_config.output_folder,
            tiles_path=self.io_config.tiles_path,
            infer_coco_path=self.io_config.input_coco,
            infer_gdf=gpd.read_file(self.io_config.input_gpkg) if self.io_config.input_gpkg else None,
            infer_gdf_columns_to_pass=set(self.io_config.infer_gdf_columns_to_pass) if self.io_config.infer_gdf_columns_to_pass else set(),
            background_executor=self.background_executor
        )

        # Initialize AOI configuration (Area of Interest, used by the Tilerizer)
        self.infer_aois_config = parse_tilerizer_aoi_config(
            aoi_config=self.io_config.aoi_config,
            aoi_type=self.io_config.aoi_type,
            aois={infer_aoi_name: self.io_config.aoi}
        )

        # If an infer_gdf from a previous pipeline run is provided, make sure to pass the special columns if present
        for special_column_name in [object_id_column_name, tile_path_column_name]:
            if self.data_state.infer_gdf is not None and special_column_name in self.data_state.infer_gdf.columns:
                self.data_state.infer_gdf_columns_to_pass.update([special_column_name])

        green_print("Pipeline initialized")

    def __call__(self):
        try:
            # Run each component in the pipeline, sequentially
            for component_id, (component_type, component_config) in enumerate(self.config.components_configs):
                component = self._get_component(component_id, component_type, component_config)
                self.data_state = component.run(self.data_state)

            # Final cleanup of side processes (COCO files generation...) at the end of the pipeline
            self.data_state.clean_side_processes()

            # Register any files that might have been missed during async processing
            self._register_known_component_files()

            green_print("Pipeline finished")
        finally:
            self.background_executor.shutdown(wait=True)

        return self.data_state

    def _get_component(self, component_id, component_type, component_config):
        if component_type == 'tilerizer':
            return TilerizerComponent(component_config, self.io_config.output_folder, component_id, self.infer_aois_config)
        elif component_type == 'detector':
            return DetectorComponent(component_config, self.io_config.output_folder, component_id)
        elif component_type == 'aggregator':
            return AggregatorComponent(component_config, self.io_config.output_folder, component_id)
        elif component_type == 'segmenter':
            return SegmenterComponent(component_config, self.io_config.output_folder, component_id)
        elif component_type == 'classifier':
            return ClassifierComponent(component_config, self.io_config.output_folder, component_id)
        else:
            raise ValueError(f'Invalid component {component_config}')

    def _register_known_component_files(self):
        """Register any output files that might have been missed during async processing"""
        for component_id, (component_type, component_config) in enumerate(self.config.components_configs):
            component_path = Path(self.io_config.output_folder) / get_component_folder_name(component_id, component_type)

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
        pass  # TODO
