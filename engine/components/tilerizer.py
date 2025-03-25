from pathlib import Path

import geopandas as gpd

from geodataset.aoi import AOIConfig
from geodataset.tilerize import RasterPolygonTilerizer, RasterTilerizer, LabeledRasterTilerizer, RasterPolygonTilerizer

from engine.components.base import BaseComponent
from engine.config_parsers.tilerizer import TilerizerConfig
from engine.data_state import DataState
from engine.utils import ground_truth_aoi_name, infer_aoi_name


class TilerizerComponent(BaseComponent):
    name = 'tilerizer'

    def __init__(self,
                 config: TilerizerConfig,
                 parent_output_path: str,
                 component_id: int,
                 infer_aois_config: AOIConfig,
                 ground_truth_aois_config: AOIConfig):
        super().__init__(config, parent_output_path, component_id)
        self.infer_aois_config = infer_aois_config
        self.ground_truth_aois_config = ground_truth_aois_config

    def __call__(self, data_state: DataState) -> DataState:
        if data_state.infer_gdf is not None and data_state.infer_gdf.crs is None:
            raise ValueError(
                "infer_gdf must have a CRS."
                " Please make sure you add an Aggregator in the pipeline before the Tilerizer."
            )
        if data_state.ground_truth_gdf is not None and data_state.ground_truth_gdf.crs is None:
            raise ValueError(
                "ground_truth_gdf must have a CRS."
                " Please input a valid ground truth GeoPackage."
            )

        if self.config.other_labels_attributes_column_names is not None:
            data_state.infer_gdf_columns_to_pass.update(self.config.other_labels_attributes_column_names)

        if self.config.tile_type == 'tile':
            if data_state.infer_gdf is not None and data_state.ground_truth_gdf is not None:
                # First, generating the ground truth COCO and tiles
                tilerizer = self.get_labeled_tilerizer(
                    data_state=data_state,
                    labels_gdf=data_state.ground_truth_gdf,
                    aois_config=self.ground_truth_aois_config,
                    other_labels_attributes_column_names=list(data_state.ground_truth_gdf_columns_to_pass)
                )
                ground_truth_coco_path = tilerizer.generate_coco_dataset()[ground_truth_aoi_name]
                tiles_path = tilerizer.tiles_path
                tiles_names = [tile.generate_name() for tile in tilerizer.aois_tiles[ground_truth_aoi_name]]

                # Then, generating the results COCO, based on the ground truth tiles
                second_result = tilerizer.generate_additional_coco_dataset(
                    labels_gdf=data_state.infer_gdf,
                    aoi_name_mapping={ground_truth_aoi_name: infer_aoi_name},
                    geopackage_layer_name=None,
                    main_label_category_column_name=self.config.main_label_category_column_name,
                    other_labels_attributes_column_names=list(data_state.infer_gdf_columns_to_pass.union(self.config.other_labels_attributes_column_names))
                )

                infer_coco_path = second_result[infer_aoi_name]

            elif data_state.infer_gdf is not None:
                # No ground truth, so only generate results COCO and tiles
                tilerizer = self.get_labeled_tilerizer(
                    data_state=data_state,
                    labels_gdf=data_state.infer_gdf,
                    aois_config=self.infer_aois_config,
                    other_labels_attributes_column_names=list(data_state.infer_gdf_columns_to_pass.union(self.config.other_labels_attributes_column_names))
                )

                ground_truth_coco_path = None
                infer_coco_path = tilerizer.generate_coco_dataset()[infer_aoi_name]
                tiles_path = tilerizer.tiles_path
                tiles_names = [tile.generate_name() for tile in tilerizer.aois_tiles[infer_aoi_name]]

            elif data_state.ground_truth_gdf is not None:
                # No results GeoDataFrame, so only generate ground truth COCO and tiles
                tilerizer = self.get_labeled_tilerizer(
                    data_state=data_state,
                    labels_gdf=data_state.ground_truth_gdf,
                    aois_config=self.ground_truth_aois_config,
                    other_labels_attributes_column_names=list(data_state.ground_truth_gdf_columns_to_pass)
                )

                ground_truth_coco_path = tilerizer.generate_coco_dataset()[ground_truth_aoi_name]
                infer_coco_path = None
                tiles_path = tilerizer.tiles_path
                tiles_names = [tile.generate_name() for tile in tilerizer.aois_tiles[ground_truth_aoi_name]]

            else:
                # Tilerize without any labels
                tilerizer = self.get_unlabeled_tilerizer(
                    data_state=data_state,
                    aois_config=self.infer_aois_config
                )

                tilerizer.generate_tiles()
                infer_coco_path = None
                ground_truth_coco_path = None
                tiles_path = tilerizer.tiles_path
                tiles_names = [tile.generate_name() for tile in tilerizer.aois_tiles[infer_aoi_name]]

        elif self.config.tile_type == 'polygon':
            if data_state.ground_truth_gdf is not None:
                # Only ground truth data available
                tilerizer = self.get_polygon_tilerizer(data_state=data_state,
                                                       labels_gdf=data_state.ground_truth_gdf,
                                                       aois_config=self.ground_truth_aois_config,
                                                       other_labels_attributes_column_names=list(data_state.ground_truth_gdf_columns_to_pass))
                ground_truth_coco_path = tilerizer.generate_coco_dataset()[ground_truth_aoi_name]
                tiles_path = tilerizer.tiles_path
                tiles_names = [tile.generate_name() for tile in tilerizer.aois_tiles[ground_truth_aoi_name]]

            elif (data_state.infer_gdf is not None) or (data_state.infer_gdf is not None and data_state.ground_truth_gdf is not None):
                # Only inference data available, if both available select inference mode
                # Check if we need to set default category IDs
                if self.config.main_label_category_column_name is not None and \
                   self.config.main_label_category_column_name not in data_state.infer_gdf.columns:
                    print(f"Warning: Missing category column '{self.config.main_label_category_column_name}' - adding default category 0")
                    data_state.infer_gdf[self.config.main_label_category_column_name] = 0

                # Format other labels attributes column names
                if self.config.other_labels_attributes_column_names != []:
                    for col_name in self.config.other_labels_attributes_column_names:
                        data_state.infer_gdf_columns_to_pass.add(col_name)

                infer_tilerizer = self.get_polygon_tilerizer(
                    data_state=data_state,
                    labels_gdf=data_state.infer_gdf,
                    aois_config=self.self.infer_aois_config,
                    other_labels_attributes_column_names=list(data_state.infer_gdf_columns_to_pass)
                )
                infer_coco_path = infer_tilerizer.generate_coco_dataset()[ground_truth_aoi_name]
                ground_truth_coco_path = None
                tiles_path = infer_tilerizer.tiles_folder_path
                tiles_names = self._collect_polygon_tile_names(infer_tilerizer, infer_aoi_name)
            else:
                raise ValueError("Polygon tilerization requires either inference or ground truth data")

        else:
            raise ValueError(f"Invalid tile type: {self.config.tile_type}. Expected 'tile' or 'polygon'.")

        self.config.to_yaml(self.output_path / "tilerizer_config.yaml")

        return self.update_data_state(data_state, tiles_path, infer_coco_path, ground_truth_coco_path, tiles_names)

    def update_data_state(self,
                         data_state: DataState,
                         tiles_path: str,
                         infer_coco_path: str,
                         ground_truth_coco_path: str,
                         tiles_names: list[str]) -> DataState:
        # Register the component folder
        data_state = self.register_outputs_base(data_state)

        # Register important output files (not all tiles)
        if infer_coco_path:
            data_state.register_output_file(self.name, self.component_id, 'infer_coco', Path(infer_coco_path))
        if ground_truth_coco_path:
            data_state.register_output_file(self.name, self.component_id, 'ground_truth_coco', Path(ground_truth_coco_path))

        data_state.tiles_path = tiles_path
        data_state.infer_coco_path = infer_coco_path
        data_state.ground_truth_coco_path = ground_truth_coco_path
        data_state.tiles_names = tiles_names

        return data_state

    def get_labeled_tilerizer(self,
                              data_state: DataState,
                              labels_gdf: gpd.GeoDataFrame,
                              aois_config: AOIConfig,
                              other_labels_attributes_column_names: list[str]):
        tilerizer = LabeledRasterTilerizer(
            raster_path=data_state.imagery_path,
            labels_path=None,
            labels_gdf=labels_gdf,
            output_path=self.output_path,
            tile_size=self.config.tile_size,
            tile_overlap=self.config.tile_overlap,
            aois_config=aois_config,
            scale_factor=self.config.scale_factor,
            ground_resolution=self.config.ground_resolution,
            ignore_black_white_alpha_tiles_threshold=self.config.ignore_black_white_alpha_tiles_threshold,
            min_intersection_ratio=self.config.min_intersection_ratio,
            ignore_tiles_without_labels=self.config.ignore_tiles_without_labels,
            main_label_category_column_name=self.config.main_label_category_column_name,
            other_labels_attributes_column_names=other_labels_attributes_column_names,
            temp_dir=self.temp_path
        )

        return tilerizer

    def get_unlabeled_tilerizer(self, data_state: DataState, aois_config: AOIConfig):
        tilerizer = RasterTilerizer(
            raster_path=data_state.imagery_path,
            output_path=self.output_path,
            tile_size=self.config.tile_size,
            tile_overlap=self.config.tile_overlap,
            aois_config=aois_config,
            scale_factor=self.config.scale_factor,
            ground_resolution=self.config.ground_resolution,
            ignore_black_white_alpha_tiles_threshold=self.config.ignore_black_white_alpha_tiles_threshold,
            temp_dir=self.temp_path
        )

        return tilerizer

    def get_polygon_tilerizer(self, data_state: DataState, labels_gdf: gpd.GeoDataFrame, aois_config: AOIConfig, other_labels_attributes_column_names: list[str]):
        """Creates a polygon tilerizer for labeled data"""
        tilerizer = RasterPolygonTilerizer(
            raster_path=data_state.imagery_path,
            output_path=self.output_path,
            labels_path=None,
            labels_gdf=labels_gdf,
            tile_size=self.config.tile_size,
            use_variable_tile_size=self.config.use_variable_tile_size,
            variable_tile_size_pixel_buffer=self.config.variable_tile_size_pixel_buffer,
            aois_config=aois_config,
            scale_factor=self.config.scale_factor,
            ground_resolution=self.config.ground_resolution,
            main_label_category_column_name=self.config.main_label_category_column_name,
            other_labels_attributes_column_names=other_labels_attributes_column_names,
            temp_dir=self.temp_path
        )

        return tilerizer

    def _collect_polygon_tile_names(self, tilerizer, aoi_name):
        """Collects tile names from a polygon tilerizer for a specific AOI"""
        tile_names = []

        # Path to the tiles folder for the specified AOI
        aoi_tiles_path = tilerizer.tiles_folder_path / aoi_name

        # If the path exists, collect all tile filenames
        if aoi_tiles_path.exists():
            tile_names = [tile_path.name for tile_path in aoi_tiles_path.glob("*.tif")]
    
        return tile_names
