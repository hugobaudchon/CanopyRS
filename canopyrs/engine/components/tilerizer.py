from typing import Optional
from pathlib import Path

import geopandas as gpd
import rasterio

from geodataset.aoi import AOIConfig
from geodataset.tilerize import RasterTilerizer, LabeledRasterTilerizer, RasterPolygonTilerizer

from canopyrs.engine.components.base import BaseComponent
from canopyrs.engine.config_parsers.tilerizer import TilerizerConfig
from canopyrs.engine.data_state import DataState
from canopyrs.engine.utils import infer_aoi_name


class TilerizerComponent(BaseComponent):
    name = 'tilerizer'

    def __init__(self,
                 config: TilerizerConfig,
                 parent_output_path: str,
                 component_id: int,
                 infer_aois_config: Optional[AOIConfig] = None):
        super().__init__(config, parent_output_path, component_id)
        self.infer_aois_config = infer_aois_config

    def __call__(self, data_state: DataState) -> DataState:

        if data_state.imagery_path is None:
            raise ValueError("No imagery_path specified in data_state. Cannot create tilerizer.")

        self._check_crs_match(data_state)

        if self.config.other_labels_attributes_column_names is not None:
            data_state.infer_gdf_columns_to_pass.update(self.config.other_labels_attributes_column_names)

        if self.config.tile_type == 'tile':
            if data_state.infer_gdf is not None:
                # Generate results COCO and tiles
                tilerizer = self.get_labeled_tilerizer(
                    data_state=data_state,
                    labels_gdf=data_state.infer_gdf,
                    aois_config=self.infer_aois_config,
                    other_labels_attributes_column_names=list(data_state.infer_gdf_columns_to_pass.union(self.config.other_labels_attributes_column_names))
                )

                infer_coco_path = tilerizer.generate_coco_dataset()[infer_aoi_name]
                tiles_path = tilerizer.tiles_path
                tiles_names = [tile.generate_name() for tile in tilerizer.aois_tiles[infer_aoi_name]]
            else:
                # Tilerize without any labels
                tilerizer = self.get_unlabeled_tilerizer(
                    data_state=data_state,
                    aois_config=self.infer_aois_config
                )

                tilerizer.generate_tiles()
                infer_coco_path = None
                tiles_path = tilerizer.tiles_path
                tiles_names = [tile.generate_name() for tile in tilerizer.aois_tiles[infer_aoi_name]]

        elif self.config.tile_type == 'polygon':
            if data_state.infer_gdf is not None:
                tilerizer = self.get_polygon_tilerizer(
                    data_state=data_state,
                    labels_gdf=data_state.infer_gdf,
                    aois_config=self.infer_aois_config,
                    other_labels_attributes_column_names=list(data_state.infer_gdf_columns_to_pass.union(self.config.other_labels_attributes_column_names))
                )
                infer_coco_path = tilerizer.generate_coco_dataset()[infer_aoi_name]
                tiles_path = tilerizer.tiles_folder_path
                tiles_names = self._collect_polygon_tile_names(tilerizer, infer_aoi_name)
            else:
                raise ValueError("Polygon tilerization requires detection/segmentation inference data.")

        else:
            raise ValueError(f"Invalid tile type: {self.config.tile_type}. Expected 'tile' or 'polygon'.")

        self.config.to_yaml(self.output_path / "tilerizer_config.yaml")

        return self.update_data_state(data_state, tiles_path, infer_coco_path, tiles_names)

    def update_data_state(self,
                         data_state: DataState,
                         tiles_path: str,
                         infer_coco_path: str,
                         tiles_names: list[str]) -> DataState:
        # Register the component folder
        data_state = self.register_outputs_base(data_state)

        # Register important output files (not all tiles)
        if infer_coco_path:
            data_state.register_output_file(self.name, self.component_id, 'infer_coco', Path(infer_coco_path))

        data_state.tiles_path = tiles_path
        data_state.infer_coco_path = infer_coco_path
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

    def get_unlabeled_tilerizer(self,
                                data_state: DataState,
                                aois_config: AOIConfig):

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

    def get_polygon_tilerizer(self,
                              data_state: DataState,
                              labels_gdf: gpd.GeoDataFrame,
                              aois_config: AOIConfig,
                              other_labels_attributes_column_names: list[str]):

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
            coco_n_workers=self.config.coco_n_workers,
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

    def _check_crs_match(self, data_state: DataState):
        """Checks if the CRS of the raster and the GeoDataFrame match."""
        if data_state.infer_gdf is None:
            return  # No GeoDataFrame to check against, continue

        raster_crs = None
        try:
            with rasterio.open(data_state.imagery_path) as src:
                raster_crs = src.crs
        except Exception as e:
            raise RuntimeError(f"Failed to open raster file at '{data_state.imagery_path}': {e}")

        gdf_crs = data_state.infer_gdf.crs

        if (raster_crs is not None and gdf_crs is None):
            raise ValueError(
                "The raster has a CRS but the infer_gdf does not. "
                "Please ensure the GeoDataFrame is projected accordingly, by using an Aggregator component before the Tilerizer."
            )
        elif (raster_crs is None and gdf_crs is not None):
            raise ValueError(
                "The raster has no CRS, but the infer_gdf does. "
                "This is ambiguous, as the raster is in a pixel coordinate system whereas the GeoDataFrame is in a geographic coordinate system."
            )
