"""
TilerizerComponent with simplified architecture.

Single __call__() method returns ComponentResult.
Pipeline handles state updates. Tilerizer handles its own file I/O internally.
"""

from pathlib import Path
from typing import Set, Optional, List

import geopandas as gpd
import rasterio

from geodataset.aoi import AOIConfig
from geodataset.tilerize import RasterTilerizer, LabeledRasterTilerizer, RasterPolygonTilerizer

from canopyrs.engine.constants import Col, StateKey, INFER_AOI_NAME
from canopyrs.engine.components.base import BaseComponent, ComponentResult, validate_requirements
from canopyrs.engine.config_parsers.tilerizer import TilerizerConfig
from canopyrs.engine.data_state import DataState


class TilerizerComponent(BaseComponent):
    """
    Creates image tiles from raster imagery.

    Tile types:
        - 'tile': Unlabeled regular grid tiles (for inference)
        - 'tile_labeled': Labeled regular grid tiles (for training data)
        - 'polygon': Per-polygon tiles (for classifier input)

    Requirements vary by tile_type:
        - 'tile': imagery_path only
        - 'tile_labeled': imagery_path + infer_gdf
        - 'polygon': imagery_path + infer_gdf

    Produces:
        - tiles_path  (always)
        - infer_coco_path (only for 'tile_labeled' and 'polygon')
    """

    name = 'tilerizer'

    BASE_REQUIRES_STATE = {StateKey.IMAGERY_PATH}
    BASE_REQUIRES_COLUMNS: Set[str] = set()

    BASE_PRODUCES_STATE = {StateKey.TILES_PATH}
    BASE_PRODUCES_COLUMNS: Set[str] = set()

    BASE_STATE_HINTS = {
        StateKey.IMAGERY_PATH: "Tilerizer needs an imagery_path to the raster file.",
        StateKey.INFER_GDF: "This tile_type requires a GeoDataFrame with labels/polygons.",
    }

    BASE_COLUMN_HINTS = {
        Col.GEOMETRY: "GeoDataFrame must have a 'geometry' column.",
    }

    def __init__(
        self,
        config: TilerizerConfig,
        parent_output_path: str = None,
        component_id: int = None,
        infer_aois_config: Optional[AOIConfig] = None
    ):
        super().__init__(config, parent_output_path, component_id)
        self.infer_aois_config = infer_aois_config

        # Validate tile_type
        if config.tile_type not in ['tile', 'tile_labeled', 'polygon']:
            raise ValueError(
                f"Invalid tile_type: '{config.tile_type}'. "
                f"Must be 'tile', 'tile_labeled', or 'polygon'."
            )

        # Set base requirements
        self.requires_state = set(self.BASE_REQUIRES_STATE)
        self.requires_columns = set(self.BASE_REQUIRES_COLUMNS)
        self.produces_state = set(self.BASE_PRODUCES_STATE)
        self.produces_columns = set(self.BASE_PRODUCES_COLUMNS)

        # Set hints
        self.state_hints = dict(self.BASE_STATE_HINTS)
        self.column_hints = dict(self.BASE_COLUMN_HINTS)

        # Tile-type-specific requirements
        if config.tile_type == 'tile':
            # Unlabeled tiles - no additional requirements or produces
            pass

        elif config.tile_type == 'tile_labeled':
            # Labeled tiles - requires infer_gdf, produces COCO
            self.requires_state.add(StateKey.INFER_GDF)
            self.requires_columns.add(Col.GEOMETRY)
            self.produces_state.add(StateKey.INFER_COCO_PATH)
            self.state_hints[StateKey.INFER_GDF] = (
                f"tile_type='tile_labeled' requires infer_gdf with labels. "
                f"Use tile_type='tile' for unlabeled tiles."
            )

        elif config.tile_type == 'polygon':
            # Polygon tiles - requires infer_gdf, produces COCO
            self.requires_state.add(StateKey.INFER_GDF)
            self.requires_columns.add(Col.GEOMETRY)
            self.produces_state.add(StateKey.INFER_COCO_PATH)
            self.state_hints[StateKey.INFER_GDF] = (
                f"tile_type='polygon' requires infer_gdf with polygons."
            )

    @classmethod
    def run_standalone(
        cls,
        config: TilerizerConfig,
        imagery_path: str,
        output_path: str,
        infer_gdf: gpd.GeoDataFrame = None,
        infer_aois_config: Optional[AOIConfig] = None,
    ) -> 'DataState':
        """
        Run tilerizer standalone on raster imagery.

        Args:
            config: Tilerizer configuration (tile_type determines requirements)
            imagery_path: Path to the raster file
            output_path: Where to save outputs
            infer_gdf: GeoDataFrame with labels/polygons
                        (required for tile_type='tile_labeled' or 'polygon')
            infer_aois_config: Area of Interest configuration (optional)

        Returns:
            DataState with tiling results (access .tiles_path for tile directory)

        Example:
            result = TilerizerComponent.run_standalone(
                config=TilerizerConfig(tile_type='tile', tile_size=512, ...),
                imagery_path='./raster.tif',
                output_path='./output',
            )
            print(result.tiles_path)
        """
        from canopyrs.engine.pipeline import run_component
        return run_component(
            component=cls(config, infer_aois_config=infer_aois_config),
            output_path=output_path,
            imagery_path=imagery_path,
            infer_gdf=infer_gdf,
        )

    @validate_requirements
    def __call__(self, data_state: DataState) -> ComponentResult:
        """
        Create tiles from raster imagery.

        Returns ComponentResult - Pipeline handles state updates.
        Tilerizer handles its own file I/O internally via geodataset.
        """
        self._check_crs_match(data_state)

        # Handle config columns
        columns_to_pass = data_state.infer_gdf_columns_to_pass.copy()
        if self.config.other_labels_attributes_column_names:
            columns_to_pass.update(self.config.other_labels_attributes_column_names)

        columns_to_pass = [col for col in columns_to_pass if col not in {Col.GEOMETRY, Col.TILE_PATH}]  # already taken care of by COCO format

        # Process based on tile_type
        if self.config.tile_type == 'tile':
            if data_state.infer_gdf is not None:
                raise ValueError(
                    "infer_gdf provided but tile_type='tile' creates unlabeled tiles. "
                    "Use tile_type='tile_labeled' or 'polygon' if labels are needed for subsequent components, like a prompted Segmenter."
                )
            # Unlabeled tiles only
            tiles_path, infer_coco_path = self._process_unlabeled_tiles(data_state)

        elif self.config.tile_type == 'tile_labeled':
            # Labeled regular grid tiles
            tiles_path, infer_coco_path = self._process_labeled_tiles(
                data_state, columns_to_pass
            )

        elif self.config.tile_type == 'polygon':
            # Polygon tiles
            tiles_path, infer_coco_path = self._process_polygon_tiles(
                data_state, columns_to_pass
            )

        else:
            raise ValueError(f"Invalid tile_type: {self.config.tile_type}")

        # Save config
        if self.output_path:
            self.config.to_yaml(self.output_path / "tilerizer_config.yaml")

        # Register the COCO file that geodataset already wrote (if any)
        output_files = {}
        if infer_coco_path is not None:
            output_files['coco'] = infer_coco_path

        return ComponentResult(
            gdf=None,  # Tilerizer doesn't modify the GDF
            produced_columns=columns_to_pass,
            state_updates={
                StateKey.TILES_PATH: tiles_path,
                StateKey.INFER_COCO_PATH: infer_coco_path,
            },
            save_gpkg=False,
            save_coco=False,  # COCO handled internally by tilerizer
            output_files=output_files,
        )

    def _process_labeled_tiles(self, data_state: DataState, columns_to_pass: Set[str]):
        """Process labeled regular grid tiles (tile_type='tile_labeled')."""
        tilerizer = LabeledRasterTilerizer(
            raster_path=data_state.imagery_path,
            labels_path=None,
            labels_gdf=data_state.infer_gdf,
            output_path=self.output_path,
            tile_size=self.config.tile_size,
            tile_overlap=self.config.tile_overlap,
            aois_config=self.infer_aois_config,
            scale_factor=self.config.scale_factor,
            ground_resolution=self.config.ground_resolution,
            ignore_black_white_alpha_tiles_threshold=self.config.ignore_black_white_alpha_tiles_threshold,
            min_intersection_ratio=self.config.min_intersection_ratio,
            ignore_tiles_without_labels=self.config.ignore_tiles_without_labels,
            main_label_category_column_name=self.config.main_label_category_column_name,
            other_labels_attributes_column_names=list(columns_to_pass),
        )
        coco_paths = tilerizer.generate_coco_dataset()
        return tilerizer.tiles_path, coco_paths.get(INFER_AOI_NAME)

    def _process_unlabeled_tiles(self, data_state: DataState):
        """Process unlabeled tiles (tile_type='tile' without infer_gdf)."""
        tilerizer = RasterTilerizer(
            raster_path=data_state.imagery_path,
            output_path=self.output_path,
            tile_size=self.config.tile_size,
            tile_overlap=self.config.tile_overlap,
            aois_config=self.infer_aois_config,
            scale_factor=self.config.scale_factor,
            ground_resolution=self.config.ground_resolution,
            ignore_black_white_alpha_tiles_threshold=self.config.ignore_black_white_alpha_tiles_threshold,
        )
        tilerizer.generate_tiles()
        return tilerizer.tiles_path, None

    def _process_polygon_tiles(self, data_state: DataState, columns_to_pass: Set[str]):
        """Process polygon tiles (tile_type='polygon')."""
        tilerizer = RasterPolygonTilerizer(
            raster_path=data_state.imagery_path,
            output_path=self.output_path,
            labels_path=None,
            labels_gdf=data_state.infer_gdf,
            tile_size=self.config.tile_size,
            use_variable_tile_size=self.config.use_variable_tile_size,
            variable_tile_size_pixel_buffer=self.config.variable_tile_size_pixel_buffer,
            aois_config=self.infer_aois_config,
            scale_factor=self.config.scale_factor,
            ground_resolution=self.config.ground_resolution,
            main_label_category_column_name=self.config.main_label_category_column_name,
            other_labels_attributes_column_names=list(columns_to_pass),
            coco_n_workers=self.config.coco_n_workers,
        )
        coco_paths = tilerizer.generate_coco_dataset()
        return tilerizer.tiles_folder_path, coco_paths.get(INFER_AOI_NAME)

    def _check_crs_match(self, data_state: DataState):
        """Check if the CRS of the raster and GeoDataFrame match."""
        if data_state.infer_gdf is None:
            return

        try:
            with rasterio.open(data_state.imagery_path) as src:
                raster_crs = src.crs
        except Exception as e:
            raise RuntimeError(f"Failed to open raster: {e}")

        gdf_crs = data_state.infer_gdf.crs

        if raster_crs is not None and gdf_crs is None:
            raise ValueError("Raster has CRS but infer_gdf does not.")
        elif raster_crs is None and gdf_crs is not None:
            raise ValueError("Raster has no CRS but infer_gdf does.")
