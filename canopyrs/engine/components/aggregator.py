"""
AggregatorComponent with simplified architecture.

Single __call__() method returns ComponentResult.
Pipeline handles all I/O (saving gpkg, COCO generation, state updates).
"""

from pathlib import Path
from typing import Set
import warnings

from geodataset.aggregator import Aggregator
from geodataset.utils import GeoPackageNameConvention, TileNameConvention

from canopyrs.engine.constants import Col, StateKey, INFER_AOI_NAME
from canopyrs.engine.components.base import BaseComponent, ComponentResult, ComponentValidationError, validate_requirements
from canopyrs.engine.config_parsers import AggregatorConfig
from canopyrs.engine.data_state import DataState


class AggregatorComponent(BaseComponent):
    """
    Aggregates overlapping detections/segmentations from tiled inference.

    Requirements:
        - infer_gdf with geometry, object_id, tile_path columns
        - Score columns based on config weights (detector_score/segmenter_score)

    Produces:
        - Merged GeoDataFrame with aggregator_score
        - GeoPackage files (aggregated + pre-aggregated)
        - COCO file
    """

    name = 'aggregator'

    BASE_REQUIRES_STATE = {StateKey.INFER_GDF, StateKey.PRODUCT_NAME}
    BASE_REQUIRES_COLUMNS = {Col.GEOMETRY, Col.OBJECT_ID, Col.TILE_PATH}

    BASE_PRODUCES_STATE = {StateKey.INFER_GDF, StateKey.INFER_COCO_PATH}
    BASE_PRODUCES_COLUMNS = {Col.AGGREGATOR_SCORE}

    BASE_STATE_HINTS = {
        StateKey.INFER_GDF: (
            "Aggregator needs a GeoDataFrame with detections/segmentations. "
            "Add a detector or segmenter before aggregator in the pipeline."
        ),
    }

    BASE_COLUMN_HINTS = {
        Col.GEOMETRY: "GeoDataFrame must have a 'geometry' column with polygon geometries.",
        Col.OBJECT_ID: "Each detection needs a unique 'canopyrs_object_id'. Created by detector/segmenter.",
        Col.TILE_PATH: "Each detection needs a 'tile_path' column indicating source tile.",
    }

    def __init__(
        self,
        config: AggregatorConfig,
        parent_output_path: str = None,
        component_id: int = None
    ):
        super().__init__(config, parent_output_path, component_id)

        # Set base requirements
        self.requires_state = set(self.BASE_REQUIRES_STATE)
        self.requires_columns = set(self.BASE_REQUIRES_COLUMNS)
        self.produces_state = set(self.BASE_PRODUCES_STATE)
        self.produces_columns = set(self.BASE_PRODUCES_COLUMNS)

        # Set hints
        self.state_hints = dict(self.BASE_STATE_HINTS)
        self.column_hints = dict(self.BASE_COLUMN_HINTS)

        # Add config-dependent requirements
        if config.detector_score_weight > 0:
            self.requires_columns.add(Col.DETECTOR_SCORE)
            self.column_hints[Col.DETECTOR_SCORE] = (
                f"Config has detector_score_weight={config.detector_score_weight} > 0, "
                f"so '{Col.DETECTOR_SCORE}' column is required. "
                f"Add a detector before aggregator, or set detector_score_weight=0."
            )

        if config.segmenter_score_weight > 0:
            self.requires_columns.add(Col.SEGMENTER_SCORE)
            self.column_hints[Col.SEGMENTER_SCORE] = (
                f"Config has segmenter_score_weight={config.segmenter_score_weight} > 0, "
                f"so '{Col.SEGMENTER_SCORE}' column is required. "
                f"Add a segmenter before aggregator, or set segmenter_score_weight=0."
            )

    @classmethod
    def run_standalone(
        cls,
        config: AggregatorConfig,
        infer_gdf: 'gpd.GeoDataFrame',
        output_path: str,
        product_name: str = "standalone",
    ) -> 'DataState':
        """
        Run aggregator standalone on a GeoDataFrame of detections/segmentations.

        Args:
            config: Aggregator configuration
            infer_gdf: GeoDataFrame with geometry, object_id, and tile_path columns
            output_path: Where to save outputs
            product_name: Name for output files (used in gpkg naming)

        Returns:
            DataState with aggregated results (access .infer_gdf for the GeoDataFrame)

        Example:
            result = AggregatorComponent.run_standalone(
                config=AggregatorConfig(nms_threshold=0.5, ...),
                infer_gdf=my_detections_gdf,
                output_path='./output',
            )
            print(result.infer_gdf)
        """
        from canopyrs.engine.pipeline import run_component
        return run_component(
            component=cls(config),
            output_path=output_path,
            infer_gdf=infer_gdf,
            product_name=product_name,
        )

    @validate_requirements
    def __call__(self, data_state: DataState) -> ComponentResult:
        """
        Aggregate overlapping detections/segmentations.

        Returns ComponentResult - Pipeline handles I/O and state updates.
        """
        # Suppress geographic CRS area warnings from geopandas
        warnings.filterwarnings('ignore', message='.*Geometry is in a geographic CRS.*')

        infer_gdf = data_state.infer_gdf
        columns_to_pass = data_state.infer_gdf_columns_to_pass

        # Build score columns and weights from config
        score_cols = []
        weights = []

        if self.config.detector_score_weight > 0:
            score_cols.append(Col.DETECTOR_SCORE)
            weights.append(self.config.detector_score_weight)

        if self.config.segmenter_score_weight > 0:
            score_cols.append(Col.SEGMENTER_SCORE)
            weights.append(self.config.segmenter_score_weight)

        # Generate output names
        gpkg_name, pre_agg_gpkg_name = self._get_gpkg_names(data_state)

        # Drop some previous components columns that can interfere with aggregation
        for col in [Col.AGGREGATOR_SCORE, 'tile_id']:
            if col in infer_gdf.columns:
                infer_gdf = infer_gdf.drop(columns=[col])
                if col in columns_to_pass:
                    columns_to_pass.remove(col)

        # Run aggregation (geodataset Aggregator handles its own file saving)
        aggregator = Aggregator.from_gdf(
            output_path=self.output_path / gpkg_name if self.output_path else None,
            gdf=infer_gdf,
            tiles_paths_column=Col.TILE_PATH,
            polygons_column=Col.GEOMETRY,
            scores_column=score_cols if score_cols else None,
            other_attributes_columns=list(columns_to_pass),
            scores_weights=weights if weights else None,
            scores_weighting_method=self.config.scores_weighting_method,
            min_centroid_distance_weight=self.config.min_centroid_distance_weight,
            score_threshold=self.config.score_threshold,
            nms_threshold=self.config.nms_threshold,
            nms_algorithm=self.config.nms_algorithm,
            best_geom_keep_area_ratio=self.config.best_geom_keep_area_ratio,
            edge_band_buffer_percentage=self.config.edge_band_buffer_percentage,
            pre_aggregated_output_path=self.output_path / pre_agg_gpkg_name if self.output_path else None,
        )

        result_gdf = aggregator.polygons_gdf

        # Determine category column for COCO
        coco_categories_col = None
        if Col.SEGMENTER_CLASS in result_gdf.columns:
            coco_categories_col = Col.SEGMENTER_CLASS
        elif Col.DETECTOR_CLASS in result_gdf.columns:
            coco_categories_col = Col.DETECTOR_CLASS

        # Register the GeoPackages that geodataset already wrote
        output_files = {}
        if self.output_path:
            output_files['gpkg'] = self.output_path / gpkg_name
            output_files['pre_aggregated_gpkg'] = self.output_path / pre_agg_gpkg_name

        return ComponentResult(
            gdf=result_gdf,
            produced_columns=columns_to_pass | {Col.AGGREGATOR_SCORE},
            objects_are_new=False,
            save_gpkg=False,  # Aggregator already saves via geodataset
            save_coco=True,
            coco_scores_column=Col.AGGREGATOR_SCORE,
            coco_categories_column=coco_categories_col,
            output_files=output_files,
        )

    def _get_gpkg_names(self, data_state: DataState) -> tuple:
        """Generate GeoPackage names using the product name from data state."""

        try:
            _, scale_factor, ground_resolution, _, _, _ = TileNameConvention().parse_name(
                Path(data_state.infer_gdf[Col.TILE_PATH].iloc[0]).name
            )
        except Exception as e:
            scale_factor = 1.0
            ground_resolution = None

        gpkg_name = GeoPackageNameConvention.create_name(
            product_name=data_state.product_name,
            fold=INFER_AOI_NAME,
            scale_factor=scale_factor,
            ground_resolution=ground_resolution
        )

        pre_agg_gpkg_name = GeoPackageNameConvention.create_name(
            product_name=data_state.product_name,
            fold=f'{INFER_AOI_NAME}notaggregated',
            scale_factor=scale_factor,
            ground_resolution=ground_resolution
        )

        return gpkg_name, pre_agg_gpkg_name
