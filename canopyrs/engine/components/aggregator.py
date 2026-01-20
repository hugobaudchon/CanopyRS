from pathlib import Path

import geopandas as gpd

from geodataset.aggregator import Aggregator
from geodataset.utils import GeoPackageNameConvention, TileNameConvention

from canopyrs.engine.components.base import BaseComponent
from canopyrs.engine.config_parsers import AggregatorConfig
from canopyrs.engine.data_state import DataState
from canopyrs.engine.utils import infer_aoi_name, generate_future_coco, tile_path_column_name


class AggregatorComponent(BaseComponent):
    name = 'aggregator'

    def __init__(self, config: AggregatorConfig, parent_output_path: str, component_id: int):
        super().__init__(config, parent_output_path, component_id)

    def __call__(self, data_state: DataState) -> DataState:
        # Determine which score columns are available and their corresponding weights.
        available_scores = []
        scores_weights = []
        if 'detector_score' in data_state.infer_gdf.columns:
            available_scores.append('detector_score')
            scores_weights.append(self.config.detector_score_weight)
        if 'segmenter_score' in data_state.infer_gdf.columns:
            available_scores.append('segmenter_score')
            scores_weights.append(self.config.segmenter_score_weight)

        # Determine naming for the aggregated output (using the first tile’s name).
        tiles_path = data_state.infer_gdf[tile_path_column_name].unique().tolist()
        product_name, scale_factor, ground_resolution, _, _, _ = TileNameConvention().parse_name(
            Path(tiles_path[0]).name
        )
        gpkg_name = GeoPackageNameConvention.create_name(
            product_name=product_name,
            fold=infer_aoi_name,
            scale_factor=scale_factor,
            ground_resolution=ground_resolution
        )
        pre_aggregated_gpkg_name = GeoPackageNameConvention.create_name(
            product_name=product_name,
            fold=f'{infer_aoi_name}notaggregated',
            scale_factor=scale_factor,
            ground_resolution=ground_resolution
        )

        # Run the aggregator
        aggregator = Aggregator.from_gdf(
            output_path=self.output_path / gpkg_name,
            gdf=data_state.infer_gdf,
            tiles_paths_column=tile_path_column_name,
            polygons_column='geometry',
            scores_column=available_scores if available_scores else None,
            other_attributes_columns=list(data_state.infer_gdf_columns_to_pass),
            scores_weights=scores_weights if scores_weights else None,
            scores_weighting_method=self.config.scores_weighting_method,
            min_centroid_distance_weight=self.config.min_centroid_distance_weight,
            score_threshold=self.config.score_threshold,
            nms_threshold=self.config.nms_threshold,
            nms_algorithm=self.config.nms_algorithm,
            best_geom_keep_area_ratio=self.config.best_geom_keep_area_ratio,
            edge_band_buffer_percentage=self.config.edge_band_buffer_percentage,
            pre_aggregated_output_path=self.output_path / pre_aggregated_gpkg_name,
        )
        results_gdf = aggregator.polygons_gdf

        # Generate COCO output asynchronously and update the data state
        columns_to_pass = data_state.infer_gdf_columns_to_pass.union({'aggregator_score'})

        if results_gdf.empty:
            print(f"[AggregatorComponent] Skip COCO generation: empty results GeoDataFrame.")
            future_coco = None
        else:
            future_coco = generate_future_coco(
                future_key='infer_coco_path',
                executor=data_state.background_executor,
                component_name=self.name,
                component_id=self.component_id,
                description="Aggregator inference",
                gdf=results_gdf,
                tiles_paths_column=tile_path_column_name,
                polygons_column='geometry',
                scores_column='aggregator_score',
                categories_column='segmenter_class' if 'segmenter_class' in results_gdf.columns
                                else 'detector_class' if 'detector_class' in results_gdf.columns
                                else None,
                other_attributes_columns=columns_to_pass,
                output_path=self.output_path,
                use_rle_for_labels=False,
                n_workers=4,
                coco_categories_list=None
            )

        return self.update_data_state(data_state, results_gdf, columns_to_pass, future_coco)

    def update_data_state(self,
                         data_state: DataState,
                         results_gdf: gpd.GeoDataFrame,
                         columns_to_pass: set,
                         future_coco: tuple) -> DataState:
        # Register the component folder
        data_state = self.register_outputs_base(data_state)

        # Register the GeoPackage files by finding them in the output directory
        # This approach avoids needing access to gpkg_name variables
        for file_path in self.output_path.glob("*.gpkg"):
            if "notaggregated" in file_path.name:
                data_state.register_output_file(self.name, self.component_id, 'pre_aggregated_gpkg', file_path)
            else:
                data_state.register_output_file(self.name, self.component_id, 'gpkg', file_path)

        data_state.update_infer_gdf(results_gdf)
        data_state.infer_gdf_columns_to_pass = columns_to_pass
        data_state.side_processes.append(future_coco)

        return data_state
