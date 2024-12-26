import geopandas as gpd

from geodataset.aggregator import Aggregator

from engine.components.base import BaseComponent
from engine.config_parsers import AggregatorConfig
from engine.data_state import DataState


class AggregatorComponent(BaseComponent):
    name = 'aggregator'

    def __init__(self, config: AggregatorConfig, parent_output_path: str, component_id: int):
        super().__init__(config, parent_output_path, component_id)

    def run(self, data_state: DataState) -> DataState:
        agg_dict = {'geometry': lambda x: list(x)}

        for column in data_state.results_gdf_columns_to_pass:
            if column in data_state.results_gdf.columns:
                agg_dict[column] = list

        grouped_gdf = data_state.results_gdf.groupby('tiles_path').agg(agg_dict)
        tiles_path = grouped_gdf.index.tolist()
        grouped_gdf = grouped_gdf.reset_index().to_dict(orient='list')

        scores = {}
        scores_weights = {}
        if 'detector_score' in grouped_gdf:
            scores['detector_score'] = grouped_gdf['detector_score']
            scores_weights['detector_score'] = self.config.detector_score_weight
        if 'segmenter_score' in grouped_gdf:
            scores['segmenter_score'] = grouped_gdf['segmenter_score']
            scores_weights['segmenter_score'] = self.config.segmenter_score_weight

        other_attributes = {}
        for column_name in data_state.results_gdf_columns_to_pass:
            if column_name not in ['detector_score', 'segmenter_score']:
                other_attributes[column_name] = grouped_gdf[column_name]

        aggregator = Aggregator.from_polygons(
            output_path=self.output_path / 'temp_name.gpkg',
            tiles_paths=tiles_path,
            polygons=grouped_gdf['geometry'],
            scores=scores,
            other_attributes=other_attributes,
            scores_weights=scores_weights,
            scores_weighting_method=self.config.scores_weighting_method,
            score_threshold=self.config.score_threshold,
            nms_threshold=self.config.nms_threshold,
            nms_algorithm=self.config.nms_algorithm
        )

        results_gdf = aggregator.polygons_gdf
        results_gdf.rename(columns={'score': 'score_aggregator'}, inplace=True)

        return self.update_data_state(data_state, results_gdf)

    def update_data_state(self,
                         data_state: DataState,
                         results_gdf: gpd.GeoDataFrame) -> DataState:
        data_state.results_gdf = results_gdf

        return data_state
