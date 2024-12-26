import geopandas as gpd

from geodataset.dataset import UnlabeledRasterDataset

from engine.components.base import BaseComponent
from engine.config_parsers import DetectorConfig
from engine.data_state import DataState
from engine.models.registry import DETECTOR_REGISTRY
from engine.models.utils import collate_fn_images


class DetectorComponent(BaseComponent):
    name = 'detector'

    def __init__(self, config: DetectorConfig, parent_output_path: str, component_id: int):
        super().__init__(config, parent_output_path, component_id)
        if config.model in DETECTOR_REGISTRY:
            self.detector = DETECTOR_REGISTRY[config.model](config)
        else:
            raise ValueError(f'Invalid detector model {config.model}')

    def run(self, data_state: DataState) -> DataState:
        infer_ds = UnlabeledRasterDataset(data_state.tiles_path,
                                          transform=None,
                                          fold='infer')                 # TODO change 'infer' to a variable

        tiles_paths, boxes, boxes_scores, classes = self.detector.infer(infer_ds, collate_fn_images)

        print(len(boxes), len(boxes[0]))

        gdf_items = []
        for i in range(len(tiles_paths)):
            for j in range(len(boxes[i])):
                gdf_items.append({
                    'tiles_path': tiles_paths[i],
                    'geometry': boxes[i][j],
                    'detector_score': boxes_scores[i][j],
                    'detector_class': classes[i][j]
                })

        results_gdf = gpd.GeoDataFrame(        # TODO should I store this as results_gdf or as multiple lists in data_state? Maybe results_gdf should be kept for the aggregated results, in CRS coordinates?
            data=gdf_items,
            geometry='geometry',
            crs=None
        )

        return self.update_data_state(data_state, results_gdf)

    def update_data_state(self,
                          data_state: DataState,
                          results_gdf: gpd.GeoDataFrame) -> DataState:
        data_state.results_gdf = results_gdf
        data_state.results_gdf_columns_to_pass.extend(['detector_score', 'detector_class'])
        return data_state
