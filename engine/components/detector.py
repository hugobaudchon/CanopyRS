import geopandas as gpd

from geodataset.dataset import UnlabeledRasterDataset

from engine.components.base import BaseComponent
from engine.config_parsers import DetectorConfig
from engine.data_state import DataState
from engine.models.registry import DETECTOR_REGISTRY
from engine.models.utils import collate_fn_images
from engine.utils import generate_future_coco


class DetectorComponent(BaseComponent):
    name = 'detector'

    def __init__(self, config: DetectorConfig, parent_output_path: str, component_id: int):
        super().__init__(config, parent_output_path, component_id)
        if config.model in DETECTOR_REGISTRY:
            self.detector = DETECTOR_REGISTRY[config.model](config)
        else:
            raise ValueError(f'Invalid detector model {config.model}')

    def __call__(self, data_state: DataState) -> DataState:
        infer_ds = UnlabeledRasterDataset(
            root_path=data_state.tiles_path,
            transform=None,
            fold=None   # load all tiles (they could have either groundtruth or infer aois)
        )

        tiles_paths, boxes, boxes_scores, classes = self.detector.infer(infer_ds, collate_fn_images)

        future_coco = generate_future_coco(
            future_key='infer_coco_path',
            description="Detector inference",
            tiles_paths=tiles_paths,
            tile_names_order_reference=data_state.tiles_names,
            polygons=boxes,
            scores=boxes_scores,
            categories=classes,
            other_attributes={},
            output_path=self.output_path,
            use_rle_for_labels=True,
            n_workers=2,
            coco_categories_list=None
        )

        gdf_items = []
        for i in range(len(tiles_paths)):
            for j in range(len(boxes[i])):
                gdf_items.append({
                    'tiles_path': tiles_paths[i],
                    'geometry': boxes[i][j],
                    'detector_score': boxes_scores[i][j],
                    'detector_class': classes[i][j]
                })

        results_gdf = gpd.GeoDataFrame(
            data=gdf_items,
            geometry='geometry',
            crs=None
        )

        return self.update_data_state(data_state, results_gdf, future_coco)

    def update_data_state(self,
                          data_state: DataState,
                          results_gdf: gpd.GeoDataFrame,
                          future_coco: tuple) -> DataState:
        data_state.infer_gdf = results_gdf
        data_state.infer_gdf_columns_to_pass.extend(['detector_score', 'detector_class'])
        data_state.side_processes.append(future_coco)
        return data_state
