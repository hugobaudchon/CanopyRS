import geopandas as gpd

from geodataset.dataset import UnlabeledRasterDataset

from engine.components.base import BaseComponent
from engine.config_parsers import DetectorConfig
from engine.data_state import DataState
from engine.models.registry import DETECTOR_REGISTRY
from engine.models.utils import collate_fn_images
from engine.utils import generate_future_coco, object_id_column_name


class DetectorComponent(BaseComponent):
    name = 'detector'

    def __init__(self, config: DetectorConfig, parent_output_path: str, component_id: int):
        super().__init__(config, parent_output_path, component_id)
        if config.model in DETECTOR_REGISTRY:
            self.detector = DETECTOR_REGISTRY[config.model](config)
        else:
            raise ValueError(f'Invalid detector model {config.model}')

    def __call__(self, data_state: DataState) -> DataState:
        # Find the tiles
        infer_ds = UnlabeledRasterDataset(
            root_path=data_state.tiles_path,
            transform=None, # transform=None because the detector class will apply its own transform
            fold=None   # load all tiles (they could have either groundtruth or infer aois)
        )

        # Run inference
        tiles_paths, boxes, boxes_scores, classes = self.detector.infer(infer_ds, collate_fn_images)

        # Combine results into a GeoDataFrame
        results_gdf, new_columns = self.combine_as_gdf(tiles_paths, boxes, boxes_scores, classes)

        # Generate COCO output asynchronously and update the data state
        columns_to_pass = data_state.infer_gdf_columns_to_pass.union(new_columns)

        future_coco = generate_future_coco(
            future_key='infer_coco_path',
            component_name=self.name,
            component_id=self.component_id,
            description="Detector inference",
            gdf=results_gdf,
            tiles_paths_column='tile_path',
            polygons_column='geometry',
            scores_column='detector_score',
            categories_column='detector_class',
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
        data_state.update_infer_gdf(results_gdf)
        data_state.infer_gdf_columns_to_pass = columns_to_pass
        data_state.side_processes.append(future_coco)
        return data_state

    @staticmethod
    def combine_as_gdf(tiles_paths, boxes, boxes_scores, classes) -> (gpd.GeoDataFrame, set):
        gdf_items = []
        for i in range(len(tiles_paths)):
            for j in range(len(boxes[i])):
                gdf_items.append({
                    'tile_path': tiles_paths[i],
                    'geometry': boxes[i][j],
                    'detector_score': boxes_scores[i][j],
                    'detector_class': classes[i][j]
                })

        gdf = gpd.GeoDataFrame(
            data=gdf_items,
            geometry='geometry',
            crs=None
        )

        # Assigning a unique ID to each object detected
        gdf[object_id_column_name] = range(len(gdf))

        new_columns = {object_id_column_name, 'detector_score', 'detector_class'}

        return gdf, new_columns
