from pathlib import Path

import geopandas as gpd

from geodataset.dataset import DetectionLabeledRasterCocoDataset, UnlabeledRasterDataset

from engine.components.base import BaseComponent
from engine.config_parsers import SegmenterConfig
from engine.data_state import DataState
from engine.models.registry import SEGMENTER_REGISTRY
from engine.utils import infer_aoi_name, generate_future_coco


class SegmenterComponent(BaseComponent):
    name = 'segmenter'

    def __init__(self, config: SegmenterConfig, parent_output_path: str, component_id: int):
        super().__init__(config, parent_output_path, component_id)
        if config.model in SEGMENTER_REGISTRY:
            self.segmenter = SEGMENTER_REGISTRY[config.model](config)
        else:
            raise ValueError(f'Invalid detector model {config.model}')

    def __call__(self, data_state: DataState) -> DataState:
        data_paths = [data_state.tiles_path]
        if self.segmenter.REQUIRES_BOX_PROMPT:
            assert data_state.infer_coco_path is not None, \
                ("The selected Segmenter model requires a COCO file with boxes to prompt. Either input it,"
                 " add a tilerizer before the segmenter, add a detector before the"
                 " segmenter in the pipeline, or choose a segmenter model that doesn't require boxes prompts.")
            data_paths.append(Path(data_state.infer_coco_path).parent)
            dataset = DetectionLabeledRasterCocoDataset(
                fold=infer_aoi_name,
                root_path=data_paths,
                box_padding_percentage=self.config.box_padding_percentage,
                transform=None
            )
        else:
            dataset = UnlabeledRasterDataset(
                fold=infer_aoi_name,
                root_path=data_paths,
                transform=None
            )

        tiles_paths, tiles_masks_polygons, tiles_masks_scores = self.segmenter.infer_on_dataset(dataset)

        attributes_data = {}
        for attribute_name in data_state.infer_gdf_columns_to_pass:
            attributes_data[attribute_name] = get_attribute_from_dataset(dataset, attribute_name, tiles_paths, tiles_masks_polygons)

        future_coco = generate_future_coco(
            future_key='infer_coco_path',
            description="Segmenter inference",
            tiles_paths=tiles_paths,
            tile_names_order_reference=data_state.tiles_names,
            polygons=tiles_masks_polygons,
            scores=tiles_masks_scores,
            categories=None,
            other_attributes={},
            output_path=self.output_path,
            use_rle_for_labels=True,
            n_workers=2,
            coco_categories_list=None
        )

        gdf_items = []
        for i in range(len(tiles_paths)):
            for j in range(len(tiles_masks_polygons[i])):
                data = {
                    'tiles_path': tiles_paths[i],
                    'geometry': tiles_masks_polygons[i][j],
                    'segmenter_score': tiles_masks_scores[i][j]
                }

                for attribute_name, attribute in attributes_data.items():
                    data[attribute_name] = attribute[i][j]

                gdf_items.append(data)

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
        data_state.infer_gdf_columns_to_pass.append('segmenter_score')
        data_state.side_processes.append(future_coco)
        return data_state


def get_attribute_from_dataset(dataset, attribute_name, tiles_paths, tiles_masks_polygons):
    if attribute_name in dataset.tiles[dataset.tiles_path_to_id_mapping[tiles_paths[0].name]]['labels'][0]['other_attributes']:
        attribute = []
        for i, tile_path in enumerate(tiles_paths):
            tile_boxes_attribute = []
            if tile_path.name in dataset.tiles_path_to_id_mapping:
                tile_id = dataset.tiles_path_to_id_mapping[tile_path.name]
                for annotation in dataset.tiles[tile_id]['labels']:
                    tile_boxes_attribute.append(annotation['other_attributes'][attribute_name])

            attribute.append(tile_boxes_attribute)
    else:
        attribute = [[None for _ in tile_masks] for tile_masks in tiles_masks_polygons]
    return attribute