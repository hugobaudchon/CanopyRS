from typing import List
import geopandas as gpd
from shapely.geometry import Polygon

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
        """Update data state with detector results"""
        data_state = self.register_outputs_base(data_state)
        data_state.update_infer_gdf(results_gdf)

        # Ensure detector_class and detector_score are passed
        columns_to_pass.add('detector_class')
        columns_to_pass.add('detector_score')
        # object_id_column_name should already be in columns_to_pass by default if it's in the GDF
        if object_id_column_name in results_gdf.columns:
            columns_to_pass.add(object_id_column_name)

        data_state.infer_gdf_columns_to_pass = columns_to_pass

        if future_coco is not None:
            data_state.side_processes.append(future_coco)

        return data_state

    def combine_as_gdf(self,
                       tiles_paths: List[str],
                       boxes: List[List[Polygon]],
                       scores: List[List[float]],
                       classes: List[List[int]]) -> gpd.GeoDataFrame:
        """
        Combines detector outputs into a GeoDataFrame.
        Each row is a detected object with its geometry, score, class, and tile path.
        """
        all_polygons_data = []
        current_object_id = 0  # Initialize a counter for unique object IDs

        for i, tile_path in enumerate(tiles_paths):
            tile_boxes = boxes[i]
            tile_scores = scores[i]
            tile_classes = classes[i]  # These will be '1' if detector_config.num_classes = 1

            for box_geom, score, cls_id in zip(tile_boxes, tile_scores, tile_classes):
                all_polygons_data.append({
                    'geometry': box_geom,
                    'tile_path': str(tile_path),
                    'detector_score': score,
                    'detector_class': cls_id,  # This will store the class ID from the detector
                    object_id_column_name: current_object_id # Assign unique ID
                })
                current_object_id += 1
        
        if not all_polygons_data:
            # Handle case with no detections
            return gpd.GeoDataFrame(columns=['geometry', 'tile_path', 'detector_score', 'detector_class', object_id_column_name], crs=None) # Or appropriate CRS

        gdf = gpd.GeoDataFrame(all_polygons_data, crs=None)  # Set CRS appropriately if known, or handle later
        
        # Ensure geometry is valid
        gdf['geometry'] = gdf['geometry'].buffer(0)
        gdf = gdf[gdf.is_valid]
        gdf = gdf[~gdf.is_empty]

        new_columns = {object_id_column_name, 'detector_score', 'detector_class'}
        
        print(f"DetectorComponent: Generated GDF with {len(gdf)} detections.")
        return gdf, new_columns
