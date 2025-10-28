import warnings
from typing import List, Any, Tuple, Set
import geopandas as gpd
import pandas as pd
from pathlib import Path

from geodataset.dataset import InstanceSegmentationLabeledRasterCocoDataset
from geodataset.utils import GeoPackageNameConvention

from canopyrs.engine.components.base import BaseComponent
from canopyrs.engine.config_parsers import ClassifierConfig
from canopyrs.engine.data_state import DataState
from canopyrs.engine.models.registry import CLASSIFIER_REGISTRY
from canopyrs.engine.models.utils import collate_fn_infer_image_masks
from canopyrs.engine.utils import generate_future_coco, object_id_column_name, infer_aoi_name, parse_product_name, \
    tile_path_column_name


class ClassifierComponent(BaseComponent):
    """Component for classifying tiles"""
    name = 'classifier'

    def __init__(self, config: ClassifierConfig, parent_output_path: str, component_id: int):
        super().__init__(config, parent_output_path, component_id)
        if config.model in CLASSIFIER_REGISTRY:
            self.classifier = CLASSIFIER_REGISTRY[config.model](config)
        else:
            raise ValueError(f'Invalid classifier model {config.model}')

    def __call__(self, data_state: DataState) -> DataState:
        """Run classification on tiles"""

        if not data_state.tiles_path or not data_state.infer_coco_path:
            raise ValueError("ClassifierComponent requires tiles_path and infer_coco_path from a previous (polygon) tilerizer.")

        # Create dataset from tile paths and associated coco
        infer_ds = InstanceSegmentationLabeledRasterCocoDataset(
            root_path=[data_state.tiles_path, Path(data_state.infer_coco_path).parent],
            transform=None,
            fold=infer_aoi_name,
            other_attributes_names_to_pass=[object_id_column_name]
        )

        infer_result = self.classifier.infer(
            infer_ds,
            collate_fn_infer_image_masks
        )

        tiles_paths, class_scores, class_predictions, object_ids = infer_result

        # Use the combine_as_gdf method for consistent handling
        results_gdf, columns_to_pass = self.combine_as_gdf(
            data_state,
            tiles_paths,
            class_scores,
            class_predictions,
            object_ids
        )

        # Generate COCO format data
        future_coco = generate_future_coco(
            future_key='infer_coco_path',
            executor=data_state.background_executor,
            component_name=self.name,
            component_id=self.component_id,
            description="Classifier inference",
            gdf=results_gdf,
            tiles_paths_column=tile_path_column_name,
            polygons_column='geometry',
            scores_column='classifier_score',
            categories_column='classifier_class',
            other_attributes_columns=columns_to_pass,
            output_path=self.output_path,
            use_rle_for_labels=False,
            n_workers=4,
            coco_categories_list=None
        )

        return self.update_data_state(data_state, results_gdf, columns_to_pass, future_coco)

    def combine_as_gdf(self,
                       data_state: DataState,
                       tiles_paths: List[str],
                       class_scores: List[List[float]],
                       class_predictions: List[int],
                       object_ids: List[Any]) -> Tuple[gpd.GeoDataFrame, Set[str]]:
        """
        Combines classifier outputs with the existing infer_gdf from data_state.
        """

        # Prepare classifier results for merging
        classifier_data = {
            object_id_column_name: object_ids,
            'classifier_class': class_predictions,
            'classifier_score': [scores[pred_idx] 
                                 for scores, pred_idx in zip(class_scores, class_predictions)],
            'classifier_scores': class_scores  # Store the full list of scores for all classes
        }
        results_df = pd.DataFrame(classifier_data)

        new_columns = {'classifier_score', 'classifier_class', 'classifier_scores'}

        if data_state.infer_gdf is not None:
            # Merge results with the original infer_gdf
            original_infer_gdf = data_state.infer_gdf.copy()
            if all(obj_id is not None for obj_id in object_ids):
                merged_gdf = original_infer_gdf.merge(results_df, on=object_id_column_name, how='left')
            elif all(isinstance(tile_path, str) for tile_path in tiles_paths):
                # If object IDs are not available, use tile paths for merging, as there should be a one-to-one correspondence with polygons being classified
                results_df[tile_path_column_name] = tiles_paths
                del results_df[object_id_column_name]  # Remove object_id column if it exists
                merged_gdf = original_infer_gdf.merge(results_df, on=tile_path_column_name, how='left')
                results_df[object_id_column_name] = list(range(len(results_df)))  # Add new object IDs
                new_columns.add(object_id_column_name)
            else:
                raise ValueError(
                    f"Neither object IDs nor tile paths are available for merging. "
                    f"Please make sure your inputs contain either {object_id_column_name}"
                    f" or {tile_path_column_name} columns/attributes."
                )

            merged_gdf = gpd.GeoDataFrame(merged_gdf, geometry='geometry', crs=original_infer_gdf.crs)  # df to gdf

            # Check for unmerged items (polygons in infer_gdf that didn't get a classification result)
            if len(merged_gdf) < len(original_infer_gdf):
                warnings.warn(
                    f"Warning: {len(original_infer_gdf) - len(merged_gdf)} polygons from infer_gdf were not"
                    f" found in classifier results during merge."
                )
        else:
            # If there are no infer_gdf yet, create a new GeoDataFrame with the results
            merged_gdf = gpd.GeoDataFrame(results_df, geometry=None, crs=None)
            warnings.warn("No infer_gdf found in data_state. The resulting gdf won't have a CRS.")
        
        unclassified_rows = merged_gdf['classifier_class'].isnull().sum()
        if unclassified_rows > 0:
             warnings.warn(f"Classifier: {unclassified_rows} polygons remain unclassified after merge (NaN in classifier_class).")

        return merged_gdf, new_columns

    def update_data_state(self,
                          data_state: DataState,
                          results_gdf: gpd.GeoDataFrame,
                          columns_to_pass: set,
                          future_coco: tuple) -> DataState:
        """Update data state with classification results"""
        # Register the component folder and outputs
        data_state = self.register_outputs_base(data_state)

        product_name, scale_factor, ground_resolution, aoi = parse_product_name(results_gdf[tile_path_column_name].iloc[0])
        gpkg_name = GeoPackageNameConvention.create_name(
            product_name=product_name,
            fold=infer_aoi_name
        )

        gpkg_path = self.output_path / gpkg_name
        results_gdf.to_file(gpkg_path, driver='GPKG')
        if not results_gdf.empty:
            data_state.register_output_file(self.name, self.component_id, 'gpkg', gpkg_path)
            data_state.update_infer_gdf(results_gdf)
        else:
            print("Classifier: Results GDF is empty, data_state.infer_gdf not updated by classifier.")
        data_state.infer_gdf_columns_to_pass = columns_to_pass
        if future_coco is not None:
            data_state.side_processes.append(future_coco)

        return data_state
