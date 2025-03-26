import geopandas as gpd
from pathlib import Path

from geodataset.dataset import ClassificationLabeledRasterCocoDataset

from engine.components.base import BaseComponent
from engine.config_parsers import ClassifierConfig
from engine.data_state import DataState
from engine.models.registry import CLASSIFIER_REGISTRY
from engine.models.utils import collate_fn_classification
from engine.utils import infer_aoi_name, generate_future_coco, object_id_column_name


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
        # Create dataset from tile paths
        infer_ds = ClassificationLabeledRasterCocoDataset(
            root_path=data_state.tiles_path,
            transform=None,
            fold=infer_aoi_name,
            include_polygon_id=True,
        )

        # Run inference
        tiles_paths, class_scores, class_predictions = self.classifier.infer(
            infer_ds,
            collate_fn_classification
        )

        # Use the combine_as_gdf method for consistent handling
        results_gdf, new_columns = self.combine_as_gdf(
            data_state.infer_gdf, tiles_paths, class_scores, class_predictions
        )

        # If no geometries were preserved from previous components, add tile centroids
        if all(geometry is None for geometry in results_gdf['geometry']):
            self._add_tile_centroids(results_gdf)

        # Update columns to be passed forward
        columns_to_pass = data_state.infer_gdf_columns_to_pass.union(new_columns)

        # Generate COCO format data
        future_coco = generate_future_coco(
            future_key='infer_coco_path',
            component_name=self.name,
            component_id=self.component_id,
            description="Classifier inference",
            gdf=results_gdf,
            tiles_paths_column='tile_path',
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

    def _add_tile_centroids(self, gdf):
        """Add tile centroid geometry to the GeoDataFrame"""
        # This is a placeholder - in a real implementation,
        # you would calculate the geographic center of each tile
        # based on its metadata or transform
        pass

    @staticmethod
    def combine_as_gdf(infer_gdf, tiles_paths, class_scores, class_predictions) -> (gpd.GeoDataFrame, set):
        """Convert classifier predictions to a GeoDataFrame"""
        gdf_items = []
        
        # If we have an existing GDF with geometries, use those
        if infer_gdf is not None:
            # Group by tile path to match predictions with existing geometries
            for i, tile_path in enumerate(tiles_paths):
                matching_rows = infer_gdf[infer_gdf['tile_path'] == tile_path]
                
                if not matching_rows.empty:
                    # For each existing polygon in this tile, add classification results
                    for _, row in matching_rows.iterrows():
                        gdf_items.append({
                            'tile_path': tile_path,
                            'geometry': row['geometry'],
                            'classifier_score': max(class_scores[i]) if class_scores[i] else 0.0,
                            'classifier_class': class_predictions[i],
                            'classifier_scores': class_scores[i],
                            object_id_column_name: row[object_id_column_name]
                        })
        else:
            # No existing geometries, create new entries with tile centroids
            for i, tile_path in enumerate(tiles_paths):
                gdf_items.append({
                    'tile_path': tile_path,
                    'geometry': None,  # Will be populated with tile center
                    'classifier_score': max(class_scores[i]) if class_scores[i] else 0.0,
                    'classifier_class': class_predictions[i],
                    'classifier_scores': class_scores[i]
                })

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            data=gdf_items,
            geometry='geometry',
            crs=None
        )

        # Add object IDs if not present
        if object_id_column_name not in gdf.columns:
            gdf[object_id_column_name] = range(len(gdf))

        new_columns = {'classifier_score', 'classifier_class', 'classifier_scores', object_id_column_name}

        return gdf, new_columns

    def update_data_state(self,
                        data_state: DataState,
                        results_gdf: gpd.GeoDataFrame,
                        columns_to_pass: set,
                        future_coco: tuple) -> DataState:
        """Update data state with classification results"""
        # Register the component folder and outputs
        data_state = self.register_outputs_base(data_state)

        # Use update_infer_gdf instead of direct assignment
        data_state.update_infer_gdf(results_gdf)
        data_state.infer_gdf_columns_to_pass = columns_to_pass
        data_state.side_processes.append(future_coco)

        return data_state
