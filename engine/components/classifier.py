import geopandas as gpd
from pathlib import Path

from geodataset.dataset import UnlabeledRasterDataset, ClassificationLabeledRasterCocoDataset

from engine.components.base import BaseComponent
from engine.config_parsers import ClassifierConfig
from engine.data_state import DataState
from engine.models.registry import CLASSIFIER_REGISTRY
from engine.models.utils import collate_fn_images
from engine.utils import infer_aoi_name, generate_future_coco


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
        # TODO: include polygon id?
        infer_ds = ClassificationLabeledRasterCocoDataset(
            root_path=data_state.tiles_path,
            transform=None,
            fold=infer_aoi_name  # Only use tiles with the inference AOI
        )

        # Run inference
        tiles_paths, class_scores, class_predictions = self.classifier.infer(
            infer_ds,
            collate_fn_images
        )

        # Build GeoDataFrame with results
        gdf_items = []
        for i, tile_path in enumerate(tiles_paths):
            # For tile-level classification, we use the tile center as geometry
            gdf_items.append({
                'tile_path': tile_path,
                'geometry': None,  # Will be populated with tile center or extent
                'classifier_score': max(class_scores[i]) if class_scores[i] else 0.0,
                'classifier_class': class_predictions[i],
                'classifier_scores': class_scores[i],  # All class scores
            })

        # Create GeoDataFrame
        results_gdf = gpd.GeoDataFrame(
            data=gdf_items,
            geometry='geometry',
            crs=None
        )

        # Add tile centroids as geometry
        self._add_tile_centroids(results_gdf)

        # Update columns to be passed forward
        columns_to_pass = data_state.infer_gdf_columns_to_pass.union({
            'classifier_score',
            'classifier_class',
            'classifier_scores'
        })

        # Generate COCO format data (for compatibility with other components)
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

    def update_data_state(self,
                        data_state: DataState,
                        results_gdf: gpd.GeoDataFrame,
                        columns_to_pass: set,
                        future_coco: tuple) -> DataState:
        """Update data state with classification results"""
        # Register the component folder and outputs
        data_state = self.register_outputs_base(data_state)

        # Update the inference GeoDataFrame
        data_state.infer_gdf = results_gdf
        data_state.infer_gdf_columns_to_pass = columns_to_pass
        data_state.side_processes.append(future_coco)

        return data_state
