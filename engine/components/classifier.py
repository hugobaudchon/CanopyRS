import geopandas as gpd
from pathlib import Path
import shapely
from shapely import Polygon

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
        infer_result = self.classifier.infer(
            infer_ds,
            collate_fn_classification
        )

        # Check if we got object IDs back (4 values instead of 3)
        if len(infer_result) == 4:
            tiles_paths, class_scores, class_predictions, object_ids = infer_result
            print(f"Retrieved {len(object_ids)} object IDs from classifier")
        else:
            tiles_paths, class_scores, class_predictions = infer_result
            object_ids = None
            print("No object IDs returned from classifier")

        # Use the combine_as_gdf method for consistent handling
        results_gdf, new_columns = self.combine_as_gdf(
            data_state.infer_gdf, tiles_paths, class_scores, class_predictions, object_ids
        )

        # If no geometries were preserved from previous components, add tile centroids
        if all(geometry is None for geometry in results_gdf['geometry']):
            self._add_tile_centroids(results_gdf)

        # Update columns to be passed forward
        columns_to_pass = data_state.infer_gdf_columns_to_pass.union(new_columns)

        # Check if we're dealing with polygon tiles by examining the first tile name
        first_tile_path = Path(results_gdf['tile_path'].iloc[0])
        is_polygon_tile = 'polygontile' in first_tile_path.name

        if is_polygon_tile:
            # For polygon tiles, use a custom function to generate COCO output
            future_coco = self._generate_future_coco_for_polygon_tiles(
                results_gdf=results_gdf,
                columns_to_pass=columns_to_pass
            )
        else:
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

    def _generate_future_coco_for_polygon_tiles(self, results_gdf, columns_to_pass):
        """Custom method to handle COCO generation for polygon tiles"""
        from concurrent.futures import ProcessPoolExecutor
        from geodataset.utils import CocoNameConvention
        from engine.utils import generate_coco

        # Parse the first tile name to extract components
        first_tile_path = Path(results_gdf['tile_path'].iloc[0])
        tile_name = first_tile_path.name

        # Parse polygon tile name - format: date_productname_polygontile_fold_grXpXX_index.tif
        parts = tile_name.split('_')
        if len(parts) >= 5:
            # Extract product name (can be multiple parts)
            product_name_parts = []
            for i in range(1, len(parts)):
                if parts[i] != 'polygontile' and not parts[i].startswith('gr') and not parts[i].startswith('sc'):
                    product_name_parts.append(parts[i])
                else:
                    break
            product_name = '_'.join(product_name_parts)

            # Find fold/aoi (usually the part after 'polygontile')
            polygon_idx = parts.index('polygontile') if 'polygontile' in parts else -1
            aoi = parts[polygon_idx + 1] if polygon_idx >= 0 and polygon_idx + 1 < len(parts) else 'infer'

            # Find ground resolution
            gr_part = next((p for p in parts if p.startswith('gr')), None)
            ground_resolution = None
            if gr_part:
                try:
                    ground_resolution = float(gr_part[2:].replace('p', '.'))
                except ValueError:
                    ground_resolution = None

            # Find scale factor
            sc_part = next((p for p in parts if p.startswith('sc')), None)
            scale_factor = None
            if sc_part:
                try:
                    scale_factor = float(sc_part[2:].replace('p', '.'))
                except ValueError:
                    scale_factor = None
        else:
            # Default values if parsing fails
            product_name = "classifier_output"
            aoi = "infer"
            ground_resolution = None
            scale_factor = None

        # Create COCO output name
        coco_output_name = CocoNameConvention().create_name(
            product_name=product_name,
            fold=aoi,
            scale_factor=scale_factor,
            ground_resolution=ground_resolution
        )

        coco_output_path = self.output_path / coco_output_name

        # Use the same function as in utils.py but without parsing tile name
        future_coco_process = ProcessPoolExecutor(max_workers=1).submit(
            generate_coco,
            description="Classifier inference",
            gdf=results_gdf,
            tiles_paths_column='tile_path',
            polygons_column='geometry',
            scores_column='classifier_score',
            categories_column='classifier_class',
            other_attributes_columns=columns_to_pass,
            coco_output_path=coco_output_path,
            use_rle_for_labels=False,
            n_workers=4,
            coco_categories_list=None
        )

        future_coco = (
            'infer_coco_path',
            future_coco_process,
            {
                'component_name': self.name,
                'component_id': self.component_id,
                'file_type': 'coco',
                'expected_path': str(coco_output_path)
            }
        )

        return future_coco

    def _add_tile_centroids(self, gdf):
        """Add tile centroid geometry to the GeoDataFrame"""
        # This is a placeholder - in a real implementation,
        # you would calculate the geographic center of each tile
        # based on its metadata or transform
        pass

    @staticmethod
    def combine_as_gdf(infer_gdf, tiles_paths, class_scores, class_predictions, object_ids=None) -> (gpd.GeoDataFrame, set):
        """Convert classifier predictions to a GeoDataFrame"""

        # If we have an existing GDF with geometries, preserve them
        if infer_gdf is not None:
            # Make a copy of the input GDF to preserve all columns and geometries
            results_gdf = infer_gdf.copy()

            # Add classifier results columns
            results_gdf['classifier_class'] = None
            results_gdf['classifier_score'] = 0.0
            results_gdf['classifier_scores'] = None

            if object_ids is not None:
                print(f"Using object IDs for direct mapping of classifier results")

                # Create a mapping from object ID to classifier results
                id_to_classifier = {}
                for i, obj_id in enumerate(object_ids):
                    id_to_classifier[obj_id] = {
                        'class': class_predictions[i],
                        'score': max(class_scores[i]) if class_scores[i] else 0.0,
                        'scores': class_scores[i]
                    }

                # Apply the mapping to the GDF
                for i, row in results_gdf.iterrows():
                    obj_id = row['canopyrs_object_id']
                    if obj_id in id_to_classifier:
                        results_gdf.at[i, 'classifier_class'] = id_to_classifier[obj_id]['class']
                        results_gdf.at[i, 'classifier_score'] = id_to_classifier[obj_id]['score']
                        results_gdf.at[i, 'classifier_scores'] = id_to_classifier[obj_id]['scores']

                # Count how many rows matched by object ID
                matched_count = sum(1 for obj_id in results_gdf['canopyrs_object_id'] if obj_id in id_to_classifier)
                print(f"Matched {matched_count} out of {len(results_gdf)} rows using object IDs")
            else:
                # Fall back to tile path matching if no object IDs are available
                print("No object IDs available. Falling back to tile path matching.")

                # Create a mapping from tile path to classifier results
                tile_to_classifier = {}
                for i, tile_path in enumerate(tiles_paths):
                    tile_to_classifier[tile_path] = {
                        'class': class_predictions[i],
                        'score': max(class_scores[i]) if class_scores[i] else 0.0,
                        'scores': class_scores[i]
                    }

                # Find paths that work in both datasets
                common_paths = set(results_gdf['tile_path']) & set(tile_to_classifier.keys())
                if common_paths:
                    print(f"Found {len(common_paths)} common tile paths for matching.")
                    for i, row in results_gdf.iterrows():
                        tile_path = row['tile_path']
                        if tile_path in tile_to_classifier:
                            results_gdf.at[i, 'classifier_class'] = tile_to_classifier[tile_path]['class']
                            results_gdf.at[i, 'classifier_score'] = tile_to_classifier[tile_path]['score']
                            results_gdf.at[i, 'classifier_scores'] = tile_to_classifier[tile_path]['scores']
                else:
                    print("No common paths found. Unable to match classifier results.")

        else:
            # If we have no existing GDF, create a new one
            print("No existing GDF found. Creating new GDF with classifier results.")
            gdf_items = []
            for i, tile_path in enumerate(tiles_paths):
                # Get object ID if available
                obj_id = object_ids[i] if object_ids is not None else i

                gdf_items.append({
                    'tile_path': tile_path,
                    'geometry': shapely.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),  # Placeholder
                    'classifier_score': max(class_scores[i]) if class_scores[i] else 0.0,
                    'classifier_class': class_predictions[i],
                    'classifier_scores': class_scores[i],
                    object_id_column_name: obj_id
                })

            # Create GeoDataFrame
            results_gdf = gpd.GeoDataFrame(
                data=gdf_items,
                geometry='geometry',
                crs=None
            )

        # Report on classifier results
        classifier_count = results_gdf['classifier_class'].notnull().sum()
        print(f"Added classifier results to {classifier_count} out of {len(results_gdf)} rows")

        new_columns = {'classifier_score', 'classifier_class', 'classifier_scores'}

        return results_gdf, new_columns

    def update_data_state(self,
                         data_state: DataState,
                         results_gdf: gpd.GeoDataFrame,
                         columns_to_pass: set,
                         future_coco: tuple) -> DataState:
        """Update data state with classification results"""
        # Register the component folder and outputs
        data_state = self.register_outputs_base(data_state)

        # Debug information
        print(f"Classifier results GDF columns: {results_gdf.columns.tolist()}")
        print(f"Classifier results GDF shape: {results_gdf.shape}")

        if data_state.infer_gdf is not None:
            print(f"Existing infer_gdf columns: {data_state.infer_gdf.columns.tolist()}")
            print(f"Existing infer_gdf shape: {data_state.infer_gdf.shape}")

            # Add classifier columns to the existing GDF
            merged_gdf = data_state.infer_gdf.copy()

            # Ensure object_id_column_name exists in both GDFs
            if object_id_column_name not in merged_gdf.columns:
                merged_gdf[object_id_column_name] = range(len(merged_gdf))

            # First, create the new columns with default values
            merged_gdf['classifier_class'] = None
            merged_gdf['classifier_score'] = 0.0
            # For classifier_scores (which is a list), use a Python object column
            merged_gdf['classifier_scores'] = None

            # Create a mapping of tile_path + object_id to classifier results
            classifier_results = {}
            for _, row in results_gdf.iterrows():
                key = (row['tile_path'], row[object_id_column_name])
                classifier_results[key] = {
                    'classifier_class': row['classifier_class'],
                    'classifier_score': row['classifier_score'],
                    'classifier_scores': row['classifier_scores']
                }

            # Apply classifier results to the merged GDF row by row
            for i, row in merged_gdf.iterrows():
                key = (row['tile_path'], row[object_id_column_name])
                if key in classifier_results:
                    # Set each column individually to avoid the list issue
                    merged_gdf.at[i, 'classifier_class'] = classifier_results[key]['classifier_class']
                    merged_gdf.at[i, 'classifier_score'] = classifier_results[key]['classifier_score'] 
                    merged_gdf.at[i, 'classifier_scores'] = classifier_results[key]['classifier_scores']

            # Update the data state with the merged GDF
            data_state.infer_gdf = merged_gdf
        else:
            # If no existing GDF, use the classifier results directly
            data_state.infer_gdf = results_gdf

        # Ensure these columns are passed forward in the pipeline
        columns_to_pass.update({'classifier_score', 'classifier_class', 'classifier_scores'})
        data_state.infer_gdf_columns_to_pass = columns_to_pass

        # Add COCO generation to side processes
        if future_coco is not None:
            data_state.side_processes.append(future_coco)

        # Save the GeoPackage directly for debugging
        gpkg_path = self.output_path / f"{self.name}_results.gpkg"
        data_state.infer_gdf.to_file(gpkg_path, driver="GPKG")
        print(f"Saved classifier results to: {gpkg_path}")

        return data_state
