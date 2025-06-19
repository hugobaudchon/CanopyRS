# engine/benchmark/classifier/benchmark.py
from pathlib import Path
import pandas as pd
import geopandas as gpd
import json

from engine.benchmark.classifier.evaluator import ClassifierCocoEvaluator
# from engine.benchmark.classifier.find_optimal_parameters import find_optimal_classifier_params_placeholder # If needed later
from engine.config_parsers import PipelineConfig, InferIOConfig # If running pipeline
# from engine.pipeline import Pipeline # If running pipeline
from engine.utils import merge_coco_jsons, object_id_column_name # Assuming object_id_column_name is relevant
from geodataset.utils import COCOGenerator # For creating COCO from GDF

# from data.segmentation_classification.preprocessed_datasets import DATASET_REGISTRY # If using dataset registry

# Define your canonical class list here or import from a central place
CANONICAL_COCO_CATEGORIES = [
    # {"id": 1, "name": "class1", "supercategory": "object"},
    # {"id": 2, "name": "class2", "supercategory": "object"},
    # Add all your classes here
]
if not CANONICAL_COCO_CATEGORIES:
    print("WARNING: CANONICAL_COCO_CATEGORIES is not defined in benchmark.py. Evaluation might fail or use arbitrary category IDs.")


class ClassifierBenchmarker:
    def __init__(self,
                 output_folder: str or Path,
                 fold_name: str, # e.g., 'test', 'valid', or 'inference_run_XYZ'
                 raw_data_root: Path or str = None, 
                 categories_config_path=None, 
                 class_mapping_path=None): # Optional if providing direct paths
        self.output_folder = Path(output_folder)
        self.fold_name = fold_name
        self.raw_data_root = Path(raw_data_root) if raw_data_root else None
        self.output_folder.mkdir(parents=True, exist_ok=True)
        # Load class configurations
        self.canonical_categories = self._load_json_config(categories_config_path)
        self.class_mapping = self._load_json_config(class_mapping_path)

    # def get_preprocessed_datasets(self, dataset_names: str or list[str]):
    #     # Adapt or reuse if you have a similar dataset registry for segmentation/classification tasks
    #     pass

    def _load_json_config(self, config_path: str or Path):
        """Helper to load a JSON configuration file."""
        if isinstance(config_path, str):
            config_path = Path(config_path)
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            print("Using empty CANONICAL_COCO_CATEGORIES - evaluation may fail")
            return CANONICAL_COCO_CATEGORIES

    def _prepare_coco_from_gpkg(self,
                                gdf_path: str or Path,
                                coco_output_path: str or Path,
                                tiles_path_column: str, # Column in GDF pointing to tile images
                                geometry_column: str,   # Usually 'geometry'
                                class_column: str,      # Column with class names/IDs
                                score_column: Optional[str] = None, # For predictions
                                other_attributes_columns: Optional[list[str]] = None,
                                is_ground_truth: bool = False,
                                class_mapping: Optional[Dict] = None):
        """Helper to convert a GeoPackage to COCO JSON for evaluation."""
        gdf = gpd.read_file(gdf_path)
        Path(coco_output_path).parent.mkdir(parents=True, exist_ok=True)

        # Apply class mapping if provided
        if class_mapping:
            gdf[class_column] = gdf[class_column].map(class_mapping).fillna(gdf[class_column])

        # Ensure consistent CRS if necessary, though COCOGenerator expects pixel coords
        # If GDF is in world coords, it needs conversion. COCOGenerator.from_gdf handles this.

        if other_attributes_columns is None:
            other_attributes_columns = []
        
        # Ensure the class column is passed if it's not already in other_attributes
        # and also object_id_column if present and needed.
        if class_column not in other_attributes_columns:
             other_attributes_columns.append(class_column)
        if object_id_column_name in gdf.columns and object_id_column_name not in other_attributes_columns:
             other_attributes_columns.append(object_id_column_name)


        coco_gen = COCOGenerator.from_gdf(
            description=f"{'Ground Truth' if is_ground_truth else 'Predictions'} for {Path(gdf_path).stem}",
            gdf=gdf,
            tiles_paths_column=tiles_path_column,
            polygons_column=geometry_column,
            scores_column=score_column if not is_ground_truth else None,
            categories_column=class_column,
            other_attributes_columns=list(set(other_attributes_columns)), # Unique columns
            output_path=Path(coco_output_path),
            use_rle_for_labels=True, # RLE is standard for segmentation
            n_workers=4, # Adjust as needed
            coco_categories_list=CANONICAL_COCO_CATEGORIES
        )
        coco_gen.generate_coco()
        return str(coco_output_path)

    def benchmark_single_run(self,
                             run_name: str, # A name for this evaluation run
                             preds_gpkg_path: str or Path,
                             truth_gpkg_path: str or Path, # Or a truth COCO JSON path
                             # These columns must exist in the respective GPKGs
                             preds_tile_path_col: str = 'tile_path',
                             preds_class_col: str = 'predicted_class', # e.g., 'class_name' or 'class_id'
                             preds_score_col: str = 'prediction_score', # e.g., 'score'
                             truth_tile_path_col: str = 'tile_path',
                             truth_class_col: str = 'true_class', # e.g., 'class_name' or 'class_id'
                             # Optional: if your GDFs contain other attributes to carry to COCO
                             preds_other_attrs: Optional[list[str]] = None,
                             truth_other_attrs: Optional[list[str]] = None
                             ):
        """
        Benchmarks a single set of predictions (GPKG) against ground truth (GPKG or COCO JSON).
        """
        evaluator = ClassifierCocoEvaluator()
        run_output_folder = self.output_folder / self.fold_name / run_name
        run_output_folder.mkdir(parents=True, exist_ok=True)

        # Prepare COCO JSON from prediction GPKG
        preds_coco_path = run_output_folder / f"{Path(preds_gpkg_path).stem}.json"
        self._prepare_coco_from_gpkg(
            gdf_path=preds_gpkg_path,
            coco_output_path=preds_coco_path,
            tiles_path_column=preds_tile_path_col,
            geometry_column='geometry',
            class_column=preds_class_col,
            score_column=preds_score_col,
            other_attributes_columns=preds_other_attrs,
            is_ground_truth=False
        )

        # Prepare COCO JSON from truth GPKG (if truth is GPKG)
        # If truth_gpkg_path is already a COCO JSON, you can use it directly.
        if str(truth_gpkg_path).lower().endswith(".gpkg"):
            truth_coco_path_eval = run_output_folder / f"{Path(truth_gpkg_path).stem}.json"
            self._prepare_coco_from_gpkg(
                gdf_path=truth_gpkg_path,
                coco_output_path=truth_coco_path_eval,
                tiles_path_column=truth_tile_path_col,
                geometry_column='geometry',
                class_column=truth_class_col,
                other_attributes_columns=truth_other_attrs,
                is_ground_truth=True
            )
        elif str(truth_gpkg_path).lower().endswith(".json"):
            truth_coco_path_eval = truth_gpkg_path
        else:
            raise ValueError("truth_gpkg_path must be a .gpkg or .json file")


        print(f"\nEvaluating tile-level metrics for run: {run_name}...")
        tile_metrics = evaluator.tile_level(
            preds_coco_path=str(preds_coco_path),
            truth_coco_path=str(truth_coco_path_eval)
            # max_dets can be adjusted if needed
        )
        tile_metrics['run_name'] = run_name
        
        metrics_df = pd.DataFrame([tile_metrics])
        metrics_file = run_output_folder / "tile_level_metrics.csv"
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Tile-level metrics for {run_name} saved to {metrics_file}")
        print(metrics_df.to_string())

        # Placeholder for raster-level (currently not implemented)
        # raster_metrics = evaluator.raster_level_placeholder()
        # print(f"Raster-level metrics for {run_name}: {raster_metrics}")

        return metrics_df

    # Placeholder for a method similar to DetectorBenchmarker.benchmark that iterates over datasets
    def benchmark_datasets_placeholder(self, dataset_names: list[str], preds_gpkg_pattern: str, truth_gpkg_pattern: str):
        """
        Iterates over predefined datasets, finds corresponding prediction and truth files,
        and runs benchmark_single_run for each.
        This is a placeholder and needs to be adapted to how your datasets are structured.
        """
        warnings.warn("benchmark_datasets_placeholder is not yet implemented.")
        # Logic would involve:
        # 1. self.get_preprocessed_datasets(dataset_names)
        # 2. For each dataset and product:
        #    - Construct paths to preds_gpkg and truth_gpkg (or truth_coco)
        #    - Call self.benchmark_single_run(...)
        #    - Aggregate metrics
        pass