from pathlib import Path
import pandas as pd
import json
from typing import Optional, Dict

from engine.benchmark.classifier.evaluator import ClassifierCocoEvaluator
from engine.config_parsers import ClassifierConfig, PipelineConfig, InferIOConfig
from engine.pipeline import Pipeline
from engine.utils import merge_coco_jsons

"""
# Define your canonical class list here or import from a central place
CANONICAL_COCO_CATEGORIES = [
    # {"id": 1, "name": "class1", "supercategory": "object"},
    # {"id": 2, "name": "class2", "supercategory": "object"},
    # Add all your classes here
]
if not CANONICAL_COCO_CATEGORIES:
    print("WARNING: CANONICAL_COCO_CATEGORIES is not defined in benchmark.py. Evaluation might fail or use arbitrary category IDs.")
"""

class ClassifierBenchmarker:
    def __init__(self,
                 output_folder: str or Path,
                 fold_name: str, # e.g., 'test', 'valid', or 'inference_run_XYZ'
                 raw_data_root: Path or str = None, 
                 categories_config_path=None):
        self.output_folder = Path(output_folder)
        self.fold_name = fold_name
        self.raw_data_root = Path(raw_data_root)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        assert fold_name in ['test', 'valid'], f'Fold {fold_name} not supported. Supported folds are "test" and "valid".'

        # Load categories if provided
        if categories_config_path and Path(categories_config_path).exists():
            with open(categories_config_path, 'r') as f:
                self.canonical_categories = json.load(f)
            print(f"Loaded {len(self.canonical_categories)} categories from {categories_config_path}")
        else:
            self.canonical_categories = []
            print("Warning: No categories loaded")

    def get_preprocessed_datasets(self, dataset_names: str or list[str]):
        # Placeholder for dataset registery
        datasets = {}
        for dataset_name in list(dataset_names):
            # datasets[dataset_name] = DATASET_REGISTRY[dataset_name]()
            # datasets[dataset_name].verify_dataset(root_output_path=self.raw_data_root, folds=[self.fold_name])
            pass
        return datasets

    def infer_single_product(self,
                             product_name: str,
                             product_tiles_path: str or Path,
                             classifier_config: ClassifierConfig,
                             input_gpkg: str or Path = None,
                             output_folder: str or Path = None):
        """
        Run classifier inference on a single product.
        Similar to DetectorBenchmarker.infer_single_product but for classification.
        """
        if output_folder is None:
            output_folder = self.output_folder / self.fold_name / product_name

        io_config = InferIOConfig(
            input_imagery=None,
            tiles_path=str(product_tiles_path),
            input_gpkg=str(input_gpkg) if input_gpkg else None,
            output_folder=str(output_folder),
        )

        components_configs = [("classifier", classifier_config)]
        pipeline_config = PipelineConfig(components_configs=components_configs)
        pipeline = Pipeline(io_config, pipeline_config)
        pipeline()

        # Get the outputs from the pipeline
        classifier_coco = pipeline.data_state.get_output_file('classifier', 0, 'coco')
        classifier_gpkg = pipeline.data_state.get_output_file('classifier', 0, 'gpkg')

        return classifier_coco, classifier_gpkg

    def benchmark_single_run(self,
                             run_name: str,
                             product_tiles_path: str or Path,
                             classifier_config: ClassifierConfig,
                             truth_coco_path: str or Path,
                             input_gpkg: str or Path = None):
        """
        Benchmark a single classifier run against ground truth.
        """
        evaluator = ClassifierCocoEvaluator()
        run_output_folder = self.output_folder / self.fold_name / run_name
        run_output_folder.mkdir(parents=True, exist_ok=True)

        # Run inference
        preds_coco_path, preds_gpkg_path = self.infer_single_product(
            product_name=run_name,
            product_tiles_path=product_tiles_path,
            classifier_config=classifier_config,
            input_gpkg=input_gpkg,
            output_folder=run_output_folder
        )

        print(f"Evaluating tile-level metrics for run: {run_name}...")
        tile_metrics = evaluator.tile_level(
            preds_coco_path=str(preds_coco_path),
            truth_coco_path=str(truth_coco_path)
        )
        tile_metrics['run_name'] = run_name
        
        metrics_df = pd.DataFrame([tile_metrics])
        metrics_file = run_output_folder / "tile_level_metrics.csv"
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Tile-level metrics for {run_name} saved to {metrics_file}")
        print(metrics_df.to_string())

        return metrics_df

    def benchmark(self,
                  classifier_config: ClassifierConfig,
                  dataset_names: str or list[str]):
        """
        Run classifier benchmark on full datasets.
        Similar structure to DetectorBenchmarker.benchmark.
        """
        datasets = self.get_preprocessed_datasets(dataset_names)
        evaluator = ClassifierCocoEvaluator()

        all_tile_level_metrics = []
        
        for dataset_name, dataset in datasets.items():
            dataset_preds_cocos = []
            dataset_truths_cocos = []
            
            for location, product_name, tiles_path, input_gpkg, truths_coco in dataset.iter_fold(self.raw_data_root, fold="test"):
                preds_coco_json, preds_gpkg = self.infer_single_product(
                    product_name=product_name,
                    product_tiles_path=tiles_path,
                    classifier_config=classifier_config,
                    input_gpkg=input_gpkg
                )

                tile_metrics = evaluator.tile_level(
                    preds_coco_path=preds_coco_json,
                    truth_coco_path=truths_coco
                )
                tile_metrics['location'] = location
                tile_metrics['product_name'] = product_name
                all_tile_level_metrics.append(tile_metrics)
                dataset_preds_cocos.append(preds_coco_json)
                dataset_truths_cocos.append(truths_coco)

            # Merge COCO files for dataset-level evaluation
            merged_preds_coco_path = self.output_folder / self.fold_name / f"{dataset_name}_merged_preds_coco.json"
            merged_truths_coco_path = self.output_folder / self.fold_name / f"{dataset_name}_merged_truths_coco.json"
            merge_coco_jsons(dataset_preds_cocos, merged_preds_coco_path)
            merge_coco_jsons(dataset_truths_cocos, merged_truths_coco_path)

            # Dataset-level evaluation
            print(f"Evaluating COCO metrics for the whole dataset {dataset_name}...")
            dataset_tile_level_metrics = evaluator.tile_level(
                preds_coco_path=str(merged_preds_coco_path),
                truth_coco_path=str(merged_truths_coco_path)
            )
            dataset_tile_level_metrics['location'] = dataset_name
            dataset_tile_level_metrics['product_name'] = "average_over_rasters"
            all_tile_level_metrics.append(dataset_tile_level_metrics)

        # Save metrics
        tile_level_metrics_file = self.output_folder / self.fold_name / "tile_level_metrics.csv"
        tile_level_metrics_df = pd.DataFrame(all_tile_level_metrics)
        tile_level_metrics_df.to_csv(tile_level_metrics_file, index=False)
        print(f"Tile-level metrics saved to {tile_level_metrics_file}")

        return tile_level_metrics_df