from abc import ABC
from pathlib import Path

import numpy as np
import pandas as pd

from canopyrs.engine.benchmark.detector.evaluator import CocoEvaluator
from canopyrs.engine.benchmark.detector.find_optimal_detector_aggregator import find_optimal_detector_aggregator
from canopyrs.engine.config_parsers import DetectorConfig, AggregatorConfig, PipelineConfig, InferIOConfig, SegmenterConfig
from canopyrs.engine.pipeline import Pipeline
from canopyrs.engine.utils import merge_coco_jsons

from canopyrs.data.detection.preprocessed_datasets import DATASET_REGISTRY


class BaseBenchmarker(ABC):
    """
    Base class for running benchmarks at tile and raster level.
    Can be extended for detection or segmentation tasks.
    """
    def __init__(self,
                 output_folder: str | Path,
                 fold_name: str,
                 raw_data_root: Path | str,
                 eval_iou_threshold: float | list[float] = 0.75):
        """
        Initialize the benchmarker.

        output_folder: root folder where metrics and intermediate files are saved.
        fold_name: dataset fold to use ('test' or 'valid').
        raw_data_root: base path to preprocessed datasets.
        eval_iou_threshold: single IoU float or list of IoUs for RF1-style averaging.
        """
        self.output_folder = Path(output_folder)
        self.fold_name = fold_name
        self.raw_data_root = Path(raw_data_root)
        
        # Normalize to list internally
        if isinstance(eval_iou_threshold, (list, tuple)):
            self.eval_iou_threshold_list = [float(t) for t in eval_iou_threshold]
        else:
            self.eval_iou_threshold_list = [float(eval_iou_threshold)]

        # Primary IoU for scalar-only contexts (labels/legacy paths)
        self.eval_iou_threshold = float(self.eval_iou_threshold_list[0])

        self.output_folder.mkdir(parents=True, exist_ok=True)

        assert fold_name in ['test', 'valid'], f'Fold {fold_name} not supported. Supported folds are "test" and "valid".'

    def _get_preprocessed_datasets(self, dataset_names: str | list[str]):
        """
        Load and validate requested datasets for the current fold.
        """
        datasets = {}
        for dataset_name in list(dataset_names):
            assert dataset_name in DATASET_REGISTRY, f'Dataset {dataset_name} not supported. Supported datasets are {DATASET_REGISTRY.keys()}.'
            datasets[dataset_name] = DATASET_REGISTRY[dataset_name]()
            datasets[dataset_name].verify_dataset(root_output_path=self.raw_data_root, folds=[self.fold_name])
        return datasets

    def _infer_single_product(self,
                              product_name: str,
                              product_tiles_path: str | Path,
                              pipeline_config: PipelineConfig,
                              component_name: str,
                              output_folder: str | Path = None):
        """
        Run inference for one product and return paths to outputs.
        
        pipeline_config: Pre-configured PipelineConfig from child class
        component_name: 'detector' or 'segmenter' - used to retrieve output files
        """

        if output_folder is None:
            output_folder = self.output_folder / self.fold_name / product_name

        io_config = InferIOConfig(
            input_imagery=None,
            tiles_path=str(product_tiles_path),
            output_folder=str(output_folder),
        )

        pipeline = Pipeline(io_config, pipeline_config)
        pipeline()

        model_output = pipeline.data_state.get_output_file(component_name, 0, 'coco')
        # Check if pipeline has aggregator (will be component index 1)
        if len(pipeline_config.components_configs) > 1:
            aggregator_output = pipeline.data_state.get_output_file('aggregator', 1, 'gpkg')
        else:
            aggregator_output = None

        return model_output, aggregator_output

    def _find_optimal_nms_iou_threshold(self,
                                        pipeline_config: PipelineConfig,
                                        component_name: str,
                                        iou_type: str,
                                        aggregator_config: AggregatorConfig,
                                        dataset_names: list[str],
                                        nms_iou_thresholds: list[float],
                                        nms_score_thresholds: list[float],
                                        eval_at_ground_resolution: float = 0.045,
                                        n_workers: int = 6):

        """
        Find the optimal NMS IoU threshold by evaluating different thresholds on the validation set.
        
        pipeline_config: Pre-configured PipelineConfig from child class (without aggregator)
        component_name: 'detector' or 'segmenter'
        iou_type: 'bbox' or 'segm'
        aggregator_config: Base AggregatorConfig from child class (will be modified for grid search)
        """
        datasets = self._get_preprocessed_datasets(dataset_names)

        print(f"Finding optimal NMS IoU threshold for datasets: {dataset_names}. Inferring rasters...")

        raster_names: list[str] = []
        preds_coco_jsons: list[str] = []
        truths_gdfs: list[str] = []
        aois_gdfs: list[str] = []
        tiles_roots: list[str] = []
        for dataset_name, dataset in datasets.items():
            for location, product_name, tiles_path, aoi_gpkg, truths_gpkg, truths_coco in dataset.iter_fold(self.raw_data_root, fold="valid"):
                preds_coco_json, _ = self._infer_single_product(
                    product_name=product_name,
                    product_tiles_path=tiles_path,
                    pipeline_config=pipeline_config,
                    component_name=component_name,
                    output_folder=self.output_folder / self.fold_name / 'tile_predictions' / product_name
                )

                raster_names.append(f"{location}/{product_name}")
                preds_coco_jsons.append(preds_coco_json)
                truths_gdfs.append(truths_gpkg)
                aois_gdfs.append(aoi_gpkg)
                tiles_roots.append(str(tiles_path))

        print(f"Datasets inferred. Starting NMS IoU threshold search...")
        nms_search_output_folder = self.output_folder / self.fold_name / 'NMS_search'
        
        aggregators_results_df = find_optimal_detector_aggregator(
            output_folder=str(nms_search_output_folder),
            raster_names=raster_names,
            preds_coco_jsons=preds_coco_jsons,
            truths_gdfs=truths_gdfs,
            aois_gdfs=aois_gdfs,
            tiles_roots=tiles_roots,
            ground_resolution=eval_at_ground_resolution,
            nms_iou_thresholds=nms_iou_thresholds,
            nms_score_thresholds=nms_score_thresholds,
            eval_iou_threshold=self.eval_iou_threshold_list,
            n_workers=n_workers,
            iou_type=iou_type,
            aggregator_config=aggregator_config,
        )

        csv_output_path = nms_search_output_folder / 'optimal_nms_iou_threshold_search.csv'
        aggregators_results_df.to_csv(csv_output_path, index=False)
        print(f"Optimal NMS IoU threshold search results saved to {csv_output_path}")

        # Find optimal NMS IoU threshold based on highest f1 metric:
        results_average = aggregators_results_df[aggregators_results_df['raster_name'] == 'average_over_rasters']
        best_aggregator = results_average.sort_values('f1', ascending=False).iloc[0]
        best_aggregator_iou = best_aggregator['nms_iou_threshold']
        best_aggregator_score_threshold = best_aggregator['nms_score_threshold']

        if self.eval_iou_threshold_list and len(self.eval_iou_threshold_list) > 1:
            iou_label = f"{int(self.eval_iou_threshold_list[0] * 100)}:{int(self.eval_iou_threshold_list[-1] * 100)}"
        else:
            iou_label = str(int(self.eval_iou_threshold * 100))

        print(f"Best NMS IoU threshold: {best_aggregator_iou}, Best Score threshold: {best_aggregator_score_threshold}, with an RF1_{iou_label} of {best_aggregator['f1']}")

        # Create and return optimal aggregator config with the best thresholds
        optimal_aggregator_config = aggregator_config.model_copy(deep=True)
        optimal_aggregator_config.nms_threshold = best_aggregator_iou
        optimal_aggregator_config.score_threshold = best_aggregator_score_threshold
        
        return optimal_aggregator_config

    def _benchmark(self,
                   pipeline_config_with_aggregator: PipelineConfig,
                   component_name: str,
                   iou_type: str,
                   dataset_names: str | list[str]):

        """
        Runs the model on the entire test dataset, recording both tile-level and raster-level metrics for each
         individual product and also aggregated for each dataset (which are made of 1 or more products).
         
        pipeline_config_with_aggregator: Pre-configured PipelineConfig from child class (with aggregator)
        component_name: 'detector' or 'segmenter'
        iou_type: 'bbox' or 'segm'
        """
        
        # Verify that the last component is an aggregator
        assert len(pipeline_config_with_aggregator.components_configs) > 1, "Pipeline config must have at least 2 components (model + aggregator)"
        assert pipeline_config_with_aggregator.components_configs[-1][0] == 'aggregator', "Last component must be 'aggregator'"

        datasets = self._get_preprocessed_datasets(dataset_names)

        evaluator = CocoEvaluator()

        all_tile_level_metrics = []
        all_raster_level_metrics = []
        for dataset_name, dataset in datasets.items():
            dataset_preds_cocos = []
            dataset_truths_cocos = []
            dataset_raster_level_metrics = []
            for location, product_name, tiles_path, aoi_gpkg, truths_gpkg, truths_coco in dataset.iter_fold(self.raw_data_root, fold="test"):
                if aoi_gpkg is not None and truths_gpkg is not None:
                    do_raster_level_eval = True
                else:
                    do_raster_level_eval = False

                # Only use aggregator if raster-level eval is needed
                if do_raster_level_eval:
                    pipeline_config = pipeline_config_with_aggregator
                else:
                    # Create pipeline without aggregator (pop the last component since it's the aggregator)
                    pipeline_config = PipelineConfig(components_configs=pipeline_config_with_aggregator.components_configs[:-1])

                preds_coco_json, preds_aggregated_gpkg = self._infer_single_product(
                    product_name=product_name,
                    product_tiles_path=tiles_path,
                    pipeline_config=pipeline_config,
                    component_name=component_name
                )

                tile_metrics = evaluator.tile_level(
                    iou_type=iou_type,
                    preds_coco_path=preds_coco_json,
                    truth_coco_path=truths_coco,
                    max_dets=[1, 10, 100, dataset.tile_level_eval_maxDets]
                )
                tile_metrics['location'] = location
                tile_metrics['product_name'] = product_name
                all_tile_level_metrics.append(tile_metrics)
                dataset_preds_cocos.append(preds_coco_json)
                dataset_truths_cocos.append(truths_coco)

                if do_raster_level_eval and preds_aggregated_gpkg is not None:
                    raster_metrics = evaluator.raster_level_multi_iou_thresholds(
                        iou_type=iou_type,
                        preds_gpkg_path=preds_aggregated_gpkg,
                        truth_gpkg_path=truths_gpkg,
                        aoi_gpkg_path=aoi_gpkg,
                        ground_resolution=dataset.ground_resolution,
                        iou_thresholds=self.eval_iou_threshold_list
                    )
                    raster_metrics['iou_thresholds'] = ",".join([f"{t:.2f}" for t in self.eval_iou_threshold_list])
                    # Log per-IoU scores for traceability
                    if 'f1_per_iou' in raster_metrics:
                        raster_metrics['f1_per_iou'] = [float(v) for v in raster_metrics['f1_per_iou']]
                    if 'precision_per_iou' in raster_metrics:
                        raster_metrics['precision_per_iou'] = [float(v) for v in raster_metrics['precision_per_iou']]
                    if 'recall_per_iou' in raster_metrics:
                        raster_metrics['recall_per_iou'] = [float(v) for v in raster_metrics['recall_per_iou']]
                    raster_metrics['location'] = location
                    raster_metrics['product_name'] = product_name
                    all_raster_level_metrics.append(raster_metrics)
                    dataset_raster_level_metrics.append(raster_metrics)

            # Merge the COCO JSON files for the dataset
            merged_preds_coco_path = self.output_folder / self.fold_name / f"{dataset_name}_merged_preds_coco.json"
            merged_truths_coco_path = self.output_folder / self.fold_name / f"{dataset_name}_merged_truths_coco.json"
            merge_coco_jsons(dataset_preds_cocos, merged_preds_coco_path)
            merge_coco_jsons(dataset_truths_cocos, merged_truths_coco_path)

            # Evaluate the merged COCO JSON files
            print(f"\nEvaluating COCO metrics for the whole dataset {dataset_name}...")
            dataset_tile_level_metrics = evaluator.tile_level(
                iou_type=iou_type,
                preds_coco_path=str(merged_preds_coco_path),
                truth_coco_path=str(merged_truths_coco_path),
                max_dets=[1, 10, 100, dataset.tile_level_eval_maxDets]
            )
            dataset_tile_level_metrics['location'] = dataset_name
            dataset_tile_level_metrics['product_name'] = "average_over_rasters"
            all_tile_level_metrics.append(dataset_tile_level_metrics)

            # Compute weighted average of raster-level metrics for the dataset, weighted by the number of num_truths
            if dataset_raster_level_metrics:
                metrics_to_average = ['precision', 'recall', 'f1']

                merged_df = pd.DataFrame(dataset_raster_level_metrics)
                weights = merged_df['num_truths']
                total_truths = weights.sum()

                weighted_avg_metrics = {}
                for metric in metrics_to_average:
                    weighted_avg_metrics[metric] = (merged_df[metric] * weights).sum() / total_truths

                # carry over total count and labeling
                weighted_avg_metrics['num_truths'] = int(total_truths)
                weighted_avg_metrics['location'] = dataset_name
                weighted_avg_metrics['product_name'] = "average_over_rasters"
                weighted_avg_metrics['iou_thresholds'] = ",".join([f"{t:.2f}" for t in self.eval_iou_threshold_list])

                # Weighted average per-IoU scores for logging
                if 'f1_per_iou' in merged_df.columns and merged_df['f1_per_iou'].notna().all():
                    stacked_f1 = np.stack(merged_df['f1_per_iou'].to_list())
                    weighted_avg_metrics['f1_per_iou'] = (stacked_f1 * weights.to_numpy()[:, None]).sum(axis=0) / total_truths
                if 'precision_per_iou' in merged_df.columns and merged_df['precision_per_iou'].notna().all():
                    stacked_prec = np.stack(merged_df['precision_per_iou'].to_list())
                    weighted_avg_metrics['precision_per_iou'] = (stacked_prec * weights.to_numpy()[:, None]).sum(axis=0) / total_truths
                if 'recall_per_iou' in merged_df.columns and merged_df['recall_per_iou'].notna().all():
                    stacked_rec = np.stack(merged_df['recall_per_iou'].to_list())
                    weighted_avg_metrics['recall_per_iou'] = (stacked_rec * weights.to_numpy()[:, None]).sum(axis=0) / total_truths

                all_raster_level_metrics.append(weighted_avg_metrics)

        # Save the metrics to CSV files
        tile_level_metrics_file = self.output_folder / self.fold_name / "tile_level_metrics.csv"
        tile_level_metrics_df = pd.DataFrame(all_tile_level_metrics)
        tile_level_metrics_df.to_csv(tile_level_metrics_file, index=False)
        print(f"Tile-level metrics saved to {tile_level_metrics_file}")
        # Save the raster-level metrics to CSV files
        raster_level_metrics_file = self.output_folder / self.fold_name / "raster_level_metrics.csv"
        raster_level_metrics_df = pd.DataFrame(all_raster_level_metrics)
        raster_level_metrics_df.to_csv(raster_level_metrics_file, index=False)
        print(f"Raster-level metrics saved to {raster_level_metrics_file}")
