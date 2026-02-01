from abc import ABC
from pathlib import Path

import numpy as np
import pandas as pd

from canopyrs.engine.benchmark.base.evaluator import CocoEvaluator
from canopyrs.engine.benchmark.base.find_optimal_detector_aggregator import find_optimal_detector_aggregator
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

        last_model_component_id = None
        last_aggregator_component_id = None
        for component_id, (component_type, _) in enumerate(pipeline.config.components_configs):
            if component_type == component_name:
                last_model_component_id = component_id
            if component_type == 'aggregator':
                last_aggregator_component_id = component_id

        model_coco_output = pipeline.data_state.get_output_file(component_name, last_model_component_id, 'coco')
        model_gpkg_output = pipeline.data_state.get_output_file(component_name, last_model_component_id, 'pre_aggregated_gpkg')
        if last_aggregator_component_id is not None:
            aggregator_output = pipeline.data_state.get_output_file('aggregator', last_aggregator_component_id, 'gpkg')
        else:
            aggregator_output = None

        return model_coco_output, model_gpkg_output, aggregator_output

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
        model_gpkg_outputs: list[str] = []
        truths_gdfs: list[str] = []
        aois_gdfs: list[str] = []
        tiles_roots: list[str] = []
        for dataset_name, dataset in datasets.items():
            for location, product_name, tiles_path, aoi_gpkg, truths_gpkg, truths_coco in dataset.iter_fold(self.raw_data_root, fold="valid"):
                _, model_gpkg_output, _ = self._infer_single_product(
                    product_name=product_name,
                    product_tiles_path=tiles_path,
                    pipeline_config=pipeline_config,
                    component_name=component_name,
                    output_folder=self.output_folder / self.fold_name / 'tile_predictions' / product_name
                )

                raster_names.append(f"{location}/{product_name}")
                model_gpkg_outputs.append(str(model_gpkg_output))
                truths_gdfs.append(truths_gpkg)
                aois_gdfs.append(aoi_gpkg)
                tiles_roots.append(str(tiles_path))

        print(f"Datasets inferred. Starting NMS IoU threshold search...")
        nms_search_output_folder = self.output_folder / self.fold_name / 'NMS_search'
        
        aggregators_results_df = find_optimal_detector_aggregator(
            output_folder=str(nms_search_output_folder),
            raster_names=raster_names,
            model_gpkg_outputs=model_gpkg_outputs,
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
        # Fallback to per-raster rows when there's only one raster and no averaged row.
        if results_average.empty:
            results_average = aggregators_results_df

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

                preds_coco_json, _, preds_aggregated_gpkg = self._infer_single_product(
                    product_name=product_name,
                    product_tiles_path=tiles_path,
                    pipeline_config=pipeline_config,
                    component_name=component_name
                )

                tile_metrics = evaluator.tile_level(
                    iou_type=iou_type,
                    preds_coco_path=preds_coco_json,
                    truth_coco_path=truths_coco,
                    max_dets=[1, 10, 100, dataset.tile_level_eval_maxDets],
                    images_common_ground_resolution=dataset.ground_resolution
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
                max_dets=[1, 10, 100, dataset.tile_level_eval_maxDets],
                images_common_ground_resolution=dataset.ground_resolution
            )
            dataset_tile_level_metrics['location'] = dataset_name
            dataset_tile_level_metrics['product_name'] = "average_over_rasters"
            all_tile_level_metrics.append(dataset_tile_level_metrics)

            # Compute weighted average of raster-level metrics for the dataset, weighted by the number of num_truths
            if dataset_raster_level_metrics:
                metrics_to_average = ['precision', 'recall', 'f1']

                merged_df = pd.DataFrame(dataset_raster_level_metrics)
                weights = merged_df['num_truths']
                weights_np = weights.to_numpy()
                total_truths = weights_np.sum()

                weighted_avg_metrics = {}

                # getting size labels
                size_labels: list[str] = []
                if 'size_labels' in merged_df.columns and not merged_df['size_labels'].empty:
                    first_labels = merged_df['size_labels'].iloc[0]
                    if isinstance(first_labels, (list, tuple)):
                        size_labels = list(first_labels)

                # weighted average of metrics
                for metric in metrics_to_average:
                    weighted_avg_metrics[metric] = (merged_df[metric] * weights).sum() / total_truths

                    # weighted average for single-IoU threshold metrics (e.g., precision_50, recall_75, f1_50, etc.)
                    for suffix in ['50', '75']:
                        col_name = f"{metric}_{suffix}"
                        if col_name in merged_df.columns and merged_df[col_name].notna().all():
                            weighted_avg_metrics[col_name] = (merged_df[col_name] * weights).sum() / total_truths

                    # per-IoU metrics
                    per_iou_col = f"{metric}_per_iou"
                    if per_iou_col in merged_df.columns and merged_df[per_iou_col].notna().all():
                        stacked_metric = np.stack(merged_df[per_iou_col].to_list())
                        weighted_avg_metrics[per_iou_col] = (stacked_metric * weights_np[:, None]).sum(axis=0) / total_truths

                    # per size-label metrics
                    for size_label in size_labels:
                        col_name = f"{metric}_{size_label}"
                        if col_name in merged_df.columns:
                            weighted_avg_metrics[col_name] = (merged_df[col_name] * weights).sum() / total_truths

                        # per size-label, per-IoU metrics
                        per_iou_size_col = f"{metric}_per_iou_{size_label}"
                        if per_iou_size_col in merged_df.columns and merged_df[per_iou_size_col].notna().all():
                            stacked_metric = np.stack(merged_df[per_iou_size_col].to_list())
                            weighted_avg_metrics[per_iou_size_col] = (stacked_metric * weights_np[:, None]).sum(axis=0) / total_truths

                # carry over total count and labeling
                weighted_avg_metrics['num_truths'] = int(total_truths)
                weighted_avg_metrics['location'] = dataset_name
                weighted_avg_metrics['product_name'] = "average_over_rasters"
                weighted_avg_metrics['iou_thresholds'] = ",".join([f"{t:.2f}" for t in self.eval_iou_threshold_list])

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

        return tile_level_metrics_df, raster_level_metrics_df

    @classmethod
    def compute_mean_std_metric_tables(cls,
                                       inputs: list[str | Path | pd.DataFrame],
                                       output_csv: str | Path) -> pd.DataFrame:
        """
        Given multiple metric tables (either in-memory DataFrames or CSV paths), compute mean/std
        per row across tables. Output has the same number of rows as the inputs; location/product_name
        and passthrough columns are preserved from the first table.
        """
        dfs: list[pd.DataFrame] = []
        for item in inputs:
            if isinstance(item, pd.DataFrame):
                dfs.append(item)
            else:
                dfs.append(pd.read_csv(item))

        if not dfs:
            raise ValueError("No inputs provided to compute_mean_std_metric_tables.")

        # Basic sanity checks
        required_cols = {'location', 'product_name'}
        for idx, df in enumerate(dfs):
            missing = required_cols - set(df.columns)
            if missing:
                raise ValueError(f"Input #{idx} missing required columns: {missing}")
        row_counts = {len(df) for df in dfs}
        if len(row_counts) != 1:
            raise ValueError(f"All inputs must have the same number of rows; got counts {row_counts}")

        n_rows = len(dfs[0])
        skip_mean_std = {'tp', 'fp', 'fn'}
        passthrough_only = {'iou_thresholds', 'num_images', 'num_truths', 'location', 'product_name', 'size_labels'}

        summary_rows: list[dict] = []
        for row_idx in range(n_rows):
            base = dfs[0].iloc[row_idx]
            # Optional sanity: ensure location/product_name match across dfs for this row
            for df in dfs[1:]:
                if (df.at[row_idx, 'location'] != base['location'] or
                        df.at[row_idx, 'product_name'] != base['product_name']):
                    raise ValueError(f"location/product_name mismatch at row {row_idx}")

            row_summary: dict[str, object] = {
                'location': base['location'],
                'product_name': base['product_name'],
            }

            for col in dfs[0].columns:
                if col in skip_mean_std:
                    continue
                if col in {'location', 'product_name'}:
                    continue

                values = [df.at[row_idx, col] for df in dfs]

                if col in passthrough_only:
                    row_summary[col] = values[0]
                    continue

                # Numeric columns
                if all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
                    arr = np.array(values, dtype=float)
                    row_summary[f"{col}_mean"] = float(arr.mean())
                    row_summary[f"{col}_std"] = float(arr.std(ddof=0))
                    continue

                # List-like columns with consistent length
                if all(isinstance(v, (list, tuple, np.ndarray)) for v in values):
                    arrays = [np.array(v, dtype=float) for v in values]
                    lengths = {a.shape[0] for a in arrays}
                    if len(lengths) == 1:
                        stacked = np.vstack(arrays)
                        row_summary[f"{col}_mean"] = stacked.mean(axis=0).tolist()
                        row_summary[f"{col}_std"] = stacked.std(axis=0, ddof=0).tolist()
                    continue

            summary_rows.append(row_summary)

        summary_df = pd.DataFrame(summary_rows)
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_csv, index=False)
        print(f"Summarized metrics with mean/std saved to {output_csv}")

        return summary_df

    @classmethod
    def merge_tile_and_raster_summaries(cls,
                                        tile_csv: str | Path | pd.DataFrame,
                                        raster_csv: str | Path | pd.DataFrame,
                                        output_csv: str | Path | None = None,
                                        tile_prefix: str = "",
                                        raster_prefix: str = "") -> pd.DataFrame:
        """
        Merge tile-level and raster-level summary tables on (location, product_name),
        prefixing their metric columns to avoid collisions. Identity columns are left untouched.
        """
        def _to_df(obj):
            if isinstance(obj, pd.DataFrame):
                return obj.copy()
            return pd.read_csv(obj)

        tile_df = _to_df(tile_csv)
        raster_df = _to_df(raster_csv)

        required_cols = {'location', 'product_name'}
        identity_cols = {'location', 'product_name', 'iou_thresholds', 'num_images', 'num_truths'}

        def _apply_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
            if not prefix:
                return df
            rename_map = {c: f"{prefix}_{c}" for c in df.columns if c not in identity_cols}
            return df.rename(columns=rename_map)

        for name, df in [('tile', tile_df), ('raster', raster_df)]:
            missing = required_cols - set(df.columns)
            if missing:
                raise ValueError(f"{name} summary missing required columns: {missing}")

        tile_df = _apply_prefix(tile_df, tile_prefix)
        raster_df = _apply_prefix(raster_df, raster_prefix)

        overlap = [c for c in raster_df.columns if c in tile_df.columns and c not in required_cols]
        raster_unique = raster_df.drop(columns=overlap)

        merged = pd.merge(tile_df, raster_unique, on=['location', 'product_name'], how='left') # using left in case some datasets dont have raster-level metrics

        if output_csv is not None:
            output_csv = Path(output_csv)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(output_csv, index=False)
            print(f"Merged tile and raster summaries saved to {output_csv}")

        return merged
