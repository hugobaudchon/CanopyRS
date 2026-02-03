from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import os
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from tqdm import tqdm

from canopyrs.engine.benchmark.base.evaluator import CocoEvaluator
from canopyrs.engine.config_parsers import AggregatorConfig, PipelineConfig, InferIOConfig
from canopyrs.engine.pipeline import Pipeline


def eval_single_aggregator(
        output_path: str,
        model_gpkg_output: str,
        truth_gdf: str,
        tiles_root: str,
        aoi_gdf: str,
        eval_iou_threshold: float | list[float],
        ground_resolution: float,
        iou_type: str,
        aggregator_config: AggregatorConfig,
):
    if iou_type not in ('bbox', 'segm'):
        raise ValueError(f"Unsupported iou_type: {iou_type}. Expected 'bbox' or 'segm'.")

    with open(os.devnull, "w") as devnull, \
        contextlib.redirect_stdout(devnull), \
        contextlib.redirect_stderr(devnull):
        output_path = Path(output_path) / f"nmsiou_{str(aggregator_config.nms_threshold).replace('.', 'p')}_nmsscorethresh_{str(aggregator_config.score_threshold).replace('.', 'p')}"
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # Setup IO config for the aggregator pipeline
        io_config = InferIOConfig(
            input_imagery=None,
            tiles_path=str(tiles_root),
            input_gpkg=str(model_gpkg_output),
            output_folder=str(output_path),
        )

        # Create a single-component pipeline with only the aggregator
        pipeline_config = PipelineConfig(components_configs=[
            ('aggregator', aggregator_config)
        ])

        # Run the pipeline
        pipeline = Pipeline(io_config, pipeline_config)
        pipeline()

        # Get aggregator output from pipeline
        aggregator_output_path = pipeline.data_state.get_output_file('aggregator', 0, 'gpkg')

        # Evaluate the predictions (multi-IoU also handles single-threshold via length-1 list)
        evaluator = CocoEvaluator()
        if isinstance(eval_iou_threshold, (list, tuple)):
            iou_list = [float(t) for t in eval_iou_threshold]
        else:
            iou_list = [float(eval_iou_threshold)]
        metrics = evaluator.raster_level_multi_iou_thresholds(
            iou_type=iou_type,
            preds_gpkg_path=str(aggregator_output_path),
            truth_gpkg_path=truth_gdf,
            aoi_gpkg_path=aoi_gdf,
            ground_resolution=ground_resolution,
            iou_thresholds=iou_list
        )

    return metrics


def average_metrics_by_raster(results_df: pd.DataFrame):
    grouping_cols = ['nms_iou_threshold', 'nms_score_threshold']

    # Metrics to be averaged and summed:
    avg_metrics = [
        "precision", "recall", "f1",
    ]
    sum_metrics = ["num_images", "num_truths", "num_preds"]

    aggregated_rows = []
    # Group by the aggregator hyperparameters.
    for params, group in results_df.groupby(grouping_cols):
        # Start with the grouping keys.
        aggregated_record = {col: val for col, val in zip(grouping_cols, params)}
        aggregated_record['raster_name'] = 'average_over_rasters'

        # Compute weighted averages for the average metrics.
        for metric in avg_metrics:
            if metric in group.columns:
                weights = group["num_truths"]
                if weights.sum() > 0:
                    aggregated_record[metric] = np.average(group[metric], weights=weights)
                else:
                    aggregated_record[metric] = group[metric].mean()

        # Sum the specified sum metrics.
        for metric in sum_metrics:
            if metric in group.columns:
                aggregated_record[metric] = group[metric].sum()

        aggregated_rows.append(aggregated_record)

    # Create a DataFrame for the aggregated (averaged) rows.
    aggregated_df = pd.DataFrame(aggregated_rows)
    # Append the aggregated results to the original results DataFrame.
    results_df = pd.concat([results_df, aggregated_df], ignore_index=True)
    return results_df


def find_optimal_detector_aggregator(
        output_folder: str,
        raster_names: list[str],
        model_gpkg_outputs: list[str],
        truths_gdfs: list[str],
        aois_gdfs: list[str],
        tiles_roots: list[str],
        ground_resolution: float,
        nms_iou_thresholds: list[float],
        nms_score_thresholds: list[float],
        eval_iou_threshold: float | list[float],
        n_workers: int,
        iou_type: str,
        aggregator_config: AggregatorConfig = None,
):

    assert len(raster_names) == len(model_gpkg_outputs) == len(truths_gdfs) == len(tiles_roots) == len(aois_gdfs), \
        "The number of elements in raster_names, model_gpkg_outputs, truths_gdfs, tiles_roots and aois_gdfs must be the same."

    if isinstance(eval_iou_threshold, (list, tuple)):
        normalized_iou_thresholds = [float(t) for t in eval_iou_threshold]
    else:
        normalized_iou_thresholds = [float(eval_iou_threshold)]

    # Create a base aggregator config if not provided
    if aggregator_config is None:
        aggregator_config = AggregatorConfig()

    # Create a list to hold all parameter combinations
    tasks = []
    for nms_iou_threshold in nms_iou_thresholds:
        for nms_score_threshold in nms_score_thresholds:
            for raster_name, model_gpkg_output, truth_gdf, tiles_root, aoi_gdf in zip(raster_names, model_gpkg_outputs, truths_gdfs, tiles_roots, aois_gdfs):
                # Create a copy of aggregator config with grid search parameters
                task_aggregator_config = aggregator_config.model_copy(deep=True)
                task_aggregator_config.nms_threshold = nms_iou_threshold
                task_aggregator_config.score_threshold = nms_score_threshold
                
                tasks.append({
                    "raster_name": raster_name,
                    "nms_iou_threshold": nms_iou_threshold,
                    "nms_score_threshold": nms_score_threshold,
                    "model_gpkg_output": model_gpkg_output,
                    "truth_gdf": truth_gdf,
                    "tiles_root": tiles_root,
                    "aoi_gdf": aoi_gdf,
                    "aggregator_config": task_aggregator_config
                })

    results_list = []

    # Parallelize using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Map futures to parameters
        future_to_params = {}
        for params in tasks:
            future = executor.submit(
                eval_single_aggregator,
                output_path=f"{output_folder}/{params['raster_name']}",
                model_gpkg_output=params["model_gpkg_output"],
                truth_gdf=params["truth_gdf"],
                tiles_root=params["tiles_root"],
                aoi_gdf=params["aoi_gdf"],
                eval_iou_threshold=normalized_iou_thresholds,
                ground_resolution=ground_resolution,
                iou_type=iou_type,
                aggregator_config=params["aggregator_config"],
            )
            future_to_params[future] = params

        # Collect the results as they complete
        with tqdm(total=len(future_to_params), desc="Grid search", unit="task") as pbar:
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    metrics = future.result()
                    record = params.copy()
                    record.update(metrics)
                    results_list.append(record)
                except Exception:
                    print(f"Parameters {params} generated an exception:")
                    traceback.print_exc()
                finally:
                    pbar.update(1)

    results_df = pd.DataFrame(results_list)

    # Compute weighted average of all metrics for each parameter combination, over the different rasters.
    # The weights are the number of truth bbox in each raster ('num_truths').
    results_df = average_metrics_by_raster(results_df)

    return results_df
