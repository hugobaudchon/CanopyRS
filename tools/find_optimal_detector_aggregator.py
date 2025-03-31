import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from geodataset.aggregator import Aggregator

from engine.benchmark.evaluator import CocoEvaluator


def eval_single_aggregator(
        output_path: str,
        preds_coco_json: str,
        truth_gdf: str,
        tiles_root: str,
        nms_iou_threshold: float,
        nms_score_threshold: float,
        min_centroid_distance_weight: float,
        ground_resolution: float
):
    output_path = Path(output_path) / f"nmsiou_{str(nms_iou_threshold).replace('.', 'p')}_mincentroid_{str(min_centroid_distance_weight).replace('.', 'p')}"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    aggregator_output_path = output_path / 'aggregator_output.gpkg'

    Aggregator.from_coco(
        output_path=aggregator_output_path,
        tiles_folder_path=tiles_root,
        coco_json_path=preds_coco_json,
        scores_names=['detector_score'],
        other_attributes_names=None,
        scores_weights=[1.0],
        min_centroid_distance_weight=min_centroid_distance_weight,
        score_threshold=nms_score_threshold,
        nms_threshold=nms_iou_threshold,
        nms_algorithm='iou',
        best_geom_keep_area_ratio=0.5,
        pre_aggregated_output_path=None
    )

    # Evaluate the predictions
    evaluator = CocoEvaluator()
    metrics = evaluator.raster_level(
        iou_type='bbox',
        preds_gpkg_path=str(aggregator_output_path),
        truth_gpkg_path=truth_gdf,
        ground_resolution=ground_resolution
    )

    return metrics


def average_metrics_by_raster(results_df: pd.DataFrame):
    grouping_cols = ['nms_iou_threshold', 'nms_score_threshold', 'min_centroid_distance_weight']
    # Columns that we don't want to aggregate (identification columns)
    ignore_cols = set(grouping_cols + ['raster_name', 'preds_coco_json', 'truth_gdf', 'tiles_root'])

    # Metrics to be averaged and summed:
    avg_metrics = [
        "AP", "AP50", "AP75",
        "AP_small", "AP_medium", "AP_large",
        "AR", "AR50", "AR75", "AR_max", "AR_small", "AR_medium", "AR_large"
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

        # Compute F1 metrics using the aggregated average values.
        def compute_f1(ap, ar):
            if (ap + ar) > 0:
                return 2 * ap * ar / (ap + ar)
            return 0

        if "AP" in aggregated_record and "AR" in aggregated_record:
            aggregated_record["F1"] = compute_f1(aggregated_record["AP"], aggregated_record["AR"])
        if "AP50" in aggregated_record and "AR50" in aggregated_record:
            aggregated_record["F1_50"] = compute_f1(aggregated_record["AP50"], aggregated_record["AR50"])
        if "AP75" in aggregated_record and "AR75" in aggregated_record:
            aggregated_record["F1_75"] = compute_f1(aggregated_record["AP75"], aggregated_record["AR75"])
        if "AP_small" in aggregated_record and "AR_small" in aggregated_record:
            aggregated_record["F1_small"] = compute_f1(aggregated_record["AP_small"], aggregated_record["AR_small"])
        if "AP_medium" in aggregated_record and "AR_medium" in aggregated_record:
            aggregated_record["F1_medium"] = compute_f1(aggregated_record["AP_medium"], aggregated_record["AR_medium"])
        if "AP_large" in aggregated_record and "AR_large" in aggregated_record:
            aggregated_record["F1_large"] = compute_f1(aggregated_record["AP_large"], aggregated_record["AR_large"])

        aggregated_rows.append(aggregated_record)

    # Create a DataFrame for the aggregated (averaged) rows.
    aggregated_df = pd.DataFrame(aggregated_rows)
    # Append the aggregated results to the original results DataFrame.
    results_df = pd.concat([results_df, aggregated_df], ignore_index=True)
    return results_df


def find_optimal_detector_aggregator(
        output_folder: str,
        raster_names: list[str],
        preds_coco_jsons: list[str],
        truths_gdfs: list[str],
        tiles_roots: list[str],
        ground_resolution: float,
        nms_iou_thresholds: list[float],
        min_centroid_distance_weights: list[float],
        min_nms_score_threshold: float,
        n_workers: int
):

    assert len(raster_names) == len(preds_coco_jsons) == len(truths_gdfs) == len(tiles_roots), \
        "The number of elements in raster_names, preds_coco_jsons, truths_gdfs, and tiles_roots must be the same."

    # Create a list to hold all parameter combinations
    tasks = []
    for nms_iou_threshold in nms_iou_thresholds:
        for min_centroid_distance_weight in min_centroid_distance_weights:
            for raster_name, preds_coco_json, truth_gdf, tiles_root in itertools.product(
                    raster_names, preds_coco_jsons, truths_gdfs, tiles_roots):
                tasks.append({
                    "raster_name": raster_name,
                    "nms_iou_threshold": nms_iou_threshold,
                    "nms_score_threshold": min_nms_score_threshold,
                    "min_centroid_distance_weight": min_centroid_distance_weight,
                    "preds_coco_json": preds_coco_json,
                    "truth_gdf": truth_gdf,
                    "tiles_root": tiles_root
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
                preds_coco_json=params["preds_coco_json"],
                truth_gdf=params["truth_gdf"],
                tiles_root=params["tiles_root"],
                nms_iou_threshold=params["nms_iou_threshold"],
                nms_score_threshold=min_nms_score_threshold,
                min_centroid_distance_weight=params["min_centroid_distance_weight"],
                ground_resolution=ground_resolution
            )
            future_to_params[future] = params

        # Collect the results as they complete
        for future in as_completed(future_to_params):
            params = future_to_params[future]
            try:
                metrics = future.result()
                record = params.copy()
                record.update(metrics)
                results_list.append(record)
            except Exception as exc:
                print(f"Parameters {params} generated an exception: {exc}")

    results_df = pd.DataFrame(results_list)

    # Compute weighted average of all metrics for each parameter combination, over the different rasters.
    # The weights are the number of truth bbox in each raster ('num_truths').
    results_df = average_metrics_by_raster(results_df)

    return results_df


