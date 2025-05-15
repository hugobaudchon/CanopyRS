from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from geodataset.aggregator import Aggregator

from engine.benchmark.detector.evaluator import CocoEvaluator


def eval_single_aggregator(
        output_path: str,
        preds_coco_json: str,
        truth_gdf: str,
        tiles_root: str,
        aoi_gdf: str,
        nms_iou_threshold: float,
        nms_score_threshold: float,
        eval_iou_threshold: float,
        ground_resolution: float
):
    try:
        print(f"Evaluating NMS IOU threshold: {nms_iou_threshold}, NMS score threshold: {nms_score_threshold}, output will be at {output_path}")

        output_path = Path(output_path) / f"nmsiou_{str(nms_iou_threshold).replace('.', 'p')}_nmsscorethresh_{str(nms_score_threshold).replace('.', 'p')}"
        Path(output_path).mkdir(parents=True, exist_ok=True)
        aggregator_output_path = output_path / 'aggregator_output.gpkg'

        Aggregator.from_coco(
            polygon_type='bbox',
            output_path=aggregator_output_path,
            tiles_folder_path=tiles_root,
            coco_json_path=preds_coco_json,
            scores_names=['detector_score'],
            other_attributes_names=None,
            scores_weights=[1.0],
            min_centroid_distance_weight=1.0,
            score_threshold=nms_score_threshold,
            nms_threshold=nms_iou_threshold,
            nms_algorithm='iou',
            edge_band_buffer_percentage=0.05,
            best_geom_keep_area_ratio=0.5,
            pre_aggregated_output_path=None
        )

        # Evaluate the predictions
        evaluator = CocoEvaluator()
        metrics = evaluator.raster_level_single_iou_threshold(
            iou_type='bbox',
            preds_gpkg_path=str(aggregator_output_path),
            truth_gpkg_path=truth_gdf,
            aoi_gpkg_path=aoi_gdf,
            ground_resolution=ground_resolution,
            iou_threshold=eval_iou_threshold
        )

    except Exception as e:
        traceback.print_exc()
        raise e

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
        preds_coco_jsons: list[str],
        truths_gdfs: list[str],
        aois_gdfs: list[str],
        tiles_roots: list[str],
        ground_resolution: float,
        nms_iou_thresholds: list[float],
        nms_score_thresholds: list[float],
        eval_iou_threshold: float,
        n_workers: int
):

    assert len(raster_names) == len(preds_coco_jsons) == len(truths_gdfs) == len(tiles_roots) == len(aois_gdfs), \
        "The number of elements in raster_names, preds_coco_jsons, truths_gdfs, tiles_roots and aois_gdfs must be the same."

    # Create a list to hold all parameter combinations
    tasks = []
    for nms_iou_threshold in nms_iou_thresholds:
        for nms_score_threshold in nms_score_thresholds:
            for raster_name, preds_coco_json, truth_gdf, tiles_root, aoi_gdf in zip(raster_names, preds_coco_jsons, truths_gdfs, tiles_roots, aois_gdfs):
                tasks.append({
                    "raster_name": raster_name,
                    "nms_iou_threshold": nms_iou_threshold,
                    "nms_score_threshold": nms_score_threshold,
                    "preds_coco_json": preds_coco_json,
                    "truth_gdf": truth_gdf,
                    "tiles_root": tiles_root,
                    "aoi_gdf": aoi_gdf
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
                aoi_gdf=params["aoi_gdf"],
                nms_iou_threshold=params["nms_iou_threshold"],
                nms_score_threshold=params["nms_score_threshold"],
                eval_iou_threshold=eval_iou_threshold,
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
                print(f"Parameters {params} generated an exception:")
                traceback.print_exc()

    results_df = pd.DataFrame(results_list)

    # Compute weighted average of all metrics for each parameter combination, over the different rasters.
    # The weights are the number of truth bbox in each raster ('num_truths').
    results_df = average_metrics_by_raster(results_df)

    return results_df


