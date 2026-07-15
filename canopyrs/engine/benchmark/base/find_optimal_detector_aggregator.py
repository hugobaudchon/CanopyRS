from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import os
from pathlib import Path
import traceback

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from geodataset.aggregator import Aggregator as GdAggregator

from canopyrs.engine.benchmark.base.evaluator import CocoEvaluator
from canopyrs.engine.components.aggregator import Aggregator as AggregatorComponent
from canopyrs.engine.config_parsers import AggregatorConfig
from canopyrs.engine.constants import Col
from canopyrs.engine.data import Tiles, Objects
from canopyrs.engine.pipeline import Pipeline


def eval_single_aggregator(
        output_path: str,
        model_run_dir: str,
        truth_gdf: str,
        aoi_gdf: str,
        eval_iou_threshold: float | list[float],
        ground_resolution: float,
        iou_type: str,
        aggregator_config: AggregatorConfig,
):
    """Legacy per-grid-cell path: reload the run, aggregate, export a gpkg and evaluate one
    (nms_threshold, score_threshold) cell. Kept as the reference implementation for validating
    ``grid_search_single_raster_iou`` — the grid search itself no longer calls it."""
    if iou_type not in ('bbox', 'segm'):
        raise ValueError(f"Unsupported iou_type: {iou_type}. Expected 'bbox' or 'segm'.")

    with open(os.devnull, "w") as devnull, \
        contextlib.redirect_stdout(devnull), \
        contextlib.redirect_stderr(devnull):
        output_path = Path(output_path) / f"nmsiou_{str(aggregator_config.nms_threshold).replace('.', 'p')}_nmsscorethresh_{str(aggregator_config.score_threshold).replace('.', 'p')}"
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # Reload the model-only run (Tiles + Objects, in tile-pixel coords, scores intact) and seed an
        # aggregator-only pipeline from it — no pixel-coords gpkg round-trip.
        prior = Pipeline.from_dir(model_run_dir)
        pipeline = Pipeline.from_config(
            [('aggregator', aggregator_config)],
            tiles=prior.latest(Tiles),
            objects=prior.latest(Objects),
            output_dir=str(output_path),
        )
        pipeline.run(verbose=False)

        aggregator_output_path = pipeline.export("gpkg")

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


def _score_threshold_mask(scores: np.ndarray, score_threshold: float) -> np.ndarray:
    """Survivors at a given score threshold, mirroring GdAggregator._remove_low_score_polygons
    (a falsy threshold, or an all-zero score column, filters nothing)."""
    if not score_threshold or not scores.any():
        return np.ones(len(scores), dtype=bool)
    return scores >= score_threshold


def grid_search_single_raster_iou(
        output_path: str,
        model_run_dir: str,
        truth_gdf: str,
        aoi_gdf: str,
        eval_iou_thresholds: list[float],
        ground_resolution: float,
        iou_type: str,
        aggregator_config: AggregatorConfig,
        nms_iou_threshold: float,
        nms_score_thresholds: list[float],
) -> list[dict]:
    """Grid-search the score-threshold axis for one (raster, nms_iou_threshold) pair.

    This is the parallel task unit: the nms_iou axis is irreducible (greedy-NMS survivor sets
    aren't monotone in the IoU threshold, so each nms_iou needs its own full pass), so it is
    parallelized across tasks rather than collapsed. The score axis, by contrast, collapses
    exactly: NMS runs once at ``min(nms_score_thresholds)``, then every higher score threshold is
    derived by masking survivors on ``aggregator_score`` — a polygon below a score threshold can
    never suppress or carve one above it, so pre-filtering by score commutes with NMS. One
    evaluation context is built on the survivors and reused for every score threshold via masks.

    The model outputs are loaded and georeferenced within the task (so per raster this repeats
    once per nms_iou — cheap next to NMS). Aggregator gpkg artifacts are written once per task
    (``nmsiou_*/aggregator.gpkg``).

    Returns one dict per score threshold: {'nms_iou_threshold', 'nms_score_threshold',
    'aggregator_config', **metrics}.
    """
    if iou_type not in ('bbox', 'segm'):
        raise ValueError(f"Unsupported iou_type: {iou_type}. Expected 'bbox' or 'segm'.")

    min_score_threshold = min(nms_score_thresholds)
    rows: list[dict] = []

    with open(os.devnull, "w") as devnull, \
        contextlib.redirect_stdout(devnull), \
        contextlib.redirect_stderr(devnull):
        prior = Pipeline.from_dir(model_run_dir)
        objects = prior.latest(Objects)
        if objects is None or len(objects) == 0:
            print(f"Grid search: no model outputs found in {model_run_dir}; skipping raster.")
            return rows

        # Georeference the model outputs (same path the aggregator component runs).
        component = AggregatorComponent(aggregator_config)
        tiles = objects.linked("tiles")
        crs = tiles.df[Col.TILE_METADATA].iloc[0]["crs"] if len(tiles) else None
        polygons_gdf, tiles_extent_gdf = component._georeference(objects, tiles, crs)
        scores_names, scores_weights = component._scores()
        tile_ids_to_path = component._tile_paths(tiles, tiles_extent_gdf[Col.TILE_ID])

        # Single NMS run at the minimum score threshold; higher score thresholds are exact subsets.
        out_dir = Path(output_path) / f"nmsiou_{str(nms_iou_threshold).replace('.', 'p')}"
        agg = GdAggregator(
            output_path=out_dir / "aggregator.gpkg",
            polygons_gdf=polygons_gdf,
            scores_names=scores_names,
            other_attributes_names=[Col.PREV_OBJECT_ID],
            scores_weights=scores_weights,
            tiles_extent_gdf=tiles_extent_gdf,
            tile_ids_to_path=tile_ids_to_path,
            scores_weighting_method=aggregator_config.scores_weighting_method,
            min_centroid_distance_weight=aggregator_config.min_centroid_distance_weight,
            score_threshold=min_score_threshold,
            nms_threshold=nms_iou_threshold,
            nms_algorithm=aggregator_config.nms_algorithm,
            edge_band_buffer_percentage=aggregator_config.edge_band_buffer_percentage,
            best_geom_keep_area_ratio=aggregator_config.best_geom_keep_area_ratio,
        )
        survivors = agg.polygons_gdf

        truth_gdf_loaded = gpd.read_file(truth_gdf)
        aoi_gdf_loaded = gpd.read_file(aoi_gdf) if aoi_gdf is not None else None

        context = CocoEvaluator.build_raster_eval_context(
            iou_type=iou_type,
            preds=survivors,
            truths=truth_gdf_loaded,
            aoi=aoi_gdf_loaded,
            ground_resolution=ground_resolution,
        )
        survivor_scores = survivors[Col.AGGREGATOR_SCORE].to_numpy(dtype=float)

        for score_threshold in nms_score_thresholds:
            mask = _score_threshold_mask(survivor_scores, score_threshold)
            metrics = CocoEvaluator.evaluate_raster_from_context(
                context, iou_thresholds=eval_iou_thresholds, pred_mask=mask)
            cell_config = aggregator_config.model_copy(deep=True)
            cell_config.nms_threshold = nms_iou_threshold
            cell_config.score_threshold = score_threshold
            rows.append({
                'nms_iou_threshold': nms_iou_threshold,
                'nms_score_threshold': score_threshold,
                'aggregator_config': cell_config,
                **metrics,
            })

    return rows


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
        model_run_dirs: list[str],
        truths_gdfs: list[str],
        aois_gdfs: list[str],
        ground_resolution: float,
        nms_iou_thresholds: list[float],
        nms_score_thresholds: list[float],
        eval_iou_threshold: float | list[float],
        n_workers: int,
        iou_type: str,
        aggregator_config: AggregatorConfig = None,
):
    """Grid-search NMS thresholds over all rasters.

    Parallelized per (raster, nms_iou_threshold): each worker loads and georeferences its raster's
    model outputs, runs NMS once for its nms_iou (at the minimum score threshold), derives the
    score-threshold axis by filtering survivors, and evaluates every score threshold against a
    precomputed evaluation context. See ``grid_search_single_raster_iou`` for the exactness
    argument. This granularity keeps all workers busy even with few rasters (tasks =
    n_rasters x |nms_iou_thresholds|) while still collapsing the score axis.
    """

    assert len(raster_names) == len(model_run_dirs) == len(truths_gdfs) == len(aois_gdfs), \
        "The number of elements in raster_names, model_run_dirs, truths_gdfs and aois_gdfs must be the same."
    assert nms_iou_thresholds and nms_score_thresholds, \
        "nms_iou_thresholds and nms_score_thresholds must be non-empty."

    if isinstance(eval_iou_threshold, (list, tuple)):
        normalized_iou_thresholds = [float(t) for t in eval_iou_threshold]
    else:
        normalized_iou_thresholds = [float(eval_iou_threshold)]

    # Create a base aggregator config if not provided
    if aggregator_config is None:
        aggregator_config = AggregatorConfig()

    results_list = []

    # Parallelize using ProcessPoolExecutor, one task per (raster, nms_iou_threshold)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_params = {}
        for raster_name, model_run_dir, truth_gdf, aoi_gdf in zip(raster_names, model_run_dirs, truths_gdfs, aois_gdfs):
            for nms_iou_threshold in nms_iou_thresholds:
                future = executor.submit(
                    grid_search_single_raster_iou,
                    output_path=f"{output_folder}/{raster_name}",
                    model_run_dir=model_run_dir,
                    truth_gdf=truth_gdf,
                    aoi_gdf=aoi_gdf,
                    eval_iou_thresholds=normalized_iou_thresholds,
                    ground_resolution=ground_resolution,
                    iou_type=iou_type,
                    aggregator_config=aggregator_config,
                    nms_iou_threshold=nms_iou_threshold,
                    nms_score_thresholds=nms_score_thresholds,
                )
                future_to_params[future] = {
                    "raster_name": raster_name,
                    "model_run_dir": model_run_dir,
                    "truth_gdf": truth_gdf,
                    "aoi_gdf": aoi_gdf,
                }

        # Collect the results as they complete
        with tqdm(total=len(future_to_params), desc="Grid search", unit="task") as pbar:
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    for row in future.result():
                        results_list.append({**params, **row})
                except Exception:
                    print(f"Raster {params['raster_name']} generated an exception:")
                    traceback.print_exc()
                finally:
                    pbar.update(1)

    results_df = pd.DataFrame(results_list)

    # Compute weighted average of all metrics for each parameter combination, over the different rasters.
    # The weights are the number of truth bbox in each raster ('num_truths').
    results_df = average_metrics_by_raster(results_df)

    return results_df
