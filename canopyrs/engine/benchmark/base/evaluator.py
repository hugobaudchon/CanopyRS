import copy
import warnings
from pathlib import Path

from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import Optional, Dict, List

import numpy as np
import geopandas as gpd
import rasterio
from shapely import make_valid
from shapely.affinity import affine_transform
from geodataset.utils import get_utm_crs

from faster_coco_eval.core.coco import COCO
from faster_coco_eval.core.faster_eval_api import COCOeval_faster      # speeds up raster level evaluation by 10-100x


class AlignmentStrategy(Enum):
    BASE_RSPLIT_1 = "base_rsplit_1"
    EXACT_MATCH = "exact"


class AlignmentError(Exception):
    pass


class LowMatchRateWarning(UserWarning):
    pass


@dataclass
class AlignmentReport:
    num_gt_images: int
    num_pred_images: int
    num_matched: int
    num_gt_without_pred: int
    num_pred_outside_gt: int
    match_rate: float
    unmatched_gt_files: List[str] = dataclass_field(default_factory=list)
    gt_filename_samples: List[str] = dataclass_field(default_factory=list)
    pred_filename_samples: List[str] = dataclass_field(default_factory=list)
    strategy_used: str = AlignmentStrategy.EXACT_MATCH.value

    def print_report(self, verbose: bool = True):
        print("\n" + "=" * 70)
        print("COCO DATASET ALIGNMENT REPORT")
        print("=" * 70)
        print("Strategy: {}".format(self.strategy_used))
        print("\nDataset Sizes:")
        print("  Ground Truth images: {}".format(self.num_gt_images))
        print("  Prediction images: {}".format(self.num_pred_images))
        print("\nAlignment Results:")
        print("  Successfully matched: {} ({:.1%} of GT)".format(
            self.num_matched, self.match_rate
        ))
        print("  GT tiles without predictions: {}".format(self.num_gt_without_pred))
        print("  Prediction tiles outside GT scope: {}".format(self.num_pred_outside_gt))
        print("    (These will be ignored - expected for extended prediction areas)")

        if verbose and self.unmatched_gt_files:
            print("\nUnmatched GT tiles (first 10):")
            for filename in self.unmatched_gt_files[:10]:
                print("    - {}".format(filename))
            if len(self.unmatched_gt_files) > 10:
                print("    ... and {} more".format(len(self.unmatched_gt_files) - 10))

        if verbose and self.gt_filename_samples:
            print("\nSample GT filenames (first 3):")
            for filename in self.gt_filename_samples[:3]:
                print("    - {}".format(filename))

        if verbose and self.pred_filename_samples:
            print("\nSample Pred filenames (first 3):")
            for filename in self.pred_filename_samples[:3]:
                print("    - {}".format(filename))

        print("=" * 70 + "\n")

    def to_dict(self) -> Dict:
        return {
            'num_gt_images': self.num_gt_images,
            'num_pred_images': self.num_pred_images,
            'num_matched': self.num_matched,
            'num_gt_without_pred': self.num_gt_without_pred,
            'num_pred_outside_gt': self.num_pred_outside_gt,
            'match_rate': self.match_rate,
            'strategy_used': self.strategy_used,
            'unmatched_gt_files': self.unmatched_gt_files,
        }


class CocoEvaluator:
    small_max_sq_meters = 16
    medium_max_sq_meters = 100
    gsd_sample_size = 50
    gsd_rel_tol = 0.01

    size_thresholds_sq_meters = {
        'tiny': (0, 9),
        'small': (9, 25),
        'medium': (25, 49),
        'large': (49, 100),
        'giant': (100, float('inf'))
    }
    size_labels = ['tiny', 'small', 'medium', 'large', 'giant']

    @classmethod
    def _get_area_ranges_pixels_from_gsd(cls, ground_resolution: float) -> dict:
        pixel_area_per_sq_meter = 1 / (ground_resolution ** 2)
        area_ranges = {}
        for category, (min_m2, max_m2) in cls.size_thresholds_sq_meters.items():
            min_px = min_m2 * pixel_area_per_sq_meter
            max_px = max_m2 * pixel_area_per_sq_meter if max_m2 != float('inf') else 1e10
            area_ranges[category] = (min_px, max_px)
        return area_ranges

    @classmethod
    def get_size_label(cls, area: float, area_ranges: dict) -> str:
        for label in cls.size_labels:
            min_a, max_a = area_ranges[label]
            if min_a <= area < max_a:
                return label
        return cls.size_labels[-1]

    def tile_level(
            self,
            iou_type: str,
            preds_coco_path: str,
            truth_coco_path: str,
            max_dets: list = (1, 10, 100),
            images_common_ground_resolution: float = None,
    ) -> dict:

        truth_coco = COCO(str(truth_coco_path))
        preds_coco = COCO(str(preds_coco_path))

        for ann in truth_coco.dataset['annotations']:
            if 'score' in ann:
                del ann['score']

        # Align predictions to truth based on file name
        align_coco_datasets_by_name(truth_coco, preds_coco)

        return self._evaluate_coco(
            truth_coco, preds_coco, iou_type, max_dets,
            images_common_ground_resolution=(
                images_common_ground_resolution
            ),
        )

    # ------------------------------------------------------------------
    # Low-level helpers reusable by subclasses (classifier, etc.)
    # ------------------------------------------------------------------

    def _evaluate_coco(
            self,
            truth_coco,
            preds_coco,
            iou_type: str,
            max_dets: list = (1, 10, 100),
            images_common_ground_resolution: float = None,
    ) -> dict:
        """Evaluate pre-aligned COCO objects for a single IoU type.

        Does **not** copy the inputs.  Callers that need to run
        multiple evaluations on the same data should copy first.
        """
        coco_evaluator = Summarize2COCOEval(
            cocoGt=truth_coco,
            cocoDt=preds_coco,
            iouType=iou_type,
        )
        coco_evaluator.params.maxDets = list(max_dets)

        if images_common_ground_resolution is not None:
            ar = self._get_area_ranges_pixels_from_gsd(
                images_common_ground_resolution)
            coco_evaluator.params.areaRng = [
                [0, 1e10],
                [ar['tiny'][0], ar['tiny'][1]],
                [ar['small'][0], ar['small'][1]],
                [ar['medium'][0], ar['medium'][1]],
                [ar['large'][0], ar['large'][1]],
                [ar['giant'][0], ar['giant'][1]],
            ]
            coco_evaluator.params.areaRngLbl = [
                'all', 'tiny', 'small',
                'medium', 'large', 'giant',
            ]

        coco_evaluator.evaluate()
        coco_evaluator.accumulate()

        metrics = coco_evaluator.summarize_to_dict()
        metrics['num_images'] = len(
            truth_coco.dataset.get('images', []))
        metrics['num_truths'] = len(
            truth_coco.dataset.get('annotations', []))
        metrics['num_preds'] = len(
            preds_coco.dataset.get('annotations', []))
        metrics['iou_type'] = iou_type

        if (hasattr(coco_evaluator, 'ious')
                and coco_evaluator.ious):
            metrics['mIoU'] = self._compute_miou(
                coco_evaluator)

        return metrics

    @staticmethod
    def _compute_miou(coco_evaluator) -> float:
        """Mean IoU from a COCOeval object's *ious* dict."""
        try:
            all_ious = []
            ious = getattr(coco_evaluator, 'ious', None)
            if ious:
                for _key, iou_matrix in ious.items():
                    if isinstance(iou_matrix, np.ndarray):
                        flat = iou_matrix.flatten()
                        valid = flat[
                            (flat > 0) & (flat <= 1)]
                        all_ious.extend(valid.tolist())
            if all_ious:
                return float(np.mean(all_ious))
            return 0.0
        except (ValueError, TypeError) as exc:
            print("Error computing mIoU: {}".format(exc))
            return 0.0

    def evaluate_class_agnostic(
            self,
            truth_coco,
            preds_coco,
            max_dets: list = (1, 10, 100),
    ) -> dict:
        """Segmentation eval ignoring category_id.

        Deep-copies datasets, forces all category_id to 1,
        then delegates to ``_evaluate_coco``.
        """
        truth_copy = COCO()
        truth_copy.dataset = copy.deepcopy(
            truth_coco.dataset)
        pred_copy = COCO()
        pred_copy.dataset = copy.deepcopy(
            preds_coco.dataset)

        one_cat = [
            {'id': 1, 'name': 'object', 'supercategory': ''}
        ]
        truth_copy.dataset['categories'] = one_cat
        pred_copy.dataset['categories'] = list(one_cat)

        for ann in truth_copy.dataset.get(
                'annotations', []):
            ann['category_id'] = 1
        for ann in pred_copy.dataset.get(
                'annotations', []):
            ann['category_id'] = 1

        truth_copy.createIndex()
        pred_copy.createIndex()

        return self._evaluate_coco(
            truth_copy, pred_copy, 'segm', max_dets)

    def compute_classification_metrics(
            self,
            truth_coco,
            preds_coco,
    ) -> dict:
        """Classification accuracy for matched instances.

        Matches GT → pred by bbox IoU > 0.5, then checks
        ``category_id`` agreement.
        """
        total_matches = 0
        correct_class = 0
        per_class_stats = {}

        for img_id in truth_coco.getImgIds():
            gt_anns = truth_coco.loadAnns(
                truth_coco.getAnnIds(imgIds=img_id))
            pred_anns = preds_coco.loadAnns(
                preds_coco.getAnnIds(imgIds=img_id))

            if not gt_anns or not pred_anns:
                continue

            for gt_ann in gt_anns:
                gt_cat = gt_ann['category_id']
                if gt_cat not in per_class_stats:
                    per_class_stats[gt_cat] = {
                        'total': 0, 'correct': 0,
                    }

                best_iou = 0.0
                best_pred = None
                for pred_ann in pred_anns:
                    iou = self._compute_bbox_iou(
                        gt_ann['bbox'],
                        pred_ann['bbox'],
                    )
                    if iou > best_iou:
                        best_iou = iou
                        best_pred = pred_ann

                if best_iou > 0.5 and best_pred is not None:
                    total_matches += 1
                    per_class_stats[gt_cat]['total'] += 1
                    if best_pred['category_id'] == gt_cat:
                        correct_class += 1
                        per_class_stats[gt_cat][
                            'correct'] += 1

        accuracy = (
            correct_class / total_matches
            if total_matches > 0 else 0.0
        )

        per_class_acc = {}
        for cat_id, stats in per_class_stats.items():
            per_class_acc[cat_id] = (
                stats['correct'] / stats['total']
                if stats['total'] > 0 else 0.0
            )

        return {
            'overall_accuracy': accuracy,
            'total_matched_instances': total_matches,
            'correctly_classified': correct_class,
            'per_class_accuracy': per_class_acc,
            'per_class_counts': per_class_stats,
        }

    @staticmethod
    def _compute_bbox_iou(box1, box2) -> float:
        """IoU between two COCO-format bboxes [x, y, w, h]."""
        x1_min, y1_min = box1[0], box1[1]
        x1_max = x1_min + box1[2]
        y1_max = y1_min + box1[3]

        x2_min, y2_min = box2[0], box2[1]
        x2_max = x2_min + box2[2]
        y2_max = y2_min + box2[3]

        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)

        if xi_max <= xi_min or yi_max <= yi_min:
            return 0.0

        intersection = (
            (xi_max - xi_min) * (yi_max - yi_min))
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def raster_level_single_iou_threshold(iou_type: str,
                                          preds_gpkg_path: str,
                                          truth_gpkg_path: str,
                                          aoi_gpkg_path: str or None,
                                          ground_resolution: float = 0.045,
                                          iou_threshold: float = 0.5) -> dict:
        """
        Backward-compatible single-IoU evaluation wrapper.
        """
        return CocoEvaluator.raster_level_multi_iou_thresholds(
            iou_type=iou_type,
            preds_gpkg_path=preds_gpkg_path,
            truth_gpkg_path=truth_gpkg_path,
            aoi_gpkg_path=aoi_gpkg_path,
            ground_resolution=ground_resolution,
            iou_thresholds=[float(iou_threshold)]
        )

    @staticmethod
    def raster_level_multi_iou_thresholds(iou_type: str,
                                          preds_gpkg_path: str,
                                          truth_gpkg_path: str,
                                          aoi_gpkg_path: str or None,
                                          ground_resolution: float = 0.045,
                                          iou_thresholds: list[float] | None = None) -> dict:
        """
        Compute precision, recall, and F1 averaged over one or more IoU thresholds
        between prediction and ground-truth GeoDataFrames (RF1-style).

        iou_type: type of IoU to compute (e.g., 'bbox', 'segm').
        preds_gpkg_path: path to GeoDataFrame with a 'geometry' column and a 'score', 'aggregator_score', 'detector_score' or 'segmentation_score' column (will be checked in that order).
        truth_gpkg_path: path to GeoDataFrame with a 'geometry' column.
        aoi_gpkg_path: path to GeoDataFrame with a 'geometry' column (optional).
        iou_thresholds: List of IoU thresholds to average over (e.g. [0.50, 0.55, ..., 0.95]).
        """

        if iou_thresholds is None or len(iou_thresholds) == 0:
            iou_thresholds = [0.5]
        # De-duplicate and sort for deterministic output
        iou_thresholds = sorted({float(t) for t in iou_thresholds})

        # Load the prediction and ground truth GeoDataFrames
        infer_gdf = gpd.read_file(preds_gpkg_path)
        truth_gdf = gpd.read_file(truth_gpkg_path)

        # Apply IoU type on polygons
        if iou_type == 'segm':
            infer_gdf = infer_gdf
            truth_gdf = truth_gdf
        elif iou_type == 'bbox':
            infer_gdf['geometry'] = infer_gdf.geometry.envelope
            truth_gdf['geometry'] = truth_gdf.geometry.envelope
        else:
            raise ValueError(f"Unsupported IoU type: {iou_type}. Supported types are 'bbox' and 'segm'.")

        common_crs = truth_gdf.crs
        if not common_crs.is_projected:
            bounds = truth_gdf.total_bounds
            centroid_lon = (bounds[0] + bounds[2]) / 2.0
            centroid_lat = (bounds[1] + bounds[3]) / 2.0
            common_crs = get_utm_crs(centroid_lon, centroid_lat)

        infer_gdf = infer_gdf.to_crs(common_crs)
        truth_gdf = truth_gdf.to_crs(common_crs)

        # Only keep the truth and inference geometries that are inside the AOI (40% overlap minimum)
        if aoi_gpkg_path is not None:
            aoi_gdf = gpd.read_file(aoi_gpkg_path).to_crs(common_crs)
            aoi_union = aoi_gdf.geometry.unary_union
            truth_gdf = filter_min_overlap(truth_gdf, aoi_union, min_frac=0.4)
            infer_gdf = filter_min_overlap(infer_gdf, aoi_union, min_frac=0.4)
            truth_gdf = gpd.overlay(truth_gdf, aoi_gdf, how='intersection')
            infer_gdf = gpd.overlay(infer_gdf, aoi_gdf, how='intersection')
        else:
            warnings.warn("AOI GPKG path is None. No AOI filtering will be applied."
                          " Please make sure the truth gpkg extent matches the prediction one or the metrics will be"
                          " low if the truth gpkg extent is much larger than the prediction one (i.e if truth gpkg has"
                          " train, valid and test folds sections, you only want to eval against valid or test areas).")

        truth_gdf, infer_gdf = move_gdfs_to_ground_resolution(truth_gdf, infer_gdf, ground_resolution)

        area_ranges = CocoEvaluator._get_area_ranges_pixels_from_gsd(ground_resolution)
        size_labels = CocoEvaluator.size_labels

        score_column_name = None
        for score_col in ['score', 'aggregator_score', 'detector_score', 'segmentation_score']:
            if score_col in infer_gdf.columns:
                score_column_name = score_col
                break

        if score_column_name is None:
            raise ValueError("No valid score column found in predictions GeoDataFrame. "
                             "Please ensure it contains 'score', 'aggregator_score', 'detector_score' or 'segmentation_score'.")

        infer_gdf = infer_gdf.sort_values(score_column_name, ascending=False).reset_index(drop=True)
        truth_gdf = truth_gdf.reset_index(drop=True)

        # Precompute size categories based on area (pixel space after transform), after sorting/resets
        gt_sizes = [CocoEvaluator.get_size_label(a, area_ranges) for a in truth_gdf.geometry.area]
        pred_sizes = [CocoEvaluator.get_size_label(a, area_ranges) for a in infer_gdf.geometry.area]

        gt_counts = {lbl: 0 for lbl in size_labels}
        pred_counts = {lbl: 0 for lbl in size_labels}
        for lbl in gt_sizes:
            gt_counts[lbl] += 1
        for lbl in pred_sizes:
            pred_counts[lbl] += 1
        print(f"GT per size: {gt_counts} | Preds per size: {pred_counts}")

        truth_sindex = truth_gdf.sindex

        # Pre-compute candidate IoUs for each prediction to reuse across thresholds
        candidate_ious: list[list[tuple[int, float]]] = []
        for _, pred in infer_gdf.iterrows():
            candidates = list(truth_sindex.intersection(pred.geometry.bounds))
            cand_iou: list[tuple[int, float]] = []
            for idx in candidates:
                truth_geom = truth_gdf.at[idx, "geometry"]
                inter_area = pred.geometry.intersection(truth_geom).area
                union_area = pred.geometry.union(truth_geom).area
                iou = inter_area / union_area if union_area > 0 else 0.0
                cand_iou.append((idx, iou))
            # Sort descending IoU to speed thresholded greedy matching
            cand_iou.sort(key=lambda x: x[1], reverse=True)
            candidate_ious.append(cand_iou)

        size_labels_with_all = ['all'] + size_labels
        per_size_iou_results: dict[str, list[dict]] = {lbl: [] for lbl in size_labels_with_all}

        # Precompute index lists per size for reuse
        gt_indices_by_size = {lbl: [i for i, s in enumerate(gt_sizes) if lbl == 'all' or s == lbl]
                              for lbl in size_labels_with_all}
        pred_indices_by_size = {lbl: [i for i, s in enumerate(pred_sizes) if lbl == 'all' or s == lbl]
                                for lbl in size_labels_with_all}

        for iou_thresh in iou_thresholds:
            for size_lbl in size_labels_with_all:
                gt_indices = gt_indices_by_size[size_lbl]
                pred_indices = pred_indices_by_size[size_lbl]

                if not gt_indices and not pred_indices:
                    per_size_iou_results[size_lbl].append({
                        'iou_threshold': iou_thresh,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1': 0.0,
                        'tp': 0,
                        'fp': 0,
                        'fn': 0
                    })
                    continue

                gt_idx_to_local = {idx: local for local, idx in enumerate(gt_indices)}
                matched = np.zeros(len(gt_indices), dtype=bool)
                tp = 0
                fp = 0

                for pred_idx in pred_indices:
                    cand_list = candidate_ious[pred_idx]
                    match_local_idx = None
                    for gt_idx, iou in cand_list:
                        if iou < iou_thresh:
                            break  # remaining candidates have lower IoU
                        if gt_idx not in gt_idx_to_local:
                            continue
                        local_idx = gt_idx_to_local[gt_idx]
                        if not matched[local_idx]:
                            match_local_idx = local_idx
                            break
                    if match_local_idx is not None:
                        tp += 1
                        matched[match_local_idx] = True
                    else:
                        fp += 1

                fn = (~matched).sum()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                per_size_iou_results[size_lbl].append({
                    'iou_threshold': iou_thresh,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn
                })

        # Aggregate across IoU thresholds (RF1_50:95-style averaging) for the "all" bin
        all_results = per_size_iou_results['all']
        precisions = [m['precision'] for m in all_results]
        recalls = [m['recall'] for m in all_results]
        f1s = [m['f1'] for m in all_results]

        metrics = {
            'precision': float(np.mean(precisions)) if precisions else 0.0,
            'recall': float(np.mean(recalls)) if recalls else 0.0,
            'f1': float(np.mean(f1s)) if f1s else 0.0,
            'tp': all_results[0]['tp'] if all_results else 0,
            'fp': all_results[0]['fp'] if all_results else 0,
            'fn': all_results[0]['fn'] if all_results else 0,
            'num_truths': len(truth_gdf),
            'num_preds': len(infer_gdf),
            'num_images': 1,
            # Keep sorted per-IoU scores to log downstream
            'precision_per_iou': precisions,
            'recall_per_iou': recalls,
            'f1_per_iou': f1s,
            'iou_thresholds': iou_thresholds,
            'size_labels': [s for s in size_labels_with_all if s != 'all']
        }

        def _find_iou_index(target_iou: float) -> int | None:
            for idx, iou in enumerate(iou_thresholds):
                if np.isclose(iou, target_iou):
                    return idx
            return None

        # Expose common single-threshold metrics when 0.50 or 0.75 are requested
        for target_iou, suffix in [(0.5, '50'), (0.75, '75')]:
            idx = _find_iou_index(target_iou)
            if idx is not None and idx < len(all_results):
                result = all_results[idx]
                metrics[f'precision_{suffix}'] = float(result['precision'])
                metrics[f'recall_{suffix}'] = float(result['recall'])
                metrics[f'f1_{suffix}'] = float(result['f1'])

        # Average per-size over IoU thresholds
        for lbl in size_labels_with_all:
            if lbl == 'all':
                continue
            results = per_size_iou_results[lbl]
            prec_list = [m['precision'] for m in results]
            rec_list = [m['recall'] for m in results]
            f1_list = [m['f1'] for m in results]
            metrics[f'precision_{lbl}'] = float(np.mean(prec_list)) if prec_list else 0.0
            metrics[f'recall_{lbl}'] = float(np.mean(rec_list)) if rec_list else 0.0
            metrics[f'f1_{lbl}'] = float(np.mean(f1_list)) if f1_list else 0.0
            metrics[f'precision_per_iou_{lbl}'] = [float(m['precision']) for m in results]
            metrics[f'recall_per_iou_{lbl}'] = [float(m['recall']) for m in results]
            metrics[f'f1_per_iou_{lbl}'] = [float(m['f1']) for m in results]

        print(metrics)

        return metrics


def move_gdfs_to_ground_resolution(truth_gdf: gpd.GeoDataFrame, infer_gdf: gpd.GeoDataFrame, ground_resolution: float) -> (gpd.GeoDataFrame, gpd.GeoDataFrame):
    # Make sure both have a CRS
    assert truth_gdf.crs is not None, "Truth GeoDataFrame must have a CRS"
    assert infer_gdf.crs is not None, "Inference GeoDataFrame must have a CRS"

    # If the two datasets use different CRS, reproject the inference to match truth
    if truth_gdf.crs != infer_gdf.crs:
        infer_gdf = infer_gdf.to_crs(truth_gdf.crs)

    # If the truth CRS is not projected (i.e. not in linear units such as meters),
    # reproject both to an appropriate UTM (meter-based) CRS.
    if not truth_gdf.crs.is_projected:
        bounds = truth_gdf.total_bounds  # [minx, miny, maxx, maxy]
        centroid_lon = (bounds[0] + bounds[2]) / 2.0
        centroid_lat = (bounds[1] + bounds[3]) / 2.0
        utm_crs = get_utm_crs(centroid_lon, centroid_lat)
        truth_gdf = truth_gdf.to_crs(utm_crs)
        infer_gdf = infer_gdf.to_crs(utm_crs)

    # Compute the overall minimum coordinates from the truth data
    b1 = truth_gdf.total_bounds  # [minx, miny, maxx, maxy]
    b2 = infer_gdf.total_bounds
    combined_bounds = [
        min(b1[0], b2[0]),  # minx
        min(b1[1], b2[1]),  # miny
        max(b1[2], b2[2]),  # maxx
        max(b1[3], b2[3])  # maxy
    ]

    # Create an affine transformation that translates and scales coordinates:
    # new_x = (old_x - minx) / ground_resolution
    # new_y = (old_y - miny) / ground_resolution
    affine_params = [
        1 / ground_resolution, 0,     # scale x, no rotation
        0, 1 / ground_resolution,       # scale y, no rotation
        -combined_bounds[0] / ground_resolution,      # translation in x
        -combined_bounds[1] / ground_resolution       # translation in y
    ]

    # Apply the affine transformation to each geometry in both GeoDataFrames.
    truth_gdf['geometry'] = truth_gdf.geometry.apply(lambda geom: affine_transform(geom, affine_params))
    infer_gdf['geometry'] = infer_gdf.geometry.apply(lambda geom: affine_transform(geom, affine_params))
    
    # Validate and repair geometries in both GeoDataFrames
    truth_gdf = validate_and_repair_gdf(truth_gdf, "ground truth")
    infer_gdf = validate_and_repair_gdf(infer_gdf, "predictions")
    
    # Return the updated GeoDataFrames in a dictionary.
    return truth_gdf, infer_gdf

def validate_and_repair_gdf(gdf: gpd.GeoDataFrame, name: str) -> gpd.GeoDataFrame:
    """Helper function to validate and repair geometries in a GeoDataFrame."""
    original_count = len(gdf)
    invalid_mask = ~gdf.is_valid
    n_invalid = invalid_mask.sum()
    
    if n_invalid > 0:
        print(f"Found {n_invalid} invalid {name} geometries after transformation. Repairing...")
        
        # Apply repair logic directly using vectorized operations
        repaired_geometries = []
        for geom in gdf.geometry:
            if not geom.is_valid:
                repaired = make_valid(geom)
                # Handle GeometryCollection
                if repaired.geom_type == 'GeometryCollection':
                    polygons = [g for g in repaired.geoms 
                               if g.geom_type in ['Polygon', 'MultiPolygon']]
                    if polygons:
                        repaired_geometries.append(max(polygons, key=lambda g: g.area))
                    else:
                        repaired_geometries.append(None)
                else:
                    repaired_geometries.append(repaired)
            else:
                repaired_geometries.append(geom)

        gdf['geometry'] = repaired_geometries

        # Remove None/invalid geometries
        gdf = gdf[gdf.geometry.notna() & gdf.is_valid].copy()
        
        # Report final counts
        removed_count = original_count - len(gdf)
        
        if removed_count > 0:
            print(f"Removed {removed_count} &  Kept {len(gdf)}/{original_count} {name} geometries after repair")
        else:
            print(f"Repaired all {n_invalid} {name} geometries")
    else:
        print(f"All {original_count} {name} geometries are valid")
    
    return gdf

def align_coco_datasets_by_name(
    truth_coco: COCO,
    preds_coco: COCO,
    alignment_strategy: AlignmentStrategy = (
        AlignmentStrategy.EXACT_MATCH),
    return_report: bool = False,
    min_match_rate_warning: float = 0.0,
    min_match_rate_error: float = 0.0,
    verbose: bool = False,
) -> Optional[AlignmentReport]:
    if alignment_strategy == AlignmentStrategy.BASE_RSPLIT_1:
        def get_match_key(file_name: str) -> str:
            stem = Path(file_name).stem
            parts = stem.rsplit('_', 1)
            return parts[0] if len(parts) > 1 else stem
    else:
        def get_match_key(file_name: str) -> str:
            return file_name

    gt_images = truth_coco.dataset.get('images', [])
    pred_images = preds_coco.dataset.get('images', [])

    preds_by_key = {
        get_match_key(img['file_name']): img
        for img in pred_images
    }
    id_mapping = {}
    new_preds_images = []
    unmatched_gt = []

    for truth_img in gt_images:
        truth_file = truth_img['file_name']
        truth_key = get_match_key(truth_file)
        truth_id = truth_img['id']

        if truth_key in preds_by_key:
            preds_img = preds_by_key[truth_key]
            id_mapping[preds_img['id']] = truth_id
            new_img = preds_img.copy()
            new_img['id'] = truth_id
            new_preds_images.append(new_img)
        else:
            new_preds_images.append(truth_img.copy())
            unmatched_gt.append(truth_file)

    preds_coco.dataset['images'] = new_preds_images

    new_preds_annotations = []
    for ann in preds_coco.dataset.get('annotations', []):
        orig_img_id = ann['image_id']
        if orig_img_id in id_mapping:
            ann['image_id'] = id_mapping[orig_img_id]
            new_preds_annotations.append(ann)
    preds_coco.dataset['annotations'] = new_preds_annotations
    preds_coco.createIndex()

    # --- match-rate validation (opt-in) -----------------------
    num_gt = len(gt_images)
    num_pred = len(pred_images)
    num_matched = len(id_mapping)
    match_rate = (
        num_matched / num_gt if num_gt > 0 else 0.0)

    if (min_match_rate_error > 0
            and match_rate < min_match_rate_error):
        raise AlignmentError(
            "Critical alignment failure: only {:.1%} of "
            "GT tiles matched predictions "
            "(threshold: {:.1%}).".format(
                match_rate, min_match_rate_error))

    if (min_match_rate_warning > 0
            and match_rate < min_match_rate_warning):
        warnings.warn(
            "Low alignment rate: {:.1%} of GT tiles "
            "matched (threshold: {:.1%}). "
            "{} GT tiles have no predictions.".format(
                match_rate, min_match_rate_warning,
                len(unmatched_gt)),
            LowMatchRateWarning,
        )

    if not return_report:
        return None

    num_gt_without_pred = len(unmatched_gt)
    num_pred_outside_gt = num_pred - num_matched

    report = AlignmentReport(
        num_gt_images=num_gt,
        num_pred_images=num_pred,
        num_matched=num_matched,
        num_gt_without_pred=num_gt_without_pred,
        num_pred_outside_gt=num_pred_outside_gt,
        match_rate=match_rate,
        unmatched_gt_files=unmatched_gt,
        gt_filename_samples=[
            img['file_name'] for img in gt_images[:3]],
        pred_filename_samples=[
            img['file_name'] for img in pred_images[:3]],
        strategy_used=alignment_strategy.value,
    )

    if verbose:
        report.print_report(verbose=True)

    return report

def filter_min_overlap(gdf, aoi_geom, min_frac=0.4):
    from shapely.validation import make_valid
    # Make geometries valid to avoid TopologyException during intersection
    gdf = gdf.copy()
    gdf['geometry'] = gdf.geometry.apply(lambda g: make_valid(g) if g is not None and not g.is_valid else g)
    aoi_geom = make_valid(aoi_geom) if not aoi_geom.is_valid else aoi_geom

    orig_areas = gdf.geometry.area
    inter_areas = gdf.geometry.intersection(aoi_geom).area
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = inter_areas.div(orig_areas.replace({0: np.nan}))
    mask = frac >= min_frac
    return gdf[mask.fillna(False)].copy()


class Summarize2COCOEval(COCOeval_faster):
    def summarize_custom(self):
        max_dets_index = len(self.params.maxDets) - 1

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                s = self.eval['precision']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            mean_s = np.mean(s[s > -1]) if len(s[s > -1]) > 0 else -1
            stat_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
            print(stat_string)
            return mean_s, stat_string

        def _summarizeDets():
            area_labels = set(self.params.areaRngLbl)
            use_extended_sizes = all(label in area_labels for label in ['tiny', 'giant'])

            stats: list[float] = []
            stats_strings: list[str] = []

            def push(ap=1, iouThr=None, areaRng='all', maxDets=100):
                value, stat_string = _summarize(ap, iouThr=iouThr, areaRng=areaRng, maxDets=maxDets)
                stats.append(value)
                stats_strings.append(stat_string)

            # AP metrics
            push(1, maxDets=self.params.maxDets[max_dets_index])
            push(1, iouThr=.5, maxDets=self.params.maxDets[max_dets_index])
            push(1, iouThr=.75, maxDets=self.params.maxDets[max_dets_index])

            if use_extended_sizes:
                push(1, areaRng='tiny', maxDets=self.params.maxDets[max_dets_index])
            push(1, areaRng='small', maxDets=self.params.maxDets[max_dets_index])
            push(1, areaRng='medium', maxDets=self.params.maxDets[max_dets_index])
            push(1, areaRng='large', maxDets=self.params.maxDets[max_dets_index])
            if use_extended_sizes:
                push(1, areaRng='giant', maxDets=self.params.maxDets[max_dets_index])

            # AR metrics
            push(0, maxDets=self.params.maxDets[0])
            push(0, maxDets=self.params.maxDets[1])
            push(0, maxDets=self.params.maxDets[2])
            if len(self.params.maxDets) > 3:
                push(0, maxDets=self.params.maxDets[3])

            push(0, iouThr=.5, maxDets=self.params.maxDets[max_dets_index])
            push(0, iouThr=.75, maxDets=self.params.maxDets[max_dets_index])

            if use_extended_sizes:
                push(0, areaRng='tiny', maxDets=self.params.maxDets[max_dets_index])
            push(0, areaRng='small', maxDets=self.params.maxDets[max_dets_index])
            push(0, areaRng='medium', maxDets=self.params.maxDets[max_dets_index])
            push(0, areaRng='large', maxDets=self.params.maxDets[max_dets_index])
            if use_extended_sizes:
                push(0, areaRng='giant', maxDets=self.params.maxDets[max_dets_index])

            return np.array(stats), stats_strings

        def _summarizeKps():
            stats = np.zeros((10,))
            stats_strings = ['' for _ in range(10)]
            stats[0], stats_strings[0] = _summarize(1, maxDets=20)
            stats[1], stats_strings[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2], stats_strings[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3], stats_strings[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4], stats_strings[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5], stats_strings[5] = _summarize(0, maxDets=20)
            stats[6], stats_strings[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7], stats_strings[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8], stats_strings[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9], stats_strings[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats, stats_strings

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType in ['segm', 'bbox']:
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        else:
            raise Exception('Unknown iouType: {}'.format(iouType))
        self.stats, stats_strings = summarize()
        return stats_strings

    def summarize_to_dict(self):
        self.summarize_custom()
        stats = self.stats
        iou_type = self.params.iouType

        # Build metric names in the exact order metrics are pushed in summarize_custom()
        if iou_type == 'keypoints':
            metric_names = [
                "AP", "AP50", "AP75", "AP_medium", "AP_large",
                "AR", "AR50", "AR75", "AR_medium", "AR_large",
            ]
            ar_names: list[str] = ["AR"]
        else:
            area_labels = set(self.params.areaRngLbl)
            use_extended_sizes = all(label in area_labels for label in ['tiny', 'giant'])

            metric_names = ["AP", "AP50", "AP75"]
            if use_extended_sizes:
                metric_names.append("AP_tiny")
            metric_names.extend(["AP_small", "AP_medium", "AP_large"])
            if use_extended_sizes:
                metric_names.append("AP_giant")

            ar_names: list[str] = []
            for max_det in self.params.maxDets[:3]:
                ar_names.append(f"AR_{max_det}")
            if len(self.params.maxDets) > 3:
                ar_names.append(f"AR_{self.params.maxDets[3]}")
            metric_names.extend(ar_names)

            metric_names.extend(["AR50", "AR75"])

            if use_extended_sizes:
                metric_names.append("AR_tiny")
            metric_names.extend(["AR_small", "AR_medium", "AR_large"])
            if use_extended_sizes:
                metric_names.append("AR_giant")

        metrics_dict = {name: float(value) for name, value in zip(metric_names, stats)}

        # Provide a plain 'AR' alias (largest maxDets) for callers expecting it.
        if iou_type != 'keypoints' and ar_names:
            metrics_dict.setdefault("AR", metrics_dict[ar_names[-1]])

        return metrics_dict
