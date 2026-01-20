import warnings

import numpy as np
import geopandas as gpd
import cv2
from scipy.spatial.distance import directed_hausdorff, cdist
from shapely.affinity import affine_transform
from shapely import make_valid
from rasterio import features
from rasterio.transform import from_bounds
from geodataset.utils import get_utm_crs

from faster_coco_eval.core.coco import COCO
from faster_coco_eval.core.faster_eval_api import COCOeval_faster

from concurrent.futures import ThreadPoolExecutor, as_completed

# HELPER FUNCTIONS 
def compute_boundary_iou_and_f1(
    pred_mask: np.ndarray,
    truth_mask: np.ndarray,
    boundary_width: int = 2
) -> dict:
    """
    Compute Boundary IoU and Boundary F1 for a single instance pair.
    
    Boundary IoU: IoU computed only in a narrow band around the mask contours.
    Boundary F1: F1 score over boundary pixels.
    
    Args:
        pred_mask: Binary prediction mask
        truth_mask: Binary ground truth mask
        boundary_width: Width of boundary band in pixels
        
    Returns:
        Dict with 'boundary_iou' and 'boundary_f1'
    """
    # Extract boundaries
    pred_boundary = extract_boundary_band(pred_mask, width=boundary_width)
    truth_boundary = extract_boundary_band(truth_mask, width=boundary_width)
    
    # Boundary IoU: IoU within the boundary band
    boundary_union = np.logical_or(pred_boundary, truth_boundary)
    if boundary_union.sum() == 0:
        return {'boundary_iou': 1.0, 'boundary_f1': 1.0}
    
    # IoU within boundary region
    pred_in_boundary = np.logical_and(pred_mask, boundary_union)
    truth_in_boundary = np.logical_and(truth_mask, boundary_union)
    boundary_intersection = np.logical_and(pred_in_boundary, truth_in_boundary).sum()
    boundary_union_area = np.logical_or(pred_in_boundary, truth_in_boundary).sum()
    boundary_iou = boundary_intersection / boundary_union_area if boundary_union_area > 0 else 0.0
    
    # Boundary F1: F1 over boundary pixels
    pred_boundary_pixels = extract_boundary(pred_mask)
    truth_boundary_pixels = extract_boundary(truth_mask)
    
    # Dilate truth boundary for tolerance
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    truth_dilated = cv2.dilate(truth_boundary_pixels.astype(np.uint8), kernel, iterations=1)
    pred_dilated = cv2.dilate(pred_boundary_pixels.astype(np.uint8), kernel, iterations=1)
    
    tp = np.logical_and(pred_boundary_pixels, truth_dilated).sum()
    fp = pred_boundary_pixels.sum() - tp
    fn = truth_boundary_pixels.sum() - np.logical_and(truth_boundary_pixels, pred_dilated).sum()
    
    boundary_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    boundary_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall) if (boundary_precision + boundary_recall) > 0 else 0.0
    
    return {
        'boundary_iou': float(boundary_iou),
        'boundary_f1': float(boundary_f1)
    }


def aggregate_dice_miou_metrics(
    all_iou_scores: list[float],
    all_dice_scores: list[float],
    size_stratified: dict,
    iou_threshold: float,
) -> dict:
    """Aggregate global + size-wise Dice / mIoU stats with IoU threshold suffix."""

    metrics: dict[str, float | int] = {}
    
    # Create suffix based on IoU threshold
    iou_suffix = f"_{int(iou_threshold * 100)}"  # e.g., "_50" for 0.5, "_75" for 0.75

    if all_iou_scores:
        ious = np.array(all_iou_scores, dtype=float)
        dices = np.array(all_dice_scores, dtype=float)

        metrics.update({
            f'instance_mIoU_mean{iou_suffix}': float(ious.mean()),
            f'instance_mIoU_median{iou_suffix}': float(np.median(ious)),
            f'instance_mIoU_std{iou_suffix}': float(ious.std()),
            f'instance_mIoU_p25{iou_suffix}': float(np.percentile(ious, 25)),
            f'instance_mIoU_p75{iou_suffix}': float(np.percentile(ious, 75)),
            f'instance_Dice_mean{iou_suffix}': float(dices.mean()),
            f'instance_Dice_median{iou_suffix}': float(np.median(dices)),
            f'instance_Dice_std{iou_suffix}': float(dices.std()),
            f'instance_Dice_p25{iou_suffix}': float(np.percentile(dices, 25)),
            f'instance_Dice_p75{iou_suffix}': float(np.percentile(dices, 75)),
        })
    else:
        metrics.update({
            f'instance_mIoU_mean{iou_suffix}': 0.0,
            f'instance_mIoU_median{iou_suffix}': 0.0,
            f'instance_mIoU_std{iou_suffix}': 0.0,
            f'instance_mIoU_p25{iou_suffix}': 0.0,
            f'instance_mIoU_p75{iou_suffix}': 0.0,
            f'instance_Dice_mean{iou_suffix}': 0.0,
            f'instance_Dice_median{iou_suffix}': 0.0,
            f'instance_Dice_std{iou_suffix}': 0.0,
            f'instance_Dice_p25{iou_suffix}': 0.0,
            f'instance_Dice_p75{iou_suffix}': 0.0,
        })

    # Size-wise averages (only over matched instances)
    for size_cat in ['tiny', 'small', 'medium', 'large', 'giant']:
        cat = size_stratified[size_cat]
        if cat['iou']:
            metrics[f'instance_mIoU_{size_cat}{iou_suffix}'] = float(np.mean(cat['iou']))
        else:
            metrics[f'instance_mIoU_{size_cat}{iou_suffix}'] = 0.0

        if cat['dice']:
            metrics[f'instance_Dice_{size_cat}{iou_suffix}'] = float(np.mean(cat['dice']))
            metrics[f'num_matched_{size_cat}{iou_suffix}'] = int(len(cat['dice']))
        else:
            metrics[f'instance_Dice_{size_cat}{iou_suffix}'] = 0.0
            metrics[f'num_matched_{size_cat}{iou_suffix}'] = 0

    return metrics


def aggregate_pq_metrics(
    total_tp: int,
    total_fp: int,
    total_fn: int,
    all_iou_scores: list[float],
    size_stratified: dict,
    iou_threshold: float,
) -> dict:
    """Aggregate PQ / SQ / RQ globally and per size with IoU threshold suffix."""

    metrics: dict[str, float | int] = {}
    
    # Create suffix based on IoU threshold
    iou_suffix = f"_{int(iou_threshold * 100)}"  # e.g., "_50" for 0.5, "_75" for 0.75

    if all_iou_scores:
        ious = np.array(all_iou_scores, dtype=float)
        SQ = float(ious.mean())
    else:
        SQ = 0.0

    denom_rq = total_tp + 0.5 * total_fp + 0.5 * total_fn
    RQ = float(total_tp / denom_rq) if denom_rq > 0 else 0.0
    PQ = SQ * RQ

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    det_f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    metrics.update({
        f'PQ{iou_suffix}': float(PQ),
        f'SQ{iou_suffix}': float(SQ),
        f'RQ{iou_suffix}': float(RQ),
        f'detection_precision{iou_suffix}': float(precision),
        f'detection_recall{iou_suffix}': float(recall),
        f'detection_f1{iou_suffix}': float(det_f1),
        f'mean_instance_IoU{iou_suffix}': float(SQ),
    })

    # Size-stratified PQ / SQ / RQ
    for size_cat in ['tiny', 'small', 'medium', 'large', 'giant']:
        cat = size_stratified[size_cat]
        tp_c, fp_c, fn_c = cat['tp'], cat['fp'], cat['fn']

        if cat['iou']:
            SQ_c = float(np.mean(cat['iou']))
        else:
            SQ_c = 0.0

        denom_c = tp_c + 0.5 * fp_c + 0.5 * fn_c
        RQ_c = float(tp_c / denom_c) if denom_c > 0 else 0.0
        PQ_c = SQ_c * RQ_c

        metrics[f'PQ_{size_cat}{iou_suffix}'] = float(PQ_c)
        metrics[f'SQ_{size_cat}{iou_suffix}'] = float(SQ_c)
        metrics[f'RQ_{size_cat}{iou_suffix}'] = float(RQ_c)
        metrics[f'num_TP_{size_cat}{iou_suffix}'] = int(tp_c)
        metrics[f'num_FP_{size_cat}{iou_suffix}'] = int(fp_c)
        metrics[f'num_FN_{size_cat}{iou_suffix}'] = int(fn_c)

    return metrics


def aggregate_boundary_metrics(
    all_boundary_iou: list[float],
    all_boundary_f1: list[float],
    size_stratified: dict,
    iou_threshold: float,
) -> dict:
    """Aggregate boundary IoU / F1 globally and per size with IoU threshold suffix."""

    metrics: dict[str, float | int] = {}
    
    # Create suffix based on IoU threshold
    iou_suffix = f"_{int(iou_threshold * 100)}"

    if all_boundary_iou:
        b_iou = np.array(all_boundary_iou, dtype=float)
        b_f1 = np.array(all_boundary_f1, dtype=float)
        metrics.update({
            f'mean_boundary_IoU{iou_suffix}': float(b_iou.mean()),
            f'std_boundary_IoU{iou_suffix}': float(b_iou.std()),
            f'mean_boundary_F1{iou_suffix}': float(b_f1.mean()),
            f'std_boundary_F1{iou_suffix}': float(b_f1.std()),
        })
    else:
        metrics.update({
            f'mean_boundary_IoU{iou_suffix}': 0.0,
            f'std_boundary_IoU{iou_suffix}': 0.0,
            f'mean_boundary_F1{iou_suffix}': 0.0,
            f'std_boundary_F1{iou_suffix}': 0.0,
        })

    for size_cat in ['tiny', 'small', 'medium', 'large', 'giant']:
        cat = size_stratified[size_cat]
        if cat['boundary_iou']:
            metrics[f'boundary_IoU_{size_cat}{iou_suffix}'] = float(np.mean(cat['boundary_iou']))
        else:
            metrics[f'boundary_IoU_{size_cat}{iou_suffix}'] = 0.0

    return metrics


def compute_instance_quality_metrics(
    truth_coco: COCO,
    preds_coco: COCO,
    iou_threshold: float = 0.5,
    compute_dice_miou: bool = True,
    compute_pq: bool = True,
    compute_boundaries: bool = False,
    boundary_width: int = 2,
    area_ranges: dict | None = None,
    num_workers: int = 8,
    verbose: bool = True,
) -> dict:
    """
    Unified instance-level quality metrics for COCO-format datasets.

    - Runs a single greedy matching loop per image (tile)
    - Parallelized across images with ThreadPoolExecutor
    - Optionally computes:
        * Dice / mIoU for matched instances
        * PQ / SQ / RQ (global + size-stratified)
        * Boundary IoU / Boundary F1
    """
    if verbose:
        print("\n" + "=" * 80)
        print("INSTANCE QUALITY METRICS (UNIFIED, MULTI-THREADED)")
        print("=" * 80)
        print(f"  → IoU threshold: {iou_threshold}")
        print(f"  → compute_dice_miou = {compute_dice_miou}")
        print(f"  → compute_pq        = {compute_pq}")
        print(f"  → compute_boundaries= {compute_boundaries}")
        print(f"  → num_workers       = {num_workers}")

    img_ids = truth_coco.getImgIds()
    num_images = len(img_ids)
    if verbose:
        print(f"\n  → Processing {num_images} images...")

    # -------------------------------------------------------------------------
    # PER-IMAGE WORKER
    # -------------------------------------------------------------------------
    def process_single_image(img_id: int) -> dict:
        """
        Runs the greedy matching for a single image and returns partial stats.
        """
        img_info = truth_coco.loadImgs(img_id)[0]
        file_name = img_info.get('file_name', f'image_{img_id}')

        truth_ann_ids = truth_coco.getAnnIds(imgIds=img_id)
        pred_ann_ids = preds_coco.getAnnIds(imgIds=img_id)

        truth_anns = truth_coco.loadAnns(truth_ann_ids)
        pred_anns = preds_coco.loadAnns(pred_ann_ids)

        if verbose:
            print(f"  [img_id={img_id}] {file_name} → GT={len(truth_anns)}, Pred={len(pred_anns)}")

        # Per-image accumulators
        local_tp = 0
        local_fp = 0
        local_fn = 0

        local_iou_scores: list[float] = []
        local_dice_scores: list[float] = []
        local_boundary_iou: list[float] = []
        local_boundary_f1: list[float] = []

        local_size = {
            'tiny': {
                'tp': 0, 'fp': 0, 'fn': 0,
                'iou': [], 'dice': [],
                'boundary_iou': [], 'boundary_f1': []
            },
            'small': {
                'tp': 0, 'fp': 0, 'fn': 0,
                'iou': [], 'dice': [],
                'boundary_iou': [], 'boundary_f1': []
            },
            'medium': {
                'tp': 0, 'fp': 0, 'fn': 0,
                'iou': [], 'dice': [],
                'boundary_iou': [], 'boundary_f1': []
            },
            'large': {
                'tp': 0, 'fp': 0, 'fn': 0,
                'iou': [], 'dice': [],
                'boundary_iou': [], 'boundary_f1': []
            },
            'giant': {
                'tp': 0, 'fp': 0, 'fn': 0,
                'iou': [], 'dice': [],
                'boundary_iou': [], 'boundary_f1': []
            },
        }


        # Case: no GT → all preds are FP
        if len(truth_anns) == 0:
            local_fp = len(pred_anns)
            if compute_pq:
                for pred_ann in pred_anns:
                    pred_area = pred_ann.get('area', 0.0)
                    size_cat = _get_size_category(pred_area, area_ranges)
                    local_size[size_cat]['fp'] += 1

            return {
                'tp': local_tp,
                'fp': local_fp,
                'fn': local_fn,
                'all_iou': local_iou_scores,
                'all_dice': local_dice_scores,
                'all_boundary_iou': local_boundary_iou,
                'all_boundary_f1': local_boundary_f1,
                'size_stratified': local_size,
            }

        # Case: no preds → all GT are FN
        if len(pred_anns) == 0:
            local_fn = len(truth_anns)
            if compute_pq:
                for truth_ann in truth_anns:
                    gt_area = truth_ann.get('area', 0.0)
                    size_cat = _get_size_category(gt_area, area_ranges)
                    local_size[size_cat]['fn'] += 1

            return {
                'tp': local_tp,
                'fp': local_fp,
                'fn': local_fn,
                'all_iou': local_iou_scores,
                'all_dice': local_dice_scores,
                'all_boundary_iou': local_boundary_iou,
                'all_boundary_f1': local_boundary_f1,
                'size_stratified': local_size,
            }

        # Precompute masks & basic stats ONCE per image
        truth_masks = [truth_coco.annToMask(ann) for ann in truth_anns]
        truth_sums = [m.sum() for m in truth_masks]

        # Sort predictions by descending score
        pred_anns = sorted(pred_anns, key=lambda x: x.get('score', 0), reverse=True)
        pred_masks = [preds_coco.annToMask(ann) for ann in pred_anns]

        matched_truth_indices = set()

        # Greedy matching: each pred picks best unmatched GT by IoU
        for pred_ann, pred_mask in zip(pred_anns, pred_masks):
            best_iou = 0.0
            best_truth_idx = None
            best_truth_ann = None
            best_truth_mask = None
            best_intersection = 0
            best_truth_sum = 0

            # Loop over GTs
            for truth_idx, (truth_ann, truth_mask, truth_sum) in enumerate(
                zip(truth_anns, truth_masks, truth_sums)
            ):
                if truth_idx in matched_truth_indices:
                    continue

                # IoU in mask space
                intersection = np.logical_and(pred_mask, truth_mask).sum()
                if intersection == 0:
                    continue
                union = np.logical_or(pred_mask, truth_mask).sum()
                if union == 0:
                    continue
                iou = intersection / union

                if iou > best_iou:
                    best_iou = iou
                    best_truth_idx = truth_idx
                    best_truth_ann = truth_ann
                    best_truth_mask = truth_mask
                    best_intersection = intersection
                    best_truth_sum = truth_sum

            if best_iou >= iou_threshold and best_truth_idx is not None:
                # TRUE POSITIVE
                matched_truth_indices.add(best_truth_idx)
                local_tp += 1

                instance_iou = best_iou
                local_iou_scores.append(instance_iou)

                gt_area = best_truth_ann.get('area', float(best_truth_sum))
                size_cat = _get_size_category(gt_area, area_ranges)

                if compute_pq:
                    local_size[size_cat]['tp'] += 1
                    local_size[size_cat]['iou'].append(instance_iou)

                # Dice / mIoU per instance
                if compute_dice_miou:
                    pred_sum = pred_mask.sum()
                    dice = (
                        2 * best_intersection / (pred_sum + best_truth_sum)
                        if (pred_sum + best_truth_sum) > 0 else 0.0
                    )
                    local_dice_scores.append(dice)
                    local_size[size_cat]['dice'].append(dice)

                # Boundary metrics per instance
                if compute_boundaries:
                    boundary_metrics = compute_boundary_iou_and_f1(
                        pred_mask, best_truth_mask, boundary_width
                    )
                    b_iou = boundary_metrics['boundary_iou']
                    b_f1 = boundary_metrics['boundary_f1']
                    local_boundary_iou.append(b_iou)
                    local_boundary_f1.append(b_f1)
                    local_size[size_cat]['boundary_iou'].append(b_iou)
                    local_size[size_cat]['boundary_f1'].append(b_f1)

            else:
                # FALSE POSITIVE
                local_fp += 1
                if compute_pq:
                    pred_area = pred_ann.get('area', float(pred_mask.sum()))
                    size_cat = _get_size_category(pred_area, area_ranges)
                    local_size[size_cat]['fp'] += 1

        # Unmatched GT → FN
        num_unmatched = len(truth_anns) - len(matched_truth_indices)
        local_fn += num_unmatched
        if compute_pq and num_unmatched > 0:
            for truth_idx, truth_ann in enumerate(truth_anns):
                if truth_idx not in matched_truth_indices:
                    gt_area = truth_ann.get('area', 0.0)
                    size_cat = _get_size_category(gt_area, area_ranges)
                    local_size[size_cat]['fn'] += 1

        return {
            'tp': local_tp,
            'fp': local_fp,
            'fn': local_fn,
            'all_iou': local_iou_scores,
            'all_dice': local_dice_scores,
            'all_boundary_iou': local_boundary_iou,
            'all_boundary_f1': local_boundary_f1,
            'size_stratified': local_size,
        }

    # -------------------------------------------------------------------------
    # RUN PER-IMAGE IN PARALLEL
    # -------------------------------------------------------------------------
    total_tp = 0
    total_fp = 0
    total_fn = 0

    all_iou_scores: list[float] = []
    all_dice_scores: list[float] = []
    all_boundary_iou: list[float] = []
    all_boundary_f1: list[float] = []

    # global size-stratified accumulator
    size_stratified_global = {
        'tiny': {
            'tp': 0, 'fp': 0, 'fn': 0,
            'iou': [], 'dice': [],
            'boundary_iou': [], 'boundary_f1': []
        },
        'small': {
            'tp': 0, 'fp': 0, 'fn': 0,
            'iou': [], 'dice': [],
            'boundary_iou': [], 'boundary_f1': []
        },
        'medium': {
            'tp': 0, 'fp': 0, 'fn': 0,
            'iou': [], 'dice': [],
            'boundary_iou': [], 'boundary_f1': []
        },
        'large': {
            'tp': 0, 'fp': 0, 'fn': 0,
            'iou': [], 'dice': [],
            'boundary_iou': [], 'boundary_f1': []
        },
        'giant': {
            'tp': 0, 'fp': 0, 'fn': 0,
            'iou': [], 'dice': [],
            'boundary_iou': [], 'boundary_f1': []
        },
    }

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(process_single_image, img_id): img_id for img_id in img_ids}
        for i, fut in enumerate(as_completed(futures), start=1):
            img_id = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"[ERROR] while processing img_id={img_id}: {e}")
                continue

            total_tp += res['tp']
            total_fp += res['fp']
            total_fn += res['fn']

            all_iou_scores.extend(res['all_iou'])
            all_dice_scores.extend(res['all_dice'])
            all_boundary_iou.extend(res['all_boundary_iou'])
            all_boundary_f1.extend(res['all_boundary_f1'])

            # merge size-stratified
            for size_cat in ['tiny', 'small', 'medium', 'large', 'giant']:
                src = res['size_stratified'][size_cat]
                dst = size_stratified_global[size_cat]
                dst['tp'] += src['tp']
                dst['fp'] += src['fp']
                dst['fn'] += src['fn']
                dst['iou'].extend(src['iou'])
                dst['dice'].extend(src['dice'])
                dst['boundary_iou'].extend(src['boundary_iou'])
                dst['boundary_f1'].extend(src['boundary_f1'])

            if verbose and (i % 10 == 0 or i == num_images):
                print(f"  → Progress: {i}/{num_images} images aggregated")

    # -------------------------------------------------------------------------
    # BASE METRICS (TP/FP/FN, precision/recall)
    # -------------------------------------------------------------------------
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    
    iou_suffix = f"_{int(iou_threshold * 100)}"
    
    metrics: dict[str, float | int] = {
        f'num_matched_instances{iou_suffix}': int(total_tp),
        f'num_false_positives{iou_suffix}': int(total_fp),
        f'num_false_negatives{iou_suffix}': int(total_fn),
        f'detection_precision{iou_suffix}': float(precision),
        f'detection_recall{iou_suffix}': float(recall),
        'iou_threshold': float(iou_threshold),  # Keep this without suffix for reference
    }

     # ---- Dice / mIoU ----
    if compute_dice_miou:
        metrics.update(aggregate_dice_miou_metrics(
            all_iou_scores=all_iou_scores,
            all_dice_scores=all_dice_scores,
            size_stratified=size_stratified_global,
            iou_threshold=iou_threshold,
        ))

    # ---- PQ / SQ / RQ ----
    if compute_pq:
        metrics.update(aggregate_pq_metrics(
            total_tp=total_tp,
            total_fp=total_fp,
            total_fn=total_fn,
            all_iou_scores=all_iou_scores,
            size_stratified=size_stratified_global,
            iou_threshold=iou_threshold,
        ))

    # ---- Boundary metrics ----
    if compute_boundaries:
        metrics.update(aggregate_boundary_metrics(
            all_boundary_iou=all_boundary_iou,
            all_boundary_f1=all_boundary_f1,
            size_stratified=size_stratified_global,
            iou_threshold=iou_threshold,
        ))

    if verbose:
        print("\n[instance_quality] Done.")
        print(f"  → TP={total_tp}, FP={total_fp}, FN={total_fn}")
        print(f"  → detection_precision={metrics[f'detection_precision{iou_suffix}']:.4f}, "
              f"recall={metrics[f'detection_recall{iou_suffix}']:.4f}")

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
    
    print("Validating geometries after coordinate transformation...")
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

def align_coco_datasets_by_name(truth_coco: COCO, preds_coco: COCO) -> None:
    """
    Align the predictions COCO dataset to follow the order of the truth COCO dataset,
    matching based on the image 'file_name'. For any truth image missing in preds,
    insert a dummy image (with no annotations) so that the image IDs match.
    This function updates the preds_coco in-place.
    """
    preds_by_name = {img['file_name']: img for img in preds_coco.dataset.get('images', [])}
    id_mapping = {}
    new_preds_images = []
    for truth_img in truth_coco.dataset.get('images', []):
        file_name = truth_img['file_name']
        truth_id = truth_img['id']
        if file_name in preds_by_name:
            preds_img = preds_by_name[file_name]
            id_mapping[preds_img['id']] = truth_id
            new_img = preds_img.copy()
            new_img['id'] = truth_id
            new_preds_images.append(new_img)
        else:
            new_preds_images.append(truth_img.copy())
    preds_coco.dataset['images'] = new_preds_images

    new_preds_annotations = []
    for ann in preds_coco.dataset.get('annotations', []):
        orig_img_id = ann['image_id']
        if orig_img_id in id_mapping:
            ann['image_id'] = id_mapping[orig_img_id]
            new_preds_annotations.append(ann)
    preds_coco.dataset['annotations'] = new_preds_annotations
    preds_coco.createIndex()


def gdf_to_coco_single_image(gdf: gpd.GeoDataFrame, width: int, height: int, is_ground_truth: bool):
    geometries = gdf['geometry'].tolist()
    scores = gdf['aggregator_score'].tolist() if not is_ground_truth else None
    categories = [1] * len(geometries)
    image_id = 1
    annotations = []

    for i, geometry in enumerate(geometries):
        if geometry.is_empty or not geometry.is_valid:
            continue

        if geometry.geom_type == "Polygon":
            segmentation = [np.array(geometry.exterior.coords).flatten().tolist()]
        elif geometry.geom_type == "MultiPolygon":
            segmentation = []
            for polygon in geometry.geoms:
                segmentation.append(np.array(polygon.exterior.coords).flatten().tolist())
        else:
            raise ValueError(f"Unsupported geometry type: {geometry.geom_type}")

        annotation = {
            'id': i + 1,
            'image_id': image_id,
            'category_id': categories[i],
            'segmentation': segmentation,
            'area': geometry.area,
            'bbox': list(geometry.bounds),
            'iscrowd': 0
        }

        if not is_ground_truth:
            annotation['score'] = scores[i]

        annotations.append(annotation)

    coco = {
        'images': [{
            'id': image_id,
            'file_name': 'dummy_raster_name.tif',
            'width': width,
            'height': height
        }],
        'annotations': annotations,
        'categories': [{
            'id': 1, 'name': 'object'
        }]
    }
    return coco


def filter_min_overlap(gdf, aoi_geom, min_frac=0.4):
    orig_areas = gdf.geometry.area
    inter_areas = gdf.geometry.intersection(aoi_geom).area
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = inter_areas.div(orig_areas.replace({0: np.nan}))
    mask = frac >= min_frac
    return gdf[mask.fillna(False)].copy()

def extract_boundary(mask: np.ndarray) -> np.ndarray:
    """Extract boundary pixels from a binary mask using erosion."""
    if mask.sum() == 0:
        return mask
    
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    boundary = mask.astype(np.uint8) - eroded
    
    return boundary

def extract_boundary_band(mask: np.ndarray, width: int = 2) -> np.ndarray:
    """
    Extract a narrow band around the mask boundary.
    
    Args:
        mask: Binary mask
        width: Width of boundary band in pixels
        
    Returns:
        Binary mask with boundary band
    """
    if mask.sum() == 0:
        return mask
    
    # Dilate and erode to get boundary band
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*width+1, 2*width+1))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    boundary_band = dilated - eroded
    
    return boundary_band


def _get_size_category(area: float, area_ranges: dict) -> str:
    """✅ UPDATED: Helper to determine size category from area for 5 categories."""
    if area_ranges['tiny'][0] <= area < area_ranges['tiny'][1]:
        return 'tiny'
    elif area_ranges['small'][0] <= area < area_ranges['small'][1]:
        return 'small'
    elif area_ranges['medium'][0] <= area < area_ranges['medium'][1]:
        return 'medium'
    elif area_ranges['large'][0] <= area < area_ranges['large'][1]:
        return 'large'
    else:
        return 'giant'

# ===========================================================================
# COCO EVALUATOR CLASS
# ===========================================================================

class CocoEvaluator:
    small_max_sq_meters = 16
    #small_max_pixels = 1024
    small_max_pixels = 1940
   # small_max_pixels = 3824
    
    medium_max_sq_meters = 100
    #medium_max_pixels = 9216
    medium_max_pixels = 7383
    #medium_max_pixels = 10850
    
    DEFAULT_GROUND_RESOLUTION = 0.045 

    SIZE_THRESHOLDS_SQ_METERS = {
        'tiny': (0, 9),
        'small': (9, 25),
        'medium': (25, 49),
        'large': (49, 100),
        'giant': (100, float('inf'))
    }

    @classmethod
    def get_area_ranges_pixels(cls, ground_resolution: float) -> dict:
        """
        Convert square meter thresholds to pixel area based on ground resolution.
        
        Args:
            ground_resolution: Resolution in meters per pixel (e.g., 0.045 m/px)
        
        Returns:
            Dict with pixel area ranges for each size category
        """
        pixel_area_per_sq_meter = 1 / (ground_resolution ** 2)
        
        area_ranges = {}
        for category, (min_m2, max_m2) in cls.SIZE_THRESHOLDS_SQ_METERS.items():
            min_px = min_m2 * pixel_area_per_sq_meter
            max_px = max_m2 * pixel_area_per_sq_meter if max_m2 != float('inf') else 1e10
            area_ranges[category] = (min_px, max_px)
        
        return area_ranges
    
    @classmethod
    def get_size_category_labels(cls) -> list[str]:
        """Get list of size category labels."""
        return list(cls.SIZE_THRESHOLDS_SQ_METERS.keys())

    def tile_level(self,
               iou_type: str,
               preds_coco_path: str,
               truth_coco_path: str,
                ground_resolution: float = DEFAULT_GROUND_RESOLUTION,
               max_dets: list[int] = (1, 10, 100),
               compute_segmentation_quality: bool = False,
               compute_pq: bool = True) -> dict:
        """
        Evaluate predictions at tile level using COCO metrics.
        
        Args:
            iou_type: 'bbox' or 'segm'
            preds_coco_path: Path to predictions COCO JSON
            truth_coco_path: Path to ground truth COCO JSON
            max_dets: Max detections per image for evaluation
            compute_instance_segmentation_quality: If True, compute instance-wise mIoU/Dice
                                                conditioned on detection (IoU >= 0.5)
        
        Returns:
            Dictionary with COCO metrics and optional instance segmentation quality metrics
        """
        area_ranges = self.get_area_ranges_pixels(ground_resolution)

        truth_coco = COCO(str(truth_coco_path))
        preds_coco = COCO(str(preds_coco_path))

        for ann in truth_coco.dataset['annotations']:
            if 'score' in ann:
                del ann['score']

        align_coco_datasets_by_name(truth_coco, preds_coco)

        # Run COCO evaluation
        coco_evaluator = Summarize2COCOEval(
            cocoGt=truth_coco,
            cocoDt=preds_coco,
            iouType=iou_type
        )
        coco_evaluator.params.maxDets = max_dets


        area_rng_list = [
            [0, 1e10],  # all
            [area_ranges['tiny'][0], area_ranges['tiny'][1]],
            [area_ranges['small'][0], area_ranges['small'][1]],
            [area_ranges['medium'][0], area_ranges['medium'][1]],
            [area_ranges['large'][0], area_ranges['large'][1]],
            [area_ranges['giant'][0], area_ranges['giant'][1]]
        ]
        area_rng_lbl = ['all', 'tiny', 'small', 'medium', 'large', 'giant']
    
        coco_evaluator.params.areaRng = area_rng_list
        coco_evaluator.params.areaRngLbl = area_rng_lbl


        coco_evaluator.evaluate()
        coco_evaluator.accumulate()

        metrics = coco_evaluator.summarize_to_dict()
        num_images = len(truth_coco.dataset.get('images', []))
        num_truths = len(truth_coco.dataset.get('annotations', []))
        num_preds = len(preds_coco.dataset.get('annotations', []))

        metrics['num_images'] = num_images
        metrics['num_truths'] = num_truths
        metrics['num_preds'] = num_preds

        # Add F1 scores
        def safe_f1(ap, ar):
            if ap is None or ar is None or ap == -1 or ar == -1:
                return 0.0
            if (ap + ar) == 0:
                return 0.0
            return float(2 * ap * ar / (ap + ar))
        
        metrics['F1'] = safe_f1(metrics['AP'], metrics['AR'])
        metrics['F1_50'] = safe_f1(metrics['AP50'], metrics['AR50'])
        metrics['F1_75'] = safe_f1(metrics['AP75'], metrics['AR75'])
        for size_cat in ['tiny', 'small', 'medium', 'large', 'giant']:
            ap_key = f'AP_{size_cat}'
            ar_key = f'AR_{size_cat}'
            if ap_key in metrics and ar_key in metrics:
                metrics[f'F1_{size_cat}'] = safe_f1(metrics[ap_key], metrics[ar_key])
    
        # =========================================================================
        # INSTANCE SEGMENTATION QUALITY (conditioned on detection)
        # =========================================================================
        if compute_segmentation_quality and iou_type == 'segm':
            print("\n" + "="*80)
            print("Computing instance quality metrics (Dice/mIoU, PQ, boundaries)...")
            print("="*80)
            
            # Compute at IoU=0.5
            inst_metrics_50 = compute_instance_quality_metrics(
                truth_coco=truth_coco,
                preds_coco=preds_coco,
                iou_threshold=0.5,
                compute_dice_miou=True,
                compute_pq=True,
                compute_boundaries=True,
                area_ranges=area_ranges,
                verbose=True,
            )
            
            # Compute at IoU=0.75
            inst_metrics_75 = compute_instance_quality_metrics(
                truth_coco=truth_coco,
                preds_coco=preds_coco,
                iou_threshold=0.75,
                compute_dice_miou=True,
                compute_pq=True,
                compute_boundaries=True,
                area_ranges=area_ranges,
                verbose=True,
            )
            
            # Merge both into metrics dict
            metrics.update(inst_metrics_50)
            metrics.update(inst_metrics_75)

        return metrics
   
    def raster_level(self,
                     iou_type: str,
                     preds_gpkg_path: str,
                     truth_gpkg_path: str,
                     aoi_gpkg_path: str or None,
                     ground_resolution: float) -> dict:
        
        area_ranges = self.get_area_ranges_pixels(ground_resolution)

        truth_gdf = gpd.read_file(truth_gpkg_path)
        infer_gdf = gpd.read_file(preds_gpkg_path)

        # Only keep the truth and inference geometries that are inside the AOI (40% overlap minimum)
        if aoi_gpkg_path is not None:
            aoi_gdf = gpd.read_file(aoi_gpkg_path)
            common_crs = aoi_gdf.crs
            truth_gdf = truth_gdf.to_crs(common_crs)
            infer_gdf = infer_gdf.to_crs(common_crs)
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

        print(f"DEBUG: Before coordinate transform - Truth: {len(truth_gdf)}, Preds: {len(infer_gdf)}")
        truth_gdf, infer_gdf = move_gdfs_to_ground_resolution(truth_gdf, infer_gdf, ground_resolution)
        print(f"DEBUG: After coordinate transform - Truth: {len(truth_gdf)}, Preds: {len(infer_gdf)}")

        b1 = truth_gdf.total_bounds  # [minx, miny, maxx, maxy]
        b2 = infer_gdf.total_bounds
        combined_bounds = [
            min(b1[0], b2[0]),  # minx
            min(b1[1], b2[1]),  # miny
            max(b1[2], b2[2]),  # maxx
            max(b1[3], b2[3])  # maxy
        ]
        width = int((combined_bounds[2] - combined_bounds[0]))
        height = int((combined_bounds[3] - combined_bounds[1]))

        truth_coco = gdf_to_coco_single_image(
            gdf=truth_gdf,
            width=width,
            height=height,
            is_ground_truth=True
        )
        infer_coco = gdf_to_coco_single_image(
            gdf=infer_gdf,
            width=width,
            height=height,
            is_ground_truth=False
        )

        coco_gt_obj = COCO()
        coco_gt_obj.dataset = truth_coco
        coco_gt_obj.createIndex()

        coco_dt_obj = COCO()
        coco_dt_obj.dataset = infer_coco
        coco_dt_obj.createIndex()

        # Initialize and run COCOeval
        twice_max_dets = len(truth_gdf) * 2    # We consider up to twice the number of ground truth objects as predictions to be evaluated
        coco_evaluator = Summarize2COCOEval(
            cocoGt=coco_gt_obj,
            cocoDt=coco_dt_obj,
            iouType=iou_type
        )
        coco_evaluator.params.maxDets = [1, 10, 100, twice_max_dets]

        area_rng_list = [
            [0, 1e10],
            [area_ranges['tiny'][0], area_ranges['tiny'][1]],
            [area_ranges['small'][0], area_ranges['small'][1]],
            [area_ranges['medium'][0], area_ranges['medium'][1]],
            [area_ranges['large'][0], area_ranges['large'][1]],
            [area_ranges['giant'][0], area_ranges['giant'][1]]
        ]
        area_rng_lbl = ['all', 'tiny', 'small', 'medium', 'large', 'giant']

        coco_evaluator.params.areaRng = area_rng_list
        coco_evaluator.params.areaRngLbl = area_rng_lbl

        coco_evaluator.evaluate()
        coco_evaluator.accumulate()

        # Get metrics as a dictionary and save as JSON
        metrics = coco_evaluator.summarize_to_dict()
        num_images = len(coco_gt_obj.dataset.get('images', []))
        num_truths = len(coco_gt_obj.dataset.get('annotations', []))
        num_preds = len(coco_dt_obj.dataset.get('annotations', []))

        metrics['num_images'] = num_images
        metrics['num_truths'] = num_truths
        metrics['num_preds'] = num_preds

        # Adding some composite metrics with zero-division protection
        # Please note that these are neither standard COCO metrics, nor standard F1 scores as AP and AR are not direct analogs to precision and recall.
        def safe_f1(ap, ar):
            """Safely compute F1 score, returning 0 if either metric is 0 or invalid."""
            if ap is None or ar is None or ap == -1 or ar == -1:
                return 0.0
            if (ap + ar) == 0:
                return 0.0
            return float(2 * ap * ar / (ap + ar))
        
        metrics['F1'] = safe_f1(metrics['AP'], metrics['AR'])
        metrics['F1_50'] = safe_f1(metrics['AP50'], metrics['AR50'])
        metrics['F1_75'] = safe_f1(metrics['AP75'], metrics['AR75'])
        for size_cat in ['tiny', 'small', 'medium', 'large', 'giant']:
            ap_key = f'AP_{size_cat}'
            ar_key = f'AR_{size_cat}'
            if ap_key in metrics and ar_key in metrics:
                metrics[f'F1_{size_cat}'] = safe_f1(metrics[ap_key], metrics[ar_key])
        return metrics

    @staticmethod
    def raster_level_single_iou_threshold(iou_type: str,
                                          preds_gpkg_path: str,
                                          truth_gpkg_path: str,
                                          aoi_gpkg_path: str or None,
                                          ground_resolution: float = 0.045,
                                          iou_threshold: float = 0.5) -> dict:
        """
        Compute precision, recall, and F1 score at a given IoU threshold
        between prediction and ground-truth GeoDataFrames.

        iou_type: type of IoU to compute (e.g., 'bbox', 'segm').
        preds_gpkg_path: path to GeoDataFrame with a 'geometry' column and a 'score', 'aggregator_score', 'detector_score' or 'segmentation_score' column (will be checked in that order).
        truth_gpkg_path: path to GeoDataFrame with a 'geometry' column.
        aoi_gpkg_path: path to GeoDataFrame with a 'geometry' column (optional).
        iou_threshold: IoU threshold for a match (default: 0.5).
        """

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

        # Sort predictions by descending score
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

        # Add a matched flag to ground truths
        truth_gdf["matched"] = False
        truth_sindex = truth_gdf.sindex

        tp = 0  # True positives
        fp = 0  # False positives

        # Greedy match each prediction
        for _, pred in infer_gdf.iterrows():
            # Bounding-box preselection for potential matches
            candidates = list(truth_sindex.intersection(pred.geometry.bounds))
            best_iou = 0
            best_idx = None

            # Find best IoU among unmatched candidates
            for idx in candidates:
                if truth_gdf.at[idx, "matched"]:
                    continue
                truth_geom = truth_gdf.at[idx, "geometry"]
                inter_area = pred.geometry.intersection(truth_geom).area
                union_area = pred.geometry.union(truth_geom).area
                iou = inter_area / union_area if union_area > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            # Assign match or count as false positive
            if best_iou >= iou_threshold:
                tp += 1
                truth_gdf.at[best_idx, "matched"] = True
            else:
                fp += 1

        # False negatives: ground truths never matched
        fn = (~truth_gdf["matched"]).sum()

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics = {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn,
                   'num_truths': len(truth_gdf), 'num_preds': len(infer_gdf), 'num_images': 1}
        print(metrics)

        return metrics
    
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
            # ✅ HARDCODED FOR 5 SIZE CATEGORIES: tiny, small, medium, large, huge
            # Total metrics: 3 AP thresholds + 5 AP sizes + 4 AR maxDets + 2 AR IoU + 5 AR sizes = 19
            stats = np.zeros((19,))
            stats_strings = ['' for _ in range(19)]
            
            idx = 0

            # ===== AP METRICS =====
            # Overall AP metrics (3)
            stats[idx], stats_strings[idx] = _summarize(1, maxDets=self.params.maxDets[max_dets_index])
            idx += 1
            stats[idx], stats_strings[idx] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[max_dets_index])
            idx += 1
            stats[idx], stats_strings[idx] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[max_dets_index])
            idx += 1
            
            # AP by size (5)
            stats[idx], stats_strings[idx] = _summarize(1, areaRng='tiny', maxDets=self.params.maxDets[max_dets_index])
            idx += 1
            stats[idx], stats_strings[idx] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[max_dets_index])
            idx += 1
            stats[idx], stats_strings[idx] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[max_dets_index])
            idx += 1
            stats[idx], stats_strings[idx] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[max_dets_index])
            idx += 1
            stats[idx], stats_strings[idx] = _summarize(1, areaRng='giant', maxDets=self.params.maxDets[max_dets_index])
            idx += 1

            # ===== AR METRICS =====
            # AR at different max detections (4)
            stats[idx], stats_strings[idx] = _summarize(0, maxDets=self.params.maxDets[0])
            idx += 1
            stats[idx], stats_strings[idx] = _summarize(0, maxDets=self.params.maxDets[1])
            idx += 1
            stats[idx], stats_strings[idx] = _summarize(0, maxDets=self.params.maxDets[2])
            idx += 1
            stats[idx], stats_strings[idx] = (_summarize(0, maxDets=self.params.maxDets[3])
                                              if len(self.params.maxDets) > 3
                                              else _summarize(0, maxDets=self.params.maxDets[2]))
            idx += 1

            # AR at specific IoU thresholds (2)
            stats[idx], stats_strings[idx] = _summarize(0, iouThr=.5, maxDets=self.params.maxDets[max_dets_index])
            idx += 1
            stats[idx], stats_strings[idx] = _summarize(0, iouThr=.75, maxDets=self.params.maxDets[max_dets_index])
            idx += 1

            # AR by size (5)
            stats[idx], stats_strings[idx] = _summarize(0, areaRng='tiny', maxDets=self.params.maxDets[max_dets_index])
            idx += 1
            stats[idx], stats_strings[idx] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[max_dets_index])
            idx += 1
            stats[idx], stats_strings[idx] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[max_dets_index])
            idx += 1
            stats[idx], stats_strings[idx] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[max_dets_index])
            idx += 1
            stats[idx], stats_strings[idx] = _summarize(0, areaRng='giant', maxDets=self.params.maxDets[max_dets_index])
            idx += 1

            return stats, stats_strings

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
        """✅ HARDCODED FOR 5 SIZE CATEGORIES."""
        self.summarize_custom()
        stats = self.stats
        
        # Hardcoded metric names for 5 size categories
        metric_names = [
            "AP", "AP50", "AP75",
            "AP_tiny", "AP_small", "AP_medium", "AP_large", "AP_giant",
            "AR_1", "AR_10", "AR_100", "AR",
            "AR50", "AR75",
            "AR_tiny", "AR_small", "AR_medium", "AR_large", "AR_giant"
        ]
        
        metrics_dict = {name: float(value) for name, value in zip(metric_names, stats)}
        
        return metrics_dict