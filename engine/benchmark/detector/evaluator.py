import warnings

import numpy as np
import geopandas as gpd
from shapely import make_valid
from shapely.affinity import affine_transform
from geodataset.utils import get_utm_crs

from faster_coco_eval.core.coco import COCO
from faster_coco_eval.core.faster_eval_api import COCOeval_faster      # speeds up raster level evaluation by 10-100x


class CocoEvaluator:
    small_max_sq_meters = 16
    medium_max_sq_meters = 100

    def tile_level(self,
                   iou_type: str,
                   preds_coco_path: str,
                   truth_coco_path: str,
                   max_dets: list[int] = (1, 10, 100)) -> dict:

        truth_coco = COCO(str(truth_coco_path))
        preds_coco = COCO(str(preds_coco_path))

        for ann in truth_coco.dataset['annotations']:
            if 'score' in ann:
                del ann['score']  # avoids crash when score is None (truth shouldn't have scores)

        # Align predictions to truth based on file name
        align_coco_datasets_by_name(truth_coco, preds_coco)

        # Set up and run COCO evaluation
        coco_evaluator = Summarize2COCOEval(
            cocoGt=truth_coco,
            cocoDt=preds_coco,
            iouType=iou_type
        )
        coco_evaluator.params.maxDets = max_dets
        coco_evaluator.evaluate()
        coco_evaluator.accumulate()

        # Get metrics as a dictionary and add some debug info
        metrics = coco_evaluator.summarize_to_dict()
        num_images = len(truth_coco.dataset.get('images', []))
        num_truths = len(truth_coco.dataset.get('annotations', []))
        num_preds = len(preds_coco.dataset.get('annotations', []))

        metrics['num_images'] = num_images
        metrics['num_truths'] = num_truths
        metrics['num_preds'] = num_preds

        return metrics

    def raster_level(self,
                     iou_type: str,
                     preds_gpkg_path: str,
                     truth_gpkg_path: str,
                     aoi_gpkg_path: str or None,
                     ground_resolution: float) -> dict:

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

        truth_gdf, infer_gdf = move_gdfs_to_ground_resolution(truth_gdf, infer_gdf, ground_resolution)

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

        pixel_small = self.small_max_sq_meters / (ground_resolution ** 2)
        pixel_medium = self.medium_max_sq_meters / (ground_resolution ** 2)
        coco_evaluator.params.areaRng = [
            [0, 1e10],                    # all
            [0, pixel_small],             # small
            [pixel_small, pixel_medium],  # medium
            [pixel_medium, 1e10]           # large
        ]
        coco_evaluator.params.areaRngLbl = ['all', 'small', 'medium', 'large']

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

        # Adding some composite metrics.
        # Please note that these are neither standard COCO metrics, nor standard F1 scores as AP and AR are not direct analogs to precision and recall.
        metrics['F1'] = 2 * metrics['AP'] * metrics['AR'] / (metrics['AP'] + metrics['AR'])
        metrics['F1_50'] = 2 * metrics['AP50'] * metrics['AR50'] / (metrics['AP50'] + metrics['AR50'])
        metrics['F1_75'] = 2 * metrics['AP75'] * metrics['AR75'] / (metrics['AP75'] + metrics['AR75'])
        metrics['F1_small'] = 2 * metrics['AP_small'] * metrics['AR_small'] / (metrics['AP_small'] + metrics['AR_small'])
        metrics['F1_medium'] = 2 * metrics['AP_medium'] * metrics['AR_medium'] / (metrics['AP_medium'] + metrics['AR_medium'])
        metrics['F1_large'] = 2 * metrics['AP_large'] * metrics['AR_large'] / (metrics['AP_large'] + metrics['AR_large'])

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
            # Now 15 metrics instead of 13.
            stats = np.zeros((15,))
            stats_strings = ['' for _ in range(15)]

            # AP metrics
            stats[0], stats_strings[0] = _summarize(1, maxDets=self.params.maxDets[max_dets_index])
            stats[1], stats_strings[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[max_dets_index])
            stats[2], stats_strings[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[max_dets_index])
            stats[3], stats_strings[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[max_dets_index])
            stats[4], stats_strings[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[max_dets_index])
            stats[5], stats_strings[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[max_dets_index])

            # AR metrics (average recall over different max detections)
            stats[6], stats_strings[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7], stats_strings[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8], stats_strings[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9], stats_strings[9] = (_summarize(0, maxDets=self.params.maxDets[3])
                                          if len(self.params.maxDets) > 3
                                          else _summarize(0, maxDets=self.params.maxDets[2]))

            # New: AR at specific IoU thresholds (mAR50 and mAR75)
            stats[10], stats_strings[10] = _summarize(0, iouThr=.5, maxDets=self.params.maxDets[max_dets_index])
            stats[11], stats_strings[11] = _summarize(0, iouThr=.75, maxDets=self.params.maxDets[max_dets_index])

            # AR for different object sizes
            stats[12], stats_strings[12] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[max_dets_index])
            stats[13], stats_strings[13] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[max_dets_index])
            stats[14], stats_strings[14] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[max_dets_index])
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
        self.summarize_custom()
        stats = self.stats
        metric_names = [
            "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
            "AR_1", "AR_10", "AR_100", "AR", "AR50", "AR75", "AR_small", "AR_medium", "AR_large"
        ]
        metrics_dict = {name: float(value) for name, value in zip(metric_names, stats)}
        return metrics_dict
    