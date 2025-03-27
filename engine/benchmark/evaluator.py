import numpy as np
from geodataset.geodata import Raster
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import geopandas as gpd

from engine.config_parsers.evaluator import EvaluatorConfig


class CocoEvaluator:
    def tile_level(self, config: EvaluatorConfig, preds_coco_path: str, truth_coco_path: str) -> dict:
        if config.type == 'instance_detection':
            iou_type = 'bbox'
        elif config.type == 'instance_segmentation':
            iou_type = 'segm'
        else:
            raise ValueError(f'Unknown evaluator type: {config.type}')

        print("Truth COCO path:", truth_coco_path)
        print("Predictions COCO path:", preds_coco_path)

        truth_coco = COCO(truth_coco_path)
        preds_coco = COCO(preds_coco_path)

        # Debug prints before alignment
        truth_images = truth_coco.dataset.get('images', [])
        preds_images = preds_coco.dataset.get('images', [])
        print("Before alignment:")
        print(f"  Number of truth images: {len(truth_images)}")
        print(f"  Number of prediction images: {len(preds_images)}")
        truth_file_names = [img['file_name'] for img in truth_images]
        preds_file_names = [img['file_name'] for img in preds_images]
        print("  Truth file names (first 10):", truth_file_names[:10])
        print("  Prediction file names (first 10):", preds_file_names[:10])

        # Align predictions to truth based on file name
        align_coco_datasets_by_name(truth_coco, preds_coco)

        # Debug prints after alignment
        aligned_preds_images = preds_coco.dataset.get('images', [])
        aligned_preds_file_names = [img['file_name'] for img in aligned_preds_images]
        print("After alignment:")
        print(f"  Number of aligned prediction images: {len(aligned_preds_images)}")
        print("  Aligned prediction file names (first 10):", aligned_preds_file_names[:10])

        # Check that every truth image has a corresponding prediction image
        missing = []
        for truth_img in truth_coco.dataset.get('images', []):
            file_name = truth_img['file_name']
            if file_name not in aligned_preds_file_names:
                missing.append(file_name)
        if missing:
            print("Warning: The following truth images are missing in predictions after alignment:")
            for fname in missing:
                print("  ", fname)
        else:
            print("All truth images have corresponding predictions after alignment.")

        # Set up and run COCO evaluation
        coco_evaluator = Summarize2COCOEval(
            cocoGt=truth_coco,
            cocoDt=preds_coco,
            iouType=iou_type
        )
        coco_evaluator.params.maxDets = config.max_dets
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

        print("Final evaluation metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        return metrics

    def raster_level(self, config: EvaluatorConfig, preds_gpkg_path: str, truth_gpkg_path: str, imagery_path: str) -> dict:
        raster = Raster(
            path=imagery_path,
            ground_resolution=config.raster_eval_ground_resolution,
        )

        truth_gdf = gpd.read_file(truth_gpkg_path)
        infer_gdf = gpd.read_file(preds_gpkg_path)

        truth_coco = gdf_to_coco_single_image(
            gdf=truth_gdf,
            raster=raster,
            is_ground_truth=True
        )
        infer_coco = gdf_to_coco_single_image(
            gdf=infer_gdf,
            raster=raster,
            is_ground_truth=False
        )

        coco_gt_obj = COCO()
        coco_gt_obj.dataset = truth_coco
        coco_gt_obj.createIndex()

        coco_dt_obj = COCO()
        coco_dt_obj.dataset = infer_coco
        coco_dt_obj.createIndex()

        # Initialize and run COCOeval
        coco_evaluator = Summarize2COCOEval(
            cocoGt=coco_gt_obj,
            cocoDt=coco_dt_obj,
            iouType='bbox'
        )
        coco_evaluator.params.maxDets = [1, 10, 100, len(truth_gdf)]
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

        return metrics


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


def gdf_to_coco_single_image(gdf, raster, is_ground_truth=False):
    pixel_coordinates_gdf = raster.adjust_geometries_to_raster_crs_if_necessary(gdf)
    pixel_coordinates_gdf = raster.adjust_geometries_to_raster_pixel_coordinates(pixel_coordinates_gdf)

    geometries = pixel_coordinates_gdf['geometry'].tolist()
    scores = pixel_coordinates_gdf['aggregator_score'].tolist() if not is_ground_truth else None
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
            'file_name': str(raster.name),
            'width': raster.metadata['width'],
            'height': raster.metadata['height']
        }],
        'annotations': annotations,
        'categories': [{
            'id': 1, 'name': 'object'
        }]
    }
    return coco


class Summarize2COCOEval(COCOeval):
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
            stats = np.zeros((13,))
            stats_strings = ['' for _ in range(13)]
            stats[0], stats_strings[0] = _summarize(1, maxDets=self.params.maxDets[max_dets_index])
            stats[1], stats_strings[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[max_dets_index])
            stats[2], stats_strings[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[max_dets_index])
            stats[3], stats_strings[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[max_dets_index])
            stats[4], stats_strings[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[max_dets_index])
            stats[5], stats_strings[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[max_dets_index])
            stats[6], stats_strings[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7], stats_strings[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8], stats_strings[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9], stats_strings[9] = _summarize(0, maxDets=self.params.maxDets[3]) if len(self.params.maxDets) > 3 else _summarize(0, maxDets=self.params.maxDets[2])
            stats[10], stats_strings[10] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[max_dets_index])
            stats[11], stats_strings[11] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[max_dets_index])
            stats[12], stats_strings[12] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[max_dets_index])
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
            "AR_1", "AR_10", "AR_100", "AR_max", "AR_small", "AR_medium", "AR_large"
        ]
        metrics_dict = {name: float(value) for name, value in zip(metric_names, stats)}
        return metrics_dict
