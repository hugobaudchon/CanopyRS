import json
import os
import copy
import random
from warnings import warn

import cv2
import numpy as np
from geodataset.geodata import Raster
from matplotlib import patches, pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from engine.components.base import BaseComponent
from engine.config_parsers.evaluator import EvaluatorConfig
from engine.data_state import DataState


class EvaluatorComponent(BaseComponent):
    name = 'evaluator'

    def __init__(self, config: EvaluatorConfig, parent_output_path: str, component_id: int):
        super().__init__(config, parent_output_path, component_id)

    def __call__(self, data_state: DataState) -> DataState:
        # making sure COCO files are generated before starting evaluation
        data_state.clean_side_processes()

        if self.config.type == 'instance_detection':
            iou_type = 'bbox'
        elif self.config.type == 'instance_segmentation':
            iou_type = 'segm'
        else:
            raise ValueError(f'Unknown evaluator type: {self.config.type}')

        print(f"Evaluating the predictions at the '{self.config.level}' level.")

        if self.config.level == 'tile':
            if data_state.infer_coco_path is not None and data_state.ground_truth_coco_path is not None:
                # Load ground truth and raw predictions
                truth_coco = COCO(data_state.ground_truth_coco_path)
                raw_preds_coco = COCO(data_state.infer_coco_path)

                # Apply NMS on a deepcopy of the raw predictions.
                nms_threshold = getattr(self.config, 'nms_threshold', 0.5)
                nms_preds_coco = self.apply_nms_to_preds(copy.deepcopy(raw_preds_coco), nms_threshold=nms_threshold)

                # Align BOTH raw and NMS predictions to the ground truth images
                align_coco_datasets_by_name(truth_coco, raw_preds_coco)
                align_coco_datasets_by_name(truth_coco, nms_preds_coco)

                # Use the aligned NMS predictions for evaluation
                preds_coco = nms_preds_coco

                # Align the predictions COCO images to the order of the ground truth COCO images
                align_coco_datasets_by_name(truth_coco, preds_coco)

                # (Optional) Visualization of predictions and ground truth before evaluation
                # visualize_preds_and_truth(truth_coco, preds_coco, data_state.tiles_path / 'groundtruth', 12)

                coco_evaluator = Summarize2COCOEval(
                    cocoGt=truth_coco,
                    cocoDt=preds_coco,
                    iouType=iou_type
                )
                coco_evaluator.params.maxDets = self.config.max_dets
                coco_evaluator.evaluate()
                coco_evaluator.accumulate()

                # Get metrics as a dictionary and save as JSON
                metrics = coco_evaluator.summarize_to_dict()
                num_images = len(truth_coco.dataset.get('images', []))
                num_truths = len(truth_coco.dataset.get('annotations', []))
                num_preds = len(preds_coco.dataset.get('annotations', []))
                with open(self.output_path / "coco_metrics_tiles.json", "w") as f:
                    json.dump({
                        "message": "Aggregated results at the tile level.",
                        "metrics": metrics,
                        "num_images": num_images,
                        "num_truths": num_truths,
                        "num_preds": num_preds
                    }, f, indent=2)

                # Save visualizations for 9 random images (with seed 0 for reproducibility).
                # We assume the images are stored in data_state.tiles_path / 'groundtruth'
                images_dir = data_state.tiles_path / 'groundtruth'
                self.save_random_visualizations(truth_coco, raw_preds_coco, nms_preds_coco, images_dir)
            else:
                raise Exception("Missing a COCO file.")

        elif self.config.level == 'raster':
            if data_state.infer_gdf is not None and data_state.ground_truth_gdf is not None:
                truth_gdf = data_state.ground_truth_gdf.copy()
                infer_gdf = data_state.infer_gdf.copy()
                if infer_gdf is None or truth_gdf.crs is None:
                    warn("No CRS found in one of the GeoDataFrames. Skipping Evaluation for the aggregated results.")

                raster = Raster(
                    path=data_state.imagery_path,
                    ground_resolution=self.config.raster_eval_ground_resolution,
                )

                # Convert GeoDataFrames to COCO format for a single image
                coco_gt = gdf_to_coco_single_image(
                    gdf=truth_gdf,
                    raster=raster,
                    is_ground_truth=True
                )
                coco_dt = gdf_to_coco_single_image(
                    gdf=infer_gdf,
                    raster=raster,
                    is_ground_truth=False
                )

                # Perform COCO evaluation
                coco_gt_obj = COCO()
                coco_gt_obj.dataset = coco_gt
                coco_gt_obj.createIndex()

                coco_dt_obj = COCO()
                coco_dt_obj.dataset = coco_dt
                coco_dt_obj.createIndex()

                # Initialize and run COCOeval
                coco_evaluator = Summarize2COCOEval(
                    cocoGt=coco_gt_obj,
                    cocoDt=coco_dt_obj,
                    iouType=iou_type
                )
                coco_evaluator.params.maxDets = [1, 10, 100, len(truth_gdf)]
                coco_evaluator.evaluate()
                coco_evaluator.accumulate()

                # Get metrics as a dictionary and save as JSON
                metrics = coco_evaluator.summarize_to_dict()
                num_images = len(coco_gt['images'])
                num_truths = len(coco_gt['annotations'])
                num_preds = len(coco_dt['annotations'])
                with open(self.output_path / "coco_metrics_raster.json", "w") as f:
                    json.dump({
                        "message": f"Aggregated results at the raster level with ground resolution {self.config.raster_eval_ground_resolution}.",
                        "metrics": metrics,
                        "num_images": num_images,
                        "num_truths": num_truths,
                        "num_preds": num_preds
                    }, f, indent=2)
            else:
                raise Exception("Missing a GeoDataFrame.")
        else:
            raise ValueError(f"Unsupported evaluation level: {self.config.level}")

        return self.update_data_state(data_state)

    def apply_nms_to_preds(self, preds_coco: COCO, nms_threshold: float = 0.5) -> COCO:
        """
        Apply Non-Maximum Suppression (NMS) to each image's detections
        in the given COCO predictions instance.
        """
        import torch
        from torchvision.ops import nms

        new_annotations = []
        # Organize annotations by image id
        anns_by_image = {}
        for ann in preds_coco.dataset.get('annotations', []):
            anns_by_image.setdefault(ann['image_id'], []).append(ann)

        for img in preds_coco.dataset.get('images', []):
            image_id = img['id']
            anns = anns_by_image.get(image_id, [])
            if not anns:
                continue

            boxes = []
            scores = []
            for ann in anns:
                # COCO bbox format: [x, y, width, height]
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                scores.append(ann.get('score', 1.0))

            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            scores_tensor = torch.tensor(scores, dtype=torch.float32)
            keep_indices = nms(boxes_tensor, scores_tensor, nms_threshold)

            # Collect only the annotations kept by NMS
            for idx in keep_indices:
                new_annotations.append(anns[idx])

        # Update the COCO dataset with filtered annotations and rebuild the index.
        preds_coco.dataset['annotations'] = new_annotations
        preds_coco.createIndex()
        return preds_coco

    def save_random_visualizations(self, truth_coco: COCO, raw_preds_coco: COCO, nms_preds_coco: COCO, images_dir) -> None:
        """
        Save 9 figures (PNG) for 9 random images (using seed 0) where each figure
        displays three subplots:
            1. Image with raw predictions overlaid (red boxes)
            2. Image with NMS predictions overlaid (blue boxes)
            3. Image with ground truth annotations overlaid (green boxes)
        The figures are saved in the 'visualizations' folder under self.output_path.
        """
        # Helper: Build a mapping from image_id to annotations
        def build_ann_dict(coco_obj):
            ann_dict = {}
            for ann in coco_obj.dataset.get('annotations', []):
                ann_dict.setdefault(ann['image_id'], []).append(ann)
            return ann_dict

        truth_ann_dict = build_ann_dict(truth_coco)
        raw_preds_ann_dict = build_ann_dict(raw_preds_coco)
        nms_preds_ann_dict = build_ann_dict(nms_preds_coco)

        images = truth_coco.dataset.get('images', [])
        if len(images) < 9:
            sample_images = images
        else:
            random.seed(0)
            sample_images = random.sample(images, 9)

        # Create directory for visualizations
        vis_dir = self.output_path / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        for i, img_meta in enumerate(sample_images):
            file_name = img_meta['file_name']
            image_id = img_meta['id']
            image_path = os.path.join(str(images_dir), file_name)

            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not load image {file_name}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Create a figure with 3 subplots (adjusted size for higher resolution)
            fig, axs = plt.subplots(1, 3, figsize=(48, 12))
            # --- Raw Predictions ---
            axs[0].imshow(image)
            axs[0].set_title("Raw Predictions", fontsize=20)
            for ann in raw_preds_ann_dict.get(image_id, []):
                x, y, w, h = ann['bbox']
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
                axs[0].add_patch(rect)
            axs[0].axis('off')
            # --- NMS Predictions ---
            axs[1].imshow(image)
            axs[1].set_title("NMS Predictions", fontsize=20)
            for ann in nms_preds_ann_dict.get(image_id, []):
                x, y, w, h = ann['bbox']
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='blue', facecolor='none')
                axs[1].add_patch(rect)
            axs[1].axis('off')
            # --- Ground Truth ---
            axs[2].imshow(image)
            axs[2].set_title("Ground Truth", fontsize=20)
            for ann in truth_ann_dict.get(image_id, []):
                x, y, w, h = ann['bbox']
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='green', facecolor='none')
                axs[2].add_patch(rect)
            axs[2].axis('off')

            plt.suptitle(f"Visualization for {file_name}", fontsize=24)
            output_file = vis_dir / f"visualization_{i+1}.png"
            plt.savefig(str(output_file), bbox_inches='tight')
            plt.close(fig)
            print(f"Saved visualization: {output_file}")

    def update_data_state(self, data_state: DataState) -> DataState:
        # Register the component folder
        data_state = self.register_outputs_base(data_state)

        # Register metrics files (now JSON)
        metrics_tiles_path = self.output_path / "coco_metrics_tiles.json"
        metrics_raster_path = self.output_path / "coco_metrics_raster.json"

        if metrics_tiles_path.exists():
            data_state.register_output_file(self.name, self.component_id, 'metrics_tiles', metrics_tiles_path)
        if metrics_raster_path.exists():
            data_state.register_output_file(self.name, self.component_id, 'metrics_raster', metrics_raster_path)

        return data_state


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
