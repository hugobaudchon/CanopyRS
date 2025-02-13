import json
import os
from pathlib import Path
from warnings import warn

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from geodataset.geodata import Raster
from matplotlib import patches, pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rasterio.transform import rowcol
from shapely import unary_union
from yaml import warnings

from engine.components.base import BaseComponent
from engine.config_parsers import TilerizerConfig
from engine.config_parsers.evaluator import EvaluatorConfig
from engine.data_state import DataState


class EvaluatorComponent(BaseComponent):
    name = 'evaluator'

    def __init__(self, config: EvaluatorConfig, parent_output_path: str, component_id: int):
        super().__init__(config, parent_output_path, component_id)

    def __call__(self, data_state: DataState) -> DataState:
        if self.config.type == 'instance_detection':
            iou_type = 'bbox'
        elif self.config.type == 'instance_segmentation':
            iou_type = 'segm'
        else:
            raise ValueError(f'Unknown evaluator type: {self.config.type}')

        if self.config.level == 'tile':
            if data_state.infer_coco_path is not None and data_state.ground_truth_coco_path is not None:
                truth_coco = COCO(data_state.ground_truth_coco_path)
                preds_coco = COCO(data_state.infer_coco_path)

                # Align the predictions COCO images to the order of the ground truth COCO images
                align_coco_datasets_by_name(truth_coco, preds_coco)

                # visualize_preds_and_truth(truth_coco, preds_coco, data_state.tiles_path / 'groundtruth', 12)

                coco_evaluator = Summarize2COCOEval(
                    cocoGt=truth_coco,
                    cocoDt=preds_coco,
                    iouType=iou_type
                )
                coco_evaluator.params.maxDets = self.config.max_dets
                coco_evaluator.evaluate()
                coco_evaluator.accumulate()
                stats_strings = coco_evaluator.summarize_custom()

                # Save stats to disk
                with open(self.output_path / "coco_metrics_tiles.txt", "w") as f:
                    f.write(f"Aggregated results at the tile level.\n")
                    for string in stats_strings:
                        f.write(string + "\n")
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
                coco_evaluator.params.maxDets = [1, 10, 100, len(infer_gdf)]
                coco_evaluator.evaluate()
                coco_evaluator.accumulate()
                stats_strings = coco_evaluator.summarize_custom()

                # Save stats to disk
                with open(self.output_path / "coco_metrics_raster.txt", "w") as f:
                    f.write(f"Aggregated results at the raster level with ground resolution {self.config.raster_eval_ground_resolution}.\n")
                    for string in stats_strings:
                        f.write(string + "\n")
            else:
                raise Exception("Missing a GeoDataFrame.")

        return self.update_data_state(data_state)

    def update_data_state(self, data_state: DataState) -> DataState:
        return data_state


def align_coco_datasets_by_name(truth_coco: COCO, preds_coco: COCO) -> None:
    """
    Align the predictions COCO dataset to follow the order of the truth COCO dataset,
    matching based on the image 'file_name'. For any truth image missing in preds,
    insert a dummy image (with no annotations) so that the image IDs match.

    This function updates the preds_coco in-place.
    """
    # Build a mapping from file_name to preds image metadata.
    preds_by_name = {img['file_name']: img for img in preds_coco.dataset.get('images', [])}

    # This mapping will help us update annotation image IDs.
    # It maps the original preds image id to the new image id (which will match the truth).
    id_mapping = {}

    # Build a new list of prediction images ordered as in truth.
    new_preds_images = []
    for truth_img in truth_coco.dataset.get('images', []):
        file_name = truth_img['file_name']
        truth_id = truth_img['id']
        if file_name in preds_by_name:
            preds_img = preds_by_name[file_name]
            # Record mapping from original preds id to the truth id.
            id_mapping[preds_img['id']] = truth_id
            # Make a copy and update its id to match the truth.
            new_img = preds_img.copy()
            new_img['id'] = truth_id
            new_preds_images.append(new_img)
        else:
            # If the truth image is missing in predictions, add a dummy image.
            # The dummy uses the truth image metadata.
            new_preds_images.append(truth_img.copy())

    # Update the predictions dataset with the new ordered images.
    preds_coco.dataset['images'] = new_preds_images

    # Update predictions annotations:
    # Only keep annotations for images that are in the truth.
    new_preds_annotations = []
    for ann in preds_coco.dataset.get('annotations', []):
        orig_img_id = ann['image_id']
        if orig_img_id in id_mapping:
            # Update the annotation's image_id to match the truth.
            ann['image_id'] = id_mapping[orig_img_id]
            new_preds_annotations.append(ann)
        else:
            # If an annotation's image doesn't match any truth image,
            # skip it.
            continue
    preds_coco.dataset['annotations'] = new_preds_annotations

    # Refresh the COCO index in the predictions object.
    preds_coco.createIndex()


def gdf_to_coco_single_image(gdf, raster, is_ground_truth=False):
    # Load the raster to get height, width, and transform
    pixel_coordinates_gdf = raster.adjust_geometries_to_raster_crs_if_necessary(gdf)
    pixel_coordinates_gdf = raster.adjust_geometries_to_raster_pixel_coordinates(pixel_coordinates_gdf)

    geometries = pixel_coordinates_gdf['geometry'].tolist()
    scores = pixel_coordinates_gdf['aggregator_score'].tolist() if not is_ground_truth else None
    categories = [1] * len(geometries)
    image_id = 1
    annotations = []

    # Transform geometries to raster pixel coordinates
    for i, geometry in enumerate(geometries):
        if geometry.is_empty or not geometry.is_valid:
            continue  # Skip invalid or empty geometries

        # Transform geometry to pixel coordinates
        if geometry.geom_type == "Polygon":
            segmentation = [np.array(geometry.exterior.coords).flatten().tolist()]  # Flatten list of tuples
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
        '''
        Compute and display summary metrics for evaluation results.
        Note this function can *only* be applied on the default parameter setting
        '''

        max_dets_index = len(self.params.maxDets) - 1

        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
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
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        else:
            raise Exception('Unknown iouType: {}'.format(iouType))
        self.stats, stats_strings = summarize()
        return stats_strings


def visualize_preds_and_truth(truth_coco, preds_coco, images_dir, num_images=12):
    """
    Visualize the first `num_images` images from the truth_coco dataset, overlaying
    the ground truth annotations (in green) and prediction annotations (in red).

    Args:
        truth_coco: The COCO object for ground truth.
        preds_coco: The COCO object for predictions.
        images_dir: Directory where the image files are stored.
        num_images: Number of images to visualize.
    """
    # Build dictionaries mapping image_id to their annotations
    gt_ann_dict = {}
    for ann in truth_coco.dataset.get('annotations', []):
        gt_ann_dict.setdefault(ann['image_id'], []).append(ann)

    pred_ann_dict = {}
    for ann in preds_coco.dataset.get('annotations', []):
        pred_ann_dict.setdefault(ann['image_id'], []).append(ann)

    # Get the first `num_images` images from truth_coco
    images = truth_coco.dataset.get('images', [])[:num_images]

    # Setup matplotlib grid; here we use 3 rows x 4 cols for 12 images.
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    for ax, img_meta in zip(axes, images):
        file_name = img_meta['file_name']
        image_path = os.path.join(images_dir, file_name)

        # Load image (using cv2 here; ensure your images_dir is correct)
        image = cv2.imread(image_path)
        if image is None:
            ax.set_title(f"Image not found: {file_name}")
            ax.axis("off")
            continue
        # Convert BGR to RGB for plotting with matplotlib
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)

        # Draw ground truth boxes in green
        for ann in gt_ann_dict.get(img_meta['id'], []):
            # The COCO bbox format is [x, y, width, height]
            x, y, w, h = ann['bbox']
            rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                     edgecolor='green', facecolor='none')
            ax.add_patch(rect)

        # Draw prediction boxes in red
        for ann in pred_ann_dict.get(img_meta['id'], []):
            x, y, w, h = ann['bbox']
            rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                     edgecolor='red', facecolor='none')
            ax.add_patch(rect)

        ax.set_title(file_name)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
