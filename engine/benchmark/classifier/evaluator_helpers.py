# Helper methods for category-agnostic evaluation
from faster_coco_eval.core.coco import COCO
import numpy as np


def compute_iou_mask(mask1, mask2):
    """Compute IoU between two binary masks"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def compute_iou_bbox(box1, box2):
    """Compute IoU between two bboxes [x, y, w, h]"""
    x1_min, y1_min = box1[0], box1[1]
    x1_max, y1_max = x1_min + box1[2], y1_min + box1[3]
    
    x2_min, y2_min = box2[0], box2[1]
    x2_max, y2_max = x2_min + box2[2], y2_min + box2[3]
    
    # Intersection
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    if xi_max <= xi_min or yi_max <= yi_min:
        return 0.0
    
    intersection = (xi_max - xi_min) * (yi_max - yi_min)
    
    # Union
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union
