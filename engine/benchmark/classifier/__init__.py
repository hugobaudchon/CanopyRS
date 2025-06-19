# engine/benchmark/classifier/evaluator.py
import warnings
import geopandas as gpd
import numpy as np
from faster_coco_eval.core.coco import COCO
from faster_coco_eval.core.faster_eval_api import COCOeval_faster # Or your Summarize2COCOEval

# Potentially reuse align_coco_datasets_by_name, gdf_to_coco_single_image, filter_min_overlap
# from the original evaluator.py or geodataset.utils if they are generic enough.
# For now, let's assume we might need slightly adapted versions or can use them as is.
# We'll import Summarize2COCOEval from the detector's evaluator for now, or define it here if needed.
from engine.benchmark.detector.evaluator import Summarize2COCOEval, align_coco_datasets_by_name 

class ClassifierCocoEvaluator: # Renamed to avoid confusion if both are imported
    def __init__(self):
        # small_max_sq_meters, medium_max_sq_meters could be relevant if area-based AP/AR for segm is desired
        pass

    def tile_level(self,
                   preds_coco_path: str,
                   truth_coco_path: str,
                   max_dets: list[int] = (1, 10, 100)) -> dict:
        """
        Evaluates instance segmentation predictions at the tile level using COCO metrics.
        """
        truth_coco = COCO(str(truth_coco_path))
        preds_coco = COCO(str(preds_coco_path))

        # Remove scores from truth annotations if they exist
        for ann in truth_coco.dataset.get('annotations', []):
            if 'score' in ann:
                del ann['score']

        align_coco_datasets_by_name(truth_coco, preds_coco)

        coco_evaluator = Summarize2COCOEval( # Or COCOeval_faster directly
            cocoGt=truth_coco,
            cocoDt=preds_coco,
            iouType='segm'  # <<< KEY CHANGE FOR SEGMENTATION
        )
        coco_evaluator.params.maxDets = max_dets # Ensure this is appropriate for segmentation
        coco_evaluator.evaluate()
        coco_evaluator.accumulate()

        metrics = coco_evaluator.summarize_to_dict()
        metrics['num_images'] = len(truth_coco.dataset.get('images', []))
        metrics['num_truths'] = len(truth_coco.dataset.get('annotations', []))
        metrics['num_preds'] = len(preds_coco.dataset.get('annotations', []))
        
        # Add F1 scores (optional, but often useful)
        if metrics.get('AP', -1) > -1 and metrics.get('AR', -1) > -1 and (metrics['AP'] + metrics['AR']) > 0:
             metrics['F1'] = 2 * metrics['AP'] * metrics['AR'] / (metrics['AP'] + metrics['AR'])
        if metrics.get('AP50', -1) > -1 and metrics.get('AR50', -1) > -1 and (metrics['AP50'] + metrics['AR50']) > 0:
             metrics['F1_50'] = 2 * metrics['AP50'] * metrics['AR50'] / (metrics['AP50'] + metrics['AR50'])
        if metrics.get('AP75', -1) > -1 and metrics.get('AR75', -1) > -1 and (metrics['AP75'] + metrics['AR75']) > 0:
             metrics['F1_75'] = 2 * metrics['AP75'] * metrics['AR75'] / (metrics['AP75'] + metrics['AR75'])

        return metrics

    def raster_level_placeholder(self, *args, **kwargs) -> dict:
        """
        Placeholder for future raster-level evaluation for segmentation/classification.
        """
        warnings.warn("Raster-level evaluation for classifier is not yet implemented.")
        return {"status": "not_implemented", "reason": "raster_level_placeholder"}

    # Placeholder for future tile classification metrics (non-COCO)
    def tile_classification_metrics_placeholder(self, preds_df, truth_df, class_names) -> dict:
        warnings.warn("Tile classification metrics (e.g., overall F1, per-class F1) not yet implemented.")
        # Example:
        # from sklearn.metrics import classification_report
        # report = classification_report(truth_df['class_id'], preds_df['predicted_class_id'], target_names=class_names, output_dict=True)
        # return report
        return {"status": "not_implemented", "reason": "tile_classification_metrics_placeholder"}
