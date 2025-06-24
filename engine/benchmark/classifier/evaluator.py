# engine/benchmark/classifier/evaluator.py
import warnings
import json
import tempfile
from pathlib import Path
import numpy as np
from faster_coco_eval.core.coco import COCO
from faster_coco_eval.core.faster_eval_api import COCOeval_faster

class ClassifierCocoEvaluator:
    small_max_sq_meters = 16
    medium_max_sq_meters = 100

    def tile_level(self,
                   preds_coco_path: str,
                   truth_coco_path: str,
                   max_dets: list[int] = (1, 10, 100),
                   evaluate_bbox: bool = False,
                   score_combination: Optional[Dict] = None) -> dict:
        """
        Tile-level evaluation for instance segmentation with classes.

        Args:
            preds_coco_path: Path to predictions COCO file
            truth_coco_path: Path to ground truth COCO file  
            max_dets: Maximum detections to consider
            evaluate_bbox: If True, also evaluate bbox IoU and return both metrics
            score_combination: Dict with 'weights' and 'method' for combining scores
                            Example: {
                                'weights': {'detector_score': 0.5, 'classifier_score': 0.3, 'segmentation_score': 0.2},
                                'method': 'weighted_arithmetic_mean'  # or 'weighted_geometric_mean'
                            }

        Returns:
            dict: Metrics dictionary. If evaluate_bbox=True, contains both 'segm' and 'bbox' keys
        """
        # Preprocess predictions to fix score field names
        processed_preds_path = self._preprocess_predictions_coco(
            preds_coco_path,
            score_combination=score_combination
        )

        # Load COCO files
        truth_coco = COCO(str(truth_coco_path))
        preds_coco = COCO(str(processed_preds_path))

        # Remove scores from ground truth annotations (if any)
        for ann in truth_coco.dataset['annotations']:
            if 'score' in ann:
                del ann['score']

        # Align predictions to truth based on file name
        self._align_coco_datasets_by_name(truth_coco, preds_coco)

        results = {}
        # Always evaluate segmentation (primary metric for classification)
        print("Evaluating segmentation IoU...")
        segm_metrics = self._evaluate_single_iou_type(truth_coco, preds_coco, 'segm', max_dets)
        results['segm'] = segm_metrics

        # Optionally evaluate bbox
        if evaluate_bbox:
            print("Evaluating bbox IoU...")
            bbox_metrics = self._evaluate_single_iou_type(truth_coco, preds_coco, 'bbox', max_dets)
            results['bbox'] = bbox_metrics

        # Clean up temporary file
        if processed_preds_path != preds_coco_path:
            Path(processed_preds_path).unlink()

        # Return segm metrics directly if only evaluating segmentation
        # Return both if evaluating both
        return results if evaluate_bbox else segm_metrics

    def _evaluate_single_iou_type(self, truth_coco: COCO, preds_coco: COCO, 
                                  iou_type: str, max_dets: list[int]) -> dict:
        """Evaluate a single IoU type (segm or bbox)"""

        # Create fresh COCO objects to avoid interference between evaluations
        truth_coco_copy = COCO()
        truth_coco_copy.dataset = truth_coco.dataset.copy()
        truth_coco_copy.createIndex()

        preds_coco_copy = COCO()
        preds_coco_copy.dataset = preds_coco.dataset.copy()
        preds_coco_copy.createIndex()

        # Set up and run COCO evaluation
        coco_evaluator = Summarize2COCOEval(
            cocoGt=truth_coco_copy,
            cocoDt=preds_coco_copy,
            iouType=iou_type
        )
        coco_evaluator.params.maxDets = max_dets
        coco_evaluator.evaluate()
        coco_evaluator.accumulate()

        # Get metrics as a dictionary and add debug info
        metrics = coco_evaluator.summarize_to_dict()
        num_images = len(truth_coco_copy.dataset.get('images', []))
        num_truths = len(truth_coco_copy.dataset.get('annotations', []))
        num_preds = len(preds_coco_copy.dataset.get('annotations', []))

        metrics['num_images'] = num_images
        metrics['num_truths'] = num_truths
        metrics['num_preds'] = num_preds
        metrics['iou_type'] = iou_type

        return metrics

    def _preprocess_predictions_coco(self, preds_coco_path: str, 
                                     score_combination: Optional[Dict] = None) -> str:
        """
        Preprocess predictions COCO file to standardize score field names.
        Can also combine multiple scores using weighted averaging.

        Args:
            preds_coco_path: Path to predictions COCO file
            score_combination: Dict with score combination settings

        Returns:
            Path to processed file (may be the same as input if no changes needed).
        """
        # Load the original predictions file
        with open(preds_coco_path, 'r') as f:
            coco_data = json.load(f)

        # Check if we need to fix score fields
        score_field_candidates = [
            'classifier_score',
            'aggregator_score',
            'detector_score',
            'segmentation_score',
            'prediction_score'
        ]

        # Check what score fields exist in annotations
        existing_score_fields = set()
        for ann in coco_data.get('annotations', []):
            # Check if score already exists at top level
            if 'score' in ann:
                existing_score_fields.add('score')

            # Check in other_attributes
            other_attrs = ann.get('other_attributes', {})
            for field in score_field_candidates:
                if field in other_attrs:
                    existing_score_fields.add(field)

        # Handle score combination if specified
        if score_combination and 'weights' in score_combination:
            print(f"Combining scores using {score_combination.get('method', 'weighted_arithmetic_mean')}")
            self._combine_scores(coco_data, score_combination)
            print("Score combination completed")

        # Handle single score field selection (existing logic)
        elif not ('score' in existing_score_fields and len(existing_score_fields) == 1):
            score_field_to_use = self._select_primary_score_field(existing_score_fields)
            if score_field_to_use:
                print(f"Converting '{score_field_to_use}' to 'score' in predictions")
                self._move_score_to_top_level(coco_data, score_field_to_use)
            else:
                print("WARNING: No score fields found in predictions.")
                return preds_coco_path
        else:
            print("Predictions already have 'score' field, no preprocessing needed")
            return preds_coco_path

        # Create temporary file with processed data
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='_fixed_predictions.json', delete=False)
        with temp_file as f:
            json.dump(coco_data, f, indent=2)

        return temp_file.name

    def _combine_scores(self, coco_data: Dict, score_combination: Dict):
        """Combine multiple scores using specified weights and method"""
        weights = score_combination['weights']
        method = score_combination.get('method', 'weighted_arithmetic_mean')

        # Validate weights sum to 1
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            print(f"WARNING: Weights sum to {weight_sum:.3f}, normalizing to 1.0")
            weights = {k: v/weight_sum for k, v in weights.items()}

        for ann in coco_data.get('annotations', []):
            other_attrs = ann.get('other_attributes', {})
            scores_to_combine = {}

            # Collect available scores
            for score_name, weight in weights.items():
                if score_name in other_attrs:
                    scores_to_combine[score_name] = other_attrs[score_name]
                elif score_name in ann:
                    scores_to_combine[score_name] = ann[score_name]

            if scores_to_combine:
                if method == 'weighted_arithmetic_mean':
                    combined_score = self._weighted_arithmetic_mean(scores_to_combine, weights)
                elif method == 'weighted_geometric_mean':
                    combined_score = self._weighted_geometric_mean(scores_to_combine, weights)
                else:
                    raise ValueError(f"Unknown score combination method: {method}")

                ann['score'] = combined_score
            else:
                print(f"WARNING: No scores found to combine for annotation {ann.get('id', 'unknown')}")

    def _weighted_arithmetic_mean(self, scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """Compute weighted arithmetic mean of scores"""
        total_score = 0.0
        total_weight = 0.0

        for score_name, score_value in scores.items():
            if score_name in weights:
                weight = weights[score_name]
                total_score += score_value * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _weighted_geometric_mean(self, scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """Compute weighted geometric mean of scores"""
        import math

        log_sum = 0.0
        total_weight = 0.0

        for score_name, score_value in scores.items():
            if score_name in weights and score_value > 0:
                weight = weights[score_name]
                log_sum += weight * math.log(score_value)
                total_weight += weight

        return math.exp(log_sum / total_weight) if total_weight > 0 else 0.0

    def _select_primary_score_field(self, existing_score_fields: set) -> Optional[str]:
        """Select primary score field when no combination is specified"""
        if 'classifier_score' in existing_score_fields:
            return 'classifier_score'
        elif 'aggregator_score' in existing_score_fields:
            return 'aggregator_score'
        elif 'detector_score' in existing_score_fields:
            return 'detector_score'
        elif existing_score_fields:
            candidates = existing_score_fields - {'score'}
            return list(candidates)[0] if candidates else None
        return None

    def _move_score_to_top_level(self, coco_data: Dict, score_field_to_use: str):
        """Move selected score field to top level of annotations"""
        for ann in coco_data.get('annotations', []):
            other_attrs = ann.get('other_attributes', {})
            if score_field_to_use in other_attrs:
                ann['score'] = other_attrs[score_field_to_use]
            elif score_field_to_use in ann:
                ann['score'] = ann[score_field_to_use]

    def _align_coco_datasets_by_name(self, truth_coco: COCO, preds_coco: COCO) -> None:
        """
        Align the predictions COCO dataset to follow the order of the truth COCO dataset,
        matching based on the image 'file_name'.
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

    def _evaluate_single_iou_type(self, truth_coco: COCO, preds_coco: COCO, 
                                  iou_type: str, max_dets: list[int]) -> dict:
        """Evaluate a single IoU type (segm or bbox)"""

        # Create fresh COCO objects to avoid interference between evaluations
        truth_coco_copy = COCO()
        truth_coco_copy.dataset = truth_coco.dataset.copy()
        truth_coco_copy.createIndex()

        preds_coco_copy = COCO()
        preds_coco_copy.dataset = preds_coco.dataset.copy()
        preds_coco_copy.createIndex()

        # Set up and run COCO evaluation
        coco_evaluator = Summarize2COCOEval(
            cocoGt=truth_coco_copy,
            cocoDt=preds_coco_copy,
            iouType=iou_type
        )
        coco_evaluator.params.maxDets = max_dets
        coco_evaluator.evaluate()
        coco_evaluator.accumulate()

        # Get metrics as a dictionary and add debug info
        metrics = coco_evaluator.summarize_to_dict()
        num_images = len(truth_coco_copy.dataset.get('images', []))
        num_truths = len(truth_coco_copy.dataset.get('annotations', []))
        num_preds = len(preds_coco_copy.dataset.get('annotations', []))

        metrics['num_images'] = num_images
        metrics['num_truths'] = num_truths
        metrics['num_preds'] = num_preds
        metrics['iou_type'] = iou_type

        # Add mIoU computation
        if hasattr(coco_evaluator, 'eval') and coco_evaluator.eval:
            miou = self._compute_miou(coco_evaluator.eval)
            metrics['mIoU'] = miou

        return metrics

    def _compute_miou(self, eval_result):
        """Compute mean IoU from COCO evaluation results"""
        try:
            # Extract IoU values from evaluation results
            iou_scores = np.array([*eval_result.get('matched').values()])
            if iou_scores is not None and isinstance(iou_scores, np.ndarray):
                # Filter valid IoU values (> 0, typically <= 1)
                valid_ious = iou_scores[(iou_scores > 0) & (iou_scores <= 1)]
                if len(valid_ious) > 0:
                    return float(np.mean(valid_ious))
            return 0.0
        except:
            return 0.0


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
            stats = np.zeros((15,))
            stats_strings = ['' for _ in range(15)]

            stats[0], stats_strings[0] = _summarize(1, maxDets=self.params.maxDets[max_dets_index])
            stats[1], stats_strings[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[max_dets_index])
            stats[2], stats_strings[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[max_dets_index])
            stats[3], stats_strings[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[max_dets_index])
            stats[4], stats_strings[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[max_dets_index])
            stats[5], stats_strings[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[max_dets_index])
            stats[6], stats_strings[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7], stats_strings[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8], stats_strings[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9], stats_strings[9] = (_summarize(0, maxDets=self.params.maxDets[3])
                                          if len(self.params.maxDets) > 3
                                          else _summarize(0, maxDets=self.params.maxDets[2]))
            stats[10], stats_strings[10] = _summarize(0, iouThr=.5, maxDets=self.params.maxDets[max_dets_index])
            stats[11], stats_strings[11] = _summarize(0, iouThr=.75, maxDets=self.params.maxDets[max_dets_index])
            stats[12], stats_strings[12] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[max_dets_index])
            stats[13], stats_strings[13] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[max_dets_index])
            stats[14], stats_strings[14] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[max_dets_index])
            return stats, stats_strings

        if not self.eval:
            raise Exception('Please run accumulate() first')
        
        self.stats, stats_strings = _summarizeDets()
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
