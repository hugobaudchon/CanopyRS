# engine/benchmark/classifier/evaluator.py
import warnings
import json
import tempfile
import copy
from pathlib import Path
import numpy as np
from faster_coco_eval.core.coco import COCO
from pycocotools import mask as mask_util
from typing import Optional, Dict

from canopyrs.engine.benchmark.base.evaluator import (
    AlignmentStrategy,
    AlignmentReport,
    AlignmentError,
    LowMatchRateWarning,
    align_coco_datasets_by_name,
    Summarize2COCOEval,
)

class ClassifierCocoEvaluator:
    small_max_sq_meters = 16
    medium_max_sq_meters = 100
    
    def __init__(self, 
                 alignment_strategy: AlignmentStrategy = AlignmentStrategy.BASE_RSPLIT_1,
                 min_match_rate_warning: float = 0.95,
                 min_match_rate_error: float = 0.50,
                 verbose: bool = True):
        """
        Initialize classifier evaluator.
        
        Args:
            alignment_strategy: Strategy for matching GT and prediction tiles
            min_match_rate_warning: Warn if match rate below this (default 0.95)
            min_match_rate_error: Raise error if match rate below this (default 0.50)
            verbose: Print detailed alignment reports
        """
        self.alignment_strategy = alignment_strategy
        self.min_match_rate_warning = min_match_rate_warning
        self.min_match_rate_error = min_match_rate_error
        self.verbose = verbose
        self.last_alignment_report: Optional[AlignmentReport] = None

    def tile_level(self,
                   preds_coco_path: str,
                   truth_coco_path: str,
                   max_dets: list[int] = (1, 10, 100),
                   evaluate_bbox: bool = False,
                   score_combination: Optional[Dict] = None,
                   evaluate_class_agnostic: bool = False,
                   min_gt_coverage: float = 0.0) -> dict:
        """
        Tile-level evaluation for instance segmentation with classes.

        Args:
            preds_coco_path: Path to predictions COCO file
            truth_coco_path: Path to ground truth COCO file
            max_dets: Maximum detections to consider
            evaluate_bbox: If True, also evaluate bbox IoU
            score_combination: Dict with 'weights' and 'method' for scores
            evaluate_class_agnostic: If True, also compute class-agnostic
                segmentation metrics (ignores category_id)
            min_gt_coverage: Optional filter to exclude tiles with too little
                GT annotation coverage. Coverage is computed per tile as:
                (sum of GT mask areas) / (image_width * image_height).
                Set to 0.0 to disable.

        Returns:
            dict: Metrics dictionary with 'segm' key (and optionally 'bbox',
                'class_agnostic_segm', 'classification_accuracy')
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
        alignment_report = align_coco_datasets_by_name(
            truth_coco,
            preds_coco,
            alignment_strategy=self.alignment_strategy,
            return_report=True,
        )
        self.last_alignment_report = alignment_report
        
        # Print alignment report
        if self.verbose:
            alignment_report.print_report(verbose=True)
        
        # Validate alignment quality
        if alignment_report.match_rate < self.min_match_rate_error:
            raise AlignmentError(
                "Critical alignment failure: only {:.1%} of GT tiles matched "
                "predictions (threshold: {:.1%}).\n"
                "This indicates incompatible tile naming conventions.\n"
                "GT samples: {}\n"
                "Pred samples: {}\n"
                "Consider using AlignmentStrategy.EXACT_MATCH or check your "
                "tiling configuration.".format(
                    alignment_report.match_rate,
                    self.min_match_rate_error,
                    alignment_report.gt_filename_samples,
                    alignment_report.pred_filename_samples
                )
            )

        if alignment_report.match_rate < self.min_match_rate_warning:
            warnings.warn(
                "Low alignment rate: {:.1%} of GT tiles matched "
                "(threshold: {:.1%}). {} GT tiles have no predictions. "
                "This may indicate a pipeline issue.".format(
                    alignment_report.match_rate,
                    self.min_match_rate_warning,
                    alignment_report.num_gt_without_pred
                ),
                LowMatchRateWarning
            )

        # Optional: filter out tiles with very low GT coverage
        if min_gt_coverage > 0:
            truth_coco, preds_coco, filter_report = self._filter_by_min_gt_coverage(
                truth_coco=truth_coco,
                preds_coco=preds_coco,
                min_gt_coverage=min_gt_coverage
            )
            if self.verbose:
                print("\n" + "="*70)
                print("GT COVERAGE FILTER")
                print("="*70)
                print("min_gt_coverage: {:.3%}".format(min_gt_coverage))
                print("Images kept: {} / {}".format(filter_report['num_images_kept'], filter_report['num_images_before']))
                print("Images dropped: {}".format(filter_report['num_images_dropped']))
                print("GT ann kept: {} / {}".format(filter_report['num_gt_anns_kept'], filter_report['num_gt_anns_before']))
                print("Pred ann kept: {} / {}".format(filter_report['num_pred_anns_kept'], filter_report['num_pred_anns_before']))

        results = {}
        # Always evaluate segmentation (primary metric for classification)
        print("Evaluating segmentation IoU...")
        segm_metrics = self._evaluate_single_iou_type(truth_coco, preds_coco, 'segm', max_dets)
        results['segm'] = segm_metrics

        # Optionally evaluate bbox
        if evaluate_bbox:
            print("Evaluating bbox IoU...")
            bbox_metrics = self._evaluate_single_iou_type(
                truth_coco, preds_coco, 'bbox', max_dets)
            results['bbox'] = bbox_metrics

        # Optionally evaluate class-agnostic segmentation
        if evaluate_class_agnostic:
            print("\nEvaluating class-agnostic segmentation...")
            print("(This ignores category_id and only checks mask overlap)")
            ca_metrics = self._evaluate_class_agnostic_segmentation(
                truth_coco, preds_coco, max_dets)
            results['class_agnostic_segm'] = ca_metrics

            # Also compute classification accuracy
            print("\nComputing classification accuracy...")
            class_metrics = self._compute_classification_metrics(
                truth_coco, preds_coco)
            results['classification'] = class_metrics

        # Clean up temporary file
        if processed_preds_path != preds_coco_path:
            Path(processed_preds_path).unlink()

        # Return appropriate results based on what was evaluated
        if evaluate_class_agnostic or evaluate_bbox:
            return results
        else:
            return segm_metrics

    def classification_only(self,
                            preds_coco_path: str,
                            truth_coco_path: str,
                            score_combination: Optional[Dict] = None,
                            min_gt_coverage: float = 0.0) -> dict:
        processed_preds_path = self._preprocess_predictions_coco(
            preds_coco_path,
            score_combination=score_combination,
        )

        truth_coco = COCO(str(truth_coco_path))
        preds_coco = COCO(str(processed_preds_path))

        for ann in truth_coco.dataset['annotations']:
            if 'score' in ann:
                del ann['score']

        alignment_report = align_coco_datasets_by_name(
            truth_coco,
            preds_coco,
            alignment_strategy=self.alignment_strategy,
            return_report=True,
        )
        self.last_alignment_report = alignment_report

        if self.verbose:
            alignment_report.print_report(verbose=True)

        if alignment_report.match_rate < self.min_match_rate_error:
            raise AlignmentError(
                "Critical alignment failure: only {:.1%} of GT tiles matched "
                "predictions (threshold: {:.1%}).\n"
                "This indicates incompatible tile naming conventions.\n"
                "GT samples: {}\n"
                "Pred samples: {}\n"
                "Consider using AlignmentStrategy.EXACT_MATCH or check your "
                "tiling configuration.".format(
                    alignment_report.match_rate,
                    self.min_match_rate_error,
                    alignment_report.gt_filename_samples,
                    alignment_report.pred_filename_samples,
                )
            )

        if alignment_report.match_rate < self.min_match_rate_warning:
            warnings.warn(
                "Low alignment rate: {:.1%} of GT tiles matched "
                "(threshold: {:.1%}). {} GT tiles have no predictions. "
                "This may indicate a pipeline issue.".format(
                    alignment_report.match_rate,
                    self.min_match_rate_warning,
                    alignment_report.num_gt_without_pred,
                ),
                LowMatchRateWarning,
            )

        if min_gt_coverage > 0:
            truth_coco, preds_coco, filter_report = self._filter_by_min_gt_coverage(
                truth_coco=truth_coco,
                preds_coco=preds_coco,
                min_gt_coverage=min_gt_coverage,
            )
            if self.verbose:
                print("\n" + "=" * 70)
                print("GT COVERAGE FILTER")
                print("=" * 70)
                print("min_gt_coverage: {:.3%}".format(min_gt_coverage))
                print("Images kept: {} / {}".format(
                    filter_report['num_images_kept'],
                    filter_report['num_images_before'],
                ))

        cls_metrics = self._compute_classification_metrics(truth_coco, preds_coco)
        cls_metrics['num_images'] = len(truth_coco.getImgIds())
        cls_metrics['num_truths'] = len(truth_coco.getAnnIds())
        cls_metrics['num_preds'] = len(preds_coco.getAnnIds())

        if processed_preds_path != preds_coco_path:
            Path(processed_preds_path).unlink()

        return cls_metrics

    def _filter_by_min_gt_coverage(self, truth_coco: COCO, preds_coco: COCO,
                                   min_gt_coverage: float):
        gt_images = truth_coco.dataset.get('images', [])
        gt_anns = truth_coco.dataset.get('annotations', [])
        pred_images = preds_coco.dataset.get('images', [])
        pred_anns = preds_coco.dataset.get('annotations', [])

        num_images_before = len(gt_images)
        num_gt_anns_before = len(gt_anns)
        num_pred_anns_before = len(pred_anns)

        gt_area_by_img = {}
        for ann in gt_anns:
            img_id = ann.get('image_id')
            if img_id is None:
                continue

            area = ann.get('area')
            if area is None:
                seg = ann.get('segmentation')
                if isinstance(seg, dict):
                    try:
                        area = float(mask_util.area(seg))
                    except (TypeError, ValueError):
                        area = 0.0
                else:
                    area = 0.0

            gt_area_by_img[img_id] = gt_area_by_img.get(img_id, 0.0) + float(area)

        kept_img_ids = set()
        new_gt_images = []
        for img in gt_images:
            img_id = img.get('id')
            w = float(img.get('width', 0) or 0)
            h = float(img.get('height', 0) or 0)
            denom = w * h
            if denom <= 0:
                continue

            cov = gt_area_by_img.get(img_id, 0.0) / denom
            if cov >= min_gt_coverage:
                kept_img_ids.add(img_id)
                new_gt_images.append(img)

        new_gt_anns = [ann for ann in gt_anns if ann.get('image_id') in kept_img_ids]
        new_pred_images = [img for img in pred_images if img.get('id') in kept_img_ids]
        new_pred_anns = [ann for ann in pred_anns if ann.get('image_id') in kept_img_ids]

        truth_coco.dataset['images'] = new_gt_images
        truth_coco.dataset['annotations'] = new_gt_anns
        truth_coco.createIndex()

        preds_coco.dataset['images'] = new_pred_images
        preds_coco.dataset['annotations'] = new_pred_anns
        preds_coco.createIndex()

        report = {
            'num_images_before': num_images_before,
            'num_images_kept': len(new_gt_images),
            'num_images_dropped': num_images_before - len(new_gt_images),
            'num_gt_anns_before': num_gt_anns_before,
            'num_gt_anns_kept': len(new_gt_anns),
            'num_pred_anns_before': num_pred_anns_before,
            'num_pred_anns_kept': len(new_pred_anns),
        }

        return truth_coco, preds_coco, report

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

        if hasattr(coco_evaluator, 'ious') and coco_evaluator.ious:
            metrics['mIoU'] = self._compute_miou(coco_evaluator)

        return metrics

    def _compute_miou(self, coco_evaluator) -> float:
        try:
            all_ious = []
            if hasattr(coco_evaluator, 'ious') and coco_evaluator.ious:
                for _, iou_matrix in coco_evaluator.ious.items():
                    if isinstance(iou_matrix, np.ndarray):
                        flat_ious = iou_matrix.flatten()
                        valid = flat_ious[(flat_ious > 0) & (flat_ious <= 1)]
                        all_ious.extend(valid.tolist())

            if all_ious:
                return float(np.mean(all_ious))
            return 0.0
        except ValueError as e:
            print(f"Error computing mIoU: {e}")
            return 0.0

    def _preprocess_predictions_coco(
            self,
            preds_coco_path: str,
            score_combination: Optional[Dict] = None,
    ) -> str:
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
        with open(preds_coco_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        score_field_candidates = [
            'classifier_score', 'aggregator_score', 'detector_score', 
            'segmentation_score', 'prediction_score'
        ]

        # Check what score fields exist in annotations
        existing_score_fields = set()
        for ann in coco_data.get('annotations', []):
            if 'score' in ann:
                existing_score_fields.add('score')
            
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
                print("Converting '{}' to 'score' in predictions".format(
                    score_field_to_use))
                self._move_score_to_top_level(coco_data, score_field_to_use)
            else:
                print("WARNING: No score fields found in predictions. "
                      "Adding default score=1.0")
                # Add default score to all annotations
                for ann in coco_data.get('annotations', []):
                    if 'score' not in ann:
                        ann['score'] = 1.0
        else:
            print("Predictions already have 'score' field, "
                  "no preprocessing needed")
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
            for score_name in weights:
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

    def _evaluate_class_agnostic_segmentation(self, truth_coco: COCO,
                                               preds_coco: COCO,
                                               max_dets: list) -> Dict:
        """
        Evaluate segmentation ignoring category_id.
        All annotations treated as same class.
        """
        # Create copies and force all to same category
        # Use deepcopy to handle binary RLE masks properly
        truth_copy = COCO()
        truth_copy.dataset = copy.deepcopy(truth_coco.dataset)
        pred_copy = COCO()
        pred_copy.dataset = copy.deepcopy(preds_coco.dataset)

        # Force all categories to ID 1
        truth_copy.dataset['categories'] = [{
            'id': 1, 'name': 'object', 'supercategory': ''
        }]
        pred_copy.dataset['categories'] = [{
            'id': 1, 'name': 'object', 'supercategory': ''
        }]

        # Change all annotation categories to 1
        for ann in truth_copy.dataset.get('annotations', []):
            ann['category_id'] = 1
        for ann in pred_copy.dataset.get('annotations', []):
            ann['category_id'] = 1

        truth_copy.createIndex()
        pred_copy.createIndex()

        # Run evaluation
        coco_eval = Summarize2COCOEval(
            cocoGt=truth_copy,
            cocoDt=pred_copy,
            iouType='segm'
        )
        coco_eval.params.maxDets = max_dets
        coco_eval.evaluate()
        coco_eval.accumulate()

        metrics = coco_eval.summarize_to_dict()
        metrics['num_images'] = len(truth_copy.getImgIds())
        metrics['num_truths'] = len(truth_copy.getAnnIds())
        metrics['num_preds'] = len(pred_copy.getAnnIds())

        if hasattr(coco_eval, 'ious') and coco_eval.ious:
            metrics['mIoU'] = self._compute_miou(coco_eval)

        return metrics

    def _compute_classification_metrics(self, truth_coco: COCO,
                                        preds_coco: COCO) -> Dict:
        """
        Compute classification accuracy for matched instances.
        Matches instances by IoU > 0.5, then checks if category_id matches.
        """
        total_matches = 0
        correct_class = 0
        per_class_stats = {}

        # For each image
        for img_id in truth_coco.getImgIds():
            gt_anns = truth_coco.loadAnns(
                truth_coco.getAnnIds(imgIds=img_id))
            pred_anns = preds_coco.loadAnns(
                preds_coco.getAnnIds(imgIds=img_id))

            if not gt_anns or not pred_anns:
                continue

            # Match GT to predictions by IoU
            for gt_ann in gt_anns:
                gt_cat = gt_ann['category_id']
                if gt_cat not in per_class_stats:
                    per_class_stats[gt_cat] = {
                        'total': 0, 'correct': 0
                    }

                # Find best matching prediction
                best_iou = 0.0
                best_pred = None

                for pred_ann in pred_anns:
                    # Compute bbox IoU as proxy
                    iou = self._compute_bbox_iou(
                        gt_ann['bbox'], pred_ann['bbox'])

                    if iou > best_iou:
                        best_iou = iou
                        best_pred = pred_ann

                # If matched with IoU > 0.5
                if best_iou > 0.5 and best_pred is not None:
                    total_matches += 1
                    per_class_stats[gt_cat]['total'] += 1

                    if best_pred['category_id'] == gt_cat:
                        correct_class += 1
                        per_class_stats[gt_cat]['correct'] += 1

        accuracy = correct_class / total_matches if total_matches > 0 else 0.0

        # Compute per-class accuracy
        per_class_acc = {}
        for cat_id, stats in per_class_stats.items():
            if stats['total'] > 0:
                per_class_acc[cat_id] = stats['correct'] / stats['total']
            else:
                per_class_acc[cat_id] = 0.0

        return {
            'overall_accuracy': accuracy,
            'total_matched_instances': total_matches,
            'correctly_classified': correct_class,
            'per_class_accuracy': per_class_acc,
            'per_class_counts': per_class_stats
        }

    def _compute_bbox_iou(self, box1, box2):
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

        return intersection / union if union > 0 else 0.0

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

    def diagnose_alignment(self, preds_coco_path: str,
                           truth_coco_path: str) -> AlignmentReport:
        """
        Diagnose alignment between GT and predictions without evaluation.

        Args:
            preds_coco_path: Path to predictions COCO file
            truth_coco_path: Path to ground truth COCO file

        Returns:
            AlignmentReport with detailed diagnostics
        """
        print("\n" + "="*70)
        print("ALIGNMENT DIAGNOSTIC MODE")
        print("="*70)
        print("GT COCO: {}".format(truth_coco_path))
        print("Pred COCO: {}".format(preds_coco_path))

        # Load COCO files
        truth_coco = COCO(str(truth_coco_path))
        preds_coco = COCO(str(preds_coco_path))

        # Check categories
        gt_cats = truth_coco.dataset.get('categories', [])
        pred_cats = preds_coco.dataset.get('categories', [])

        print("\nCategory Analysis:")
        print("  GT categories: {}".format(
            [(c['id'], c['name']) for c in gt_cats]))
        print("  Pred categories: {}".format(
            [(c['id'], c['name']) for c in pred_cats]))

        if gt_cats == pred_cats:
            print("  Categories match")
        else:
            print("  WARNING: Category mismatch detected")

        # Perform alignment (on copies to avoid modifying originals)
        preds_coco_copy = COCO()
        preds_coco_copy.dataset = preds_coco.dataset.copy()
        preds_coco_copy.createIndex()

        report = align_coco_datasets_by_name(
            truth_coco,
            preds_coco_copy,
            alignment_strategy=self.alignment_strategy,
            return_report=True,
        )
        report.print_report(verbose=True)

        # Provide recommendations
        print("Recommendations:")
        if report.match_rate >= 0.98:
            print("  Alignment looks good!")
        elif report.match_rate >= 0.80:
            print("  Match rate is acceptable but could be improved.")
            print("  Check if some tiles were not generated in predictions.")
        else:
            print("  Match rate is poor. Possible issues:")
            print("  - Different tiling strategies (grid vs polygon)")
            print("  - Different tile naming conventions")
            print("  - GT and predictions from different rasters")
            print("  Try: AlignmentStrategy.EXACT_MATCH or regenerate "
                  "predictions")

        print("="*70 + "\n")
        return report
