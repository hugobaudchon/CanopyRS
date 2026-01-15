# engine/benchmark/classifier/evaluator.py
import warnings
import json
import tempfile
from pathlib import Path
import numpy as np
from faster_coco_eval.core.coco import COCO
from faster_coco_eval.core.faster_eval_api import COCOeval_faster
from typing import Optional, Dict, List
from dataclasses import dataclass, field as dataclass_field
from enum import Enum


class AlignmentStrategy(Enum):
    """Strategy for aligning prediction and ground truth COCO datasets"""
    BASE_RSPLIT_1 = "base_rsplit_1"  # Current default: rsplit('_', 1)[0]
    EXACT_MATCH = "exact"             # Exact filename matching


class AlignmentError(Exception):
    """Raised when COCO alignment fails critically"""
    pass


class LowMatchRateWarning(UserWarning):
    """Raised when match rate is below acceptable threshold"""
    pass


@dataclass
class AlignmentReport:
    """Report of COCO dataset alignment results"""
    num_gt_images: int
    num_pred_images: int
    num_matched: int
    num_gt_without_pred: int
    num_pred_outside_gt: int
    match_rate: float
    unmatched_gt_files: List[str] = dataclass_field(default_factory=list)
    gt_filename_samples: List[str] = dataclass_field(default_factory=list)
    pred_filename_samples: List[str] = dataclass_field(default_factory=list)
    strategy_used: str = "base_rsplit_1"
    
    def print_report(self, verbose: bool = True):
        """Print formatted alignment report"""
        print("\n" + "="*70)
        print("COCO DATASET ALIGNMENT REPORT")
        print("="*70)
        print("Strategy: {}".format(self.strategy_used))
        print("\nDataset Sizes:")
        print("  Ground Truth images: {}".format(self.num_gt_images))
        print("  Prediction images: {}".format(self.num_pred_images))
        print("\nAlignment Results:")
        print("  Successfully matched: {} ({:.1%} of GT)".format(
            self.num_matched, self.match_rate))
        print("  GT tiles without predictions: {}".format(
            self.num_gt_without_pred))
        print("  Prediction tiles outside GT scope: {}".format(
            self.num_pred_outside_gt))
        print("    (These will be ignored - expected for extended "
              "prediction areas)")

        if self.match_rate < 0.98:
            print("\n  WARNING: Match rate {:.1%} is below 98%".format(
                self.match_rate))
            print("    This may indicate a naming mismatch between GT "
                  "and predictions.")

        if verbose and self.unmatched_gt_files:
            print("\nUnmatched GT tiles (first 10):")
            for filename in self.unmatched_gt_files[:10]:
                print("    - {}".format(filename))
            if len(self.unmatched_gt_files) > 10:
                print("    ... and {} more".format(
                    len(self.unmatched_gt_files) - 10))

        if verbose and self.gt_filename_samples:
            print("\nSample GT filenames (first 3):")
            for filename in self.gt_filename_samples[:3]:
                print("    - {}".format(filename))

        if verbose and self.pred_filename_samples:
            print("\nSample Pred filenames (first 3):")
            for filename in self.pred_filename_samples[:3]:
                print("    - {}".format(filename))

        print("="*70 + "\n")
    
    def to_dict(self) -> Dict:
        """Convert report to dictionary for serialization"""
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
                   evaluate_class_agnostic: bool = False) -> dict:
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
        alignment_report = self._align_coco_datasets_by_name(truth_coco, preds_coco)
        self.last_alignment_report = alignment_report
        
        # Print alignment report
        if self.verbose:
            alignment_report.print_report(verbose=True)
        
        # Validate alignment quality
        self._validate_alignment(alignment_report)

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

    def _preprocess_predictions_coco(self, preds_coco_path: str, score_combination: Optional[Dict] = None) -> str:
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

    def _align_coco_datasets_by_name(self, truth_coco: COCO,
                                      preds_coco: COCO) -> AlignmentReport:
        """
        Align predictions COCO to truth COCO, matching by filename.

        Args:
            truth_coco: Ground truth COCO dataset
            preds_coco: Predictions COCO dataset (modified in-place)

        Returns:
            AlignmentReport with detailed matching statistics
        """
        # Get strategy-specific filename matcher
        if self.alignment_strategy == AlignmentStrategy.BASE_RSPLIT_1:
            def get_match_key(file_name: str) -> str:
                stem = Path(file_name).stem
                parts = stem.rsplit('_', 1)
                return parts[0] if len(parts) > 1 else stem
        else:  # EXACT_MATCH
            def get_match_key(file_name: str) -> str:
                return file_name

        # Build prediction lookup
        gt_images = truth_coco.dataset.get('images', [])
        pred_images = preds_coco.dataset.get('images', [])

        preds_by_key = {get_match_key(img['file_name']): img
                        for img in pred_images}
        id_mapping = {}
        new_preds_images = []
        unmatched_gt = []

        # Match GT to predictions
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
                # GT without prediction: add dummy image
                new_preds_images.append(truth_img.copy())
                unmatched_gt.append(truth_file)

        # Update prediction dataset
        preds_coco.dataset['images'] = new_preds_images

        # Remap annotation image_ids
        new_preds_annotations = []
        for ann in preds_coco.dataset.get('annotations', []):
            orig_img_id = ann['image_id']
            if orig_img_id in id_mapping:
                ann['image_id'] = id_mapping[orig_img_id]
                new_preds_annotations.append(ann)

        preds_coco.dataset['annotations'] = new_preds_annotations
        preds_coco.createIndex()

        # Build alignment report
        num_gt = len(gt_images)
        num_pred = len(pred_images)
        num_matched = len(id_mapping)
        num_gt_without_pred = len(unmatched_gt)
        num_pred_outside_gt = num_pred - num_matched
        match_rate = num_matched / num_gt if num_gt > 0 else 0.0

        report = AlignmentReport(
            num_gt_images=num_gt,
            num_pred_images=num_pred,
            num_matched=num_matched,
            num_gt_without_pred=num_gt_without_pred,
            num_pred_outside_gt=num_pred_outside_gt,
            match_rate=match_rate,
            unmatched_gt_files=unmatched_gt,
            gt_filename_samples=[img['file_name'] for img in gt_images[:3]],
            pred_filename_samples=[img['file_name']
                                   for img in pred_images[:3]],
            strategy_used=self.alignment_strategy.value
        )

        return report


    def _compute_miou(self, eval_result):
        """Compute mean IoU from COCO evaluation results"""
        try:
            # The 'matched' key is a dictionary where keys are (imgId, catId) and values are arrays of IoU scores.
            # We need to flatten these arrays and filter for valid IoU values.
            all_ious = []
            if 'matched' in eval_result:
                for key, iou_array in eval_result['matched'].items():
                    # Ensure iou_array is a numpy array and contains numbers
                    if isinstance(iou_array, np.ndarray):
                        all_ious.extend(iou_array.tolist())
                    elif isinstance(iou_array, list): # Fallback for list of lists or similar
                        for item in iou_array:
                            if isinstance(item, (int, float)) and item > 0:
                                all_ious.append(item)
            
            # Filter valid IoU values (> 0, typically <= 1)
            valid_ious = np.array([iou for iou in all_ious if iou > 0 and iou <= 1])

            if len(valid_ious) > 0:
                miou = float(np.mean(valid_ious))
                return miou
            
            return 0.0
            
        except Exception as e:
            print(f"Error computing mIoU: {e}")
            return 0.0

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

    def _validate_alignment(self, report: AlignmentReport):
        """Validate alignment quality and raise warnings/errors as needed"""
        if report.match_rate < self.min_match_rate_error:
            raise AlignmentError(
                "Critical alignment failure: only {:.1%} of GT tiles matched "
                "predictions (threshold: {:.1%}).\n"
                "This indicates incompatible tile naming conventions.\n"
                "GT samples: {}\n"
                "Pred samples: {}\n"
                "Consider using AlignmentStrategy.EXACT_MATCH or check your "
                "tiling configuration.".format(
                    report.match_rate,
                    self.min_match_rate_error,
                    report.gt_filename_samples,
                    report.pred_filename_samples
                )
            )

        if report.match_rate < self.min_match_rate_warning:
            warnings.warn(
                "Low alignment rate: {:.1%} of GT tiles matched "
                "(threshold: {:.1%}). {} GT tiles have no predictions. "
                "This may indicate a pipeline issue.".format(
                    report.match_rate,
                    self.min_match_rate_warning,
                    report.num_gt_without_pred
                ),
                LowMatchRateWarning
            )

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

        report = self._align_coco_datasets_by_name(truth_coco,
                                                    preds_coco_copy)
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

    def _evaluate_class_agnostic_segmentation(self, truth_coco: COCO,
                                               preds_coco: COCO,
                                               max_dets: list) -> Dict:
        """
        Evaluate segmentation ignoring category_id.
        All annotations treated as same class.
        """
        # Create copies and force all to same category
        truth_copy = COCO()
        truth_copy.dataset = json.loads(json.dumps(truth_coco.dataset))
        pred_copy = COCO()
        pred_copy.dataset = json.loads(json.dumps(preds_coco.dataset))

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
        coco_eval.summarize()

        metrics = {
            'AP': coco_eval.stats[0],
            'AP50': coco_eval.stats[1],
            'AP75': coco_eval.stats[2],
            'AR': coco_eval.stats[8],
            'num_images': len(truth_copy.getImgIds()),
            'num_truths': len(truth_copy.getAnnIds()),
            'num_preds': len(pred_copy.getAnnIds()),
        }

        if hasattr(coco_eval, 'eval') and coco_eval.eval:
            miou = self._compute_miou(coco_eval.eval)
            metrics['mIoU'] = miou

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
