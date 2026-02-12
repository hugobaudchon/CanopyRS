# engine/benchmark/classifier/evaluator.py
import warnings
import json
import tempfile
from pathlib import Path
from typing import Optional, Dict

from faster_coco_eval.core.coco import COCO

from canopyrs.engine.benchmark.base.evaluator import (
    AlignmentStrategy,
    AlignmentReport,
    AlignmentError,
    LowMatchRateWarning,
    CocoEvaluator,
    align_coco_datasets_by_name,
)


class ClassifierCocoEvaluator(CocoEvaluator):
    """Classifier-specific evaluator.

    Inherits mIoU, class-agnostic evaluation, and
    classification metrics from ``CocoEvaluator``.
    Adds score preprocessing and alignment orchestration.
    """

    def __init__(
            self,
            alignment_strategy: AlignmentStrategy = (
                AlignmentStrategy.BASE_RSPLIT_1),
            min_match_rate_warning: float = 0.95,
            min_match_rate_error: float = 0.50,
            verbose: bool = True,
    ):
        super().__init__()
        self.alignment_strategy = alignment_strategy
        self.min_match_rate_warning = min_match_rate_warning
        self.min_match_rate_error = min_match_rate_error
        self.verbose = verbose
        self.last_alignment_report: Optional[
            AlignmentReport] = None

    def tile_level(self,
                   preds_coco_path: str,
                   truth_coco_path: str,
                   max_dets: list = (1, 10, 100),
                   evaluate_bbox: bool = False,
                   evaluate_class_agnostic: bool = False,
                   ) -> dict:
        """
        Tile-level evaluation for instance segmentation
        with classes.

        Args:
            preds_coco_path: Path to predictions COCO file
            truth_coco_path: Path to ground truth COCO file
            max_dets: Maximum detections to consider
            evaluate_bbox: Also evaluate bbox IoU
            evaluate_class_agnostic: Also compute
                class-agnostic segmentation metrics

        Returns:
            dict with 'segm' key (and optionally 'bbox',
                'class_agnostic_segm', 'classification')
        """
        # Preprocess predictions to fix score field names
        processed_preds_path = self._preprocess_predictions_coco(
            preds_coco_path)

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

        results = {}
        # Always evaluate segmentation
        print("Evaluating segmentation IoU...")
        segm_metrics = self._evaluate_coco(
            truth_coco, preds_coco, 'segm', max_dets)
        results['segm'] = segm_metrics

        # Optionally evaluate bbox
        if evaluate_bbox:
            print("Evaluating bbox IoU...")
            bbox_metrics = self._evaluate_coco(
                truth_coco, preds_coco, 'bbox', max_dets)
            results['bbox'] = bbox_metrics

        # Optionally evaluate class-agnostic segmentation
        if evaluate_class_agnostic:
            print("\nEvaluating class-agnostic segmentation...")
            ca_metrics = self.evaluate_class_agnostic(
                truth_coco, preds_coco, max_dets)
            results['class_agnostic_segm'] = ca_metrics

            print("\nComputing classification accuracy...")
            class_metrics = self.compute_classification_metrics(
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

    def classification_only(
            self,
            preds_coco_path: str,
            truth_coco_path: str,
    ) -> dict:
        processed_preds_path = self._preprocess_predictions_coco(
            preds_coco_path)

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

        cls_metrics = self.compute_classification_metrics(
            truth_coco, preds_coco)
        cls_metrics['num_images'] = len(truth_coco.getImgIds())
        cls_metrics['num_truths'] = len(truth_coco.getAnnIds())
        cls_metrics['num_preds'] = len(preds_coco.getAnnIds())

        if processed_preds_path != preds_coco_path:
            Path(processed_preds_path).unlink()

        return cls_metrics

    def _preprocess_predictions_coco(
            self,
            preds_coco_path: str,
    ) -> str:
        """Standardise the ``score`` field in a predictions COCO.

        If both ``segmentation_score`` and ``classifier_score``
        are present (under ``other_attributes``), the final
        ``score`` is set to their arithmetic mean.  Otherwise
        the best available score field is promoted.

        Returns
        -------
        str
            Path to the (possibly temporary) processed file.
        """
        with open(preds_coco_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        score_field_candidates = [
            'classifier_score', 'aggregator_score',
            'detector_score', 'segmentation_score',
            'prediction_score',
        ]

        existing_score_fields = set()
        for ann in coco_data.get('annotations', []):
            if 'score' in ann:
                existing_score_fields.add('score')
            other_attrs = ann.get('other_attributes', {})
            for field in score_field_candidates:
                if field in other_attrs:
                    existing_score_fields.add(field)

        # --- simple average when both scores are available ---
        seg_key = 'segmentation_score'
        cls_key = 'classifier_score'
        if (
            seg_key in existing_score_fields
            and cls_key in existing_score_fields
        ):
            print(
                "Averaging {} and {} -> score".format(
                    seg_key, cls_key))
            for ann in coco_data.get('annotations', []):
                oa = ann.get('other_attributes', {})
                s = oa.get(seg_key, 0.0)
                c = oa.get(cls_key, 0.0)
                ann['score'] = (s + c) / 2.0

        # --- fallback: promote a single score field ----------
        elif not (
            'score' in existing_score_fields
            and len(existing_score_fields) == 1
        ):
            field = self._select_primary_score_field(
                existing_score_fields)
            if field:
                print("Promoting '{}' -> 'score'".format(
                    field))
                self._move_score_to_top_level(
                    coco_data, field)
            else:
                print(
                    "WARNING: No score fields found. "
                    "Adding default score=1.0")
                for ann in coco_data.get(
                        'annotations', []):
                    if 'score' not in ann:
                        ann['score'] = 1.0
        else:
            print(
                "Predictions already have 'score', "
                "no preprocessing needed")
            return preds_coco_path

        tmp = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='_fixed_predictions.json',
            delete=False,
        )
        with tmp as f:
            json.dump(coco_data, f, indent=2)
        return tmp.name

    def _select_primary_score_field(
            self, existing_score_fields: set,
    ) -> Optional[str]:
        """Pick the best available score field."""
        for candidate in (
            'classifier_score',
            'aggregator_score',
            'detector_score',
        ):
            if candidate in existing_score_fields:
                return candidate
        others = existing_score_fields - {'score'}
        return next(iter(others), None)

    def _move_score_to_top_level(
            self, coco_data: Dict,
            score_field: str,
    ):
        """Promote a score from other_attributes."""
        for ann in coco_data.get('annotations', []):
            oa = ann.get('other_attributes', {})
            if score_field in oa:
                ann['score'] = oa[score_field]
            elif score_field in ann:
                ann['score'] = ann[score_field]

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
