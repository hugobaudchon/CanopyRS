#!/usr/bin/env python
"""Smoke tests for the classifier benchmarker refactoring.

Run from the repo root with the rubisco_models conda env active:

    conda activate rubisco_models
    python tests/engine/test_refactor_smoke.py

All tests are import / introspection checks — no data files needed.
"""
import sys
import inspect
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

passed = 0
failed = 0


def check(label, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {label}")
    else:
        failed += 1
        msg = f"  FAIL  {label}"
        if detail:
            msg += f"  ({detail})"
        print(msg)


# ------------------------------------------------------------------
# 1. Imports
# ------------------------------------------------------------------
print("\n=== 1. Imports ===")

try:
    from canopyrs.engine.benchmark.base.evaluator import (
        CocoEvaluator,
        AlignmentStrategy,
        AlignmentReport,
        AlignmentError,
        LowMatchRateWarning,
        align_coco_datasets_by_name,
        Summarize2COCOEval,
    )
    check("base evaluator imports", True)
except Exception as exc:
    check("base evaluator imports", False, str(exc))

try:
    from canopyrs.engine.benchmark.classifier.evaluator import (
        ClassifierCocoEvaluator,
    )
    check("classifier evaluator import", True)
except Exception as exc:
    check("classifier evaluator import", False, str(exc))

try:
    from canopyrs.engine.benchmark.classifier.benchmark import (
        ClassifierBenchmarker,
    )
    check("classifier benchmarker import", True)
except Exception as exc:
    check("classifier benchmarker import", False, str(exc))

try:
    from canopyrs.engine.benchmark import (
        ClassifierBenchmarker as CB2,
    )
    check("top-level ClassifierBenchmarker re-export", True)
except Exception as exc:
    check("top-level ClassifierBenchmarker re-export", False, str(exc))

# ------------------------------------------------------------------
# 2. Inheritance
# ------------------------------------------------------------------
print("\n=== 2. Inheritance ===")

check(
    "ClassifierCocoEvaluator inherits CocoEvaluator",
    issubclass(ClassifierCocoEvaluator, CocoEvaluator),
)

# ------------------------------------------------------------------
# 3. Base evaluator has new methods
# ------------------------------------------------------------------
print("\n=== 3. Base evaluator new methods ===")

e = CocoEvaluator()
for method_name in (
    "_evaluate_coco",
    "_compute_miou",
    "evaluate_class_agnostic",
    "compute_classification_metrics",
    "_compute_bbox_iou",
):
    check(
        f"CocoEvaluator.{method_name} exists",
        hasattr(e, method_name),
    )

# ------------------------------------------------------------------
# 4. Removed methods are gone from classifier
# ------------------------------------------------------------------
print("\n=== 4. Removed methods gone from classifier ===")

ce = ClassifierCocoEvaluator()

# These were classifier-only methods that should be fully deleted
for method_name in (
    "_evaluate_single_iou_type",
    "_evaluate_class_agnostic_segmentation",
    "_compute_classification_metrics",
    "_filter_by_min_gt_coverage",
    "_combine_scores",
    "_weighted_arithmetic_mean",
    "_weighted_geometric_mean",
):
    check(
        f"ClassifierCocoEvaluator.{method_name} removed",
        not hasattr(ce, method_name),
        "still exists" if hasattr(ce, method_name) else "",
    )

# These moved to CocoEvaluator and are now inherited (should exist)
for method_name in (
    "_compute_miou",
    "_compute_bbox_iou",
):
    check(
        f"ClassifierCocoEvaluator.{method_name} inherited from base",
        hasattr(ce, method_name),
    )

# ------------------------------------------------------------------
# 5. Signatures are clean (no score_combination / min_gt_coverage)
# ------------------------------------------------------------------
print("\n=== 5. Signature checks ===")

for method_name in ("tile_level", "classification_only"):
    sig = inspect.signature(getattr(ce, method_name))
    for removed_param in ("score_combination", "min_gt_coverage"):
        check(
            f"ClassifierCocoEvaluator.{method_name} "
            f"has no '{removed_param}'",
            removed_param not in sig.parameters,
        )

# ------------------------------------------------------------------
# 6. align_coco_datasets_by_name has new params
# ------------------------------------------------------------------
print("\n=== 6. align_coco_datasets_by_name new params ===")

sig_align = inspect.signature(align_coco_datasets_by_name)
for param in (
    "min_match_rate_warning",
    "min_match_rate_error",
    "verbose",
):
    check(
        f"align_coco_datasets_by_name has '{param}'",
        param in sig_align.parameters,
    )

# ------------------------------------------------------------------
# 7. pipeline_outputs_root is a ClassifierBenchmarker constructor arg
#    (run-level path, not a ClassifierConfig field)
# ------------------------------------------------------------------
print("\n=== 7. ClassifierBenchmarker pipeline_outputs_root ===")

try:
    import inspect
    from canopyrs.engine.config_parsers.classifier import (
        ClassifierConfig,
    )
    from canopyrs.engine.benchmark.classifier.benchmark import (
        ClassifierBenchmarker,
    )

    check(
        "ClassifierConfig.pipeline_outputs_root removed",
        "pipeline_outputs_root" not in ClassifierConfig.__fields__,
    )
    init_params = inspect.signature(
        ClassifierBenchmarker.__init__
    ).parameters
    check(
        "ClassifierBenchmarker.__init__ accepts pipeline_outputs_root",
        "pipeline_outputs_root" in init_params,
    )
except Exception as exc:
    check("ClassifierBenchmarker import", False, str(exc))

# ------------------------------------------------------------------
# 8. Dead code files are deleted
# ------------------------------------------------------------------
print("\n=== 8. Dead code removed ===")

classifier_dir = (
    Path(REPO_ROOT)
    / "canopyrs"
    / "engine"
    / "benchmark"
    / "classifier"
)
for dead_file in (
    "evaluator_helpers.py",
    "find_optimal_parameters.py",
):
    check(
        f"{dead_file} deleted",
        not (classifier_dir / dead_file).exists(),
    )

test_dir = Path(REPO_ROOT) / "test"
check(
    "test_score_combination.py deleted",
    not (test_dir / "test_score_combination.py").exists(),
)

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print("\n" + "=" * 50)
total = passed + failed
print(f"RESULTS: {passed}/{total} passed, {failed} failed")
if failed:
    print("SOME CHECKS FAILED")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
    sys.exit(0)
