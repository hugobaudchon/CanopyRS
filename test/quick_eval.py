#!/usr/bin/env python3
"""Quick evaluation script for the classifier benchmark.

This script supports two modes:
1) Direct COCO evaluation (pred COCO vs GT COCO)
2) Running the classifier benchmarker for a single product, using a
   pre-existing pipeline output folder to resolve tiles/input_coco/input_gpkg.
"""

import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from canopyrs.engine.benchmark.classifier.benchmark import (
        ClassifierBenchmarker,
    )
    from canopyrs.engine.benchmark.classifier.evaluator import (
        ClassifierCocoEvaluator,
    )
    from canopyrs.engine.config_parsers import ClassifierConfig, TilerizerConfig

    # ========================================================================
    # CONFIGURE YOUR PATHS HERE
    # ========================================================================
    RUN_WITH_BENCHMARKER = False

    TARGET_PRODUCT_NAME = "20210902_sblz3_p4rtk_rgb"

    RAW_DATA_ROOT = (
        "/home/mila/a/arthur.ouaknine/scratch/data/"
    )
    PIPELINE_OUTPUTS_ROOT = (
        "/network/scratch/a/arthur.ouaknine/temp/"
        "inference_pipeline_outputs_z3_jan26"
    )
    OUTPUT_FOLDER = (
        "/network/scratch/a/arthur.ouaknine/temp/"
        "benchmark_outputs_quebec_trees"
    )

    CLASSIFIER_YAML = (
        "/home/mila/a/arthur.ouaknine/code/CanopyRS/"
        "config/pipeline_w_classification/classifier.yaml"
    )
    TILERIZER_YAML = (
        "/home/mila/a/arthur.ouaknine/code/CanopyRS/"
        "config/pipeline_w_classification/"
        "tilerizer_classifier.yaml"
    )
    INPUT_IMAGERY = (
        "/network/scratch/a/arthur.ouaknine/data/"
        "quebec_trees/rasters/"
        "2021-09-02-sbl-z3-rgb-cog.tif"
    )

    PREDICTIONS_COCO = (
        "/network/scratch/a/arthur.ouaknine/temp/"
        "quebec_trees/quebectree_z3_predicted_jan26/"
        "2021-09-02-sbl-z3-rgb-cog/"
        "2021_09_02_sbl_z3_rgb_cog/"
        "2021_09_02_sbl_z3_rgb_cog_coco_sf0p8_test.json"
    )
    GROUND_TRUTH_COCO = (
        "/network/scratch/a/arthur.ouaknine/temp/"
        "quebec_trees/20210902_sblz3_p4rtk_rgb/"
        "20210902_sblz3_p4rtk_rgb/"
        "2021-09-02-sbl-z3-rgb-cog/"
        "2021_09_02_sbl_z3_rgb_cog/"
        "2021_09_02_sbl_z3_rgb_cog_coco_sf0p8_test.json"
    )

    # Evaluation options
    EVALUATE_CLASS_AGNOSTIC = True  # Full diagnostic
    EVALUATE_BBOX = False           # Also check bbox metrics
    # ========================================================================
    # END CONFIGURATION
    # ========================================================================
    if RUN_WITH_BENCHMARKER:
        if not Path(RAW_DATA_ROOT).exists():
            print(f"ERROR: RAW_DATA_ROOT not found: {RAW_DATA_ROOT}")
            return

        if not Path(PIPELINE_OUTPUTS_ROOT).exists():
            print(
                "ERROR: PIPELINE_OUTPUTS_ROOT not found: "
                f"{PIPELINE_OUTPUTS_ROOT}"
            )
            return

        if not Path(CLASSIFIER_YAML).exists():
            print(f"ERROR: CLASSIFIER_YAML not found: {CLASSIFIER_YAML}")
            return
    else:
        if not Path(PREDICTIONS_COCO).exists():
            print(f"ERROR: Predictions file not found: {PREDICTIONS_COCO}")
            print("Please edit the PREDICTIONS_COCO path in this script.")
            return

        if not Path(GROUND_TRUTH_COCO).exists():
            print(f"ERROR: Ground truth file not found: {GROUND_TRUTH_COCO}")
            print("Please edit the GROUND_TRUTH_COCO path in this script.")
            return
    print("\n" + "=" * 70)
    if RUN_WITH_BENCHMARKER:
        print("QUICK CLASSIFIER BENCHMARK (single product)")
        print("=" * 70)
        print(f"Target product: {TARGET_PRODUCT_NAME}")
        print(f"RAW_DATA_ROOT: {RAW_DATA_ROOT}")
        print(f"PIPELINE_OUTPUTS_ROOT: {PIPELINE_OUTPUTS_ROOT}")
        print(f"OUTPUT_FOLDER: {OUTPUT_FOLDER}")
        print("=" * 70 + "\n")
    else:
        print("QUICK COCO EVALUATION")
        print("=" * 70)
        print(f"Predictions: {PREDICTIONS_COCO}")
        print(f"Ground Truth: {GROUND_TRUTH_COCO}")
        print(f"Class-agnostic eval: {EVALUATE_CLASS_AGNOSTIC}")
        print("=" * 70 + "\n")
    evaluator = ClassifierCocoEvaluator(verbose=True)

    if RUN_WITH_BENCHMARKER:
        classifier_config = ClassifierConfig.from_yaml(path=CLASSIFIER_YAML)
        classifier_config.pipeline_outputs_root = PIPELINE_OUTPUTS_ROOT

        tilerizer_config = None
        if TILERIZER_YAML and Path(TILERIZER_YAML).exists():
            tilerizer_config = TilerizerConfig.from_yaml(
                path=TILERIZER_YAML,
            )
            print(f"Tilerizer config loaded: {TILERIZER_YAML}")

        input_imagery = INPUT_IMAGERY if INPUT_IMAGERY else None

        benchmarker = ClassifierBenchmarker(
            output_folder=OUTPUT_FOLDER,
            fold_name="test",
            raw_data_root=RAW_DATA_ROOT,
        )

        datasets = benchmarker._get_preprocessed_datasets(
            ["QuebecTrees"],
            pipeline_outputs_root=PIPELINE_OUTPUTS_ROOT,
        )
        dataset = datasets["QuebecTrees"]

        matched = False
        for (
            _location,
            product_name,
            tiles_path,
            input_gpkg,
            input_coco,
            truths_coco,
        ) in dataset.iter_fold_classifier(
            root_output_path=RAW_DATA_ROOT,
            fold="test",
        ):
            if product_name != TARGET_PRODUCT_NAME:
                continue
            matched = True

            run_output_folder = (
                Path(OUTPUT_FOLDER)
                / benchmarker.fold_name
                / product_name
            )

            preds_coco_path, _preds_gpkg_path, tilerizer_coco = (
                benchmarker._infer_classifier_single_product(
                    product_name=product_name,
                    product_tiles_path=tiles_path,
                    classifier_config=classifier_config,
                    input_gpkg=input_gpkg,
                    input_coco=input_coco,
                    tilerizer_config=tilerizer_config,
                    input_imagery=input_imagery,
                    output_folder=run_output_folder,
                )
            )

            # When tilerizer ran, its output COCO (with GT
            # labels) serves as ground truth for alignment.
            effective_truth = (
                str(tilerizer_coco)
                if tilerizer_coco is not None
                else str(truths_coco)
            )

            results = evaluator.tile_level(
                preds_coco_path=str(preds_coco_path),
                truth_coco_path=effective_truth,
                evaluate_class_agnostic=EVALUATE_CLASS_AGNOSTIC,
                evaluate_bbox=EVALUATE_BBOX,
            )
            break

        if not matched:
            print(
                "ERROR: TARGET_PRODUCT_NAME not found in dataset fold: "
                f"{TARGET_PRODUCT_NAME}"
            )
            return
    else:
        results = evaluator.tile_level(
            preds_coco_path=PREDICTIONS_COCO,
            truth_coco_path=GROUND_TRUTH_COCO,
            evaluate_class_agnostic=EVALUATE_CLASS_AGNOSTIC,
            evaluate_bbox=EVALUATE_BBOX,
        )
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    # Standard metrics
    segm = results['segm'] if 'segm' in results else results
    print("\n### STANDARD METRICS (Segmentation + Classification)")
    print(f"  mAP (IoU=0.50:0.95): {segm['AP']:.1%}")
    print(f"  mAP@0.50:            {segm['AP50']:.1%}")
    print(f"  mAP@0.75:            {segm['AP75']:.1%}")
    print(f"  mAR:                 {segm['AR']:.1%}")
    print(f"  mAR@0.50:            {segm['AR50']:.1%}")
    print(f"  mIoU:                {segm.get('mIoU', 0.0):.3f}")
    print(f"\n  Images evaluated:    {segm['num_images']}")
    print(f"  Ground truth anns:   {segm['num_truths']}")
    print(f"  Prediction anns:     {segm['num_preds']}")
    # Class-agnostic metrics
    if EVALUATE_CLASS_AGNOSTIC and 'class_agnostic_segm' in results:
        ca = results['class_agnostic_segm']
        print("\n### CLASS-AGNOSTIC METRICS (Segmentation Only)")
        print(f"  mAP (IoU=0.50:0.95): {ca['AP']:.1%}")
        print(f"  mAP@0.50:            {ca['AP50']:.1%}")
        print(f"  mAP@0.75:            {ca['AP75']:.1%}")
        print(f"  mAR:                 {ca['AR']:.1%}")
        print(f"  mIoU:                {ca.get('mIoU', 0.0):.3f}")
    # Classification metrics
    if EVALUATE_CLASS_AGNOSTIC and 'classification' in results:
        cls = results['classification']
        print("\n### CLASSIFICATION METRICS (For Matched Instances)")
        print(f"  Overall accuracy:    {cls['overall_accuracy']:.1%}")
        print(
            "  Correctly classified: "
            f"{cls['correctly_classified']}/{cls['total_matched_instances']}"
        )

        # Top 5 classes by count
        if cls['per_class_counts']:
            print("\n  Top classes by count:")
            counts = [
                (cat_id, stats['total'])
                for cat_id, stats in cls['per_class_counts'].items()
            ]
            counts.sort(key=lambda x: x[1], reverse=True)
            for cat_id, count in counts[:5]:
                acc = cls['per_class_accuracy'].get(cat_id, 0.0)
                print(
                    f"    Category {cat_id:2d}: {count:4d} instances, "
                    f"{acc:.1%} accuracy"
                )

    # Bbox metrics
    if EVALUATE_BBOX and 'bbox' in results:
        bbox = results['bbox']
        print("\n### BBOX METRICS")
        print(f"  mAP@0.50:            {bbox['AP50']:.1%}")
        print(f"  mAP@0.75:            {bbox['AP75']:.1%}")
    # Diagnostic interpretation
    if (
        EVALUATE_CLASS_AGNOSTIC
        and 'class_agnostic_segm' in results
        and 'classification' in results
    ):
        std_map = segm['AP50']
        ca_map = results['class_agnostic_segm']['AP50']
        cls_acc = results['classification']['overall_accuracy']
        print("\n" + "=" * 70)
        print("DIAGNOSTIC INTERPRETATION")
        print("=" * 70)

        if ca_map > 0.70 and cls_acc < 0.30:
            print("\n✓ SEGMENTATION IS GOOD, CLASSIFICATION IS BAD")
            print(f"  Segmentation quality: {ca_map:.1%} (good)")
            print(f"  Classification accuracy: {cls_acc:.1%} (poor)")
            print("\n  → Focus on improving the classifier")
        elif ca_map < 0.30:
            print("\n✗ SEGMENTATION IS POOR")
            print(f"  Class-agnostic mAP@0.50: {ca_map:.1%}")
            print(f"  Classification accuracy: {cls_acc:.1%}")
            print("\n  → Primary issue is segmentation/localization")
            print("  → Check model weights, thresholds, mask quality")
        elif ca_map > 0.50 and cls_acc > 0.50:
            print("\n⚠ BOTH ARE MODERATE")
            print(f"  Segmentation quality: {ca_map:.1%}")
            print(f"  Classification accuracy: {cls_acc:.1%}")
            print("\n  → Both components need improvement")
        else:
            print(f"\n  Standard mAP@0.50: {std_map:.1%}")
            print(f"  Class-agnostic mAP@0.50: {ca_map:.1%}")
            print(f"  Classification accuracy: {cls_acc:.1%}")
    print("\n" + "=" * 70)
    print("Alignment report available in evaluator.last_alignment_report")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
