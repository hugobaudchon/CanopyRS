#!/usr/bin/env python3
"""
Quick COCO Evaluation Script

Simple script to evaluate predictions vs ground truth COCO files.
Just modify the paths below and run.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from engine.benchmark.classifier.evaluator import ClassifierCocoEvaluator


def main():
    # ========================================================================
    # CONFIGURE YOUR PATHS HERE
    # ========================================================================
    
    PREDICTIONS_COCO = "/network/scratch/a/arthur.ouaknine/temp/quebectree_z3_predicted_jan26/2021-09-02-sbl-z3-rgb-cog/2021_09_02_sbl_z3_rgb_cog/2021_09_02_sbl_z3_rgb_cog_coco_sf0p8_test.json"
    GROUND_TRUTH_COCO = "/network/scratch/a/arthur.ouaknine/temp/quebectree_z3_labelled_jan26/2021-09-02-sbl-z3-rgb-cog/2021_09_02_sbl_z3_rgb_cog/2021_09_02_sbl_z3_rgb_cog_coco_sf0p8_test.json"
    
    # Evaluation options
    EVALUATE_CLASS_AGNOSTIC = True  # Set to True for full diagnostic
    EVALUATE_BBOX = False           # Set to True to also check bbox metrics
    
    # ========================================================================
    # END CONFIGURATION
    # ========================================================================
    
    # Validate paths
    if not Path(PREDICTIONS_COCO).exists():
        print(f"ERROR: Predictions file not found: {PREDICTIONS_COCO}")
        print("Please edit the PREDICTIONS_COCO path in this script.")
        return
    
    if not Path(GROUND_TRUTH_COCO).exists():
        print(f"ERROR: Ground truth file not found: {GROUND_TRUTH_COCO}")
        print("Please edit the GROUND_TRUTH_COCO path in this script.")
        return
    
    print("\n" + "="*70)
    print("QUICK COCO EVALUATION")
    print("="*70)
    print(f"Predictions: {PREDICTIONS_COCO}")
    print(f"Ground Truth: {GROUND_TRUTH_COCO}")
    print(f"Class-agnostic eval: {EVALUATE_CLASS_AGNOSTIC}")
    print("="*70 + "\n")
    
    # Run evaluation
    evaluator = ClassifierCocoEvaluator(verbose=True)
    
    results = evaluator.tile_level(
        preds_coco_path=PREDICTIONS_COCO,
        truth_coco_path=GROUND_TRUTH_COCO,
        evaluate_class_agnostic=EVALUATE_CLASS_AGNOSTIC,
        evaluate_bbox=EVALUATE_BBOX
    )
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
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
        print(f"  Correctly classified: {cls['correctly_classified']}/{cls['total_matched_instances']}")
        
        # Top 5 classes by count
        if cls['per_class_counts']:
            print("\n  Top classes by count:")
            counts = [(cat_id, stats['total']) 
                     for cat_id, stats in cls['per_class_counts'].items()]
            counts.sort(key=lambda x: x[1], reverse=True)
            for cat_id, count in counts[:5]:
                acc = cls['per_class_accuracy'].get(cat_id, 0.0)
                print(f"    Category {cat_id:2d}: {count:4d} instances, {acc:.1%} accuracy")
    
    # Bbox metrics
    if EVALUATE_BBOX and 'bbox' in results:
        bbox = results['bbox']
        print("\n### BBOX METRICS")
        print(f"  mAP@0.50:            {bbox['AP50']:.1%}")
        print(f"  mAP@0.75:            {bbox['AP75']:.1%}")
    
    # Diagnostic interpretation
    if EVALUATE_CLASS_AGNOSTIC and 'class_agnostic_segm' in results and 'classification' in results:
        std_map = segm['AP50']
        ca_map = results['class_agnostic_segm']['AP50']
        cls_acc = results['classification']['overall_accuracy']
        
        print("\n" + "="*70)
        print("DIAGNOSTIC INTERPRETATION")
        print("="*70)
        
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
    
    print("\n" + "="*70)
    print("Alignment report available in evaluator.last_alignment_report")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
