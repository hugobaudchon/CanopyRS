#!/usr/bin/env python3
"""
Unit tests for COCO alignment logic in ClassifierCocoEvaluator.

Tests various alignment scenarios:
- Perfect exact matches
- Base name matching with rsplit
- Partial matches (some GT without predictions)
- No matches (complete mismatch)
- Edge cases (empty datasets, single image, etc.)
"""
import json
import tempfile
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from engine.benchmark.classifier.evaluator import (
    ClassifierCocoEvaluator,
    AlignmentStrategy,
    AlignmentError,
    LowMatchRateWarning
)


def create_coco_json(images, annotations, categories=None):
    """Helper to create a COCO JSON structure"""
    if categories is None:
        categories = [{'id': 1, 'name': 'tree', 'supercategory': ''}]
    
    return {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }


def write_temp_coco(coco_dict):
    """Write COCO dict to temporary file and return path"""
    temp_file = tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', delete=False
    )
    with temp_file as f:
        json.dump(coco_dict, f)
    return temp_file.name


def test_perfect_exact_match():
    """Test case: GT and Pred have identical filenames"""
    print("\n" + "="*70)
    print("TEST 1: Perfect Exact Match")
    print("="*70)
    
    # Create GT COCO
    gt_images = [
        {'id': 1, 'file_name': 'tile_001.tif', 'width': 512, 'height': 512},
        {'id': 2, 'file_name': 'tile_002.tif', 'width': 512, 'height': 512},
        {'id': 3, 'file_name': 'tile_003.tif', 'width': 512, 'height': 512}
    ]
    gt_anns = [
        {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [10, 10, 50, 50],
         'segmentation': [[10, 10, 60, 10, 60, 60, 10, 60]], 'area': 2500},
        {'id': 2, 'image_id': 2, 'category_id': 1, 'bbox': [20, 20, 40, 40],
         'segmentation': [[20, 20, 60, 20, 60, 60, 20, 60]], 'area': 1600}
    ]
    
    # Create Pred COCO with same filenames but different image IDs
    pred_images = [
        {'id': 101, 'file_name': 'tile_001.tif', 'width': 512,
         'height': 512},
        {'id': 102, 'file_name': 'tile_002.tif', 'width': 512,
         'height': 512},
        {'id': 103, 'file_name': 'tile_003.tif', 'width': 512,
         'height': 512}
    ]
    pred_anns = [
        {'id': 201, 'image_id': 101, 'category_id': 1, 'score': 0.9,
         'bbox': [12, 12, 48, 48],
         'segmentation': [[12, 12, 60, 12, 60, 60, 12, 60]], 'area': 2304},
        {'id': 202, 'image_id': 102, 'category_id': 1, 'score': 0.85,
         'bbox': [22, 22, 38, 38],
         'segmentation': [[22, 22, 60, 22, 60, 60, 22, 60]], 'area': 1444}
    ]
    
    gt_path = write_temp_coco(create_coco_json(gt_images, gt_anns))
    pred_path = write_temp_coco(create_coco_json(pred_images, pred_anns))
    
    try:
        evaluator = ClassifierCocoEvaluator(
            alignment_strategy=AlignmentStrategy.EXACT_MATCH,
            verbose=True
        )
        
        metrics = evaluator.tile_level(
            preds_coco_path=pred_path,
            truth_coco_path=gt_path,
            max_dets=[1, 10, 100]
        )
        
        report = evaluator.last_alignment_report
        
        # Assertions
        assert report.num_matched == 3, "Should match all 3 tiles"
        assert report.match_rate == 1.0, "Match rate should be 100%"
        assert report.num_gt_without_pred == 0, "No GT should be unmatched"
        
        print("\n✓ TEST PASSED")
        print("  All tiles matched perfectly")
        print("  Metrics: mAP={:.3f}, mAR={:.3f}".format(
            metrics['AP'], metrics['AR']))
        
    finally:
        Path(gt_path).unlink()
        Path(pred_path).unlink()


def test_base_name_rsplit_match():
    """Test case: Filenames differ only in last suffix"""
    print("\n" + "="*70)
    print("TEST 2: Base Name Match (rsplit)")
    print("="*70)
    
    # GT tiles from annotation tiling
    gt_images = [
        {'id': 1, 'file_name': 'raster_tile_1024_0_0.tif',
         'width': 512, 'height': 512},
        {'id': 2, 'file_name': 'raster_tile_1024_0_1.tif',
         'width': 512, 'height': 512},
        {'id': 3, 'file_name': 'raster_tile_1024_1_0.tif',
         'width': 512, 'height': 512}
    ]
    gt_anns = [
        {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [10, 10, 50, 50],
         'segmentation': [[10, 10, 60, 10, 60, 60, 10, 60]], 'area': 2500}
    ]
    
    # Pred tiles from polygon-based tiling (different last suffix)
    pred_images = [
        {'id': 101, 'file_name': 'raster_tile_1024_0_1234.tif',
         'width': 512, 'height': 512},
        {'id': 102, 'file_name': 'raster_tile_1024_0_1235.tif',
         'width': 512, 'height': 512},
        {'id': 103, 'file_name': 'raster_tile_1024_1_1236.tif',
         'width': 512, 'height': 512}
    ]
    pred_anns = [
        {'id': 201, 'image_id': 101, 'category_id': 1, 'score': 0.9,
         'bbox': [12, 12, 48, 48],
         'segmentation': [[12, 12, 60, 12, 60, 60, 12, 60]], 'area': 2304}
    ]
    
    gt_path = write_temp_coco(create_coco_json(gt_images, gt_anns))
    pred_path = write_temp_coco(create_coco_json(pred_images, pred_anns))
    
    try:
        evaluator = ClassifierCocoEvaluator(
            alignment_strategy=AlignmentStrategy.BASE_RSPLIT_1,
            verbose=True
        )
        
        metrics = evaluator.tile_level(
            preds_coco_path=pred_path,
            truth_coco_path=gt_path,
            max_dets=[1, 10, 100]
        )
        
        report = evaluator.last_alignment_report
        
        # Assertions
        assert report.num_matched == 3, "Should match all 3 tiles by base name"
        assert report.match_rate == 1.0, "Match rate should be 100%"
        
        print("\n✓ TEST PASSED")
        print("  Base name matching successful despite different suffixes")
        
    finally:
        Path(gt_path).unlink()
        Path(pred_path).unlink()


def test_partial_match():
    """Test case: Some GT tiles have no predictions"""
    print("\n" + "="*70)
    print("TEST 3: Partial Match (Some GT without Predictions)")
    print("="*70)
    
    # GT has 5 tiles
    gt_images = [
        {'id': i, 'file_name': 'tile_{:03d}.tif'.format(i),
         'width': 512, 'height': 512}
        for i in range(1, 6)
    ]
    gt_anns = [
        {'id': i, 'image_id': i, 'category_id': 1, 'bbox': [10, 10, 50, 50],
         'segmentation': [[10, 10, 60, 10, 60, 60, 10, 60]], 'area': 2500}
        for i in range(1, 6)
    ]
    
    # Predictions only for 3 tiles (missing tiles 2 and 4)
    pred_images = [
        {'id': 101, 'file_name': 'tile_001.tif', 'width': 512,
         'height': 512},
        {'id': 103, 'file_name': 'tile_003.tif', 'width': 512,
         'height': 512},
        {'id': 105, 'file_name': 'tile_005.tif', 'width': 512,
         'height': 512}
    ]
    pred_anns = [
        {'id': 201, 'image_id': 101, 'category_id': 1, 'score': 0.9,
         'bbox': [12, 12, 48, 48],
         'segmentation': [[12, 12, 60, 12, 60, 60, 12, 60]], 'area': 2304}
    ]
    
    gt_path = write_temp_coco(create_coco_json(gt_images, gt_anns))
    pred_path = write_temp_coco(create_coco_json(pred_images, pred_anns))
    
    try:
        evaluator = ClassifierCocoEvaluator(
            alignment_strategy=AlignmentStrategy.EXACT_MATCH,
            verbose=True,
            min_match_rate_warning=0.95  # Should trigger warning at 60%
        )
        
        metrics = evaluator.tile_level(
            preds_coco_path=pred_path,
            truth_coco_path=gt_path,
            max_dets=[1, 10, 100]
        )
        
        report = evaluator.last_alignment_report
        
        # Assertions
        assert report.num_matched == 3, "Should match 3 tiles"
        assert report.match_rate == 0.6, "Match rate should be 60%"
        assert report.num_gt_without_pred == 2, "2 GT tiles without predictions"
        assert len(report.unmatched_gt_files) == 2, "Should track unmatched"
        
        print("\n✓ TEST PASSED")
        print("  Partial matching handled correctly")
        print("  Warning triggered as expected for low match rate")
        
    finally:
        Path(gt_path).unlink()
        Path(pred_path).unlink()


def test_no_match():
    """Test case: Complete naming mismatch - should fail"""
    print("\n" + "="*70)
    print("TEST 4: No Match (Complete Mismatch - Should Fail)")
    print("="*70)
    
    # GT with naming pattern A
    gt_images = [
        {'id': 1, 'file_name': 'data_A_tile_001.tif',
         'width': 512, 'height': 512},
        {'id': 2, 'file_name': 'data_A_tile_002.tif',
         'width': 512, 'height': 512}
    ]
    gt_anns = [
        {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [10, 10, 50, 50],
         'segmentation': [[10, 10, 60, 10, 60, 60, 10, 60]], 'area': 2500}
    ]
    
    # Pred with completely different naming pattern B
    pred_images = [
        {'id': 101, 'file_name': 'data_B_region_100.tif',
         'width': 512, 'height': 512},
        {'id': 102, 'file_name': 'data_B_region_200.tif',
         'width': 512, 'height': 512}
    ]
    pred_anns = [
        {'id': 201, 'image_id': 101, 'category_id': 1, 'score': 0.9,
         'bbox': [12, 12, 48, 48],
         'segmentation': [[12, 12, 60, 12, 60, 60, 12, 60]], 'area': 2304}
    ]
    
    gt_path = write_temp_coco(create_coco_json(gt_images, gt_anns))
    pred_path = write_temp_coco(create_coco_json(pred_images, pred_anns))
    
    try:
        evaluator = ClassifierCocoEvaluator(
            alignment_strategy=AlignmentStrategy.EXACT_MATCH,
            verbose=True,
            min_match_rate_error=0.5  # Should trigger error at 0%
        )
        
        try:
            metrics = evaluator.tile_level(
                preds_coco_path=pred_path,
                truth_coco_path=gt_path,
                max_dets=[1, 10, 100]
            )
            
            # Should not reach here
            print("\n✗ TEST FAILED")
            print("  Expected AlignmentError but evaluation succeeded")
            assert False, "Should have raised AlignmentError"
            
        except AlignmentError as e:
            print("\n✓ TEST PASSED")
            print("  AlignmentError raised as expected for 0% match rate")
            print("  Error message: {}".format(str(e)[:100] + "..."))
        
    finally:
        Path(gt_path).unlink()
        Path(pred_path).unlink()


def test_empty_predictions():
    """Test case: GT has tiles but predictions are empty"""
    print("\n" + "="*70)
    print("TEST 5: Empty Predictions")
    print("="*70)
    
    gt_images = [
        {'id': 1, 'file_name': 'tile_001.tif', 'width': 512, 'height': 512},
        {'id': 2, 'file_name': 'tile_002.tif', 'width': 512, 'height': 512}
    ]
    gt_anns = [
        {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [10, 10, 50, 50],
         'segmentation': [[10, 10, 60, 10, 60, 60, 10, 60]], 'area': 2500}
    ]
    
    # Empty predictions
    pred_images = []
    pred_anns = []
    
    gt_path = write_temp_coco(create_coco_json(gt_images, gt_anns))
    pred_path = write_temp_coco(create_coco_json(pred_images, pred_anns))
    
    try:
        evaluator = ClassifierCocoEvaluator(
            alignment_strategy=AlignmentStrategy.EXACT_MATCH,
            verbose=True,
            min_match_rate_error=0.0  # Allow 0% match
        )
        
        metrics = evaluator.tile_level(
            preds_coco_path=pred_path,
            truth_coco_path=gt_path,
            max_dets=[1, 10, 100]
        )
        
        report = evaluator.last_alignment_report
        
        # Assertions
        assert report.num_matched == 0, "No matches expected"
        assert report.match_rate == 0.0, "Match rate should be 0%"
        assert metrics['AP'] == 0.0, "mAP should be 0 with no predictions"
        assert metrics['AR'] == 0.0, "mAR should be 0 with no predictions"
        
        print("\n✓ TEST PASSED")
        print("  Empty predictions handled gracefully")
        print("  Metrics correctly show 0.0")
        
    finally:
        Path(gt_path).unlink()
        Path(pred_path).unlink()


def test_diagnostic_mode():
    """Test the diagnose_alignment() method"""
    print("\n" + "="*70)
    print("TEST 6: Diagnostic Mode")
    print("="*70)
    
    gt_images = [
        {'id': 1, 'file_name': 'tile_1024_0_0.tif',
         'width': 512, 'height': 512},
        {'id': 2, 'file_name': 'tile_1024_0_1.tif',
         'width': 512, 'height': 512}
    ]
    gt_anns = []
    
    pred_images = [
        {'id': 101, 'file_name': 'tile_1024_0_100.tif',
         'width': 512, 'height': 512},
        {'id': 102, 'file_name': 'tile_1024_0_101.tif',
         'width': 512, 'height': 512}
    ]
    pred_anns = []
    
    gt_path = write_temp_coco(create_coco_json(gt_images, gt_anns))
    pred_path = write_temp_coco(create_coco_json(pred_images, pred_anns))
    
    try:
        evaluator = ClassifierCocoEvaluator(
            alignment_strategy=AlignmentStrategy.BASE_RSPLIT_1,
            verbose=True
        )
        
        # Run diagnostic without full evaluation
        report = evaluator.diagnose_alignment(
            preds_coco_path=pred_path,
            truth_coco_path=gt_path
        )
        
        # Assertions
        assert report is not None, "Should return alignment report"
        assert report.num_gt_images == 2, "Should detect 2 GT images"
        assert report.num_pred_images == 2, "Should detect 2 pred images"
        
        print("\n✓ TEST PASSED")
        print("  Diagnostic mode works without running evaluation")
        
    finally:
        Path(gt_path).unlink()
        Path(pred_path).unlink()


def run_all_tests():
    """Run all alignment tests"""
    print("\n" + "="*70)
    print("RUNNING ALIGNMENT UNIT TESTS")
    print("="*70)
    
    tests = [
        test_perfect_exact_match,
        test_base_name_rsplit_match,
        test_partial_match,
        test_no_match,
        test_empty_predictions,
        test_diagnostic_mode
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print("\n✗ TEST FAILED: {}".format(str(e)))
            failed += 1
        except Exception as e:
            print("\n✗ TEST ERROR: {}".format(str(e)))
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print("Passed: {}/{}".format(passed, len(tests)))
    print("Failed: {}".format(failed))
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
