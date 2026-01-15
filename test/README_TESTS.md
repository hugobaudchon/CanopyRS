# CanopyRS Classifier Evaluation Tests

This directory contains comprehensive tests for the classification evaluation pipeline.

## Test Files

### Unit Tests

#### `test_alignment_unit.py`
Tests for COCO dataset alignment logic.

**Run:**
```bash
python test/test_alignment_unit.py
```

**Test Cases:**
1. **Perfect Exact Match**: GT and Pred have identical filenames
2. **Base Name Match**: Filenames differ only in last suffix (rsplit logic)
3. **Partial Match**: Some GT tiles without predictions
4. **No Match**: Complete naming mismatch (should fail with error)
5. **Empty Predictions**: GT has tiles but predictions are empty
6. **Diagnostic Mode**: Test `diagnose_alignment()` method

**What It Tests:**
- Alignment strategies (EXACT_MATCH vs BASE_RSPLIT_1)
- Match rate calculation
- Unmatched tile tracking
- Error/warning thresholds
- Alignment report generation

---

#### `test_score_combination.py`
Tests for score combination and selection logic.

**Run:**
```bash
python test/test_score_combination.py
```

**Test Cases:**
1. **Weighted Arithmetic Mean**: Combines detector + classifier + segmentation scores
2. **Weighted Geometric Mean**: Alternative combination method
3. **Score Field Priority**: Tests automatic score selection order
4. **Missing Scores**: Handles when requested scores don't exist
5. **Weight Normalization**: Normalizes weights that don't sum to 1.0
6. **No Scores**: Graceful handling when no scores provided

**What It Tests:**
- Score combination algorithms
- Weight normalization
- Score field selection priority
- Missing score handling
- Edge cases

---

### Integration Tests

#### `test/evaluator_classifier.py`
Main evaluation script showing usage examples.

**Run:**
```bash
python test/evaluator_classifier.py
```

**Features:**
- Loads real COCO files for evaluation
- Supports multiple evaluation runs
- Generates summary tables
- Saves results to CSV/JSON
- Demonstrates alignment diagnostics

**Configure:**
Edit the `CONFIGURATION` section in the script:
- `CATEGORIES_FILE`: Path to categories JSON
- `OUTPUT_DIR`: Where to save results
- `EVALUATION_RUNS`: List of (name, pred_path, truth_path) tuples

---

## Quick Start

### 1. Run All Unit Tests

```bash
cd /path/to/CanopyRS

# Test alignment
python test/test_alignment_unit.py

# Test score combination
python test/test_score_combination.py
```

### 2. Diagnose Alignment Issues

If you're getting bad evaluation results, use diagnostic mode:

```python
from engine.benchmark.classifier.evaluator import (
    ClassifierCocoEvaluator,
    AlignmentStrategy
)

evaluator = ClassifierCocoEvaluator(
    alignment_strategy=AlignmentStrategy.BASE_RSPLIT_1,
    verbose=True
)

report = evaluator.diagnose_alignment(
    preds_coco_path="path/to/predictions.json",
    truth_coco_path="path/to/ground_truth.json"
)
```

This will show:
- How many tiles matched
- Which GT tiles have no predictions
- Sample filenames from both datasets
- Category alignment check
- Recommendations for fixing issues

### 3. Run Full Evaluation

```python
evaluator = ClassifierCocoEvaluator(
    alignment_strategy=AlignmentStrategy.BASE_RSPLIT_1,
    min_match_rate_warning=0.95,
    min_match_rate_error=0.50,
    verbose=True
)

# With score combination
score_combination = {
    'weights': {
        'detector_score': 0.4,
        'classifier_score': 0.2,
        'segmentation_score': 0.4
    },
    'method': 'weighted_arithmetic_mean'
}

metrics = evaluator.tile_level(
    preds_coco_path="predictions.json",
    truth_coco_path="ground_truth.json",
    evaluate_bbox=True,  # Also evaluate bounding boxes
    score_combination=score_combination
)

# Access alignment report
report = evaluator.last_alignment_report
print("Match rate: {:.1%}".format(report.match_rate))
```

---

## Common Issues & Solutions

### Issue 1: Low Match Rate (<95%)

**Symptoms:**
```
WARNING: Match rate 60.0% is below 95%
GT tiles without predictions: 40
```

**Possible Causes:**
1. GT and predictions use different tiling strategies
2. Some tiles missing from prediction pipeline
3. Predictions filtered out (e.g., by confidence threshold)

**Solutions:**
1. Check tiling configuration matches
2. Verify all GT tiles processed in prediction pipeline
3. Use `diagnose_alignment()` to see unmatched tiles

---

### Issue 2: Complete Mismatch (0% match)

**Symptoms:**
```
AlignmentError: only 0.0% of GT tiles matched predictions
```

**Possible Causes:**
1. Different tile naming conventions
2. GT from one raster, predictions from another
3. Wrong alignment strategy

**Solutions:**
1. Try `AlignmentStrategy.EXACT_MATCH` instead of BASE_RSPLIT_1
2. Verify GT and predictions from same raster
3. Check filename patterns with `diagnose_alignment()`

---

### Issue 3: Score Field Not Found

**Symptoms:**
```
WARNING: No score fields found in predictions
```

**Possible Causes:**
1. Predictions don't have score field in COCO JSON
2. Score in wrong location (top-level vs other_attributes)

**Solutions:**
1. Ensure predictions have 'score' or specific score fields
2. Check COCO generation includes scores
3. Use score_combination to specify which fields to use

---

## Alignment Strategies

### BASE_RSPLIT_1 (Default)

Best for: Polygon-based classification tiles that differ from GT grid tiles

**How it works:**
```python
filename = "raster_tile_1024_0_123.tif"
stem = "raster_tile_1024_0_123"
base = stem.rsplit('_', 1)[0]  # "raster_tile_1024_0"
```

Matches tiles that differ only in the **last underscore segment**.

**Example:**
- GT: `raster_tile_1024_0_0.tif` → base: `raster_tile_1024_0`
- Pred: `raster_tile_1024_0_1234.tif` → base: `raster_tile_1024_0`
- ✓ **Match!**

### EXACT_MATCH

Best for: GT and predictions from identical tiling process

**How it works:**
Filenames must match exactly.

**Example:**
- GT: `tile_001.tif`
- Pred: `tile_001.tif`
- ✓ **Match!**

But:
- GT: `tile_001.tif`
- Pred: `tile_001_pred.tif`
- ✗ **No match**

---

## Score Combination Methods

### Weighted Arithmetic Mean

```python
combined = Σ(weight_i * score_i) / Σ(weight_i)
```

**Use when:** Scores are on similar scales and you want a simple average.

### Weighted Geometric Mean

```python
combined = exp(Σ(weight_i * log(score_i)) / Σ(weight_i))
```

**Use when:** You want to penalize low scores more heavily (one low score significantly reduces combined score).

---

## Expected Metrics

### Good Alignment
- Match rate: **>98%**
- GT without predictions: **<2%**
- No warnings in alignment report

### Acceptable Alignment
- Match rate: **95-98%**
- Warning issued but evaluation proceeds
- Some tiles missing predictions (expected if model skipped areas)

### Poor Alignment
- Match rate: **<95%**
- Strong warning issued
- Consider fixing naming or tiling configuration

### Critical Failure
- Match rate: **<50%**
- AlignmentError raised
- Evaluation blocked
- Must fix naming mismatch

---

## Tips for Writing Your Own Tests

### 1. Use Helper Functions

```python
def create_simple_coco(n_images=5, n_anns_per_image=3):
    images = [
        {'id': i, 'file_name': f'tile_{i:03d}.tif',
         'width': 512, 'height': 512}
        for i in range(1, n_images + 1)
    ]
    
    annotations = []
    ann_id = 1
    for img_id in range(1, n_images + 1):
        for _ in range(n_anns_per_image):
            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': 1,
                'bbox': [10, 10, 50, 50],
                'segmentation': [[10, 10, 60, 10, 60, 60, 10, 60]],
                'area': 2500,
                'score': 0.9
            })
            ann_id += 1
    
    categories = [{'id': 1, 'name': 'tree', 'supercategory': ''}]
    
    return {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }
```

### 2. Clean Up Temp Files

```python
try:
    # Your test code
    evaluator.tile_level(pred_path, gt_path)
finally:
    # Always clean up
    Path(pred_path).unlink()
    Path(gt_path).unlink()
```

### 3. Test Edge Cases

- Empty datasets
- Single image
- All correct predictions
- All wrong predictions
- Missing categories
- Malformed COCO files

---

## Troubleshooting Test Failures

### Import Errors

```bash
ModuleNotFoundError: No module named 'engine'
```

**Fix:** Ensure project root is in Python path:
```python
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
```

### File Not Found

```bash
ERROR: Predictions file not found
```

**Fix:** Use absolute paths or verify working directory:
```python
pred_path = Path('/absolute/path/to/predictions.json')
assert pred_path.exists(), f"File not found: {pred_path}"
```

### COCO Loading Errors

```bash
JSONDecodeError: Expecting value
```

**Fix:** Validate COCO structure:
```python
required_keys = ['images', 'annotations', 'categories']
assert all(k in coco_dict for k in required_keys)
```

---

## Contributing

When adding new tests:

1. **Name clearly**: `test_<feature>_<scenario>.py`
2. **Document**: Add docstring explaining what's tested
3. **Isolate**: Each test should be independent
4. **Clean up**: Always remove temporary files
5. **Assert**: Use assertions to validate expected behavior
6. **Report**: Print clear pass/fail messages

---

## Contact

For questions about tests or evaluation issues, refer to the main CanopyRS documentation or open an issue in the repository.
