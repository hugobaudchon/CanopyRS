# Evaluation

CanopyRS provides tools for finding optimal NMS parameters and benchmarking models on test datasets.

## Finding optimal NMS parameters

To find the optimal NMS parameters for your model (`nms_iou_threshold` and `nms_score_threshold`), use the `find_optimal_raster_nms.py` tool script. This script runs a grid search over NMS parameters and evaluates results using COCO evaluation metrics at a chosen IoU threshold.

### IoU threshold options

- `--eval_iou_threshold 0.75` for RF1₇₅ (default)
- `--eval_iou_threshold 50:95` for COCO-style sweep (RF1₅₀:₉₅)
- Comma-separated lists are also accepted: `--eval_iou_threshold 0.50,0.65,0.80`

### Example: Finding NMS parameters for RF1₇₅

To find NMS parameters for the `default_detection_multi_NQOS_best` model on the validation set of SelvaBox and Detectree2:

```bash
python -m canopyrs.tools.detection.find_optimal_raster_nms \
  -c config/default_detection_multi_NQOS_best/detector.yaml \
  -d SelvaBox Detectree2 \
  -r <DATA_ROOT> \
  -o <OUTPUT_PATH> \
  --n_workers 6 \
  --eval_iou_threshold 0.75
```

### Example: Finding NMS parameters for RF1₅₀:₉₅

```bash
python -m canopyrs.tools.detection.find_optimal_raster_nms \
  -c config/default_detection_multi_NQOS_best/detector.yaml \
  -d SelvaBox Detectree2 \
  -r <DATA_ROOT> \
  -o <OUTPUT_PATH> \
  --n_workers 6 \
  --eval_iou_threshold 50:95
```

### Performance notes

Depending on how many rasters there are in the datasets you select, the search could take from a few tens of minutes to a few hours. If you have lots of CPU cores, we recommend increasing the number of workers.

For more information on parameters:

```bash
python -m canopyrs.tools.detection.find_optimal_raster_nms --help
```

## Benchmarking

To benchmark a model on test or valid sets of datasets, use the `benchmark.py` tool script.

This script runs the model and evaluates results using tile-level COCO metrics (mAP and mAR).

### Raster-level evaluation (RF1)

To run raster-level evaluation (RF1) in addition to tile-level, you must pass values for `--nms_threshold` and `--score_threshold`. To find these parameter values, run `find_optimal_raster_nms.py` as described above.

The benchmark will then run a single raster-level aggregation with those values and report RF1 at the chosen IoU setting.

### Important: Use consistent IoU thresholds

Always use the **same** `--eval_iou_threshold` value when finding NMS parameters and when running the final benchmark. If you optimize NMS for RF1₇₅ but benchmark with RF1₅₀:₉₅, your NMS parameters will not be optimal for that metric.

### Example: Benchmarking with RF1₇₅

To benchmark the `default_detection_multi_NQOS_best` model on the test set of SelvaBox and Detectree2:

```bash
python -m canopyrs.tools.detection.benchmark \
  -c config/default_detection_multi_NQOS_best/detector.yaml \
  -d SelvaBox Detectree2 \
  -r <DATA_ROOT> \
  -o <OUTPUT_PATH> \
  --nms_threshold 0.7 \
  --score_threshold 0.5 \
  --eval_iou_threshold 0.75
```

By default, evaluation is done on the test set.

For more information on parameters:

```bash
python -m canopyrs.tools.detection.benchmark --help
```
