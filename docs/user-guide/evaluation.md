# Evaluation

CanopyRS provides tools for finding optimal NMS parameters and benchmarking models on test datasets.

## Finding optimal NMS parameters

To find the optimal NMS parameters for your model (`nms_iou_threshold` and `nms_score_threshold`), use the `find_optimal_raster_nms.py` tool script. This script runs a grid search over NMS parameters and evaluates results using COCO evaluation metrics at a chosen IoU threshold.

### IoU threshold options

- `--eval_iou_threshold 0.75` for RF1₇₅ (default)
- `--eval_iou_threshold 50:95` for COCO-style sweep (RF1<sub>50:95</sub>)
- Comma-separated lists are also accepted: `--eval_iou_threshold 0.50,0.65,0.80`

### Example: Finding NMS parameters for RF1₇₅

To find NMS parameters for the DINO Swin-L multi-NQOS detector on the validation set of SelvaBox and Detectree2:

=== "Linux / macOS"

    ```bash
    python -m canopyrs.tools.detection.find_optimal_raster_nms \
      -c canopyrs/config/detectors/dino_swinL_multi_NQOS.yaml \
      -d SelvaBox Detectree2 \
      -r <DATA_ROOT> \
      -o <OUTPUT_PATH> \
      --n_workers 6 \
      --eval_iou_threshold 0.75
    ```

=== "Windows (PowerShell)"

    ```powershell
    python -m canopyrs.tools.detection.find_optimal_raster_nms `
      -c canopyrs/config/detectors/dino_swinL_multi_NQOS.yaml `
      -d SelvaBox Detectree2 `
      -r <DATA_ROOT> `
      -o <OUTPUT_PATH> `
      --n_workers 6 `
      --eval_iou_threshold 0.75
    ```

### Example: Finding NMS parameters for RF1<sub>50:95</sub>

=== "Linux / macOS"

    ```bash
    python -m canopyrs.tools.detection.find_optimal_raster_nms \
      -c canopyrs/config/detectors/dino_swinL_multi_NQOS.yaml \
      -d SelvaBox Detectree2 \
      -r <DATA_ROOT> \
      -o <OUTPUT_PATH> \
      --n_workers 6 \
      --eval_iou_threshold 50:95
    ```

=== "Windows (PowerShell)"

    ```powershell
    python -m canopyrs.tools.detection.find_optimal_raster_nms `
      -c canopyrs/config/detectors/dino_swinL_multi_NQOS.yaml `
      -d SelvaBox Detectree2 `
      -r <DATA_ROOT> `
      -o <OUTPUT_PATH> `
      --n_workers 6 `
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

To run raster-level evaluation (RF1) in addition to tile-level, you must pass values for `--nms_threshold` and `--score_threshold`. To find these parameter values, run `find_optimal_raster_nms.py` on the validation set of one (or more) datasets, as described above.

The benchmark will then run a single raster-level aggregation with those values and report RF1 at the chosen IoU setting.

### Important: Use consistent IoU thresholds

Always use the **same** `--eval_iou_threshold` value when finding NMS parameters and when running the final benchmark. If you optimize NMS for RF1₇₅ but benchmark with RF1<sub>50:95</sub>, your NMS parameters will not be optimal for that metric.

### Example: Benchmarking with RF1<sub>50:95</sub>

To benchmark the DINO Swin-L multi-NQOS detector on the test set of SelvaBox and Detectree2:

=== "Linux / macOS"

    ```bash
    python -m canopyrs.tools.detection.benchmark \
      -c canopyrs/config/detectors/dino_swinL_multi_NQOS.yaml \
      -d SelvaBox Detectree2 \
      -r <DATA_ROOT> \
      -o <OUTPUT_PATH> \
      --nms_threshold 0.7 \
      --score_threshold 0.5 \
      --eval_iou_threshold 50:95
    ```

=== "Windows (PowerShell)"

    ```powershell
    python -m canopyrs.tools.detection.benchmark `
      -c canopyrs/config/detectors/dino_swinL_multi_NQOS.yaml `
      -d SelvaBox Detectree2 `
      -r <DATA_ROOT> `
      -o <OUTPUT_PATH> `
      --nms_threshold 0.7 `
      --score_threshold 0.5 `
      --eval_iou_threshold 50:95
    ```

By default, evaluation is done on the test set.

For more information on parameters:

```bash
python -m canopyrs.tools.detection.benchmark --help
```

## Programmatic usage

You can also use the benchmarker classes directly in Python for more control.

### Detector example

```python
from canopyrs.engine.benchmark import DetectorBenchmarker
from canopyrs.engine.config_parsers import DetectorConfig, AggregatorConfig

detector_config = DetectorConfig.from_yaml("canopyrs/config/detectors/dino_swinL_multi_NQOS.yaml")

# COCO-style IoU sweep for RF1_50:95
eval_ious = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

# Detector-only: all confidence comes from the detector
aggregator_base = AggregatorConfig(
    nms_algorithm="iou",
    detector_score_weight=1.0,
    segmenter_score_weight=0.0,
)

# Step 1: Find optimal NMS parameters on validation set
valid_benchmarker = DetectorBenchmarker(
    output_folder="./output/benchmark/valid",
    fold_name="valid",
    raw_data_root="/data/canopyrs",
    eval_iou_threshold=eval_ious,
)

best_aggregator = valid_benchmarker.find_optimal_nms_iou_threshold(
    detector_config=detector_config,
    base_aggregator_config=aggregator_base,
    dataset_names=["SelvaBox", "Detectree2"],
    nms_iou_thresholds=[i / 20 for i in range(1, 21)],# these parameters define over which values the grid search should be ran
    nms_score_thresholds=[i / 20 for i in range(1, 21)],
    n_workers=6,
)

# Step 2: Benchmark on test set using optimal NMS parameters
test_benchmarker = DetectorBenchmarker(
    output_folder="./output/benchmark/test",
    fold_name="test",
    raw_data_root="/data/canopyrs",
    eval_iou_threshold=eval_ious,
)

tile_metrics, raster_metrics = test_benchmarker.benchmark(
    detector_config=detector_config,
    aggregator_config=best_aggregator,
    dataset_names=["SelvaBox", "Detectree2"],
)
```

### Segmenter example (end-to-end)

```python
from canopyrs.engine.benchmark import SegmenterBenchmarker
from canopyrs.engine.config_parsers import SegmenterConfig, AggregatorConfig

segmenter_config = SegmenterConfig.from_yaml("canopyrs/config/segmenters/mask2former_swinL_multi_selvamask.yaml")

# COCO-style IoU sweep for RF1_50:95
eval_ious = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

# End-to-end segmenter: all confidence comes from the segmenter
aggregator_base = AggregatorConfig(
    nms_algorithm="iou",
    detector_score_weight=0.0,
    segmenter_score_weight=1.0,
)

# Step 1: Find optimal NMS parameters on validation set
valid_benchmarker = SegmenterBenchmarker(
    output_folder="./output/benchmark/valid",
    fold_name="valid",
    raw_data_root="/data/canopyrs",
    eval_iou_threshold=eval_ious,
)

best_aggregator = valid_benchmarker.find_optimal_nms_iou_threshold(
    segmenter_config=segmenter_config,
    base_aggregator_config=aggregator_base,
    dataset_names=["SelvaMask"],
    nms_iou_thresholds=[i / 20 for i in range(1, 21)],
    nms_score_thresholds=[i / 20 for i in range(1, 21)],
    n_workers=6,
)

# Step 2: Benchmark on test set using optimal NMS parameters
test_benchmarker = SegmenterBenchmarker(
    output_folder="./output/benchmark/test",
    fold_name="test",
    raw_data_root="/data/canopyrs",
    eval_iou_threshold=eval_ious,
)

tile_metrics, raster_metrics = test_benchmarker.benchmark(
    segmenter_config=segmenter_config,
    aggregator_config=best_aggregator,
    dataset_names=["SelvaMask"],
)
```

### Segmenter example (detector + prompted SAM3)

```python
from canopyrs.engine.benchmark import SegmenterBenchmarker
from canopyrs.engine.config_parsers import DetectorConfig, SegmenterConfig, AggregatorConfig

detector_config = DetectorConfig.from_yaml("canopyrs/config/detectors/dino_swinL_multi_NQOS_selvamask_FT.yaml")
segmenter_config = SegmenterConfig.from_yaml("canopyrs/config/segmenters/sam3_multi_selvamask_FT.yaml")

# COCO-style IoU sweep for RF1_50:95
eval_ious = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

# Prompted SAM3: final score is a weighted blend of detector and segmenter confidence.
# Equal weights (0.5/0.5) is a reasonable default; rebalance if one model is more reliable.
aggregator_base = AggregatorConfig(
    nms_algorithm="ioa-disambiguate",
    detector_score_weight=0.5,
    segmenter_score_weight=0.5,
)

# Step 1: Find optimal NMS parameters on validation set
valid_benchmarker = SegmenterBenchmarker(
    output_folder="./output/benchmark/valid",
    fold_name="valid",
    raw_data_root="/data/canopyrs",
    eval_iou_threshold=eval_ious,
)

best_aggregator = valid_benchmarker.find_optimal_nms_iou_threshold(
    segmenter_config=segmenter_config,
    prompter_detector_config=detector_config,
    base_aggregator_config=aggregator_base,
    dataset_names=["SelvaMask"],
    nms_iou_thresholds=[i / 20 for i in range(1, 21)],      # these parameters define over which values the grid search should be ran
    nms_score_thresholds=[i / 20 for i in range(1, 21)],
    n_workers=6,
)

# Step 2: Benchmark on test set
test_benchmarker = SegmenterBenchmarker(
    output_folder="./output/benchmark/test",
    fold_name="test",
    raw_data_root="/data/canopyrs",
    eval_iou_threshold=eval_ious,
)

tile_metrics, raster_metrics = test_benchmarker.benchmark(
    segmenter_config=segmenter_config,
    prompter_detector_config=detector_config,
    aggregator_config=best_aggregator,
    dataset_names=["SelvaMask"],
)
```

### Aggregating results across seeds

If you ran multiple training seeds, you can compute mean/std across runs for both tile-level and raster-level metrics, then merge them into a single table:

```python
# Assuming you collected per-seed results:
# tile_metrics_list = [tile_metrics_seed1, tile_metrics_seed2, tile_metrics_seed3]
# raster_metrics_list = [raster_metrics_seed1, raster_metrics_seed2, raster_metrics_seed3]

summary_tile = DetectorBenchmarker.compute_mean_std_metric_tables(
    tile_metrics_list,
    output_csv="./output/benchmark/tile_summary.csv",
)

summary_raster = DetectorBenchmarker.compute_mean_std_metric_tables(
    raster_metrics_list,
    output_csv="./output/benchmark/raster_summary.csv",
)

# Merge tile and raster summaries into one table
combined = DetectorBenchmarker.merge_tile_and_raster_summaries(
    summary_tile,
    summary_raster,
    output_csv="./output/benchmark/combined_summary.csv",
    tile_prefix="tile",
    raster_prefix="raster",
)
```
