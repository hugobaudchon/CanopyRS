# Configuration

CanopyRS pipelines are defined in YAML. A pipeline config lists the components to run and their parameters.

## Pipeline config structure

```yaml
components_configs:
  - tilerizer:
      tile_type: tile
      tile_size: 1777
      tile_overlap: 0.75
      ground_resolution: 0.045

  - detector: detectors/dino_swinL_multi_NQOS.yaml

  - aggregator:
      nms_algorithm: 'iou'
      score_threshold: 0.5
      nms_threshold: 0.7
      edge_band_buffer_percentage: 0.05
      scores_weights: { 'detector_score': 1.0 }
```

Each entry is either:

- **Inline config** — parameters specified directly in the pipeline YAML
- **Reference** — a path to a reusable component config, for example a detector in `canopyrs/config/detectors/` or a segmenter in `canopyrs/config/segmenters/`. See our [Model Zoo](model-zoo.md).

## Component config files

Reusable component configs live in `canopyrs/config/detectors/` and `canopyrs/config/segmenters/`. They can be referenced from any pipeline config by relative path. See our [Model Zoo](model-zoo.md) for the full list.

Example detector config (`detectors/dino_swinL_multi_NQOS.yaml`):

```yaml
model: dino_detrex
architecture: dino/configs/dino-swin/dino_swin_large_384_5scale_36ep.py
checkpoint_path: 'https://huggingface.co/CanopyRS/dino-swin-l-384-multi-NQOS/resolve/main/model_best.pth'
batch_size: 1
box_predictions_per_image: 500
num_classes: 1
```

## Key parameters

### Tilerizer

| Parameter | Description |
|---|---|
| `tile_type` | `tile`, `tile_labeled`, or `polygon` |
| `tile_size` | Tile size in pixels |
| `tile_overlap` | Overlap ratio between tiles (0–1) |
| `ground_resolution` | Target ground resolution in meters/pixel |

### Aggregator

| Parameter | Description |
|---|---|
| `nms_algorithm` | `iou` or `ioa-disambiguate` |
| `score_threshold` | Minimum score to keep a detection |
| `nms_threshold` | IoU threshold for suppression |
| `scores_weights` | Weight per score column for combined scoring |
| `scores_weighting_method` | How to combine scores (e.g. `weighted_geometric_mean`) |
