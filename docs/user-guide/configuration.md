<style>
table td:first-child {
  white-space: nowrap;
}
</style>

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
- **Reference** — a path to a reusable component config, for example a detector in `canopyrs/config/detectors/` or a segmenter in `canopyrs/config/segmenters/`. See our [Model Zoo](model-zoo.md) for the full list.

    Example reusable detector config (`detectors/dino_swinL_multi_NQOS.yaml`):

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
| `ground_resolution` | Target ground resolution in meters/pixel (cannot be set together with `scale_factor`) |
| `scale_factor` | Scale factor to resize tiles (cannot be set together with `ground_resolution`). A value of 1.0 keeps the source raster resolution as-is. |

### Detector

| Parameter | Description |
|---|---|
| `model` | Model type identifier (e.g. `dino_detrex`, `faster_rcnn_detectron2`, `deepforest`) |
| `architecture` | Model architecture configuration path |
| `checkpoint_path` | Path or URL to pre-trained model weights |
| `batch_size` | Inference batch size |
| `num_classes` | Number of classes to detect (typically `1` for trees) |
| `box_predictions_per_image` | Maximum number of predictions per image |
| `augmentation_image_size` | Resize image size. A single int (e.g. `1024`) for a fixed size, or a `[min, max]` list (e.g. `[1024, 1777]`) where images smaller than min are upscaled and images larger than max are downscaled |

### Segmenter

| Parameter | Description |
|---|---|
| `model` | Model type identifier (e.g. `sam2`, `sam3`, `detectree2`) |
| `architecture` | Model architecture variant (e.g. `l` for large) |
| `checkpoint_path` | Path or URL to pre-trained model weights |
| `image_batch_size` | Number of images processed at once (sometimes ignored by models like SAM that rely exclusively on `box_batch_size`)|
| `box_batch_size` | Number of prompt boxes processed at once per image |
| `augmentation_image_size` | Resize image size. A single int (e.g. `1024`) for a fixed size, or a `[min, max]` list (e.g. `[1024, 1777]`) where images smaller than min are upscaled and images larger than max are downscaled |
| `pp_n_workers` | Number of workers for parallel post-processing. Reduce this number if you have a limited amount of CPU cores. |
| `pp_down_scale_masks_px` | Downscale masks to this pixel size for more efficient post-processing |
| `pp_simplify_tolerance` | Polygon simplification tolerance |
| `pp_remove_rings` | Remove holes/rings from segmentation masks |
| `pp_remove_small_geoms` | Remove geometries smaller than this area threshold |

### Aggregator

| Parameter | Description |
|---|---|
| `nms_algorithm` | `iou` (detection or segmentation) or `ioa-disambiguate` (segmentation only) |
| `score_threshold` | Minimum score to keep a detection |
| `nms_threshold` | IoU threshold for suppression |
| `scores_weights` | Weight per score column for combined scoring |
| `scores_weighting_method` | How to combine scores (e.g. `weighted_geometric_mean`) |

### Classifier

*To be added.*
