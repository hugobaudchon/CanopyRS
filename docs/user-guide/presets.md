# Presets

CanopyRS ships with preset pipeline configurations for common use cases. Each preset is a folder under `canopyrs/config/` containing a `pipeline.yaml`.

## Available presets

We provide different default config files depending on your GPU resources and use case. You can find these config files in the `config/` folder—feel free to copy and adapt them to optimize inference on your data.

### Detection only

| Config name | Description |
|---|---|
| `default_detection_multi_NQOS_best` | The best model from our paper, a DINO + Swin L-384 trained on a mixture of multi-resolution datasets including SelvaBox. NMS hyper-parameters found using the RF1₇₅ metric. Best quality, ~10 GB GPU memory. |
| `default_detection_single_S_medium` | A single resolution (6 cm/px) DINO + ResNet-50 model. Medium quality but faster and much lower memory footprint compared to models with Swin L-384 backbones. |
| `default_detection_single_S_low` | A single resolution (10 cm/px) Faster R-CNN + ResNet-50 model. Worse quality, but even faster and even lower memory footprint. |

### Detection + Segmentation

| Config name | Description |
|---|---|
| `default_segmentation_multi_NQOS_best` | Same as `default_detection_multi_NQOS_best`, but with SAM2 chained after the detection model to provide instance segmentations. Best quality, ~10 GB GPU memory. |
| `default_segmentation_multi_NQOS_best_S` | **(Recommended)** Same as `default_segmentation_multi_NQOS_best`, but inference is optimized for smaller trees (up to ~15m), by using a lower score threshold before NMS, and tiles with smaller spatial extent and higher GSD (4cm/px). |
| `default_segmentation_multi_NQOS_best_L` | **(Recommended)** Same as `default_segmentation_multi_NQOS_best`, but inference is optimized for larger trees (up to ~60m), by using a lower score threshold before NMS, and tiles with larger spatial extent and lower GSD (7cm/px). |

## How to use a preset

Pass the preset folder name to `infer.py` with the `-c` flag:

```bash
python infer.py -c default_segmentation_multi_NQOS_best_S -i image.tif -o ./out
```

## Customizing a preset

Copy a preset folder, edit its `pipeline.yaml`, and point `-c` to your copy. All inline parameters can be tweaked without changing any code.
