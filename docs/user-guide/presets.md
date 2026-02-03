# Presets

CanopyRS ships with preset pipeline configurations for common use cases. Each preset is a folder under `canopyrs/config/` containing a `pipeline.yaml`.

## Available presets

### Detection only

| Preset | Model | GPU Memory | Resolution | Best for |
|---|---|---|---|---|
| `default_detection_single_S_low` | Faster R-CNN | ~4 GB | 10 cm/px | Quick tests, low-resource machines |
| `default_detection_single_S_medium` | DINO Swin-S | ~6 GB | 6 cm/px | Good balance of speed and quality |
| `default_detection_multi_NQOS_best` | DINO Swin-L-384 | ~10 GB | 4.5 cm/px | Highest detection quality |

### Detection + Segmentation

| Preset | Model | Best for |
|---|---|---|
| `default_segmentation_multi_NQOS_best` | DINO + SAM2 | General segmentation |
| `default_segmentation_multi_NQOS_best_S` | DINO + SAM2 | Small trees (recommended) |
| `default_segmentation_multi_NQOS_best_L` | DINO + SAM2 | Large trees (recommended) |

## How to use a preset

Pass the preset folder name to `infer.py`:

```bash
python infer.py --config default_segmentation_multi_NQOS_best_S --input image.tif --output ./out
```

## Customizing a preset

Copy a preset folder, edit its `pipeline.yaml`, and point `--config` to your copy. All inline parameters can be tweaked without changing any code.
