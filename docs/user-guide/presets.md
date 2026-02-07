# Presets

CanopyRS ships with preset pipeline configurations for common use cases. Each preset is a YAML file under `canopyrs/config/pipelines/`.

## Available presets

We provide different preset configs depending on your GPU resources and use case. You can find these config files in `canopyrs/config/pipelines/`—feel free to copy and adapt them to optimize inference on your data.

### Instance Segmentation (SAM&nbsp;3)

| Config file | Architecture | Train dataset(s) | Performance/Requirements | Quality | Description |
|---|---|---|---|---|---|
| `preset_seg_multi_NQOS_selvamask_SAM3_FT_quality.yaml` | DINO&nbsp;+&nbsp;Swin&#8209;L&nbsp;384&nbsp;+&nbsp;SAM&nbsp;3 | Multi-resolution, multi-dataset (NeonTrees, QuebecTrees, OAM-TCD, SelvaBox), fine-tuned on SelvaMask | ~10 GB GPU memory | Best | **(Recommended for best quality)** SelvaMask fine-tuned detector and SAM 3 segmenter at high resolution (4.5 cm/px GSD) for best quality. |
| `preset_seg_multi_NQOS_selvamask_SAM3_FT_fast.yaml` | DINO&nbsp;+&nbsp;Swin&#8209;L&nbsp;384&nbsp;+&nbsp;SAM&nbsp;3 | Multi-resolution, multi-dataset (NeonTrees, QuebecTrees, OAM-TCD, SelvaBox), fine-tuned on SelvaMask | ~10 GB GPU memory | High | **(Recommended for a bit faster, high-quality inference)** SelvaMask fine-tuned detector and SAM 3 segmenter at lower resolution (7 cm/px GSD) and reduced tile overlap for faster inference. |

### Instance Segmentation (SAM&nbsp;2)

| Config file | Architecture | Train dataset(s) | Performance/Requirements | Quality | Description |
|---|---|---|---|---|---|
| `preset_seg_multi_NQOS_SAM2.yaml` | DINO&nbsp;+&nbsp;Swin&#8209;L&nbsp;384&nbsp;+&nbsp;SAM&nbsp;2 | Multi-resolution, multi-dataset (NeonTrees, QuebecTrees, OAM-TCD, SelvaBox) | ~10 GB GPU memory | Best | Same detector as `preset_det_multi_NQOS_dino_swinL.yaml`, with SAM 2 chained after detection for instance segmentations. |
| `preset_seg_multi_NQOS_SAM2_smalltrees.yaml` | DINO&nbsp;+&nbsp;Swin&#8209;L&nbsp;384&nbsp;+&nbsp;SAM&nbsp;2 | Multi-resolution, multi-dataset (NeonTrees, QuebecTrees, OAM-TCD, SelvaBox) | ~10 GB GPU memory | Best | **(Recommended for small trees)** Optimized for smaller trees (up to ~15 m) with 4 cm/px GSD and smaller tiles. |
| `preset_seg_multi_NQOS_SAM2_largetrees.yaml` | DINO&nbsp;+&nbsp;Swin&#8209;L&nbsp;384&nbsp;+&nbsp;SAM&nbsp;2 | Multi-resolution, multi-dataset (NeonTrees, QuebecTrees, OAM-TCD, SelvaBox) | ~10 GB GPU memory | Best | **(Recommended for large trees)** Optimized for larger trees (up to ~60 m) with 7 cm/px GSD and larger tiles. |

### Instance Detection only

| Config file | Architecture | Train dataset(s) | Performance/Requirements | Quality | Description |
|---|---|---|---|---|---|
| `preset_det_multi_NQOS_dino_swinL.yaml` | DINO&nbsp;+&nbsp;Swin&#8209;L&nbsp;384 | Multi-resolution, multi-dataset (NeonTrees, QuebecTrees, OAM-TCD, SelvaBox) | ~10 GB GPU memory | Best | The best detection model from our paper. NMS hyper-parameters found using the RF1₇₅ metric. |
| `preset_det_single_S_dino_r50.yaml` | DINO&nbsp;+&nbsp;ResNet&#8209;50 | SelvaBox (single resolution, 6 cm/px) | Faster, lower memory | Medium | Single resolution model with lower memory footprint compared to Swin L-384 backbones. |
| `preset_det_single_S_fasterrcnn_r50.yaml` | Faster&nbsp;R&#8209;CNN&nbsp;+&nbsp;ResNet&#8209;50 | SelvaBox (single resolution, 10 cm/px) | Fastest, lowest memory | Low | Faster and lower memory footprint but lower quality. |

## How to use a preset

Pass the preset file name to `infer.py` with the `-c` flag:

```bash
python infer.py -c preset_seg_multi_NQOS_SAM2_smalltrees.yaml -i <PATH_TO_TIF> -o <PATH_TO_OUTPUT_FOLDER>
```

## Customizing a preset

Copy a preset YAML file, edit it, and point `-c` to your copy. All inline parameters can be tweaked without changing any code. See [Configuration](configuration.md) for further details on how to build your own pipeline.
