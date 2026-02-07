# Model Zoo

This page lists every model supported by CanopyRS, grouped by pipeline stage.
If you just want a ready-made configuration, see [Presets](presets.md).
If you want to build a custom pipeline, use the tables below to pick models for each stage and reference them in your `pipeline.yaml`.

All checkpoints are **downloaded automatically** the first time a model is used.

---

## Detectors

Detectors produce bounding boxes around individual trees.

| Config file | Architecture | Training data | Checkpoint | Config |
|---|---|---|---|---|
| `dino_swinL_multi_NQOS` | DINO&nbsp;+&nbsp;Swin&#8209;L&nbsp;384 | Multi-resolution, multi-dataset (NeonTrees, QuebecTrees, OAM-TCD, SelvaBox) | [HuggingFace (CanopyRS)](https://huggingface.co/CanopyRS/dino-swin-l-384-multi-NQOS) | [YAML](https://github.com/hugobaudchon/CanopyRS/blob/main/canopyrs/config/detectors/dino_swinL_multi_NQOS.yaml) |
| `dino_swinL_multi_NQOS_selvamask_FT` | DINO&nbsp;+&nbsp;Swin&#8209;L&nbsp;384 | Multi-resolution, multi-dataset (NeonTrees, QuebecTrees, OAM-TCD, SelvaBox), fine-tuned on SelvaMask | [HuggingFace (CanopyRS)](https://huggingface.co/CanopyRS/dino-swin-l-384-multi-NQOS-selvamask-FT) | [YAML](https://github.com/hugobaudchon/CanopyRS/blob/main/canopyrs/config/detectors/dino_swinL_multi_NQOS_selvamask_FT.yaml) |
| `dino_r50_single_S` | DINO&nbsp;+&nbsp;ResNet&#8209;50 | SelvaBox (single resolution, 6 cm/px) | [HuggingFace (CanopyRS)](https://huggingface.co/CanopyRS/dino-resnet50-single-6p0cm-S) | [YAML](https://github.com/hugobaudchon/CanopyRS/blob/main/canopyrs/config/detectors/dino_r50_single_S.yaml) |
| `fasterrcnn_r50_single_S` | Faster&nbsp;R&#8209;CNN&nbsp;+&nbsp;ResNet&#8209;50 | SelvaBox (single resolution, 10 cm/px) | [HuggingFace (CanopyRS)](https://huggingface.co/CanopyRS/fasterrcnn-resnet50-single-10p0cm-S) | [YAML](https://github.com/hugobaudchon/CanopyRS/blob/main/canopyrs/config/detectors/fasterrcnn_r50_single_S.yaml) |

#### External models

These detectors come from third-party projects. CanopyRS wraps them so they can be used as pipeline components.

| Config file | Architecture | Project | Training data | Checkpoint | Config |
|---|---|---|---|---|---|
| `deepforest` | RetinaNet&nbsp;+&nbsp;ResNet&#8209;50 | [DeepForest](https://github.com/weecology/DeepForest) | NeonTrees | [DeepForest](https://github.com/weecology/DeepForest/releases) | [YAML](https://github.com/hugobaudchon/CanopyRS/blob/main/canopyrs/config/detectors/deepforest.yaml) |

---

## Segmenters

Segmenters produce per-tree instance masks. They can be **prompted** (fed bounding boxes from a detector) or **unprompted** (run directly on tiles).

### Prompted segmenters

These models take bounding boxes as input and produce a mask for each box. Chain them after a detector.

| Config file | Architecture | Training data | Checkpoint | Config |
|---|---|---|---|---|
| `sam2_L` | SAM&nbsp;2&nbsp;Large | SA-1B (foundation model) | [Meta](https://github.com/facebookresearch/sam2) | [YAML](https://github.com/hugobaudchon/CanopyRS/blob/main/canopyrs/config/segmenters/sam2_L.yaml) |
| `sam3_multi_selvamask_FT` | SAM&nbsp;3 | SA-1B, fine-tuned on SelvaMask | [HuggingFace (CanopyRS)](https://huggingface.co/CanopyRS/sam3-multi-selvabox-selvamask-FT) | [YAML](https://github.com/hugobaudchon/CanopyRS/blob/main/canopyrs/config/segmenters/sam3_multi_selvamask_FT.yaml) |

### Unprompted segmenters

These models perform end-to-end instance segmentation without requiring bounding box prompts.

| Config file | Architecture | Training data | Checkpoint | Config |
|---|---|---|---|---|
| `mask2former_swinL_multi_selvamask` | Mask2Former&nbsp;+&nbsp;Swin&#8209;L | SelvaMask | [HuggingFace (CanopyRS)](https://huggingface.co/CanopyRS/mask2former-swin-l-384-multi-selvamask) | [YAML](https://github.com/hugobaudchon/CanopyRS/blob/main/canopyrs/config/segmenters/mask2former_swinL_multi_selvamask.yaml) |
| `maskrcnn_r50_multi_selvamask` | Mask&nbsp;R&#8209;CNN&nbsp;+&nbsp;ResNet&#8209;50 | SelvaMask | [HuggingFace (CanopyRS)](https://huggingface.co/CanopyRS/maskrcnn-resnet50-multi-selvamask) | [YAML](https://github.com/hugobaudchon/CanopyRS/blob/main/canopyrs/config/segmenters/maskrcnn_r50_multi_selvamask.yaml) |

#### External models

These segmenters come from third-party projects. CanopyRS wraps them so they can be used as pipeline components.

| Config file | Architecture | Project | Training data | Checkpoint | Config |
|---|---|---|---|---|---|
| `detectree2_flexi` | Mask&nbsp;R&#8209;CNN&nbsp;+&nbsp;ResNet&#8209;101 | [Detectree2](https://github.com/PatBall1/detectree2) | Detectree2 + urban data | [Zenodo](https://zenodo.org/records/15863800) | [YAML](https://github.com/hugobaudchon/CanopyRS/blob/main/canopyrs/config/segmenters/detectree2_flexi.yaml) |
| `detectree2_randresizefull` | Mask&nbsp;R&#8209;CNN&nbsp;+&nbsp;ResNet&#8209;101 | [Detectree2](https://github.com/PatBall1/detectree2) | Detectree2 | [Zenodo](https://zenodo.org/records/15863800) | [YAML](https://github.com/hugobaudchon/CanopyRS/blob/main/canopyrs/config/segmenters/detectree2_randresizefull.yaml) |

---

## Using a model in a custom pipeline

To use any model listed above, reference its config in your `pipeline.yaml`:

**Detector:**

```yaml
- detector: detectors/dino_swinL_multi_NQOS
```

**Segmenter:**

```yaml
- segmenter: segmenters/sam2_L
```

See [Configuration](configuration.md) for the full list of parameters you can override.
