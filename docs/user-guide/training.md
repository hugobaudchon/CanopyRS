# Training

We provide a `train.py` script to train detector models on preprocessed datasets. You must download the datasets first (see [Data](data.md)).

## Prerequisites

Our training pipeline requires [wandb](https://wandb.ai/site) to be installed and configured for logging purposes.

## Configuration

To train a model, you need to copy and modify a `detector.yaml` config file. For example, copy from `config/default_detection_multi_NQOS_best/detector.yaml` and modify the following parameters:

### Model parameters

| Parameter | Description |
|---|---|
| `model` | Model type: `dino_detrex` for detrex-based DINO models or `faster_rcnn_detectron2` for detectron2-based Faster R-CNN models |
| `architecture` | Model architecture (see supported architectures below) |
| `checkpoint_path` | Path to pretrained model checkpoint. Keep our pretrained checkpoint to fine-tune, or replace with a [detrex](https://detrex.readthedocs.io/en/latest/tutorials/Model_Zoo.html) COCO checkpoint |

### Supported architectures

| Model type | Architecture |
|---|---|
| DINO (Swin-L) | `dino-swin/dino_swin_large_384_5scale_36ep.py` |
| DINO (ResNet-50) | `dino-resnet/dino_r50_4scale_24ep.py` |
| Faster R-CNN | `COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml` |

### Data parameters

| Parameter | Description |
|---|---|
| `data_root_path` | Path to your dataset root folder (the `<DATA_ROOT>` folder where extracted datasets are located) |
| `train_dataset_names` | List of location folder names to train on |
| `valid_dataset_names` | List of location folder names to validate on |
| `train_output_path` | Path to output folder for model checkpoints and logs |
| `wandb_project` | Name of the wandb project to log to |

### Dataset locations

**SelvaBox** has three locations:
- `brazil_zf2`
- `ecuador_tiputini`
- `panama_aguasalud`

**Detectree2** has one location:
- `malaysia_detectree2`

You can choose to train on all locations or a subset of them.

### Other parameters

You can also modify parameters such as `batch_size`, `lr`, and more in the config file.

## Running training

Run the training script with:

```bash
python train.py \
  -m detector \
  -c config/default_detection_multi_NQOS_best/detector.yaml
```

## Example workflow

1. Download datasets:
   ```bash
   python -m canopyrs.tools.detection.download_datasets -d SelvaBox Detectree2 -o /data/canopyrs
   ```

2. Copy and modify a detector config:
   ```bash
   cp -r config/default_detection_multi_NQOS_best config/my_custom_detector
   # Edit config/my_custom_detector/detector.yaml with your settings
   ```

3. Run training:
   ```bash
   python train.py -m detector -c config/my_custom_detector/detector.yaml
   ```
