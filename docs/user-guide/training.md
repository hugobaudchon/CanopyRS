# Training

We provide a `train.py` script to train detector and segmenter models on preprocessed datasets. You must download the datasets first (see [Data](data.md)).

## Prerequisites

Our training pipeline requires [wandb](https://wandb.ai/site) to be installed and configured for logging purposes.

## Detector training

To train a detector, copy and modify one of the config files under `canopyrs/config/detectors/`. For example, start from `dino_swinL_multi_NQOS.yaml`.

### Workflow

1. Download datasets:
   ```
   python -m canopyrs.tools.detection.download_datasets -d SelvaBox Detectree2 -o /data
   ```

2. Copy and modify a detector config:
   ```
   cp canopyrs/config/detectors/dino_swinL_multi_NQOS.yaml canopyrs/config/detectors/my_detector.yaml
   ```
   Edit `my_detector.yaml` and update the training-specific fields marked with `TODO`:

      - `data_root_path` — path to your dataset root folder
      - `train_output_path` — path for model checkpoints and logs
      - `train_dataset_names` / `valid_dataset_names` — location folders to use. See [Data](data.md) for more info on data structure
      - `wandb_project` — your wandb project name

3. Run training:

=== "Linux / macOS"

    ```bash
    python train.py \
      -m detector \
      -c canopyrs/config/detectors/my_detector.yaml
    ```

=== "Windows (PowerShell)"

    ```powershell
    python train.py `
      -m detector `
      -c canopyrs/config/detectors/my_detector.yaml
    ```

### Configuration reference

#### Model parameters

| Parameter | Description |
|---|---|
| `model` | Model type: `dino_detrex` for detrex-based DINO models or `faster_rcnn_detectron2` for detectron2-based Faster R-CNN models |
| `architecture` | Model architecture (see supported architectures below) |
| `checkpoint_path` | Path to pretrained model checkpoint. Keep our pretrained checkpoint to fine-tune, or replace with a [detrex](https://detrex.readthedocs.io/en/latest/tutorials/Model_Zoo.html) COCO checkpoint. If left as `null` for a detectron2 model (Faster R-CNN), it will download a pretrained COCO checkpoint automatically. |

#### Supported architectures

| Model type | Architecture |
|---|---|
| DINO (Swin-L) | `dino-swin/dino_swin_large_384_5scale_36ep.py` |
| DINO (ResNet-50) | `dino-resnet/dino_r50_4scale_24ep.py` |
| Faster R-CNN | `COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml` |

#### Data parameters

| Parameter | Description |
|---|---|
| `data_root_path` | Path to your dataset root folder (the `<DATA_ROOT>` folder where extracted datasets are located) |
| `train_dataset_names` | List of location folder names to train on |
| `valid_dataset_names` | List of location folder names to validate on |
| `train_output_path` | Path to output folder for model checkpoints and logs |
| `wandb_project` | Name of the wandb project to log to |

#### Dataset locations

**SelvaBox** has three locations:
- `brazil_zf2`
- `ecuador_tiputini`
- `panama_aguasalud`

**Detectree2** has one location:
- `malaysia_detectree2`

You can choose to train on all locations or a subset of them.

#### Other parameters

You can also modify parameters such as `batch_size`, `lr`, and more in the config file.

## Segmenter training

To train a segmenter, copy and modify one of the config files under `canopyrs/config/segmenters/`. For example, start from `mask2former_swinL_multi_selvamask.yaml`.

### Workflow

1. Download datasets:
   ```
   python -m canopyrs.tools.detection.download_datasets -d SelvaMask -o /data
   ```

2. Copy and modify a segmenter config:
   ```
   cp canopyrs/config/segmenters/mask2former_swinL_multi_selvamask.yaml canopyrs/config/segmenters/my_segmenter.yaml
   ```
   Edit `my_segmenter.yaml` and update the training-specific fields marked with `TODO`:

      - `data_root_path` — path to your dataset root folder
      - `train_output_path` — path for model checkpoints and logs
      - `train_dataset_names` / `valid_dataset_names` — location folders to use. See [Data](data.md) for more info on data structure
      - `wandb_project` — your wandb project name

3. Run training:

=== "Linux / macOS"

    ```bash
    python train.py \
      -m segmenter \
      -c canopyrs/config/segmenters/my_segmenter.yaml
    ```

=== "Windows (PowerShell)"

    ```powershell
    python train.py `
      -m segmenter `
      -c canopyrs/config/segmenters/my_segmenter.yaml
    ```

### Configuration reference

#### Model parameters

| Parameter | Description |
|---|---|
| `model` | Model type: `mask2former_detrex` for Mask2Former, `mask_rcnn_detectron2` for Mask R-CNN, or `sam3` for SAM 3 |
| `architecture` | Model architecture (see supported architectures below) |
| `checkpoint_path` | Path to pretrained model checkpoint. Keep our pretrained checkpoint to fine-tune, or replace with your own or a detrex/sam3 pretrained checkpoint. If left as `null` for a detectron2 model (Mask R-CNN), it will download a pretrained COCO checkpoint automatically. |

#### Supported architectures

| Model type | Architecture |
|---|---|
| Mask2Former (Swin-L) | `mask2former/configs/maskformer2_swin_large_IN21k_384_bs16_100ep.py` |
| Mask R-CNN (ResNet-50) | `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml` |
| SAM 3 | `l` |

#### Data parameters

| Parameter | Description |
|---|---|
| `data_root_path` | Path to your dataset root folder (the `<DATA_ROOT>` folder where extracted datasets are located) |
| `train_dataset_names` | List of location folder names to train on |
| `valid_dataset_names` | List of location folder names to validate on |
| `train_output_path` | Path to output folder for model checkpoints and logs |
| `wandb_project` | Name of the wandb project to log to |

#### Other parameters

You can also modify parameters such as `batch_size`, `lr`, and more in the config file.
