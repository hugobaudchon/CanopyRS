<h1 align="center">CanopyRS</h1>

![Project Title or Logo](assets/canopyrs_banner.png)

Canopy Remote Sensing is a project and pipeline designed for processing geospatial orthomosaics,
specifically for the purpose of detecting, segmenting and in the future classifying trees in a forest.
The pipeline includes several components such as tiling, detection, aggregation, and segmentation,
which can be chained one after the other depending on the application.

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/hugobaudchon/CanopyRS.git
cd CanopyRS
```

Install the required Python packages:

#### Linux
```bash
conda create -n canopyrs python=3.10
conda activate canopyrs
conda install -c conda-forge gdal
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/detectron2.git@70f4543
pip install git+https://github.com/IDEA-Research/detrex.git
```

[//]: # ()
[//]: # (#### Windows)

[//]: # (Coming soon)

## Configuration

Each component is configurable via YAML configuration files.

While default configuration files are provided in the `config` directory,
you can also create your own configuration files by creating a new folder under `config/`, adding a `pipeline.yaml` script,
and setup your desired list of component configuration files.

## Usage

### Inference

The main entry point of the inference pipeline is `infer.py`. 
This script accepts command-line arguments specifying the config to use and the input and output paths:

```bash
python infer.py -c <CONFIG_NAME> -i <INPUT_PATH> -o <OUTPUT_PATH>
```

We provide different default config files depending on your GPU resources:

| Config name                            | Description                                                                                                                                                                      |
|----------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `default_detection_multi_NQOS_best`    | The best multi-dataset, multi-resolution DINO + Swin L-384 model from our paper, trained on 4 datasets including SelvaBox. Best quality, and memory footprint is at about 10 GB. |
| `default_segmentation_multi_NQOS_best` | Same as `default_segmentation_multi_NQOS_best`, but with SAM2 chained after the detection model to provide instance segmentations. Best quality, and memory footprint is at about 10 GB.           |
| `default_detection_single_S_medium`    | A single resolution (6 cm/px) DINO + ResNet-50 model. Medium quality but faster and much lower memory footprint compared to models with Swin L-384 backbones.                    |
| `default_detection_single_S_low`       | A single resolution (10 cm/px) Faster R-CNN + ResNet-50 model. Worse quality, but even faster and even lower memory footprint.                                                   |

Example run for a single raster/orthomosaic (`-i`) with our default config:
```bash
python infer.py -c default_detection_multi_NQOS_best -i /path/to/raster.tif -o <OUTPUT_PATH>
```

Example run for a folder of tiles/images (`-t`) with our default config:
```bash
python infer.py -c default_detection_multi_NQOS_best -t /path/to/tiles/folder -o <OUTPUT_PATH>
```

### Data

In order to train or benchmark models, you will need data. In addition to SelvaBox, we provide 5 other pre-processed datasets:

- **SelvaBox** (~31.5 GB): https://huggingface.co/datasets/CanopyRS/SelvaBox
- **Detectree2** (~1.5 GB): https://huggingface.co/datasets/CanopyRS/Detectree2
- **NeonTreeEvaluation** (~3.3 GB): https://huggingface.co/datasets/CanopyRS/NeonTreeEvaluation
- **OAM-TCD** (~32.2 GB): https://huggingface.co/datasets/CanopyRS/OAM-TCD
- **BCI50ha** (~27.0 GB): https://huggingface.co/datasets/CanopyRS/BCI50ha
- **QuebecTrees** (~6.0 GB): https://huggingface.co/datasets/CanopyRS/QuebecTrees

To download and extract datasets automatically and use it with our benchmark or training scripts, we provide a tool.

For example, to download SelvaBox and Detectree2 datasets, you can use the following command:

```bash
python -m tools.detection.download_datasets \
  -d SelvaBox Detectree2 \
  -o <DATA_ROOT>
```

After extraction, they will be in COCO format (the same as geodataset's tilerizers output).

Your `<DATA_ROOT>` folder will contain one or more 'locations' folders, each containing individual 'rasters' folders, themsevles containing .json COCO annotations and tiles for minimum one fold (train, valid, test...).

For our SelvaBox and Detectree2 datasets example, the structure should look like this:

```
<DATA_ROOT>
├── brazil_zf2                         (-> Brazil location of SelvaBox)
│   ├── 20240130_zf2quad_m3m_rgb       (-> one of the Brazil location rasters for SelvaBox)
│   │   ├── tiles/
│   │   │  ├── valid/
│   │   │  │  ├── 20240130_zf2quad_m3m_rgb_tile_valid_1777_gr0p045_0_6216.tif
│   │   │  │  ├── ...
│   │   ├──  20240130_zf2quad_m3m_rgb_coco_gr0p045_valid.json
│   │   └──  ...
│   ├── 20240130_zf2tower_m3m_rgb
│   ├── 20240130_zf2transectew_m3m_rgb
│   └── 20240131_zf2campirana_m3m_rgb
├── ecuador_tiputini                   (-> Ecuador location of SelvaBox)
│   ├── ...
├── malaysia_detectree2                (-> Malaysia location of Detectree2)
│   ├── ...
└── panama_aguasalud                   (-> Panama location of SelvaBox)
```

Each additional dataset will add one or more locations folders.

### Evaluation

#### Find optimal NMS parameters for Raster-level evaluation ($RF1_{75}$)
To find the optimal NMS parameters for your model, i.e. `nms_iou_threshold` and `nms_score_threshold`,
you can use the `find_optimal_raster_nms.py` tool script. This script will run a grid search over the NMS parameters and evaluate the results using the COCO evaluation metrics.
Depending on how many Rasters there are in the datasets you select, it could take from a few tens of minutes to a few hours. If you have lots of CPU cores, we recommend to increase the number of workers.

You have to pass the path of a detection model config file, compatible with CanopyRS.

For example to find NMS parameters for the `default_detection_multi_NQOS_best` default model (DINO+Swin L-384 trained on NQOS datasets) on the validation set of SelvaBox and Detectree2 datasets,
you can use the following command (make sure to download the data first, see `Data` section):

```bash
python -m tools.detection.find_optimal_raster_nms \
  -c config/default_detection_multi_NQOS_best/detector.yaml \
  -d SelvaBox Detectree2 \
  -r <DATA_ROOT> \
  -o <OUTPUT_PATH> \
  --n_workers 6
```

For more information on parameters, you can use the `--help` flag:
```bash
python -m tools.detection.find_optimal_raster_nms --help
```

#### Benchmarking
To benchmark a model on the test or valid fold of some datasets, you can use the `benchmark.py` tool script. 


This script will run the model on the desired fold and evaluate the results using the COCO evaluation metrics (mAP and mAR).

If you provide `nms_threshold` and `score_threshold` parameters, it will also compute the $RF1_{75}$ metric by running an NMS at the raster level for datasets that have raster-level annotations.

```bash
python -m tools.detection.benchmark \
  -c config/default_detection_multi_NQOS_best/detector.yaml \
  -d SelvaBox Detectree2 \
  -r <DATA_ROOT> \
  -o <OUTPUT_PATH> \
  --nms_threshold 0.7 \
  --score_threshold 0.5
```

By default the evaluation is done on the test set. 
For more information on parameters, you can use the `--help` flag:
```bash
python -m tools.detection.benchmark --help
```
### Training

Coming soon
