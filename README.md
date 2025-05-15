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
python infer.py -c <config_name> -i <input_path> -o <output_path>
```

Example run for a single raster/orthomosaic (`-i`):
```bash
python infer.py -c default -i /path/to/raster.tif -o /path/to/output/folder
```

Example run for a folder of tiles/images (`-t`):
```bash
python infer.py -c default -t /path/to/tiles/folder -o /path/to/output/folder
```

### Data

In order to train or evaluate models, you will need data. In addition to SelvaBox, we provide 5 other pre-processed datasets:

- **SelvaBox**: TODO
- **Detectree2**: TODO
- **NeonTreeEvaluation**: TODO
- **OamTcd**: TODO
- **BCI50ha**: TODO
- **QuebecTrees**: TODO
To download and extract datasets, you can use one of our tools TODO

### Evaluation
#### Simple tile-level evaluation
Coming soon

TODO: expliquer comment telecharger modeles / changer config default et expliquer comment la modifier si necessaire

#### Benchmark for tile-level and raster-level

### Training

Coming soon
