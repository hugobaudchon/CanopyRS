# CanopyRS

This project is a pipeline for processing geospatial data, specifically for the purpose of detecting and segmenting trees in a forest. The pipeline includes several tasks such as tiling, detection, aggregation, and segmentation. Each task is configurable via YAML configuration files.

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
```

#### Windows
On Windows, we need to install torch first to have GPU support. You may change the index url to your installed cuda version (see [pytorch](https://pytorch.org/get-started/locally/)).
```bash
conda create -n canopyrs python=3.10
conda activate canopyrs
conda install -c conda-forge gdal
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Configuration

See config/

## Usage

### Inference

The main entry point of the pipeline is `main.py`. This script accepts command-line arguments specifying the task and subtask to perform, and the path to the configuration file.

Example run:

```bash
python main.py -t pipeline -c default -i /path/to/raster.tif -o /path/to/output/folder
```

### Evaluation

Similar as inference, but now we input a ground truth geopackage (optional), an aoi geopackage, and use a different pipeline (default_eval):

```bash
python main.py -t pipeline -c default_eval -i /path/to/raster.tif -gt /path/to/groundtruth/geopackage.gpkg -aoi /path/to/aoi/geopackage.gpkg -o /path/to/output/folder
```
