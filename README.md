# CanopyRS

Canopy Remote Sensing is a project and pipeline designed for processing geospatial data, specifically for the purpose of detecting and segmenting trees in a forest. The pipeline includes several tasks such as tiling, detection, aggregation, and segmentation. Each task is configurable via YAML configuration files.

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

#### Windows
Coming soon

## Configuration

See config/

## Usage

### Inference

The main entry point of the pipeline is `main.py`. This script accepts command-line arguments specifying the task and subtask to perform, and the path to the configuration file.

Example run:

```bash
python infer.py -t pipeline -c default -i /path/to/raster.tif -o /path/to/output/folder
```

### Evaluation

Coming soon
