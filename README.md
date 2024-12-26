# CanopyRS

This project is a pipeline for processing geospatial data, specifically for the purpose of detecting and segmenting trees in a forest. The pipeline includes several tasks such as tiling, detection, aggregation, and segmentation. Each task is configurable via YAML configuration files.

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/hugobaudchon/CanopyRS.git
cd CanopyRS
```

Install the required Python packages:

```bash
pip install -r requirements.txt
pip install -r requirements2.txt
```

## Configuration

See config/

## Usage

The main entry point of the pipeline is `main.py`. This script accepts command-line arguments specifying the task and subtask to perform, and the path to the configuration file.

Example run:

```bash
python main.py -t pipeline -c default -i /path/to/raster.tif -o /path/to/output/folder
```

