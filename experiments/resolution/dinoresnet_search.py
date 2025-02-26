#!/usr/bin/env python3
import os
import time
import yaml
import subprocess
from itertools import product

# Grid search parameters
batch_sizes = [
    4,
    # 8,
    # 16
]
max_epochs_list = [200, 1000, 5000]
lrs = [1e-4, 5e-5]

# Path to the base YAML config file
base_config_path = "experiences/resolution/detector_dinoresnet.yaml"
# Directory to store the generated config files
config_dir = "experiences/resolution/grid_configs_dinoresnet"
os.makedirs(config_dir, exist_ok=True)

# SLURM job script to call
sbatch_script = "sbatch_train_dino_resnet_gridsearch.sh"

# Dataset configurations for three dataset variants
dataset_configs = [
    {
        "compressed": "ours_gr0p045_1777px.tar.gz",
        "train_dataset_names": [
            'tilerized_3555_0p5_0p045_None/panama_aguasalud',
            'tilerized_3555_0p5_0p045_None/ecuador_tiputini',
            'tilerized_3555_0p5_0p045_None/brazil_zf2'
        ],
        "valid_dataset_names": [
            'tilerized_1777_0p5_0p045_None/panama_aguasalud',
            'tilerized_1777_0p5_0p045_None/ecuador_tiputini',
            'tilerized_1777_0p5_0p045_None/brazil_zf2'
        ],
        "augmentation_image_size": 1777,
        "augmentation_train_crop_size_range": [1600, 1955]
    },
    {
        "compressed": "ours_gr0p06_1333px.tar.gz",
        "train_dataset_names": [
            'tilerized_2666_0p5_0p06_None/panama_aguasalud',
            'tilerized_2666_0p5_0p06_None/ecuador_tiputini',
            'tilerized_2666_0p5_0p06_None/brazil_zf2'
        ],
        "valid_dataset_names": [
            'tilerized_1333_0p5_0p06_None/panama_aguasalud',
            'tilerized_1333_0p5_0p06_None/ecuador_tiputini',
            'tilerized_1333_0p5_0p06_None/brazil_zf2'
        ],
        "augmentation_image_size": 1333,
        "augmentation_train_crop_size_range": [1200, 1466]
    },
    {
        "compressed": "ours_gr0p1_800px.tar.gz",
        "train_dataset_names": [
            'tilerized_1600_0p5_0p1_None/panama_aguasalud',
            'tilerized_1600_0p5_0p1_None/ecuador_tiputini',
            'tilerized_1600_0p5_0p1_None/brazil_zf2'
        ],
        "valid_dataset_names": [
            'tilerized_800_0p5_0p1_None/panama_aguasalud',
            'tilerized_800_0p5_0p1_None/ecuador_tiputini',
            'tilerized_800_0p5_0p1_None/brazil_zf2'
        ],
        "augmentation_image_size": 800,
        "augmentation_train_crop_size_range": [720, 880]
    },


    # for showing that its lower resolution that drops performance, not smaller images
    {
        "compressed": "ours_gr0p06_1333px.tar.gz",
        "train_dataset_names": [
            'tilerized_2666_0p5_0p06_None/panama_aguasalud',
            'tilerized_2666_0p5_0p06_None/ecuador_tiputini',
            'tilerized_2666_0p5_0p06_None/brazil_zf2'
        ],
        "valid_dataset_names": [
            'tilerized_1333_0p5_0p06_None/panama_aguasalud',
            'tilerized_1333_0p5_0p06_None/ecuador_tiputini',
            'tilerized_1333_0p5_0p06_None/brazil_zf2'
        ],
        "augmentation_image_size": 1777,
        "augmentation_train_crop_size_range": [1200, 1466]
    },
    {
        "compressed": "ours_gr0p1_800px.tar.gz",
        "train_dataset_names": [
            'tilerized_1600_0p5_0p1_None/panama_aguasalud',
            'tilerized_1600_0p5_0p1_None/ecuador_tiputini',
            'tilerized_1600_0p5_0p1_None/brazil_zf2'
        ],
        "valid_dataset_names": [
            'tilerized_800_0p5_0p1_None/panama_aguasalud',
            'tilerized_800_0p5_0p1_None/ecuador_tiputini',
            'tilerized_800_0p5_0p1_None/brazil_zf2'
        ],
        "augmentation_image_size": 1333,
        "augmentation_train_crop_size_range": [720, 880]
    },
    {
        "compressed": "ours_gr0p1_800px.tar.gz",
        "train_dataset_names": [
            'tilerized_1600_0p5_0p1_None/panama_aguasalud',
            'tilerized_1600_0p5_0p1_None/ecuador_tiputini',
            'tilerized_1600_0p5_0p1_None/brazil_zf2'
        ],
        "valid_dataset_names": [
            'tilerized_800_0p5_0p1_None/panama_aguasalud',
            'tilerized_800_0p5_0p1_None/ecuador_tiputini',
            'tilerized_800_0p5_0p1_None/brazil_zf2'
        ],
        "augmentation_image_size": 1777,
        "augmentation_train_crop_size_range": [720, 880]
    },
]

# Load the base YAML configuration
with open(base_config_path, "r") as f:
    base_config = yaml.safe_load(f)

# Iterate over each dataset configuration and grid search combinations
for dataset_config in dataset_configs:
    for batch_size, max_epochs, lr in product(batch_sizes, max_epochs_list, lrs):
        # Create a copy of the base configuration
        config = base_config.copy()

        # Update grid search parameters
        config["batch_size"] = batch_size
        config["max_epochs"] = max_epochs
        config["lr"] = lr

        # Update dataset-specific parameters
        config["train_dataset_names"] = dataset_config["train_dataset_names"]
        config["valid_dataset_names"] = dataset_config["valid_dataset_names"]
        config["augmentation_image_size"] = dataset_config["augmentation_image_size"]
        config["augmentation_train_crop_size_range"] = dataset_config["augmentation_train_crop_size_range"]

        # Create a unique filename using a timestamp and parameter values
        timestamp = int(time.time())
        # Using the compressed file name (without extension) to indicate the dataset variant
        variant_name = dataset_config["compressed"].split('.')[0]
        config_filename = f"config_{variant_name}_{dataset_config["augmentation_image_size"]}_bs{batch_size}_epochs{max_epochs}_lr{lr}_{timestamp}.yaml"
        config_path = os.path.join(config_dir, config_filename)

        # Save the modified configuration to file
        with open(config_path, "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

        # Build the sbatch command, passing the dataset compressed file name and config file path as arguments.
        cmd = ["sbatch", sbatch_script, dataset_config["compressed"], config_path]
        print("Submitting job with command:", " ".join(cmd))
        
        # Submit the job
        subprocess.run(cmd)
