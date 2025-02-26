#!/usr/bin/env python3
import os
import random
import time

import numpy as np
import yaml
import subprocess


NUM_RANDOM_SAMPLES = 30

# Grid search parameters
batch_sizes = [
    8, 
    16,
    32
]
max_epochs_list = [200, 500, 1000]
lr_min, lr_max = 1e-4, 1e-2

# Path to the base YAML config file
base_config_path = "experiences/resolution/detector_fasterrcnn.yaml"
# Directory to store the generated config files
config_dir = "experiences/resolution/grid_configs"
os.makedirs(config_dir, exist_ok=True)

# SLURM job script to call
sbatch_script = "sbatch_train_fasterrcnn_gridsearch.sh"

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

random.seed(42)
np.random.seed(42)


# Function to sample a learning rate from log-uniform distribution
def sample_log_uniform(min_val, max_val):
    # Sample uniformly in log space
    log_val = np.random.uniform(np.log10(min_val), np.log10(max_val))
    # Convert back to normal space
    val = 10 ** log_val
    # Round to a reasonable precision for learning rates
    return round(val, 6)


# Iterate over each dataset configuration
for dataset_config in dataset_configs:
    # For each dataset, generate N random hyperparameter combinations
    for i in range(NUM_RANDOM_SAMPLES):
        # Sample categorical parameters
        batch_size = random.choice(batch_sizes)
        max_epochs = random.choice(max_epochs_list)

        # Sample learning rate from log-uniform distribution
        lr = sample_log_uniform(lr_min, lr_max)

        # Create a copy of the base configuration
        config = base_config.copy()

        # Update hyperparameters
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
        config_filename = f"config_{variant_name}_{dataset_config['augmentation_image_size']}_bs{batch_size}_epochs{max_epochs}_lr{lr:.6f}_{timestamp}.yaml"
        config_path = os.path.join(config_dir, config_filename)

        # Save the modified configuration to file
        with open(config_path, "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

        # GPU selection logic using SLURM gres
        if batch_size == 32 and dataset_config["augmentation_image_size"] >= 1777:
            gres_arg = "--gres=gpu:A100l:1"
        else:
            gres_arg = "--gres=gpu:RTX8000:1"

        # Time request logic using SLURM --time flag
        if max_epochs == 200 or dataset_config["augmentation_image_size"] <= 800:
            time_arg = "--time=1-00:00:00"  # 1 day in D-HH:MM:SS format
        else:
            time_arg = "--time=2-00:00:00"  # 2 days

        # Build the sbatch command, passing the dataset compressed file name and config file path as arguments
        cmd = [
            "sbatch",
            gres_arg,
            time_arg,
            sbatch_script,
            dataset_config["compressed"],
            config_path
        ]
        print(f"[{i+1}/{NUM_RANDOM_SAMPLES}] Submitting job with command:")
        print(" ".join(cmd))
        print(f"  Parameters: batch_size={batch_size}, max_epochs={max_epochs}, lr={lr:.6f}, "
              f"gres_arg={gres_arg}, time_arg={time_arg}")

        # Submit the job
        subprocess.run(cmd)

        # Small delay to avoid overwhelming the scheduler
        time.sleep(1)
