#!/usr/bin/env python3
import os
import random
import time

import numpy as np
import yaml
import subprocess
from itertools import product


# Grid search parameters
seeds_list = [
    1,
    # 2,
    # 3
]
batch_sizes = [
    4,
    8, 
    16,
    # 32
]
max_epochs_list = [
    200,
    500
]                   # TODO change this to a final unique value
box_nms_thresh_list = [
    0.5,
    # 0.6,
    # 0.7
]

scheduler_type = "WarmupCosineLR" # "WarmupMultiStepLR"
lrs_list = [
    # 5e-4,
    1e-3,
    2e-3,
    5e-3
    # 1e-2
]

# Path to the base YAML config file
base_config_path = "experiments/resolution/detector_fasterrcnn.yaml"
# Directory to store the generated config files
config_dir = "experiments/resolution/grid_configs_fasterrcnn"
os.makedirs(config_dir, exist_ok=True)

# SLURM job script to call
sbatch_script = "sbatch_train_fasterrcnn_gridsearch.sh"

# Dataset configurations for three dataset variants
dataset_configs = [
    # {
    #     "compressed": "quebectrees_gr0p06_1333px.tar.gz",
    #     "train_dataset_names": [
    #         'tilerized_2666_0p5_0p06_None/quebec_trees',
    #     ],
    #     "valid_dataset_names": [
    #         'tilerized_1333_0p5_0p06_None/quebec_trees',
    #     ],
    #     "augmentation_image_size": 1333,
    #     "augmentation_train_crop_size_range": [1200, 1466]
    # },
    {
        "compressed": "quebectrees_gr0p06_1333px.tar.gz",
        "train_dataset_names": [
            'tilerized_2666_0p5_0p06_None/quebec_trees',
        ],
        "valid_dataset_names": [
            'tilerized_1333_0p5_0p06_None/quebec_trees',
        ],
        "augmentation_image_size": 1333,
        "augmentation_train_crop_size_range": [550, 1466]
    },
    # {
    #     "compressed": "quebectrees_gr0p025_1333px.tar.gz",
    #     "train_dataset_names": [
    #         'tilerized_2666_0p5_0p025_None/quebec_trees',
    #     ],
    #     "valid_dataset_names": [
    #         'tilerized_1333_0p5_0p025_None/quebec_trees',
    #     ],
    #     "augmentation_image_size": 1333,
    #     "augmentation_train_crop_size_range": [1200, 1466]
    # },
]

# Load the base YAML configuration
with open(base_config_path, "r") as f:
    base_config = yaml.safe_load(f)


# # Function to sample a learning rate from log-uniform distribution
# def sample_log_uniform(min_val, max_val):
#     # Sample uniformly in log space
#     log_val = np.random.uniform(np.log10(min_val), np.log10(max_val))
#     # Convert back to normal space
#     val = 10 ** log_val
#     # Round to a reasonable precision for learning rates
#     return round(val, 6)


# Iterate over each dataset configuration
for dataset_config in dataset_configs:
    # For each dataset, generate N random hyperparameter combinations
    combinations = product(batch_sizes, max_epochs_list, lrs_list, box_nms_thresh_list, seeds_list)
    for i, (batch_size, max_epochs, lr, box_nms_thresh, seed) in enumerate(combinations):
        # Create a copy of the base configuration
        config = base_config.copy()

        # Update hyperparameters
        config["batch_size"] = batch_size
        config["max_epochs"] = max_epochs
        config["scheduler_type"] = scheduler_type
        config["lr"] = lr
        config["scheduler_epochs_steps"] = [int(max_epochs*0.8), int(max_epochs*0.9)]
        config["box_nms_thresh"] = box_nms_thresh
        config["seed"] = seed

        # Update dataset-specific parameters
        config["train_dataset_names"] = dataset_config["train_dataset_names"]
        config["valid_dataset_names"] = dataset_config["valid_dataset_names"]
        config["augmentation_image_size"] = dataset_config["augmentation_image_size"]
        config["augmentation_train_crop_size_range"] = dataset_config["augmentation_train_crop_size_range"]

        # Create a unique filename using a timestamp and parameter values
        timestamp = int(time.time())
        # Using the compressed file name (without extension) to indicate the dataset variant
        variant_name = dataset_config["compressed"].split('.')[0]
        config_filename = f"config_{variant_name}_{dataset_config['augmentation_image_size']}_bs{batch_size}_epochs{max_epochs}_lr{lr:.6f}_nmsthresh{box_nms_thresh}_{timestamp}.yaml"
        config_path = os.path.join(config_dir, config_filename)

        # Save the modified configuration to file
        with open(config_path, "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

        # GPU selection logic using SLURM gres
        if batch_size == 32 and dataset_config["augmentation_image_size"] >= 1777:
            gres_arg = "--gres=gpu:a100l:1"
        else:
            gres_arg = "--gres=gpu:rtx8000:1"

        # Time request logic using SLURM --time flag
        if max_epochs <= 200 and dataset_config["augmentation_image_size"] <= 1333:
            time_arg = "--time=1-00:00:00"  # 1 day in D-HH:MM:SS format
        elif dataset_config["augmentation_image_size"] <= 1333 or (max_epochs <= 200 and dataset_config["augmentation_image_size"] <= 1777):
            time_arg = "--time=2-00:00:00"  # 2 days
        else:
            time_arg = "--time=3-00:00:00"  # 3 days

        # Build the sbatch command, passing the dataset compressed file name and config file path as arguments
        cmd = [
            "sbatch",
            gres_arg,
            time_arg,
            sbatch_script,
            dataset_config["compressed"],
            config_path
        ]
        print(f"Submitting job with command:")
        print(" ".join(cmd))
        print(f"  Parameters: batch_size={batch_size}, max_epochs={max_epochs}, lr={lr:.6f}, "
              f"gres_arg={gres_arg}, time_arg={time_arg}")

        # Submit the job
        subprocess.run(cmd)

        # Small delay to avoid overwhelming the scheduler
        time.sleep(1)
