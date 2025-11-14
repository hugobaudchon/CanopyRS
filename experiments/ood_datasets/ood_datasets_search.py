#!/usr/bin/env python3
import math
import os
import time
import yaml
import subprocess
from itertools import product

from engine.config_parsers.detector import DetectorConfig

# Grid search parameters


model_architecture_list = [
    "fasterrcnn",
    # "retinanet",
    # "dinoresnet",
    # "dinoswin"
]

datasets_configs_to_train = [
    # 'neon_trees',
    # 'ours',
    # 'all_datasets',
    'all_datasets_but_ours',
    # 'all_datasets_but_ours_and_oamtcd'
]

seeds_list = [
    # 1,
    33,
    42
]

batch_sizes = [
    4, 
    # 8,
    # 16
]
max_epochs_list = [
    # 40,
    # 80,
    # 120,
    200,
    # 500,
    # 1000
]
scheduler_type = "WarmupCosineLR"  # "WarmupCosineLR" or "WarmupMultiStepLR", "WarmupExponentialLR"
scheduler_gamma = 0.1 #0.1
scheduler_epochs_steps_percents = [0.8, 0.9]   # [0.8, 0.9]
lrs = [
    # fasterrcnn
    # 1e-3,
    # 2e-3,
    5e-3,

    # dino
    # 5e-4,
    # 1e-4,
    # 5e-5
]

augmentation_drop_annotation_random_prob_list = [
    0,                                                      # SHOULD USUALLY HAVE 0, more than 0 is just for specific experiments for rebuttal of the paper!!!
    # 0.2,
    # 0.4,
    # 0.6,
    # 0.8,
    # 0.9,
    # 0.95
]

base_path = 'experiments/ood_datasets'
experience_name = f'detector_experience_OOD_rebuttal_extra'
train_output_path = f'/network/scratch/h/hugo.baudchon/training/{experience_name}'
wandb_project = experience_name

partition_arg = "--partition=long"

# Mapping from architecture string to its corresponding config paths and sbatch script
arch_to_config = {
    "fasterrcnn": {
        "model": "faster_rcnn_detectron2",
        "architecture": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        "checkpoint_path": None  # checkpoint is automatically gathered by detectron2
    },
    "retinanet": {
        "model": "retinanet_detectron2",
        "architecture": "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
        "checkpoint_path": None  # checkpoint is automatically gathered by detectron2
    },
    "dinoresnet": {
        "model": "dino_detrex",
        "architecture": "dino-resnet/dino_r50_4scale_24ep.py",
        "checkpoint_path": "/home/mila/h/hugo.baudchon/CanopyRS/pretrained_models/dino_r50_4scale_24ep.pth"
    },
    "dinoswin": {
        "model": "dino_detrex",
        "architecture": "dino-swin/dino_swin_large_384_5scale_36ep.py",
        "checkpoint_path": "/home/mila/h/hugo.baudchon/CanopyRS/pretrained_models/dino_swin_large_384_5scale_36ep.pth"
    }
}

# Common dataset configurations (merged from your scripts)
dataset_configs = {
    "all_datasets": {
        "train_dataset_names": [
            "extracted/brazil_zf2",
            "extracted/ecuador_tiputini",
            "extracted/panama_aguasalud",
            "extracted/quebec_trees",
            "extracted/unitedstates_neon",
            "extracted/global_oamtcd"
        ],
        "valid_dataset_names": [
            "extracted/brazil_zf2",
            "extracted/ecuador_tiputini",
            "extracted/panama_aguasalud",
            "extracted/quebec_trees",
            "extracted/unitedstates_neon",
            "extracted/global_oamtcd"
        ],
        "augmentation_early_conditional_image_size": 2000,  # resizes neon tree train images from 1200 to 2000 to then being able to crop at as low as 666 pixels = 40m
        "augmentation_image_size": [1024, 1777],
        "augmentation_train_crop_size_range": [666, 2666]
    },
    "all_datasets_but_ours": {
        "train_dataset_names": [
            "extracted/quebec_trees",
            "extracted/unitedstates_neon",
            "extracted/global_oamtcd"
        ],
        "valid_dataset_names": [
            "extracted/quebec_trees",
            "extracted/unitedstates_neon",
            "extracted/global_oamtcd"
        ],
        "augmentation_early_conditional_image_size": 2000,
        "augmentation_image_size": [1024, 1777],
        "augmentation_train_crop_size_range": [666, 2666]
    },
    "all_datasets_but_ours_and_oamtcd": {
        "train_dataset_names": [
            "extracted/quebec_trees",
            "extracted/unitedstates_neon",
        ],
        "valid_dataset_names": [
            "extracted/quebec_trees",
            "extracted/unitedstates_neon",
        ],
        "augmentation_early_conditional_image_size": 2000,
        "augmentation_image_size": [1024, 1777],
        "augmentation_train_crop_size_range": [666, 2666]
    },
    "ours": {
        "train_dataset_names": [
            "extracted/brazil_zf2",
            "extracted/ecuador_tiputini",
            "extracted/panama_aguasalud",
        ],
        "valid_dataset_names": [
            "extracted/brazil_zf2",
            "extracted/ecuador_tiputini",
            "extracted/panama_aguasalud",
        ],
        "augmentation_early_conditional_image_size": 2000,
        "augmentation_image_size": [1024, 1777],
        "augmentation_train_crop_size_range": [666, 2666]
    },
    "neon_trees": {
        "train_dataset_names": [
            "extracted/unitedstates_neon",
        ],
        "valid_dataset_names": [
            "extracted/unitedstates_neon",
        ],
        "augmentation_early_conditional_image_size": 2000,
        "augmentation_image_size": [1024, 1777],
        "augmentation_train_crop_size_range": [666, 2666]
    }
}


def select_sbatch_args(arch, batch_size, aug_img_size, max_epochs):
    """
    Returns a tuple (gres_arg, time_arg, cpus_arg, mem_arg)
    based on the current model architecture and parameters.
    """
    max_aug_img_size = max(aug_img_size)
    if arch == "fasterrcnn" or arch == "retinanet":
        if batch_size == 32 and max_aug_img_size >= 1777:
            gres_arg = "--gres=gpu:a100l:1"
        elif max_aug_img_size >= 1777:
            gres_arg = "--gres=gpu:rtx8000:2"
        else:
            gres_arg = "--gres=gpu:rtx8000:1"
        if max_epochs <= 200 and max_aug_img_size <= 1333:
            time_arg = "--time=1-00:00:00"
        elif max_aug_img_size <= 1333 or (max_epochs <= 200 and max_aug_img_size <= 1777):
            time_arg = "--time=2-00:00:00"
        else:
            time_arg = "--time=3-00:00:00"
        cpus_arg = "--cpus-per-task=4"
        mem_arg = "--mem=40G"
        return gres_arg, time_arg, cpus_arg, mem_arg
    elif arch == "dinoresnet":
        if (batch_size >= 8 and max_aug_img_size >= 1333) or max_aug_img_size >= 1777:
            gres_arg = "--gres=gpu:rtx8000:2"
        elif max_aug_img_size >= 1333:
            gres_arg = "--gres=gpu:l40s:1"
        else:
            gres_arg = "--gres=gpu:rtx8000:1"
        time_arg = "--time=16:00:00"
        cpus_arg = "--cpus-per-task=4"
        mem_arg = "--mem=40G"
        return gres_arg, time_arg, cpus_arg, mem_arg
    elif arch == "dinoswin":
        if batch_size >= 8 or max_aug_img_size >= 1777:
            gres_arg = "--gres=gpu:rtx8000:4"
            cpus_arg = "--cpus-per-task=8"
            mem_arg = "--mem=40G"
        else:
            gres_arg = "--gres=gpu:l40s:2"
            cpus_arg = "--cpus-per-task=6"
            mem_arg = "--mem=40G"

        # time_hours = math.ceil(1 * max_epochs)
        # # split into days and hours
        # days, hours = divmod(time_hours, 24)
        # # build Slurm time string
        # if days > 0:
        #     time_arg = f"--time={days}-{hours:02d}:00:00"
        # else:
        #     time_arg = f"--time={hours}:00:00"
        time_arg = "--time=2-12:00:00"
        return gres_arg, time_arg, cpus_arg, mem_arg


# Main grid search loop over each architecture, dataset config, and hyperparameter combination
for train_dataset_config in datasets_configs_to_train:
    dataset_config = dataset_configs[train_dataset_config]

    for arch in model_architecture_list:
        # Retrieve config information for this architecture
        arch_info = arch_to_config[arch]
        base_config_path = f"{base_path}/base_config.yaml"
        config_dir = f"{base_path}/{experience_name}"
        os.makedirs(config_dir, exist_ok=True)

        # Load the base YAML configuration for the current architecture
        with open(base_config_path, "r") as f:
            base_config = yaml.safe_load(f)

        # Loop over each dataset configuration and grid search combination
        for batch_size, max_epochs, lr, seed, augmentation_drop_annotation_random_prob in product(batch_sizes, max_epochs_list, lrs, seeds_list, augmentation_drop_annotation_random_prob_list):
            # Create a copy of the base configuration
            config = base_config.copy()

            # Update outputs
            config["train_output_path"] = train_output_path
            config["wandb_project"] = wandb_project

            # Update hyperparameters
            config["batch_size"] = batch_size
            config["max_epochs"] = max_epochs
            config["scheduler_type"] = scheduler_type
            config["lr"] = lr
            config["scheduler_epochs_steps"] = [int(max_epochs * perc) for perc in scheduler_epochs_steps_percents]
            config["scheduler_gamma"] = scheduler_gamma
            config["seed"] = seed
            config["augmentation_drop_annotation_random_prob"] = augmentation_drop_annotation_random_prob

            # Include the model architecture in the configuration
            config["model"] = arch_info["model"]
            config["architecture"] = arch_info["architecture"]
            config["checkpoint_path"] = arch_info["checkpoint_path"]

            # Update dataset–specific parameters
            config["train_dataset_names"] = dataset_config["train_dataset_names"]
            config["valid_dataset_names"] = dataset_config["valid_dataset_names"]
            config["augmentation_image_size"] = dataset_config["augmentation_image_size"]
            config["augmentation_train_crop_size_range"] = dataset_config["augmentation_train_crop_size_range"]
            config["augmentation_early_conditional_image_size"] = dataset_config["augmentation_early_conditional_image_size"]

            # Create a unique filename (including the architecture name)
            timestamp = int(time.time())
            config_filename = f"config_{arch}_{experience_name}_{'_'.join(map(str, dataset_config['augmentation_image_size']))}_bs{batch_size}_epochs{max_epochs}_lr{lr:.6f}_anndropprob{augmentation_drop_annotation_random_prob:.2f}_{seed}_{timestamp}.yaml"
            config_path = os.path.join(config_dir, config_filename)

            # Save the modified configuration to file
            with open(config_path, "w") as outfile:
                yaml.dump(config, outfile, default_flow_style=False)

            # Verify config
            config = DetectorConfig.from_yaml(config_path)

            # Get SLURM arguments based on architecture and parameters
            gres_arg, time_arg, cpus_arg, mem_arg = select_sbatch_args(
                arch, batch_size, dataset_config["augmentation_image_size"], max_epochs
            )

            # Build the sbatch command; add cpus and mem args if defined
            cmd = ["sbatch", gres_arg, time_arg, cpus_arg, mem_arg, partition_arg]
            cmd.extend([f'{base_path}/sbatch_train.sh', config_path])

            print("Submitting job with command:")
            print(" ".join(cmd))
            print(f"  Parameters: arch={arch}, batch_size={batch_size}, max_epochs={max_epochs}, lr={lr:.6f}, gres_arg={gres_arg}, time_arg={time_arg}")

            # Submit the job
            subprocess.run(cmd)

            # Small delay to avoid overwhelming the scheduler
            time.sleep(1)
