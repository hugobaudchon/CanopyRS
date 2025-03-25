#!/usr/bin/env python3
import os
import time
import yaml
import subprocess
from itertools import product

# Grid search parameters

image_extent = "40m"

model_architecture_list = [
    # "fasterrcnn",
    "dinoresnet",
    # "dinoswin"
]
seeds_list = [
    1,
]
batch_sizes = [
    4, 
    8, 
    # 16
]
max_epochs_list = [
    200,
    500,
    # 1000
]
scheduler_type = "WarmupCosineLR"  # or "WarmupMultiStepLR"
lrs = [
    # fasterrcnn
    # 1e-3,
    # 2e-3,
    # 5e-3,

    # dino
    
    1e-4,
    5e-5
]

experience_name = f'detector_experience_resolution_{image_extent}'
train_output_path = f'/network/scratch/h/hugo.baudchon/training/{experience_name}'
wandb_project = experience_name

# Mapping from architecture string to its corresponding config paths and sbatch script
arch_to_config = {
    "fasterrcnn": {
        "model": "faster_rcnn_detectron2",
        "architecture": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        "checkpoint_path": None # checkpoint is automatically gathered by detectron2
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
    '40m':
    [
        {
            "compressed": "ours_gr0p045_888px.tar.gz",
            "train_dataset_names": [
                "tilerized_1777_0p5_0p045_None/panama_aguasalud",
                "tilerized_1777_0p5_0p045_None/ecuador_tiputini",
                "tilerized_1777_0p5_0p045_None/brazil_zf2"
            ],
            "valid_dataset_names": [
                "tilerized_888_0p5_0p045_None/panama_aguasalud",
                "tilerized_888_0p5_0p045_None/ecuador_tiputini",
                "tilerized_888_0p5_0p045_None/brazil_zf2"
            ],
            "augmentation_image_size": 888,
            "augmentation_train_crop_size_range": [800, 977]
        },
        {
            "compressed": "ours_gr0p06_666px.tar.gz",
            "train_dataset_names": [
                "tilerized_1333_0p5_0p06_None/panama_aguasalud",
                "tilerized_1333_0p5_0p06_None/ecuador_tiputini",
                "tilerized_1333_0p5_0p06_None/brazil_zf2"
            ],
            "valid_dataset_names": [
                "tilerized_666_0p5_0p06_None/panama_aguasalud",
                "tilerized_666_0p5_0p06_None/ecuador_tiputini",
                "tilerized_666_0p5_0p06_None/brazil_zf2"
            ],
            "augmentation_image_size": 666,
            "augmentation_train_crop_size_range": [600, 733]
        },
        {
            "compressed": "ours_gr0p1_400px.tar.gz",
            "train_dataset_names": [
                "tilerized_800_0p5_0p1_None/panama_aguasalud",
                "tilerized_800_0p5_0p1_None/ecuador_tiputini",
                "tilerized_800_0p5_0p1_None/brazil_zf2"
            ],
            "valid_dataset_names": [
                "tilerized_400_0p5_0p1_None/panama_aguasalud",
                "tilerized_400_0p5_0p1_None/ecuador_tiputini",
                "tilerized_400_0p5_0p1_None/brazil_zf2"
            ],
            "augmentation_image_size": 400,
            "augmentation_train_crop_size_range": [360, 440]
        },
        {
            "compressed": "ours_gr0p06_666px.tar.gz",
            "train_dataset_names": [
                "tilerized_1333_0p5_0p06_None/panama_aguasalud",
                "tilerized_1333_0p5_0p06_None/ecuador_tiputini",
                "tilerized_1333_0p5_0p06_None/brazil_zf2"
            ],
            "valid_dataset_names": [
                "tilerized_666_0p5_0p06_None/panama_aguasalud",
                "tilerized_666_0p5_0p06_None/ecuador_tiputini",
                "tilerized_666_0p5_0p06_None/brazil_zf2"
            ],
            "augmentation_image_size": 888,
            "augmentation_train_crop_size_range": [600, 733]
        },
        {
            "compressed": "ours_gr0p1_400px.tar.gz",
            "train_dataset_names": [
                "tilerized_800_0p5_0p1_None/panama_aguasalud",
                "tilerized_800_0p5_0p1_None/ecuador_tiputini",
                "tilerized_800_0p5_0p1_None/brazil_zf2"
            ],
            "valid_dataset_names": [
                "tilerized_400_0p5_0p1_None/panama_aguasalud",
                "tilerized_400_0p5_0p1_None/ecuador_tiputini",
                "tilerized_400_0p5_0p1_None/brazil_zf2"
            ],
            "augmentation_image_size": 888,
            "augmentation_train_crop_size_range": [360, 440]
        },
        {
            "compressed": "ours_gr0p1_400px.tar.gz",
            "train_dataset_names": [
                "tilerized_800_0p5_0p1_None/panama_aguasalud",
                "tilerized_800_0p5_0p1_None/ecuador_tiputini",
                "tilerized_800_0p5_0p1_None/brazil_zf2"
            ],
            "valid_dataset_names": [
                "tilerized_400_0p5_0p1_None/panama_aguasalud",
                "tilerized_400_0p5_0p1_None/ecuador_tiputini",
                "tilerized_400_0p5_0p1_None/brazil_zf2"
            ],
            "augmentation_image_size": 666,
            "augmentation_train_crop_size_range": [360, 440]
        },
    ],


    '80m':
    [
        {
            "compressed": "ours_gr0p045_1777px.tar.gz",
            "train_dataset_names": [
                "tilerized_3555_0p5_0p045_None/panama_aguasalud",
                "tilerized_3555_0p5_0p045_None/ecuador_tiputini",
                "tilerized_3555_0p5_0p045_None/brazil_zf2"
            ],
            "valid_dataset_names": [
                "tilerized_1777_0p5_0p045_None/panama_aguasalud",
                "tilerized_1777_0p5_0p045_None/ecuador_tiputini",
                "tilerized_1777_0p5_0p045_None/brazil_zf2"
            ],
            "augmentation_image_size": 1777,
            "augmentation_train_crop_size_range": [1600, 1955]
        },
        {
            "compressed": "ours_gr0p06_1333px.tar.gz",
            "train_dataset_names": [
                "tilerized_2666_0p5_0p06_None/panama_aguasalud",
                "tilerized_2666_0p5_0p06_None/ecuador_tiputini",
                "tilerized_2666_0p5_0p06_None/brazil_zf2"
            ],
            "valid_dataset_names": [
                "tilerized_1333_0p5_0p06_None/panama_aguasalud",
                "tilerized_1333_0p5_0p06_None/ecuador_tiputini",
                "tilerized_1333_0p5_0p06_None/brazil_zf2"
            ],
            "augmentation_image_size": 1333,
            "augmentation_train_crop_size_range": [1200, 1466]
        },
        {
            "compressed": "ours_gr0p1_800px.tar.gz",
            "train_dataset_names": [
                "tilerized_1600_0p5_0p1_None/panama_aguasalud",
                "tilerized_1600_0p5_0p1_None/ecuador_tiputini",
                "tilerized_1600_0p5_0p1_None/brazil_zf2"
            ],
            "valid_dataset_names": [
                "tilerized_800_0p5_0p1_None/panama_aguasalud",
                "tilerized_800_0p5_0p1_None/ecuador_tiputini",
                "tilerized_800_0p5_0p1_None/brazil_zf2"
            ],
            "augmentation_image_size": 800,
            "augmentation_train_crop_size_range": [720, 880]
        },
        {
            "compressed": "ours_gr0p06_1333px.tar.gz",
            "train_dataset_names": [
                "tilerized_2666_0p5_0p06_None/panama_aguasalud",
                "tilerized_2666_0p5_0p06_None/ecuador_tiputini",
                "tilerized_2666_0p5_0p06_None/brazil_zf2"
            ],
            "valid_dataset_names": [
                "tilerized_1333_0p5_0p06_None/panama_aguasalud",
                "tilerized_1333_0p5_0p06_None/ecuador_tiputini",
                "tilerized_1333_0p5_0p06_None/brazil_zf2"
            ],
            "augmentation_image_size": 1777,
            "augmentation_train_crop_size_range": [1200, 1466]
        },
        {
            "compressed": "ours_gr0p1_800px.tar.gz",
            "train_dataset_names": [
                "tilerized_1600_0p5_0p1_None/panama_aguasalud",
                "tilerized_1600_0p5_0p1_None/ecuador_tiputini",
                "tilerized_1600_0p5_0p1_None/brazil_zf2"
            ],
            "valid_dataset_names": [
                "tilerized_800_0p5_0p1_None/panama_aguasalud",
                "tilerized_800_0p5_0p1_None/ecuador_tiputini",
                "tilerized_800_0p5_0p1_None/brazil_zf2"
            ],
            "augmentation_image_size": 1777,
            "augmentation_train_crop_size_range": [720, 880]
        },
        {
            "compressed": "ours_gr0p1_800px.tar.gz",
            "train_dataset_names": [
                "tilerized_1600_0p5_0p1_None/panama_aguasalud",
                "tilerized_1600_0p5_0p1_None/ecuador_tiputini",
                "tilerized_1600_0p5_0p1_None/brazil_zf2"
            ],
            "valid_dataset_names": [
                "tilerized_800_0p5_0p1_None/panama_aguasalud",
                "tilerized_800_0p5_0p1_None/ecuador_tiputini",
                "tilerized_800_0p5_0p1_None/brazil_zf2"
            ],
            "augmentation_image_size": 1333,
            "augmentation_train_crop_size_range": [720, 880]
        }
    ]
}

def select_sbatch_args(arch, batch_size, aug_img_size, max_epochs):
    """
    Returns a tuple (gres_arg, time_arg, cpus_arg, mem_arg)
    based on the current model architecture and parameters.
    """
    if arch == "fasterrcnn":
        if batch_size == 32 and aug_img_size >= 1777:
            gres_arg = "--gres=gpu:a100l:1"
        elif aug_img_size >= 1777:
            gres_arg = "--gres=gpu:l40s:1"
        else:
            gres_arg = "--gres=gpu:rtx8000:1"
        if max_epochs <= 200 and aug_img_size <= 1333:
            time_arg = "--time=1-00:00:00"
        elif aug_img_size <= 1333 or (max_epochs <= 200 and aug_img_size <= 1777):
            time_arg = "--time=2-00:00:00"
        else:
            time_arg = "--time=3-00:00:00"
        return gres_arg, time_arg, None, None
    elif arch == "dinoresnet":
        if batch_size >= 8 or aug_img_size >= 1777:
            gres_arg = "--gres=gpu:rtx8000:2"
        else:
            gres_arg = "--gres=gpu:l40s:1"
        time_arg = "--time=2-00:00:00"
        cpus_arg = "--cpus-per-task=8"
        mem_arg = "--mem=40G"
        return gres_arg, time_arg, cpus_arg, mem_arg
    elif arch == "dinoswin":
        if batch_size >= 8 or aug_img_size >= 1777:
            gres_arg = "--gres=gpu:rtx8000:4"
        else:
            gres_arg = "--gres=gpu:rtx8000:2"
        time_arg = "--time=2-00:00:00"
        cpus_arg = "--cpus-per-task=8"
        mem_arg = "--mem=40G"
        return gres_arg, time_arg, cpus_arg, mem_arg

# Main grid search loop over each architecture, dataset config, and hyperparameter combination
for arch in model_architecture_list:
    # Retrieve config information for this architecture
    arch_info = arch_to_config[arch]
    base_config_path = "experiments/resolution/base_config.yaml"
    config_dir = f"experiments/resolution/{experience_name}"
    os.makedirs(config_dir, exist_ok=True)

    # Load the base YAML configuration for the current architecture
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Loop over each dataset configuration and grid search combination
    for dataset_config in dataset_configs[image_extent]:
        for batch_size, max_epochs, lr, seed in product(batch_sizes, max_epochs_list, lrs, seeds_list):
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
            config["scheduler_epochs_steps"] = [int(max_epochs * 0.8), int(max_epochs * 0.9)]
            config["seed"] = seed

            # Include the model architecture in the configuration
            config["model"] = arch_info["model"]
            config["architecture"] = arch_info["architecture"]
            config["checkpoint_path"] = arch_info["checkpoint_path"]

            # Update dataset–specific parameters
            config["train_dataset_names"] = dataset_config["train_dataset_names"]
            config["valid_dataset_names"] = dataset_config["valid_dataset_names"]
            config["augmentation_image_size"] = dataset_config["augmentation_image_size"]
            config["augmentation_train_crop_size_range"] = dataset_config["augmentation_train_crop_size_range"]

            # Create a unique filename (including the architecture name)
            timestamp = int(time.time())
            variant_name = dataset_config["compressed"].split('.')[0]
            config_filename = f"config_{arch}_{variant_name}_{dataset_config['augmentation_image_size']}_bs{batch_size}_epochs{max_epochs}_lr{lr:.6f}_{seed}_{timestamp}.yaml"
            config_path = os.path.join(config_dir, config_filename)

            # Save the modified configuration to file
            with open(config_path, "w") as outfile:
                yaml.dump(config, outfile, default_flow_style=False)

            # Get SLURM arguments based on architecture and parameters
            gres_arg, time_arg, cpus_arg, mem_arg = select_sbatch_args(
                arch, batch_size, dataset_config["augmentation_image_size"], max_epochs
            )

            # Build the sbatch command; add cpus and mem args if defined
            cmd = ["sbatch", gres_arg, time_arg]
            if cpus_arg is not None:
                cmd.append(cpus_arg)
            if mem_arg is not None:
                cmd.append(mem_arg)
            cmd.extend(['experiments/resolution/sbatch_train.sh', dataset_config["compressed"], config_path])

            print("Submitting job with command:")
            print(" ".join(cmd))
            print(f"  Parameters: arch={arch}, batch_size={batch_size}, max_epochs={max_epochs}, lr={lr:.6f}, gres_arg={gres_arg}, time_arg={time_arg}")

            # Submit the job
            subprocess.run(cmd)

            # Small delay to avoid overwhelming the scheduler
            time.sleep(1)
