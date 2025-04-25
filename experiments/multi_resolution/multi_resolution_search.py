#!/usr/bin/env python3
import os
import time
import yaml
import subprocess
from itertools import product

from engine.config_parsers.detector import DetectorConfig

# Grid search parameters


exp_name = "multi_resolution"

model_architecture_list = [
    # "fasterrcnn",
    # "dinoresnet",
    "dinoswin"
]
seeds_list = [
    1,
    33,
    42
]


batch_sizes = [
    4, 
    # 8,
    # 16
]
max_epochs_list = [
    # 200,
    500,
    # 1000
]
scheduler_type = "WarmupCosineLR"  # "WarmupCosineLR" or "WarmupMultiStepLR"
lrs = [
    # fasterrcnn
    # 1e-3,
    # 2e-3,
    # 5e-3,

    # dino
    # 1e-4,
    5e-5
]

base_path = 'experiments/multi_resolution'
experience_name = f'detector_experience_multi_resolution'
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
dataset_config = {
        "compressed": [
            "ours_gr0p045_1777px.tar.gz",
            "ours_gr0p06_1333px.tar.gz",
            "ours_gr0p1_800px.tar.gz",

            "ours_gr0p045_888px.tar.gz"     # for validation @40m only
        ],
        "train_dataset_names": [
            # training on 0.045m resolution
            "tilerized_3555_0p5_0p045_None/panama_aguasalud",
            "tilerized_3555_0p5_0p045_None/ecuador_tiputini",
            "tilerized_3555_0p5_0p045_None/brazil_zf2",
        ],
        "valid_dataset_names": [
            # validation on 0.045m resolution @ 80m
            "tilerized_1777_0p5_0p045_None/panama_aguasalud",
            "tilerized_1777_0p5_0p045_None/ecuador_tiputini",
            "tilerized_1777_0p5_0p045_None/brazil_zf2",

            # # also validation on 0.06m resolution @ 80m
            # "tilerized_1333_0p5_0p06_None/panama_aguasalud",
            # "tilerized_1333_0p5_0p06_None/ecuador_tiputini",
            # "tilerized_1333_0p5_0p06_None/brazil_zf2",

            # # also validation on 0.1m resolution @ 80m
            # "tilerized_800_0p5_0p1_None/panama_aguasalud",
            # "tilerized_800_0p5_0p1_None/ecuador_tiputini",
            # "tilerized_800_0p5_0p1_None/brazil_zf2",

            # # also validation on 0.045m resolution @ 40m
            # "tilerized_888_0p5_0p045_None/panama_aguasalud",
            # "tilerized_888_0p5_0p045_None/ecuador_tiputini",
            # "tilerized_888_0p5_0p045_None/brazil_zf2",
        ],
        "augmentation_image_size": [1024, 1777], #[720, 1955],                 # resizing range for training (random or deterministic with p_crop prob) and for test (just a deterministic [min,max] resize if image is too small or too large)
        "augmentation_train_crop_size_range": [666, 2666] #[666, 2222] #[720, 1955]  #[666, 2222]     # cropping range for training
}


def select_sbatch_args(arch, batch_size, aug_img_size, max_epochs):
    """
    Returns a tuple (gres_arg, time_arg, cpus_arg, mem_arg)
    based on the current model architecture and parameters.
    """
    max_aug_img_size = max(aug_img_size)
    if arch == "fasterrcnn":
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
        time_arg = "--time=2-12:00:00"
        return gres_arg, time_arg, cpus_arg, mem_arg


# Main grid search loop over each architecture, dataset config, and hyperparameter combination
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
        config_filename = f"config_{arch}_{exp_name}_{'_'.join(map(str, dataset_config['augmentation_image_size']))}_bs{batch_size}_epochs{max_epochs}_lr{lr:.6f}_{seed}_{timestamp}.yaml"
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
        cmd.extend([f'{base_path}/sbatch_train.sh', config_path] + dataset_config["compressed"])

        print("Submitting job with command:")
        print(" ".join(cmd))
        print(f"  Parameters: arch={arch}, batch_size={batch_size}, max_epochs={max_epochs}, lr={lr:.6f}, gres_arg={gres_arg}, time_arg={time_arg}")

        # Submit the job
        subprocess.run(cmd)

        # Small delay to avoid overwhelming the scheduler
        time.sleep(1)
