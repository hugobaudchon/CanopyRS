"""
Detectree2 model configuration and setup.

Portions based on the Detectree2 project:
    https://github.com/PatBall1/detectree2

The MIT License (MIT)
Copyright (c) 2022, James G. C. Ball

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import json
import pickle
from pathlib import Path

from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo


def load_class_mapping(file_path: str):
    """Function to load class-to-index mapping from a file.
    -> Taken from https://github.com/PatBall1/detectree2

    Args:
        file_path: Path to the file (json or pickle)

    Returns:
        class_to_idx: Loaded class-to-index mapping
    """
    file_ext = Path(file_path).suffix

    if file_ext == '.json':
        with open(file_path, 'r') as f:
            class_to_idx = json.load(f)
    elif file_ext == '.pkl':
        with open(file_path, 'rb') as f:
            class_to_idx = pickle.load(f)
    else:
        raise ValueError("Unsupported file format. Use '.json' or '.pkl'.")

    return class_to_idx

def setup_detectree2_cfg(
    base_model: str = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
    trains=("trees_train",),
    tests=("trees_val",),
    update_model=None,
    workers=2,
    ims_per_batch=2,
    gamma=0.1,
    backbone_freeze=3,
    warm_iter=120,
    momentum=0.9,
    batch_size_per_im=1024,
    base_lr=0.0003389,
    weight_decay=0.001,
    max_iter=1000,
    eval_period=100,
    resize="fixed",    # "fixed" or "random" or "rand_fixed"
    imgmode="rgb",
    num_bands=3,
    class_mapping_file=None,
):
    """Set up config object # noqa: D417.
    -> Taken from https://github.com/PatBall1/detectree2

    Args:
        base_model: base pre-trained model from detectron2 model_zoo
        trains: names of registered data to use for training
        tests: names of registered data to use for evaluating models
        update_model: updated pre-trained model from detectree2 model_garden
        workers: number of workers for dataloader
        ims_per_batch: number of images per batch
        gamma: gamma for learning rate scheduler
        backbone_freeze: backbone layer to freeze
        warm_iter: number of iterations for warmup
        momentum: momentum for optimizer
        batch_size_per_im: batch size per image
        base_lr: base learning rate
        weight_decay: weight decay for optimizer
        max_iter: maximum number of iterations
        num_classes: number of classes
        eval_period: number of iterations between evaluations
        out_dir: directory to save outputs
        resize: resize strategy for images
        imgmode: image mode (rgb or multispectral)
        num_bands: number of bands in the image
        class_mapping_file: path to class mapping file
    """

    # Load the class mapping if provided
    if class_mapping_file:
        class_mapping = load_class_mapping(class_mapping_file)
        num_classes = len(
            class_mapping)    # Set the number of classes based on the mapping
    else:
        num_classes = 1    # Default to 1 class if no mapping is provided

    # Validate the resize parameter
    if resize not in {"fixed", "random", "rand_fixed"}:
        raise ValueError(
            f"Invalid resize option '{resize}'. Must be 'fixed', 'random', or 'rand_fixed'."
        )

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(base_model))
    cfg.DATASETS.TRAIN = trains
    cfg.DATASETS.TEST = tests
    cfg.DATALOADER.NUM_WORKERS = workers
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.GAMMA = gamma
    cfg.MODEL.BACKBONE.FREEZE_AT = backbone_freeze
    cfg.SOLVER.WARMUP_ITERS = warm_iter
    cfg.SOLVER.MOMENTUM = momentum
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = batch_size_per_im
    cfg.SOLVER.WEIGHT_DECAY = weight_decay
    cfg.SOLVER.BASE_LR = base_lr

    if update_model is not None:
        cfg.MODEL.WEIGHTS = update_model
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base_model)

    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.TEST.EVAL_PERIOD = eval_period
    cfg.RESIZE = resize
    cfg.INPUT.MIN_SIZE_TRAIN = 1000
    cfg.IMGMODE = imgmode    # "rgb" or "ms" (multispectral)
    if num_bands > 3:
        # Adjust PIXEL_MEAN and PIXEL_STD for the number of bands
        default_pixel_mean = cfg.MODEL.PIXEL_MEAN
        default_pixel_std = cfg.MODEL.PIXEL_STD
        # Extend or truncate the PIXEL_MEAN and PIXEL_STD based on num_bands
        cfg.MODEL.PIXEL_MEAN = (
            default_pixel_mean * (num_bands // len(default_pixel_mean)) +
            default_pixel_mean[:num_bands % len(default_pixel_mean)])
        cfg.MODEL.PIXEL_STD = (
            default_pixel_std * (num_bands // len(default_pixel_std)) +
            default_pixel_std[:num_bands % len(default_pixel_std)])
    return cfg
