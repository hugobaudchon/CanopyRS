from typing import List, Optional, Tuple, Union

from canopyrs.engine.config_parsers.base import BaseConfig


class DetectorConfig(BaseConfig):
    # General model definition
    model: str = 'faster_rcnn_detectron2'
    architecture: str = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
    checkpoint_path: Optional[str] = None
    batch_size: int = 8
    num_classes: int = 1
    box_predictions_per_image: Optional[int] = 500
    anchor_sizes: Optional[list] = [[32], [64], [128], [256], [512]]
    aspect_ratios: Optional[tuple] = ((0.5, 1.0, 2.0),) * 5
    box_score_thresh: Optional[float] = 0.05
    box_nms_thresh: Optional[float] = 0.5

    # Training Data and Output path
    data_root_path: str = None   # Parent folder of datasets
    train_dataset_names: List[str] = []     # Sub-folders names in root_path (parent folder)
    valid_dataset_names: List[str] = []     # Sub-folders names in root_path (parent folder)
    test_dataset_names: List[str] = []     # Sub-folders names in root_path (parent folder)
    train_output_path: str = None

    # Training Params
    main_metric: str = 'bbox/AP'
    seed: int = 42
    wandb_project: Optional[str] = None

    augmentation_image_size: Union[int, Tuple[int, int]] = 1024     # Final image size for both training and evaluation. If a tuple, the image will be resized randomly within the range for training (with prob augmentation_train_crop_prob), and will be resized deterministically for evaluation/infer.

    augmentation_early_conditional_image_size: Optional[int] = None
    augmentation_early_image_resize_test_only: Optional[int] = None  # allows evaluation at different image resolution for a given dataset and a model with a resize range (for example first resize to 800px to simulate 10cm/px, then resize to 1024px which is in model range of [1024, 1777]px)
    augmentation_train_crop_size_range: List[int] = [784, 2048]  # TODO implement: add support for floats too, ex: [0.6, 1.0] would crop between 60% and 100% of the image size
    # augmentation_crop_min_size = 333    # TODO implement: Minimum size of the crop, useful if cropping with float range (ex: [0.6, 1.0]) and some datasets have very small images by default (NeonTree at 400x400px...)
    augmentation_crop_prob: float = 0.5
    augmentation_crop_fallback_to_augmentation_image_size: bool = True    # If True, if no crop is selected based on augmentation_crop_prob, the image is cropped to augmentation_image_size. If False, the original image is kept.
    augmentation_crop_min_intersection_ratio: float = 0.5
    augmentation_flip_horizontal: bool = True
    augmentation_flip_vertical: bool = True
    augmentation_rotation: float = 30
    augmentation_rotation_prob: float = 0.5
    augmentation_contrast: float = 0.2
    augmentation_contrast_prob: float = 0.5
    augmentation_brightness: float = 0.2
    augmentation_brightness_prob: float = 0.5
    augmentation_saturation: float = 0.2
    augmentation_saturation_prob: float = 0.5
    augmentation_hue: int = 10  # in the [0-180] range
    augmentation_hue_prob: float = 0.5
    augmentation_drop_annotation_random_prob: float = 0.0        # for experiment where we randomly drop % of boxes to compare with detectree2 that kept images with min 40% of annotation cover

    lr: Optional[float] = 1e-4
    max_epochs: int = 100
    freeze_layers: Optional[int] = -1
    train_log_interval: int = 10
    eval_epoch_interval: int = 1
    grad_accumulation_steps: int = 1
    backbone_model_pretrained: Optional[bool] = True
    scheduler_type: str = 'WarmupMultiStepLR' # 'WarmupExponentialLR'
    scheduler_epochs_steps: Optional[List[int]] = [10, 20, 30]  # for WarmupMultiStepLR
    scheduler_gamma: Optional[float] = 0.1  # for WarmupExponentialLR (corresponds to end value, or 'decay') and WarmupMultiStepLR
    scheduler_warmup_steps: Optional[int] = 1000
    dataloader_num_workers: int = 4
    use_gradient_checkpointing: Optional[bool] = False  # Used for detrex training
    use_amp: bool = True        # Automatic Mixed Precision
