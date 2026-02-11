from typing import Optional, List
from canopyrs.engine.config_parsers.base import BaseConfig
from pydantic import BaseModel, Field

class SegmenterConfig(BaseConfig):
    # General model definition and inference params
    model: str = 'sam2'
    architecture: Optional[str] = 'l'
    checkpoint_path: Optional[str] = None
    num_classes: int = 1
    image_batch_size: int = 1
    box_batch_size: Optional[int] = 250
    max_prompts_per_image: Optional[int] = 64

    # might be used for maskrcnn
    box_predictions_per_image: Optional[int] = 500
    anchor_sizes: Optional[list] = [[32], [64], [128], [256], [512]]
    aspect_ratios: Optional[tuple] = ((0.5, 1.0, 2.0),) * 5
    box_score_thresh: Optional[float] = 0.05
    box_nms_thresh: Optional[float] = 0.5
    
    # Inference post-processing
    pp_n_workers: int = 8
    pp_down_scale_masks_px: Optional[int] = 512
    pp_simplify_tolerance: float = 0.0
    pp_remove_rings: bool = True
    pp_remove_small_geoms: float = 50

    box_padding_percentage: float = 0.00

    # Data paths
    data_root_path: str = "../selvamask"
    train_dataset_names: List[str] = None
    valid_dataset_names: List[str] = None
    test_dataset_names: List[str] = []
    
    # Training hyperparameters
    seed: int = 42
    batch_size: int = 2
    max_epochs: int = 50
    lr: float = 1e-5
    eval_epoch_interval: int = 2
    main_metric: str = "val/mIoU"
    freeze_layers: Optional[int] = -1
    grad_accumulation_steps: int = 1
    use_amp: bool = True
    use_gradient_checkpointing: Optional[bool] = False
    dataloader_num_workers: int = 4
    backbone_model_pretrained: Optional[bool] = True
    scheduler_type: str = 'WarmupMultiStepLR'
    scheduler_epochs_steps: Optional[List[int]] = [10, 20, 30]
    scheduler_gamma: Optional[float] = 0.1
    scheduler_warmup_steps: Optional[int] = 1000

    # Training logs
    train_log_interval: int = 50
    wandb_project: str = "canopyrs-sam2-finetune"
    train_output_path: str = "./output/sam2_training"
    
    # Data Augmentation (compatible with AugmentationAdder from detector training)
    augmentation_image_size: List[int] = [1024, 1777]
    augmentation_early_conditional_image_size: int = 2000
    augmentation_early_image_resize_test_only: Optional[int] = None  # allows evaluation at different image resolution for a given dataset and a model with a resize range (for example first resize to 800px to simulate 10cm/px, then resize to 1024px which is in model range of [1024, 1777]px)
    
    augmentation_flip_horizontal: bool = True
    augmentation_flip_vertical: bool = True
    
    augmentation_rotation: float = 30.0
    augmentation_rotation_prob: float = 0.5
    
    augmentation_brightness: float = 0.2
    augmentation_brightness_prob: float = 0.5
    augmentation_contrast: float = 0.2
    augmentation_contrast_prob: float = 0.5
    augmentation_saturation: float = 0.2
    augmentation_saturation_prob: float = 0.5
    augmentation_hue: int = 10
    augmentation_hue_prob: float = 0.3
    
    augmentation_train_crop_size_range: List[int] = [666, 2666]
    augmentation_crop_min_intersection_ratio: float = 0.5
    augmentation_crop_prob: float = 0.5
    augmentation_crop_fallback_to_augmentation_image_size: bool = False

    augmentation_drop_annotation_random_prob: float = 0.0        # for experiment where we randomly drop % of boxes to compare with detectree2 that kept images with min 40% of annotation cover
    
    # SAM2/3-specific training params
    weight_decay: float = 0.01
    steps_per_epoch: int = 100
    lr_image_encoder: float = 1.0e-5
    lr_others: float = 3.0e-5

    freeze_image_encoder: bool = False
    freeze_prompt_encoder: bool = False
    freeze_mask_decoder: bool = False
    
    box_noise_scale: float = 0.1  # Jitter scale for box prompts during training

    loss_weight_mask: float = 20.0   # focal
    loss_weight_dice: float = 1.0
    loss_weight_iou: float = 1.0
    loss_iou_use_l1: bool = True

    visualize_batches: bool = False
    visualize_every_n_steps: int = 200
    visualize_samples_per_batch: int = 2
    visualize_path : str = "./output/sam2_segmenter"
    max_validation_vis_samples: int = 5

    wandb_enabled: bool = True

    use_detector_boxes: bool = True
    detector_config_path: str = "config/default_detection_multi_NQOS_best/pipeline.yaml"
    detector_cache_dir: str = "./output/detector_cache"
    eval_pipeline_config_path: Optional[str] = None
    coco_eval_output_dir: Optional[str] = None