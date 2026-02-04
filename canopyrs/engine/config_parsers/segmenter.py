from typing import Optional, List
from canopyrs.engine.config_parsers.base import BaseConfig
from pydantic import BaseModel, Field

class SegmenterConfig(BaseConfig):
    # General model definition
    model: str = 'sam2'
    architecture: Optional[str] = 'l'
    checkpoint_path: Optional[str] = None
    image_batch_size: int = 1
    box_batch_size: Optional[int] = 250
    max_prompts_per_image: Optional[int] = 64
    
    # Post-processing
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
    
    # Training hyperparameters
    seed: int = 42
    batch_size: int = 2
    max_epochs: int = 50
    lr: float = 1e-5
    weight_decay: float = 0.01
    steps_per_epoch: int = 100
    lr_image_encoder: float = 1.0e-5
    lr_others: float = 3.0e-5
    
    # Model freezing (RECOMMENDED: freeze encoder, train decoder)
    freeze_image_encoder: bool = False
    freeze_prompt_encoder: bool = False
    freeze_mask_decoder: bool = False
    
    box_noise_scale: float = 0.1  # Jitter scale for box prompts during training
    
    # ============================================================================
    # Data Augmentation (compatible with AugmentationAdder from detector training)
    # ============================================================================
    
    # Image size augmentation
    augmentation_image_size: List[int] = [1024, 1777]
    augmentation_early_conditional_image_size: int = 2000
    
    # Flip augmentations
    augmentation_flip_horizontal: bool = True
    augmentation_flip_vertical: bool = True
    
    # Rotation augmentation
    augmentation_rotation: float = 30.0
    augmentation_rotation_prob: float = 0.5
    
    # Color augmentations
    augmentation_brightness: float = 0.2
    augmentation_brightness_prob: float = 0.5
    augmentation_contrast: float = 0.2
    augmentation_contrast_prob: float = 0.5
    augmentation_saturation: float = 0.2
    augmentation_saturation_prob: float = 0.5
    augmentation_hue: int = 10
    augmentation_hue_prob: float = 0.3
    
    # Crop augmentations
    augmentation_train_crop_size_range: List[int] = [666, 2666]
    augmentation_crop_min_intersection_ratio: float = 0.5
    augmentation_crop_prob: float = 0.5
    augmentation_crop_fallback_to_augmentation_image_size: bool = False
    
    # ============================================================================
    # Loss Configuration
    # ============================================================================
    loss_weight_mask: float = 20.0   # focal
    loss_weight_dice: float = 1.0
    loss_weight_iou: float = 1.0
    loss_iou_use_l1: bool = True

    # ============================================================================
    # Evaluation & Checkpointing
    # ============================================================================
    eval_epoch_interval: int = 2
    main_metric: str = "val/mIoU"

    # Debug / visualization
    visualize_batches: bool = False
    visualize_every_n_steps: int = 200
    visualize_samples_per_batch: int = 2
    visualize_path : str = "./output/sam2_segmenter"
    max_validation_vis_samples: int = 5

    # ============================================================================
    # Hardware & Performance
    # ============================================================================
    use_amp: bool = True
    dataloader_num_workers: int = 4
    
    # ============================================================================
    # Logging & Output
    # ============================================================================
    train_log_interval: int = 50
    wandb_project: str = "canopyrs-sam2-finetune"
    wandb_enabled: bool = True
    train_output_path: str = "./output/sam2_training"

    # ============================================================================
    # Optional full CanopyRS inference + evaluation hooks
    # ============================================================================

    #Use detector boxes as SAM prompt
    use_detector_boxes: bool = True
    detector_config_path: str = "config/default_detection_multi_NQOS_best/pipeline.yaml"
    detector_cache_dir: str = "./output/detector_cache"
    eval_pipeline_config_path: Optional[str] = None
    coco_eval_output_dir: Optional[str] = None