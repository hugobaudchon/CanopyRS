from typing import Optional, List
from engine.config_parsers.base import BaseConfig


class SegmenterConfig(BaseConfig):
    # General model definition
    model: str = 'sam2'
    architecture: Optional[str] = 'l'
    checkpoint_path: str = None
    device: str = 'cuda'
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
    
    # Model freezing (RECOMMENDED: freeze encoder, train decoder)
    freeze_image_encoder: bool = True
    freeze_prompt_encoder: bool = False
    freeze_mask_decoder: bool = False
    
    # Scheduler
    scheduler_type: str = "cosine"  # "step", "cosine", "polynomial"
    scheduler_epochs_steps: Optional[List[int]] = None
    scheduler_gamma: Optional[float] = 0.1
    scheduler_warmup_steps: Optional[int] = 500
    
    # ============================================================================
    # Data Augmentation (compatible with AugmentationAdder from detector training)
    # ============================================================================
    use_augmentation: bool = True
    
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
    # Prompts for SAM2 Training
    # ============================================================================
    use_box_prompts: bool = True  # Use ground truth boxes (REQUIRED)
    use_point_prompts: bool = False  # Optional: add point prompts
    num_point_prompts: int = 5  # Positive points inside mask
    num_negative_points: int = 1  # Negative points outside mask
    
    # ============================================================================
    # Loss Configuration
    # ============================================================================
    use_focal_loss: bool = True
    use_dice_loss: bool = True
    focal_loss_weight: float = 20.0
    dice_loss_weight: float = 1.0
    iou_loss_weight: float = 1.0
    
    # ============================================================================
    # Evaluation & Checkpointing
    # ============================================================================
    eval_epoch_interval: int = 2
    main_metric: str = "val/mIoU"
    save_checkpoint_interval: int = 5
    # Debug / visualization
    visualize_batches: bool = False
    visualize_every_n_steps: int = 200
    visualize_samples_per_batch: int = 2
    visualize_path : str = "./output/sam2_segmenter"
    
    # ============================================================================
    # Hardware & Performance
    # ============================================================================
    use_amp: bool = True
    dataloader_num_workers: int = 4
    gradient_accumulation_steps: int = 2
    
    # ============================================================================
    # Logging & Output
    # ============================================================================
    train_log_interval: int = 50
    wandb_project: str = "canopyrs-sam2-finetune"
    wandb_enabled: bool = True
    train_output_path: str = "./output/sam2_training"