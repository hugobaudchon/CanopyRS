import os
import uuid
from datetime import datetime
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import wandb
from canopyrs.engine.config_parsers import SegmenterConfig
from sam2.sam2_image_predictor import SAM2ImagePredictor
from detectron2.config import get_cfg
from .loss_fns import sam_loss
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from detectron2.data import build_detection_train_loader, build_detection_test_loader, DatasetMapper, DatasetCatalog
from .augmentation import AugmentationAdder
import math
import gc
import json
import tempfile
from canopyrs.engine.models.segmenter.train_sam.dataset import register_sam2_dataset_with_masks, register_sam2_dataset_with_predicted_boxes

def run_coco_evaluations(
    config: SegmenterConfig,
    checkpoint_path: Path,
    epoch: int,
    valid_detector_coco_data=None,
) -> dict[str, float]:
    """
    Run full pipeline (detector → SAM2) evaluation on validation set.
    Returns COCO metrics comparing predictions to GT.
    """
    from  canopyrs.engine.config_parsers import InferIOConfig, PipelineConfig
    from  canopyrs.engine.config_parsers.base import get_config_path
    from  canopyrs.engine.pipeline import Pipeline
    from  canopyrs.engine.benchmark.detector.evaluator import CocoEvaluator
    from copy import deepcopy
    
    print(f"\n[coco_eval] Running COCO evaluation at epoch {epoch}...")
    
    # Get paths from config
    pipeline_config_path = getattr(config, 'eval_pipeline_config_path', None)
    if not pipeline_config_path:
        print("[coco_eval] No eval_pipeline_config_path set, skipping.")
        return {}
    
    dataset_root = Path(config.data_root_path)
    output_root = Path(getattr(config, 'coco_eval_output_dir', None))
    ground_resolution = 0.045
    gr_token = str(ground_resolution).replace('.', 'p')
    
    # Load pipeline config
    pipeline_config = PipelineConfig.from_yaml(get_config_path(pipeline_config_path))

    # Override segmenter checkpoint with current model
    for component_type, component_config in pipeline_config.components_configs:
        if component_type == 'segmenter':
            component_config.checkpoint_path = str(checkpoint_path)
    
    # Remove tilerizer if present (we have tiles already)
    if pipeline_config.components_configs[0][0] == 'tilerizer':
        pipeline_config.components_configs.pop(0)

    # OK HERE i need to find the boxes coco file from the first run and if found remove detector
    if valid_detector_coco_data is not None:
        if pipeline_config.components_configs[0][0] == 'detector':
            pipeline_config.components_configs.pop(0)
            print("[coco_eval] Removed tilerizer and detector from pipeline for evaluation.")
        # Save detector coco data to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_coco_file:
            json.dump(valid_detector_coco_data, tmp_coco_file)
            tmp_coco_file_path = tmp_coco_file.name
        # Now set the detector component to use this coco file as input
        for component_type, component_config in pipeline_config.components_configs:
            if component_type == 'segmenter':
                component_config.input_coco_path = tmp_coco_file_path
                print(f"[coco_eval] Using segmenter COCO predictions from {tmp_coco_file_path}")
                break
    
    all_metrics = {}
    
    # Run on each validation dataset
    for dataset_name in config.valid_dataset_names:
        site_root = dataset_root / dataset_name
        tiles_path = site_root / 'tiles' / 'valid'
        
        if not tiles_path.exists():
            print(f"[coco_eval] Skipping {dataset_name}: tiles not found at {tiles_path}")
            continue
        
        # GT COCO file path
        gt_coco_path = site_root / f"{dataset_name}_coco_gr{gr_token}_valid.json"
        if not gt_coco_path.exists():
            print(f"[coco_eval] Skipping {dataset_name}: GT COCO not found at {gt_coco_path}")
            continue
        
        # Create IO config
        io_config = InferIOConfig(
            input_imagery=str(tiles_path),  # Required field
            tiles_path=str(tiles_path),
            output_folder=str(output_root),
        )
        
        # Run pipeline
        print(f"[coco_eval] Running pipeline on {dataset_name}...")
        pipeline = Pipeline(io_config, deepcopy(pipeline_config))
        data_state = pipeline()
        
        # Get predictions COCO path (use raw detector output, not aggregated)
        pred_coco_path = None
        if hasattr(data_state, 'component_output_files'):
            # Try to get segmenter output first
            for comp_id, files in data_state.component_output_files.items():
                if 'segmenter' in comp_id.lower() or comp_id.startswith('1_'):
                    pred_coco_path = files.get('coco')
                    break
            # Fallback to any coco file
            if not pred_coco_path:
                for comp_id, files in data_state.component_output_files.items():
                    if 'coco' in files:
                        pred_coco_path = files['coco']
                        break
        
        if not pred_coco_path or not Path(pred_coco_path).exists():
            print(f"[coco_eval] No predictions COCO found for {dataset_name}")
            continue
        
        # Run COCO evaluation
        print(f"[coco_eval] Evaluating {dataset_name}...")
        evaluator = CocoEvaluator()
        
        try:
            metrics = evaluator.tile_level(
                iou_type='segm',
                preds_coco_path=str(pred_coco_path),
                truth_coco_path=str(gt_coco_path),
                ground_resolution=ground_resolution,
                max_dets=[1, 10, 100, 400],
            )
            
            # Keep only key metrics
            key_metrics = {k: v for k, v in metrics.items() 
                          if k in {'AP', 'AP50', 'AP75', 'AR', 'AR50', 'AR75'}}
            
            print(f"[coco_eval] {dataset_name}: {key_metrics}")
            
            # Prefix with dataset name
            for k, v in key_metrics.items():
                all_metrics[f"coco/{dataset_name}/{k}"] = float(v)
                
        except Exception as e:
            print(f"[coco_eval] Evaluation failed for {dataset_name}: {e}")
            continue
    
    # Compute mean across datasets
    if all_metrics:
        metric_names = ['AP', 'AP50', 'AP75', 'AR', 'AR50', 'AR75']
        for metric_name in metric_names:
            values = [v for k, v in all_metrics.items() if k.endswith(f'/{metric_name}')]
            if values:
                all_metrics[f"coco/mean/{metric_name}"] = float(np.mean(values))
    
    return all_metrics

def visualize_predictions(image, gt_masks, pred_masks, boxes, save_path, step, max_samples=20):
    """
    Visualize ALL ground truth vs predicted masks on the same images.
    
    Args:
        image: numpy array (H, W, 3)
        gt_masks: numpy array (N, H, W)
        pred_masks: torch tensor (N, H, W) - logits
        boxes: numpy array (N, 4)
        save_path: Path to save visualization
        step: current training step
        max_samples: max number of masks to show
    """
    save_path.mkdir(parents=True, exist_ok=True)
    num_masks = min(len(gt_masks), max_samples)
    
    # Create figure with 3 subplots (one row)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # ========== SUBPLOT 1: Image + ALL Boxes ==========
    axes[0].imshow(image)
    for i in range(num_masks):
        x1, y1, x2, y2 = boxes[i]
        rect = mpatches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, edgecolor='lime', linewidth=1)
        axes[0].add_patch(rect)
    axes[0].set_title(f'Image + {num_masks} Boxes')
    axes[0].axis('off')
    
    # ========== SUBPLOT 2: Image + ALL Ground Truth Masks ==========
    axes[1].imshow(image)
    # Combine all GT masks with different colors
    combined_gt = np.zeros_like(image, dtype=np.float32)
    for i in range(num_masks):
        # Use different colors for each mask
        color_mask = np.zeros_like(image, dtype=np.float32)
        color_mask[:, :, 0] = gt_masks[i] * (i % 3 == 0) * 255  # Red channel
        color_mask[:, :, 1] = gt_masks[i] * (i % 3 == 1) * 255  # Green channel
        color_mask[:, :, 2] = gt_masks[i] * (i % 3 == 2) * 255  # Blue channel
        combined_gt = np.maximum(combined_gt, color_mask)
    
    axes[1].imshow(combined_gt.astype(np.uint8), alpha=0.5)
    axes[1].set_title(f'{num_masks} Ground Truth Masks')
    axes[1].axis('off')
    
    # ========== SUBPLOT 3: Image + ALL Predicted Masks ==========
    axes[2].imshow(image)
    # Combine all predictions with different colors
    combined_pred = np.zeros_like(image, dtype=np.float32)
    for i in range(num_masks):
        pred_prob = torch.sigmoid(pred_masks[i]).detach().cpu().numpy()
        # Use different colors for each mask
        color_mask = np.zeros_like(image, dtype=np.float32)
        color_mask[:, :, 0] = pred_prob * (i % 3 == 0) * 255  # Red channel
        color_mask[:, :, 1] = pred_prob * (i % 3 == 1) * 255  # Green channel
        color_mask[:, :, 2] = pred_prob * (i % 3 == 2) * 255  # Blue channel
        combined_pred = np.maximum(combined_pred, color_mask)
    
    axes[2].imshow(combined_pred.astype(np.uint8), alpha=0.5)
    
    # Calculate overall IoU
    all_iou = []
    for i in range(num_masks):
        pred_binary = (torch.sigmoid(pred_masks[i]).detach().cpu().numpy() > 0.5).astype(float)
        inter = (gt_masks[i] * pred_binary).sum()
        union = (gt_masks[i] + pred_binary).clip(0, 1).sum()
        iou = inter / (union + 1e-6)
        all_iou.append(iou)
    
    mean_iou = np.mean(all_iou)
    axes[2].set_title(f'{num_masks} Predictions (Mean IoU={mean_iou:.3f})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path / f'step_{step:06d}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def build_sam2_model(config: SegmenterConfig):
    """Build SAM2 using HuggingFace pretrained models."""
    MODEL_MAPPING = {
        't': "facebook/sam2-hiera-tiny",
        's': "facebook/sam2-hiera-small",
        'b': "facebook/sam2-hiera-base-plus",
        'l': "facebook/sam2-hiera-large",
    }
    
    model_name = MODEL_MAPPING[config.architecture]
    
    print(f"\n{'='*80}")
    print(f"Building SAM2 Model")
    print(f"{'='*80}")
    print(f"Architecture: {config.architecture}")
    print(f"Model: {model_name}")
    
    if config.checkpoint_path:
        print(f"Loading from fine-tuned checkpoint: {config.checkpoint_path}")
        predictor = SAM2ImagePredictor.from_pretrained(model_name)
        state_dict = torch.load(config.checkpoint_path, map_location='cuda')
        if 'model_state_dict' in state_dict:
            predictor.model.load_state_dict(state_dict['model_state_dict'])
        else:
            predictor.model.load_state_dict(state_dict)
        print(f"✓ Fine-tuned weights loaded")
    else:
        print(f"Loading pretrained model from HuggingFace: {model_name}")
        predictor = SAM2ImagePredictor.from_pretrained(model_name)
        print(f"✓ Model loaded from HuggingFace")
    
    # Apply freezing strategy
    freeze_image_encoder = getattr(config, 'freeze_image_encoder', True)
    freeze_prompt_encoder = getattr(config, 'freeze_prompt_encoder', False)
    freeze_mask_decoder = getattr(config, 'freeze_mask_decoder', False)
    
    print(f"\nFreezing strategy:")
    print(f"  Image Encoder: {'FROZEN' if freeze_image_encoder else 'TRAINABLE'}")
    print(f"  Prompt Encoder: {'FROZEN' if freeze_prompt_encoder else 'TRAINABLE'}")
    print(f"  Mask Decoder: {'FROZEN' if freeze_mask_decoder else 'TRAINABLE'}")
    print("\nFreezing strategy flags:")
    print(f"  freeze_image_encoder  = {freeze_image_encoder}")
    print(f"  freeze_prompt_encoder = {freeze_prompt_encoder}")
    print(f"  freeze_mask_decoder   = {freeze_mask_decoder}")
    # Components that are NOT used in single-image prediction (video/memory stuff)
    memory_components = [
        "memory_attention",
        "memory_encoder", 
        "obj_ptr_proj",
        "maskmem_tpos_enc",
        "no_mem_embed",
        "no_mem_pos_enc",
        "no_obj_ptr",
        "mask_downsample",
    ]
    for name, param in predictor.model.named_parameters():
        param.requires_grad = True
        if any(mem_comp in name for mem_comp in memory_components):
            param.requires_grad = False
        elif freeze_image_encoder and "image_encoder" in name:
            param.requires_grad = False
        elif freeze_prompt_encoder and "prompt_encoder" in name:
            param.requires_grad = False
        elif freeze_mask_decoder and "mask_decoder" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    total_params = sum(p.numel() for p in predictor.model.parameters())
    trainable_params = sum(p.numel() for p in predictor.model.parameters() if p.requires_grad)
    
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen: {total_params - trainable_params:,}")
    print(f"  Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    print(f"{'='*80}\n")
        # After the freezing loop, add this:
    # Detailed breakdown of "Other" category
    print("\n  Breakdown of 'Other' parameters:")
    other_modules = {}
    for name, p in predictor.model.named_parameters():
        if "image_encoder" not in name and "mask_decoder" not in name and "prompt_encoder" not in name:
            # Get the top-level module name
            module_name = name.split('.')[0]
            if module_name not in other_modules:
                other_modules[module_name] = {'total': 0, 'trainable': 0}
            other_modules[module_name]['total'] += p.numel()
            if p.requires_grad:
                other_modules[module_name]['trainable'] += p.numel()
    
    for mod, counts in sorted(other_modules.items(), key=lambda x: -x[1]['total']):
        print(f"    {mod}: {counts['trainable']:,} / {counts['total']:,} trainable")

    print("\nTrainable parameter breakdown by module:")
    enc_total = enc_trainable = 0
    dec_total = dec_trainable = 0
    prompt_total = prompt_trainable = 0
    other_total = other_trainable = 0

    for name, p in predictor.model.named_parameters():
        n = p.numel()
        if "image_encoder" in name:
            enc_total += n
            if p.requires_grad:
                enc_trainable += n
        elif "mask_decoder" in name:  # adjust if your module name differs
            dec_total += n
            if p.requires_grad:
                dec_trainable += n
        elif "prompt_encoder" in name:
            prompt_total += n
            if p.requires_grad:
                prompt_trainable += n
        else:
            other_total += n
            if p.requires_grad:
                other_trainable += n

    print(f"  Image encoder: {enc_trainable:,} / {enc_total:,} trainable")
    print(f"  Mask decoder : {dec_trainable:,} / {dec_total:,} trainable")
    print(f"  Prompt encoder: {prompt_trainable:,} / {prompt_total:,} trainable")
    print(f"  Other        : {other_trainable:,} / {other_total:,} trainable")

    
    return predictor


def build_sam2_train_loader(cfg, dataset_name, batch_size=1):
    """
    Build SAM2 training DataLoader using Detectron2's infrastructure.
    Same pattern as TrainerWithValidation.build_train_loader().
    """
    # Get the dataset
    dataset_dicts = DatasetCatalog.get(dataset_name)
    
    loader = build_detection_train_loader(
        cfg,
        dataset=dataset_dicts,
        mapper=DatasetMapper(
            cfg,
            augmentations=AugmentationAdder.get_augmentation_detectron2_train(cfg),
            is_train=True,
            instance_mask_format="bitmask",
            use_instance_mask=True,
        ),
        total_batch_size=batch_size,  # SAM2 uses batch_size=1
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )
    
    return loader

def build_sam2_test_loader(cfg, dataset_name):
    """
    Build SAM2 validation DataLoader.
    Same pattern as TrainerWithValidation.build_test_loader().
    """
    
    dataset_dicts = DatasetCatalog.get(dataset_name)
    
    return build_detection_test_loader(
        cfg,
        dataset_name,
        mapper=DatasetMapper(
            cfg,
            augmentations=AugmentationAdder.get_augmentation_detectron2_test(cfg),
            is_train=True,
            instance_mask_format="bitmask",
            use_instance_mask=True,
        )
    )


def setup_sam2_datasets(config: SegmenterConfig):
    """
    Register SAM2 datasets with annotations.
    Returns dataset names (not loaders - those are built separately).
    """
    print("Setting up datasets...")
    use_detector_boxes = getattr(config, 'use_detector_boxes', True)
    valid_detector_coco_data = None

    if use_detector_boxes:
        print("Using PREDICTED boxes for training prompts")
        
        detector_config = getattr(config, 'detector_config_path', None)
        cache_dir = getattr(config, 'detector_cache_dir', '/tmp/detector_cache')
        
        if detector_config is None:
            raise ValueError("detector_config_path must be set when use_gt_boxes=False")
        print("Running detector on train datasets to get boxes...")
        train_dataset_name = register_sam2_dataset_with_predicted_boxes(
            root_path=[f"{config.data_root_path}/{path}" for path in config.train_dataset_names],
            fold="train",
            detector_config_path=detector_config,
            cache_dir=cache_dir,
            force_binary_class=True
        )
        
        # For validation, you might want to still use GT boxes for fair comparison
        # Or use predicted boxes for consistency - your choice
        print("Running detector on valid datasets to get boxes...")
        valid_dataset_name = register_sam2_dataset_with_predicted_boxes(
            root_path=[f"{config.data_root_path}/{path}" for path in config.valid_dataset_names],
            fold="valid",
            detector_config_path=detector_config,
            cache_dir=cache_dir,
            force_binary_class=True
        )
    else:
        print("Using GT boxes for training prompts")
        
        train_dataset_name = register_sam2_dataset_with_masks(
            root_path=[f"{config.data_root_path}/{path}" for path in config.train_dataset_names],
            fold="train",
            force_binary_class=True
        )
        
        valid_dataset_name = register_sam2_dataset_with_masks(
            root_path=[f"{config.data_root_path}/{path}" for path in config.valid_dataset_names],
            fold="valid",
            force_binary_class=True
        )
    
    return train_dataset_name, valid_dataset_name


def setup_sam2_trainer(train_dataset_name: str, valid_dataset_name: str, config: SegmenterConfig, model_name: str):
    """
    Set up SAM2 training components.
    Same pattern as setup_trainer() for Detectron2.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = get_cfg()
    
    # Add augmentation config
    AugmentationAdder.modify_detectron2_augmentation_config(config, cfg)
    
    # Dataset config
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (valid_dataset_name,)
    
    # Dataloader config
    num_workers = min(getattr(config, 'dataloader_num_workers', 4), 2)  # Cap at 2 for memory
    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.INPUT.MASK_FORMAT = "bitmask"
    
    # Output directory
    output_dir = Path(config.train_output_path) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR = str(output_dir)
    
    # Save config
    config.to_yaml(str(output_dir / "config.yaml"))
    
    train_loader = build_sam2_train_loader(cfg, train_dataset_name, batch_size=getattr(config, 'batch_size', 1))
    valid_loader = build_sam2_test_loader(cfg, valid_dataset_name)
    
    train_dataset_dicts = DatasetCatalog.get(train_dataset_name)
    valid_dataset_dicts = DatasetCatalog.get(valid_dataset_name)
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset_dicts)} images")
    print(f"  Validation: {len(valid_dataset_dicts)} images")
    
    # Build model
    predictor = build_sam2_model(config)
    predictor.model.to(device)
    predictor.model.train()

    encoder_params = []
    non_encoder_params = []
    for name, p in predictor.model.named_parameters():
        if not p.requires_grad:
            continue
        if "image_encoder" in name:
            encoder_params.append(p)
        else:
            non_encoder_params.append(p)

    # Optimizer
    trainable_params = [p for p in predictor.model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": config.lr_image_encoder},
            {"params": non_encoder_params, "lr": config.lr_others},
        ],
        weight_decay=getattr(config, 'weight_decay', 0.01),
    )
    num_opt_params = sum(p.numel() for g in optimizer.param_groups for p in g['params'])
    print(f"Optimizer has {num_opt_params:,} parameters")
    # Scheduler
    batch_size = getattr(config, 'batch_size', 1)
    dataset_size = len(train_dataset_dicts)
    steps_per_epoch = math.ceil(dataset_size / batch_size)
    print(f"\nSteps per epoch: {steps_per_epoch} "
          f"(dataset_size={dataset_size}, batch_size={batch_size})")

    max_steps = config.max_epochs * steps_per_epoch

    warmup_steps = int(0.05 * max_steps)  # e.g. 5% warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        # cosine decay  from 1 down to ~0 over remaining steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    
    scaler = torch.amp.GradScaler(enabled=getattr(config, 'use_amp', True))
    
    print(f"\nTraining configuration:")
    print(f"  Max epochs: {config.max_epochs}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {max_steps}")
    print(f"  LR (image encoder): {config.lr_image_encoder}")
    print(f"  LR (others)       : {config.lr_others}")
    print(f"  Output dir: {output_dir}")
    
    return {
        'predictor': predictor,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'scaler': scaler,
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'output_dir': output_dir,
        'device': device,
        'max_steps': max_steps,
        'steps_per_epoch': steps_per_epoch,
    }


def train_step(predictor, batch, optimizer, scaler, device, config, global_step=0, vis_dir=None):
    """
    Single training step that supports arbitrary DataLoader batch_size.
    `batch` is a list of dicts (Detectron2 style), length = batch_size.
    Now: box-only prompts (no points).
    """
    if len(batch) == 0:
        return None

    weight_dict = {
        'loss_mask': getattr(config, 'loss_weight_mask', 20.0),
        'loss_dice': getattr(config, 'loss_weight_dice', 1.0),
        'loss_iou': getattr(config, 'loss_weight_iou', 1.0),
    }

    use_amp = getattr(config, 'use_amp', True)
    visualize = getattr(config, 'visualize_batches', False)
    visualize_every = getattr(config, 'visualize_every_n_steps', 100)

    total_loss = 0.0
    total_loss_mask = 0.0
    total_loss_dice = 0.0
    total_loss_iou = 0.0
    all_ious = []
    num_valid_samples = 0

    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type='cuda', enabled=use_amp):
        for sample_idx, data in enumerate(batch):
            # --- Extract image ---
            if "image" not in data:
                continue

            image = data["image"].permute(1, 2, 0).cpu().numpy()
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)

            # --- Extract instances ---
            if "instances" not in data or len(data["instances"]) == 0:
                continue

            instances = data["instances"]
            
            boxes = instances.gt_boxes.tensor.cpu().numpy()
            # Adding box jitter
            if getattr(config, 'box_noise_scale', 0.0) > 0:
                # Scale is relative to box size (e.g., 0.1 = 10% jitter)
                box_widths = boxes[:, 2] - boxes[:, 0]
                box_heights = boxes[:, 3] - boxes[:, 1]
                
                # Random noise for x1, y1, x2, y2
                noise = np.random.normal(0, config.box_noise_scale, boxes.shape)
                
                # Scale noise by box dimensions
                noise[:, 0] *= box_widths
                noise[:, 2] *= box_widths
                noise[:, 1] *= box_heights
                noise[:, 3] *= box_heights
                
                boxes = boxes + noise
                # Clip to image boundaries
                boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, image.shape[1])
                boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, image.shape[0])
                
            masks = instances.gt_masks.tensor.cpu().numpy()

            if len(boxes) == 0:
                continue

            # --- Subsample to max prompts ---
            max_prompts = getattr(config, 'max_prompts_per_image', 64)
            if len(boxes) > max_prompts:
                idx = np.random.choice(len(boxes), max_prompts, replace=False)
                boxes = boxes[idx]
                masks = masks[idx]

            num_masks = len(boxes)

            # -----------------------------
            # BOX-ONLY PROMPTS (NO POINTS)
            # -----------------------------
            freeze_image_encoder = getattr(config, 'freeze_image_encoder', True)
            
            if freeze_image_encoder:
                # Use set_image for frozen encoder (faster, no gradients needed)
                with torch.no_grad():
                    predictor.set_image(image)
            else:
                # Manually encode to preserve gradient chain for training
                # SAM2Transforms does: ToTensor (uint8->float 0-1) -> Resize(1024) -> Normalize(ImageNet)
                
                # 1. Convert numpy [H,W,3] uint8 to tensor [1,3,H,W] float [0,1]
                img_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(device)
                img_tensor = img_tensor / 255.0  # Normalize to [0, 1]
                img_tensor = img_tensor.unsqueeze(0)  # [1, 3, H, W]
                
                # 2. Resize to SAM2's expected resolution (1024x1024)
                img_size = predictor._transforms.resolution  # 1024
                img_resized = F.interpolate(
                    img_tensor,
                    size=(img_size, img_size),
                    mode='bilinear',
                    align_corners=False,
                )
                
                # 3. Apply ImageNet normalization (same as SAM2Transforms)
                mean = torch.tensor(predictor._transforms.mean, device=device).view(1, 3, 1, 1)
                std = torch.tensor(predictor._transforms.std, device=device).view(1, 3, 1, 1)
                img_normalized = (img_resized - mean) / std
                
                # 4. Run through image encoder WITH gradients
                backbone_out = predictor.model.forward_image(img_normalized)
                
                # 5. Prepare backbone features (this is what _prepare_backbone_features does)
                _, vision_feats, _, feat_sizes = predictor.model._prepare_backbone_features(backbone_out)
                
                # 6. Add no_mem_embed if configured
                if predictor.model.directly_add_no_mem_embed:
                    vision_feats[-1] = vision_feats[-1] + predictor.model.no_mem_embed
                
                # 7. Reshape features to match expected format
                # vision_feats are in (HW, B, C) format, need to convert to (B, C, H, W)
                feats = [
                    feat.permute(1, 2, 0).view(1, -1, *feat_size)
                    for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
                ][::-1]
                
                # 8. Store in predictor (mimicking set_image behavior)
                predictor._features = {
                    "image_embed": feats[-1],
                    "high_res_feats": feats[:-1],
                }
                predictor._orig_hw = [(image.shape[0], image.shape[1])]
                predictor._is_image_set = True
            
            orig_h, orig_w = predictor._orig_hw[-1]

            # No point_coords, no point_labels
            _, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                point_coords=None,
                point_labels=None,
                box=boxes,
                mask_logits=None,
                normalize_coords=True,
            )

            if unnorm_box is None or unnorm_box.shape[0] == 0:
                continue

            # points=None here
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=None,
                boxes=unnorm_box,
                masks=None,
            )

            batched_mode = unnorm_box.shape[0] > 1

            high_res_features = [
                feat_level[-1].unsqueeze(0)
                for feat_level in predictor._features["high_res_feats"]
            ]
            multimask_output = False
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
            )

            prd_masks = predictor._transforms.postprocess_masks(
                low_res_masks, predictor._orig_hw[-1]
            )
            # prd_masks: [N, K, H, W]
            num_masks_out, num_candidates, H, W = prd_masks.shape

            if multimask_output:
                # Select the best mask based on predicted IoU score
                best_idx = prd_scores.argmax(dim=1)  # [N]
                rows = torch.arange(num_masks_out, device=prd_masks.device)
                prd_masks_best = prd_masks[rows, best_idx]      # [N, H, W] logits
                prd_ious_best = prd_scores[rows, best_idx]      # [N]
            else:
                # Only one mask returned, take index 0
                prd_masks_best = prd_masks[:, 0]                # [N, H, W] logits
                prd_ious_best = prd_scores[:, 0]                # [N]

            prd_mask_prob = torch.sigmoid(prd_masks_best)
            gt_mask = torch.tensor(masks.astype(np.float32), device=device)

            if gt_mask.shape[1:] != (orig_h, orig_w):
                gt_mask = F.interpolate(
                    gt_mask.unsqueeze(1),
                    size=(orig_h, orig_w),
                    mode='nearest',
                ).squeeze(1)

            # --- optional visualization ---
            if (
                visualize
                and vis_dir is not None
                and global_step % visualize_every == 0
                and sample_idx == 0
            ):
                visualize_predictions(
                    image=image,
                    gt_masks=gt_mask.cpu().numpy(),
                    pred_masks=prd_masks_best,
                    boxes=boxes,
                    save_path=vis_dir,
                    step=global_step,
                    max_samples=20,
                )

            # --- Loss for this sample ---
            losses = sam_loss(
                pred_masks=prd_masks_best,
                gt_masks=gt_mask,
                pred_ious=prd_ious_best,
                num_objects=num_masks,
                weight_dict=weight_dict,
                focal_alpha=getattr(config, 'loss_focal_alpha', 0.25),
                focal_gamma=getattr(config, 'loss_focal_gamma', 2.0),
                iou_use_l1=getattr(config, 'loss_iou_use_l1', True),
            )

            total_loss      = total_loss      + losses['loss']
            total_loss_mask = total_loss_mask + losses['loss_mask']
            total_loss_dice = total_loss_dice + losses['loss_dice']
            total_loss_iou  = total_loss_iou  + losses['loss_iou']

            # IoU logging
            pred_mask_binary = (prd_mask_prob > 0.5).float()
            inter = (gt_mask * pred_mask_binary).sum((1, 2))
            union = gt_mask.sum((1, 2)) + pred_mask_binary.sum((1, 2)) - inter + 1e-6
            iou = inter / union    # [N]
            all_ious.append(iou.detach())
            num_valid_samples += 1

        if num_valid_samples == 0:
            predictor.reset_predictor()
            torch.cuda.empty_cache()
            return None

        total_loss      = total_loss      / num_valid_samples
        total_loss_mask = total_loss_mask / num_valid_samples
        total_loss_dice = total_loss_dice / num_valid_samples
        total_loss_iou  = total_loss_iou  / num_valid_samples

    # single backward for the whole batch
    scaler.scale(total_loss).backward()
    torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=0.1)
    scaler.step(optimizer)
    scaler.update()

    predictor.reset_predictor()

    all_ious_cat = torch.cat(all_ious, dim=0)
    mean_iou = all_ious_cat.mean().item()

    return {
        'train/loss':      total_loss.detach().item(),
        'train/loss_mask': total_loss_mask.detach().item(),
        'train/loss_dice': total_loss_dice.detach().item(),
        'train/loss_iou':  total_loss_iou.detach().item(),
        'train/mean_iou':  mean_iou,
    }

def validate_sam2(predictor, valid_loader, device, config, epoch, vis_dir=None):
    """
    Validate SAM2 model with both simple AP and COCO mAP metrics.
    """
    print(f"\nValidating...")
    predictor.model.eval()
    
    all_ious = []
    all_losses = []
    all_loss_masks = []
    all_loss_dices = []
    all_loss_ious = []
    num_batches = 0
    
    # Visualization settings
    visualize = getattr(config, 'visualize_validation', False)
    max_vis_samples = getattr(config, 'max_validation_vis_samples', 3)
    vis_count = 0
    
    if visualize and vis_dir is not None:
        val_vis_dir = vis_dir / f'validation_epoch_{epoch}'
        val_vis_dir.mkdir(parents=True, exist_ok=True)
    else:
        val_vis_dir = None
    
    with torch.no_grad():
        pbar = tqdm(valid_loader, desc="Validation")
        
        for batch_idx, batch in enumerate(pbar, 1):

            if len(batch) == 0:
                continue
            
            data = batch[0]
            
            # Use batch_idx as image_id if not provided
            image_id = data.get("image_id", batch_idx)
            
            # Extract image
            image = data["image"].permute(1, 2, 0).cpu().numpy()
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            
            # Extract instances
            if "instances" not in data or len(data["instances"]) == 0:
                continue
            
            instances = data["instances"]
            boxes = instances.gt_boxes.tensor.cpu().numpy()
            masks = instances.gt_masks.tensor.cpu().numpy()
            
            if len(boxes) == 0:
                continue
            
            # Subsample to max prompts
            max_prompts = getattr(config, 'max_prompts_per_image', 64)
            if len(boxes) > max_prompts:
                idx = np.random.choice(len(boxes), max_prompts, replace=False)
                boxes = boxes[idx]
                masks = masks[idx]
            
            num_masks = len(boxes)
            
            # Set image
            predictor.set_image(image)
            orig_h, orig_w = predictor._orig_hw[-1]
            
            # Prepare prompts
            _, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                point_coords=None,
                point_labels=None,
                box=boxes,
                mask_logits=None,
                normalize_coords=True,
            )
            
            if unnorm_box is None or unnorm_box.shape[0] == 0:
                predictor.reset_predictor()
                continue
            
            # Encode prompts
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=None,
                boxes=unnorm_box,
                masks=None,
            )
            
            # Decode masks
            batched_mode = unnorm_box.shape[0] > 1
            
            high_res_features = [
                feat_level[-1].unsqueeze(0)
                for feat_level in predictor._features["high_res_feats"]
            ]
            multimask_output = False
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
            )
            prd_masks = predictor._transforms.postprocess_masks(
                low_res_masks, predictor._orig_hw[-1]
            )
            num_masks_out, num_candidates, H, W = prd_masks.shape

            if multimask_output:
                best_idx = prd_scores.argmax(dim=1)
                rows = torch.arange(num_masks_out, device=prd_masks.device)
                prd_masks_best = prd_masks[rows, best_idx]
                prd_ious_best = prd_scores[rows, best_idx]
            else:
                prd_masks_best = prd_masks[:, 0]
                prd_ious_best = prd_scores[:, 0]

            prd_mask_prob = torch.sigmoid(prd_masks_best)
            gt_mask = torch.tensor(masks.astype(np.float32), device=device)

            # Resize GT if needed
            if gt_mask.shape[1:] != (orig_h, orig_w):
                gt_mask = F.interpolate(
                    gt_mask.unsqueeze(1),
                    size=(orig_h, orig_w),
                    mode='nearest',
                ).squeeze(1)
            
            # --- Loss calculation ---
            weight_dict = {
                'loss_mask': getattr(config, 'loss_weight_mask', 20.0),
                'loss_dice': getattr(config, 'loss_weight_dice', 1.0),
                'loss_iou': getattr(config, 'loss_weight_iou', 1.0),
            }
            
            losses = sam_loss(
                pred_masks=prd_masks_best.detach(),
                gt_masks=gt_mask,
                pred_ious=prd_ious_best.detach(),
                num_objects=num_masks,
                weight_dict=weight_dict,
                focal_alpha=getattr(config, 'loss_focal_alpha', 0.25),
                focal_gamma=getattr(config, 'loss_focal_gamma', 2.0),
                iou_use_l1=getattr(config, 'loss_iou_use_l1', True),
            )
            
            # Compute IoU for each mask
            pred_mask_binary = (prd_mask_prob > 0.5).float()
            inter = (gt_mask * pred_mask_binary).sum((1, 2))
            union = gt_mask.sum((1, 2)) + pred_mask_binary.sum((1, 2)) - inter + 1e-6
            iou = inter / union

            all_losses.append(losses['loss'].detach().cpu().item())
            all_loss_masks.append(losses['loss_mask'].detach().cpu().item())
            all_loss_dices.append(losses['loss_dice'].detach().cpu().item())
            all_loss_ious.append(losses['loss_iou'].detach().cpu().item())
            all_ious.extend(iou.detach().cpu().numpy().tolist())
            
            num_batches += 1
            
            if visualize and val_vis_dir is not None and vis_count < max_vis_samples:
                visualize_predictions(
                    image=image,
                    gt_masks=gt_mask.cpu().numpy(),
                    pred_masks=prd_masks_best.detach(),
                    boxes=boxes,
                    save_path=val_vis_dir,
                    step=vis_count,
                    max_samples=20
                )
                vis_count += 1
            predictor.reset_predictor()
            pbar.set_postfix({
                'batches': num_batches,
                'mean_iou': f"{np.mean(all_ious):.4f}",
            })
    torch.cuda.empty_cache()

    # Compute final metrics
    metrics = {
        'val/loss': np.mean(all_losses) if all_losses else 0.0,
        'val/loss_mask': np.mean(all_loss_masks) if all_loss_masks else 0.0,
        'val/loss_dice': np.mean(all_loss_dices) if all_loss_dices else 0.0,
        'val/loss_iou': np.mean(all_loss_ious) if all_loss_ious else 0.0,
        'val/mean_iou': np.mean(all_ious) if all_ious else 0.0,
        'val/median_iou': np.median(all_ious) if all_ious else 0.0,
        'val/std_iou': np.std(all_ious) if all_ious else 0.0,
        'val/num_samples': len(all_ious),
    }
    
    print(f"\n✓ Validation complete:")
    print(f"  Batches: {num_batches}")
    print(f"  Samples: {metrics['val/num_samples']}")
    print(f"  Mean IoU: {metrics['val/mean_iou']:.4f}")
    print(f"  Loss: {metrics['val/loss']:.4f}")
    
    return metrics

def train_sam2(config: SegmenterConfig):
    """
    Train SAM2 - following Detectron2's train_detectron2_fasterrcnn pattern.
    """
    print(f"\n{'='*80}")
    print(f"SAM2 FINE-TUNING")
    print(f"{'='*80}\n")
    
    train_dataset_name, valid_dataset_name = setup_sam2_datasets(config)
    
    print(f"Setting up trainer for dataset: {train_dataset_name}")
    
    u = uuid.uuid4()
    now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    if slurm_job_id:
        model_name = f"{config.model}_{now}_{slurm_job_id}"
    else:
        model_name = f"{config.model}_{now}_{u.hex[:4]}"
    
    trainer_components = setup_sam2_trainer(train_dataset_name, valid_dataset_name, config, model_name)
    
    # Extract components
    predictor = trainer_components['predictor']
    optimizer = trainer_components['optimizer']
    scheduler = trainer_components['scheduler']
    scaler = trainer_components['scaler']
    train_loader = trainer_components['train_loader']
    valid_loader = trainer_components['valid_loader'] 
    output_dir = trainer_components['output_dir']
    device = trainer_components['device']
    steps_per_epoch = trainer_components['steps_per_epoch']
    
    visualize = getattr(config, 'visualize_batches', False)
    focal_loss = getattr(config, 'loss_weight_mask', 0.0)
    vis_dir = None
    if visualize:
        vis_dir = Path(getattr(config, 'visualize_path', output_dir / 'visualizations'))
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Wandb
    # Keep track of which layers are frozen
    freeze_layers = []
    if getattr(config, 'freeze_image_encoder', True):
        freeze_layers.append('img')
    if getattr(config, 'freeze_prompt_encoder', False):
        freeze_layers.append('prompt')
    if getattr(config, 'freeze_mask_decoder', False):
        freeze_layers.append('mask')
    freeze_str = '_'.join(freeze_layers) if freeze_layers else 'none'

    noise_scale = getattr(config, 'box_noise_scale', 0.0)
    lr_image_encoder = getattr(config, 'lr_image_encoder', 1e-5)

    if config.wandb_enabled:
        wandb.init(
            project=config.wandb_project,
            name=f"{model_name}_noise{noise_scale}_lrie{lr_image_encoder}",
            config=vars(config)
        )

    best_val_iou = 0.0
    best_val_ap = 0.0 

    # Training loop
    print("Starting training...")
    
    global_step = 0

    for epoch in range(config.max_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.max_epochs}")
        print(f"{'='*60}")
        data_iter = iter(train_loader)
        # Training 
        predictor.model.train()
        steps_per_epoch = 1
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}")
        i = 0
        for _ in pbar:
            # In case my iterator ends
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            
            metrics = train_step(predictor, batch, optimizer, scaler,device, config , global_step=global_step, vis_dir=vis_dir)
            
            if metrics is None:
                continue
            
            global_step += 1
            scheduler.step()
            pbar.set_postfix({
                'loss': f"{metrics['train/loss']:.4f}",
                'iou': f"{metrics['train/mean_iou']:.4f}",
                'loss_mask': f"{metrics['train/loss_mask']:.4f}",
                'loss_dice': f"{metrics['train/loss_dice']:.4f}",
                'loss_iou': f"{metrics['train/loss_iou']:.4f}",
            })
            
            # Log to wandb
            if wandb.run:
                wandb.log(
                    {
                        **metrics,
                        "lr/group_0": optimizer.param_groups[0]["lr"],
                        "lr/group_1": optimizer.param_groups[1]["lr"],
                    },
                    step=global_step,
                )
            del batch

        # Clean up before validation
        del data_iter
        gc.collect()
        torch.cuda.empty_cache()

        val_metrics = validate_sam2(
            predictor=predictor,
            valid_loader=valid_loader,
            device=device,
            config=config,
            epoch=epoch + 1,
            vis_dir=vis_dir
        )
       
        latest_path = output_dir / "model_latest.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': predictor.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_iou': best_val_iou,
        }, latest_path)
        torch.save(predictor.model.state_dict(), latest_path)

        ########################################################################################
        # COCO evaluations
        print(f"\n[coco_eval] Running full pipeline evaluations at epoch {epoch + 1}...")
        coco_metrics = {}
        try:
            coco_metrics = run_coco_evaluations(
                config=config,
                checkpoint_path=latest_path,
                epoch=epoch + 1,
                )
        except Exception as exc:
                print(f"[coco_eval] Error while running evaluations: {exc}")

        # Log validation metrics
        if coco_metrics:
            val_metrics.update(coco_metrics)
            print(f"\n✓ COCO evaluation metrics:")
            for k, v in coco_metrics.items():
                print(f"  {k}: {v:.4f}")
        ########################################################################################


        if wandb.run:
            wandb.log(val_metrics, step=global_step)

        current_ap = val_metrics.get('coco/AP', 0.0)
        current_iou = val_metrics['val/mean_iou']
        # Save best model
        if current_ap > best_val_ap:
            best_val_ap = current_ap
            best_path = output_dir / "model_best.pt"
            torch.save(predictor.model.state_dict(), best_path)
            print(f"✓ New best model saved: AP={best_val_ap:.4f}, IoU={best_val_iou:.4f}")

        if current_iou > best_val_iou:
            best_val_iou = current_iou
            best_path = output_dir / "model_best_iou.pt"
            torch.save(predictor.model.state_dict(), best_path)
            print(f"✓ New best model (IoU) saved: IoU={best_val_iou:.4f}")
            
            ## Log best model to W&B (overwrites previous version)
            #if wandb.run:
            #    artifact = wandb.Artifact(
            #        name="model-best",
            #        type="model",
            #        description=f"Best model at epoch {epoch + 1} with IoU={best_val_iou:.4f}",
            #        metadata={
            #            "epoch": epoch + 1,
            #            "val_iou": best_val_iou,
            #            "val_loss": val_metrics['val/loss'],
            #        }
            #    )
            #    artifact.add_file(str(best_path))
            #    wandb.log_artifact(artifact, aliases=["latest", "best"])
            #    print(f"✓ Best model logged to W&B")

        # IF we
        
        # Log latest checkpoint to W&B (overwrites previous version)
        #if wandb.run:
        #    artifact = wandb.Artifact(
        #        name="model-latest",
        #        type="checkpoint",
        #        description=f"Latest checkpoint at epoch {epoch + 1}",
        #        metadata={
        #            "epoch": epoch + 1,
        #            "val_iou": val_metrics['val/mean_iou'],
        #            "val_loss": val_metrics['val/loss'],
        #            "best_val_iou": best_val_iou,
        #        }
        #    )
        #    artifact.add_file(str(latest_path))
        #    wandb.log_artifact(artifact, aliases=["latest"])
        #    print(f"✓ Latest checkpoint logged to W&B")
            
        gc.collect()

    # Final model
    final_path = output_dir / "model_final.pt"
    torch.save(predictor.model.state_dict(), final_path)
    
    if wandb.run:
        wandb.finish()
    
    print(f"Training completed. Model saved in {output_dir}")
