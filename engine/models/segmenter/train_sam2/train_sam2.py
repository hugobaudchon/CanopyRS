import os
import uuid
from datetime import datetime
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import wandb
from engine.config_parsers import SegmenterConfig
from sam2.sam2_image_predictor import SAM2ImagePredictor
from detectron2.config import get_cfg
from .loss_fns import sam2_loss
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from detectron2.data import build_detection_train_loader, build_detection_test_loader, DatasetMapper, DatasetCatalog
from .augmentation import AugmentationAdder

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
    
    for name, param in predictor.model.named_parameters():
        if freeze_image_encoder and "image_encoder" in name:
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
    
    return predictor




def build_sam2_train_loader(cfg, dataset_name):
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
        total_batch_size=1,  # SAM2 uses batch_size=1
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
    
    from engine.models.segmenter.train_sam2.dataset import register_sam2_dataset_with_masks    
    
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
    cfg.DATALOADER.NUM_WORKERS = config.dataloader_num_workers
    cfg.INPUT.MASK_FORMAT = "bitmask"
    
    # Output directory
    output_dir = Path(config.train_output_path) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR = str(output_dir)
    
    # Save config
    config.to_yaml(str(output_dir / "config.yaml"))
    
    train_loader = build_sam2_train_loader(cfg, train_dataset_name)
    valid_loader = build_sam2_test_loader(cfg, valid_dataset_name)
    
    print(f"\nDataset sizes:")
    print(f"  Training: {len(DatasetCatalog.get(train_dataset_name))} images")
    print(f"  Validation: {len(DatasetCatalog.get(valid_dataset_name))} images")
    
    # Build model
    predictor = build_sam2_model(config)
    predictor.model.to(device)
    predictor.model.train()
    
    # Optimizer
    trainable_params = [p for p in predictor.model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.lr,
        weight_decay=getattr(config, 'weight_decay', 1e-4),
    )
    
    # Scheduler
    steps_per_epoch = len(DatasetCatalog.get(train_dataset_name))
    print(f"\nSteps per epoch: {steps_per_epoch}")
    max_steps = config.max_epochs * steps_per_epoch
    
    
    scheduler_epochs_steps = getattr(config, 'scheduler_epochs_steps', None)

    if scheduler_epochs_steps is None or len(scheduler_epochs_steps) == 0:
        step_size = 10  # every 10 epochs
    else:
        step_size = scheduler_epochs_steps[0]
    scheduler_gamma = getattr(config, 'scheduler_gamma', 0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,  # number of epochs
        gamma=scheduler_gamma,
    )
    scaler = torch.amp.GradScaler(enabled=getattr(config, 'use_amp', True))
    
    print(f"\nTraining configuration:")
    print(f"  Max epochs: {config.max_epochs}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {max_steps}")
    print(f"  Learning rate: {config.lr}")
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
    Single training step - matches your working version with Detectron2 data loading.
    """
    # Extract from Detectron2 format
    if len(batch) == 0:
        return None
    
    data = batch[0]

    # Extract image
    image = data["image"].permute(1, 2, 0).cpu().numpy()
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Extract instances
    if "instances" not in data or len(data["instances"]) == 0:
        return None
    
    instances = data["instances"]
    boxes = instances.gt_boxes.tensor.cpu().numpy()
    masks = instances.gt_masks.tensor.cpu().numpy()
    
    if len(boxes) == 0:
        return None
    
    # Subsample to max prompts
    max_prompts = getattr(config, 'max_prompts_per_image', 64)
    if len(boxes) > max_prompts:
        idx = np.random.choice(len(boxes), max_prompts, replace=False)
        boxes = boxes[idx]
        masks = masks[idx]
    
    num_masks = len(boxes)

    input_point = np.zeros((num_masks, 1, 2), dtype=np.float32)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        input_point[i, 0] = [cx, cy]
    
    input_label = np.ones((num_masks, 1), dtype=np.float32)
    
    # Train with SAM2
    with torch.amp.autocast(device_type='cuda', enabled=getattr(config, 'use_amp', True)):
        # Set image
        predictor.set_image(image)
        orig_h, orig_w = predictor._orig_hw[-1]
        
        _, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
            point_coords=input_point,
            point_labels=input_label,
            box=boxes,
            mask_logits=None,
            normalize_coords=True,
        )
        
        if unnorm_box is None or unnorm_box.shape[0] == 0:
            return None
        
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=(unnorm_coords, labels),
            boxes=unnorm_box,
            masks=None,
        )
        
        # Decode masks
        batched_mode = unnorm_coords.shape[0] > 1
        
        high_res_features = [
            feat_level[-1].unsqueeze(0)
            for feat_level in predictor._features["high_res_feats"]
        ]
        
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )
        
        prd_masks = predictor._transforms.postprocess_masks(
            low_res_masks, predictor._orig_hw[-1]
        )
        num_masks, num_candidates, H, W = prd_masks.shape

        best_idx = prd_scores.argmax(dim=1)  # [N]
        rows = torch.arange(num_masks, device=prd_masks.device)

        prd_masks_best = prd_masks[rows, best_idx]      # [N, H, W] (logits)
        prd_ious_best = prd_scores[rows, best_idx]      # [N]

        prd_mask = torch.sigmoid(prd_masks_best)
        gt_mask = torch.tensor(masks.astype(np.float32), device=device)
        
        # Resize GT if needed
        if gt_mask.shape[1:] != (orig_h, orig_w):
            gt_mask = F.interpolate(
                gt_mask.unsqueeze(1),
                size=(orig_h, orig_w),
                mode='nearest',
            ).squeeze(1)

        
        visualize = getattr(config, 'visualize_batches', False)
        visualize_every = getattr(config, 'visualize_every_n_steps', 100)
        
        if visualize and vis_dir is not None and global_step % visualize_every == 0:
            visualize_predictions(
                image=image,
                gt_masks=gt_mask.cpu().numpy(),
                pred_masks=prd_masks_best,  # Logits for visualization
                boxes=boxes,
                save_path=vis_dir,
                step=global_step,
                max_samples=20
            )
    
        
        # SAM2 official loss (focal + dice + iou)
        weight_dict = {
            'loss_mask': getattr(config, 'loss_weight_mask', 1.0),
            'loss_dice': getattr(config, 'loss_weight_dice', 1.0),
            'loss_iou': getattr(config, 'loss_weight_iou', 1.0),
        }
        
        losses = sam2_loss(
            pred_masks=prd_masks_best,  # Use logits
            gt_masks=gt_mask,
            pred_ious=prd_ious_best,
            num_objects=num_masks,
            weight_dict=weight_dict,
            focal_alpha=getattr(config, 'loss_focal_alpha', 0.25),
            focal_gamma=getattr(config, 'loss_focal_gamma', 2.0),
            iou_use_l1=getattr(config, 'loss_iou_use_l1', False),
        )
        
        total_loss = losses['loss']
            
        # Backward
        scaler.scale(total_loss).backward()
        torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    predictor.reset_predictor()
    
    # Compute IoU for logging
    with torch.no_grad():
        pred_mask_binary = prd_mask > 0.5
        inter = (gt_mask * pred_mask_binary).sum((1, 2))
        union = gt_mask.sum((1, 2)) + pred_mask_binary.sum((1, 2)) - inter + 1e-6
        iou = inter / union
    
    return {
        'train/loss': losses['loss'].item(),
        'train/loss_mask': losses['loss_mask'].item(),
        'train/loss_dice': losses['loss_dice'].item(),
        'train/loss_iou': losses['loss_iou'].item(),
        'train/mean_iou': iou.mean().item(),
    }

def validate_sam2(predictor, valid_loader, device, config, epoch, vis_dir=None):
    """
    Validate SAM2 model
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
                # Use random sampling to get diverse samples
                idx = np.random.choice(len(boxes), max_prompts, replace=False)
                boxes = boxes[idx]
                masks = masks[idx]
            
            num_masks = len(boxes)
            
            # Create point prompts
            input_point = np.zeros((num_masks, 1, 2), dtype=np.float32)
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                input_point[i, 0] = [cx, cy]
            
            input_label = np.ones((num_masks, 1), dtype=np.float32)
            
            # Set image
            predictor.set_image(image)
            orig_h, orig_w = predictor._orig_hw[-1]
            
            # Prepare prompts
            _, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                point_coords=input_point,
                point_labels=input_label,
                box=boxes,
                mask_logits=None,
                normalize_coords=True,
            )
            
            if unnorm_box is None or unnorm_box.shape[0] == 0:
                predictor.reset_predictor()
                continue
            
            # Encode prompts
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels),
                boxes=unnorm_box,
                masks=None,
            )
            
            # Decode masks
            batched_mode = unnorm_coords.shape[0] > 1
            
            high_res_features = [
                feat_level[-1].unsqueeze(0)
                for feat_level in predictor._features["high_res_feats"]
            ]
            
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
            )
            
            prd_masks = predictor._transforms.postprocess_masks(
                low_res_masks, predictor._orig_hw[-1]
            )
            
            # Select best mask
            num_masks_out, num_candidates, H, W = prd_masks.shape
            best_idx = prd_scores.argmax(dim=1)
            rows = torch.arange(num_masks_out, device=prd_masks.device)
            
            prd_masks_best = prd_masks[rows, best_idx]
            prd_ious_best = prd_scores[rows, best_idx]
            prd_mask = torch.sigmoid(prd_masks_best)
            
            # Get ground truth masks
            gt_mask = torch.tensor(masks.astype(np.float32), device=device)
            
            # Resize GT if needed
            if gt_mask.shape[1:] != (orig_h, orig_w):
                gt_mask = F.interpolate(
                    gt_mask.unsqueeze(1),
                    size=(orig_h, orig_w),
                    mode='nearest',
                ).squeeze(1)
            
            weight_dict = {
                'loss_mask': getattr(config, 'loss_weight_mask', 1.0),
                'loss_dice': getattr(config, 'loss_weight_dice', 1.0),
                'loss_iou': getattr(config, 'loss_weight_iou', 1.0),
            }
            
            losses = sam2_loss(
                pred_masks=prd_masks_best.detach(),
                gt_masks=gt_mask,
                pred_ious=prd_ious_best.detach(),
                num_objects=num_masks,
                weight_dict=weight_dict,
                focal_alpha=getattr(config, 'loss_focal_alpha', 0.25),
                focal_gamma=getattr(config, 'loss_focal_gamma', 2.0),
                iou_use_l1=getattr(config, 'loss_iou_use_l1', False),
            )
            
            pred_mask_binary = (prd_mask.detach() > 0.5).float()
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
            
            # Update progress bar
            pbar.set_postfix({
                'batches': num_batches,
                'mean_iou': f"{np.mean(all_ious):.4f}",
            })
    
    # Clear cache after validation
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
    print(f"  Median IoU: {metrics['val/median_iou']:.4f}")
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
    
    visualize = getattr(config, 'visualize_batches', False)
    vis_dir = None
    if visualize:
        vis_dir = Path(getattr(config, 'visualize_path', output_dir / 'visualizations'))
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Wandb
    if config.wandb_enabled:
        wandb.init(
            project=config.wandb_project,
            name=model_name,
            config=vars(config),
        )

    steps_per_epoch = len(DatasetCatalog.get(train_dataset_name))
    best_val_iou = 0.0

    # Training loop
    print("Starting training...")
    
    global_step = 0
    data_iter = iter(train_loader)

    for epoch in range(config.max_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.max_epochs}")
        print(f"{'='*60}")
        
        # Training 
        predictor.model.train()
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}")
        i = 0
        for batch in pbar:
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
            pbar.set_postfix({
                'loss': f"{metrics['train/loss']:.4f}",
                'iou': f"{metrics['train/mean_iou']:.4f}",
                'loss_mask': f"{metrics['train/loss_mask']:.4f}",
                'loss_dice': f"{metrics['train/loss_dice']:.4f}",
                'loss_iou': f"{metrics['train/loss_iou']:.4f}",
            })
            
            # Log to wandb
            if wandb.run:
                wandb.log(metrics, step=global_step)

        val_metrics = validate_sam2(
            predictor=predictor,
            valid_loader=valid_loader,
            device=device,
            config=config,
            epoch=epoch + 1,
            vis_dir=vis_dir
        )
        
        # Log validation metrics
        if wandb.run:
            wandb.log(val_metrics, step=global_step)
        
        # Save best model
        if val_metrics['val/mean_iou'] > best_val_iou:
            best_val_iou = val_metrics['val/mean_iou']
            best_path = output_dir / "model_best.pt"
            torch.save(predictor.model.state_dict(), best_path)
            print(f"✓ New best model saved: IoU={best_val_iou:.4f}")

        # Save checkpoint
        ckpt_path = output_dir / f"model_epoch_{epoch + 1}.pt"
        torch.save(predictor.model.state_dict(), ckpt_path)
        print(f"✓ Checkpoint saved: {ckpt_path}")

        # Step scheduler
        scheduler.step()
    
    # Final model
    final_path = output_dir / "model_final.pt"
    torch.save(predictor.model.state_dict(), final_path)
    
    if wandb.run:
        wandb.finish()
    
    print(f"Training completed. Model saved in {output_dir}")