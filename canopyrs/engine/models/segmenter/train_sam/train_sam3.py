import os
import uuid
import math
import gc
from datetime import datetime
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import wandb
from PIL import Image
from canopyrs.engine.config_parsers import SegmenterConfig

from transformers import Sam3TrackerProcessor, Sam3TrackerModel
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, build_detection_test_loader, DatasetMapper, DatasetCatalog
from canopyrs.engine.models.segmenter.train_sam.augmentation import AugmentationAdder
from canopyrs.engine.models.segmenter.train_sam.loss_fns import sam_loss
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import cv2
from canopyrs.engine.models.segmenter.train_sam.dataset import register_sam2_dataset_with_masks, register_sam2_dataset_with_predicted_boxes
from canopyrs.engine.config_parsers import InferIOConfig, PipelineConfig
from canopyrs.engine.config_parsers.base import get_config_path
from canopyrs.engine.pipeline import Pipeline
from canopyrs.engine.benchmark.detector.evaluator import CocoEvaluator
from copy import deepcopy
    
def run_coco_evaluations(
    config: SegmenterConfig,
    checkpoint_path: Path,
    epoch: int,
    valid_detector_coco_data=None,
) -> dict[str, float]:
    """
    Run full pipeline (detector → SAM) evaluation on validation set.
    Returns COCO metrics comparing predictions to GT.
    """

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
        dataset_output_dir = output_root / dataset_name
        io_config = InferIOConfig(
            input_imagery=str(tiles_path),  # Required field
            tiles_path=str(tiles_path),
            output_folder=str(dataset_output_dir),
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

def visualize_predictions(
    image,
    gt_masks,
    pred_masks,
    boxes,
    save_path,
    step,
    max_samples=20,
    pred_boxes=None,   # <- NEW
):
    """Visualize ground truth (Green) vs predicted masks (Red),
    plus GT boxes (yellow) and optionally matched predicted boxes (cyan)."""
    save_path.mkdir(parents=True, exist_ok=True)
    num_masks = min(len(gt_masks), max_samples)
    
    # Ensure image is HxWxC
    H, W = image.shape[:2]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Helper to create transparent overlay
    def create_overlay(masks, color, target_shape=None):
        # color should be (R, G, B) tuple
        if len(masks) == 0:
            return np.zeros((target_shape[0], target_shape[1], 4)) if target_shape else np.zeros((100, 100, 4))
            
        H_target, W_target = target_shape if target_shape else masks[0].shape
        overlay = np.zeros((H_target, W_target, 4), dtype=np.float32)
        
        # Create a single combined mask for the color layer
        combined_mask = np.zeros((H_target, W_target), dtype=np.float32)
        
        for i in range(len(masks)):
            mask = masks[i]
            if mask.shape != (H_target, W_target):
                mask = cv2.resize(mask.astype(np.float32), (W_target, H_target), interpolation=cv2.INTER_NEAREST)
            combined_mask = np.maximum(combined_mask, mask)
            
        # Apply color
        for c in range(3):
            overlay[:, :, c] = color[c]
            
        # Alpha channel: 0.6 where mask is present
        overlay[:, :, 3] = (combined_mask > 0.5).astype(np.float32) * 0.6
            
        return overlay

    # 1. Image + Boxes
    axes[0].imshow(image)

    # GT boxes (yellow)
    for i in range(min(num_masks, len(boxes))):
        x1, y1, x2, y2 = boxes[i]
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor='yellow', linewidth=2
        )
        axes[0].add_patch(rect)

    # Matched predicted boxes (cyan), if provided
    if pred_boxes is not None:
        for i in range(min(num_masks, len(pred_boxes))):
            px1, py1, px2, py2 = pred_boxes[i]
            rect_p = mpatches.Rectangle(
                (px1, py1), px2 - px1, py2 - py1,
                fill=False, edgecolor='cyan', linewidth=2, linestyle='--'
            )
            axes[0].add_patch(rect_p)

        axes[0].set_title(f'Image + {num_masks} GT boxes (yellow) + matched pred boxes (cyan)')
    else:
        axes[0].set_title(f'Image + {num_masks} GT Boxes')

    axes[0].axis('off')
    
    # 2. Ground Truth (GREEN)
    axes[1].imshow(image)
    if num_masks > 0:
        gt_overlay = create_overlay(gt_masks[:num_masks], color=(0, 1, 0), target_shape=(H, W))
        axes[1].imshow(gt_overlay)
    axes[1].set_title(f'{num_masks} Ground Truth (Green)')
    axes[1].axis('off')
    
    # 3. Predictions (RED)
    axes[2].imshow(image)
    
    processed_preds = []
    all_iou = []
    
    for i in range(num_masks):
        if isinstance(pred_masks, torch.Tensor):
            pred_prob = torch.sigmoid(pred_masks[i]).detach().cpu().numpy()
        else:
            pred_prob = pred_masks[i]
        
        processed_preds.append(pred_prob)
        
        # Calculate IoU
        gt_mask_i = gt_masks[i]
        if pred_prob.shape != gt_mask_i.shape:
            pred_prob_resized = cv2.resize(
                pred_prob,
                (gt_mask_i.shape[1], gt_mask_i.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            pred_prob_resized = pred_prob

        pred_binary = (pred_prob_resized > 0.5).astype(float)
        inter = (gt_mask_i * pred_binary).sum()
        union = (gt_mask_i + pred_binary).clip(0, 1).sum()
        all_iou.append(inter / (union + 1e-6))
    
    if num_masks > 0:
        pred_overlay = create_overlay(processed_preds, color=(1, 0, 0), target_shape=(H, W))
        axes[2].imshow(pred_overlay)
    
    mean_iou = np.mean(all_iou) if all_iou else 0.0
    axes[2].set_title(f'Predictions (Red) - Mean IoU={mean_iou:.3f}')
    axes[2].axis('off')
    
    plt.tight_layout()
    out_file = save_path / f'step_{step:06d}.png'
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

def build_sam3_model(config: SegmenterConfig):
    """Build SAM3 Tracker model with proper freezing strategy """
    MODEL_MAPPING = {
        't': "facebook/sam3",
        's': "facebook/sam3",
        'b': "facebook/sam3",
        'l': "facebook/sam3",
    }

    model_name = MODEL_MAPPING.get(config.architecture, "facebook/sam3")

    print(f"\n{'='*80}")
    print(f"Building SAM3 Tracker Model (PVS)")
    print(f"{'='*80}")
    print(f"Architecture: {config.architecture}")
    print(f"Model: {model_name}")

    # 🔁 Tracker instead of PCS model
    processor = Sam3TrackerProcessor.from_pretrained(model_name)
    model = Sam3TrackerModel.from_pretrained(model_name)

    if config.checkpoint_path:
        print(f"Loading from fine-tuned checkpoint: {config.checkpoint_path}")
        state_dict = torch.load(config.checkpoint_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'], strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
        print(f"✓ Fine-tuned weights loaded")

    # Freezing strategy (same as SAM2)
    freeze_image_encoder = getattr(config, 'freeze_image_encoder', True)
    freeze_prompt_encoder = getattr(config, 'freeze_prompt_encoder', False)
    freeze_mask_decoder = getattr(config, 'freeze_mask_decoder', False)

    print(f"\nFreezing strategy:")
    print(f"  Image Encoder: {'FROZEN' if freeze_image_encoder else 'TRAINABLE'}")
    print(f"  Prompt Encoder: {'FROZEN' if freeze_prompt_encoder else 'TRAINABLE'}")
    print(f"  Mask Decoder: {'FROZEN' if freeze_mask_decoder else 'TRAINABLE'}")

    # Sam3Tracker parameter names look like SAM2:
    #   vision_encoder.*, prompt_encoder.*, mask_decoder.*
    for name, param in model.named_parameters():
        param.requires_grad = True  # default

        # Freeze vision encoder
        if freeze_image_encoder and ("vision_encoder" in name or "image_encoder" in name):
            param.requires_grad = False
        # Freeze prompt encoder
        elif freeze_prompt_encoder and "prompt_encoder" in name:
            param.requires_grad = False
        # Freeze mask decoder
        elif freeze_mask_decoder and "mask_decoder" in name:
            param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen: {total_params - trainable_params:,}")
    print(f"  Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    print(f"{'='*80}\n")

    return processor, model


def setup_sam3_datasets(config: SegmenterConfig):
    """Register SAM3 datasets."""
    print("Setting up datasets...")
    
    from canopyrs.engine.models.segmenter.train_sam.dataset import register_sam2_dataset_with_masks    
    use_detector_boxes = getattr(config, 'use_detector_boxes', True)
    if use_detector_boxes:
        print("Using predicted boxes from detector for prompts.")
        detector_config = getattr(config, 'detector_config_path', None)
        cache_dir = getattr(config, 'detector_cache_dir', '/tmp/detector_cache')
        
        if detector_config is None:
            raise ValueError("detector_config_path must be set when use_detector_boxes=True")
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


def build_sam3_train_loader(cfg, dataset_name, batch_size=1):
    """Build SAM3 training DataLoader."""
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
        total_batch_size=batch_size,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )
    
    return loader


def build_sam3_test_loader(cfg, dataset_name):
    """Build SAM3 validation DataLoader."""
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


def setup_sam3_trainer(train_dataset_name: str, valid_dataset_name: str, config: SegmenterConfig, model_name: str):
    """Set up SAM3 training components."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = get_cfg()
    AugmentationAdder.modify_detectron2_augmentation_config(config, cfg)
    
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (valid_dataset_name,)
    cfg.DATALOADER.NUM_WORKERS = min(getattr(config, 'dataloader_num_workers', 4), 2)
    cfg.INPUT.MASK_FORMAT = "bitmask"
    
    output_dir = Path(config.train_output_path) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR = str(output_dir)
    
    config.to_yaml(str(output_dir / "config.yaml"))
    
    train_loader = build_sam3_train_loader(cfg, train_dataset_name, batch_size=getattr(config, 'batch_size', 1))
    valid_loader = build_sam3_test_loader(cfg, valid_dataset_name)
    
    train_dataset_dicts = DatasetCatalog.get(train_dataset_name)
    valid_dataset_dicts = DatasetCatalog.get(valid_dataset_name)
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset_dicts)} images")
    print(f"  Validation: {len(valid_dataset_dicts)} images")
    
    # Build model
    processor, model = build_sam3_model(config)
    model.to(device)
    model.train()
    
    encoder_params = []
    non_encoder_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "vision_encoder" in name or "image_encoder" in name:
            encoder_params.append(p)
        else:
            non_encoder_params.append(p)
    
    lr_encoder = getattr(config, 'lr_image_encoder', config.lr * 0.1)
    lr_others = getattr(config, 'lr_others', config.lr)
    
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": lr_encoder},
        {"params": non_encoder_params, "lr": lr_others},
    ], weight_decay=getattr(config, 'weight_decay', 0.01))
    
    batch_size = getattr(config, 'batch_size', 1)
    dataset_size = len(train_dataset_dicts)
    steps_per_epoch = math.ceil(dataset_size / batch_size)
    max_steps = config.max_epochs * steps_per_epoch
    warmup_steps = int(0.05 * max_steps)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(enabled=getattr(config, 'use_amp', True))
    
    print(f"\nTraining configuration:")
    print(f"  Max epochs: {config.max_epochs}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {max_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  LR (encoder): {lr_encoder}")
    print(f"  LR (others): {lr_others}")
    print(f"  Output dir: {output_dir}")
    
    return {
        'processor': processor,
        'model': model,
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


def train_step(processor, model, batch, optimizer, scaler, device, config, global_step=0, vis_dir=None):
    """Single SAM3 training step (PVS-style)."""
    if len(batch) == 0:
        return None

    weight_dict = {
        'loss_mask': getattr(config, 'loss_weight_mask', 0.0),
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
            if "image" not in data:
                continue

            # C,H,W → H,W,C, uint8 in [0,255]
            image = data["image"].permute(1, 2, 0).cpu().numpy()
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

            if "instances" not in data or len(data["instances"]) == 0:
                continue

            instances = data["instances"]
            boxes = instances.gt_boxes.tensor.cpu().numpy()
            masks = instances.gt_masks.tensor.cpu().numpy()

            if len(boxes) == 0:
                continue

            H, W = image.shape[:2]

            # Box jitter (like SAM2)
            if getattr(config, 'box_noise_scale', 0.0) > 0:
                box_widths = boxes[:, 2] - boxes[:, 0]
                box_heights = boxes[:, 3] - boxes[:, 1]
                noise = np.random.normal(0, config.box_noise_scale, boxes.shape)
                noise[:, 0] *= box_widths
                noise[:, 2] *= box_widths
                noise[:, 1] *= box_heights
                noise[:, 3] *= box_heights
                boxes = boxes + noise
                boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, W)
                boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, H)

            # Subsample prompts
            max_prompts = getattr(config, 'max_prompts_per_image', 64)
            if len(boxes) > max_prompts:
                idx = np.random.choice(len(boxes), max_prompts, replace=False)
                boxes = boxes[idx]
                masks = masks[idx]

            # Clip boxes to image bounds (just in case)
            boxes[:, 0] = np.clip(boxes[:, 0], 0, W)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, H)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, W)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, H)

            num_boxes = len(boxes)
            if num_boxes == 0:
                continue

            pil_image = Image.fromarray(image).convert("RGB")

            input_boxes = [boxes.tolist()]  # [batch=1, num_objects, 4]

            inputs = processor(
                images=pil_image,
                input_boxes=input_boxes,
                return_tensors="pt",
            ).to(device)

            # One mask per object (no multi-mask)
            outputs = model(**inputs, multimask_output=False)

            # pred_masks raw logits at low res
            pred_masks_all = outputs.pred_masks  # shapes vary; handle both 4D and 5D
            pred_ious_all = outputs.iou_scores  # (1, Nobj)

            # possible shapes:
            # (1, Nobj, Hm, Wm) or (1, Nobj, 1, Hm, Wm)
            if pred_masks_all.ndim == 5:
                # (B, Nobj, 1, Hm, Wm) -> (B, Nobj, Hm, Wm)
                pred_masks_all = pred_masks_all[:, :, 0]

            pred_masks_all = pred_masks_all.squeeze(0)  # (Nobj, Hm, Wm)
            pred_ious_all = pred_ious_all.squeeze(0)  # (Nobj,)
            pred_ious_all = pred_ious_all.flatten()

            num_pred = pred_masks_all.shape[0]
            num_matched = min(num_pred, num_boxes)
            if num_matched == 0:
                continue

            # Align by index: first K GT boxes ↔ first K predicted masks
            pred_masks_logits = pred_masks_all[:num_matched]  # (Nm, Hm, Wm)
            pred_iou_scores = pred_ious_all[:num_matched]  # (Nm,)
            
            eps = 1e-6
            pred_iou_scores_clamped = pred_iou_scores.clamp(eps, 1 - eps)
            pred_iou_logits = torch.log(pred_iou_scores_clamped / (1 - pred_iou_scores_clamped))
            
            gt_mask = torch.tensor(
                masks[:num_matched].astype(np.float32),
                device=device,
            )  # (Nm, H_orig, W_orig)

            # Resize GT masks down to mask decoder resolution
            Hm, Wm = pred_masks_logits.shape[-2:]
            if gt_mask.shape[1:] != (Hm, Wm):
                gt_mask = F.interpolate(
                    gt_mask.unsqueeze(1),
                    size=(Hm, Wm),
                    mode='nearest',
                ).squeeze(1)  # (Nm, Hm, Wm)

            # ---- Estimate IoU per instance (for logging / IoU loss target) ----
            with torch.no_grad():
                pred_mask_sigmoid = torch.sigmoid(pred_masks_logits)
                pred_mask_binary = (pred_mask_sigmoid > 0.5).float()
                inter = (gt_mask * pred_mask_binary).sum((1, 2))
                union = gt_mask.sum((1, 2)) + pred_mask_binary.sum((1, 2)) - inter + 1e-6
                actual_iou_calculated = inter / union  # (Nm,)

            # ---- Optional visualization ----
            if visualize and vis_dir is not None and global_step % visualize_every == 0:
                print(f"Visualizing training predictions at step {global_step}...")
                pred_probs = pred_mask_sigmoid.detach().cpu().numpy()

                visualize_predictions(
                    image=image,
                    gt_masks=gt_mask.cpu().numpy(),
                    pred_masks=pred_probs,
                    boxes=boxes[:num_matched],
                    save_path=vis_dir,
                    step=global_step,
                    max_samples=20,
                )
          
            # ---- Compute SAM2-style loss on matched masks ----
            losses = sam_loss(
                pred_masks=pred_masks_logits,
                gt_masks=gt_mask,
                pred_ious=pred_iou_logits,
                num_objects=num_matched,
                weight_dict=weight_dict,
                focal_alpha=getattr(config, 'loss_focal_alpha', 0.25),
                focal_gamma=getattr(config, 'loss_focal_gamma', 2.0),
                iou_use_l1=getattr(config, 'loss_iou_use_l1', True),
            )

            total_loss += losses['loss']
            total_loss_mask += losses['loss_mask']
            total_loss_dice += losses['loss_dice']
            total_loss_iou += losses['loss_iou']

            all_ious.append(actual_iou_calculated.detach())
            num_valid_samples += 1

        # end for data in batch

        if num_valid_samples == 0:
            torch.cuda.empty_cache()
            return None

        total_loss = total_loss / num_valid_samples
        total_loss_mask = total_loss_mask / num_valid_samples
        total_loss_dice = total_loss_dice / num_valid_samples
        total_loss_iou = total_loss_iou / num_valid_samples

    # ---- Backward + optimizer step ----
    scaler.scale(total_loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    scaler.step(optimizer)

    scaler.update()

    all_ious_cat = torch.cat(all_ious, dim=0)
    mean_iou = all_ious_cat.mean().item()

    return {
        'train/loss': float(total_loss.detach().item()),
        'train/loss_mask': float(total_loss_mask.detach().item()),
        'train/loss_dice': float(total_loss_dice.detach().item()),
        'train/loss_iou': float(total_loss_iou.detach().item()),
        'train/mean_iou': mean_iou,
    }


def validate_sam3(processor, model, valid_loader, device, config, epoch, vis_dir=None):
    """Validate SAM3 tracker model (PVS-style)."""
    print(f"\nValidating...")
    model.eval()

    all_ious = []
    all_losses = []
    all_loss_masks = []
    all_loss_dices = []
    all_loss_ious = []
    num_batches = 0

    visualize = getattr(config, 'visualize_validation', False)
    max_vis_samples = getattr(config, 'max_validation_vis_samples', 3)
    vis_count = 0

    if visualize and vis_dir is not None:
        val_vis_dir = vis_dir / f'validation_epoch_{epoch}'
        val_vis_dir.mkdir(parents=True, exist_ok=True)
    else:
        val_vis_dir = None

    weight_dict = {
        'loss_mask': getattr(config, 'loss_weight_mask', 0.0),
        'loss_dice': getattr(config, 'loss_weight_dice', 1.0),
        'loss_iou': getattr(config, 'loss_weight_iou', 1.0),
    }

    with torch.no_grad():
        pbar = tqdm(valid_loader, desc="Validation")

        for batch_idx, batch in enumerate(pbar, 1):
            if len(batch) == 0:
                continue

            # val loader uses batch size 1 → batch[0]
            data = batch[0]

            if "image" not in data:
                continue

            # C,H,W → H,W,C, uint8
            image = data["image"].permute(1, 2, 0).cpu().numpy()
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

            if "instances" not in data or len(data["instances"]) == 0:
                continue

            instances = data["instances"]
            boxes = instances.gt_boxes.tensor.cpu().numpy()   # (Ng, 4)
            masks = instances.gt_masks.tensor.cpu().numpy()   # (Ng, H, W)

            if len(boxes) == 0:
                continue

            H, W = image.shape[:2]

            # Clip boxes
            boxes[:, 0] = np.clip(boxes[:, 0], 0, W)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, H)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, W)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, H)

            # Remove degenerate boxes
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            if not valid.any():
                continue
            boxes = boxes[valid]
            masks = masks[valid]

            # Subsample prompts if needed
            max_prompts = getattr(config, 'max_prompts_per_image', 64)
            if len(boxes) > max_prompts:
                idx = np.random.choice(len(boxes), max_prompts, replace=False)
                boxes = boxes[idx]
                masks = masks[idx]

            if len(boxes) == 0:
                continue

            try:
                pil_image = Image.fromarray(image).convert("RGB")

                input_boxes = [boxes.tolist()]  # [batch=1, num_objects, 4]

                inputs = processor(
                    images=pil_image,
                    input_boxes=input_boxes,
                    return_tensors="pt",
                ).to(device)

                # One mask per object
                outputs = model(**inputs, multimask_output=False)

                pred_masks_all = outputs.pred_masks
                pred_ious_all = outputs.iou_scores  # (1, Nobj)
                # Possible shapes:
                # (1, Nobj, Hm, Wm) or (1, Nobj, 1, Hm, Wm)
                if pred_masks_all.ndim == 5:
                    pred_masks_all = pred_masks_all[:, :, 0]

                pred_masks_all = pred_masks_all.squeeze(0)  # (Nobj, Hm, Wm)
                pred_ious_all = pred_ious_all.squeeze(0)  # (Nobj,)
                pred_ious_all = pred_ious_all.flatten()
                num_pred = pred_masks_all.shape[0]

                num_matched = min(num_pred, len(boxes))
                if num_matched == 0:
                    continue

                # Align by index: ith GT box ↔ ith predicted mask
                pred_masks_logits = pred_masks_all[:num_matched]  # (Nm, Hm, Wm)
                pred_iou_scores = pred_ious_all[:num_matched]  # (Nm,)
                
                eps = 1e-6
                pred_iou_scores_clamped = pred_iou_scores.clamp(eps, 1 - eps)
                pred_iou_logits = torch.log(pred_iou_scores_clamped / (1 - pred_iou_scores_clamped))
            
                gt_mask = torch.tensor(
                    masks[:num_matched].astype(np.float32),
                    device=device,
                )  # (Nm, H_orig, W_orig)

                # Resize GT masks to pred resolution
                Hm, Wm = pred_masks_logits.shape[-2:]
                if gt_mask.shape[1:] != (Hm, Wm):
                    gt_mask = F.interpolate(
                        gt_mask.unsqueeze(1),
                        size=(Hm, Wm),
                        mode='nearest',
                    ).squeeze(1)

                # IoU & losses
                pred_mask_sigmoid = torch.sigmoid(pred_masks_logits)
                pred_mask_binary = (pred_mask_sigmoid > 0.5).float()
                inter = (gt_mask * pred_mask_binary).sum((1, 2))
                union = gt_mask.sum((1, 2)) + pred_mask_binary.sum((1, 2)) - inter + 1e-6
                actual_iou_scores = inter / union  # (Nm,)

                losses = sam_loss(
                    pred_masks=pred_masks_logits,
                    gt_masks=gt_mask,
                    pred_ious=pred_iou_logits,
                    num_objects=num_matched,
                    weight_dict=weight_dict,
                    focal_alpha=getattr(config, 'loss_focal_alpha', 0.25),
                    focal_gamma=getattr(config, 'loss_focal_gamma', 2.0),
                    iou_use_l1=getattr(config, 'loss_iou_use_l1', True),
                )

                all_losses.append(losses['loss'].detach().cpu().item())
                all_loss_masks.append(losses['loss_mask'].detach().cpu().item())
                all_loss_dices.append(losses['loss_dice'].detach().cpu().item())
                all_loss_ious.append(losses['loss_iou'].detach().cpu().item())
                all_ious.extend(actual_iou_scores.detach().cpu().numpy().tolist())

                num_batches += 1

                # Visualization (optional)
                if visualize and val_vis_dir is not None and vis_count < max_vis_samples:
                    pred_probs = pred_mask_sigmoid.detach().cpu().numpy()

                    visualize_predictions(
                        image=image,
                        gt_masks=gt_mask.cpu().numpy(),
                        pred_masks=pred_probs,
                        boxes=boxes[:num_matched],
                        save_path=val_vis_dir,
                        step=vis_count,
                        max_samples=20,
                    )
                    vis_count += 1

                pbar.set_postfix({
                    'batches': num_batches,
                    'mean_iou': f"{np.mean(all_ious):.4f}" if all_ious else "0.0000",
                })

            except Exception as e:
                print(f"\n⚠️  Error processing batch {batch_idx}: {e}")
                continue

    torch.cuda.empty_cache()

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
def load_checkpoint(checkpoint_path: Path, model, optimizer=None, scheduler=None):
    """
    Load a training checkpoint and restore model/optimizer/scheduler states.
    
    Args:
        checkpoint_path: Path to checkpoint file (.pt)
        model: The model to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        
    Returns:
        dict with 'epoch' and 'best_val_iou' if available, else empty dict
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle both full checkpoint and weights-only formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Full checkpoint (like model_latest.pt)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ Optimizer state restored")
            
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✓ Scheduler state restored")
        
        info = {
            'epoch': checkpoint.get('epoch', 0),
            'best_val_iou': checkpoint.get('best_val_iou', 0.0),
        }
        print(f"✓ Checkpoint loaded: epoch={info['epoch']}, best_val_iou={info['best_val_iou']:.4f}")
        return info
    else:
        # Weights-only checkpoint (like model_best_iou.pt)
        model.load_state_dict(checkpoint)
        print("✓ Model weights loaded (weights-only checkpoint)")
        return {}

def train_sam3(config: SegmenterConfig):
    """Train SAM3 """
    print(f"\n{'='*80}")
    print(f"SAM3 FINE-TUNING")
    print(f"{'='*80}\n")
    
    train_dataset_name, valid_dataset_name = setup_sam3_datasets(config)
    
    u = uuid.uuid4()
    now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    if slurm_job_id:
        model_name = f"{config.model}_{now}_{slurm_job_id}"
    else:
        model_name = f"{config.model}_{now}_{u.hex[:4]}"
    
    trainer_components = setup_sam3_trainer(train_dataset_name, valid_dataset_name, config, model_name)
    
    processor = trainer_components['processor']
    model = trainer_components['model']
    optimizer = trainer_components['optimizer']
    scheduler = trainer_components['scheduler']
    scaler = trainer_components['scaler']
    train_loader = trainer_components['train_loader']
    valid_loader = trainer_components['valid_loader']
    output_dir = trainer_components['output_dir']
    device = trainer_components['device']
    steps_per_epoch = trainer_components['steps_per_epoch']
    
    visualize = getattr(config, 'visualize_batches', False)
    vis_dir = None
    if visualize:
        vis_dir = Path(getattr(config, 'visualize_path', output_dir / 'visualizations'))
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Wandb
    freeze_layers = []
    if getattr(config, 'freeze_image_encoder', False):
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
            name=f"{model_name}_seed{config.seed}",
            config=vars(config),
        )
    
    best_val_iou = 0.0
    best_val_ap = 0.0 
    global_step = 0
    start_epoch = 0
    resume_path = getattr(config, 'resume_checkpoint_path', None)
    if resume_path:
        resume_info = load_checkpoint(
            Path(resume_path),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = resume_info.get('epoch', 0)
        best_val_iou = resume_info.get('best_val_iou', 0.0)
        # Adjust global_step based on resumed epoch
        global_step = start_epoch * steps_per_epoch
        print(f"Resuming training from epoch {start_epoch + 1}")
    
    print("Starting training...")
    
    for epoch in range(start_epoch, config.max_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.max_epochs}")
        print(f"{'='*60}")
        
        data_iter = iter(train_loader)
        model.train()
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}")
        
        for _ in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            
            metrics = train_step(processor, model, batch, optimizer, scaler, device, config, global_step=global_step, vis_dir=vis_dir)
            
            if metrics is None:
                continue
            
            global_step += 1
            scheduler.step()
            
            pbar.set_postfix({
                'loss': f"{metrics['train/loss']:.4f}",
                'iou': f"{metrics['train/mean_iou']:.4f}",
            })
            
            if wandb.run:
                wandb.log({
                    **metrics,
                    "lr/group_0": optimizer.param_groups[0]["lr"],
                    "lr/group_1": optimizer.param_groups[1]["lr"],
                }, step=global_step)
            
            del batch
        
        del data_iter
        gc.collect()
        torch.cuda.empty_cache()
        
        val_metrics = validate_sam3(
            processor=processor,
            model=model,
            valid_loader=valid_loader,
            device=device,
            config=config,
            epoch=epoch + 1,
            vis_dir=vis_dir
        )

        latest_path = output_dir / "model_latest.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_iou': best_val_iou,
        }, latest_path)

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
        
        current_ap = val_metrics.get('coco/mean/AP', 0.0)
        current_iou = val_metrics['val/mean_iou']
        # Save best model
        if current_ap > best_val_ap:
            best_val_ap = current_ap
            best_path = output_dir / "model_best.pt"
            torch.save(model.state_dict(), best_path)
            print(f"✓ New best model saved: AP={best_val_ap:.4f}, IoU={best_val_iou:.4f}")

        if current_iou > best_val_iou:
            best_val_iou = current_iou
            best_path = output_dir / "model_best_iou.pt"
            torch.save(model.state_dict(), best_path)
            print(f"✓ New best model (IoU) saved: IoU={best_val_iou:.4f}")
        
        
        gc.collect()
    
    final_path = output_dir / "model_final.pt"
    torch.save(model.state_dict(), final_path)
    
    if wandb.run:
        wandb.finish()
    
    print(f"Training completed. Model saved in {output_dir}")