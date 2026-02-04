import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from geodataset.dataset import InstanceSegmentationLabeledRasterCocoDataset
from geodataset.utils import decode_coco_segmentation
import albumentations as A
import rasterio
import uuid


class Sam2InstanceSegmentationDataset(InstanceSegmentationLabeledRasterCocoDataset):
    """
    SAM2 training dataset based on InstanceSegmentationLabeledRasterCocoDataset.
    
    Inherits all the tile loading, COCO parsing, and augmentation logic.
    Returns data in SAM2-compatible format.
    
    Parameters
    ----------
    fold: str
        The dataset fold to load (e.g., 'train', 'valid', 'test')
    root_path: str or List[str] or Path or List[Path]
        The root directory of the dataset
    transform: albumentations.core.composition.Compose
        Augmentation pipeline (crop, rotate, flip, etc.)
    force_binary_class: bool
        Force all classes to be binary (1)
    max_prompts: int
        Maximum number of box prompts per image (subsample if more)
    """
    
    def __init__(
        self,
        fold: str,
        root_path: str or List[str] or Path or List[Path],
        transform: A.core.composition.Compose = None,
        force_binary_class: bool = True,
        max_prompts: int = 64,
    ):
        super().__init__(
            fold=fold,
            root_path=root_path,
            transform=transform,
            force_binary_class=force_binary_class,
        )
        self.max_prompts = max_prompts
        
        print(f"\n{'='*80}")
        print(f"SAM2 Instance Segmentation Dataset")
        print(f"{'='*80}")
        print(f"Fold: {fold}")
        print(f"Root path: {root_path}")
        print(f"Num images: {len(self.tiles)}")
        print(f"Max prompts per image: {max_prompts}")
        print(f"Augmentation: {'Enabled' if transform else 'Disabled'}")
        print(f"{'='*80}\n")
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a single image with masks and boxes in SAM2 format.
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'image': numpy.ndarray (H, W, 3) - RGB image [0-255]
            - 'masks': numpy.ndarray (N, H, W) - binary masks
            - 'boxes': numpy.ndarray (N, 4) - boxes in XYXY format
            - 'labels': numpy.ndarray (N,) - class labels
            - 'image_id': int - image index
        """
        # Get tile info
        tile_info = self.tiles[idx]
        
        # Load image
        with rasterio.open(tile_info['path']) as tile_file:
            tile = tile_file.read([1, 2, 3])  # RGB bands
        
        # Get annotations
        labels = tile_info['labels']
        masks = []
        bboxes = []
        
        for label in labels:
            bbox = decode_coco_segmentation(label, 'bbox')
            mask = decode_coco_segmentation(label, 'mask')
            
            bboxes.append(np.array([int(x) for x in bbox.bounds]))  # XYXY format
            masks.append(mask)
        
        # Category IDs
        if self.force_binary_class:
            category_ids = np.ones(len(labels), dtype=np.int32)
        else:
            category_ids = np.array([
                0 if label['category_id'] is None else label['category_id']
                for label in labels
            ], dtype=np.int32)
        
        # Apply augmentations
        if self.transform:
            transformed = self.transform(
                image=tile.transpose((1, 2, 0)),  # (H, W, 3)
                mask=np.stack(masks, axis=2),      # (H, W, N)
                bboxes=bboxes,
                labels=category_ids
            )
            image = transformed['image']  # (H, W, 3)
            masks = transformed['mask'].transpose((2, 0, 1))  # (N, H, W)
            bboxes = np.array(transformed['bboxes'])
            category_ids = np.array(transformed['labels'])
        else:
            image = tile.transpose((1, 2, 0))  # (C, H, W) -> (H, W, C)
            masks = np.array(masks)
        
        # Subsample if too many instances
        num_instances = len(bboxes)
        if num_instances > self.max_prompts:
            idx_sample = np.random.choice(num_instances, self.max_prompts, replace=False)
            bboxes = bboxes[idx_sample]
            masks = masks[idx_sample]
            category_ids = category_ids[idx_sample]
        
        # Return in simple format (no normalization - SAM2 does it internally)
        return {
            'image': image.astype(np.uint8),  # (H, W, 3) RGB [0-255]
            'masks': masks.astype(np.float32),  # (N, H, W)
            'boxes': bboxes.astype(np.float32),  # (N, 4) XYXY
            'labels': category_ids,  # (N,)
            'image_id': idx,
        }
# Add these imports at the top
import json
import hashlib
from pathlib import Path

def get_detector_predictions_for_dataset(
    tiles_paths: list[str],
    detector_config_path: str,
    cache_dir: str,
) -> dict[str, list]:
    """
    Run detector pipeline on tiles and return predicted boxes.
    
    Parameters
    ----------
    tiles_paths : list[str]
        List of tile paths to run detection on
    detector_config_path : str
        Path to detector pipeline config (e.g., 'default_detection_multi_NQOS_best')
    cache_dir : str
        Directory to cache detector outputs
    force_recompute : bool
        Force re-running detector even if cache exists
    
    Returns
    -------
    dict[str, list]
        Mapping from tile path to list of predicted boxes [x1, y1, x2, y2]
    """
    from canopyrs.engine.config_parsers import InferIOConfig, PipelineConfig
    from canopyrs.engine.config_parsers.base import get_config_path
    from canopyrs.engine.pipeline import Pipeline

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Check cache
    
    print(f"Running detector pipeline: {detector_config_path}")
    print(f"On {len(tiles_paths)} tiles...")
    
    # Get unique tile directories
    tile_dirs = list(set(str(Path(p).parent) for p in tiles_paths))
    
    # Run detector for each tile directory
    all_predictions = {}
    
    for tile_dir in tile_dirs:        
        # Load pipeline config
        config_path = get_config_path(detector_config_path)
        config = PipelineConfig.from_yaml(config_path)
        score_threshold = getattr(config, 'score_threshold',  0.5)
        # Remove tilerizer if present (we have tiles already)
        if config.components_configs[0][0] == 'tilerizer':
            config.components_configs.pop(0)
        
        # Create IO config
        io_config = InferIOConfig(
            input_imagery=tile_dir, 
            tiles_path=tile_dir,
            output_folder=str(cache_dir),
        )
        print(f"Processing tile directory: {tile_dir}")
        # Run pipeline
        pipeline = Pipeline(io_config, config)
        output = pipeline()

        coco_path = None
        if hasattr(output, 'infer_coco_path') and output.infer_coco_path:
            coco_path = output.infer_coco_path
        elif hasattr(output, 'component_output_files'):
            # Try aggregator first, then detector
            if '1_aggregator' in output.component_output_files:
                coco_path = output.component_output_files['1_aggregator'].get('coco')
            elif '0_detector' in output.component_output_files:
                coco_path = output.component_output_files['0_detector'].get('coco')

        if not coco_path or not Path(coco_path).exists():
            print(f"Warning: No COCO output found from pipeline")
            continue
        
        # Parse COCO annotations to get boxes
        with open(coco_path, 'r') as f:
            coco_data = json.load(f)
        #Get detector output
        detector_coco_path = output.component_output_files['0_detector'].get('coco')
        # Build image_id -> filename mapping
        id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        
        # Group annotations by image
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            filename = id_to_filename[img_id]
            
            # Convert to absolute path
            tile_path = str(Path(tile_dir) / filename)
            
            if tile_path not in all_predictions:
                all_predictions[tile_path] = []
            
            # COCO bbox is [x, y, width, height] -> convert to [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            box = [x, y, x + w, y + h]
            score = ann.get('score', 1.0)
            if score >= score_threshold:
                all_predictions[tile_path].append({
                    'bbox': box,
                    'score': score,
                })
    return all_predictions


def get_dataset_dicts_with_predicted_boxes(
    dataset_instance,
    predicted_boxes: dict[str, list]
):
    """
    Convert dataset to Detectron2 format using PREDICTED boxes but GT masks.
    
    For each predicted box, find the best matching GT mask based on IoU.
    
    Parameters
    ----------
    dataset_instance : InstanceSegmentationLabeledRasterCocoDataset
        Dataset instance with GT masks
    predicted_boxes : dict[str, list]
        Mapping from tile path to list of predicted boxes
    
    Returns
    -------
    list
        List of dictionaries in Detectron2 format
    """
    from shapely.geometry import box as shapely_box
    
    dataset_dicts = []
    
    for idx in range(len(dataset_instance.tiles)):
        tile_info = dataset_instance.tiles[idx]
        tile_path = str(tile_info['path'])
        
        # Create record
        record = {
            "file_name": tile_path,
            "image_id": idx,
            "height": tile_info['height'],
            "width": tile_info['width'],
        }
        
        # Get GT labels (for masks)
        gt_labels = tile_info['labels']
        
        # Get predicted boxes for this tile
        pred_boxes = predicted_boxes.get(tile_path, [])
        
        if len(pred_boxes) == 0 or len(gt_labels) == 0:
            record["annotations"] = []
            dataset_dicts.append(record)
            continue
        
        # Build GT boxes and masks for matching
        gt_boxes = []
        gt_segmentations = []
        
        for label in gt_labels:
            bbox = decode_coco_segmentation(label, 'bbox')
            gt_boxes.append(list(bbox.bounds))  # [x1, y1, x2, y2]
            
            if 'segmentation' in label:
                gt_segmentations.append(label['segmentation'])
            else:
                gt_segmentations.append(None)
        
        gt_boxes = np.array(gt_boxes)
        
        # Match predicted boxes to GT boxes using IoU
        objs = []
        used_gt_indices = set()
        
        for pred in pred_boxes:
            pred_box = pred['bbox']
            
            # Compute IoU with all GT boxes
            pred_shapely = shapely_box(*pred_box)
            
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in used_gt_indices:
                    continue
                if gt_segmentations[gt_idx] is None:
                    continue
                
                gt_shapely = shapely_box(*gt_box)
                
                inter = pred_shapely.intersection(gt_shapely).area
                union = pred_shapely.union(gt_shapely).area
                iou = inter / (union + 1e-6)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Only use if IoU > threshold (0.3)
            if best_iou > 0.3 and best_gt_idx >= 0:
                used_gt_indices.add(best_gt_idx)
                
                obj = {
                    "bbox": pred_box,  # Use PREDICTED box
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": 0,
                    "segmentation": gt_segmentations[best_gt_idx],  # Use GT mask
                }
                objs.append(obj)
        
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts



def register_sam2_dataset_with_predicted_boxes(
    fold: str,
    root_path: str or list[str],
    detector_config_path: str,
    cache_dir: str,
    force_binary_class: bool = True,
) -> str:
    """
    Register dataset with PREDICTED boxes (from detector) but GT masks.
    
    Parameters
    ----------
    fold : str
        Dataset fold ('train', 'valid', 'test')
    root_path : str or list[str]
        Root path(s) to dataset
    detector_config_path : str
        Path to detector pipeline config
    cache_dir : str
        Directory to cache detector outputs
    score_threshold : float
        Minimum score to keep a predicted box
    force_binary_class : bool
        Force binary classification
    
    Returns
    -------
    str
        Registered dataset name
    """
    # First create the base dataset to get tile paths
    dataset = Sam2InstanceSegmentationDataset(
        fold=fold,
        root_path=root_path,
        force_binary_class=force_binary_class
    )
    print("IN REGISTER, FOLD IS:", fold)
    # Get all tile paths
    tiles_paths = [str(dataset.tiles[idx]['path']) for idx in range(len(dataset.tiles))]
    
    # Run detector to get predicted boxes
    predicted_boxes = get_detector_predictions_for_dataset(
        tiles_paths=tiles_paths,
        detector_config_path=detector_config_path,
        cache_dir=cache_dir,
    )
    
    dataset_name = f"sam2_{fold}_predboxes_{uuid.uuid4().hex[:8]}"
    
    print(f"Registering dataset '{dataset_name}' with predicted boxes...")
    
    # Register with predicted box conversion function
    DatasetCatalog.register(
        dataset_name,
        lambda d=dataset, pb=predicted_boxes: get_dataset_dicts_with_predicted_boxes(d, pb)
    )
    
    
    # Register metadata
    MetadataCatalog.get(dataset_name).set(
        thing_classes=["tree"],
        evaluator_type="coco"
    )
    
    print(f"Dataset '{dataset_name}' registered with {len(dataset.tiles)} images using predicted boxes.\n")
    
    return dataset_name

def get_dataset_dicts_with_masks(dataset_instance: InstanceSegmentationLabeledRasterCocoDataset):
    """
    Convert dataset to Detectron2 format WITH segmentation masks.
    
    Parameters
    ----------
    dataset_instance : InstanceSegmentationLabeledRasterCocoDataset
        Dataset instance
    
    Returns
    -------
    list
        List of dictionaries in Detectron2 format with segmentation masks
    """
    dataset_dicts = []
    
    for idx in range(len(dataset_instance.tiles)):
        tile_info = dataset_instance.tiles[idx]
        
        # Create record
        record = {
            "file_name": str(tile_info['path']),
            "image_id": idx,
            "height": tile_info['height'],
            "width": tile_info['width'],
        }
        
        # Convert annotations
        labels = tile_info['labels']
        objs = []
        
        for label in labels:
            bbox = decode_coco_segmentation(label, 'bbox')
            
            # ✅ Get segmentation mask
            if 'segmentation' not in label:
                continue
            
            segmentation = label['segmentation']
            
            # Handle different segmentation formats
            if isinstance(segmentation, dict):
                # RLE format - keep as is
                pass
            elif isinstance(segmentation, list):
                # Polygon format - Detectron2 can handle this
                pass
            else:
                continue
            
       
            category_id = 0
            #if category_id is None:
            #    category_id = 0
            
            obj = {
                "bbox": list(bbox.bounds),  # [x1, y1, x2, y2]
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": int(category_id),
                "segmentation": segmentation,  # ✅ Include segmentation!
            }
            objs.append(obj)
        
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts


def register_sam2_dataset_with_masks(
    fold: str,
    root_path: str or list[str],
    force_binary_class: bool = True
) -> str:
    """
    Register dataset with segmentation masks for SAM2 training.
    
    Parameters
    ----------
    fold : str
        Dataset fold ('train', 'valid', 'test')
    root_path : str or list[str]
        Root path(s) to dataset
    force_binary_class : bool
        Force binary classification
    
    Returns
    -------
    str
        Registered dataset name
    """
    dataset = Sam2InstanceSegmentationDataset(
        fold=fold,
        root_path=root_path,
        force_binary_class=force_binary_class
    )
    
    dataset_name = f"sam2_{fold}_tree_masks_{uuid.uuid4().hex[:8]}"
    
    print(f"Registering dataset '{dataset_name}' with segmentation masks...")
    
    # ✅ Register with mask conversion function
    DatasetCatalog.register(
        dataset_name,
        lambda: get_dataset_dicts_with_masks(dataset)
    )
    
    # Register metadata
    MetadataCatalog.get(dataset_name).set(
        thing_classes=["tree"],
        evaluator_type="coco"
    )
    
    print(f"✅ Dataset '{dataset_name}' registered with {len(dataset.tiles)} images.\n")
    
    return dataset_name

def register_sam2_dataset(
    fold: str,
    root_path: str or List[str],
    transform: A.core.composition.Compose = None,
    force_binary_class: bool = True,
    max_prompts: int = 64,
) -> str:
    """
    Register a SAM2 dataset.
    
    Parameters
    ----------
    fold: str
        Dataset fold ('train', 'val', 'test')
    root_path: str or List[str]
        Root path(s) to dataset
    transform: albumentations.Compose
        Augmentation pipeline
    force_binary_class: bool
        Force binary classification
    max_prompts: int
        Max prompts per image
    
    Returns
    -------
    str
        Registered dataset name
    """
    import uuid
    
    dataset = Sam2InstanceSegmentationDataset(
        fold=fold,
        root_path=root_path,
        transform=transform,
        force_binary_class=force_binary_class,
        max_prompts=max_prompts,
    )
    
    dataset_name = f"sam2_{fold}_tree_{uuid.uuid4().hex[:8]}"
    
    print(f"Registering dataset '{dataset_name}'...")
    
    # Register (just return the dataset directly - no conversion needed)
    DatasetCatalog.register(dataset_name, lambda: dataset)
    
    MetadataCatalog.get(dataset_name).set(
        thing_classes=["tree"],
        evaluator_type="coco"
    )
    
    print(f"✅ Dataset '{dataset_name}' registered.\n")
    
    return dataset_name