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