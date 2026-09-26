import uuid
import pprint

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from geodataset.dataset import DetectionLabeledRasterCocoDataset
from tqdm import tqdm


class Detectron2DetectionLabeledRasterCocoDataset(DetectionLabeledRasterCocoDataset):
    def __getitem__(self, idx: int):
        tile_info = self.tiles[idx]

        labels = tile_info['labels']
        bboxes = []
        segmentations = []
        area = []
        iscrowd = []
        image_id = []

        for label in labels:
            # Get bounding box directly from COCO bbox field
            x, y, w, h = label["bbox"]
            bboxes.append(np.array([x, y, x + w, y + h], dtype=float))

            # Just pass through the segmentation if present
            if 'segmentation' in label and label['segmentation']:
                segmentations.append(label['segmentation'])
            else:
                # Keep explicit None so we can see where we don't have segs
                segmentations.append(None)

            area.append(label['area'])
            iscrowd.append(label['iscrowd'])
            image_id.append(idx)

        if self.force_binary_class:
            category_ids = np.array([1 for _ in labels])
        else:
            category_ids = np.array([
                0 if label['category_id'] is None else label['category_id']
                for label in labels
            ])

        annotations = {
            'boxes': bboxes,
            'labels': category_ids,
            'segmentations': segmentations,
            'area': area,
            'iscrowd': iscrowd,
            'image_id': image_id
        }

        return tile_info, annotations


def process_single_sample(idx, dataset_instance):
    """Process a single sample (used in the sequential loop)."""
    tile_info, annotations = dataset_instance[idx]

    record = {
        "file_name": str(tile_info['path']),
        "image_id": idx,
        "height": tile_info['height'],
        "width": tile_info['width']
    }

    objs = []
    for obj_idx, (bbox, category_id, segmentation, area, iscrowd) in enumerate(zip(
        annotations['boxes'],
        annotations['labels'],
        annotations['segmentations'],
        annotations['area'],
        annotations['iscrowd']
    )):
        # Ensure bbox is a plain Python list
        if isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()
        else:
            bbox = [float(x) for x in bbox]

        obj = {
            "bbox": bbox,
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": int(category_id) - 1,
            "iscrowd": int(iscrowd),
            "area": float(area),
        }

        # Add segmentation if valid; otherwise set an empty list
        # so that every object has a "segmentation" key and
        # utils.annotations_to_instances() won't KeyError.
        if segmentation is not None and isinstance(segmentation, list) and len(segmentation) > 0:
            clean_seg = []
            for poly in segmentation:
                if isinstance(poly, (list, np.ndarray)):
                    clean_poly = [float(x) for x in poly]
                    if len(clean_poly) >= 6:
                        clean_seg.append(clean_poly)

            if clean_seg:
                obj["segmentation"] = clean_seg
            else:
                obj["segmentation"] = []
        else:
            obj["segmentation"] = []

        objs.append(obj)

    record["annotations"] = objs
    return record


def get_dataset_dicts(dataset_instance: DetectionLabeledRasterCocoDataset, segmentation_only: bool = False):
    """
    Convert the custom dataset format to detectron2's dictionary format.
    Single-threaded version.
    """
    dataset_size = len(dataset_instance)
    print(f"Converting {dataset_size} samples to Detectron2 format...")

    dataset_dicts = []
    for idx in tqdm(range(dataset_size), desc="Processing dataset"):
        try:
            record = process_single_sample(idx, dataset_instance)
        except Exception as e:
            print(f"[ERROR] Failed to process sample idx={idx}")
            tile_info, annotations = dataset_instance[idx]
            print("[ERROR] tile_info:")
            pprint.pprint(tile_info)
            print("[ERROR] annotations (truncated):")
            print("   n_boxes:", len(annotations.get("boxes", [])))
            print("   n_segmentations:", len(annotations.get("segmentations", [])))
            raise e

        if segmentation_only: 
            # Skip this record if ANY annotation has an empty or missing segmentation
             annos = record.get("annotations", [])
             has_empty_seg = any(
                 ("segmentation" not in obj) 
                 or (not obj["segmentation"]) #empty list or falsy 
                 for obj in annos
             )
             if has_empty_seg: 
                continue

        dataset_dicts.append(record)

    print(f"Conversion complete: {len(dataset_dicts)} samples ready.")
    return dataset_dicts



def register_detection_dataset(
    fold: str,
    root_path: str | list[str],
    force_binary_class: bool = True,
    segmentation_only: bool = False
):
    """
    Register a custom dataset with detectron2 (no multiprocessing).
    """
    dataset = Detectron2DetectionLabeledRasterCocoDataset(
        fold=fold,
        root_path=root_path,
        transform=None,
        force_binary_class=force_binary_class
    )

    thing_classes = ["tree"]
    dataset_name = f"{fold}_tree_custom_{uuid.uuid4().hex}"
    print(f"Registering as dataset '{dataset_name}'...")

    DatasetCatalog.register(
        dataset_name,
        lambda: get_dataset_dicts(dataset, segmentation_only)
    )

    MetadataCatalog.get(dataset_name).set(
        thing_classes=thing_classes,
        evaluator_type="coco"
    )
    print(f"Dataset '{dataset_name}' registered.")

    return dataset_name
