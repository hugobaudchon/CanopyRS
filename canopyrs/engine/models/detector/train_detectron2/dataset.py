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

        for label in labels:
            # Get bounding box directly from COCO bbox field
            bbox = label['bbox']
            bboxes.append(np.array([int(x) for x in bbox]))

            # Just pass through the segmentation if present
            if 'segmentation' in label and label['segmentation']:
                segmentations.append(label['segmentation'])
            else:
                # Keep explicit None so we can see where we don't have segs
                segmentations.append(None)

        if self.force_binary_class:
            category_ids = np.array([1 for _ in labels])
        else:
            category_ids = np.array([
                0 if label['category_id'] is None else label['category_id']
                for label in labels
            ])

        area = np.array([(bboxe[3] - bboxe[1]) * (bboxe[2] - bboxe[0]) for bboxe in bboxes])
        iscrowd = np.zeros((len(bboxes),))
        image_id = np.array([idx])

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


def get_dataset_dicts(dataset_instance: DetectionLabeledRasterCocoDataset):
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
        dataset_dicts.append(record)

    print(f"Conversion complete: {len(dataset_dicts)} samples ready.")
    return dataset_dicts


def register_detection_dataset(
    fold: str,
    root_path: str | list[str],
    force_binary_class: bool = True,
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
        lambda: get_dataset_dicts(dataset)
    )

    MetadataCatalog.get(dataset_name).set(
        thing_classes=thing_classes,
        evaluator_type="coco"
    )
    print(f"Dataset '{dataset_name}' registered.")

    return dataset_name
