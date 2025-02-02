from pathlib import Path
from typing import List

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from geodataset.dataset import DetectionLabeledRasterCocoDataset
from geodataset.utils import decode_coco_segmentation


class Detectron2DetectionLabeledRasterCocoDataset(DetectionLabeledRasterCocoDataset):
    def __getitem__(self, idx: int):
        tile_info = self.tiles[idx]

        labels = tile_info['labels']
        bboxes = []

        for label in labels:
            bbox = decode_coco_segmentation(label, 'bbox')
            bboxes.append(np.array([int(x) for x in bbox.bounds]))

        if self.force_binary_class:
            category_ids = np.array([1 for _ in labels])
        else:
            category_ids = np.array([0 if label['category_id'] is None else label['category_id']
                                     for label in labels])

        area = np.array([(bboxe[3] - bboxe[1]) * (bboxe[2] - bboxe[0]) for bboxe in bboxes])
        # suppose all instances are not crowd
        iscrowd = np.zeros((len(bboxes),))
        # get tile id
        image_id = np.array([idx])
        # group annotations info
        annotations = {'boxes': bboxes, 'labels': category_ids, 'area': area, 'iscrowd': iscrowd, 'image_id': image_id}

        return tile_info, annotations


def get_dataset_dicts(dataset_instance: DetectionLabeledRasterCocoDataset):
    """
    Convert the custom dataset format to detectron2's dictionary format.

    Parameters
    ----------
    dataset_instance : DetectionLabeledRasterCocoDataset
        An instance of the custom dataset class

    Returns
    -------
    list
        List of dictionaries in detectron2 format
    """
    dataset_dicts = []

    for idx in range(len(dataset_instance)):
        # Get image and annotations
        tile_info, annotations = dataset_instance[idx]

        # Create record dictionary
        record = {}

        record["file_name"] = str(tile_info['path'])
        record["image_id"] = idx
        record["height"] = tile_info['height']
        record["width"] = tile_info['width']

        # Convert annotations
        objs = []
        for bbox, category_id, area, iscrowd in zip(
                annotations['boxes'],
                annotations['labels'],
                annotations['area'],
                annotations['iscrowd']
        ):
            obj = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": int(category_id) - 1,        # 0-indexed in detectron2, but 1-indexed in COCO, so substracting 1
                "iscrowd": int(iscrowd),
                "area": float(area)
            }
            objs.append(obj)
        record["annotations"] = objs

        dataset_dicts.append(record)

    return dataset_dicts


def register_detection_dataset(
        fold: str,
        root_path: str,
        dataset_name: str,
        force_binary_class: bool = True
):
    """
    Register a custom dataset with detectron2.

    Parameters
    ----------
    fold: str
        The fold of the dataset to register
    root_path: str
        The root path of the dataset
    dataset_name: str
        The name of the dataset
    force_binary_class: bool, optional
        Whether to force binary class

    """

    dataset = Detectron2DetectionLabeledRasterCocoDataset(
        fold=fold,
        root_path=root_path,
        transform=None,
        force_binary_class=force_binary_class
    )

    # Define your classes
    thing_classes = ["tree"]

    print(f"Registering as dataset '{dataset_name}'...")
    # Register the dataset
    DatasetCatalog.register(
        dataset_name,
        lambda: get_dataset_dicts(dataset)
    )

    # Register metadata
    MetadataCatalog.get(dataset_name).set(
        thing_classes=thing_classes,
        evaluator_type="coco"
    )
    print(f"Dataset '{dataset_name}' registered.")


def register_combined_datasets(dataset_names: list):
    """
    Combine multiple datasets into one.

    Parameters
    ----------
    dataset_names: list
        List of dataset names to combine

    Returns
    -------
    str
        The name of the combined dataset
    """
    combined_dataset_name = "_".join(dataset_names)
    combined_dataset_dicts = []

    for dataset_name in dataset_names:
        combined_dataset_dicts.extend(DatasetCatalog.get(dataset_name))

    DatasetCatalog.register(
        combined_dataset_name,
        lambda: combined_dataset_dicts
    )

    MetadataCatalog.get(combined_dataset_name).set(
        thing_classes=MetadataCatalog.get(dataset_names[0]).thing_classes
    )

    return combined_dataset_name


def register_multiple_detection_datasets(
        root_path: str,
        dataset_names: list,
        fold: str,
        force_binary_class: bool = True,
        combine_datasets: bool = True
) -> List[str]:
    """
    Register multiple custom datasets with detectron2.

    Parameters
    ----------
    root_path: str
        The root path of the datasets
    dataset_names: list
        List of dataset names
    fold: str
        The fold of the dataset to register (train, valid, train0, etc.)
    force_binary_class: bool, optional
        Whether to force binary class
    """
    d2_datasets_names = []
    for dataset_name in dataset_names:
        d2_name = f"{dataset_name}_{fold}"
        register_detection_dataset(
            fold=fold,
            root_path=Path(root_path) / dataset_name,
            dataset_name=d2_name,
            force_binary_class=force_binary_class
        )
        d2_datasets_names.append(d2_name)

    if combine_datasets and len(d2_datasets_names) > 1:
        d2_combined_dataset_name = register_combined_datasets(d2_datasets_names)
        return [d2_combined_dataset_name]
    else:
        return d2_datasets_names
