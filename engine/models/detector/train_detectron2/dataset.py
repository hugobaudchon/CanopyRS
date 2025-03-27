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
    Convert the custom dataset format to train_detectron2's dictionary format.

    Parameters
    ----------
    dataset_instance : DetectionLabeledRasterCocoDataset
        An instance of the custom dataset class

    Returns
    -------
    list
        List of dictionaries in train_detectron2 format
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
                "category_id": int(category_id) - 1,        # 0-indexed in train_detectron2, but 1-indexed in COCO, so substracting 1
                "iscrowd": int(iscrowd),
                "area": float(area)
            }
            objs.append(obj)
        record["annotations"] = objs

        dataset_dicts.append(record)

    return dataset_dicts


def register_detection_dataset(
        fold: str,
        root_path: str or list[str],
        dataset_name: str,
        force_binary_class: bool = True
):
    """
    Register a custom dataset with train_detectron2.

    Parameters
    ----------
    fold: str
        The fold of the dataset to register
    root_path: str or list[str]
        The list of root paths of the dataset
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

    return dataset_name
