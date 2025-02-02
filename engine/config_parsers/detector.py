from typing import List, Optional

from engine.config_parsers.base import BaseConfig


class DetectorConfig(BaseConfig):
    # General model definition
    model: str = 'faster_rcnn_detectron2'
    architecture: str = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
    checkpoint_path: Optional[str] = None
    batch_size: int = 8
    num_classes: int = 1
    box_predictions_per_image: int = 250
    anchor_sizes: tuple = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios: tuple = ((0.5, 1.0, 2.0),) * 5
    box_score_thresh: float = 0.05
    box_nms_thresh: float = 0.5

    # Training Data and Output path
    data_root_path: str = None   # Parent folder of datasets
    train_dataset_names: List[str] = []     # Sub-folders names in root_path (parent folder)
    train_output_path: str = None

    # Training Params
    main_metric: str = 'mAP'            # TODO add support for this
    lr: float = 1e-4
    max_epochs: int = 100
    freeze_layers: int = -1
    train_log_interval: int = 10
    eval_epoch_interval: int = 1
    grad_accumulation_steps: int = 1
    backbone_model_pretrained: bool = True
    scheduler_epochs_steps: List[int] = [10, 20, 30]
    scheduler_gamma: float = 0.9
    scheduler_warmup_steps: int = 1000
    dataloader_num_workers: int = 4
