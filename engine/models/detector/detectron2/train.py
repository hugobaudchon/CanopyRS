from collections.abc import Mapping
from datetime import datetime
from pprint import pprint
from typing import List

from detectron2.data import DatasetCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultTrainer, default_argument_parser
from detectron2.config import get_cfg, LazyConfig
from detectron2.model_zoo import model_zoo
import os

from engine.config_parsers import DetectorConfig
from engine.models.detector.detectron2.dataset import register_multiple_detection_datasets


class TrainerWithValidation(DefaultTrainer):
    """Custom trainer class that includes periodic validation."""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator = COCOEvaluator(
            dataset_name,
            output_dir=output_folder
        )
        evaluator._max_dets_per_image = [10, 100, cfg.TEST.DETECTIONS_PER_IMAGE]
        return evaluator


def setup_trainer(train_dataset_names: List[str], valid_dataset_names: List[str], config: DetectorConfig, model_name: str):
    """
    Set up a basic Faster R-CNN trainer with default configurations.

    Parameters
    ----------
    train_dataset_names : List[str]
        Name(s) of the registered dataset(s) to use for training
    valid_dataset_names : List[str]
        Name(s) of the registered dataset(s) to use for validation
    config : DetectorConfig
        Configuration object for the detector model
    model_name : str

    Returns
    -------
    DefaultTrainer
        Configured detectron2 trainer
    """
    cfg = get_cfg()

    # Load base configs for Faster R-CNN
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    # Load pre-trained model weights
    if config.backbone_model_pretrained:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    if config.checkpoint_path is not None:
        cfg.MODEL.WEIGHTS = config.checkpoint_path

    dataset_length = sum([len(DatasetCatalog.get(dataset_name)) for dataset_name in train_dataset_names])

    # Dataset config
    cfg.DATASETS.TRAIN = tuple(train_dataset_names)
    cfg.DATASETS.TEST = tuple(valid_dataset_names)

    # Training config
    cfg.DATALOADER.NUM_WORKERS = config.dataloader_num_workers
    cfg.SOLVER.IMS_PER_BATCH = config.batch_size  # This is the real "batch size"
    cfg.SOLVER.BASE_LR = config.lr
    cfg.SOLVER.MAX_ITER = config.max_epochs * dataset_length // config.batch_size
    cfg.SOLVER.LOG_PERIOD = config.train_log_interval
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.num_classes

    if config.scheduler_epochs_steps is not None:
        steps = [step * dataset_length // config.batch_size for step in config.scheduler_epochs_steps]
        print(f"Changing scheduler steps from {cfg.SOLVER.STEPS} to {steps}.")
        cfg.SOLVER.STEPS = steps
    if config.scheduler_gamma is not None:
        print(f"Changing scheduler gamma from {cfg.SOLVER.GAMMA} to {config.scheduler_gamma}.")
        cfg.SOLVER.GAMMA = config.scheduler_gamma
    if config.scheduler_warmup_steps is not None:
        print(f"Changing scheduler warmup steps from {cfg.SOLVER.WARMUP_ITERS} to {config.scheduler_warmup_steps}.")
        cfg.SOLVER.WARMUP_ITERS = config.scheduler_warmup_steps
    if config.freeze_layers:
        print(f"Changing freeze layers from {cfg.MODEL.BACKBONE.FREEZE_AT} to {config.freeze_layers}.")
        cfg.MODEL.BACKBONE.FREEZE_AT = config.freeze_layers
    if config.box_score_thresh is not None:
        print(f"Changing box score threshold from {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST} to {config.box_score_thresh}.")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.box_score_thresh
    if config.box_nms_thresh is not None:
        print(f"Changing box NMS threshold from {cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST} to {config.box_nms_thresh}.")
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config.box_nms_thresh

    # Evaluation config
    cfg.TEST.EVAL_PERIOD = config.eval_epoch_interval * dataset_length // config.batch_size
    cfg.TEST.DETECTIONS_PER_IMAGE = config.box_predictions_per_image

    # Output directory
    output_dir = os.path.join(config.train_output_path, model_name)
    os.makedirs(output_dir, exist_ok=True)
    cfg.OUTPUT_DIR = output_dir

    # Save config
    config.to_yaml(os.path.join(output_dir, "config.yaml"))

    # Create trainer
    trainer = TrainerWithValidation(cfg)
    trainer.resume_or_load(resume=False)

    return trainer


def train_detectron2_fasterrcnn(config: DetectorConfig):
    """
    Train a Faster R-CNN model on the custom dataset.

    Parameters
    ----------
    config : DetectorConfig
        Configuration object for the detector model
    """

    print("Setting up datasets...")
    d2_train_datasets_names = register_multiple_detection_datasets(
        root_path=config.data_root_path,
        dataset_names=config.train_dataset_names,
        fold="train",
        force_binary_class=True if config.num_classes == 1 else False
    )

    d2_valid_datasets_names = register_multiple_detection_datasets(
        root_path=config.data_root_path,
        dataset_names=config.train_dataset_names,
        fold="valid",
        force_binary_class=True if config.num_classes == 1 else False
    )

    print(f"Setting up trainer for dataset(s): {d2_train_datasets_names}")
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{config.model}_{now}"
    trainer = setup_trainer(d2_train_datasets_names, d2_valid_datasets_names, config, model_name)

    print("Starting training...")
    trainer.train()
    print(f"Training completed. Model saved in {config.train_output_path}")


def lazyconfig_to_dict(obj):
    """
    Recursively convert a LazyConfig object (which acts like nested
    namespaces/dicts) into a pure Python dict.
    """
    if isinstance(obj, Mapping):
        return {k: lazyconfig_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [lazyconfig_to_dict(v) for v in obj]
    else:
        return obj


def train_detrex_dino(config):
    from engine.models.detector.detectron2.detrex_models.dino.train_net import do_train
    print("Setting up datasets...")
    d2_train_datasets_names = register_multiple_detection_datasets(
        root_path=config.data_root_path,
        dataset_names=config.train_dataset_names,
        fold="train",
        force_binary_class=True if config.num_classes == 1 else False,
        combine_datasets=True
    )

    d2_valid_datasets_names = register_multiple_detection_datasets(
        root_path=config.data_root_path,
        dataset_names=config.train_dataset_names,
        fold="valid",
        force_binary_class=True if config.num_classes == 1 else False,
        combine_datasets=True
    )

    dataset_length = sum([len(DatasetCatalog.get(dataset_name)) for dataset_name in d2_train_datasets_names])

    # cfg = LazyConfig.load(f'/home/hugo/PycharmProjects/CanopyRS/engine/models/detector/detectron2/detrex_models/dino/configs/dino-swin/dino_swin_large_384_5scale_36ep.py')
    cfg = LazyConfig.load(f'/home/hugo/PycharmProjects/CanopyRS/engine/models/detector/detectron2/detrex_models/dino/configs/dino-resnet/dino_r50_4scale_24ep.py')
    cfg.dataloader.train.dataset.names = d2_train_datasets_names[0]
    cfg.dataloader.test.dataset.names = d2_valid_datasets_names[0]
    cfg.dataloader.train.num_workers = config.dataloader_num_workers
    cfg.dataloader.test.num_workers = config.dataloader_num_workers // 2
    cfg.train.seed = config.seed
    cfg.train.output_dir = config.train_output_path
    cfg.train.init_checkpoint = config.checkpoint_path
    cfg.train.log_period = config.train_log_interval
    cfg.train.max_iter = config.max_epochs * dataset_length // config.batch_size
    cfg.train.eval_period = config.eval_epoch_interval * dataset_length // config.batch_size
    cfg.dataloader.train.total_batch_size = config.batch_size
    cfg.model.num_classes = config.num_classes

    pprint(lazyconfig_to_dict(cfg))

    do_train(
        default_argument_parser().parse_args([]),   # passing empty list to simulate empty command line arguments
        cfg
    )
