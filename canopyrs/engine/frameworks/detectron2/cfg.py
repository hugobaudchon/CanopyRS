"""Shared detectron2/detrex model-config builders.

Used by BOTH the inference wrappers (models/{detector,segmenter}/detectron2_infer.py) and the
trainers (engine/train/detectron2/) — they live here, framework-shared and task-shared, so
inference never imports the training machinery (wandb, trainers, evaluators).
"""

import sys
from pathlib import Path

import detrex
from detectron2.config import LazyConfig, get_cfg
from detectron2.model_zoo import model_zoo

from canopyrs.engine.frameworks.detectron2.augmentation import AugmentationAdder


def get_base_detectron2_model_cfg(config):
    cfg = get_cfg()

    # Load base configs for Faster R-CNN
    cfg.merge_from_file(model_zoo.get_config_file(config.architecture))

    # Load pre-trained model weights
    if config.backbone_model_pretrained:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config.architecture)

    if config.checkpoint_path is not None:
        cfg.MODEL.WEIGHTS = config.checkpoint_path

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.num_classes
    if config.anchor_sizes is not None:
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [list(s) for s in config.anchor_sizes]
    cfg.SOLVER.AMP.ENABLED = config.use_amp

    if config.box_score_thresh is not None:
        if cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST != config.box_score_thresh:
            print(f"Changing box score threshold from {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST} to {config.box_score_thresh}.")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.box_score_thresh
    if config.box_nms_thresh is not None:
        if cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST != config.box_nms_thresh:
            print(f"Changing box NMS threshold from {cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST} to {config.box_nms_thresh}.")
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config.box_nms_thresh

    # Augmentations
    AugmentationAdder().modify_detectron2_augmentation_config(config, cfg)

    cfg.TEST.DETECTIONS_PER_IMAGE = config.box_predictions_per_image

    return cfg




def get_base_detrex_model_cfg(config):
    detrex_root = Path(next(iter(detrex.__path__))).resolve()
    # Depending on environment it might pick up the wrong detrex path
    if not (detrex_root / 'projects').exists() and (detrex_root.parent / 'projects').exists():
        detrex_root = detrex_root.parent
    if str(detrex_root) not in sys.path:
        sys.path.insert(0, str(detrex_root))

    # fixing architecture path from old CanopyRS versions
    if config.architecture == "dino-swin/dino_swin_large_384_5scale_36ep.py":
        config.architecture = "dino/configs/dino-swin/dino_swin_large_384_5scale_36ep.py"

    # loading base config
    cfg = LazyConfig.load(str(detrex_root / 'projects' / config.architecture))
    cfg.train.init_checkpoint = config.checkpoint_path

    # dino
    if hasattr(cfg.model, 'num_classes'):
        cfg.model.num_classes = config.num_classes
    elif hasattr(cfg.model, 'params') and hasattr(cfg.model.params, 'num_classes'):
        cfg.model.params.num_classes = config.num_classes

    # mask2former
    if hasattr(cfg.model, "sem_seg_head") and hasattr(cfg.model.sem_seg_head, "num_classes"):
        cfg.model.sem_seg_head.num_classes = config.num_classes

    if (
        hasattr(cfg.model, "sem_seg_head")
        and hasattr(cfg.model.sem_seg_head, "transformer_predictor")
        and hasattr(cfg.model.sem_seg_head.transformer_predictor, "num_classes")
    ):
        cfg.model.sem_seg_head.transformer_predictor.num_classes = config.num_classes

    # optimizer
    if hasattr(cfg.model, "criterion") and hasattr(cfg.model.criterion, "num_classes"):
        cfg.model.criterion.num_classes = config.num_classes

    # Custom Augmentations
    augmentation_adder = AugmentationAdder()
    cfg.dataloader.train.mapper.augmentation = augmentation_adder.get_augmentation_detrex_train(config)
    cfg.dataloader.train.mapper.augmentation_with_crop = None   # we have our own set of augmentations, including cropping, in augmentation_adder
    cfg.dataloader.test.mapper.augmentation = augmentation_adder.get_augmentation_detrex_test(config)

    # Enable AMP (mixed-precision).
    cfg.train.amp.enabled = config.use_amp

    return cfg


