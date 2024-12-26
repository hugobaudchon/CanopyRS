from engine.config_parsers.base import BaseConfig


class DetectorConfig(BaseConfig):
    # General model definition
    model: str = 'faster_rcnn'
    backbone: str = 'resnet50'
    checkpoint_path: str = None
    batch_size: int = 8
    num_classes: int = 2
    box_predictions_per_image: int = 250
    anchor_sizes: tuple = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios: tuple = ((0.5, 1.0, 2.0),) * 5
    box_score_thresh: float = 0.05
    box_nms_thresh: float = 0.5

    # Training
    main_metric: str = 'mAP'            # TODO add support for this
    lr: float = 1e-4
    max_epochs: int = 100
    trainable_layers: int = -1
    train_log_interval: int = 10
    grad_accumulation_steps: int = 1
    backbone_model_pretrained: bool = True
    scheduler_step_size: int = 10
    scheduler_warmup_steps: int = 100
    scheduler_gamma: float = 0.1
