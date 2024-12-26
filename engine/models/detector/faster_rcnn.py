from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator

from engine.config_parsers import DetectorConfig
from engine.models.backbone import build_detection_backbone
from engine.models.detector.detector_base import DetectorWrapperBase


def build_faster_rcnn(config: DetectorConfig):
    backbone = build_detection_backbone(
        backbone_model_pretrained=config.backbone_model_pretrained and not config.checkpoint_path,
        backbone=config.backbone,
        trainable_layers=config.trainable_layers
    )

    # Define an anchor generator
    anchor_generator = AnchorGenerator(
        sizes=config.anchor_sizes,
        aspect_ratios=config.aspect_ratios
    )

    # Update num classes to integrate the background
    if config.num_classes == 1:
        num_classes = 2   # 1 class + background
    else:
        num_classes = config.num_classes

    # Create the Faster R-CNN model
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_detections_per_img=config.box_predictions_per_image,
        box_score_thresh=config.box_score_thresh,
        box_nms_thresh=config.box_nms_thresh,
    )

    return model


class FasterRCNNWrapper(DetectorWrapperBase):
    def __init__(self, config: DetectorConfig):
        super().__init__(config)

        self.model = build_faster_rcnn(config)
        self.model.to(self.device)
        self.load_checkpoint(config.checkpoint_path)

    def forward(self, images, targets=None):
        return self.model(images, targets)




