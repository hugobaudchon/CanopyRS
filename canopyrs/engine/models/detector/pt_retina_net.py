from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator

from canopyrs.engine.config_parsers import DetectorConfig
from canopyrs.engine.models.backbone import build_detection_backbone
from canopyrs.engine.models.detector.detector_base import TorchVisionDetectorWrapperBase
from canopyrs.engine.models.registry import DETECTOR_REGISTRY


def build_retina_net(config: DetectorConfig):
    backbone = build_detection_backbone(
        backbone_model_pretrained=config.backbone_model_pretrained and not config.checkpoint_path,
        backbone=config.architecture,
        freeze_layers=config.freeze_layers
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
    model = RetinaNet(
        backbone=backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        detections_per_img=config.box_predictions_per_image,
        score_thresh=config.box_score_thresh,
        nms_thresh=config.box_nms_thresh
    )

    return model


@DETECTOR_REGISTRY.register('retina_net')
class RetinaNetWrapper(TorchVisionDetectorWrapperBase):
    def __init__(self, config: DetectorConfig):
        super().__init__(config)

        self.model = build_retina_net(config)
        self.model.to(self.device)
        self.load_checkpoint(config.checkpoint_path)

    def forward(self, images, targets=None):
        return self.model(images, targets)