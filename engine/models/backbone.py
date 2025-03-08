from torchvision.models import ResNet152_Weights, ResNet101_Weights, ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def build_detection_backbone(backbone_model_pretrained: bool,
                             backbone: str,
                             freeze_layers: int):
    if backbone.startswith('resnet'):
        if backbone_model_pretrained:
            if backbone == 'resnet50':
                weights = ResNet50_Weights.DEFAULT
            elif backbone == 'resnet101':
                weights = ResNet101_Weights.DEFAULT
            elif backbone == 'resnet152':
                weights = ResNet152_Weights.DEFAULT
            else:
                raise NotImplementedError(f"Pretrained weights for {backbone} not implemented."
                                          f" But can probably add the logic here.")
        else:
            weights = None

        resnet = resnet_fpn_backbone(
            backbone_name=backbone,
            weights=weights,
            trainable_layers=5 if freeze_layers == -1 or not freeze_layers else 5 - freeze_layers
        )

        return resnet
    else:
        raise NotImplementedError(f"Backbone {backbone} not implemented. But can probably add the logic here.")