from engine.config_parsers.base import BaseConfig


class EmbedderConfig(BaseConfig):
    model: str = 'resnet50'
    backbone: str = None
    checkpoint_path: str = None
    image_size: int = 224
    final_embedding_size: int = 768
    mean_std_descriptor: str = 'imagenet' #'forest_qpeb'
    use_cls_token: bool = False
    image_size_center_crop_pad: int = 224
