from engine.config_parsers.base import BaseConfig


class InferIOConfig(BaseConfig):
    input_imagery: str
    output_folder: str

    input_gpkg: str = None
    input_coco: str = None

