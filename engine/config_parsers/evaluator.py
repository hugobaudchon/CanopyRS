from engine.config_parsers import BaseConfig


class EvaluatorConfig(BaseConfig):
    type: str = 'instance_detection'
    max_dets: list[int] = [1, 10, 100, 500] # added 500 as we often have > 100 tree detections per tile
    raster_eval_ground_resolution: float = 0.05