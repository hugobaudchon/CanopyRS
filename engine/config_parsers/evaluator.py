from engine.config_parsers import BaseConfig


class EvaluatorConfig(BaseConfig):
    type: str = 'instance_detection'
    level: str = 'tile'    # or 'raster'
    max_dets: list[int] = [1, 10, 100] # Can add more values, like [1, 10, 100, 500]
    raster_eval_ground_resolution: float = 0.05