from engine.config_parsers.base import BaseConfig


class AggregatorConfig(BaseConfig):
    score_threshold: float = 0.2
    nms_threshold: float = 0.5
    nms_algorithm: str = 'iou'
    detector_score_weight: float = 0.5
    segmenter_score_weight: float = 0.5
    scores_weighting_method: str = 'weighted_geometric_mean'
