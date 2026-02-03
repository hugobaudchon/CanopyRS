from canopyrs.engine.config_parsers.base import BaseConfig


class AggregatorConfig(BaseConfig):
    score_threshold: float = 0.2
    nms_threshold: float = 0.5
    nms_algorithm: str = 'iou'
    detector_score_weight: float = 0.5
    segmenter_score_weight: float = 0.5
    scores_weighting_method: str = 'weighted_geometric_mean'
    min_centroid_distance_weight: float = None
    edge_band_buffer_percentage: float = 0.05
    best_geom_keep_area_ratio: float = 0.5
