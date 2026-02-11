from typing import Optional, Dict, Any

from pydantic import model_validator

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

    # Optional dict-style scores_weights for YAML compatibility
    scores_weights: Optional[Dict[str, float]] = None

    @model_validator(mode='before')
    @classmethod
    def parse_scores_weights(cls, data: Any) -> Any:
        """
        Convert scores_weights dict to individual weight fields.

        If scores_weights is provided (e.g., {'detector_score': 1.0}),
        set the corresponding weight fields and default others to 0.
        """
        if isinstance(data, dict) and 'scores_weights' in data and data['scores_weights'] is not None:
            scores_weights = data['scores_weights']
            # When scores_weights dict is provided, only use weights explicitly listed
            # Set others to 0 (not the default 0.5)
            data['detector_score_weight'] = scores_weights.get('detector_score', 0.0)
            data['segmenter_score_weight'] = scores_weights.get('segmenter_score', 0.0)
        return data
