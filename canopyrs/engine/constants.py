"""
Standardized constants for the CanopyRS pipeline.

These constants ensure consistent naming across all components.
"""


# Naming conventions
INFER_AOI_NAME = "infer"


class Col:
    """Standardized GeoDataFrame column names."""
    # Core columns
    GEOMETRY = "geometry"
    OBJECT_ID = "canopyrs_object_id"
    TILE_PATH = "tile_path"

    # Detector columns
    DETECTOR_SCORE = "detector_score"
    DETECTOR_CLASS = "detector_class"

    # Segmenter columns
    SEGMENTER_SCORE = "segmenter_score"
    SEGMENTER_CLASS = "segmenter_class"

    # Aggregator columns
    AGGREGATOR_SCORE = "aggregator_score"

    # Classifier columns
    CLASSIFIER_SCORE = "classifier_score"
    CLASSIFIER_CLASS = "classifier_class"
    CLASSIFIER_SCORES = "classifier_scores"  # Full list of scores for all classes


class StateKey:
    """Standardized DataState attribute names."""
    IMAGERY_PATH = "imagery_path"
    TILES_PATH = "tiles_path"
    INFER_GDF = "infer_gdf"
    INFER_COCO_PATH = "infer_coco_path"
    PRODUCT_NAME = "product_name"
