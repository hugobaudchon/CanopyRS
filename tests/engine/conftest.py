"""
Engine-specific test fixtures.

These fixtures are specific to pipeline and component testing.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from canopyrs.engine.constants import Col, StateKey
from canopyrs.engine.data_state import DataState


# =============================================================================
# Config Fixtures
# =============================================================================

@pytest.fixture
def mock_detector_config():
    """Mock detector config."""
    config = MagicMock()
    config.model = "test_detector"
    config.weights_path = "/path/to/weights.pth"
    config.device = "cpu"
    config.batch_size = 4
    return config


@pytest.fixture
def mock_segmenter_config():
    """Mock segmenter config."""
    config = MagicMock()
    config.model = "test_segmenter"
    config.weights_path = "/path/to/weights.pth"
    config.device = "cpu"
    config.batch_size = 4
    return config


@pytest.fixture
def mock_classifier_config():
    """Mock classifier config."""
    config = MagicMock()
    config.model = "test_classifier"
    config.weights_path = "/path/to/weights.pth"
    config.device = "cpu"
    config.batch_size = 4
    return config


@pytest.fixture
def mock_aggregator_config():
    """Mock aggregator config."""
    config = MagicMock()
    config.detector_score_weight = 1.0
    config.segmenter_score_weight = 0.0
    config.scores_weighting_method = "weighted_average"
    config.min_centroid_distance_weight = 0.0
    config.score_threshold = 0.5
    config.nms_threshold = 0.5
    config.nms_algorithm = "nms"
    config.best_geom_keep_area_ratio = 0.5
    config.edge_band_buffer_percentage = 0.1
    return config


@pytest.fixture
def mock_tilerizer_config():
    """Mock tilerizer config for tile mode."""
    config = MagicMock()
    config.tile_type = "tile"
    config.tile_size = 512
    config.tile_overlap = 0.1
    config.ground_resolution = None
    return config


@pytest.fixture
def mock_tilerizer_polygon_config():
    """Mock tilerizer config for polygon mode."""
    config = MagicMock()
    config.tile_type = "polygon"
    config.tile_size = 256
    config.tile_overlap = 0.0
    config.ground_resolution = None
    return config


# =============================================================================
# Component State Fixtures
# =============================================================================

@pytest.fixture
def state_ready_for_detector(sample_gdf_minimal, tmp_path):
    """DataState ready for detector component."""
    state = DataState(
        tiles_path=str(tmp_path / "tiles"),
        parent_output_path=str(tmp_path / "output"),
        product_name="test_product",
    )
    return state


@pytest.fixture
def state_ready_for_aggregator(sample_gdf_with_detector_score, tmp_path):
    """DataState ready for aggregator component."""
    state = DataState(
        imagery_path=str(tmp_path / "test_image.tif"),
        parent_output_path=str(tmp_path / "output"),
        product_name="test_product",
    )
    state.infer_gdf = sample_gdf_with_detector_score
    state.infer_gdf_columns_to_pass = {Col.DETECTOR_SCORE, Col.DETECTOR_CLASS}
    return state


@pytest.fixture
def state_ready_for_classifier(tmp_path):
    """DataState ready for classifier component."""
    state = DataState(
        tiles_path=str(tmp_path / "polygon_tiles"),
        infer_coco_path=str(tmp_path / "coco" / "annotations.json"),
        parent_output_path=str(tmp_path / "output"),
        product_name="test_product",
    )
    return state
