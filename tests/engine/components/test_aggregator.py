"""
Tests for AggregatorComponent.
"""

import pytest

from canopyrs.engine.components.aggregator import AggregatorComponent
from canopyrs.engine.constants import Col, StateKey


class TestAggregatorRequirements:
    """Tests for AggregatorComponent requirements based on config."""

    def test_requires_detector_score_when_weight_positive(self, mock_aggregator_config):
        """Aggregator requires detector_score column when weight > 0."""
        mock_aggregator_config.detector_score_weight = 1.0
        mock_aggregator_config.segmenter_score_weight = 0.0

        component = AggregatorComponent(
            config=mock_aggregator_config,
            parent_output_path=None,
            component_id=0
        )

        assert Col.DETECTOR_SCORE in component.requires_columns
        assert Col.SEGMENTER_SCORE not in component.requires_columns

    def test_requires_segmenter_score_when_weight_positive(self, mock_aggregator_config):
        """Aggregator requires segmenter_score column when weight > 0."""
        mock_aggregator_config.detector_score_weight = 0.0
        mock_aggregator_config.segmenter_score_weight = 1.0

        component = AggregatorComponent(
            config=mock_aggregator_config,
            parent_output_path=None,
            component_id=0
        )

        assert Col.SEGMENTER_SCORE in component.requires_columns
        assert Col.DETECTOR_SCORE not in component.requires_columns

    def test_requires_both_scores_when_both_weights_positive(self, mock_aggregator_config):
        """Aggregator requires both score columns when both weights > 0."""
        mock_aggregator_config.detector_score_weight = 0.5
        mock_aggregator_config.segmenter_score_weight = 0.5

        component = AggregatorComponent(
            config=mock_aggregator_config,
            parent_output_path=None,
            component_id=0
        )

        assert Col.DETECTOR_SCORE in component.requires_columns
        assert Col.SEGMENTER_SCORE in component.requires_columns

    def test_base_requirements_always_present(self, mock_aggregator_config):
        """Base requirements are always present regardless of config."""
        component = AggregatorComponent(
            config=mock_aggregator_config,
            parent_output_path=None,
            component_id=0
        )

        assert StateKey.INFER_GDF in component.requires_state
        assert StateKey.PRODUCT_NAME in component.requires_state
        assert Col.GEOMETRY in component.requires_columns
        assert Col.OBJECT_ID in component.requires_columns
        assert Col.TILE_PATH in component.requires_columns


class TestAggregatorProduces:
    """Tests for what AggregatorComponent produces."""

    def test_produces_aggregator_score_column(self, mock_aggregator_config):
        """Aggregator declares it produces aggregator_score column."""
        component = AggregatorComponent(
            config=mock_aggregator_config,
            parent_output_path=None,
            component_id=0
        )

        assert Col.AGGREGATOR_SCORE in component.produces_columns

    def test_produces_infer_gdf_state(self, mock_aggregator_config):
        """Aggregator declares it produces updated infer_gdf."""
        component = AggregatorComponent(
            config=mock_aggregator_config,
            parent_output_path=None,
            component_id=0
        )

        assert StateKey.INFER_GDF in component.produces_state
        assert StateKey.INFER_COCO_PATH in component.produces_state
