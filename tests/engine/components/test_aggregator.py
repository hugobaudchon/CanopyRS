"""
Tests for AggregatorComponent.
"""

import pytest
from unittest.mock import MagicMock, patch

from canopyrs.engine.components.aggregator import AggregatorComponent
from canopyrs.engine.components.base import ComponentValidationError
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


class TestAggregatorValidation:
    """Tests for AggregatorComponent validation."""

    def test_validation_fails_without_infer_gdf(self, mock_aggregator_config):
        """Validation fails when infer_gdf is missing."""
        component = AggregatorComponent(
            config=mock_aggregator_config,
            parent_output_path=None,
            component_id=0
        )

        available_state = {StateKey.PRODUCT_NAME}  # Missing INFER_GDF
        available_columns = set()

        with pytest.raises(ComponentValidationError) as exc_info:
            component.validate(available_state, available_columns)
        assert "infer_gdf" in str(exc_info.value).lower() or StateKey.INFER_GDF in str(exc_info.value)

    def test_validation_fails_without_product_name(self, mock_aggregator_config, sample_gdf_with_detector_score):
        """Validation fails when product_name is missing."""
        component = AggregatorComponent(
            config=mock_aggregator_config,
            parent_output_path=None,
            component_id=0
        )

        available_state = {StateKey.INFER_GDF}  # Missing PRODUCT_NAME
        available_columns = set(sample_gdf_with_detector_score.columns)

        with pytest.raises(ComponentValidationError):
            component.validate(available_state, available_columns)

    def test_validation_fails_without_required_score_column(
        self, mock_aggregator_config, sample_gdf_with_tile_path
    ):
        """Validation fails when required score column is missing."""
        mock_aggregator_config.detector_score_weight = 1.0

        component = AggregatorComponent(
            config=mock_aggregator_config,
            parent_output_path=None,
            component_id=0
        )

        available_state = {StateKey.INFER_GDF, StateKey.PRODUCT_NAME}
        available_columns = set(sample_gdf_with_tile_path.columns)  # No detector_score

        with pytest.raises(ComponentValidationError) as exc_info:
            component.validate(available_state, available_columns)
        assert Col.DETECTOR_SCORE in str(exc_info.value)

    def test_validation_passes_with_all_requirements(
        self, mock_aggregator_config, sample_gdf_with_detector_score
    ):
        """Validation passes when all requirements are met."""
        mock_aggregator_config.detector_score_weight = 1.0
        mock_aggregator_config.segmenter_score_weight = 0.0

        component = AggregatorComponent(
            config=mock_aggregator_config,
            parent_output_path=None,
            component_id=0
        )

        available_state = {StateKey.INFER_GDF, StateKey.PRODUCT_NAME}
        available_columns = set(sample_gdf_with_detector_score.columns)

        # Should not raise
        component.validate(available_state, available_columns)


class TestAggregatorHints:
    """Tests for AggregatorComponent validation hints."""

    def test_hints_for_missing_detector_score(self, mock_aggregator_config):
        """Hints explain how to resolve missing detector_score."""
        mock_aggregator_config.detector_score_weight = 1.0
        mock_aggregator_config.segmenter_score_weight = 0.0

        component = AggregatorComponent(
            config=mock_aggregator_config,
            parent_output_path=None,
            component_id=0
        )

        hint = component.column_hints.get(Col.DETECTOR_SCORE, "")
        assert "detector" in hint.lower()
        assert "weight" in hint.lower() or "0" in hint

    def test_hints_for_missing_segmenter_score(self, mock_aggregator_config):
        """Hints explain how to resolve missing segmenter_score."""
        mock_aggregator_config.detector_score_weight = 0.0
        mock_aggregator_config.segmenter_score_weight = 1.0

        component = AggregatorComponent(
            config=mock_aggregator_config,
            parent_output_path=None,
            component_id=0
        )

        hint = component.column_hints.get(Col.SEGMENTER_SCORE, "")
        assert "segmenter" in hint.lower()


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
