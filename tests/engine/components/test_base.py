"""
Tests for BaseComponent validation logic.
"""

import pytest

from canopyrs.engine.components.base import (
    BaseComponent,
    ComponentValidationError,
)
from canopyrs.engine.constants import Col, StateKey


class TestBaseComponentValidation:
    """Tests for BaseComponent validation logic."""

    def test_validate_state_missing_required(self):
        """Validation fails when required state is missing."""
        # Create a concrete component subclass for testing
        class TestComponent(BaseComponent):
            name = "test"

            def __call__(self, data_state):
                pass

        component = TestComponent(config=None, parent_output_path=None, component_id=0)
        component.requires_state = {StateKey.INFER_GDF}
        component.requires_columns = set()
        component.state_hints = {}
        component.column_hints = {}

        available_state = set()  # Empty - missing INFER_GDF
        available_columns = set()

        with pytest.raises(ComponentValidationError):
            component.validate(available_state, available_columns)

    def test_validate_columns_missing_required(self, sample_gdf_minimal):
        """Validation fails when required columns are missing."""
        class TestComponent(BaseComponent):
            name = "test"

            def __call__(self, data_state):
                pass

        component = TestComponent(config=None, parent_output_path=None, component_id=0)
        component.requires_state = {StateKey.INFER_GDF}
        component.requires_columns = {Col.DETECTOR_SCORE}  # Not in sample_gdf_minimal
        component.state_hints = {}
        component.column_hints = {Col.DETECTOR_SCORE: "Need detector score"}

        available_state = {StateKey.INFER_GDF}
        available_columns = set(sample_gdf_minimal.columns)  # Missing DETECTOR_SCORE

        with pytest.raises(ComponentValidationError) as exc_info:
            component.validate(available_state, available_columns)
        assert Col.DETECTOR_SCORE in str(exc_info.value)

    def test_validate_passes_with_all_requirements(self, sample_gdf_with_detector_score):
        """Validation passes when all requirements are met."""
        class TestComponent(BaseComponent):
            name = "test"

            def __call__(self, data_state):
                pass

        component = TestComponent(config=None, parent_output_path=None, component_id=0)
        component.requires_state = {StateKey.INFER_GDF}
        component.requires_columns = {Col.DETECTOR_SCORE, Col.GEOMETRY}
        component.state_hints = {}
        component.column_hints = {}

        available_state = {StateKey.INFER_GDF}
        available_columns = set(sample_gdf_with_detector_score.columns)

        # Should not raise
        component.validate(available_state, available_columns)

    def test_validate_returns_errors_without_raising(self):
        """Validation can return errors without raising."""
        class TestComponent(BaseComponent):
            name = "test"

            def __call__(self, data_state):
                pass

        component = TestComponent(config=None, parent_output_path=None, component_id=0)
        component.requires_state = {StateKey.INFER_GDF, StateKey.TILES_PATH}
        component.requires_columns = set()
        component.state_hints = {}
        component.column_hints = {}

        errors = component.validate(set(), set(), raise_on_error=False)
        assert len(errors) == 2
        assert any(StateKey.INFER_GDF in e for e in errors)
        assert any(StateKey.TILES_PATH in e for e in errors)
