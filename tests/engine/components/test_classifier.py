"""
Tests for ClassifierComponent.
"""

import pytest
from unittest.mock import MagicMock, patch

from canopyrs.engine.components.classifier import ClassifierComponent
from canopyrs.engine.components.base import ComponentValidationError
from canopyrs.engine.constants import Col, StateKey


class TestClassifierRequirements:
    """Tests for ClassifierComponent requirements."""

    @patch('canopyrs.engine.components.classifier.CLASSIFIER_REGISTRY')
    def test_requires_tiles_path(self, mock_registry, mock_classifier_config):
        """Classifier requires tiles_path state."""
        mock_registry.get.return_value = MagicMock
        mock_registry.__contains__ = lambda self, key: True

        component = ClassifierComponent(
            config=mock_classifier_config,
            parent_output_path=None,
            component_id=0
        )

        assert StateKey.TILES_PATH in component.requires_state

    @patch('canopyrs.engine.components.classifier.CLASSIFIER_REGISTRY')
    def test_requires_infer_coco_path(self, mock_registry, mock_classifier_config):
        """Classifier requires infer_coco_path state."""
        mock_registry.get.return_value = MagicMock
        mock_registry.__contains__ = lambda self, key: True

        component = ClassifierComponent(
            config=mock_classifier_config,
            parent_output_path=None,
            component_id=0
        )

        assert StateKey.INFER_COCO_PATH in component.requires_state


class TestClassifierValidation:
    """Tests for ClassifierComponent validation."""

    @patch('canopyrs.engine.components.classifier.CLASSIFIER_REGISTRY')
    def test_validation_fails_without_tiles_path(self, mock_registry, mock_classifier_config):
        """Validation fails when tiles_path is missing."""
        mock_registry.get.return_value = MagicMock
        mock_registry.__contains__ = lambda self, key: True

        component = ClassifierComponent(
            config=mock_classifier_config,
            parent_output_path=None,
            component_id=0
        )

        available_state = {StateKey.INFER_COCO_PATH}  # Missing TILES_PATH
        available_columns = set()

        with pytest.raises(ComponentValidationError):
            component.validate(available_state, available_columns)

    @patch('canopyrs.engine.components.classifier.CLASSIFIER_REGISTRY')
    def test_validation_fails_without_coco_path(self, mock_registry, mock_classifier_config):
        """Validation fails when infer_coco_path is missing."""
        mock_registry.get.return_value = MagicMock
        mock_registry.__contains__ = lambda self, key: True

        component = ClassifierComponent(
            config=mock_classifier_config,
            parent_output_path=None,
            component_id=0
        )

        available_state = {StateKey.TILES_PATH}  # Missing INFER_COCO_PATH
        available_columns = set()

        with pytest.raises(ComponentValidationError):
            component.validate(available_state, available_columns)

    @patch('canopyrs.engine.components.classifier.CLASSIFIER_REGISTRY')
    def test_validation_passes_with_all_requirements(self, mock_registry, mock_classifier_config):
        """Validation passes when all requirements are met."""
        mock_registry.get.return_value = MagicMock
        mock_registry.__contains__ = lambda self, key: True

        component = ClassifierComponent(
            config=mock_classifier_config,
            parent_output_path=None,
            component_id=0
        )

        available_state = {StateKey.TILES_PATH, StateKey.INFER_COCO_PATH}
        available_columns = set()

        # Should not raise
        component.validate(available_state, available_columns)


class TestClassifierProduces:
    """Tests for what ClassifierComponent produces."""

    @patch('canopyrs.engine.components.classifier.CLASSIFIER_REGISTRY')
    def test_produces_classifier_columns(self, mock_registry, mock_classifier_config):
        """Classifier declares it produces classification columns."""
        mock_registry.get.return_value = MagicMock
        mock_registry.__contains__ = lambda self, key: True

        component = ClassifierComponent(
            config=mock_classifier_config,
            parent_output_path=None,
            component_id=0
        )

        assert Col.CLASSIFIER_SCORE in component.produces_columns
        assert Col.CLASSIFIER_CLASS in component.produces_columns
        assert Col.CLASSIFIER_SCORES in component.produces_columns

    @patch('canopyrs.engine.components.classifier.CLASSIFIER_REGISTRY')
    def test_produces_state_keys(self, mock_registry, mock_classifier_config):
        """Classifier declares it produces required state keys."""
        mock_registry.get.return_value = MagicMock
        mock_registry.__contains__ = lambda self, key: True

        component = ClassifierComponent(
            config=mock_classifier_config,
            parent_output_path=None,
            component_id=0
        )

        assert StateKey.INFER_GDF in component.produces_state
        assert StateKey.INFER_COCO_PATH in component.produces_state


class TestClassifierModelRegistry:
    """Tests for classifier model registry integration."""

    @patch('canopyrs.engine.components.classifier.CLASSIFIER_REGISTRY')
    def test_invalid_model_raises_error(self, mock_registry, mock_classifier_config):
        """Invalid model name raises ValueError."""
        mock_registry.__contains__ = lambda self, key: False

        with pytest.raises(ValueError) as exc_info:
            ClassifierComponent(
                config=mock_classifier_config,
                parent_output_path=None,
                component_id=0
            )
        assert "invalid" in str(exc_info.value).lower() or "model" in str(exc_info.value).lower()
