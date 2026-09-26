from canopyrs.engine.components.base import Component, flatten_by_tile, COMPONENT_REGISTRY, register_component
from canopyrs.engine.components.tilerizer import Tilerizer
from canopyrs.engine.components.detector import Detector
from canopyrs.engine.components.segmenter import Segmenter
from canopyrs.engine.components.aggregator import Aggregator
from canopyrs.engine.components.classifier import Classifier

# Importing the component modules above runs their @register_component decorators, populating
# COMPONENT_REGISTRY (kind -> class) for Pipeline.from_config.

__all__ = ["Component", "flatten_by_tile", "COMPONENT_REGISTRY", "register_component",
           "Tilerizer", "Detector", "Segmenter", "Aggregator", "Classifier"]
