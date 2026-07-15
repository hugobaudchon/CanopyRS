"""Contract tests for the Classifier component: its one_of input shape and produced columns."""

from canopyrs.engine.components.classifier import Classifier
from canopyrs.engine.config_parsers import ClassifierConfig
from canopyrs.engine.contracts import AnyOf
from canopyrs.engine.data import Imagery, Objects
from canopyrs.engine.constants import Col, ImageKind


def _config(**overrides):
    """A minimal valid ClassifierConfig (model / architecture / num_classes are required)."""
    return ClassifierConfig(model='resnet', architecture='resnet50', num_classes=2, **overrides)


def test_requires_is_one_of_objects_or_tiles():
    clf = Classifier(_config())
    assert len(clf.requires) == 1
    req = clf.requires[0]
    assert isinstance(req, AnyOf)
    # per-object crops preferred, then whole tiles (readability is guaranteed by the imagery tree)
    assert [alt.data_type for alt in req.alternatives] == [Objects, Imagery]
    assert req.alternatives[1].kind == ImageKind.TILE


def test_produces_class_columns_without_names():
    clf = Classifier(_config())
    cols = set(clf.produces.columns)
    assert cols == {Col.CLASSIFIER_CLASS, Col.CLASSIFIER_SCORE, Col.CLASSIFIER_SCORES}
    assert Col.CLASSIFIER_CLASS_NAME not in cols   # no class_names configured


def test_produces_class_name_when_names_configured():
    clf = Classifier(_config(class_names=['deadwood', 'live']))
    assert Col.CLASSIFIER_CLASS_NAME in clf.produces.columns
