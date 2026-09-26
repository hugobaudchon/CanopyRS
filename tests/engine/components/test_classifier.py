"""Contract tests for the Classifier component: its Objects-on-Crops input and produced columns."""

from canopyrs.engine.components.classifier import Classifier
from canopyrs.engine.config_parsers import ClassifierConfig
from canopyrs.engine.contracts import Need
from canopyrs.engine.data import Crops, Objects
from canopyrs.engine.constants import Col


def _config(**overrides):
    """A minimal valid ClassifierConfig (model / architecture / num_classes are required)."""
    return ClassifierConfig(model='resnet', architecture='resnet50', num_classes=2, **overrides)


def test_requires_objects_on_crops():
    clf = Classifier(_config())
    assert len(clf.requires) == 1
    need = clf.requires[0]
    assert isinstance(need, Need)
    # one mode only: objects each pointing at their own crop — never whole tiles
    assert need.data_type is Objects
    assert need.on is Crops
    assert "imagery" in need.links


def test_produces_class_columns_without_names():
    clf = Classifier(_config())
    cols = set(clf.produces.columns)
    assert cols == {Col.CLASSIFIER_CLASS, Col.CLASSIFIER_SCORE, Col.CLASSIFIER_SCORES}
    assert Col.CLASSIFIER_CLASS_NAME not in cols   # no class_names configured


def test_produces_class_name_when_names_configured():
    clf = Classifier(_config(class_names=['deadwood', 'live']))
    assert Col.CLASSIFIER_CLASS_NAME in clf.produces.columns
