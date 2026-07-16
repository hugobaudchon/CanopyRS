"""Contract tests for the Aggregator component: its requires/produces reflect the score weights."""

from canopyrs.engine.components.aggregator import Aggregator
from canopyrs.engine.config_parsers import AggregatorConfig
from canopyrs.engine.data import Crops, Objects, Tiles
from canopyrs.engine.constants import Col


def _need(component_requires):
    """The Objects Need in the aggregator's (objects, tiles) requires pair."""
    need, tiles_need = component_requires
    assert tiles_need.data_type is Tiles   # the NMS tile frames, asked for explicitly
    return need


def test_requires_carries_weighted_score_columns():
    # only detector weighted -> only detector_score is required as an input column
    agg = Aggregator(AggregatorConfig(detector_score_weight=1.0, segmenter_score_weight=0.0,
                                      classifier_score_weight=0.0))
    need = _need(agg.requires)
    assert need.data_type is Objects
    assert set(need.columns) == {Col.DETECTOR_SCORE}
    assert need.links == ("imagery",)
    assert need.crs is None                  # pixel detections and CRS classified objects both work
    assert need.on == (Tiles, Crops)         # raw detections, or classified objects on crops


def test_requires_reflects_multiple_weights():
    agg = Aggregator(AggregatorConfig(detector_score_weight=0.5, segmenter_score_weight=0.5,
                                      classifier_score_weight=1.0))
    need = _need(agg.requires)
    assert set(need.columns) == {Col.DETECTOR_SCORE, Col.SEGMENTER_SCORE, Col.CLASSIFIER_SCORE}


def test_produces_georeferenced_aggregator_score():
    agg = Aggregator(AggregatorConfig())
    assert agg.produces.data_type is Objects
    assert Col.AGGREGATOR_SCORE in agg.produces.columns
    assert agg.produces.links == ("prev_objects",)   # survivors point back to the input they kept
    assert agg.produces.crs is True                  # georeferenced output
