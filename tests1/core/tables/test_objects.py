import pandas as pd
import pytest
from shapely.geometry import Point, box

from canopyrs1.core.constants import Col, GeomKind
from canopyrs1.core.tables.objects import Objects
from canopyrs1.core.tables.table import Table


class Images(Table):
    """A stand-in for the imagery tables."""

    id_column = Col.IMAGE_ID


def _images(n=2):
    return Images(pd.DataFrame({"name": [f"image{i}" for i in range(n)]}))


def _history():
    """Return (tiles, boxes, masks, kept): boxes found on tiles, masks made from the boxes, and
    the masks kept by an aggregation step. Only the boxes are linked to the tiles."""
    tiles = _images()
    boxes = Objects.build(
        geometry=[box(0, 0, 10, 10), box(5, 5, 20, 20), box(30, 30, 40, 40)],
        geom_kind=GeomKind.BOX,
        parent_image_id=[0, 0, 1],
        parent_imagery=tiles,
        columns={Col.DETECTOR_SCORE: [0.9, 0.8, 0.7]},
    )
    masks = Objects.build(
        geometry=[Point(5, 5).buffer(4), Point(35, 35).buffer(4)],
        geom_kind=GeomKind.MASK,
        parent_object_id=[0, 2],
        parent_objects=boxes,
        columns={Col.SEGMENTER_SCORE: [0.6, 0.5]},
    )
    kept = Objects.build(
        geometry=[Point(35, 35).buffer(4)],
        geom_kind=GeomKind.MASK,
        crs="EPSG:32618",
        parent_object_id=[1],
        parent_objects=masks,
    )
    return tiles, boxes, masks, kept


def test_build():
    tiles, boxes, _, _ = _history()
    assert list(boxes.df[Col.OBJECT_ID]) == [0, 1, 2]
    assert list(boxes.df[Col.PARENT_IMAGE_ID]) == [0, 0, 1]
    assert boxes.df[Col.PARENT_OBJECT_ID].isna().all()
    assert (boxes.df[Col.GEOM_KIND] == GeomKind.BOX).all()
    assert boxes.parent_imagery is tiles and boxes.parent_objects is None
    assert not boxes.has_crs


def test_build_nothing():
    empty = Objects.build(geometry=[], geom_kind=GeomKind.BOX)
    assert len(empty) == 0
    assert {Col.OBJECT_ID, Col.PARENT_IMAGE_ID, Col.PARENT_OBJECT_ID} <= set(empty.df.columns)


def test_get_column_follows_the_history():
    _, _, masks, kept = _history()
    assert kept.get_column(Col.SEGMENTER_SCORE).tolist() == [0.5]  # one step back
    assert kept.get_column(Col.DETECTOR_SCORE).tolist() == [0.7]  # two steps back
    assert masks.get_column(Col.DETECTOR_SCORE).tolist() == [0.9, 0.7]
    assert kept.get_column(Col.DETECTOR_SCORE).name == Col.DETECTOR_SCORE


def test_an_object_without_a_parent_gets_a_missing_value():
    _, boxes, _, _ = _history()
    masks = Objects.build(
        geometry=[Point(0, 0), Point(1, 1)],
        geom_kind=GeomKind.POINT,
        parent_object_id=[1, None],
        parent_objects=boxes,
    )
    values = masks.get_column(Col.DETECTOR_SCORE)
    assert values[0] == 0.8 and pd.isna(values[1])


def test_get_column_when_no_table_has_it():
    _, _, _, kept = _history()
    with pytest.raises(KeyError, match="classifier_score"):
        kept.get_column(Col.CLASSIFIER_SCORE)


def test_has_column_follows_the_history():
    _, _, _, kept = _history()
    assert kept.has_column(Col.DETECTOR_SCORE)
    assert kept.has_column(Col.SEGMENTER_SCORE)
    assert not kept.has_column(Col.CLASSIFIER_SCORE)


def test_get_parent_imagery_follows_the_history():
    tiles, boxes, masks, kept = _history()
    # Only the boxes are linked to the tiles: the masks and the kept masks find them through
    # their history.
    assert boxes.get_parent_imagery() is tiles
    assert masks.get_parent_imagery() is tiles
    assert kept.get_parent_imagery() is tiles
    assert kept.parent_imagery is None and kept.parent_objects is masks


def test_get_parent_imagery_without_any():
    lone = Objects.build(geometry=[Point(0, 0)], geom_kind=GeomKind.POINT)
    assert lone.get_parent_imagery() is None


def test_an_unknown_geom_kind():
    with pytest.raises(ValueError, match="Unknown geom_kind \\['polygon'\\]"):
        Objects.build(geometry=[box(0, 0, 1, 1)], geom_kind="polygon")


def test_a_parent_object_id_outside_the_parent_objects():
    _, boxes, _, _ = _history()
    with pytest.raises(ValueError, match="parent_object_id has ids outside its parent Objects"):
        Objects.build(
            geometry=[Point(0, 0)],
            geom_kind=GeomKind.POINT,
            parent_object_id=[3],
            parent_objects=boxes,
        )
