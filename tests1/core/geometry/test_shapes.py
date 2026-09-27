import pytest
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon, box

from canopyrs1.core.geometry.shapes import (
    get_largest_part,
    keep_polygon_parts,
    remove_holes,
    remove_small_parts,
    repair_polygon,
)

BOWTIE = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])  # its outline crosses itself
SPIKE = Polygon([(0, 0), (4, 0), (4, 4), (2, 4), (2, 6), (2, 4), (0, 4)])  # a line sticks out
WITH_HOLE = Polygon(box(0, 0, 10, 10).exterior, holes=[box(4, 4, 6, 6).exterior])
TWO_PARTS = MultiPolygon([box(0, 0, 10, 10), box(20, 20, 21, 21)])  # areas 100 and 1


def test_keep_polygon_parts():
    assert keep_polygon_parts(TWO_PARTS) is TWO_PARTS
    mixed = GeometryCollection(
        [
            box(0, 0, 1, 1),
            LineString([(5, 5), (6, 6)]),
            Point(9, 9),
            MultiPolygon([box(2, 2, 3, 3), box(4, 4, 5, 5)]),
        ]
    )
    kept = keep_polygon_parts(mixed)
    assert isinstance(kept, MultiPolygon) and len(kept.geoms) == 3 and kept.area == 3
    # Two boxes that only touch intersect along a line: nothing with an area is left.
    assert keep_polygon_parts(box(0, 0, 1, 1).intersection(box(1, 0, 2, 1))) == Polygon()
    assert keep_polygon_parts(GeometryCollection([box(0, 0, 1, 1)])).equals(box(0, 0, 1, 1))


def test_repair_polygon():
    assert repair_polygon(WITH_HOLE) is WITH_HOLE
    bowtie = repair_polygon(BOWTIE)
    assert bowtie.is_valid and isinstance(bowtie, MultiPolygon) and bowtie.area == pytest.approx(2)
    spike = repair_polygon(SPIKE)
    assert spike.is_valid and spike.equals(box(0, 0, 4, 4))  # the line is dropped
    flat = Polygon([(0, 0), (1, 1), (2, 2)])  # no area at all
    assert repair_polygon(flat) == Polygon()


def test_get_largest_part():
    assert get_largest_part(TWO_PARTS).equals(box(0, 0, 10, 10))
    assert get_largest_part(MultiPolygon()) == Polygon()
    assert get_largest_part(WITH_HOLE) is WITH_HOLE
    point = Point(1, 1)
    assert get_largest_part(point) is point


def test_remove_holes():
    assert remove_holes(WITH_HOLE).equals(box(0, 0, 10, 10))
    filled = remove_holes(MultiPolygon([WITH_HOLE, box(20, 20, 21, 21)]))
    assert isinstance(filled, MultiPolygon) and filled.area == 101
    assert remove_holes(Polygon()) == Polygon()


def test_remove_small_parts():
    assert remove_small_parts(TWO_PARTS, min_area=10).equals(box(0, 0, 10, 10))
    assert remove_small_parts(TWO_PARTS, min_area=1).equals(TWO_PARTS)
    assert remove_small_parts(TWO_PARTS, min_area=1000) == Polygon()
    small = box(0, 0, 1, 1)
    assert remove_small_parts(small, min_area=10) is small  # a single part is kept
