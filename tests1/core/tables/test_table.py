"""The Table base, tested with two small table types defined here."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from canopyrs1.core.tables.table import Table


class Parents(Table):
    id_column = "parent_id"


class Children(Table):
    id_column = "child_id"

    def __init__(self, df, parent_rows=None):
        super().__init__(df)
        self.parent_rows = parent_rows
        self._check_parent_ids("parent_id", parent_rows)


def _parents(n=3):
    return Parents(pd.DataFrame({"name": [f"p{i}" for i in range(n)]}))


def test_ids_are_row_numbers():
    assert list(_parents().df["parent_id"]) == [0, 1, 2]
    # Ids already in the frame are replaced: a row's id is its position.
    renumbered = Parents(pd.DataFrame({"parent_id": [10, 7], "name": ["a", "b"]}))
    assert list(renumbered.df["parent_id"]) == [0, 1]


def test_without_a_parent_its_ids_are_not_checked():
    children = Children(pd.DataFrame({"parent_id": [99]}))
    assert children.parent_rows is None


def test_a_parent():
    parents = _parents()
    children = Children(pd.DataFrame({"parent_id": [0, 2, None]}), parent_rows=parents)
    assert children.parent_rows is parents


def test_no_parent():
    children = Children(pd.DataFrame({"parent_id": [0]}))
    assert children.parent_rows is None


PARENT_IDS_OUTSIDE = [[0, 3], [-1, 0]]


@pytest.mark.parametrize("parent_ids", PARENT_IDS_OUTSIDE)
def test_a_parent_id_outside_the_parent_table(parent_ids):
    with pytest.raises(ValueError, match="Children.parent_id has ids outside its parent Parents"):
        Children(pd.DataFrame({"parent_id": parent_ids}), parent_rows=_parents())


def test_has_column():
    children = Children(pd.DataFrame({"score": [0.5, None], "empty": [None, None]}))
    assert children.has_column("score")
    assert not children.has_column("empty")
    assert not children.has_column("missing")
    assert Children(pd.DataFrame({"score": []})).has_column("score")


def test_has_crs():
    points = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:32618")
    assert Children(points).has_crs
    assert not Children(points.set_crs(None, allow_override=True)).has_crs
    assert not Children(pd.DataFrame({"parent_id": [0]})).has_crs  # no geometry at all


def test_len_and_repr():
    parents = _parents(4)
    assert len(parents) == 4
    assert repr(parents) == "Parents(4 rows)"
