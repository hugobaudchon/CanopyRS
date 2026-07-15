"""Unit tests for the v3 relational data model: FK validation and prev_objects ancestry."""

import geopandas as gpd
import pytest
from shapely.geometry import box

from canopyrs.engine.data import Tiles, Objects
from canopyrs.engine.constants import Col, BOX


def _aggregated_from(objects_seed, **columns):
    """An aggregator-style survivor derived from the first seed object (prev_objects link, no own tiles)."""
    return Objects.build(
        geometry=[box(0, 0, 100, 100)], geom_kind=BOX,
        prev_object_id=[objects_seed.df[Col.OBJECT_ID].iloc[0]],
        prev_objects=objects_seed, crs=None, **columns,
    )


def test_column_walks_ancestry(objects_seed):
    survivor = _aggregated_from(objects_seed, **{Col.AGGREGATOR_SCORE: [0.5]})
    # detector_score lives on the ancestor, not the survivor, but is reachable
    assert Col.DETECTOR_SCORE not in survivor.df.columns
    assert survivor.provides(Col.DETECTOR_SCORE)
    assert survivor.column(Col.DETECTOR_SCORE).iloc[0] == 0.9


def test_column_missing_raises(objects_seed):
    with pytest.raises(KeyError):
        objects_seed.column("nonexistent")


def test_linked_resolves_through_ancestry(objects_seed):
    survivor = _aggregated_from(objects_seed)
    # survivor has no direct tiles link; it inherits its ancestor's
    assert survivor.linked("tiles") is objects_seed.tiles


def test_dangling_fk_raises(sources_seed):
    metadata = [{"transform": [1, 0, 0, 0, -1, 0], "crs": "EPSG:32618",
                 "width": 8, "height": 8, "dtype": "uint8", "count": 3, "nodata": None}]
    with pytest.raises(ValueError):
        Tiles.build(source_id=[999], tile_metadata=metadata, sources=sources_seed)  # 999 not in sources


def test_missing_geom_kind_column_raises(tiles_seed):
    tile_id = tiles_seed.df[Col.TILE_ID].iloc[0]
    df = gpd.GeoDataFrame(
        {Col.OBJECT_ID: [0], Col.TILE_ID: [tile_id], Col.GEOMETRY: [box(0, 0, 1, 1)]},
        geometry=Col.GEOMETRY,
    )
    with pytest.raises(ValueError):
        Objects(df, tiles=tiles_seed)   # no geom_kind column at all


def test_bad_geom_kind_raises(tiles_seed):
    tile_id = tiles_seed.df[Col.TILE_ID].iloc[0]
    with pytest.raises(ValueError):
        Objects.build(geometry=[box(0, 0, 1, 1)], geom_kind="banana", tile_id=[tile_id], tiles=tiles_seed)


def test_fresh_ids_are_stamped(objects_seed):
    assert list(objects_seed.df[Col.OBJECT_ID]) == [0, 1]
