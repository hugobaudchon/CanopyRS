"""Unit tests for the relational data model: FK validation, prev_objects ancestry, and the imagery
containment tree (nearest materialized ancestor)."""

import geopandas as gpd
import pytest
from shapely.geometry import box

from canopyrs.engine.data import Imagery, Objects
from canopyrs.engine.constants import Col, GeomKind, ImageKind
from tests.conftest import make_tile_metadata


def _aggregated_from(objects_seed, **columns):
    """An aggregator-style survivor derived from the first seed object (prev_objects link, no own imagery)."""
    return Objects.build(
        geometry=[box(0, 0, 100, 100)], geom_kind=GeomKind.BOX,
        prev_object_id=[objects_seed.df[Col.OBJECT_ID].iloc[0]],
        prev_objects=objects_seed, crs=None, **columns,
    )


# --- Objects lineage ----------------------------------------------------------

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
    # survivor has no direct imagery link; it inherits its ancestor's
    assert survivor.linked("imagery") is objects_seed.imagery


def test_dangling_fk_raises(sources_seed):
    with pytest.raises(ValueError):
        Imagery.build(kind=ImageKind.TILE, parent_id=[999],
                      metadata=[make_tile_metadata()], parent=sources_seed)   # 999 not in sources


def test_missing_geom_kind_column_raises(tiles_seed):
    image_id = tiles_seed.df[Col.IMAGE_ID].iloc[0]
    df = gpd.GeoDataFrame(
        {Col.OBJECT_ID: [0], Col.IMAGE_ID: [image_id], Col.GEOMETRY: [box(0, 0, 1, 1)]},
        geometry=Col.GEOMETRY,
    )
    with pytest.raises(ValueError):
        Objects(df, imagery=tiles_seed)   # no geom_kind column at all


def test_bad_geom_kind_raises(tiles_seed):
    image_id = tiles_seed.df[Col.IMAGE_ID].iloc[0]
    with pytest.raises(ValueError):
        Objects.build(geometry=[box(0, 0, 1, 1)], geom_kind="banana",
                      image_id=[image_id], imagery=tiles_seed)


def test_fresh_ids_are_stamped(objects_seed):
    assert list(objects_seed.df[Col.OBJECT_ID]) == [0, 1]


# --- Imagery containment tree --------------------------------------------------

def test_kind_and_modalities(sources_seed, tiles_seed):
    assert sources_seed.kind == ImageKind.SOURCE
    assert tiles_seed.kind == ImageKind.TILE
    assert sources_seed.modalities == {"rgb"}


def test_resolved_paths_walk_to_materialized_ancestor(sources_seed, tiles_seed):
    """Window tiles (no own path) resolve to their source raster's file; a crop of a window tile
    resolves through both levels to the same file."""
    source_path = sources_seed.df[Col.PATH].iloc[0]
    assert list(tiles_seed.resolved_paths()) == [source_path, source_path]

    crop = Imagery.build(kind=ImageKind.TILE,
                         parent_id=tiles_seed.df[Col.IMAGE_ID].iloc[0],
                         metadata=[make_tile_metadata(width=8, height=8)],
                         parent=tiles_seed)
    assert crop.resolved_paths().iloc[0] == source_path


def test_resolved_paths_prefer_own_file(sources_seed):
    """A materialized tile reads from its own file, not its ancestor's."""
    tile = Imagery.build(kind=ImageKind.TILE,
                         parent_id=sources_seed.df[Col.IMAGE_ID].iloc[0],
                         metadata=[make_tile_metadata()],
                         path=["/on/disk/tile_0.tif"],
                         parent=sources_seed)
    assert tile.resolved_paths().iloc[0] == "/on/disk/tile_0.tif"


def test_reading_frame_carries_read_path(tiles_seed):
    from canopyrs.engine.data import READ_PATH
    frame = tiles_seed.reading_frame()
    assert READ_PATH in frame.columns
    assert frame[READ_PATH].notna().all()   # windows resolve to the source raster
