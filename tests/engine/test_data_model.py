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


def test_group_by_materialized_source_single_raster(objects_seed):
    """Pixel objects on window tiles bucket into their raster's single file, georeferenced through
    each tile's own affine (tile 0 at CRS origin (0, 64), tile 1 at (64, 64), 1 unit/pixel, north-up)."""
    (path, gdf), = objects_seed.group_by_materialized_source()
    assert path == objects_seed.imagery.resolved_paths().iloc[0]
    assert str(gdf.crs) == "EPSG:32618"
    assert list(gdf[Col.OBJECT_ID]) == list(objects_seed.df[Col.OBJECT_ID])
    assert gdf.geometry.iloc[0].bounds == (1.0, 54.0, 10.0, 63.0)     # box(1, 1, 10, 10) on tile 0
    assert gdf.geometry.iloc[1].bounds == (69.0, 44.0, 84.0, 59.0)    # box(5, 5, 20, 20) on tile 1


def test_group_by_materialized_source_many_files(tmp_path):
    """CRS objects over on-disk tiles bucket once per tile file, geometry passing through unchanged."""
    tiles = Imagery.build(kind=ImageKind.TILE,
                          metadata=[make_tile_metadata(), make_tile_metadata(x0=64.0)],
                          path=[str(tmp_path / "a.tif"), str(tmp_path / "b.tif")])
    geoms = [box(1, 1, 10, 10), box(70, 1, 80, 10)]
    objs = Objects.build(geometry=geoms, geom_kind=GeomKind.BOX,
                         image_id=list(tiles.df[Col.IMAGE_ID]), imagery=tiles, crs="EPSG:32618")
    groups = objs.group_by_materialized_source()
    assert [path for path, _ in groups] == [str(tmp_path / "a.tif"), str(tmp_path / "b.tif")]
    assert all(len(gdf) == 1 for _, gdf in groups)
    assert groups[0][1].geometry.iloc[0].equals(geoms[0])


def test_group_by_materialized_source_needs_imagery(tiles_seed):
    objs = Objects.build(geometry=[box(0, 0, 1, 1)], geom_kind=GeomKind.BOX,
                         image_id=[tiles_seed.df[Col.IMAGE_ID].iloc[0]])
    with pytest.raises(ValueError, match="must be linked"):
        objs.group_by_materialized_source()
