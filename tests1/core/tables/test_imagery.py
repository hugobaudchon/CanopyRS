from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.errors import RasterioIOError
from shapely.geometry import box

from canopyrs1.core.constants import RGB_BANDS, Col, GeomKind, Modality
from canopyrs1.core.geometry.georef import (
    get_footprint,
    make_georef,
    read_georef,
    window_georef,
)
from canopyrs1.core.tables.imagery import Crops, Imagery, Sources, Tiles
from canopyrs1.core.tables.objects import Objects

RASTER = make_georef(
    transform=[0.1, 0, 600000, 0, -0.1, 5040000],
    crs="EPSG:32618",
    width=2048,
    height=2048,
    count=3,
    dtype="uint8",
)


def _sources():
    return Sources.build(georef=[RASTER], path="ortho.tif")


def _tiles(sources):
    """Four 1024 px windows of the source, none on disk."""
    georefs = [
        window_georef(RASTER, col_off=col, row_off=row, width=1024, height=1024)
        for row in (0, 1024)
        for col in (0, 1024)
    ]
    return Tiles.build(georef=georefs, parent_image_id=0, parent_imagery=sources)


def test_a_source():
    sources = _sources()
    row = sources.df.iloc[0]
    assert row[Col.IMAGE_ID] == 0 and row[Col.PATH] == "ortho.tif"
    assert row[Col.GEOREF] == RASTER
    assert row[Col.BANDS] == RGB_BANDS
    assert row[Col.MODALITY] == Modality.RGB
    assert row[Col.PARENT_IMAGE_ID] is None and row[Col.TIMESTAMP] is None
    assert sources.parent_imagery is None


def test_tiles_are_windows_of_their_source():
    sources = _sources()
    tiles = _tiles(sources)
    assert list(tiles.df[Col.IMAGE_ID]) == [0, 1, 2, 3]
    assert list(tiles.df[Col.PARENT_IMAGE_ID]) == [0, 0, 0, 0]
    assert tiles.df[Col.PATH].isna().all()
    assert tiles.parent_imagery is sources


def test_one_value_or_one_per_image():
    tiles = Tiles.build(
        georef=[RASTER, RASTER],
        path=["a.tif", "b.tif"],
        modality=[Modality.RGB, Modality.THERMAL],
        timestamp="2024-06-13",
        columns={Col.INSTANCE_ID: [7, 7]},
    )
    assert list(tiles.df[Col.PATH]) == ["a.tif", "b.tif"]
    assert list(tiles.df[Col.MODALITY]) == [Modality.RGB, Modality.THERMAL]
    assert list(tiles.df[Col.TIMESTAMP]) == ["2024-06-13"] * 2
    assert list(tiles.df[Col.INSTANCE_ID]) == [7, 7]


def test_bands():
    tiles = Tiles.build(georef=[RASTER, RASTER], bands=(4, 5, 6))
    assert list(tiles.df[Col.BANDS]) == [[4, 5, 6], [4, 5, 6]]


def test_build_nothing():
    empty = Tiles.build(georef=[])
    assert len(empty) == 0
    assert {Col.IMAGE_ID, Col.PARENT_IMAGE_ID, Col.GEOREF, Col.PATH} <= set(empty.df.columns)


def test_crops_can_come_from_tiles_or_from_a_source():
    sources = _sources()
    tiles = _tiles(sources)
    crop = window_georef(RASTER, col_off=100, row_off=100, width=64, height=64)
    from_tiles = Crops.build(georef=[crop], parent_image_id=3, parent_imagery=tiles)
    from_source = Crops.build(georef=[crop], parent_image_id=0, parent_imagery=sources)
    assert from_tiles.parent_imagery is tiles and from_source.parent_imagery is sources


def test_a_parent_id_outside_the_parent_imagery():
    with pytest.raises(
        ValueError, match="Tiles.parent_image_id has ids outside its parent Sources"
    ):
        Tiles.build(georef=[RASTER], parent_image_id=1, parent_imagery=_sources())


def test_the_roles_are_separate_types_sharing_imagery():
    sources = _sources()
    tiles = _tiles(sources)
    assert isinstance(sources, Imagery) and isinstance(tiles, Imagery)
    assert not isinstance(tiles, Sources) and not isinstance(sources, Tiles)


def test_objects_found_on_tiles():
    tiles = _tiles(_sources())
    boxes = Objects.build(
        geometry=[box(0, 0, 10, 10), box(5, 5, 20, 20)],
        geom_kind=GeomKind.BOX,
        parent_image_id=[1, 3],
        parent_imagery=tiles,
    )
    assert boxes.get_parent_imagery() is tiles
    with pytest.raises(ValueError, match="parent_image_id has ids outside its parent Tiles"):
        Objects.build(
            geometry=[box(0, 0, 1, 1)],
            geom_kind=GeomKind.BOX,
            parent_image_id=[4],
            parent_imagery=tiles,
        )


# =============================================================================
# get_disk_paths and get_ancestor_ids
# =============================================================================


def _crops(tiles):
    """Three crops: two of tile 1 and one of tile 3, none on disk."""
    crop = window_georef(RASTER, col_off=100, row_off=100, width=64, height=64)
    return Crops.build(georef=[crop] * 3, parent_image_id=[1, 1, 3], parent_imagery=tiles)


def test_disk_paths_of_windows_are_their_nearest_parent_file():
    sources = _sources()
    tiles = _tiles(sources)
    assert list(sources.get_disk_paths()) == ["ortho.tif"]
    assert list(tiles.get_disk_paths()) == ["ortho.tif"] * 4
    assert list(_crops(tiles).get_disk_paths()) == ["ortho.tif"] * 3  # two steps up


def test_an_image_on_disk_uses_its_own_file():
    sources = _sources()
    tiles = Tiles.build(
        georef=[RASTER] * 3,
        path=["tile_0.tif", None, "tile_2.tif"],
        parent_image_id=0,
        parent_imagery=sources,
    )
    assert list(tiles.get_disk_paths()) == ["tile_0.tif", "ortho.tif", "tile_2.tif"]
    crops = Crops.build(georef=[RASTER] * 2, parent_image_id=[2, 1], parent_imagery=tiles)
    assert list(crops.get_disk_paths()) == ["tile_2.tif", "ortho.tif"]


def test_a_window_without_a_parent_on_disk_has_no_path():
    windows = Tiles.build(georef=[RASTER])  # no path, no parent
    assert windows.get_disk_paths().isna().all()


def test_get_ancestor():
    sources = _sources()
    tiles = _tiles(sources)
    crops = _crops(tiles)
    assert crops.get_ancestor(Crops) is crops
    assert crops.get_ancestor(Tiles) is tiles
    assert crops.get_ancestor(Sources) is sources  # two steps up
    with pytest.raises(ValueError, match="No Crops above this Tiles table"):
        tiles.get_ancestor(Crops)


def test_get_ancestor_ids():
    sources = _sources()
    tiles = _tiles(sources)
    crops = _crops(tiles)
    assert list(crops.get_ancestor_ids([0, 1, 2], Crops)) == [0, 1, 2]
    assert list(crops.get_ancestor_ids([0, 1, 2], Tiles)) == [1, 1, 3]
    assert list(crops.get_ancestor_ids([2, 0], Sources)) == [0, 0]  # two steps up
    assert list(tiles.get_ancestor_ids([3, 1], Tiles)) == [3, 1]
    with pytest.raises(ValueError, match="No Crops above this Tiles table"):
        tiles.get_ancestor_ids([0], Crops)


# =============================================================================
# Footprints
# =============================================================================


def test_each_image_s_geometry_is_its_footprint():
    sources = _sources()
    tiles = _tiles(sources)
    assert sources.df.crs == "EPSG:32618" and tiles.has_crs
    assert sources.df.geometry[0].equals(get_footprint(RASTER))
    for geometry, georef in zip(tiles.df.geometry, tiles.df[Col.GEOREF]):
        assert geometry.equals(get_footprint(georef))
    # The four tiles cover the source exactly (up to floating-point rounding along their seams).
    uncovered = tiles.df.geometry.union_all().symmetric_difference(sources.df.geometry[0])
    assert uncovered.area < 1e-6


def test_footprints_in_another_crs_are_moved_into_the_first_image_s():
    # Two rasters on either side of the UTM 18N / 19N line, near Montreal.
    zone_18 = make_georef(
        transform=[1, 0, 700000, 0, -1, 5040000],
        crs="EPSG:32618",
        width=100,
        height=100,
        count=3,
        dtype="uint8",
    )
    zone_19 = make_georef(
        transform=[1, 0, 230000, 0, -1, 5040000],
        crs="EPSG:32619",
        width=100,
        height=100,
        count=3,
        dtype="uint8",
    )
    sources = Sources.build(georef=[zone_18, zone_19], path=["a.tif", "b.tif"])
    assert sources.df.crs == "EPSG:32618"
    expected = gpd.GeoSeries([get_footprint(zone_19)], crs="EPSG:32619").to_crs("EPSG:32618")[0]
    assert sources.df.geometry[1].equals_exact(expected, 1e-6)
    assert sources.df[Col.GEOREF][1]["crs"] == "EPSG:32619"  # the georef itself is unchanged


def test_images_without_a_crs():
    photo = make_georef(
        transform=[1, 0, 0, 0, 1, 0],
        crs=None,
        width=64,
        height=32,
        count=3,
        dtype="uint8",
    )
    photos = Tiles.build(georef=[photo, photo], path=["a.jpg", "b.jpg"])
    assert not photos.has_crs
    assert photos.df.geometry[0].equals(box(0, 0, 64, 32))  # in pixel coordinates
    with pytest.raises(ValueError, match="with and without a CRS"):
        Tiles.build(georef=[RASTER, photo])


def test_which_tiles_cover_an_area():
    tiles = _tiles(_sources())
    # A 10 m box around the centre of the source touches all four tiles; one near a corner, one.
    centre = box(600102.4 - 5, 5039897.6 - 5, 600102.4 + 5, 5039897.6 + 5)
    corner = box(600001, 5039990, 600005, 5039995)
    assert sorted(tiles.df.sindex.query(centre, predicate="intersects")) == [0, 1, 2, 3]
    assert list(tiles.df.sindex.query(corner, predicate="intersects")) == [0]


# =============================================================================
# from_paths
# =============================================================================


def test_sources_from_one_path(rgb_raster):
    sources = Sources.from_paths(rgb_raster)
    assert isinstance(sources, Sources) and len(sources) == 1
    row = sources.df.iloc[0]
    assert row[Col.PATH] == str(rgb_raster)
    with rasterio.open(rgb_raster) as src:
        assert row[Col.GEOREF] == read_georef(src)
    assert row[Col.MODALITY] == Modality.RGB and row[Col.BANDS] == RGB_BANDS
    assert sources.df.geometry[0].equals(box(0, 0, 256, 256))  # its footprint, in its CRS
    assert sources.parent_imagery is None


def test_sources_from_several_paths(rgb_raster, unprojected_raster):
    sources = Sources.from_paths(
        [Path(rgb_raster), str(unprojected_raster)],
        modality=[Modality.RGB, Modality.RGB],
        timestamp=[0, 1],
    )
    assert list(sources.df[Col.PATH]) == [str(rgb_raster), str(unprojected_raster)]
    assert list(sources.df[Col.TIMESTAMP]) == [0, 1]
    # Each georef stays in its file's own CRS; the footprints are all in the first one's.
    assert sources.df[Col.GEOREF][1]["crs"] == "EPSG:4326"
    assert sources.df.crs == "EPSG:32618"


def test_from_paths_with_the_real_crop(real_raster):
    sources = Sources.from_paths(real_raster)
    with rasterio.open(real_raster) as src:
        assert sources.df[Col.GEOREF][0] == read_georef(src)
        assert sources.df.crs == src.crs.to_string()


def test_from_paths_of_a_raster_without_a_crs(tmp_path):
    path = tmp_path / "photo.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=64,
        height=32,
        count=3,
        dtype="uint8",
        transform=Affine.identity(),
    ) as dst:
        dst.write(np.zeros((3, 32, 64), dtype=np.uint8))
    tiles = Tiles.from_paths(path)
    assert isinstance(tiles, Tiles) and not tiles.has_crs
    assert tiles.df.geometry[0].equals(box(0, 0, 64, 32))


def test_from_paths_of_nothing():
    assert len(Sources.from_paths([])) == 0


def test_from_paths_of_a_missing_file(tmp_path):
    with pytest.raises(RasterioIOError):
        Sources.from_paths(tmp_path / "missing.tif")


# =============================================================================
# from_image_dir
# =============================================================================


def test_tiles_from_a_folder(tiles_dir):
    tiles = Tiles.from_image_dir(tiles_dir)
    assert isinstance(tiles, Tiles) and tiles.parent_imagery is None
    assert list(tiles.df[Col.PATH]) == [
        str(tiles_dir / "tile_0.tif"),
        str(tiles_dir / "tile_1.tif"),
    ]
    for path, georef in zip(tiles.df[Col.PATH], tiles.df[Col.GEOREF]):
        with rasterio.open(path) as src:
            assert georef == read_georef(src)
    # The two tiles are the top-left and top-right quarters of their raster.
    assert tiles.df.geometry[0].equals(box(0, 128, 128, 256))
    assert tiles.df.geometry[1].equals(box(128, 128, 256, 256))


def test_crops_from_a_folder(tiles_dir):
    crops = Crops.from_image_dir(tiles_dir, timestamp=3)
    assert isinstance(crops, Crops) and len(crops) == 2
    assert list(crops.df[Col.TIMESTAMP]) == [3, 3]


def test_only_geotiffs_directly_inside_the_folder(tiles_dir):
    (tiles_dir / "tile_1.tif").rename(tiles_dir / "tile_1.TIFF")
    (tiles_dir / "coco.json").write_text("{}")
    (tiles_dir / "nested").mkdir()
    (tiles_dir / "folder.tif").mkdir()
    names = [Path(p).name for p in Tiles.from_image_dir(tiles_dir).df[Col.PATH]]
    assert names == ["tile_0.tif", "tile_1.TIFF"]


def test_sorted_by_file_name(tiles_dir):
    (tiles_dir / "tile_1.tif").rename(tiles_dir / "a.tif")
    names = [Path(p).name for p in Tiles.from_image_dir(tiles_dir).df[Col.PATH]]
    assert names == ["a.tif", "tile_0.tif"]


def test_a_folder_without_images(tmp_path):
    (tmp_path / "notes.txt").write_text("")
    with pytest.raises(ValueError, match="No .tif or .tiff images in"):
        Tiles.from_image_dir(tmp_path)
