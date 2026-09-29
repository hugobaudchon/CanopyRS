import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from canopyrs1.core.constants import Col, GeomKind
from canopyrs1.core.geometry.georef import make_georef, window_georef
from canopyrs1.core.tables.imagery import Sources, Tiles
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


# =============================================================================
# get_geometry_in_image_coords and group_by_disk_path
# =============================================================================


def _raster(crs, left):
    """A 2048 px raster at 0.1 m, its top-left corner at (``left``, 5040000) in ``crs``."""
    return make_georef(
        transform=[0.1, 0, left, 0, -0.1, 5040000],
        crs=crs,
        width=2048,
        height=2048,
        count=3,
        dtype="uint8",
    )


def _same(a, b):
    """Whether geometries ``a`` and ``b`` are the same, up to floating-point rounding."""
    return a.hausdorff_distance(b) < 1e-6


# Two orthos on either side of the UTM 18N / 19N line, near Montreal.
ZONE_18 = _raster("EPSG:32618", 700000)
ZONE_19 = _raster("EPSG:32619", 230000)


def _two_zones():
    """Return (sources, tiles): the two orthos, and two 1024 px windows of each, none on disk.
    Tiles 0 and 1 are the top-left and top-right windows of ZONE_18, 2 and 3 of ZONE_19."""
    sources = Sources.build(georef=[ZONE_18, ZONE_19], path=["zone_18.tif", "zone_19.tif"])
    georefs = [
        window_georef(raster, col_off=col, row_off=0, width=1024, height=1024)
        for raster in (ZONE_18, ZONE_19)
        for col in (0, 1024)
    ]
    tiles = Tiles.build(georef=georefs, parent_image_id=[0, 0, 1, 1], parent_imagery=sources)
    return sources, tiles


def _detections(tiles):
    """One 10 px box at pixel (100, 100) of each tile, in the tile's pixel coordinates, listed
    with the tiles of both orthos interleaved."""
    return Objects.build(
        geometry=[box(100, 100, 110, 110)] * 4,
        geom_kind=GeomKind.BOX,
        parent_image_id=[0, 2, 1, 3],
        parent_imagery=tiles,
    )


def test_pixel_objects_into_their_image_s_crs():
    _, tiles = _two_zones()
    geometry = _detections(tiles).get_geometry_in_image_coords()
    assert geometry.crs is None  # the images are in two CRSs
    # Pixel (100, 100) is 10 m right of and below each tile's top-left corner; tile 1 (row 2)
    # starts 102.4 m right of its ortho's.
    assert _same(geometry[0], box(700010, 5039989, 700011, 5039990))
    assert _same(geometry[1], box(230010, 5039989, 230011, 5039990))
    assert _same(geometry[2], box(700112.4, 5039989, 700113.4, 5039990))


def test_pixel_objects_in_pixels_are_unchanged():
    _, tiles = _two_zones()
    geometry = _detections(tiles).get_geometry_in_image_coords(pixels=True)
    assert all(_same(g, box(100, 100, 110, 110)) for g in geometry)


def test_crs_objects_into_their_image_s_crs_and_pixels():
    _, tiles = _two_zones()
    in_images = _detections(tiles).get_geometry_in_image_coords()
    # The same boxes in one CRS for the whole table, zone 18N: the zone 19N ones are reprojected.
    zone_19 = gpd.GeoSeries(in_images[[1, 3]].to_list(), crs="EPSG:32619").to_crs("EPSG:32618")
    moved = [*in_images[[0, 2]], *zone_19]
    in_zone_18 = Objects.build(
        geometry=moved,
        geom_kind=GeomKind.BOX,
        crs="EPSG:32618",
        parent_image_id=[0, 1, 2, 3],
        parent_imagery=tiles,
    )
    back = in_zone_18.get_geometry_in_image_coords()
    for got, expected in zip(back, in_images[[0, 2, 1, 3]]):
        assert _same(got, expected)
    for got in in_zone_18.get_geometry_in_image_coords(pixels=True):
        assert _same(got, box(100, 100, 110, 110))


def test_objects_without_an_image():
    _, tiles = _two_zones()
    lone = Objects.build(geometry=[Point(0, 0)], geom_kind=GeomKind.POINT)
    missing = Objects.build(
        geometry=[Point(0, 0)] * 2,
        geom_kind=GeomKind.POINT,
        parent_image_id=[0, None],
        parent_imagery=tiles,
    )
    for objects in (lone, missing):
        with pytest.raises(ValueError, match="Every object needs its image"):
            objects.get_geometry_in_image_coords()
        with pytest.raises(ValueError, match="Every object needs its image"):
            objects.group_by_disk_path()


def test_group_by_disk_path():
    _, tiles = _two_zones()
    groups = _detections(tiles).group_by_disk_path()
    assert [path for path, _ in groups] == ["zone_18.tif", "zone_19.tif"]  # first seen first
    (_, zone_18), (_, zone_19) = groups
    assert zone_18.crs == "EPSG:32618" and zone_19.crs == "EPSG:32619"
    assert list(zone_18[Col.OBJECT_ID]) == [0, 2] and list(zone_18[Col.PARENT_IMAGE_ID]) == [0, 1]
    assert list(zone_19[Col.OBJECT_ID]) == [1, 3] and list(zone_19[Col.PARENT_IMAGE_ID]) == [2, 3]
    assert _same(zone_18.geometry[0], box(700010, 5039989, 700011, 5039990))
    assert _same(zone_19.geometry[1], box(230010, 5039989, 230011, 5039990))


def test_group_by_disk_path_with_tiles_on_disk():
    sources, _ = _two_zones()
    tiles = Tiles.build(
        georef=[window_georef(ZONE_18, col_off=0, row_off=0, width=1024, height=1024)] * 2,
        path=["tile_0.tif", None],
        parent_image_id=0,
        parent_imagery=sources,
    )
    objects = Objects.build(
        geometry=[box(0, 0, 1, 1)] * 3,
        geom_kind=GeomKind.BOX,
        parent_image_id=[1, 0, 1],
        parent_imagery=tiles,
    )
    groups = objects.group_by_disk_path()
    assert [(path, list(gdf[Col.OBJECT_ID])) for path, gdf in groups] == [
        ("zone_18.tif", [0, 2]),
        ("tile_0.tif", [1]),
    ]


def test_group_by_disk_path_without_a_file_on_disk():
    windows = Tiles.build(georef=[ZONE_18])  # no path, no parent
    objects = Objects.build(
        geometry=[box(0, 0, 1, 1)],
        geom_kind=GeomKind.BOX,
        parent_image_id=0,
        parent_imagery=windows,
    )
    with pytest.raises(ValueError, match="no file on disk above their image"):
        objects.group_by_disk_path()


def test_group_by_disk_path_of_nothing():
    _, tiles = _two_zones()
    empty = Objects.build(geometry=[], geom_kind=GeomKind.BOX, parent_imagery=tiles)
    assert empty.group_by_disk_path() == []


def test_tiles_of_a_raster_without_a_crs():
    # Without a CRS, an image's coordinates are its raster's own frame: here, its pixels.
    raster = make_georef(
        transform=[1, 0, 0, 0, 1, 0],
        crs=None,
        width=2048,
        height=2048,
        count=3,
        dtype="uint8",
    )
    sources = Sources.build(georef=[raster], path="no_crs.tif")
    tile = window_georef(raster, col_off=1024, row_off=512, width=1024, height=1024)
    tiles = Tiles.build(georef=[tile], parent_image_id=0, parent_imagery=sources)
    objects = Objects.build(
        geometry=[box(100, 100, 110, 110)],
        geom_kind=GeomKind.BOX,
        parent_image_id=0,
        parent_imagery=tiles,
    )
    in_raster = box(1124, 612, 1134, 622)
    assert _same(objects.get_geometry_in_image_coords()[0], in_raster)
    [(path, gdf)] = objects.group_by_disk_path()
    assert path == "no_crs.tif" and gdf.crs is None
    assert _same(gdf.geometry[0], in_raster)
