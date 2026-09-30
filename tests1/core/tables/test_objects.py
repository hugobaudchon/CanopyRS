import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from canopyrs1.core.constants import Col, GeomKind
from canopyrs1.core.geometry.georef import make_georef, window_georef
from canopyrs1.core.tables.contracts import Need
from canopyrs1.core.tables.imagery import Crops, Sources, Tiles
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
    """Return (orthos, tiles): the two orthos, on disk, and two 1024 px windows of each, none on
    disk. Tiles 0 and 1 are the top-left and top-right windows of ZONE_18, 2 and 3 of ZONE_19. The
    orthos are two places, so a Tiles table rather than Sources."""
    sources = Tiles.build(georef=[ZONE_18, ZONE_19], path=["zone_18.tif", "zone_19.tif"])
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


# =============================================================================
# from_file
# =============================================================================


def test_boxes_from_a_geopackage(box_labels):
    boxes = Objects.from_file(box_labels)
    assert list(boxes.df[Col.OBJECT_ID]) == [0, 1, 2, 3]
    assert (boxes.df[Col.GEOM_KIND] == GeomKind.BOX).all()
    assert list(boxes.df["class"]) == [0, 0, 0, 0]  # every column of the file is kept
    assert boxes.df.crs == "EPSG:32618"
    assert boxes.parent_imagery is None and boxes.df[Col.PARENT_IMAGE_ID].isna().all()


def test_masks_from_a_geopackage(polygon_labels):
    masks = Objects.from_file(polygon_labels)
    assert (masks.df[Col.GEOM_KIND] == GeomKind.MASK).all()


def test_from_a_geojson(box_labels, tmp_path):
    path = tmp_path / "labels.geojson"
    gpd.read_file(box_labels).to_file(path, driver="GeoJSON")
    assert len(Objects.from_file(path)) == 4


def test_from_a_geodataframe():
    gdf = gpd.GeoDataFrame(
        {"geom": [box(0, 0, 10, 10), Point(5, 5)]},
        geometry="geom",
        crs="EPSG:32618",
    )
    objects = Objects.from_file(gdf)
    assert list(objects.df[Col.GEOM_KIND]) == [GeomKind.BOX, GeomKind.POINT]
    assert objects.df.geometry.name == Col.GEOMETRY
    assert list(gdf.columns) == ["geom"]  # the GeoDataFrame given is left as it was


def test_the_file_s_own_kinds_are_kept():
    gdf = gpd.GeoDataFrame({Col.GEOM_KIND: [GeomKind.MASK]}, geometry=[box(0, 0, 10, 10)])
    assert Objects.from_file(gdf).df[Col.GEOM_KIND][0] == GeomKind.MASK


def test_objects_on_a_single_raster(rgb_raster, box_labels):
    sources = Sources.from_paths(rgb_raster)
    boxes = Objects.from_file(box_labels, parent_imagery=sources)
    assert boxes.parent_imagery is sources
    assert list(boxes.df[Col.PARENT_IMAGE_ID]) == [0, 0, 0, 0]
    # The first box, 10 to 30 m from the raster's bottom-left corner, in its 1 m pixels.
    in_pixels = boxes.get_geometry_in_image_coords(pixels=True)
    assert _same(in_pixels[0], box(10, 226, 30, 246))


def test_objects_on_several_rasters(rgb_raster, box_labels):
    sources = Sources.from_paths([rgb_raster, rgb_raster], timestamp=[0, 1])  # two dates
    with pytest.raises(ValueError, match="need a parent_image_id column"):
        Objects.from_file(box_labels, parent_imagery=sources)
    gdf = gpd.read_file(box_labels)
    gdf[Col.PARENT_IMAGE_ID] = [0, 1, 1, 0]
    boxes = Objects.from_file(gdf, parent_imagery=sources)
    assert list(boxes.df[Col.PARENT_IMAGE_ID]) == [0, 1, 1, 0]
    gdf[Col.PARENT_IMAGE_ID] = [0, 1, 2, 0]
    with pytest.raises(ValueError, match="ids outside its parent Sources"):
        Objects.from_file(gdf, parent_imagery=sources)


UNSUPPORTED_GEOMETRIES = [LineString([(0, 0), (1, 1)]), None]


@pytest.mark.parametrize("geometry", UNSUPPORTED_GEOMETRIES)
def test_geometries_that_aren_t_objects(geometry):
    gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), geometry])
    with pytest.raises(ValueError, match="Expected points or polygons"):
        Objects.from_file(gdf)


def test_an_empty_file(tmp_path):
    path = tmp_path / "empty.gpkg"
    gpd.GeoDataFrame({"class": []}, geometry=[], crs="EPSG:32618").to_file(path)
    assert len(Objects.from_file(path)) == 0


# =============================================================================
# from_imagery
# =============================================================================


def test_one_object_per_crop(tiles_dir):
    crops = Crops.from_image_dir(tiles_dir)
    objects = Objects.from_imagery(crops)
    assert list(objects.df[Col.PARENT_IMAGE_ID]) == [0, 1]
    assert (objects.df[Col.GEOM_KIND] == GeomKind.BOX).all()
    assert objects.parent_imagery is crops and not objects.has_crs
    assert all(_same(g, box(0, 0, 128, 128)) for g in objects.df.geometry)
    # In their images' CRS, each object is its crop's footprint.
    for got, footprint in zip(objects.get_geometry_in_image_coords(), crops.df.geometry):
        assert _same(got, footprint)
    # What the classifier needs.
    assert Need(Objects, links=("parent_imagery",), on=Crops).check(objects.schema()) == []


def test_one_object_per_tile_of_a_raster_without_a_crs():
    # Each box is in its own tile's pixels, not in the raster's frame the tile's footprint is in.
    raster = make_georef(
        transform=[1, 0, 0, 0, 1, 0],
        crs=None,
        width=2048,
        height=2048,
        count=3,
        dtype="uint8",
    )
    tile = window_georef(raster, col_off=1024, row_off=512, width=256, height=128)
    tiles = Tiles.build(georef=[tile])
    objects = Objects.from_imagery(tiles)
    assert _same(objects.df.geometry[0], box(0, 0, 256, 128))
    assert _same(objects.get_geometry_in_image_coords()[0], box(1024, 512, 1280, 640))


def test_one_object_per_image_of_nothing():
    assert len(Objects.from_imagery(Crops.build(georef=[]))) == 0


# =============================================================================
# select
# =============================================================================


def test_select_objects():
    tiles, boxes, _, _ = _history()
    kept = boxes.select([2, 0])
    assert list(kept.df[Col.OBJECT_ID]) == [0, 1]
    assert kept.parent_objects is boxes and list(kept.df[Col.PARENT_OBJECT_ID]) == [2, 0]
    # On the same images.
    assert kept.parent_imagery is tiles and list(kept.df[Col.PARENT_IMAGE_ID]) == [1, 0]
    assert list(kept.get_column(Col.DETECTOR_SCORE)) == [0.7, 0.9]
    assert kept.df.geometry[0].equals(boxes.df.geometry[2])


def test_select_objects_with_new_columns():
    _, boxes, _, _ = _history()
    trimmed = [box(0, 0, 5, 5), box(30, 30, 35, 35)]
    kept = boxes.select(
        [0, 2],
        columns={Col.AGGREGATOR_SCORE: [0.95, 0.6], Col.GEOMETRY: trimmed},
    )
    assert list(kept.df[Col.AGGREGATOR_SCORE]) == [0.95, 0.6]
    assert kept.df.geometry[1].equals(box(30, 30, 35, 35))
    assert Col.AGGREGATOR_SCORE not in boxes.df.columns  # left as they were
    assert boxes.df.geometry[0].equals(box(0, 0, 10, 10))


def test_select_objects_checks_them_like_build():
    _, boxes, _, _ = _history()
    with pytest.raises(ValueError, match="Unknown geom_kind"):
        boxes.select([0], columns={Col.GEOM_KIND: "polygon"})
    with pytest.raises(ValueError, match="has 2 values for 1 rows"):
        boxes.select([0], columns={Col.SCORE: [0.1, 0.2]})


def test_select_objects_onto_other_images():
    _, boxes, _, _ = _history()
    crops = _images(3)  # one crop per box
    on_crops = boxes.select([0, 1, 2], parent_image_id=[2, 0, 1], parent_imagery=crops)
    assert on_crops.parent_imagery is crops and on_crops.parent_objects is boxes
    assert list(on_crops.df[Col.PARENT_IMAGE_ID]) == [2, 0, 1]
    assert list(on_crops.get_column(Col.DETECTOR_SCORE)) == [0.9, 0.8, 0.7]
    # The new ids are checked against the new images.
    with pytest.raises(ValueError, match="parent_image_id has ids outside its parent Images"):
        boxes.select([0], parent_image_id=[5], parent_imagery=crops)


def test_select_onto_other_images_needs_both():
    _, boxes, _, _ = _history()
    with pytest.raises(ValueError, match="Give both parent_image_id and parent_imagery"):
        boxes.select([0], parent_image_id=[0])
    with pytest.raises(ValueError, match="Give both parent_image_id and parent_imagery"):
        boxes.select([0], parent_imagery=_images())


def test_select_pixel_objects_onto_images_with_another_georef():
    # Objects in pixel coordinates are relative to their image: moved as they are onto a window
    # 50 px to the right, they land 50 px further right on the ground. The caller has to give their
    # geometry in the new images' pixels, or keep them in a CRS.
    tile = _raster("EPSG:32618", 700000)
    window = window_georef(tile, col_off=500, row_off=0, width=1000, height=1000)
    tiles = Tiles.build(georef=[tile])
    windows = Tiles.build(georef=[window])
    boxes = Objects.build(
        geometry=[box(600, 10, 610, 20)],
        geom_kind=GeomKind.BOX,
        parent_image_id=0,
        parent_imagery=tiles,
    )
    moved_as_is = boxes.select([0], parent_image_id=0, parent_imagery=windows)
    before = boxes.get_geometry_in_image_coords()[0]
    after = moved_as_is.get_geometry_in_image_coords()[0]
    assert after.bounds[0] - before.bounds[0] == pytest.approx(50)  # 500 px of 0.1 m
    # Given in the window's pixels, it stays where it was.
    in_window = boxes.select(
        [0],
        columns={Col.GEOMETRY: [box(100, 10, 110, 20)]},
        parent_image_id=0,
        parent_imagery=windows,
    )
    assert _same(in_window.get_geometry_in_image_coords()[0], before)
