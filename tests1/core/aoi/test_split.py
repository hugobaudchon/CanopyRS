import geopandas as gpd
import pytest
from shapely.geometry import box

from canopyrs1.core.aoi.assign import assign_to_aois
from canopyrs1.core.aoi.load import load_aois
from canopyrs1.core.aoi.split import split_by_aoi
from canopyrs1.core.constants import Col, GeomKind
from canopyrs1.core.geometry.georef import make_georef, window_georef
from canopyrs1.core.tables.imagery import Sources, Tiles
from canopyrs1.core.tables.objects import Objects

UTM_18N = "EPSG:32618"

# A 400 x 100 m raster in 1 m pixels, whose top-left corner is at (0, 100).
RASTER = make_georef(
    transform=[1, 0, 0, 0, -1, 100],
    crs=UTM_18N,
    width=400,
    height=100,
    count=3,
    dtype="uint8",
)


def _assigned():
    """Return the four 100 m tiles of RASTER assigned to 'train' (x 0 to 150) and 'valid'
    (x 150 to 400): tile 1 is on the border, so it is in both."""
    sources = Sources.build(georef=[RASTER], path="ortho.tif")
    georefs = [
        window_georef(RASTER, col_off=c, row_off=0, width=100, height=100)
        for c in range(0, 400, 100)
    ]
    tiles = Tiles.build(georef=georefs, parent_image_id=0, parent_imagery=sources)
    areas = {"train": box(0, 0, 150, 100), "valid": box(150, 0, 400, 100)}
    aois = load_aois(
        {name: gpd.GeoDataFrame(geometry=[area], crs=UTM_18N) for name, area in areas.items()},
        UTM_18N,
    )
    return assign_to_aois(tiles, aois)  # train: tiles 0, 1; valid: tiles 1, 2, 3


def test_split_tiles():
    assigned = _assigned()
    folds = split_by_aoi(assigned)
    assert list(folds) == ["train", "valid"]
    train, valid = folds["train"][0], folds["valid"][0]
    assert len(train) == 2 and len(valid) == 3
    assert folds["train"][1] is None  # no objects given
    # Each fold's tiles are windows of the copies they came from.
    assert train.parent_imagery is assigned and list(train.df[Col.PARENT_IMAGE_ID]) == [0, 1]
    assert list(valid.df[Col.PARENT_IMAGE_ID]) == [2, 3, 4]
    assert set(train.df[Col.AOI]) == {"train"} and set(valid.df[Col.AOI]) == {"valid"}
    assert list(valid.get_disk_paths()) == ["ortho.tif"] * 3
    # The border tile's copies keep their cut.
    assert train.df.geometry[1].equals(box(100, 0, 150, 100))
    assert valid.df.geometry[0].equals(box(150, 0, 200, 100))


def test_split_objects_with_their_tile():
    assigned = _assigned()
    # One object on each copy: on train's tile 0 and 1, on valid's tile 1, 2 and 3.
    objects = Objects.build(
        geometry=[box(0, 0, 5, 5)] * 5,
        geom_kind=GeomKind.BOX,
        parent_image_id=[4, 0, 2, 1, 3],
        parent_imagery=assigned,
        columns={Col.DETECTOR_SCORE: [0.1, 0.2, 0.3, 0.4, 0.5]},
    )
    folds = split_by_aoi(assigned, objects)
    train_tiles, train_objects = folds["train"]
    valid_tiles, valid_objects = folds["valid"]

    # Each fold's objects are on its own tiles, pointing to the objects they came from.
    assert train_objects.parent_imagery is train_tiles and train_objects.parent_objects is objects
    assert list(train_objects.df[Col.PARENT_OBJECT_ID]) == [1, 3]
    assert list(train_objects.df[Col.PARENT_IMAGE_ID]) == [0, 1]
    assert list(valid_objects.df[Col.PARENT_OBJECT_ID]) == [0, 2, 4]
    assert list(valid_objects.df[Col.PARENT_IMAGE_ID]) == [2, 0, 1]
    assert list(valid_objects.get_column(Col.DETECTOR_SCORE)) == [0.1, 0.3, 0.5]
    # Every object is in exactly one fold.
    assert len(train_objects) + len(valid_objects) == len(objects)


def test_a_fold_without_objects():
    assigned = _assigned()
    objects = Objects.build(
        geometry=[box(0, 0, 5, 5)],
        geom_kind=GeomKind.BOX,
        parent_image_id=[0],
        parent_imagery=assigned,
    )
    folds = split_by_aoi(assigned, objects)
    assert len(folds["train"][1]) == 1 and len(folds["valid"][1]) == 0


def test_objects_on_other_tiles():
    assigned = _assigned()
    other = Objects.build(geometry=[box(0, 0, 1, 1)], geom_kind=GeomKind.BOX)
    with pytest.raises(ValueError, match="must be on the tiles to split"):
        split_by_aoi(assigned, other)


def test_no_tile():
    assert split_by_aoi(_assigned().select([])) == {}


def test_the_real_crop(real_raster):
    # Two windows of the real crop, split by AOIs covering its left and right halves.
    sources = Sources.from_paths(real_raster)
    georef = sources.df[Col.GEOREF][0]
    half = georef["width"] // 2
    windows = [
        window_georef(georef, col_off=c, row_off=0, width=half, height=georef["height"])
        for c in (0, half)
    ]
    tiles = Tiles.build(georef=windows, parent_image_id=0, parent_imagery=sources)
    minx, miny, maxx, maxy = sources.df.geometry[0].bounds
    middle = (minx + maxx) / 2
    areas = {"left": box(minx, miny, middle, maxy), "right": box(middle, miny, maxx, maxy)}
    aois = load_aois(
        {name: gpd.GeoDataFrame(geometry=[a], crs=tiles.df.crs) for name, a in areas.items()},
        tiles.df.crs,
    )
    assigned = assign_to_aois(tiles, aois)
    folds = split_by_aoi(assigned, Objects.from_imagery(assigned))
    assert list(folds) == ["left", "right"]
    for fold_tiles, fold_objects in folds.values():
        assert len(fold_objects) == len(fold_tiles)
        assert list(fold_tiles.get_disk_paths()) == [str(real_raster)] * len(fold_tiles)
