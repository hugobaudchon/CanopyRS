import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box

from canopyrs1.core.aoi.assign import assign_to_aois, get_usable_mask
from canopyrs1.core.aoi.load import load_aois
from canopyrs1.core.constants import Col
from canopyrs1.core.geometry.georef import get_footprint, make_georef, window_georef
from canopyrs1.core.tables.imagery import Crops, Sources, Tiles

UTM_18N = "EPSG:32618"

# A 400 x 200 m raster in 1 m pixels, whose top-left corner is at (0, 200).
RASTER = make_georef(
    transform=[1, 0, 0, 0, -1, 200],
    crs=UTM_18N,
    width=400,
    height=200,
    count=3,
    dtype="uint8",
)


def _tiles():
    """Return four 100 x 100 m tiles side by side along the top of RASTER, x from 0 to 400."""
    sources = Sources.build(georef=[RASTER], path="ortho.tif")
    georefs = [
        window_georef(RASTER, col_off=c, row_off=0, width=100, height=100)
        for c in range(0, 400, 100)
    ]
    return Tiles.build(
        georef=georefs,
        parent_image_id=0,
        parent_imagery=sources,
        timestamp=3,
        columns={Col.LAZY_CONDITIONS: [{"ignore_black_white_alpha_tiles_threshold": 0.8}] * 4},
    )


def _aois(**areas):
    return load_aois(
        {name: gpd.GeoDataFrame(geometry=[a], crs=UTM_18N) for name, a in areas.items()}, UTM_18N
    )


def test_tiles_inside_an_aoi():
    tiles = _tiles()
    assigned = assign_to_aois(tiles, _aois(train=box(0, 0, 200, 200)))
    assert list(assigned.df[Col.AOI]) == ["train", "train"]
    assert list(assigned.df[Col.PARENT_IMAGE_ID]) == [0, 1]
    # Wholly inside: their geometry is their whole footprint.
    for geometry, georef in zip(assigned.df.geometry, assigned.df[Col.GEOREF]):
        assert geometry.equals(get_footprint(georef))


def test_a_tile_across_two_aois_is_in_both_cut_to_each():
    # The cut runs through the middle of the second tile (x 100 to 200).
    assigned = assign_to_aois(
        _tiles(), _aois(train=box(0, 0, 150, 200), valid=box(150, 0, 400, 200))
    )
    assert list(zip(assigned.df[Col.AOI], assigned.df[Col.PARENT_IMAGE_ID])) == [
        ("train", 0),
        ("train", 1),
        ("valid", 1),
        ("valid", 2),
        ("valid", 3),
    ]
    assert assigned.df.geometry[1].equals(box(100, 100, 150, 200))
    assert assigned.df.geometry[2].equals(box(150, 100, 200, 200))
    # Both copies keep the whole tile's georef: only their geometry is cut.
    assert assigned.df[Col.GEOREF][1] == assigned.df[Col.GEOREF][2]


def test_tiles_outside_every_aoi_or_only_touching_one_are_left_out():
    # The AOI touches the second tile's right edge only, and covers the third one.
    assigned = assign_to_aois(_tiles(), _aois(test=box(200, 0, 300, 200)))
    assert list(assigned.df[Col.PARENT_IMAGE_ID]) == [2]


def test_copies_are_windows_of_their_tile():
    tiles = _tiles()
    assigned = assign_to_aois(tiles, _aois(train=box(0, 0, 400, 200)))
    assert type(assigned) is Tiles and assigned.parent_imagery is tiles
    assert list(assigned.df[Col.IMAGE_ID]) == [0, 1, 2, 3]
    assert assigned.df[Col.PATH].isna().all()
    assert list(assigned.get_disk_paths()) == ["ortho.tif"] * 4
    # What the tiles had is kept.
    assert list(assigned.df[Col.TIMESTAMP]) == [3] * 4
    assert assigned.df[Col.LAZY_CONDITIONS][0] == tiles.df[Col.LAZY_CONDITIONS][0]


def test_the_same_type_is_returned():
    crops = Crops.build(georef=[window_georef(RASTER, col_off=0, row_off=0, width=50, height=50)])
    assert type(assign_to_aois(crops, _aois(test=box(0, 0, 400, 200)))) is Crops


def test_which_tiles_have_usable_pixels_in_an_area():
    assigned = assign_to_aois(
        _tiles(), _aois(train=box(0, 0, 150, 200), valid=box(150, 0, 400, 200))
    )
    # A point at x = 120 is in tile 1, but only its train copy has usable pixels there.
    hits = assigned.df.sindex.query(box(119, 150, 121, 151), predicate="intersects")
    assert list(assigned.df[Col.AOI].iloc[hits]) == ["train"]


def test_no_tile_in_any_aoi():
    assert len(assign_to_aois(_tiles(), _aois(test=box(1000, 1000, 1100, 1100)))) == 0


def test_aois_in_another_crs():
    aois = load_aois(
        {"train": gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs=UTM_18N)}, "EPSG:32619"
    )
    with pytest.raises(ValueError, match="The AOIs are in"):
        assign_to_aois(_tiles(), aois)


def test_the_same_tiles_as_an_overlay():
    # geodataset assigned tiles with a geopandas overlay: the same (tile, AOI) pairs.
    tiles = _tiles()
    aois = _aois(train=box(0, 0, 150, 120), valid=box(150, 0, 400, 200), test=box(20, 150, 60, 190))
    assigned = assign_to_aois(tiles, aois)
    footprints = gpd.GeoDataFrame({"tile": range(4)}, geometry=tiles.df.geometry, crs=UTM_18N)
    overlay = gpd.overlay(footprints, aois, how="intersection")
    assert sorted(zip(assigned.df[Col.PARENT_IMAGE_ID], assigned.df[Col.AOI])) == sorted(
        zip(overlay["tile"], overlay["aoi"])
    )


def test_the_real_crop(real_raster):
    # Four windows of the real crop, cut by an AOI covering its left half.
    sources = Sources.from_paths(real_raster)
    georef = sources.df[Col.GEOREF][0]
    half_width, half_height = georef["width"] // 2, georef["height"] // 2
    windows = [
        window_georef(georef, col_off=c, row_off=r, width=half_width, height=half_height)
        for r in (0, half_height)
        for c in (0, half_width)
    ]
    tiles = Tiles.build(georef=windows, parent_image_id=0, parent_imagery=sources)
    minx, miny, maxx, maxy = sources.df.geometry[0].bounds
    aois = load_aois(
        {
            "left": gpd.GeoDataFrame(
                geometry=[box(minx, miny, (minx + maxx) / 2, maxy)], crs=tiles.df.crs
            )
        },
        tiles.df.crs,
    )
    assigned = assign_to_aois(tiles, aois)
    assert sorted(assigned.df[Col.PARENT_IMAGE_ID]) == [0, 2]  # the two left windows
    assert assigned.df.geometry.area.sum() == pytest.approx(
        sources.df.geometry[0].area / 2, rel=1e-3
    )


# =============================================================================
# The pixels of each copy
# =============================================================================


def _masks(assigned):
    """Return the usable mask of each row of ``assigned``."""
    crs = assigned.df.crs
    return [
        get_usable_mask(g, crs, r) for g, r in zip(assigned.df.geometry, assigned.df[Col.GEOREF])
    ]


def _border_tile(cut):
    """Return the two copies of tile 1 (x 100 to 200) cut by the AOIs 'left' and 'right' on
    either side of ``cut``, a line given as the polygon left of it, and their masks."""
    tiles = _tiles()
    right = box(0, 0, 400, 200).difference(cut)
    assigned = assign_to_aois(tiles, _aois(left=cut, right=right))
    copies = assigned.df[Col.PARENT_IMAGE_ID] == 1
    names = list(assigned.df[Col.AOI][copies])
    masks = [m for m, keep in zip(_masks(assigned), copies) if keep]
    return names, masks


def test_a_tile_wholly_inside_keeps_every_pixel():
    assigned = assign_to_aois(_tiles(), _aois(train=box(0, 0, 400, 200)))
    assert all(mask.all() for mask in _masks(assigned))


def test_a_border_tile_is_split_along_the_cut():
    # The cut at x = 150 falls on the edge between the tile's pixel columns 49 and 50.
    names, (left, right) = _border_tile(box(0, 0, 150, 200))
    assert names == ["left", "right"]
    assert left[:, :50].all() and not left[:, 50:].any()
    assert right[:, 50:].all() and not right[:, :50].any()


CUTS = [
    box(0, 0, 150.5, 200),  # through the pixel centres of column 50
    box(0, 0, 150.3, 200),  # inside column 50, left of its centre
    box(0, 0, 150.7, 200),  # inside column 50, right of its centre
    Polygon([(0, 0), (180, 0), (120, 200), (0, 200)]),  # slanted
]


@pytest.mark.parametrize("cut", CUTS)
def test_every_pixel_of_a_border_tile_is_in_exactly_one_copy(cut):
    names, masks = _border_tile(cut)
    assert names == ["left", "right"]
    assert np.array_equal(sum(m.astype(int) for m in masks), np.ones((100, 100), dtype=int))


def test_a_pixel_is_in_the_copy_holding_its_centre():
    # The cut at x = 150.3 leaves column 50's centre (150.5) on the right.
    _, (left, right) = _border_tile(box(0, 0, 150.3, 200))
    assert not left[:, 50].any() and right[:, 50].all()
    _, (left, right) = _border_tile(box(0, 0, 150.7, 200))
    assert left[:, 50].all() and not right[:, 50].any()


def test_an_aoi_with_a_hole():
    ring = box(0, 0, 400, 200).difference(Point(150, 150).buffer(20))
    assigned = assign_to_aois(_tiles(), _aois(train=ring))
    mask = _masks(assigned)[1]  # tile 1, x 100 to 200 and y 100 to 200
    assert not mask[50, 50]  # the pixel at (150.5, 149.5), in the hole
    assert mask.sum() == pytest.approx(100 * 100 - np.pi * 20**2, rel=0.02)


def test_an_aoi_covering_no_pixel_centre_of_a_tile():
    # A sliver of the tile, less than half a pixel wide: a copy with no usable pixel, which the
    # skip rule then drops when the tile is read.
    assigned = assign_to_aois(_tiles(), _aois(test=box(199.8, 0, 250, 200)))
    assert list(assigned.df[Col.PARENT_IMAGE_ID]) == [1, 2]
    assert not _masks(assigned)[0].any()


def test_masks_of_tiles_in_another_crs_than_the_table():
    # Two rasters on either side of the UTM 18N / 19N line; the table is in zone 18N.
    def raster(crs, left):
        return make_georef(
            transform=[1, 0, left, 0, -1, 5040100],
            crs=crs,
            width=100,
            height=100,
            count=3,
            dtype="uint8",
        )

    zone_19 = raster("EPSG:32619", 230000)
    tiles = Tiles.build(georef=[raster(UTM_18N, 700000), zone_19])
    # An AOI covering the left half of the zone 19N tile, drawn in zone 19N.
    half = gpd.GeoDataFrame(geometry=[box(230000, 5040000, 230050, 5040100)], crs="EPSG:32619")
    assigned = assign_to_aois(tiles, load_aois({"left": half}, tiles.df.crs))
    assert list(assigned.df[Col.PARENT_IMAGE_ID]) == [1]
    mask = _masks(assigned)[0]
    # Back in its own CRS, the cut is on the edge between columns 49 and 50, up to rounding.
    assert mask[:, :49].all() and not mask[:, 51:].any()


def test_masks_in_pixel_coordinates():
    photo = make_georef(
        transform=[1, 0, 0, 0, 1, 0],
        crs=None,
        width=64,
        height=32,
        count=3,
        dtype="uint8",
    )
    tiles = Tiles.build(georef=[photo], path="photo.jpg")
    aois = load_aois({"top": gpd.GeoDataFrame(geometry=[box(0, 0, 64, 10)])}, crs=None)
    mask = _masks(assign_to_aois(tiles, aois))[0]
    assert mask[:10].all() and not mask[10:].any()
