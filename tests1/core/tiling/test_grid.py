import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from canopyrs1.core.constants import Col, Modality
from canopyrs1.core.geometry.georef import get_footprint
from canopyrs1.core.tables.imagery import Sources, Tiles
from canopyrs1.core.tiling.grid import grid_tiles

# rgb_raster: 256 x 256 pixels of 1 m, covering x and y from 0 to 256 in UTM 18N.


def _corners(tiles):
    """Return each tile's top-left corner, in its CRS."""
    return [(g["transform"][2], g["transform"][5]) for g in tiles.df[Col.GEOREF]]


def test_a_grid_without_overlap(rgb_raster):
    sources = Sources.from_paths(rgb_raster)
    tiles = grid_tiles(sources, tile_size=128)
    assert type(tiles) is Tiles and len(tiles) == 4
    assert _corners(tiles) == [(0, 256), (128, 256), (0, 128), (128, 128)]  # row after row
    assert all((g["width"], g["height"]) == (128, 128) for g in tiles.df[Col.GEOREF])
    assert tiles.parent_imagery is sources and list(tiles.df[Col.PARENT_IMAGE_ID]) == [0] * 4
    assert tiles.df[Col.PATH].isna().all()
    assert list(tiles.get_disk_paths()) == [str(rgb_raster)] * 4


def test_overlapping_tiles_reach_past_the_edges(rgb_raster):
    tiles = grid_tiles(Sources.from_paths(rgb_raster), tile_size=128, tile_overlap=0.5)
    # Every 64 pixels: 4 rows of 4, the last ones reaching 64 pixels past the edges.
    assert len(tiles) == 16
    assert sorted({x for x, _ in _corners(tiles)}) == [0, 64, 128, 192]
    assert tiles.df.geometry.iloc[-1].equals(box(192, -64, 320, 64))


def test_a_tile_larger_than_the_source(rgb_raster):
    tiles = grid_tiles(Sources.from_paths(rgb_raster), tile_size=1000)
    assert len(tiles) == 1 and tiles.df.geometry[0].equals(box(0, -744, 1000, 256))


def test_a_ground_resolution(rgb_raster):
    # At 2 m, the 256 m raster is 128 pixels: 4 tiles of 64 pixels, 128 m each.
    tiles = grid_tiles(Sources.from_paths(rgb_raster), tile_size=64, ground_resolution=2.0)
    assert len(tiles) == 4
    assert tiles.df[Col.GEOREF][0]["transform"][0] == pytest.approx(2.0)
    assert tiles.df.geometry[0].equals(box(0, 128, 128, 256))


def test_a_scale_factor(rgb_raster):
    tiles = grid_tiles(Sources.from_paths(rgb_raster), tile_size=64, scale_factor=0.5)
    assert len(tiles) == 4 and tiles.df[Col.GEOREF][0]["transform"][0] == pytest.approx(2.0)


def test_a_source_in_lat_lon_is_tiled_in_utm(unprojected_raster):
    sources = Sources.from_paths(unprojected_raster)
    tiles = grid_tiles(sources, tile_size=128, ground_resolution=1.0)
    assert sources.df[Col.GEOREF][0]["crs"] == "EPSG:4326"  # the source keeps its own CRS
    assert {g["crs"] for g in tiles.df[Col.GEOREF]} == {"EPSG:32618"}
    assert tiles.df.crs == "EPSG:32618"
    # The tiles cover the whole source.
    footprint = gpd.GeoSeries(sources.df.geometry, crs=sources.df.crs).to_crs("EPSG:32618")[0]
    assert tiles.df.geometry.union_all().covers(footprint)


def test_no_source():
    assert len(grid_tiles(Sources.build(georef=[]), tile_size=128)) == 0


BAD_GRIDS = [(0, 0.0), (128, -0.1), (128, 1.0), (4, 0.9)]  # the last moves by 0.4 pixels


@pytest.mark.parametrize("tile_size, tile_overlap", BAD_GRIDS)
def test_grids_that_can_t_be(tile_size, tile_overlap, rgb_raster):
    with pytest.raises(ValueError, match="Tile windows need"):
        grid_tiles(Sources.from_paths(rgb_raster), tile_size=tile_size, tile_overlap=tile_overlap)


def test_the_real_crop(real_raster):
    sources = Sources.from_paths(real_raster)
    tiles = grid_tiles(sources, tile_size=512, tile_overlap=0.25)
    with rasterio.open(real_raster) as src:
        width, height = src.width, src.height
    step = 384
    assert len(tiles) == len(range(0, width, step)) * len(range(0, height, step))
    assert tiles.df.geometry.union_all().covers(sources.df.geometry[0])
    # Each tile is a window of the source at its own resolution.
    source = sources.df[Col.GEOREF][0]
    first = tiles.df[Col.GEOREF][0]
    assert first["transform"] == source["transform"]
    assert get_footprint(first).area == pytest.approx(512 * 512 * source["transform"][0] ** 2)


# =============================================================================
# Several sources of one place
# =============================================================================


def _raster(path, *, x0=0, y0=256, gsd=1.0, size=256, count=3):
    """Write a uint8 raster of ``size`` x ``size`` pixels of ``gsd`` m, its top-left corner at
    (``x0``, ``y0``) in UTM 18N, and return its path."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=count,
        dtype="uint8",
        crs="EPSG:32618",
        transform=from_origin(x0, y0, gsd, gsd),
    ) as dst:
        dst.write(np.ones((count, size, size), dtype=np.uint8))
    return path


def _tiles_of(tiles, source_id):
    return tiles.df[tiles.df[Col.PARENT_IMAGE_ID] == source_id]


def test_two_dates_share_the_grid(rgb_raster, tmp_path):
    later = _raster(tmp_path / "2025.tif")
    sources = Sources.from_paths([rgb_raster, later], timestamp=[0, 1])
    tiles = grid_tiles(sources, tile_size=128)
    # Each window twice, once per date, one after the other, with the window's number.
    assert list(tiles.df[Col.PARENT_IMAGE_ID]) == [0, 1] * 4
    assert list(tiles.df[Col.INSTANCE_ID]) == [0, 0, 1, 1, 2, 2, 3, 3]
    assert list(tiles.df[Col.TIMESTAMP]) == [0, 1] * 4
    # At the same resolution, both dates' tiles are the same window.
    first, second = _tiles_of(tiles, 0), _tiles_of(tiles, 1)
    for a, b in zip(first[Col.GEOREF], second[Col.GEOREF]):
        assert a["transform"] == b["transform"] and (a["width"], a["height"]) == (128, 128)


def test_a_coarser_modality_shows_the_same_ground_in_fewer_pixels(rgb_raster, tmp_path):
    chm = _raster(tmp_path / "chm.tif", gsd=4.0, size=64, count=1)
    sources = Sources.from_paths([rgb_raster, chm], modality=[Modality.RGB, "chm"])
    tiles = grid_tiles(sources, tile_size=128)
    rgb, heights = _tiles_of(tiles, 0), _tiles_of(tiles, 1)
    assert len(rgb) == len(heights) == 4
    for a, b in zip(rgb[Col.GEOREF], heights[Col.GEOREF]):
        assert (b["width"], b["height"]) == (32, 32)  # 128 m in 4 m pixels
        assert b["transform"][0] == pytest.approx(4.0) and b["count"] == 1
        assert get_footprint(b).equals(get_footprint(a))
    assert list(heights[Col.BANDS]) == [[1, 2, 3]] * 4  # the bands the CHM source was given


def test_the_same_ground_in_a_whole_number_of_pixels(rgb_raster, tmp_path):
    # 128 m in 3 m pixels is 42.7 pixels: 43 pixels of a little under 3 m.
    other = _raster(tmp_path / "3m.tif", gsd=3.0, size=86)
    sources = Sources.from_paths([rgb_raster, other], timestamp=[0, 1])
    georef = _tiles_of(grid_tiles(sources, tile_size=128), 1)[Col.GEOREF].iloc[0]
    assert georef["width"] == 43 and georef["transform"][0] == pytest.approx(128 / 43)


def test_the_grid_is_extended_to_cover_every_source(rgb_raster, tmp_path):
    # The second date reaches 100 m further left, and stops 100 m before the right edge.
    shifted = _raster(tmp_path / "shifted.tif", x0=-100)
    sources = Sources.from_paths([rgb_raster, shifted], timestamp=[0, 1])
    tiles = grid_tiles(sources, tile_size=128)
    main = _tiles_of(tiles, 0)
    # The main source's tiles are exactly its own grid.
    alone = grid_tiles(Sources.from_paths(rgb_raster), tile_size=128)
    assert list(main[Col.GEOREF]) == list(alone.df[Col.GEOREF])
    # One more column of windows, on the left, covering the second date only.
    left = _tiles_of(tiles, 1).geometry.bounds.minx.min()
    assert left == -128
    assert set(tiles.df.loc[tiles.df.geometry.bounds.minx == -128, Col.PARENT_IMAGE_ID]) == {1}
    # The second date has no tile in the windows it doesn't reach.
    assert (_tiles_of(tiles, 1).geometry.bounds.minx < 156).all()


def test_another_main_source(rgb_raster, tmp_path):
    chm = _raster(tmp_path / "chm.tif", gsd=4.0, size=64, count=1)
    sources = Sources.from_paths([rgb_raster, chm], modality=[Modality.RGB, "chm"])
    tiles = grid_tiles(sources, tile_size=32, main_source=1)
    # 32 CHM pixels of 4 m: 128 m, 128 RGB pixels of 1 m.
    rgb, heights = _tiles_of(tiles, 0), _tiles_of(tiles, 1)
    assert {g["width"] for g in heights[Col.GEOREF]} == {32}
    assert {g["width"] for g in rgb[Col.GEOREF]} == {128}


def test_one_ground_resolution_making_some_sources_coarser_and_others_finer(rgb_raster, tmp_path):
    chm = _raster(tmp_path / "chm.tif", gsd=4.0, size=64, count=1)
    sources = Sources.from_paths([rgb_raster, chm], modality=[Modality.RGB, "chm"])
    with pytest.warns(UserWarning, match="makes some sources coarser and others finer") as caught:
        tiles = grid_tiles(sources, tile_size=64, ground_resolution=2.0)
    ours = [str(w.message) for w in caught if "coarser" in str(w.message)]  # not rasterio's
    assert len(ours) == 1 and "chm.tif (4 per pixel, ×0.5)" in ours[0]
    assert {g["width"] for g in _tiles_of(tiles, 1)[Col.GEOREF]} == {64}  # both at 2 m


def test_one_ground_resolution_per_source(rgb_raster, tmp_path):
    chm = _raster(tmp_path / "chm.tif", gsd=4.0, size=64, count=1)
    sources = Sources.from_paths([rgb_raster, chm], modality=[Modality.RGB, "chm"])
    tiles = grid_tiles(sources, tile_size=64, ground_resolution=[2.0, 4.0])
    assert {g["width"] for g in _tiles_of(tiles, 0)[Col.GEOREF]} == {64}  # 128 m at 2 m
    assert {g["width"] for g in _tiles_of(tiles, 1)[Col.GEOREF]} == {32}  # 128 m at 4 m
    with pytest.raises(ValueError, match="Give one ground_resolution per source"):
        grid_tiles(sources, tile_size=64, ground_resolution=[2.0])
    # A numpy array works as well as a list.
    as_array = grid_tiles(sources, tile_size=64, ground_resolution=np.array([2.0, 4.0]))
    assert list(as_array.df[Col.GEOREF]) == list(tiles.df[Col.GEOREF])


def test_a_main_source_that_isn_t_there(rgb_raster):
    with pytest.raises(ValueError, match="main_source 1 isn't one of the 1 sources"):
        grid_tiles(Sources.from_paths(rgb_raster), tile_size=64, main_source=1)


def test_a_scale_factor_for_several_sources(rgb_raster, tmp_path):
    sources = Sources.from_paths([rgb_raster, _raster(tmp_path / "b.tif")], timestamp=[0, 1])
    with pytest.raises(ValueError, match="A scale_factor only works for one source"):
        grid_tiles(sources, tile_size=64, scale_factor=0.5)


def test_windows_on_the_real_crop_and_a_coarser_copy(real_raster, tmp_path):
    # The real crop, and a 4 times coarser raster over the same ground, as a second modality.
    sources = Sources.from_paths(real_raster)
    georef = sources.df[Col.GEOREF][0]
    a, _, c, _, e, f = georef["transform"]
    coarse = tmp_path / "coarse.tif"
    with rasterio.open(
        coarse,
        "w",
        driver="GTiff",
        width=georef["width"] // 4,
        height=georef["height"] // 4,
        count=1,
        dtype="uint8",
        crs=georef["crs"],
        transform=from_origin(c, f, 4 * a, -4 * e),
    ) as dst:
        dst.write(np.ones((1, georef["height"] // 4, georef["width"] // 4), dtype=np.uint8))
    both = Sources.from_paths([real_raster, coarse], modality=[Modality.RGB, "chm"])
    tiles = grid_tiles(both, tile_size=512, tile_overlap=0.25)
    rgb, coarse_tiles = _tiles_of(tiles, 0), _tiles_of(tiles, 1)
    assert list(rgb[Col.GEOREF]) == list(
        grid_tiles(sources, tile_size=512, tile_overlap=0.25).df[Col.GEOREF]
    )
    for tile in coarse_tiles[Col.GEOREF]:
        assert tile["width"] == 128


def test_a_window_only_touching_a_source_has_no_tile_there(rgb_raster, tmp_path):
    # The second date starts exactly at x = 128, where the first column of windows ends.
    right_half = _raster(tmp_path / "right.tif", x0=128, size=128)
    sources = Sources.from_paths([rgb_raster, right_half], timestamp=[0, 1])
    tiles = _tiles_of(grid_tiles(sources, tile_size=128), 1)
    assert list(tiles.geometry.bounds.minx) == [128]  # only the window it's in


def test_a_sliver_of_a_source_still_gets_its_tile(rgb_raster, tmp_path):
    # The second date covers only the last 0.5 m of the first column of windows.
    sliver = _raster(tmp_path / "sliver.tif", x0=127.5, gsd=0.5, size=64)
    sources = Sources.from_paths([rgb_raster, sliver], timestamp=[0, 1])
    tiles = _tiles_of(grid_tiles(sources, tile_size=128), 1)
    assert 0 in set(tiles.geometry.bounds.minx)


def test_the_main_source_s_tiles_are_the_windows_and_come_first(rgb_raster, tmp_path):
    later = _raster(tmp_path / "2025.tif")
    chm = _raster(tmp_path / "chm.tif", gsd=4.0, size=64, count=1)
    sources = Sources.from_paths(
        [rgb_raster, later, chm],
        modality=[Modality.RGB, Modality.RGB, "chm"],
        timestamp=[0, 1, 0],
    )
    tiles = grid_tiles(sources, tile_size=128)
    # Each window's tiles together, in the sources' order.
    assert list(tiles.df[Col.PARENT_IMAGE_ID]) == [0, 1, 2] * 4
    alone = grid_tiles(Sources.from_paths(rgb_raster), tile_size=128)
    assert list(_tiles_of(tiles, 0)[Col.GEOREF]) == list(alone.df[Col.GEOREF])


def test_overlapping_tiles_extended_to_the_left_add_main_source_tiles_along_that_edge(
    rgb_raster, tmp_path
):
    # 128 px tiles every 64 px; the second date reaches 100 m further left. The grid gains
    # columns starting at x = -128 and -64: the one at -64 covers x -64 to 64, half on the main
    # source, which gets a tile there too.
    shifted = _raster(tmp_path / "shifted.tif", x0=-100)
    sources = Sources.from_paths([rgb_raster, shifted], timestamp=[0, 1])
    main = _tiles_of(grid_tiles(sources, tile_size=128, tile_overlap=0.5), 0)
    alone = grid_tiles(Sources.from_paths(rgb_raster), tile_size=128, tile_overlap=0.5)
    assert sorted(set(main.geometry.bounds.minx)) == [-64, 0, 64, 128, 192]
    assert sorted(set(alone.df.geometry.bounds.minx)) == [0, 64, 128, 192]
    # The main source's other tiles are its usual grid.
    extra = main.geometry.bounds.minx == -64
    assert list(main[Col.GEOREF][~extra]) == list(alone.df[Col.GEOREF])
