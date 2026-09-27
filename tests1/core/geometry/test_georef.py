import json

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.windows import Window
from rasterio.windows import transform as rasterio_window_transform
from shapely.affinity import affine_transform
from shapely.geometry import MultiPolygon, Point, Polygon, box

from canopyrs1.core.geometry.georef import (crs_to_pixel, get_bounds, get_footprint, make_georef,
                                            pixel_to_crs, read_georef, window_georef)

NORTH_UP = make_georef(transform=[1.0, 0.0, 0.0, 0.0, -1.0, 256.0], crs="EPSG:32618",
                       width=256, height=256, count=3, dtype="uint8")
ROTATED = make_georef(transform=[0.3, 0.1, 500.0, 0.2, -0.3, 900.0], crs="EPSG:32618",
                      width=100, height=50, count=3, dtype="uint8")


def test_make_georef_turns_rasterio_objects_into_plain_values():
    georef = make_georef(transform=Affine(0.5, 0.0, 10.0, 0.0, -0.5, 20.0), crs=CRS.from_epsg(32618),
                         width=64, height=32, count=3, dtype="uint8", nodata=0)
    assert georef == {
        "transform": [0.5, 0.0, 10.0, 0.0, -0.5, 20.0],
        "crs": "EPSG:32618",
        "width": 64,
        "height": 32,
        "count": 3,
        "dtype": "uint8",
        "nodata": 0.0,
    }
    assert json.loads(json.dumps(georef)) == georef      # plain values only


def test_make_georef_without_crs():
    georef = make_georef(transform=[1, 0, 0, 0, 1, 0], crs=None, width=8, height=8, count=3, dtype="uint8")
    assert georef["crs"] is None and georef["nodata"] is None


def test_read_georef(rgb_raster, rgba_raster, unprojected_raster):
    with rasterio.open(rgb_raster) as src:
        assert read_georef(src) == {
            "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 256.0],
            "crs": "EPSG:32618",
            "width": 256,
            "height": 256,
            "count": 3,
            "dtype": "uint8",
            "nodata": None,
        }
    with rasterio.open(rgba_raster) as src:
        assert read_georef(src)["count"] == 4
    with rasterio.open(unprojected_raster) as src:
        assert read_georef(src)["crs"] == "EPSG:4326"


@pytest.mark.parametrize("transform", [
    Affine(1.0, 0.0, 0.0, 0.0, -1.0, 256.0),        # north-up
    Affine(0.3, 0.1, 500.0, 0.2, -0.3, 900.0),      # rotated
])
@pytest.mark.parametrize("window", [
    Window(0, 0, 128, 128),
    Window(100, 30, 50, 70),
    Window(-20, -10, 64, 64),                        # past the top-left edge
])
def test_window_georef_matches_rasterio(transform, window):
    parent = make_georef(transform=transform, crs="EPSG:32618", width=256, height=256, count=3, dtype="uint8")
    child = window_georef(parent, col_off=window.col_off, row_off=window.row_off,
                          width=window.width, height=window.height)
    assert child["transform"] == pytest.approx(list(rasterio_window_transform(window, transform))[:6])
    assert (child["width"], child["height"]) == (window.width, window.height)
    assert {k: child[k] for k in ("crs", "count", "dtype", "nodata")} == \
           {k: parent[k] for k in ("crs", "count", "dtype", "nodata")}


def test_window_georef_matches_the_tile_file(rgb_raster, tiles_dir):
    # tile_1.tif was cut from columns 128-255 of rgb_raster.
    with rasterio.open(rgb_raster) as src:
        expected = window_georef(read_georef(src), col_off=128, row_off=0, width=128, height=128)
    with rasterio.open(tiles_dir / "tile_1.tif") as tile:
        assert read_georef(tile) == expected


def test_pixel_to_crs_on_a_north_up_image():
    # Pixel rows go down, CRS y goes up: the top 10 pixel rows are the top 10 m of the raster.
    assert pixel_to_crs(box(0, 0, 10, 10), NORTH_UP).equals(box(0, 246, 10, 256))
    assert crs_to_pixel(box(0, 246, 10, 256), NORTH_UP).equals(box(0, 0, 10, 10))


@pytest.mark.parametrize("geometry", [
    box(1, 2, 30, 40),
    Point(7, 3),
    Polygon([(0, 0), (10, 0), (5, 8)], holes=[[(4, 1), (6, 1), (5, 3)]]),
    MultiPolygon([box(0, 0, 1, 1), box(5, 5, 9, 9)]),
])
@pytest.mark.parametrize("georef", [NORTH_UP, ROTATED])
def test_pixel_to_crs_matches_shapely_and_round_trips(geometry, georef):
    a, b, c, d, e, f = georef["transform"]
    expected = affine_transform(geometry, [a, b, d, e, c, f])
    in_crs = pixel_to_crs(geometry, georef)
    assert in_crs.equals_exact(expected, 1e-9)
    assert crs_to_pixel(in_crs, georef).equals_exact(geometry, 1e-9)


def test_several_geometries_keep_their_order():
    series = gpd.GeoSeries([box(0, 0, 1, 1), Point(5, 5)], index=[10, 20])
    result = pixel_to_crs(series, NORTH_UP)
    assert isinstance(result, np.ndarray) and len(result) == 2
    assert result[0].equals(box(0, 255, 1, 256)) and result[1].equals(Point(5, 251))
    assert len(pixel_to_crs([], NORTH_UP)) == 0


def test_footprint_and_bounds_of_a_north_up_image(rgb_raster):
    with rasterio.open(rgb_raster) as src:
        georef = read_georef(src)
        assert get_bounds(georef) == pytest.approx(tuple(src.bounds))
    assert get_footprint(georef).equals(box(0, 0, 256, 256))


def test_footprint_of_a_window_matches_the_tile_file(tiles_dir):
    with rasterio.open(tiles_dir / "tile_1.tif") as tile:
        assert get_bounds(read_georef(tile)) == pytest.approx((128.0, 128.0, 256.0, 256.0))


def test_footprint_of_a_rotated_image():
    footprint = get_footprint(ROTATED)
    a, b, _, d, e, _ = ROTATED["transform"]
    assert footprint.area == pytest.approx(100 * 50 * abs(a * e - b * d))
    assert footprint.area < box(*get_bounds(ROTATED)).area     # the bounds rectangle is larger
