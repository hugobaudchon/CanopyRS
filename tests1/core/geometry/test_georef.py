import json

import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.windows import Window
from rasterio.windows import transform as rasterio_window_transform

from canopyrs1.core.geometry.georef import make_georef, read_georef, window_georef


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
