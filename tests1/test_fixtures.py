"""Checks that the shared fixtures in conftest.py hold what their docstrings say."""

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import ColorInterp

from tests1.conftest import make_georef


def test_make_georef():
    georef = make_georef(width=10, height=20, gsd=0.5, x0=100.0, y0=200.0, count=4)
    assert georef["transform"] == [0.5, 0.0, 100.0, 0.0, -0.5, 200.0]
    assert (georef["width"], georef["height"], georef["count"]) == (10, 20, 4)
    assert georef["crs"] == "EPSG:32618"


def test_rgb_raster(rgb_raster):
    with rasterio.open(rgb_raster) as src:
        assert (src.count, src.width, src.height) == (3, 256, 256)
        assert src.crs.to_epsg() == 32618
        assert tuple(src.bounds) == (0.0, 0.0, 256.0, 256.0)
        assert src.colorinterp == (ColorInterp.red, ColorInterp.green, ColorInterp.blue)


def test_rgba_raster(rgba_raster):
    with rasterio.open(rgba_raster) as src:
        assert src.count == 4
        assert src.colorinterp[3] == ColorInterp.alpha
        data = src.read()
    assert (data[:, :, :128] == 0).all()             # left half: transparent and black
    assert (data[3, :, 128:] == 255).all()           # right half: opaque


def test_unprojected_raster(unprojected_raster):
    with rasterio.open(unprojected_raster) as src:
        assert src.crs.to_epsg() == 4326
        assert (src.bounds.left, src.bounds.top) == (-73.6, 45.5)


def test_tiles_dir(tiles_dir, rgb_raster):
    paths = sorted(tiles_dir.glob("*.tif"))
    assert [p.name for p in paths] == ["tile_0.tif", "tile_1.tif"]
    with rasterio.open(rgb_raster) as src:
        full = src.read()
    for path, (left, col) in zip(paths, [(0.0, 0), (128.0, 128)]):
        with rasterio.open(path) as tile:
            assert (tile.width, tile.height) == (128, 128)
            assert (tile.bounds.left, tile.bounds.top) == (left, 256.0)
            assert (tile.read() == full[:, :128, col:col + 128]).all()


def test_labels(box_labels, polygon_labels):
    boxes, polygons = gpd.read_file(box_labels), gpd.read_file(polygon_labels)
    assert len(boxes) == len(polygons) == 4
    assert boxes.crs.to_epsg() == polygons.crs.to_epsg() == 32618
    # A box fills its bounding rectangle; a round crown doesn't.
    assert np.allclose(boxes.area, boxes.envelope.area)
    assert (polygons.area < 0.9 * polygons.envelope.area).all()
