"""read_window gives the same pixels as geodataset's tiles, where geodataset is right: without
resampling, at whole-number scale factors, and when reprojecting. At other resolutions geodataset's
pixels drift (tests1/core/raster/test_read.py checks where each pixel lands).

The rasters are flat colour blocks, so that resampling one window or the whole raster, which only
differs by rounding at the edges, doesn't hide how well the pixels line up."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

from geodataset.geodata import Raster

from canopyrs1.core.geometry.georef import window_georef
from canopyrs1.core.raster.read import read_window
from canopyrs1.core.raster.resampling import get_resampled_georef

WINDOWS = [(0, 0, 64, 64), (100, 30, 50, 70), (-20, -10, 64, 64)]


def _write_blocks(path, crs, transform, size=320, block=16):
    rng = np.random.default_rng(0)
    colours = rng.integers(1, 255, (3, size // block, size // block), dtype=np.uint8)
    dst = rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=3,
        dtype="uint8",
        crs=crs,
        transform=transform,
    )
    with dst:
        dst.write(np.kron(colours, np.ones((block, block), np.uint8)))
    return path


@pytest.fixture
def utm_blocks(tmp_path):
    return _write_blocks(tmp_path / "utm.tif", "EPSG:32618", from_origin(600000, 5040000, 0.1, 0.1))


@pytest.fixture
def lonlat_blocks(tmp_path):
    return _write_blocks(tmp_path / "lonlat.tif", "EPSG:4326", from_origin(-73.6, 45.5, 1e-6, 1e-6))


def _old_and_new(path, settings, window, tmp_path):
    raster = Raster(str(path), temp_dir=tmp_path, **settings)
    col, row, width, height = window
    old_tile = raster.create_tile_metadata(window=Window(col, row, width, height), tile_id=1)
    old = old_tile.get_pixel_data()
    with rasterio.open(path) as src:
        georef = window_georef(
            get_resampled_georef(src, **settings),
            col_off=col,
            row_off=row,
            width=width,
            height=height,
        )
        new = read_window(src, georef)
    assert old.shape == new.shape
    return old.astype(int), new.astype(int)


@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("raster", ["rgb_raster", "rgba_raster"])
def test_identical_without_resampling(raster, window, request, tmp_path):
    old, new = _old_and_new(request.getfixturevalue(raster), {}, window, tmp_path)
    np.testing.assert_array_equal(old, new)


WHOLE_NUMBER_SETTINGS = [
    dict(scale_factor=0.5),
    dict(scale_factor=2),
    dict(ground_resolution=0.2),
]


@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("settings", WHOLE_NUMBER_SETTINGS)
def test_close_at_whole_number_scale_factors(utm_blocks, settings, window, tmp_path):
    old, new = _old_and_new(utm_blocks, settings, window, tmp_path)
    assert np.abs(old - new).mean() < 1


@pytest.mark.parametrize("window", WINDOWS)
def test_close_when_reprojected(lonlat_blocks, window, tmp_path):
    old, new = _old_and_new(lonlat_blocks, dict(ground_resolution=0.25), window, tmp_path)
    assert np.abs(old - new).mean() < 1
