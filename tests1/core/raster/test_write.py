import warnings

import numpy as np
import pytest
import rasterio
from rasterio.enums import ColorInterp
from rasterio.errors import NotGeoreferencedWarning
from rasterio.warp import calculate_default_transform

from canopyrs1.core.geometry.georef import make_georef, read_georef, window_georef
from canopyrs1.core.raster.read import read_window
from canopyrs1.core.raster.resampling import get_resampled_georef
from canopyrs1.core.raster.write import write_tile

RGB = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]


def _read_back(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as tile:
            return tile.read(), read_georef(tile), tile.colorinterp


def test_a_tile_keeps_its_pixels_and_georef(rgb_raster, tmp_path):
    with rasterio.open(rgb_raster) as src:
        georef = window_georef(read_georef(src), col_off=64, row_off=32, width=100, height=50)
        pixels = read_window(src, georef)
    path = write_tile(tmp_path / "tile.tif", pixels, georef, colorinterp=RGB)
    assert path == tmp_path / "tile.tif"
    written, written_georef, colours = _read_back(path)
    np.testing.assert_array_equal(written, pixels)
    assert written_georef == georef
    assert list(colours) == RGB


def test_a_resampled_and_reprojected_tile(unprojected_raster, tmp_path):
    with rasterio.open(unprojected_raster) as src:
        resampled = get_resampled_georef(src, ground_resolution=0.5)
        georef = window_georef(resampled, col_off=10, row_off=20, width=64, height=64)
        pixels = read_window(src, georef)
    written, written_georef, _ = _read_back(write_tile(tmp_path / "tile.tif", pixels, georef))
    np.testing.assert_array_equal(written, pixels)
    assert written_georef == georef
    assert written_georef["crs"] == "EPSG:32618"


def test_a_tile_without_crs_keeps_its_transform(tmp_path):
    # The aggregator places tiles by their transform, so it must survive without a CRS.
    raster = make_georef(
        transform=[1, 0, 0, 0, 1, 0],
        crs=None,
        width=256,
        height=256,
        count=3,
        dtype="uint8",
    )
    georef = window_georef(raster, col_off=128, row_off=64, width=32, height=32)
    pixels = np.full((3, 32, 32), 7, np.uint8)
    written, written_georef, _ = _read_back(write_tile(tmp_path / "tile.tif", pixels, georef))
    assert written_georef == georef
    assert written_georef["crs"] is None
    assert written_georef["transform"] == [1, 0, 128, 0, 1, 64]
    np.testing.assert_array_equal(written, pixels)


def test_only_the_bands_written(rgba_raster, tmp_path):
    # Three bands read from an RGBA raster are written as a three-band file.
    with rasterio.open(rgba_raster) as src:
        georef = window_georef(read_georef(src), col_off=128, row_off=0, width=64, height=64)
        pixels = read_window(src, georef, bands=[1, 2, 3])
    written, written_georef, colours = _read_back(
        write_tile(tmp_path / "tile.tif", pixels, georef, colorinterp=RGB)
    )
    assert written.shape == (3, 64, 64) and written_georef["count"] == 3
    assert list(colours) == RGB


def test_other_dtypes_and_nodata(tmp_path):
    georef = make_georef(
        transform=[0.5, 0, 100, 0, -0.5, 200],
        crs="EPSG:32618",
        width=8,
        height=4,
        count=2,
        dtype="uint16",
        nodata=65535,
    )
    pixels = np.arange(64, dtype=np.uint16).reshape(2, 4, 8) * 1000
    written, written_georef, _ = _read_back(write_tile(tmp_path / "tile.tif", pixels, georef))
    assert written.dtype == np.uint16
    np.testing.assert_array_equal(written, pixels)
    assert written_georef["nodata"] == 65535


def test_pixels_and_georef_must_match(tmp_path):
    georef = make_georef(
        transform=[1, 0, 0, 0, -1, 0],
        crs="EPSG:32618",
        width=64,
        height=32,
        count=3,
        dtype="uint8",
    )
    with pytest.raises(ValueError, match="32 x 64, but the georef is 64 x 32"):
        write_tile(tmp_path / "tile.tif", np.zeros((3, 64, 32), np.uint8), georef)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_zstd_compression_is_lossless(dtype, tmp_path):
    georef = make_georef(
        transform=[1, 0, 0, 0, -1, 64],
        crs="EPSG:32618",
        width=64,
        height=64,
        count=3,
        dtype=np.dtype(dtype).name,
    )
    pixels = (np.arange(3 * 64 * 64) % 251).reshape(3, 64, 64).astype(dtype)
    path = write_tile(tmp_path / "tile.tif", pixels, georef, compress="zstd")
    with rasterio.open(path) as tile:
        assert tile.compression.name.lower() == "zstd"
        np.testing.assert_array_equal(tile.read(), pixels)
    plain = write_tile(tmp_path / "plain.tif", pixels, georef)
    assert path.stat().st_size < plain.stat().st_size


def test_unknown_compression_is_refused(tmp_path):
    georef = make_georef(
        transform=[1, 0, 0, 0, -1, 8],
        crs="EPSG:32618",
        width=8,
        height=8,
        count=3,
        dtype="uint8",
    )
    with pytest.raises(ValueError, match="None or 'zstd'"):
        write_tile(tmp_path / "tile.tif", np.zeros((3, 8, 8), np.uint8), georef, compress="lzw")


# =============================================================================
# The real orthomosaic crop: 6 cm, UTM 20S, RGBA, striped, no overviews
# =============================================================================

REAL_CROP_WRITES = [
    (dict(), (0, 0, 512, 512), None),  # its own pixels
    (dict(), (2600, 1500, 512, 512), None),  # past the bottom-right edge
    (dict(ground_resolution=0.05), (1000, 600, 777, 777), None),
    (dict(ground_resolution=0.1), (-100, -100, 512, 512), "zstd"),  # past the top-left edge
    (dict(scale_factor=0.5), (300, 200, 512, 512), "zstd"),
]


@pytest.mark.parametrize("settings, window, compress", REAL_CROP_WRITES)
def test_real_crop_round_trip(real_raster, settings, window, compress, tmp_path):
    col, row, width, height = window
    with rasterio.open(real_raster) as src:
        georef = window_georef(
            get_resampled_georef(src, **settings),
            col_off=col,
            row_off=row,
            width=width,
            height=height,
        )
        pixels = read_window(src, georef)
        path = write_tile(
            tmp_path / "tile.tif",
            pixels,
            georef,
            colorinterp=src.colorinterp,
            compress=compress,
        )
        colours = src.colorinterp
    written, written_georef, written_colours = _read_back(path)
    assert pixels.shape == (4, height, width)
    np.testing.assert_array_equal(written, pixels)
    assert written_georef == georef
    assert written_colours == colours


def test_real_crop_reprojected_round_trip(real_raster, tmp_path):
    with rasterio.open(real_raster) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            "EPSG:32721",
            src.width,
            src.height,
            *src.bounds,
            resolution=0.1,
        )
        reprojected = make_georef(
            transform=transform,
            crs="EPSG:32721",
            width=width,
            height=height,
            count=src.count,
            dtype="uint8",
        )
        georef = window_georef(reprojected, col_off=400, row_off=300, width=512, height=512)
        pixels = read_window(src, georef)
    written, written_georef, _ = _read_back(write_tile(tmp_path / "tile.tif", pixels, georef))
    np.testing.assert_array_equal(written, pixels)
    assert written_georef == georef
