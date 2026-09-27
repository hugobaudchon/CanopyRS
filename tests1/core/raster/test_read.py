import warnings

import numpy as np
import pytest
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform

from canopyrs1.core.geometry.georef import make_georef, read_georef, window_georef
from canopyrs1.core.raster.read import read_window
from canopyrs1.core.raster.resampling import get_resampled_georef

BLOCK = 16  # pixels per side of each flat colour block


def _write_blocks(path, *, crs, transform, size=320, nodata=None):
    """Write a raster of flat colour blocks, BLOCK x BLOCK pixels each, and return its path."""
    rng = np.random.default_rng(0)
    colours = rng.integers(1, 255, (3, size // BLOCK, size // BLOCK), dtype=np.uint8)
    data = np.kron(colours, np.ones((BLOCK, BLOCK), np.uint8))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
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
            nodata=nodata,
        )
        with dst:
            dst.write(data)
    return path


@pytest.fixture
def utm_blocks(tmp_path):
    """10 cm pixels in UTM 18N, around Montreal."""
    transform = from_origin(600000, 5040000, 0.1, 0.1)
    return _write_blocks(tmp_path / "utm.tif", crs="EPSG:32618", transform=transform)


@pytest.fixture
def lonlat_blocks(tmp_path):
    """Pixels of 1e-6 degrees (about 8 cm x 11 cm) in WGS84, around Montreal."""
    transform = from_origin(-73.6, 45.5, 1e-6, 1e-6)
    return _write_blocks(tmp_path / "lonlat.tif", crs="EPSG:4326", transform=transform)


@pytest.fixture
def pixel_blocks(tmp_path):
    """No CRS and no transform, like a plain photo."""
    return _write_blocks(tmp_path / "photo.tif", crs=None, transform=Affine.identity())


def _open(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        return rasterio.open(path)


def _check_block_colours(src, georef, pixels, margin):
    """Check each pixel of ``pixels`` (read for ``georef``) whose centre falls inside a block of
    ``src``, at least ``margin`` file pixels from its edges: it must be exactly that block's colour.
    Return how many pixels were checked."""
    height, width = pixels.shape[1:]
    rows, cols = np.mgrid[0:height, 0:width] + 0.5
    a, b, c, d, e, f = georef["transform"]
    x, y = a * cols + b * rows + c, d * cols + e * rows + f
    if georef["crs"] != (src.crs.to_string() if src.crs else None):
        x, y = Transformer.from_crs(georef["crs"], src.crs, always_xy=True).transform(x, y)
    inverse = ~src.transform
    file_col = inverse.a * x + inverse.b * y + inverse.c
    file_row = inverse.d * x + inverse.e * y + inverse.f
    inside = (
        (file_col >= 0)
        & (file_col < src.width)
        & (file_row >= 0)
        & (file_row < src.height)
        & (np.minimum(file_col % BLOCK, BLOCK - file_col % BLOCK) >= margin)
        & (np.minimum(file_row % BLOCK, BLOCK - file_row % BLOCK) >= margin)
    )
    expected = src.read()[:, file_row[inside].astype(int), file_col[inside].astype(int)]
    np.testing.assert_array_equal(pixels[:, inside], expected)
    return inside.sum()


# =============================================================================
# On the file's own pixels
# =============================================================================

FILE_WINDOWS = [
    (0, 0, 64, 64),
    (100, 30, 50, 70),
    (192, 192, 64, 64),
]


@pytest.mark.parametrize("col, row, width, height", FILE_WINDOWS)
def test_the_file_s_pixels_are_read_as_they_are(rgb_raster, col, row, width, height):
    with rasterio.open(rgb_raster) as src:
        georef = window_georef(
            read_georef(src),
            col_off=col,
            row_off=row,
            width=width,
            height=height,
        )
        pixels = read_window(src, georef)
        assert pixels.dtype == np.uint8
        np.testing.assert_array_equal(pixels, src.read()[:, row : row + height, col : col + width])


def test_past_the_edges(rgb_raster, tmp_path):
    with rasterio.open(rgb_raster) as src:
        georef = window_georef(read_georef(src), col_off=-20, row_off=-10, width=64, height=64)
        pixels = read_window(src, georef)
        np.testing.assert_array_equal(pixels[:, 10:, 20:], src.read()[:, :54, :44])
        assert not pixels[:, :10, :].any() and not pixels[:, :, :20].any()
    # A raster with a nodata value is filled with it.
    with_nodata = _write_blocks(
        tmp_path / "nodata.tif",
        crs="EPSG:32618",
        transform=from_origin(600000, 5040000, 0.1, 0.1),
        nodata=255,
    )
    with rasterio.open(with_nodata) as src:
        georef = window_georef(read_georef(src), col_off=-20, row_off=0, width=64, height=64)
        pixels = read_window(src, georef)
        assert (pixels[:, :, :20] == 255).all()


def test_bands(rgba_raster):
    with rasterio.open(rgba_raster) as src:
        georef = window_georef(read_georef(src), col_off=100, row_off=0, width=64, height=64)
        everything = read_window(src, georef)
        assert everything.shape == (4, 64, 64)
        np.testing.assert_array_equal(read_window(src, georef, bands=[3, 1]), everything[[2, 0]])


# =============================================================================
# Resampled and reprojected: every pixel lands where it should
# =============================================================================

WINDOWS = [(0, 0), (60, 20), ("far", "far"), (-20, -20)]  # "far": the bottom-right corner


def _windows(resampled, size=64):
    for col, row in WINDOWS:
        col = resampled["width"] - size if col == "far" else col
        row = resampled["height"] - size if row == "far" else row
        yield window_georef(resampled, col_off=col, row_off=row, width=size, height=size)


RESAMPLING_SETTINGS = [
    dict(scale_factor=0.5),
    dict(scale_factor=0.37),
    dict(scale_factor=1.7),
    dict(ground_resolution=0.25),
    dict(ground_resolution=0.07),
]


@pytest.mark.parametrize("settings", RESAMPLING_SETTINGS)
def test_resampled_pixels_land_where_they_should(utm_blocks, settings):
    with rasterio.open(utm_blocks) as src:
        resampled = get_resampled_georef(src, **settings)
        for georef in _windows(resampled):
            pixels = read_window(src, georef)
            assert pixels.shape == (3, 64, 64)
            assert _check_block_colours(src, georef, pixels, margin=4) > 0


@pytest.mark.parametrize("ground_resolution", [0.25, 0.1, 0.05])
def test_reprojected_pixels_land_where_they_should(lonlat_blocks, ground_resolution):
    with rasterio.open(lonlat_blocks) as src:
        resampled = get_resampled_georef(src, ground_resolution=ground_resolution)
        assert resampled["crs"] == "EPSG:32618"
        for georef in _windows(resampled):
            assert _check_block_colours(src, georef, read_window(src, georef), margin=4) > 0


def test_a_raster_without_crs(pixel_blocks):
    with _open(pixel_blocks) as src:
        for settings in (dict(), dict(scale_factor=0.5), dict(scale_factor=0.37)):
            resampled = get_resampled_georef(src, **settings)
            for georef in _windows(resampled):
                assert _check_block_colours(src, georef, read_window(src, georef), margin=4) > 0


# =============================================================================
# The real orthomosaic crop: 6 cm, UTM 20S, RGBA, striped (full-width blocks), no overviews
# =============================================================================


def _read_all_at_once(src, georef):
    """Return the whole image described by ``georef``, reprojected from ``src`` in one warp: the
    reference each window is compared with."""
    vrt = WarpedVRT(
        src,
        crs=georef["crs"],
        transform=Affine(*georef["transform"]),
        width=georef["width"],
        height=georef["height"],
        resampling=Resampling.bilinear,
    )
    with vrt:
        return vrt.read()


def _cut(image, col, row, width, height):
    """Return the ``width`` x ``height`` window of ``image`` at (``col``, ``row``), zero outside
    it."""
    window = np.zeros((image.shape[0], height, width), image.dtype)
    top, left = max(row, 0), max(col, 0)
    part = image[:, top : row + height, left : col + width]
    window[:, top - row : top - row + part.shape[1], left - col : left - col + part.shape[2]] = part
    return window


def _corners_and_middle(georef, size=512):
    """Yield (col, row) of windows at the top-left, the middle, the bottom-right corner, and past
    the top-left edge of the image described by ``georef``."""
    yield 0, 0
    yield georef["width"] // 3, georef["height"] // 3
    yield georef["width"] - size // 2, georef["height"] - size // 2
    yield -size // 4, -size // 4


REAL_WINDOWS = [
    (0, 0, 512, 512),
    (1000, 600, 777, 777),
    (2600, 1500, 512, 512),  # past the bottom-right edge
]


@pytest.mark.parametrize("col, row, width, height", REAL_WINDOWS)
def test_real_crop_on_its_own_pixels(real_raster, col, row, width, height):
    with rasterio.open(real_raster) as src:
        georef = window_georef(
            read_georef(src),
            col_off=col,
            row_off=row,
            width=width,
            height=height,
        )
        pixels = read_window(src, georef)
        assert pixels.shape == (4, height, width)
        np.testing.assert_array_equal(pixels, _cut(src.read(), col, row, width, height))


REAL_RESAMPLING = [
    dict(ground_resolution=0.05),
    dict(ground_resolution=0.1),
    dict(scale_factor=0.5),
]


@pytest.mark.parametrize("settings", REAL_RESAMPLING)
def test_real_crop_resampled(real_raster, settings):
    # Shifting the reference by a single pixel makes the mean difference 3 to 10 on this image,
    # so a mean under 1 means every window is in the right place.
    with rasterio.open(real_raster) as src:
        resampled = get_resampled_georef(src, **settings)
        reference = _read_all_at_once(src, resampled).astype(int)
        for col, row in _corners_and_middle(resampled):
            georef = window_georef(resampled, col_off=col, row_off=row, width=512, height=512)
            pixels = read_window(src, georef).astype(int)
            assert np.abs(pixels - _cut(reference, col, row, 512, 512)).mean() < 1


def test_real_crop_reprojected(real_raster):
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
        reference = _read_all_at_once(src, reprojected).astype(int)
        for col, row in _corners_and_middle(reprojected):
            georef = window_georef(reprojected, col_off=col, row_off=row, width=512, height=512)
            pixels = read_window(src, georef).astype(int)
            assert np.abs(pixels - _cut(reference, col, row, 512, 512)).mean() < 1


@pytest.mark.parametrize("settings", [dict(), dict(ground_resolution=0.05)])
def test_real_crop_one_band_read_then_sliced(real_raster, settings):
    # The loader reads full-width bands of rows at once and slices the tiles out of them, which
    # must give exactly what reading each tile on its own gives.
    with rasterio.open(real_raster) as src:
        resampled = get_resampled_georef(src, **settings)
        band_georef = window_georef(
            resampled,
            col_off=0,
            row_off=512,
            width=resampled["width"],
            height=512,
        )
        band = read_window(src, band_georef)
        for col in (0, 1024, resampled["width"] - 512):
            tile = window_georef(resampled, col_off=col, row_off=512, width=512, height=512)
            np.testing.assert_array_equal(band[:, :, col : col + 512], read_window(src, tile))


def test_real_crop_alpha_band(real_raster):
    with rasterio.open(real_raster) as src:
        georef = window_georef(read_georef(src), col_off=-10, row_off=0, width=64, height=64)
        alpha = read_window(src, georef, bands=[4])
        assert alpha.shape == (1, 64, 64)
        assert (alpha[:, :, :10] == 0).all()  # outside the raster
        np.testing.assert_array_equal(alpha[:, :, 10:], src.read([4])[:, :64, :54])
