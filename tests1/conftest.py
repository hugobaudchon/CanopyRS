"""
Shared fixtures for the canopyrs1 tests.

Synthetic rasters and labels, written with rasterio and geopandas directly (never with canopyrs1
code, so a bug in the code under test can't hide in its own fixtures), and a helper that builds an
image's georef.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.geometry import Point, box

REAL_RASTER_PATH = Path(__file__).parent.parent / "assets" / "20240130_zf2tower_m3m_rgb_test_crop.tif"

RASTER_SIZE = 256
TILE_SIZE = 128


# =============================================================================
# Georeferencing helper
# =============================================================================

def make_georef(*, width=64, height=64, gsd=1.0, x0=0.0, y0=0.0, crs="EPSG:32618", count=3):
    """Return the georef of a north-up uint8 image whose top-left corner is at
    (x0, y0) in CRS units, with pixels of ``gsd`` CRS units. It has the same keys as the dict the
    tilerizer stores for each tile."""
    return {
        "transform": [gsd, 0.0, x0, 0.0, -gsd, y0],
        "crs": crs,
        "width": width,
        "height": height,
        "dtype": "uint8",
        "count": count,
        "nodata": None,
    }


# =============================================================================
# Rasters
# =============================================================================

def _write_raster(path, data, *, crs, x0, y0, gsd):
    """Write ``data`` (bands, height, width) as a uint8 GeoTIFF whose top-left corner is at (x0, y0)
    in CRS units, and return its path. Bands are tagged red, green, blue, then alpha."""
    count, height, width = data.shape
    with rasterio.open(path, "w", driver="GTiff", height=height, width=width, count=count,
                       dtype="uint8", crs=crs, transform=from_origin(x0, y0, gsd, gsd)) as dst:
        dst.write(data)
        dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha][:count]
    return path


def _random_rgb(seed=0):
    """Return random RGB pixels of shape (3, RASTER_SIZE, RASTER_SIZE). The seed keeps them the
    same from one run to the next."""
    return np.random.default_rng(seed).integers(0, 256, (3, RASTER_SIZE, RASTER_SIZE), dtype=np.uint8)


@pytest.fixture
def rgb_raster(tmp_path):
    """Return the path of a 256x256 RGB raster in UTM zone 18N (EPSG:32618), with 1 m pixels,
    covering x and y from 0 to 256."""
    return _write_raster(tmp_path / "rgb.tif", _random_rgb(), crs="EPSG:32618", x0=0.0, y0=256.0, gsd=1.0)


@pytest.fixture
def rgba_raster(tmp_path):
    """Return the path of the same raster as ``rgb_raster`` with a fourth, alpha band. The left half
    (columns 0 to 127) is fully transparent and black, like the empty edge of an orthomosaic; the
    right half is fully opaque."""
    rgb = _random_rgb()
    rgb[:, :, :TILE_SIZE] = 0
    alpha = np.full((1, RASTER_SIZE, RASTER_SIZE), 255, dtype=np.uint8)
    alpha[:, :, :TILE_SIZE] = 0
    return _write_raster(tmp_path / "rgba.tif", np.concatenate([rgb, alpha]),
                         crs="EPSG:32618", x0=0.0, y0=256.0, gsd=1.0)


@pytest.fixture
def unprojected_raster(tmp_path):
    """Return the path of a 256x256 RGB raster in latitude/longitude (EPSG:4326), near Montreal
    (top-left corner at 73.6 W, 45.5 N), with pixels of 1e-5 degrees (about 1 m). Its UTM zone
    is 18N (EPSG:32618)."""
    return _write_raster(tmp_path / "unprojected.tif", _random_rgb(), crs="EPSG:4326",
                         x0=-73.6, y0=45.5, gsd=1e-5)


@pytest.fixture(scope="session")
def real_raster():
    """Return the path of the small real orthomosaic crop in assets/. Skips the test if the file
    is missing."""
    if not REAL_RASTER_PATH.exists():
        pytest.skip(f"Test raster not found: {REAL_RASTER_PATH}")
    return REAL_RASTER_PATH


@pytest.fixture
def tiles_dir(rgb_raster, tmp_path):
    """Return a folder holding two 128x128 GeoTIFF tiles cut from ``rgb_raster``: tile_0.tif is its
    top-left quarter and tile_1.tif its top-right quarter. Each keeps its georeferencing."""
    out = tmp_path / "tiles"
    out.mkdir()
    with rasterio.open(rgb_raster) as src:
        for i, window in enumerate([Window(0, 0, TILE_SIZE, TILE_SIZE), Window(TILE_SIZE, 0, TILE_SIZE, TILE_SIZE)]):
            meta = src.meta.copy()
            meta.update(height=TILE_SIZE, width=TILE_SIZE, transform=window_transform(window, src.transform))
            with rasterio.open(out / f"tile_{i}.tif", "w", **meta) as dst:
                dst.write(src.read(window=window))
    return out


# =============================================================================
# Labels (in CRS coordinates, over rgb_raster)
# =============================================================================

LABEL_CENTERS = [(20, 20), (65, 65), (120, 120), (200, 200)]


@pytest.fixture
def box_labels(tmp_path):
    """Return the path of a GeoPackage with four axis-aligned boxes over ``rgb_raster`` (like a
    detection dataset), all of class 0."""
    labels = gpd.GeoDataFrame({
        "geometry": [box(x - 10, y - 10, x + 10, y + 10) for x, y in LABEL_CENTERS],
        "class": [0] * len(LABEL_CENTERS),
    }, crs="EPSG:32618")
    path = tmp_path / "box_labels.gpkg"
    labels.to_file(path, driver="GPKG")
    return path


@pytest.fixture
def polygon_labels(tmp_path):
    """Return the path of a GeoPackage with four round polygons over ``rgb_raster`` (like tree crown
    masks), at the same places as ``box_labels``, all of class 0."""
    labels = gpd.GeoDataFrame({
        "geometry": [Point(x, y).buffer(10) for x, y in LABEL_CENTERS],
        "class": [0] * len(LABEL_CENTERS),
    }, crs="EPSG:32618")
    path = tmp_path / "polygon_labels.gpkg"
    labels.to_file(path, driver="GPKG")
    return path
