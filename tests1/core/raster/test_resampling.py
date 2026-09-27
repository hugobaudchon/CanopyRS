import json
import warnings

import numpy as np
import pytest
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.warp import transform_bounds

from canopyrs1.core.geometry.georef import get_bounds, read_georef
from canopyrs1.core.raster.resampling import get_resampled_georef


@pytest.fixture
def raster_without_crs(tmp_path):
    """A 64 x 64 RGB GeoTIFF with neither a CRS nor a transform, like a plain photo."""
    path = tmp_path / "photo.tif"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        dst = rasterio.open(path, "w", driver="GTiff", width=64, height=64, count=3, dtype="uint8")
        with dst:
            dst.write(np.zeros((3, 64, 64), np.uint8))
    return path


def _resample(path, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as src:
            return get_resampled_georef(src, **kwargs), read_georef(src)


def test_neither_keeps_the_raster_as_it_is(rgb_raster):
    resampled, raster = _resample(rgb_raster)
    assert resampled == raster
    assert json.loads(json.dumps(resampled)) == resampled


@pytest.mark.parametrize("scale_factor, size", [(0.5, 128), (2, 512), (0.3, 76)])
def test_scale_factor(rgb_raster, scale_factor, size):
    resampled, raster = _resample(rgb_raster, scale_factor=scale_factor)
    assert (resampled["width"], resampled["height"], resampled["crs"]) == (size, size, "EPSG:32618")
    assert resampled["transform"] == pytest.approx([256 / size, 0, 0, 0, -256 / size, 256])
    assert get_bounds(resampled) == pytest.approx(get_bounds(raster))  # same area, other pixels


@pytest.mark.parametrize("ground_resolution, size", [(0.5, 512), (2, 128)])
def test_ground_resolution(rgb_raster, ground_resolution, size):
    resampled, _ = _resample(rgb_raster, ground_resolution=ground_resolution)
    assert (resampled["width"], resampled["height"], resampled["crs"]) == (size, size, "EPSG:32618")
    expected = [ground_resolution, 0, 0, 0, -ground_resolution, 256]
    assert resampled["transform"] == pytest.approx(expected)


def test_a_raster_in_latitude_longitude_moves_to_utm(unprojected_raster):
    resampled, raster = _resample(unprojected_raster, ground_resolution=0.5)
    assert resampled["crs"] == "EPSG:32618"
    a, b, _, d, e, _ = resampled["transform"]
    assert (a, b, d, e) == pytest.approx((0.5, 0, 0, -0.5))
    # It covers the raster's area once moved to UTM, to within a pixel.
    with rasterio.open(unprojected_raster) as src:
        left, bottom, right, top = transform_bounds(src.crs, "EPSG:32618", *src.bounds)
    assert resampled["width"] * 0.5 == pytest.approx(right - left, abs=1)
    assert resampled["height"] * 0.5 == pytest.approx(top - bottom, abs=1)
    # With a scale factor, it stays in latitude and longitude.
    assert _resample(unprojected_raster, scale_factor=0.5)[0]["crs"] == "EPSG:4326"


def test_other_properties_are_kept(rgba_raster):
    resampled, raster = _resample(rgba_raster, ground_resolution=0.5)
    kept = {k: resampled[k] for k in ("count", "dtype", "nodata")}
    assert kept == {"count": 4, "dtype": "uint8", "nodata": None}


def test_a_raster_without_crs(raster_without_crs):
    resampled, _ = _resample(raster_without_crs, scale_factor=0.5)
    assert resampled["crs"] is None and (resampled["width"], resampled["height"]) == (32, 32)
    assert resampled["transform"] == pytest.approx([2, 0, 0, 0, 2, 0])  # photo pixels, rows down
    with pytest.raises(ValueError, match="has no CRS"):
        _resample(raster_without_crs, ground_resolution=0.5)


def test_not_both(rgb_raster):
    with pytest.raises(ValueError, match="not both"):
        _resample(rgb_raster, ground_resolution=0.5, scale_factor=0.5)
