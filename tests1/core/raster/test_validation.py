import numpy as np
import pytest
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin

from canopyrs1.core.raster.validation import RasterValidationError, validate_rgb_raster


def _write(path, *, count=3, dtype="uint8", colorinterp=None):
    """Write a small raster and return its path."""
    dst = rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=count,
        dtype=dtype,
        crs="EPSG:32618",
        transform=from_origin(0, 8, 1, 1),
    )
    with dst:
        dst.write(np.zeros((count, 8, 8), dtype))
        if colorinterp is not None:
            dst.colorinterp = colorinterp
    return path


GREY = [ColorInterp.gray, ColorInterp.undefined, ColorInterp.undefined]
RGB = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]


def test_valid_rasters(rgb_raster, rgba_raster, real_raster):
    for path in (rgb_raster, rgba_raster, real_raster):
        validate_rgb_raster(path)


def test_a_missing_or_unreadable_file(tmp_path):
    with pytest.raises(RasterValidationError, match="Can't open the raster"):
        validate_rgb_raster(tmp_path / "missing.tif")
    text = tmp_path / "notes.tif"
    text.write_text("not a raster")
    with pytest.raises(RasterValidationError, match="Can't open the raster"):
        validate_rgb_raster(text)


def test_too_few_bands(tmp_path):
    path = _write(tmp_path / "two.tif", count=2)
    with pytest.raises(RasterValidationError, match="has 2 band"):
        validate_rgb_raster(path)


def test_wrong_colour_tags(tmp_path):
    path = _write(tmp_path / "grey.tif", colorinterp=GREY)
    with pytest.raises(RasterValidationError, match="strict_rgb_validation=False"):
        validate_rgb_raster(path)
    with pytest.warns(UserWarning, match="should be red, green, blue, not gray"):
        validate_rgb_raster(path, strict_rgb_validation=False)


def test_other_bands(tmp_path):
    # Bands 2 to 4 of a 5-band file, tagged red, green, blue.
    tags = [ColorInterp.undefined, *RGB, ColorInterp.undefined]
    path = _write(tmp_path / "five.tif", count=5, colorinterp=tags)
    validate_rgb_raster(path, bands=[2, 3, 4])
    with pytest.raises(RasterValidationError, match="should be red, green, blue"):
        validate_rgb_raster(path)


def test_uint8_is_required_unless_converted(tmp_path):
    path = _write(tmp_path / "wide.tif", dtype="uint16", colorinterp=RGB)
    with pytest.raises(RasterValidationError, match="should be uint8, not uint16"):
        validate_rgb_raster(path)
    validate_rgb_raster(path, require_uint8=False)
