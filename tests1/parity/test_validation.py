"""validate_rgb_raster accepts and refuses the same rasters as the old check."""

import warnings

import numpy as np
import pytest
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin

from canopyrs.engine.raster_validation import RasterValidationError as OldError
from canopyrs.engine.raster_validation import validate_raster_rgb_bands

from canopyrs1.core.raster.validation import RasterValidationError, validate_rgb_raster

RGB = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]
CASES = [
    dict(count=3, dtype="uint8", colorinterp=RGB),
    dict(count=4, dtype="uint8", colorinterp=[*RGB, ColorInterp.alpha]),
    dict(count=2, dtype="uint8", colorinterp=None),
    dict(count=3, dtype="uint8", colorinterp=[ColorInterp.gray] + [ColorInterp.undefined] * 2),
    dict(count=3, dtype="uint16", colorinterp=RGB),
    dict(count=3, dtype="float32", colorinterp=RGB),
]


def _passes(check, error, path, **kwargs):
    """Return whether ``check`` accepts the raster at ``path``, ignoring its warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            check(path, **kwargs)
        except error:
            return False
    return True


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("strict", [True, False])
def test_same_verdict(case, strict, tmp_path):
    path = tmp_path / "raster.tif"
    dst = rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=case["count"],
        dtype=case["dtype"],
        crs="EPSG:32618",
        transform=from_origin(0, 8, 1, 1),
    )
    with dst:
        dst.write(np.zeros((case["count"], 8, 8), case["dtype"]))
        if case["colorinterp"] is not None:
            dst.colorinterp = case["colorinterp"]
    old = _passes(validate_raster_rgb_bands, OldError, path, strict_color_interp=strict)
    new = _passes(validate_rgb_raster, RasterValidationError, path, strict_rgb_validation=strict)
    assert new == old
