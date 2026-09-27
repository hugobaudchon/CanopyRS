"""Checking that a raster can be used as RGB input, from its header only."""

import warnings

import rasterio
from rasterio.enums import ColorInterp

from canopyrs1.core.constants import RGB_BANDS


class RasterValidationError(ValueError):
    """Raised when a raster can't be opened or can't be used as RGB input."""


def validate_rgb_raster(
    path,
    *,
    bands=RGB_BANDS,
    strict_rgb_validation=True,
    require_uint8=True,
):
    """Check that the raster at ``path`` (a local file or a URL) can be used as RGB input, reading
    only its header:

    - it can be opened, and has the ``bands`` to read (the first three by default);
    - those bands are tagged red, green and blue. With ``strict_rgb_validation=False``, wrong tags
      only give a warning, for files whose bands are RGB but are tagged otherwise;
    - those bands are uint8, unless ``require_uint8`` is False (for rasters that are converted to
      uint8 when read).

    Returns nothing, and raises a RasterValidationError if a check fails.
    """
    try:
        src = rasterio.open(path)
    except rasterio.errors.RasterioIOError as error:
        raise RasterValidationError(f"Can't open the raster {path}: {error}") from error

    with src:
        if max(bands) > src.count:
            raise RasterValidationError(
                f"The raster {path} has {src.count} band(s), but bands {list(bands)} are to be read"
            )
        tags = [src.colorinterp[band - 1] for band in bands]
        expected = [ColorInterp.red, ColorInterp.green, ColorInterp.blue]
        if tags != expected:
            names = ", ".join(tag.name for tag in tags)
            message = (
                f"Bands {list(bands)} of the raster {path} should be red, green, blue, not {names}."
            )
            if strict_rgb_validation:
                raise RasterValidationError(
                    f"{message} If they do hold red, green and blue, pass "
                    f"strict_rgb_validation=False to only warn."
                )
            warnings.warn(message, UserWarning, stacklevel=2)
        dtypes = {src.dtypes[band - 1] for band in bands}
        if require_uint8 and dtypes != {"uint8"}:
            raise RasterValidationError(
                f"Bands {list(bands)} of the raster {path} should be uint8, not "
                f"{', '.join(sorted(dtypes))}"
            )
