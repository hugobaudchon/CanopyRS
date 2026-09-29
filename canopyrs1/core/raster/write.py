"""Writing an image's pixels to a GeoTIFF file, with its georef."""

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import ColorInterp

from canopyrs1.core.geometry.georef import Georef
from canopyrs1.core.types import PathLike


def write_tile(
    path: PathLike,
    pixels: np.ndarray,
    georef: Georef,
    colorinterp: list[ColorInterp] | None = None,
    compress: str | None = None,
) -> PathLike:
    """Write ``pixels`` (bands, height, width) to a GeoTIFF file at ``path``, placed on the ground
    by ``georef``, and return ``path``. The file holds the pixels' own band count and dtype, and
    ``georef``'s CRS (none if it has none), transform and nodata value.

    ``colorinterp`` tags each band, for example [ColorInterp.red, ColorInterp.green,
    ColorInterp.blue]; None leaves GDAL's default tags. ``compress`` is None (no compression) or
    "zstd" (lossless). Raises a ValueError if the pixels don't have ``georef``'s width and height.
    """
    # check the size
    count, height, width = pixels.shape
    if (width, height) != (georef["width"], georef["height"]):
        raise ValueError(
            f"The pixels are {width} x {height}, but the georef is "
            f"{georef['width']} x {georef['height']}"
        )

    # choose the compression
    compression = {}
    if compress == "zstd":
        # The predictor stores differences between neighbouring pixels, which compress better:
        # 2 is for integers, 3 for floating-point values.
        predictor = 3 if np.issubdtype(pixels.dtype, np.floating) else 2
        compression = {"compress": "zstd", "predictor": predictor}
    elif compress is not None:
        raise ValueError(f"compress must be None or 'zstd', not {compress!r}")

    # write
    dst = rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype=pixels.dtype,
        crs=georef["crs"],
        transform=Affine(*georef["transform"]),
        nodata=georef["nodata"],
        **compression,
    )
    with dst:
        dst.write(pixels)
        if colorinterp is not None:
            dst.colorinterp = colorinterp
    return path
