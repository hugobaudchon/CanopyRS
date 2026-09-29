"""Reading the pixels of any image, described by its georef, from a raster file, and telling
whether they are worth keeping."""

import math

import numpy as np
from affine import Affine
from pyproj import CRS
from rasterio.enums import ColorInterp, Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window


def _same_crs(a, b):
    """Return whether the CRSs ``a`` and ``b`` (strings, CRS objects or None) are the same."""
    if a is None or b is None:
        return a is None and b is None
    return CRS.from_user_input(a) == CRS.from_user_input(b)


def _is_rotated(transform):
    """Return whether a six-number transform turns the image, so its rows don't run east-west."""
    return transform[1] != 0 or transform[3] != 0


def _file_window(src, georef):
    """Return the window of ``src``'s own pixels covering the image described by ``georef``, in the
    same CRS and with neither rotated. Its offsets and size are fractional when the image's pixels
    don't line up with the file's."""
    a, _, c, _, e, f = georef["transform"]
    inverse = ~src.transform
    col = inverse.a * c + inverse.b * f + inverse.c
    row = inverse.d * c + inverse.e * f + inverse.f
    width = georef["width"] * a / src.transform.a
    height = georef["height"] * e / src.transform.e
    return Window(col, row, width, height)


def _lies_on_file_pixels(window, georef):
    """Return whether ``window`` covers whole file pixels, one image pixel per file pixel."""
    return (
        math.isclose(window.width, georef["width"], abs_tol=1e-6)
        and math.isclose(window.height, georef["height"], abs_tol=1e-6)
        and abs(window.col_off - round(window.col_off)) < 1e-6
        and abs(window.row_off - round(window.row_off)) < 1e-6
    )


def read_window(src, georef, bands=None):
    """Return the pixels of the image described by ``georef``, read from the raster ``src`` (an open
    rasterio dataset), as an array of shape (bands, height, width) in the raster's dtype.

    The image can be any window of the raster, resampled or reprojected (see
    ``get_resampled_georef`` and ``window_georef``):

    - on the raster's own pixels, they are read as they are;
    - in the raster's CRS at another resolution, they are resampled with bilinear interpolation;
    - in another CRS, or rotated, they are reprojected, with bilinear interpolation.

    Pixels outside the raster are its nodata value, or 0 if it has none. ``bands`` are the band
    numbers to read, starting at 1; None reads them all.
    """
    # choose the bands and the value outside the raster
    bands = list(bands) if bands is not None else list(range(1, src.count + 1))
    fill = src.nodata if src.nodata is not None else 0

    # reproject if the CRS differs, or if either is rotated
    reproject = (
        not _same_crs(src.crs, georef["crs"])
        or _is_rotated(src.transform)
        or _is_rotated(georef["transform"])
    )
    if reproject:
        vrt = WarpedVRT(
            src,
            crs=georef["crs"],
            transform=Affine(*georef["transform"]),
            width=georef["width"],
            height=georef["height"],
            resampling=Resampling.bilinear,
        )
        with vrt:
            return vrt.read(bands)

    # read directly if the image's pixels are exactly the file's pixels
    window = _file_window(src, georef)
    if _lies_on_file_pixels(window, georef):
        col, row = round(window.col_off), round(window.row_off)
        window = Window(col, row, georef["width"], georef["height"])
        return src.read(
            bands,
            window=window,
            boundless=True,
            fill_value=fill,
        )
    # otherwise resample (same CRS, another resolution)
    return src.read(
        bands,
        window=window,
        boundless=True,
        fill_value=fill,
        out_shape=(len(bands), georef["height"], georef["width"]),
        resampling=Resampling.bilinear,
    )


def get_alpha_band(src):
    """Return the number (starting at 1) of the alpha band of the raster ``src``, an open rasterio
    dataset, or None if it has none."""
    for number, colour in enumerate(src.colorinterp, start=1):
        if colour == ColorInterp.alpha:
            return number
    return None


def should_skip(pixels, conditions, alpha=None):
    """Return whether an image should be skipped, given its ``pixels`` (bands, height, width) and
    the ``conditions`` it must meet.

    A pixel is empty when it is black (0 in every band), white (255 in every band) or transparent
    (0 in ``alpha``, the image's alpha band as a (height, width) array, if it has one). Pixels
    outside the raster are read as 0, so they count as empty too.

    ``conditions`` is a dict: {"ignore_black_white_alpha_tiles_threshold": 0.75} skips the image
    when at least 75% of its pixels are empty; a threshold of 1 or more never skips. None or an
    empty dict never skips either. Raises a ValueError for any other key.
    """
    # check the conditions
    known_conditions = {"ignore_black_white_alpha_tiles_threshold"}
    if not conditions:
        return False
    unknown_conditions = set(conditions) - known_conditions
    if unknown_conditions:
        raise ValueError(
            f"Unknown skip conditions {sorted(unknown_conditions)}, "
            f"expected {sorted(known_conditions)}"
        )
    threshold = conditions["ignore_black_white_alpha_tiles_threshold"]
    if threshold >= 1:
        return False
    # count the empty pixels
    empty = np.all(pixels == 0, axis=0) | np.all(pixels == 255, axis=0)
    if alpha is not None:
        empty |= alpha == 0
    return bool(empty.mean() >= threshold)
