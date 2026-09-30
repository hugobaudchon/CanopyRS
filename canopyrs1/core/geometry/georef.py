"""The georef of an image: a dict saying where its pixels are on the ground, and what they hold.

Every image row stores one in Col.GEOREF, whether the image has its own file or is a window into
its parent's file. It holds only plain values, so it is saved to parquet as it is:

    {"transform": [a, b, c, d, e, f],   # pixel -> CRS: x = a*col + b*row + c, y = d*col + e*row + f
     "crs": "EPSG:32618",               # None for an image without georeferencing
     "width": 1024, "height": 1024,     # in pixels
     "count": 3,                        # the number of bands in the file
     "dtype": "uint8",
     "nodata": None}

It also moves geometries between an image's pixel coordinates and its CRS coordinates.
"""

from collections.abc import Sequence
from typing import TypedDict

import geopandas as gpd
import numpy as np
import shapely
from affine import Affine
from rasterio.io import DatasetReader
from rasterio.vrt import WarpedVRT
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from canopyrs1.core.types import CRSLike, Geometries


class Georef(TypedDict):
    """The georef of an image (see this module's docstring)."""

    transform: list[float]
    crs: str | None
    width: int
    height: int
    count: int
    dtype: str
    nodata: float | None


def make_georef(
    *,
    transform: Affine | Sequence[float],
    crs: CRSLike | None,
    width: int,
    height: int,
    count: int,
    dtype: str | np.dtype,
    nodata: float | None = None,
) -> Georef:
    """Return the georef of an image from its values. ``transform`` can be an
    ``affine.Affine`` or its first six numbers, and ``crs`` a rasterio CRS, a string, or None."""
    return {
        "transform": [float(v) for v in list(transform)[:6]],
        "crs": None if crs is None else (crs if isinstance(crs, str) else crs.to_string()),
        "width": int(width),
        "height": int(height),
        "count": int(count),
        "dtype": str(dtype),
        "nodata": None if nodata is None else float(nodata),
    }


def read_georef(src: DatasetReader | WarpedVRT) -> Georef:
    """Return the georef of a whole raster. ``src`` is an open rasterio dataset: a
    file, or a virtual raster such as a WarpedVRT."""
    return make_georef(
        transform=src.transform,
        crs=src.crs,
        width=src.width,
        height=src.height,
        count=src.count,
        dtype=src.dtypes[0],
        nodata=src.nodata,
    )


def window_georef(
    georef: Georef,
    *,
    col_off: int,
    row_off: int,
    width: int,
    height: int,
) -> Georef:
    """Return the georef of a window of the image described by ``georef``: the
    ``width`` x ``height`` pixels whose top-left pixel is at column ``col_off`` and row ``row_off``.
    The window may reach past the edges of the image."""
    a, b, c, d, e, f = georef["transform"]
    return {
        **georef,
        "transform": [a, b, a * col_off + b * row_off + c, d, e, d * col_off + e * row_off + f],
        "width": int(width),
        "height": int(height),
    }


def resize_georef(georef: Georef, width: int, height: int) -> Georef:
    """Return the georef of the same ground as ``georef``, in ``width`` x ``height`` pixels: its
    pixels are stretched to fit."""
    a, b, c, d, e, f = georef["transform"]
    x_step, y_step = georef["width"] / width, georef["height"] / height  # old pixels per new pixel
    return {
        **georef,
        "transform": [a * x_step, b * y_step, c, d * x_step, e * y_step, f],
        "width": int(width),
        "height": int(height),
    }


def _apply_transform(
    geometry: BaseGeometry | Geometries,
    transform: Sequence[float],
) -> BaseGeometry | np.ndarray:
    """Return ``geometry`` with the six-number ``transform`` applied to every coordinate."""
    a, b, c, d, e, f = transform

    def apply(xy: np.ndarray) -> np.ndarray:
        x, y = xy[:, 0], xy[:, 1]
        return np.column_stack([a * x + b * y + c, d * x + e * y + f])

    return shapely.transform(geometry, apply)


def pixel_to_crs(geometry: BaseGeometry | Geometries, georef: Georef) -> BaseGeometry | np.ndarray:
    """Return ``geometry``, given in the pixel coordinates of the image described by ``georef``,
    in the image's CRS coordinates. ``geometry`` can be one shapely geometry, or several (a list,
    an array or a GeoSeries); several geometries are returned as a numpy array, in the same
    order."""
    return _apply_transform(geometry, georef["transform"])


def crs_to_pixel(geometry: BaseGeometry | Geometries, georef: Georef) -> BaseGeometry | np.ndarray:
    """Return ``geometry``, given in the CRS coordinates of the image described by ``georef``, in
    the image's pixel coordinates. It accepts and returns the same types as ``pixel_to_crs``."""
    return _apply_transform(geometry, list(~Affine(*georef["transform"]))[:6])


def get_pixel_footprint(georef: Georef) -> Polygon:
    """Return the area covered by the image described by ``georef``, in its pixel coordinates:
    the box from (0, 0) to (width, height)."""
    width, height = georef["width"], georef["height"]
    return Polygon([(0, 0), (width, 0), (width, height), (0, height)])


def get_footprint(georef: Georef) -> Polygon:
    """Return the area covered by the image described by ``georef``, as a polygon in CRS
    coordinates: its four corners, which form a rotated rectangle if the image is rotated."""
    return pixel_to_crs(get_pixel_footprint(georef), georef)


def get_footprints(georefs: Sequence[Georef], crs: CRSLike | None) -> gpd.GeoSeries:
    """Return the footprints of the images described by ``georefs``, as a GeoSeries in ``crs``:
    each footprint is moved from its image's own CRS into ``crs``. ``crs`` is None for images in
    pixel coordinates. Raises a ValueError if some images have a CRS and others don't."""
    # check the CRSs
    crss = {georef["crs"] for georef in georefs} | {crs}
    if None in crss and len(crss) > 1:
        raise ValueError("Images with and without a CRS can't be together")

    # make the footprints, and move those in another CRS
    polygons = [get_footprint(georef) for georef in georefs]
    footprints = gpd.GeoSeries(polygons, crs=crs)
    for other_crs in crss - {crs}:
        rows = [i for i, georef in enumerate(georefs) if georef["crs"] == other_crs]
        moved = gpd.GeoSeries([polygons[i] for i in rows], crs=other_crs).to_crs(crs)
        footprints.iloc[rows] = moved.values
    return footprints


def get_bounds(georef: Georef) -> tuple[float, float, float, float]:
    """Return the (left, bottom, right, top) CRS coordinates of the smallest north-up rectangle
    that contains the image described by ``georef``."""
    return get_footprint(georef).bounds
