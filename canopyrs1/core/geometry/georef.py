"""The georeferencing dict of an image: where its pixels are on the ground, and what they hold.

Every image row stores one in Col.METADATA, whether the image has its own file or is a window into
its parent's file. It holds only plain values, so it is saved to parquet as it is:

    {"transform": [a, b, c, d, e, f],   # pixel -> CRS: x = a*col + b*row + c, y = d*col + e*row + f
     "crs": "EPSG:32618",               # None for an image without georeferencing
     "width": 1024, "height": 1024,     # in pixels
     "count": 3,                        # the number of bands in the file
     "dtype": "uint8",
     "nodata": None}

It also moves geometries between an image's pixel coordinates and its CRS coordinates.
"""

import numpy as np
import shapely
from affine import Affine
from shapely.geometry import Polygon


def make_georef(*, transform, crs, width, height, count, dtype, nodata=None):
    """Return the georeferencing dict of an image from its values. ``transform`` can be an
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


def read_georef(src):
    """Return the georeferencing dict of a whole raster. ``src`` is an open rasterio dataset: a
    file, or a virtual raster such as a WarpedVRT."""
    return make_georef(transform=src.transform, crs=src.crs, width=src.width, height=src.height,
                       count=src.count, dtype=src.dtypes[0], nodata=src.nodata)


def window_georef(georef, *, col_off, row_off, width, height):
    """Return the georeferencing dict of a window of the image described by ``georef``: the
    ``width`` x ``height`` pixels whose top-left pixel is at column ``col_off`` and row ``row_off``.
    The window may reach past the edges of the image."""
    a, b, c, d, e, f = georef["transform"]
    return {
        **georef,
        "transform": [a, b, a * col_off + b * row_off + c, d, e, d * col_off + e * row_off + f],
        "width": int(width),
        "height": int(height),
    }


def _apply_transform(geometry, transform):
    """Return ``geometry`` with the six-number ``transform`` applied to every coordinate."""
    a, b, c, d, e, f = transform
    return shapely.transform(
        geometry, lambda xy: np.column_stack([a * xy[:, 0] + b * xy[:, 1] + c, d * xy[:, 0] + e * xy[:, 1] + f]))


def pixel_to_crs(geometry, georef):
    """Return ``geometry``, given in the pixel coordinates of the image described by ``georef``,
    in the image's CRS coordinates. ``geometry`` can be one shapely geometry, or several (a list,
    an array or a GeoSeries); several geometries are returned as a numpy array, in the same order."""
    return _apply_transform(geometry, georef["transform"])


def crs_to_pixel(geometry, georef):
    """Return ``geometry``, given in the CRS coordinates of the image described by ``georef``, in
    the image's pixel coordinates. It accepts and returns the same types as ``pixel_to_crs``."""
    return _apply_transform(geometry, list(~Affine(*georef["transform"]))[:6])


def get_footprint(georef):
    """Return the area covered by the image described by ``georef``, as a polygon in CRS
    coordinates: its four corners, which form a rotated rectangle if the image is rotated."""
    width, height = georef["width"], georef["height"]
    return pixel_to_crs(Polygon([(0, 0), (width, 0), (width, height), (0, height)]), georef)


def get_bounds(georef):
    """Return the (left, bottom, right, top) CRS coordinates of the smallest north-up rectangle
    that contains the image described by ``georef``."""
    return get_footprint(georef).bounds
