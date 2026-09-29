"""Resampling a raster at the requested resolution, in a CRS in metres, without reading its pixels.

The result is the georef of the resampled raster. Its tiles are windows of it (see
``window_georef``), and each window's pixels are read from the raster file only when needed.
"""

from rasterio.warp import calculate_default_transform

from canopyrs1.core.geometry.crs import get_projected_crs
from canopyrs1.core.geometry.georef import make_georef, read_georef


def get_resampled_georef(src, *, ground_resolution=None, scale_factor=None):
    """Return the georef of the raster ``src`` (an open rasterio dataset) once resampled:

    - ``ground_resolution``: resamples to pixels of that many metres. A raster in latitude and
      longitude is moved to the UTM zone of its centre, so that its pixels can be in metres.
    - ``scale_factor``: resamples to its own pixels scaled by that factor (0.5 gives pixels twice
      as large), in its own CRS.
    - neither: as it is, like a scale factor of 1.

    Raises a ValueError if both are given, or if ``ground_resolution`` is given for a raster
    without a CRS (its pixels have no size in metres).
    """
    if ground_resolution and scale_factor:
        raise ValueError("Give a ground_resolution or a scale_factor, not both")
    georef = read_georef(src)

    # at a ground resolution, in a CRS in metres
    if ground_resolution:
        if src.crs is None:
            raise ValueError(
                f"{src.name} has no CRS, so its pixels have no size in metres: "
                f"use a scale_factor instead of a ground_resolution"
            )
        crs = get_projected_crs(src.crs, tuple(src.bounds))
        transform, width, height = calculate_default_transform(
            src.crs,
            crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=ground_resolution,
        )
        return make_georef(
            transform=transform,
            crs=crs,
            width=width,
            height=height,
            count=src.count,
            dtype=src.dtypes[0],
            nodata=src.nodata,
        )

    # at a scale factor, in its own CRS
    if scale_factor:
        width, height = int(src.width * scale_factor), int(src.height * scale_factor)
        a, b, c, d, e, f = georef["transform"]
        x_step, y_step = src.width / width, src.height / height  # raster pixels per new pixel
        return {
            **georef,
            "transform": [a * x_step, b * y_step, c, d * x_step, e * y_step, f],
            "width": width,
            "height": height,
        }

    # as it is
    return georef
