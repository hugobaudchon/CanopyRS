"""
Tile window metadata — the single source of truth for a tile's georeferencing.

A tile stores ``Col.TILE_METADATA``: a JSON-serializable snapshot of the rasterio metadata for its
window (transform, crs, size, ...). The tile's GSD lives in the transform, so it may differ from the
raster's (resampling) — consumers (loader, aggregator) read georeferencing from the tile, never the
raster. Kept serializable (transform as a 6-float list, crs as a string) so it round-trips parquet.
"""

import rasterio
from affine import Affine
from rasterio.windows import transform as _window_transform
from shapely.affinity import affine_transform
from shapely.geometry import box


def window_meta(src, window) -> dict:
    """Serializable rasterio metadata for `window` of the open raster `src`."""
    return {
        "transform": list(_window_transform(window, src.transform))[:6],
        "crs": src.crs.to_string() if src.crs else None,
        "width": int(window.width),
        "height": int(window.height),
        "dtype": src.dtypes[0],
        "count": src.count,
        "nodata": src.nodata,
    }


def serialize_meta(meta: dict) -> dict:
    """Our serializable TILE_METADATA from a rasterio-style metadata dict (Affine transform + CRS
    object) — e.g. geodataset's ``RasterTileMetadata.metadata``. Mirrors ``window_meta``'s format so
    both tilerizing paths produce identical tile metadata."""
    crs = meta.get("crs")
    return {
        "transform": list(meta["transform"])[:6],
        "crs": crs.to_string() if crs else None,
        "width": int(meta["width"]),
        "height": int(meta["height"]),
        "dtype": str(meta["dtype"]),
        "count": int(meta["count"]),
        "nodata": meta.get("nodata"),
    }


def transform_of(meta) -> Affine:
    return Affine(*meta["transform"])


def bounds_of(meta):
    """(left, bottom, right, top) of the tile window in CRS units."""
    return rasterio.transform.array_bounds(meta["height"], meta["width"], transform_of(meta))


def box_of(meta):
    """The tile window extent as a shapely box (CRS)."""
    return box(*bounds_of(meta))


def _shapely_affine(aff: Affine):
    """An ``affine.Affine`` -> shapely ``affine_transform`` 6-param list ``[a, b, d, e, xoff, yoff]``."""
    return [aff.a, aff.b, aff.d, aff.e, aff.c, aff.f]


def pixel_to_crs(geom, meta):
    """Map a tile-pixel geometry to CRS coords via the tile's own transform."""
    return affine_transform(geom, _shapely_affine(transform_of(meta)))


def crs_to_pixel(geom, meta):
    """Map a CRS geometry to the tile's pixel coords (the inverse transform)."""
    return affine_transform(geom, _shapely_affine(~transform_of(meta)))
