"""Telling points, boxes and masks apart, and cleaning up polygons (boxes and masks): repairing
them, and removing holes or small parts.

Every cleanup function returns one part as a Polygon, several parts as a MultiPolygon, and nothing
as an empty Polygon.
"""

import geopandas as gpd
import numpy as np
import shapely
from pyproj import CRS
from shapely import make_valid
from shapely.geometry import MultiPolygon, Polygon

from canopyrs1.core.constants import GeomKind

# A box fills its north-up bounding rectangle; the last 0.1% allows for floating-point rounding.
_BOX_MIN_FILL = 0.999

_POINT_TYPES = (shapely.GeometryType.POINT, shapely.GeometryType.MULTIPOINT)
_POLYGON_TYPES = (shapely.GeometryType.POLYGON, shapely.GeometryType.MULTIPOLYGON)


def _fills_its_bounds(geometries):
    """Return, for each geometry, whether it is a Polygon filling its north-up bounding
    rectangle."""
    area = shapely.area(geometries)
    return (
        (shapely.get_type_id(geometries) == shapely.GeometryType.POLYGON)
        & (area > 0)
        & (area >= _BOX_MIN_FILL * shapely.area(shapely.envelope(geometries)))
    )


def infer_geom_kind(geometries, crs=None, image_crs=None):
    """Return the GeomKind of each geometry in ``geometries`` (a list, an array or a GeoSeries), as
    a numpy array in the same order:

    - a Point or MultiPoint is a point;
    - a Polygon that fills its north-up bounding rectangle is a box: an axis-aligned rectangle;
    - any other Polygon or MultiPolygon is a mask.

    A box is drawn aligned with the image it annotates, so it may only look like one in the image's
    CRS. ``crs`` is the CRS the geometries are in (None for pixel coordinates), and ``image_crs``
    the CRS of their image, if known. A polygon is a box if it fills its bounding rectangle as
    given, or once moved from ``crs`` to ``image_crs``. Both can be strings, pyproj or rasterio
    CRSs.

    Raises a ValueError if a geometry is anything else, such as a line or a missing geometry.
    """
    geometries = np.asarray(geometries, dtype=object)
    types = shapely.get_type_id(geometries)
    unsupported = ~np.isin(types, _POINT_TYPES + _POLYGON_TYPES)
    if unsupported.any():
        found = sorted({"None" if g is None else str(g.geom_type) for g in geometries[unsupported]})
        raise ValueError(f"Expected points or polygons, found {', '.join(found)}")

    is_box = _fills_its_bounds(geometries)
    in_other_crs = (
        crs is not None
        and image_crs is not None
        and CRS.from_user_input(crs) != CRS.from_user_input(image_crs)
    )
    candidates = (types == shapely.GeometryType.POLYGON) & ~is_box
    if in_other_crs and candidates.any():
        in_image_crs = gpd.GeoSeries(geometries[candidates], crs=crs).to_crs(image_crs)
        reprojected = in_image_crs.to_numpy()
        is_box[candidates] = _fills_its_bounds(reprojected)

    kinds = np.full(len(geometries), GeomKind.MASK, dtype=object)
    kinds[np.isin(types, _POINT_TYPES)] = GeomKind.POINT
    kinds[is_box] = GeomKind.BOX
    return kinds


def _from_parts(parts):
    """Return ``parts``, a list of polygons, as one geometry: an empty Polygon if there are none,
    the Polygon itself if there is one, and a MultiPolygon if there are several."""
    if not parts:
        return Polygon()
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def keep_polygon_parts(geometry):
    """Return the polygons in ``geometry``, dropping any line or point. A Polygon or MultiPolygon
    is returned unchanged; a mix of shapes (a GeometryCollection), as produced by repairing or
    intersecting polygons, is reduced to its polygons. Returns an empty Polygon if there are
    none."""
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    parts = []
    for part in getattr(geometry, "geoms", []):
        if isinstance(part, Polygon):
            parts.append(part)
        elif isinstance(part, MultiPolygon):
            parts.extend(part.geoms)
    return _from_parts(parts)


def repair_polygon(geometry):
    """Return ``geometry`` made valid (for example, a self-crossing outline split into the parts it
    encloses), keeping only its polygons. A valid Polygon or MultiPolygon is returned unchanged.
    Returns an empty Polygon if nothing with an area is left."""
    if not geometry.is_valid:
        geometry = make_valid(geometry)
    return keep_polygon_parts(geometry)


def get_largest_part(geometry):
    """Return the part of ``geometry`` with the largest area if it is a MultiPolygon (an empty
    Polygon if it has no parts). Any other geometry is returned unchanged."""
    if not isinstance(geometry, MultiPolygon):
        return geometry
    return max(geometry.geoms, key=lambda part: part.area, default=Polygon())


def remove_holes(geometry):
    """Return ``geometry`` with the holes of each of its polygons filled. Anything other than a
    Polygon or MultiPolygon is returned unchanged."""
    if isinstance(geometry, Polygon):
        return Polygon(geometry.exterior)
    if isinstance(geometry, MultiPolygon):
        return MultiPolygon([Polygon(part.exterior) for part in geometry.geoms])
    return geometry


def remove_small_parts(geometry, min_area):
    """Return ``geometry`` without the parts whose area is below ``min_area``, if it is a
    MultiPolygon. Returns an empty Polygon if every part is too small. Any other geometry, even a
    small Polygon, is returned unchanged: this only removes the extra parts."""
    if not isinstance(geometry, MultiPolygon):
        return geometry
    return _from_parts([part for part in geometry.geoms if part.area >= min_area])
