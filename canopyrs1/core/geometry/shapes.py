"""Cleaning up polygons (boxes and masks): repairing them, and removing holes or small parts.

Every function here returns one part as a Polygon, several parts as a MultiPolygon, and nothing as
an empty Polygon.
"""

from shapely import make_valid
from shapely.geometry import MultiPolygon, Polygon


def _from_parts(parts):
    """Return ``parts``, a list of polygons, as one geometry: an empty Polygon if there are none,
    the Polygon itself if there is one, and a MultiPolygon if there are several."""
    if not parts:
        return Polygon()
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def keep_polygon_parts(geometry):
    """Return the polygons in ``geometry``, dropping any line or point. A Polygon or MultiPolygon
    is returned unchanged; a mix of shapes (a GeometryCollection), as produced by repairing or
    intersecting polygons, is reduced to its polygons. Returns an empty Polygon if there are none."""
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
