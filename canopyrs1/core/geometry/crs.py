"""Choosing a CRS in metres for data given in latitude and longitude."""

from pyproj import CRS

from canopyrs1.core.types import CRSLike


def get_utm_crs(lon: float, lat: float) -> str:
    """Return the UTM CRS of the zone that contains the point at longitude ``lon`` and latitude
    ``lat`` (in degrees), as a string: "EPSG:326xx" north of the equator, "EPSG:327xx" south of
    it."""
    zone = min(int((lon + 180) / 6) + 1, 60)  # longitude 180 belongs to zone 60
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def get_projected_crs(
    crs: CRSLike,
    bounds: tuple[float, float, float, float],
) -> CRSLike:
    """Return ``crs`` unchanged if it is projected (in metres or feet, not degrees). Otherwise,
    return the UTM CRS of the centre of ``bounds``, the (left, bottom, right, top) of the data in
    ``crs``. ``crs`` can be a string, a rasterio CRS or a pyproj CRS."""
    if CRS.from_user_input(crs).is_projected:
        return crs
    left, bottom, right, top = bounds
    return get_utm_crs((left + right) / 2, (bottom + top) / 2)
