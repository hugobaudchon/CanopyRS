import pytest
import rasterio
from rasterio.crs import CRS

from canopyrs1.core.geometry.crs import get_projected_crs, get_utm_crs


UTM_ZONES = [
    (-73.6, 45.5, "EPSG:32618"),  # Montreal
    (151.2, -33.9, "EPSG:32756"),  # Sydney
    (-180.0, 10.0, "EPSG:32601"),  # first zone
    (180.0, 10.0, "EPSG:32660"),  # last zone, not a 61st one
    (3.0, 0.0, "EPSG:32631"),  # the equator counts as north
    (-78.0, -1.0, "EPSG:32718"),  # zone edge: -78 is the start of zone 18
]


@pytest.mark.parametrize("lon, lat, expected", UTM_ZONES)
def test_get_utm_crs(lon, lat, expected):
    assert get_utm_crs(lon, lat) == expected


@pytest.mark.parametrize("crs", ["EPSG:32618", CRS.from_epsg(2950)])
def test_a_projected_crs_is_returned_unchanged(crs):
    assert get_projected_crs(crs, (0, 0, 1, 1)) is crs


def test_a_geographic_crs_becomes_utm(unprojected_raster):
    with rasterio.open(unprojected_raster) as src:
        assert get_projected_crs(src.crs, tuple(src.bounds)) == "EPSG:32618"
    assert get_projected_crs("EPSG:4326", (150.0, -34.0, 152.0, -33.0)) == "EPSG:32756"
