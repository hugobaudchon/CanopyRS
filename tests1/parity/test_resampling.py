"""The resampled georef is the same as the one geodataset's read_raster resamples the raster
onto."""

import pytest
import rasterio

from geodataset.utils import read_raster

from canopyrs1.core.raster.resampling import get_resampled_georef

SETTINGS = [
    dict(),
    dict(scale_factor=0.5),
    dict(scale_factor=0.3),
    dict(scale_factor=2),
    dict(ground_resolution=0.5),
    dict(ground_resolution=0.3),
    dict(ground_resolution=1.7),
]


@pytest.mark.parametrize("raster", ["rgb_raster", "rgba_raster", "unprojected_raster"])
@pytest.mark.parametrize("settings", SETTINGS)
def test_same_georef_as_read_raster(raster, settings, request, tmp_path):
    path = request.getfixturevalue(raster)
    _, profile, _, _, _ = read_raster(str(path), temp_dir=tmp_path, **settings)
    with rasterio.open(path) as src:
        resampled = get_resampled_georef(src, **settings)
    assert (resampled["width"], resampled["height"]) == (profile["width"], profile["height"])
    assert resampled["transform"] == pytest.approx(list(profile["transform"])[:6])
    assert resampled["crs"] == profile["crs"].to_string()
    assert (resampled["count"], resampled["dtype"]) == (profile["count"], profile["dtype"])
