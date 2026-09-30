"""The SelvaMask AOIs load, and each raster's valid and test AOIs don't overlap."""

import warnings

import geopandas as gpd
import pytest

from canopyrs1.core.aoi.load import load_aois

pytestmark = pytest.mark.integration


def test_selvamask_aois(selvamask_aoi_gpkgs):
    for raster, folds in selvamask_aoi_gpkgs.items():
        crs = gpd.read_file(folds["valid"]).crs
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no overlap between the folds
            aois = load_aois(folds, crs=crs)
        assert list(aois["aoi"]) == ["valid", "test"], raster
        for fold, area in zip(aois["aoi"], aois.geometry):
            assert area.is_valid and area.area > 0, (raster, fold)
            assert area.area == pytest.approx(gpd.read_file(folds[fold]).union_all().area)
