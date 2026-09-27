"""The names are the same as geodataset's, for many inputs."""

import itertools

import pytest

from geodataset.utils import strip_all_extensions_and_path
from geodataset.utils.file_name_conventions import (
    AoiGeoPackageConvention,
    AoiTilesImageConvention,
    CocoNameConvention,
    GeoPackageNameConvention,
    TileNameConvention,
    validate_and_convert_product_name,
)

from canopyrs1.core.naming import NameConvention as Names

PRODUCTS = ["ortho", "20240131_zf2block4_ms_m3m_rgb", "site_2", "tile_ortho"]
RESOLUTIONS = (
    [dict(ground_resolution=g) for g in (0.045, 0.03, 0.1, 1, 0.1 + 0.2, 1 / 3)]
    + [dict(scale_factor=s) for s in (0.5, 1, 2)]
    + [dict()]
)
FOLDS = ["train", "valid", "test", "infer", "finalpreds"]


RAW_NAMES = [
    "ortho",
    "My Ortho-2024",
    "__a__b__",
    "A B  C",
    "ortho.v2",
    "forêt",
    "",
]


@pytest.mark.parametrize("name", RAW_NAMES)
def test_clean_product_name(name):
    try:
        expected = validate_and_convert_product_name(name)
    except ValueError:
        with pytest.raises(ValueError):
            Names.clean_product_name(name)
    else:
        assert Names.clean_product_name(name) == expected


PATHS = [
    "ortho.tif",
    "a/b/ortho.cog.tif",
    "/data/My Site-2024.tif",
    "x.tar.gz",
]


@pytest.mark.parametrize("path", PATHS)
def test_product_name(path):
    expected = validate_and_convert_product_name(strip_all_extensions_and_path(path))
    assert Names.product_name(path) == expected


def test_tile_names():
    # geodataset can't name a tile with id 0 (it takes 0 for "no id"), so ids start at 1 here.
    combinations = itertools.product(
        PRODUCTS,
        RESOLUTIONS,
        FOLDS + [None],
        [(0, 0), (1024, 2048), (-512, 0)],
        [1, 218],
    )
    for product, resolution, aoi, (col, row), tile_id in combinations:
        kwargs = dict(
            col=col,
            row=row,
            width=1777,
            height=1024,
            tile_id=tile_id,
            aoi=aoi,
            **resolution,
        )
        expected = TileNameConvention.create_name(product_name=product, **kwargs)
        assert Names.tile(product, **kwargs) == expected


def test_other_names():
    for product, resolution, fold in itertools.product(PRODUCTS, RESOLUTIONS, FOLDS):
        old_coco = CocoNameConvention.create_name(product, fold, **resolution)
        old_gpkg = GeoPackageNameConvention.create_name(product, fold, **resolution)
        old_aoi_gpkg = AoiGeoPackageConvention.create_name(product, fold, **resolution)
        old_aoi_image = AoiTilesImageConvention.create_name(product, **resolution)
        assert Names.coco(product, fold, **resolution) == old_coco
        assert Names.gpkg(product, fold, **resolution) == old_gpkg
        assert Names.aoi_gpkg(product, fold, **resolution) == old_aoi_gpkg
        assert Names.aoi_tiles_image(product, **resolution) == old_aoi_image
