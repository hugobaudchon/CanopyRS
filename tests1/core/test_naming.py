from pathlib import Path

import pytest

from canopyrs1.core.naming import NameConvention as Names

SELVAMASK_TILE = "20240131_zf2block4_ms_m3m_rgb_tile_test_gr0p045_2220_3108_1777_1777_218.tif"
SELVAMASK_COCO = "20240131_zf2block4_ms_m3m_rgb_coco_gr0p045_test.json"


# =============================================================================
# Product names and resolution tags
# =============================================================================

CLEANED_NAMES = [
    ("ortho", "ortho"),
    ("My Ortho-2024", "my_ortho_2024"),
    ("__a__b__", "a_b"),
    ("20240130_zf2tower_m3m_rgb", "20240130_zf2tower_m3m_rgb"),
]


@pytest.mark.parametrize("name, cleaned", CLEANED_NAMES)
def test_clean_product_name(name, cleaned):
    assert Names.clean_product_name(name) == cleaned


@pytest.mark.parametrize("name", ["ortho.v2", "forêt", "", "___", "a/b"])
def test_clean_product_name_refuses_other_characters(name):
    with pytest.raises(ValueError, match="may only hold letters and digits"):
        Names.clean_product_name(name)


PRODUCT_NAMES = [
    ("ortho.tif", "ortho"),
    ("a/b/ortho.cog.tif", "ortho"),
    (Path("/data/My Site-2024.tif"), "my_site_2024"),
    ("20240130_zf2tower_m3m_rgb_test_crop.tif", "20240130_zf2tower_m3m_rgb_test_crop"),
]


@pytest.mark.parametrize("path, product_name", PRODUCT_NAMES)
def test_product_name(path, product_name):
    assert Names.product_name(path) == product_name


RESOLUTION_TAGS = [
    (None, 0.045, "gr0p045"),
    (None, 1, "gr1p0"),
    (None, 1e-5, "gr0p00001"),  # never in scientific notation
    (0.5, None, "sf0p5"),
    (2, None, "sf2p0"),
    (None, None, "sf1p0"),  # the raster's own resolution
]


@pytest.mark.parametrize("scale_factor, ground_resolution, tag", RESOLUTION_TAGS)
def test_resolution_tag(scale_factor, ground_resolution, tag):
    assert Names.resolution_tag(scale_factor, ground_resolution) == tag


# =============================================================================
# File names
# =============================================================================


def test_tile():
    name = Names.tile(
        "20240131_zf2block4_ms_m3m_rgb",
        col=2220,
        row=3108,
        width=1777,
        height=1777,
        tile_id=218,
        aoi="test",
        ground_resolution=0.045,
    )
    assert name == SELVAMASK_TILE


def test_tile_edge_cases():
    # The first tile has id 0; a tile may start past the raster's top-left edge; no AOI is "noaoi".
    name = Names.tile("ortho", col=-512, row=-8, width=4, height=4, tile_id=0)
    assert name == "ortho_tile_noaoi_sf1p0_-512_-8_4_4_0.tif"


def test_coco_and_gpkg():
    coco = Names.coco("20240131_zf2block4_ms_m3m_rgb", "test", ground_resolution=0.045)
    assert coco == SELVAMASK_COCO
    gpkg = Names.gpkg("ortho", "finalpreds", ground_resolution=0.045)
    assert gpkg == "ortho_gr0p045_finalpreds.gpkg"
    assert Names.gpkg("ortho", "finalpreds") == "ortho_sf1p0_finalpreds.gpkg"


def test_aoi_names():
    assert Names.aoi_gpkg("ortho", "valid", scale_factor=0.5) == "ortho_aoi_sf0p5_valid.gpkg"
    assert Names.aoi_tiles_image("ortho", ground_resolution=0.045) == "ortho_aoistiles_gr0p045.png"


AMBIGUOUS_NAMES = [
    lambda: Names.tile("ortho", col=0, row=0, width=4, height=4, tile_id=1, aoi="train_1"),
    lambda: Names.coco("ortho", "train_2"),
    lambda: Names.gpkg("ortho", "final-preds"),
    lambda: Names.aoi_gpkg("ortho", ""),
    lambda: Names.coco("Not Cleaned", "train"),
]


@pytest.mark.parametrize("build", AMBIGUOUS_NAMES)
def test_names_refuse_parts_that_would_make_them_ambiguous(build):
    with pytest.raises(ValueError, match="may only hold letters and digits"):
        build()


AOI_NAMES = [("train", True), ("test2", True), ("train_1", False), ("", False), ("tést", False)]


@pytest.mark.parametrize("name, valid", AOI_NAMES)
def test_check_aoi(name, valid):
    if valid:
        assert Names.check_aoi(name) == name
    else:
        with pytest.raises(ValueError, match="The AOI"):
            Names.check_aoi(name)
