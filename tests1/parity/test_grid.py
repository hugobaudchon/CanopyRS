"""grid_tiles gives the tiles geodataset's RasterTilerizer gives, without skipping any: the same
windows, in the same order, with the same georeferencing."""

import os
from contextlib import redirect_stdout

import pytest

from geodataset.tilerize import RasterTilerizer

from canopyrs1.core.constants import Col
from canopyrs1.core.tables.imagery import Sources
from canopyrs1.core.tiling.grid import grid_tiles

GRIDS = [
    dict(tile_size=512, tile_overlap=0.0),
    dict(tile_size=512, tile_overlap=0.25),
    dict(tile_size=300, tile_overlap=0.5, ground_resolution=0.1),
    dict(tile_size=256, tile_overlap=0.1, scale_factor=0.5),
]


def _geodataset_tiles(raster_path, tmp_path, grid):
    """Return geodataset's tiles for ``grid``, none skipped (its threshold of 1 or more)."""
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        tilerizer = RasterTilerizer(
            raster_path=raster_path,
            output_path=tmp_path / "geodataset",
            ignore_black_white_alpha_tiles_threshold=1.0,
            temp_dir=tmp_path,
            **grid,
        )
        return tilerizer._create_tiles()


@pytest.mark.parametrize("grid", GRIDS)
def test_same_grid_as_geodataset_on_the_real_crop(grid, real_raster, tmp_path):
    old = _geodataset_tiles(real_raster, tmp_path, grid)
    new = grid_tiles(Sources.from_paths(real_raster), **grid)
    assert len(new) == len(old)
    for tile, georef in zip(old, new.df[Col.GEOREF]):
        assert (georef["width"], georef["height"]) == (
            tile.metadata["width"],
            tile.metadata["height"],
        )
        assert georef["transform"] == pytest.approx(list(tile.metadata["transform"])[:6], abs=1e-6)
        assert georef["crs"] == tile.metadata["crs"].to_string()


def test_same_grid_as_geodataset_in_lat_lon(unprojected_raster, tmp_path):
    grid = dict(tile_size=128, tile_overlap=0.25, ground_resolution=1.0)
    old = _geodataset_tiles(unprojected_raster, tmp_path, grid)
    new = grid_tiles(Sources.from_paths(unprojected_raster), **grid)
    assert len(new) == len(old)
    for tile, georef in zip(old, new.df[Col.GEOREF]):
        assert georef["transform"] == pytest.approx(list(tile.metadata["transform"])[:6], abs=1e-6)
        assert georef["crs"] == tile.metadata["crs"].to_string()
