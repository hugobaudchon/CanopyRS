"""assign_to_aois gives the (tile, AOI) pairs geodataset's AOIFromPackageForTiles gives, on the
real crop cut into AOIs in several ways (a slanted cut, overlapping AOIs, a hole and several parts,
AOIs smaller than a tile), and the same masks away from the AOIs' edges. geodataset
drew its masks with cv2.fillPoly on coordinates truncated to whole pixels, not by pixel centres,
so the pixels along the cut can differ."""

import os
import warnings
from contextlib import redirect_stdout

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import shapely
from shapely.geometry import Point, Polygon, box

from geodataset.aoi import AOIFromPackageConfig
from geodataset.aoi.aoi_from_package import AOIFromPackageForTiles
from geodataset.tilerize import RasterTilerizer

from canopyrs1.core.aoi.assign import assign_to_aois, get_usable_mask
from canopyrs1.core.aoi.load import load_aois
from canopyrs1.core.constants import Col
from canopyrs1.core.geometry.georef import crs_to_pixel, get_footprint, read_georef, window_georef
from canopyrs1.core.tables.imagery import Tiles


def _slanted_cut(georef):
    """Two AOIs meeting along a slanted line."""
    footprint = get_footprint(georef)
    minx, miny, maxx, maxy = footprint.bounds
    width = maxx - minx
    left = Polygon(
        [(minx, miny), (minx + 0.43 * width, miny), (minx + 0.61 * width, maxy), (minx, maxy)]
    )
    return {"left": left, "right": footprint.difference(left)}


def _overlapping(georef):
    """Two AOIs sharing a band of 20% of the raster's width."""
    minx, miny, maxx, maxy = get_footprint(georef).bounds
    width = maxx - minx
    return {
        "left": box(minx, miny, minx + 0.6 * width, maxy),
        "right": box(minx + 0.4 * width, miny, maxx, maxy),
    }


def _hole_and_parts(georef):
    """The raster but a disc, and the disc with a box in a corner."""
    footprint = get_footprint(georef)
    minx, miny, maxx, maxy = footprint.bounds
    disc = Point((minx + maxx) / 2, (miny + maxy) / 2).buffer((maxx - minx) / 6)
    corner = box(minx, miny, minx + (maxx - minx) / 10, miny + (maxy - miny) / 10)
    return {"train": footprint.difference(disc), "test": disc.union(corner)}


def _small(georef):
    """An AOI of 5 x 5 m, smaller than a tile, and a sliver 0.3 pixels wide along the left edge:
    most tiles are in neither."""
    minx, miny, maxx, maxy = get_footprint(georef).bounds
    x, y = minx + 0.3 * (maxx - minx), miny + 0.3 * (maxy - miny)
    pixel = georef["transform"][0]
    return {"small": box(x, y, x + 5, y + 5), "sliver": box(minx, miny, minx + 0.3 * pixel, maxy)}


LAYOUTS = [_slanted_cut, _overlapping, _hole_and_parts, _small]


def _aoi_files(layout, georef, tmp_path):
    """Write the AOIs of ``layout`` over the raster of ``georef``, and return {name: path}."""
    paths = {}
    for name, area in layout(georef).items():
        paths[name] = tmp_path / f"{name}.gpkg"
        gpd.GeoDataFrame(geometry=[area], crs=georef["crs"]).to_file(paths[name])
    return paths


@pytest.mark.parametrize("layout", LAYOUTS)
def test_same_assignment_as_geodataset(layout, real_raster, tmp_path):
    with rasterio.open(real_raster) as src:
        raster = read_georef(src)
    paths = _aoi_files(layout, raster, tmp_path)

    # geodataset: its grid of tiles, then its AOI assignment
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        tilerizer = RasterTilerizer(
            raster_path=real_raster,
            output_path=tmp_path / "geodataset",
            tile_size=256,
            tile_overlap=0.25,
            aois_config=AOIFromPackageConfig(dict(paths)),
            ignore_black_white_alpha_tiles_threshold=1.0,
            temp_dir=tmp_path,
        )
        grid = tilerizer._create_tiles()
        old_aois_tiles, _ = AOIFromPackageForTiles(
            tiles=grid,
            tile_coordinate_step=tilerizer.tile_coordinate_step,
            associated_raster=tilerizer.raster,
            global_aoi=None,
            aois_config=AOIFromPackageConfig(dict(paths)),
            ground_resolution=None,
            scale_factor=None,
        ).get_aoi_tiles()
    old = {}
    for aoi, tiles in old_aois_tiles.items():
        for tile in tiles:
            shape = (tile.metadata["height"], tile.metadata["width"])
            old[(tile.col, tile.row, aoi)] = tile.mask if tile.mask is not None else np.ones(shape)

    # canopyrs1: the same grid, then assign_to_aois
    georefs = [
        window_georef(
            raster,
            col_off=tile.col,
            row_off=tile.row,
            width=tile.metadata["width"],
            height=tile.metadata["height"],
        )
        for tile in grid
    ]
    tiles = Tiles.build(georef=georefs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the overlapping layout
        aois = load_aois(paths, tiles.df.crs)
    assigned = assign_to_aois(tiles, aois)
    new = {}  # (col, row, aoi): (mask, georef)
    for grid_id, aoi, geometry, georef in zip(
        assigned.df[Col.PARENT_IMAGE_ID],
        assigned.df[Col.AOI],
        assigned.df.geometry,
        assigned.df[Col.GEOREF],
    ):
        mask = get_usable_mask(geometry, assigned.df.crs, georef)
        new[(grid[grid_id].col, grid[grid_id].row, aoi)] = (mask, georef)

    # The same tiles in the same AOIs.
    assert set(new) == set(old)

    # The same pixels, but within a pixel of an AOI's edge.
    cut = shapely.union_all([area.boundary for area in aois.geometry])
    different, total = 0, 0
    for key, (mask, georef) in new.items():
        wrong = np.argwhere(mask.astype(bool) != old[key].astype(bool))
        different, total = different + len(wrong), total + mask.size
        if len(wrong):
            centres = shapely.points(wrong[:, 1] + 0.5, wrong[:, 0] + 0.5)
            distances = shapely.distance(centres, crs_to_pixel(cut, georef))
            assert distances.max() <= 1.5, key
    assert different < 0.01 * total
