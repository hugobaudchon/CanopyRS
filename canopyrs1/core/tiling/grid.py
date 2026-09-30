"""Cutting the sources of one place into tiles along the same lines on the ground, as windows:
no pixel is read.

- The pixel grid is the main source's pixels at the requested resolution: its CRS, its pixel size
  and its starting corner.
- The tile windows are the squares the tiles are cut along, laid over the pixel grid.
- A tile is one tile window on one source.
"""

import math
import warnings
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry.base import BaseGeometry

from canopyrs1.core.constants import Col
from canopyrs1.core.geometry.georef import (
    Georef,
    crs_to_pixel,
    get_footprint,
    resize_georef,
    window_georef,
)
from canopyrs1.core.geometry.shapes import get_overlaps
from canopyrs1.core.raster.resampling import get_resampled_georef
from canopyrs1.core.tables.imagery import Sources, Tiles


def _tile_windows(
    pixel_grid: Georef,
    area: BaseGeometry,
    tile_size: int,
    step: int,
) -> list[Georef]:
    """Return the tile windows over ``pixel_grid`` covering ``area`` (in the pixel grid's CRS),
    row after row: ``tile_size`` pixels wide, every ``step`` pixels from the pixel grid's top-left
    pixel, starting before it or going past its last pixel where ``area`` does."""
    # the rows and columns of tile windows the area needs, past rounding errors
    min_col, min_row, max_col, max_row = crs_to_pixel(area, pixel_grid).bounds
    cols = range(math.floor(min_col / step + 1e-9), math.ceil(max_col / step - 1e-9))
    rows = range(math.floor(min_row / step + 1e-9), math.ceil(max_row / step - 1e-9))
    return [
        window_georef(
            pixel_grid, col_off=c * step, row_off=r * step, width=tile_size, height=tile_size
        )
        for r in rows
        for c in cols
    ]


def _footprints_in_pixel_grid(
    sources: Sources,
    pixel_grid: Georef,
    main_source: int,
) -> gpd.GeoSeries:
    """Return every source's footprint in the CRS of ``pixel_grid``, the main source's being the
    pixel grid's own extent."""
    footprints = sources.df.geometry
    if sources.df.crs is not None:
        footprints = footprints.to_crs(pixel_grid["crs"])
    footprints = gpd.GeoSeries(footprints.to_numpy())
    footprints[main_source] = get_footprint(pixel_grid)
    return footprints


def _tile_pixel_sizes(
    sources: Sources,
    footprints: gpd.GeoSeries,
    ground_resolution: float | Sequence[float] | None,
) -> list[float]:
    """Return the pixel size of each source's tiles, in the units of its ``footprints``' CRS:
    ``ground_resolution`` (one value, or one per source), or else each source's own. Warns if a
    single value makes some sources coarser and others finer, which may not be intended."""
    # each source's own pixel size, from its footprint's area
    georefs = sources.df[Col.GEOREF]
    natives = [
        math.sqrt(fp.area / (g["width"] * g["height"])) for fp, g in zip(footprints, georefs)
    ]
    if ground_resolution is None:
        return natives
    if np.ndim(ground_resolution) > 0:
        return [float(value) for value in ground_resolution]

    # a single value, maybe resampling some sources one way and others the other
    factors = [ground_resolution / native for native in natives]
    if max(factors) > 1.01 and min(factors) < 0.99:
        names = [Path(path).name for path in sources.get_disk_paths()]
        details = ", ".join(
            f"{name} ({native:.3g} per pixel, ×{factor:.2g})"
            for name, native, factor in zip(names, natives, factors)
        )
        warnings.warn(
            f"A ground_resolution of {ground_resolution} makes some sources coarser and others "
            f"finer: {details}. Pass one ground_resolution per source if that isn't intended",
            UserWarning,
            stacklevel=3,
        )
    return [float(ground_resolution)] * len(sources)


def _same_ground_on_source(window: Georef, source: Georef, pixel_size: float) -> Georef:
    """Return the georef of the ground of a tile ``window`` on another source: in pixels of about
    ``pixel_size`` (in the window's CRS units), a whole number of them, with the source's bands,
    dtype and nodata."""
    size = math.hypot(window["transform"][0], window["transform"][3])  # the window's pixel size
    width = max(1, round(window["width"] * size / pixel_size))
    height = max(1, round(window["height"] * size / pixel_size))
    resized = resize_georef(window, width, height)
    return {**source, **{key: resized[key] for key in ("transform", "crs", "width", "height")}}


def _tiles_of_windows(
    tile_windows: list[Georef],
    sources: Sources,
    footprints: gpd.GeoSeries,
    pixel_sizes: list[float],
    main_source: int,
) -> Tiles:
    """Return a tile per tile window and source covering it, window after window: the window
    itself on the main source, the same ground on the others (see ``_same_ground_on_source``). Each
    has the window's number as its ``instance_id``, and its source's modality, timestamp and
    bands."""
    # the sources each tile window covers, window after window
    areas = [get_footprint(window) for window in tile_windows]
    pairs = list(zip(*get_overlaps(areas, footprints)))

    # each tile window's georef on each of these sources
    georefs = sources.df[Col.GEOREF]
    tile_georefs = [
        tile_windows[w]
        if s == main_source
        else _same_ground_on_source(tile_windows[w], georefs[s], pixel_sizes[s])
        for w, s in pairs
    ]

    # the tiles, with their source's modality, timestamp and bands
    tile_sources = [s for _, s in pairs]
    per_tile = sources.df.iloc[tile_sources]
    return Tiles.build(
        georef=tile_georefs,
        parent_image_id=tile_sources,
        parent_imagery=sources,
        modality=per_tile[Col.MODALITY].to_list(),
        timestamp=per_tile[Col.TIMESTAMP].to_list(),
        columns={
            Col.BANDS: per_tile[Col.BANDS].to_list(),
            Col.INSTANCE_ID: [w for w, _ in pairs],
        },
    )


def grid_tiles(
    sources: Sources,
    *,
    tile_size: int,
    tile_overlap: float = 0.0,
    main_source: int = 0,
    ground_resolution: float | Sequence[float] | None = None,
    scale_factor: float | None = None,
) -> Tiles:
    """Return tiles of the sources of one place (dates or modalities of it), as windows of them,
    every source cut along the same lines on the ground.

    - The pixel grid is the ``main_source`` resampled at ``ground_resolution`` or
      ``scale_factor`` (see ``get_resampled_georef``).
    - The tile windows are ``tile_size`` x ``tile_size`` pixels of the pixel grid, row after row
      from its top-left corner, each overlapping its neighbours by ``tile_overlap`` of its size.
      They cover every source; the last row and column reach past their edges. With overlapping
      tiles, a source reaching past the main one's top or left edge adds tile windows along that
      edge, which the main source gets tiles in too.
    - Each tile window gives a tile on each source that covers it, all with the window's number
      as their ``instance_id``, and their source's modality, timestamp and bands. The main
      source's tiles are the tile windows themselves; another source's show the same ground at its
      own resolution, or at ``ground_resolution`` (one value, or one per source), in a whole number
      of pixels. Other sources are resampled when read, to line up with the main one.

    Warns if a single ``ground_resolution`` makes some sources coarser and others finer. Raises a
    ValueError if the tile windows can't be laid (a ``tile_size`` under 1, or a ``tile_overlap``
    not moving them by a pixel at least), if ``main_source`` isn't one of the sources, if
    ``scale_factor`` is given for several sources, or if ``ground_resolution`` has another number
    of values than there are sources."""
    # check the tile windows and the resolutions
    step = int((1 - tile_overlap) * tile_size)
    if tile_size < 1 or tile_overlap < 0 or step < 1:
        raise ValueError(
            f"Tile windows need a tile_size of at least 1 and a tile_overlap from 0 to under 1, "
            f"moving them by at least a pixel: got {tile_size} and {tile_overlap}"
        )
    if len(sources) == 0:
        return Tiles.build(georef=[])
    if not 0 <= main_source < len(sources):
        raise ValueError(f"main_source {main_source} isn't one of the {len(sources)} sources")
    if scale_factor is not None and len(sources) > 1:
        raise ValueError("A scale_factor only works for one source: use a ground_resolution")
    several = np.ndim(ground_resolution) > 0
    if several and len(ground_resolution) != len(sources):
        raise ValueError(
            f"Give one ground_resolution per source: got {len(ground_resolution)} for "
            f"{len(sources)} sources"
        )

    # the pixel grid: the main source's pixels at the requested resolution. It sets the CRS, the
    # pixel size and the starting corner the tile windows follow
    main_resolution = ground_resolution[main_source] if several else ground_resolution
    with rasterio.open(sources.get_disk_paths()[main_source]) as src:
        pixel_grid = get_resampled_georef(
            src,
            ground_resolution=main_resolution,
            scale_factor=scale_factor,
        )

    # every source's footprint, in the pixel grid's CRS. For the main source, it's the pixel
    # grid's own extent, so that with a single source the tile windows cover exactly the pixel grid
    source_footprints = _footprints_in_pixel_grid(sources, pixel_grid, main_source)

    # the pixel size of each source's tiles: the requested ground_resolution (the same for every
    # source if a single ground_resolution value is given, one per source otherwise), or else each
    # source's original pixel size
    source_pixel_sizes = _tile_pixel_sizes(sources, source_footprints, ground_resolution)

    # the tile windows over the extended pixel grid, so that every source is fully covered
    tile_windows = _tile_windows(pixel_grid, source_footprints.union_all(), tile_size, step)

    # a tile per tile window and source covering it
    return _tiles_of_windows(
        tile_windows,
        sources,
        source_footprints,
        source_pixel_sizes,
        main_source,
    )
