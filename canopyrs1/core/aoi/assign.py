"""Assigning tiles to the AOIs they are in, and masking the pixels outside."""

import geopandas as gpd
import numpy as np
from shapely.geometry.base import BaseGeometry

from canopyrs1.core.constants import Col
from canopyrs1.core.geometry.georef import Georef, crs_to_pixel
from canopyrs1.core.geometry.masks import polygon_to_mask
from canopyrs1.core.geometry.shapes import keep_polygon_parts
from canopyrs1.core.tables.imagery import Imagery
from canopyrs1.core.types import CRSLike


def assign_to_aois(tiles: Imagery, aois: gpd.GeoDataFrame) -> Imagery:
    """Return a copy of each tile per AOI it overlaps (see ``load_aois``), as a table of the same
    type whose parent is ``tiles``: its ``aoi`` is the AOI's name, and its geometry the part of the
    tile inside it. Tiles outside every AOI, or only touching one, are left out. Raises a ValueError
    if ``aois`` isn't in the CRS of ``tiles``."""
    if aois.crs != tiles.df.crs:
        raise ValueError(f"The AOIs are in {aois.crs}, the tiles in {tiles.df.crs}")

    # the part of each tile in each AOI it overlaps, in the AOIs' order, then the tiles'
    aoi_ids, tile_ids = tiles.df.sindex.query(aois.geometry, predicate="intersects")
    order = np.lexsort((tile_ids, aoi_ids))
    aoi_ids, tile_ids = aoi_ids[order], tile_ids[order]
    parts = tiles.df.geometry.iloc[tile_ids].intersection(aois.geometry.iloc[aoi_ids], align=False)
    parts = parts.map(keep_polygon_parts)  # without the lines where they only touch
    kept = (parts.area > 0).to_numpy()

    # a copy of each tile per AOI, cut to the AOI
    names = aois["aoi"].to_numpy()[aoi_ids[kept]]
    geometry = parts[kept].to_numpy()
    return tiles.select(tile_ids[kept], columns={Col.AOI: names, Col.GEOMETRY: geometry})


def get_usable_mask(geometry: BaseGeometry, crs: CRSLike | None, georef: Georef) -> np.ndarray:
    """Return the mask of an image's usable pixels: 1 where a pixel's centre is inside
    ``geometry`` (its row's geometry, in the table's ``crs``), 0 elsewhere. ``georef`` is the
    image's."""
    if crs is not None:
        geometry = gpd.GeoSeries([geometry], crs=crs).to_crs(georef["crs"]).iloc[0]
    return polygon_to_mask(crs_to_pixel(geometry, georef), georef["height"], georef["width"])
