"""The file names of tiles, COCO files and GeoPackages.

These names are a public contract: published datasets and saved runs use them, so they must not
change. They are only written, never read back: files are identified by their contents and their
georeferencing, not by their names.
"""

import re
from pathlib import Path

import numpy as np

_WORD = r"[a-zA-Z0-9]+"
_PRODUCT_NAME = rf"{_WORD}(?:_{_WORD})*"


def _check(value, pattern, what):
    """Return ``value``, or raise a ValueError if it doesn't fully match ``pattern``. An underscore
    in a fold or AOI, for example, would make the name impossible to split into its parts."""
    if not re.fullmatch(pattern, str(value)):
        raise ValueError(
            f"{what} {value!r} may only hold letters and digits"
            + (" and single underscores" if pattern == _PRODUCT_NAME else "")
        )
    return value


def _number_tag(value):
    """Return ``value`` with "p" for the decimal point, never in scientific notation: 0.045 gives
    "0p045", 2 gives "2p0"."""
    return np.format_float_positional(float(value), trim="0").replace(".", "p")


class NameConvention:
    """Builds every file name. A name holds the product name (the raster's file name, cleaned up),
    a resolution tag, and the details of the file."""

    @staticmethod
    def clean_product_name(name):
        """Return ``name`` cleaned up to be used in file names: lowercase, with spaces and dashes
        turned into single underscores, and no underscore at either end. Raises a ValueError if it
        then holds anything other than letters, digits and underscores."""
        cleaned = re.sub(r"_+", "_", name.replace(" ", "_").replace("-", "_").lower()).strip("_")
        return _check(cleaned, _PRODUCT_NAME, f"The product name {name!r}, cleaned up to")

    @staticmethod
    def product_name(path):
        """Return the product name of the file at ``path``: its file name without its folder and
        without any of its extensions ("a/b/ortho.cog.tif" gives "ortho"), cleaned up."""
        name = Path(path).name
        while Path(name).suffix:
            name = Path(name).stem
        return NameConvention.clean_product_name(name)

    @staticmethod
    def resolution_tag(scale_factor=None, ground_resolution=None):
        """Return the tag of a raster read at ``ground_resolution`` metres per pixel ("gr0p045" is
        4.5 cm), or rescaled by ``scale_factor`` ("sf0p5"). With neither, returns "sf1p0": the
        raster's own resolution."""
        if scale_factor is not None:
            return f"sf{_number_tag(scale_factor)}"
        if ground_resolution is not None:
            return f"gr{_number_tag(ground_resolution)}"
        return "sf1p0"

    @staticmethod
    def tile(
        product_name,
        *,
        col,
        row,
        width,
        height,
        tile_id,
        aoi=None,
        scale_factor=None,
        ground_resolution=None,
    ):
        """Return the file name of a tile: the ``width`` x ``height`` pixels whose top-left pixel is
        at column ``col`` and row ``row`` of the raster (negative if the tile starts past its edge),
        with its ``tile_id`` and the ``aoi`` it belongs to ("noaoi" if None).
        Example: "ortho_tile_train_gr0p045_1024_0_1024_1024_3.tif"."""
        _check(product_name, _PRODUCT_NAME, "The product name")
        aoi = _check(aoi, _WORD, "The AOI") if aoi is not None else "noaoi"
        tag = NameConvention.resolution_tag(scale_factor, ground_resolution)
        numbers = "_".join(str(int(n)) for n in (col, row, width, height, tile_id))
        return f"{product_name}_tile_{aoi}_{tag}_{numbers}.tif"

    @staticmethod
    def coco(
        product_name,
        fold,
        *,
        scale_factor=None,
        ground_resolution=None,
    ):
        """Return the file name of the COCO file of one fold: "ortho_coco_gr0p045_train.json"."""
        _check(product_name, _PRODUCT_NAME, "The product name")
        _check(fold, _WORD, "The fold")
        tag = NameConvention.resolution_tag(scale_factor, ground_resolution)
        return f"{product_name}_coco_{tag}_{fold}.json"

    @staticmethod
    def gpkg(
        product_name,
        fold,
        *,
        scale_factor=None,
        ground_resolution=None,
    ):
        """Return the file name of the GeoPackage of one fold, or of a run's final predictions when
        ``fold`` is "finalpreds": "ortho_gr0p045_finalpreds.gpkg"."""
        _check(product_name, _PRODUCT_NAME, "The product name")
        _check(fold, _WORD, "The fold")
        tag = NameConvention.resolution_tag(scale_factor, ground_resolution)
        return f"{product_name}_{tag}_{fold}.gpkg"

    @staticmethod
    def aoi_gpkg(
        product_name,
        aoi,
        *,
        scale_factor=None,
        ground_resolution=None,
    ):
        """Return the file name of the GeoPackage holding one AOI's area:
        "ortho_aoi_gr0p045_train.gpkg"."""
        _check(product_name, _PRODUCT_NAME, "The product name")
        _check(aoi, _WORD, "The AOI")
        tag = NameConvention.resolution_tag(scale_factor, ground_resolution)
        return f"{product_name}_aoi_{tag}_{aoi}.gpkg"

    @staticmethod
    def aoi_tiles_image(product_name, *, scale_factor=None, ground_resolution=None):
        """Return the file name of the picture showing which tiles are in which AOI:
        "ortho_aoistiles_gr0p045.png"."""
        _check(product_name, _PRODUCT_NAME, "The product name")
        tag = NameConvention.resolution_tag(scale_factor, ground_resolution)
        return f"{product_name}_aoistiles_{tag}.png"
