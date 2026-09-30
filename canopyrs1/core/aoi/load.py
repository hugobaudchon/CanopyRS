"""Loading AOIs, the named areas of a raster (such as its train, valid and test folds), from
vector files."""

import warnings

import geopandas as gpd
import shapely
from shapely.geometry import MultiPolygon, Polygon

from canopyrs1.core.geometry.shapes import repair_polygon
from canopyrs1.core.naming import NameConvention
from canopyrs1.core.types import CRSLike, PathLike


def _load_aoi(
    name: str,
    source: PathLike | gpd.GeoDataFrame,
    crs: CRSLike | None,
) -> Polygon | MultiPolygon:
    """Return the area of the AOI ``name``: the union of the polygons of ``source`` (a vector file
    or a GeoDataFrame), moved into ``crs``."""
    # read it
    gdf = source if isinstance(source, gpd.GeoDataFrame) else gpd.read_file(source)
    if len(gdf) == 0:
        raise ValueError(f"The AOI {name!r} has no polygon")
    others = {
        "None" if g is None else g.geom_type
        for g in gdf.geometry
        if not isinstance(g, (Polygon, MultiPolygon))
    }
    if others:
        raise ValueError(
            f"The AOI {name!r} should only hold polygons, not {', '.join(sorted(others))}"
        )

    # move it into the CRS
    if (gdf.crs is None) != (crs is None):
        where = "a CRS" if crs is not None else "pixel coordinates, without a CRS"
        raise ValueError(f"The AOI {name!r} has to be in {where}, like the images")
    if crs is not None:
        gdf = gdf.to_crs(crs)

    # one area
    return repair_polygon(shapely.union_all([repair_polygon(g) for g in gdf.geometry]))


def load_aois(
    aois: dict[str, PathLike | gpd.GeoDataFrame],
    crs: CRSLike | None,
) -> gpd.GeoDataFrame:
    """Return the AOIs ``aois`` ({name: a vector file, or a GeoDataFrame}, such as {"train":
    "train.gpkg", "valid": "valid.gpkg"}) as a GeoDataFrame in ``crs`` (None for images in pixel
    coordinates), with one row per AOI, in order: its name in ``aoi``, and its area, the union of
    its polygons. Warns if two AOIs overlap (on more than a millionth of the smaller one), as the
    pixels they share would then be in both.

    Raises a ValueError if a name holds anything other than letters and digits (it is written in
    file names), if an AOI has no polygon or anything else than polygons, or if an AOI has a CRS
    and the images don't, or the other way around."""
    # load each AOI
    names = [NameConvention.check_aoi(name) for name in aois]
    areas = [_load_aoi(name, source, crs) for name, source in aois.items()]
    loaded = gpd.GeoDataFrame({"aoi": names}, geometry=areas, crs=crs)

    # warn about overlaps, beyond floating-point rounding along shared edges
    for i, j in zip(*loaded.sindex.query(loaded.geometry, predicate="intersects")):
        shared = areas[i].intersection(areas[j]).area if i < j else 0
        smaller = min(areas[i].area, areas[j].area)
        if shared > 1e-6 * smaller:
            warnings.warn(
                f"The AOIs {names[i]!r} and {names[j]!r} overlap, on {shared / smaller:.2%} of the "
                f"smaller one: the pixels they share would be in both",
                UserWarning,
                stacklevel=2,
            )
    return loaded
