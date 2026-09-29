"""Writing objects to a GeoPackage, the file people open in GIS software."""

import json
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np

from canopyrs1.core.types import PathLike


def _to_json(value: list | dict | np.ndarray) -> str:
    """Return ``value``, a list, dict or array, as JSON text. Numpy numbers and arrays inside it
    are written as plain numbers and lists."""
    return json.dumps(value, default=lambda item: item.tolist())


def write_gpkg(gdf: gpd.GeoDataFrame, path: PathLike) -> Path:
    """Write ``gdf`` to a GeoPackage file at ``path``, creating its folder if needed, and return
    ``path``. Cells holding a list, a dict or an array (such as the score of every class) are
    written as JSON text, as a GeoPackage can't hold them. A GeoDataFrame without a CRS is written
    in its pixel coordinates, with a warning."""
    # warn about pixel coordinates
    if gdf.crs is None:
        warnings.warn(
            f"{path} is written without a CRS: its geometries are in pixel coordinates",
            UserWarning,
            stacklevel=2,
        )

    # write lists, dicts and arrays as JSON text
    out = gdf.copy()
    for column in out.columns.drop(out.geometry.name):
        nested = out[column].map(lambda value: isinstance(value, (list, dict, np.ndarray)))
        if nested.any():
            out[column] = out[column].astype(object)
            out.loc[nested, column] = out.loc[nested, column].map(_to_json)

    # write the file
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="'crs' was not provided")  # warned above
        out.to_file(path, driver="GPKG")
    return path
