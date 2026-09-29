"""Names for the types that several modules' signatures share."""

import os
from collections.abc import Sequence

import geopandas as gpd
import numpy as np
import pyproj
import rasterio.crs
from shapely.geometry.base import BaseGeometry

# A file path: a string or a pathlib.Path.
PathLike = str | os.PathLike

# A CRS: its name ("EPSG:32618"), or a pyproj or rasterio CRS.
CRSLike = str | pyproj.CRS | rasterio.crs.CRS

# Several shapely geometries: a list, a numpy array or a GeoSeries.
Geometries = Sequence[BaseGeometry] | np.ndarray | gpd.GeoSeries
