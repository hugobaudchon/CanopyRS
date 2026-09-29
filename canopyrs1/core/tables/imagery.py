"""The imagery tables: images, each with its georef, on disk or read as a window of its parent."""

import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio

from canopyrs1.core.constants import RGB_BANDS, Col, Modality
from canopyrs1.core.geometry.georef import get_footprints, read_georef
from canopyrs1.core.tables.table import Table

_IMAGE_SUFFIXES = {".tif", ".tiff"}


class Imagery(Table):
    """Holds a set of images, one per row, each described by its georef, with its footprint as
    the geometry, in the CRS of the table's first image (in pixel coordinates for images without a
    CRS).

    The georef is what places an image: its pixels are read, and coordinates converted, with it,
    in the image's own CRS. The footprint is made from it, only to find images by area across the
    table (``df.sindex``), which is why it is in one CRS for the whole table.

    An image is on disk (``path`` is its file) or a window of its parent image (``path`` is
    empty, and its pixels are read from the nearest parent on disk). Its parent is in
    ``parent_imagery``, through ``parent_image_id``: a tile's source, a crop's tile or source.
    Images without a parent (sources, or images read from a folder) are on disk.

    Sources, Tiles and Crops share this class: they differ only by their role, which the pipeline
    steps use to say what they need.
    """

    id_column = Col.IMAGE_ID

    def __init__(self, df, parent_imagery=None):
        """Wrap ``df``, which must have the ``parent_image_id`` column (``build`` writes it). Raises
        a ValueError if a parent id is outside ``parent_imagery``."""
        super().__init__(df)
        self.parent_imagery = parent_imagery
        self._check_parent_ids(Col.PARENT_IMAGE_ID, parent_imagery)

    @classmethod
    def build(
        cls,
        *,
        georef,
        path=None,
        parent_image_id=None,
        parent_imagery=None,
        bands=RGB_BANDS,
        modality=Modality.RGB,
        timestamp=None,
        columns=None,
    ):
        """Return images built from their ``georef`` (one per image), their ``path`` (None for
        windows of their parent), their parent ids and parent table, the ``bands`` to read (the
        same for every image), their ``modality`` and ``timestamp``. ``columns`` holds any other
        column, by its Col name: ``{Col.LAZY_CONDITIONS: conditions}``. Every argument except
        ``georef`` and ``bands`` can be one value for every image, or one per image."""
        georef = list(georef)
        crs = georef[0]["crs"] if georef else None  # the table's CRS: its first image's
        df = gpd.GeoDataFrame(
            {
                Col.GEOMETRY: get_footprints(georef, crs),
                Col.GEOREF: georef,
                Col.PATH: path,
                Col.PARENT_IMAGE_ID: parent_image_id,
                Col.BANDS: [list(bands)] * len(georef),
                Col.MODALITY: modality,
                Col.TIMESTAMP: timestamp,
                **(columns or {}),
            },
            geometry=Col.GEOMETRY,
            crs=crs,
        )
        return cls(df, parent_imagery=parent_imagery)

    @classmethod
    def from_paths(
        cls,
        paths,
        *,
        bands=RGB_BANDS,
        modality=Modality.RGB,
        timestamp=None,
    ):
        """Return a table of this type (Sources, Tiles or Crops) with one image per raster file in
        ``paths``: one path or a list, of local files or URLs. Each image's georef is read from its
        file's header, so no pixel is read. ``bands``, ``modality`` and ``timestamp`` are as in
        ``build``."""
        # one path or several
        if isinstance(paths, (str, os.PathLike)):
            paths = [paths]
        paths = [os.fspath(path) for path in paths]

        # read each georef from its file's header
        georefs = []
        for path in paths:
            with rasterio.open(path) as src:
                georefs.append(read_georef(src))
        return cls.build(
            georef=georefs,
            path=paths,
            bands=bands,
            modality=modality,
            timestamp=timestamp,
        )

    @classmethod
    def from_image_dir(
        cls,
        path,
        *,
        bands=RGB_BANDS,
        modality=Modality.RGB,
        timestamp=None,
    ):
        """Return a table of this type (Sources, Tiles or Crops) with one image per GeoTIFF file
        (.tif or .tiff) in the folder ``path``, sorted by file name, as ``from_paths`` does.
        Subfolders aren't searched. Raises a ValueError if the folder has no GeoTIFF file."""
        # find the images
        paths = sorted(
            file
            for file in Path(path).iterdir()
            if file.is_file() and file.suffix.lower() in _IMAGE_SUFFIXES
        )
        if not paths:
            raise ValueError(f"No .tif or .tiff images in {path}")

        # read their headers
        return cls.from_paths(
            paths,
            bands=bands,
            modality=modality,
            timestamp=timestamp,
        )

    def schema(self):
        """Return what these images offer, as a Schema (see ``Table.schema``), with their link to
        ``parent_imagery`` if they have one."""
        schema = super().schema()
        if self.parent_imagery is not None:
            schema.links.add("parent_imagery")
        return schema

    def get_disk_paths(self):
        """Return, for each image, the path of the file on disk its pixels are read from: its own
        file, or recursively its nearest parent's that is on disk. The value is missing for an
        image with no parent on disk."""
        paths = self.df[Col.PATH]
        if self.parent_imagery is None:
            return paths
        parent_paths = pd.Series(self.parent_imagery.get_disk_paths().to_numpy())
        return paths.fillna(self.df[Col.PARENT_IMAGE_ID].map(parent_paths))

    def get_ancestor(self, ancestor_type):
        """Return the nearest imagery table of ``ancestor_type`` (Sources, Tiles or Crops), going up
        ``parent_imagery`` recursively: this table itself if it is one. Raises a ValueError if
        there is none."""
        table = self
        while not isinstance(table, ancestor_type):
            if table.parent_imagery is None:
                raise ValueError(
                    f"No {ancestor_type.__name__} above this {type(self).__name__} table"
                )
            table = table.parent_imagery
        return table

    def get_ancestor_ids(self, image_ids, ancestor_type):
        """Return, for each of these ``image_ids``, the id of the image it came from in the nearest
        imagery table of ``ancestor_type`` (see ``get_ancestor``). For crops of tiles:
        ``crops.get_ancestor_ids([0, 2], Tiles)`` gives the tiles of crops 0 and 2. Raises a
        ValueError if there is no table of ``ancestor_type`` above this one."""
        ids = pd.Series(image_ids)
        table = self
        while not isinstance(table, ancestor_type):
            if table.parent_imagery is None:
                raise ValueError(
                    f"No {ancestor_type.__name__} above this {type(self).__name__} table"
                )
            ids = ids.map(pd.Series(table.df[Col.PARENT_IMAGE_ID].to_numpy()))
            table = table.parent_imagery
        return ids


class Sources(Imagery):
    """Whole input rasters, such as orthomosaics: they have no parent, and are to be tiled."""


class Tiles(Imagery):
    """The images the models read: grid tiles of a source (windows, or files on disk), or images
    read from a folder."""


class Crops(Imagery):
    """One image per object, cut around it: what the classifier reads."""
