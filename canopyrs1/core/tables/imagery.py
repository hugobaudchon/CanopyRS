"""The imagery tables: images, each with its georef, on disk or read as a window of its parent."""

import os
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio

from canopyrs1.core.constants import RGB_BANDS, Col, Modality
from canopyrs1.core.geometry.georef import Georef, get_footprints, read_georef
from canopyrs1.core.tables.contracts import Schema
from canopyrs1.core.tables.table import Table
from canopyrs1.core.types import PathLike

_IMAGE_SUFFIXES = {".tif", ".tiff"}


class Imagery(Table):
    """Holds a set of images, one per row, each described by its georef. A row's geometry is
    where its usable pixels are: its footprint, cut to its AOI if it has one (the pixels outside
    are blacked out when read). It is in the CRS of the table's first image (in pixel coordinates
    for images without a CRS).

    The georef is what places an image: its pixels are read, and coordinates converted, with it,
    in the image's own CRS; the whole image is always ``get_footprint(georef)``. The geometry is
    only there to find images by area across the table (``df.sindex``), which is why it is in one
    CRS for the whole table.

    An image is on disk (``path`` is its file) or a window of its parent image (``path`` is
    empty, and its pixels are read from the nearest parent on disk). Its parent is in
    ``parent_imagery``, through ``parent_image_id``: a tile's source, a crop's tile or source.
    Images without a parent (sources, or images read from a folder) are on disk.

    Sources, Tiles and Crops share this class: they differ only by their role, which the pipeline
    steps use to say what they need.
    """

    id_column = Col.IMAGE_ID

    def __init__(self, df: gpd.GeoDataFrame, parent_imagery: "Imagery | None" = None):
        """Wrap ``df``, which must have the ``parent_image_id`` column (``build`` writes it). Raises
        a ValueError if a parent id is outside ``parent_imagery``."""
        super().__init__(df)
        self.parent_imagery = parent_imagery
        self._check_parent_ids(Col.PARENT_IMAGE_ID, parent_imagery)

    @classmethod
    def build(
        cls,
        *,
        georef: Sequence[Georef],
        path: str | Sequence[str | None] | None = None,
        parent_image_id: int | Sequence[int] | None = None,
        parent_imagery: "Imagery | None" = None,
        bands: Sequence[int] = RGB_BANDS,
        modality: str | Sequence[str] = Modality.RGB,
        timestamp: int | Sequence[int] | None = None,
        columns: dict[str, object] | None = None,
    ) -> "Imagery":
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
        paths: PathLike | Sequence[PathLike],
        *,
        bands: Sequence[int] = RGB_BANDS,
        modality: str | Sequence[str] = Modality.RGB,
        timestamp: int | Sequence[int] | None = None,
    ) -> "Imagery":
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
        path: PathLike,
        *,
        bands: Sequence[int] = RGB_BANDS,
        modality: str | Sequence[str] = Modality.RGB,
        timestamp: int | Sequence[int] | None = None,
    ) -> "Imagery":
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

        # read the images' headers
        return cls.from_paths(
            paths,
            bands=bands,
            modality=modality,
            timestamp=timestamp,
        )

    def select(
        self,
        image_ids: Sequence[int],
        columns: dict[str, object] | None = None,
    ) -> "Imagery":
        """Return a table of the same type holding the images ``image_ids`` of this one, each as
        a window of the image it came from (its ``parent_image_id``). ``columns`` sets other
        columns, by Col name, as in ``build``: ``{Col.AOI: names}``."""
        rows = self._select_rows(image_ids, columns)
        rows[Col.PATH] = None
        rows[Col.PARENT_IMAGE_ID] = list(image_ids)
        return type(self)(rows, parent_imagery=self)

    def schema(self) -> Schema:
        """Return what these images offer, as a Schema (see ``Table.schema``), with their link to
        ``parent_imagery`` if they have one."""
        schema = super().schema()
        if self.parent_imagery is not None:
            schema.links.add("parent_imagery")
        return schema

    def get_disk_paths(self) -> pd.Series:
        """Return, for each image, the path of the file on disk its pixels are read from: its own
        file, or recursively its nearest parent's that is on disk. The value is missing for an
        image with no parent on disk."""
        paths = self.df[Col.PATH]
        if self.parent_imagery is None:
            return paths
        parent_paths = pd.Series(self.parent_imagery.get_disk_paths().to_numpy())
        return paths.fillna(self.df[Col.PARENT_IMAGE_ID].map(parent_paths))

    def get_ancestor(self, ancestor_type: type["Imagery"]) -> "Imagery":
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

    def get_ancestor_ids(
        self,
        image_ids: Sequence[int],
        ancestor_type: type["Imagery"],
    ) -> pd.Series:
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
    """Whole input rasters of one place, such as orthomosaics: several dates or modalities of it,
    each with its own timestamp or modality. They have no parent, and are to be tiled together (see
    ``grid_tiles``)."""

    def __init__(self, df: gpd.GeoDataFrame, parent_imagery: "Imagery | None" = None):
        """Wrap ``df`` (see ``Imagery``). Raises a ValueError if two sources have the same
        timestamp and modality, or if a source doesn't overlap the first: other places are tiled
        separately."""
        super().__init__(df, parent_imagery=parent_imagery)

        # each source a different date or modality
        pairs = list(zip(df[Col.TIMESTAMP], df[Col.MODALITY]))
        if len(set(pairs)) < len(pairs):
            raise ValueError(
                "Sources are one place: two of them have the same timestamp and modality. Tile "
                "other places separately"
            )

        # all of the same place
        if len(df) and not df.geometry.intersects(df.geometry.iloc[0]).all():
            raise ValueError(
                "Sources are one place: one of them doesn't overlap the first. Tile other places "
                "separately"
            )


class Tiles(Imagery):
    """The images the models read: grid tiles of a source (windows, or files on disk), or images
    read from a folder."""


class Crops(Imagery):
    """One image per object, cut around it: what the classifier reads."""
