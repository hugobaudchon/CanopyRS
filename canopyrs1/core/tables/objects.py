"""The objects table: boxes, masks and points, each found in an image."""

import geopandas as gpd
import pandas as pd

from canopyrs1.core.constants import Col, GeomKind
from canopyrs1.core.geometry.georef import crs_to_pixel, pixel_to_crs
from canopyrs1.core.geometry.shapes import infer_geom_kind
from canopyrs1.core.tables.table import Table


class Objects(Table):
    """Holds a set of boxes, masks or points, one per row, with their geometry and kind (see
    GeomKind).

    Each object can have two parents:

    - ``parent_imagery``: the imagery table of the images the objects were found in, through
      ``parent_image_id``;
    - ``parent_objects``: the objects table they were made from, through ``parent_object_id``
      (a mask's box, a kept box's detection). Following it back gives an object's history.

    A column, or the imagery, that these objects don't have is looked for recursively in their
    history (see ``get_column`` and ``get_parent_imagery``): a late step can read a score made
    several steps before.
    """

    id_column = Col.OBJECT_ID

    def __init__(self, df, parent_imagery=None, parent_objects=None):
        """Wrap ``df``, which must have the ``geom_kind``, ``parent_image_id`` and
        ``parent_object_id`` columns (``build`` writes them all). Raises a ValueError if a kind
        isn't one of GeomKind's, or if a parent id is outside its parent table."""
        super().__init__(df)
        self.parent_imagery = parent_imagery
        self.parent_objects = parent_objects
        self._check_parent_ids(Col.PARENT_IMAGE_ID, parent_imagery)
        self._check_parent_ids(Col.PARENT_OBJECT_ID, parent_objects)
        unknown = set(df[Col.GEOM_KIND].dropna()) - GeomKind.ALL
        if unknown:
            raise ValueError(
                f"Unknown geom_kind {sorted(unknown)}, expected {sorted(GeomKind.ALL)}"
            )

    @classmethod
    def build(
        cls,
        *,
        geometry,
        geom_kind,
        crs=None,
        parent_image_id=None,
        parent_imagery=None,
        parent_object_id=None,
        parent_objects=None,
        columns=None,
    ):
        """Return objects built from their ``geometry`` (in ``crs``, or in pixel coordinates if
        None) and ``geom_kind``, with their parent ids and parent tables. ``columns`` holds any
        other column, by its Col name: ``{Col.DETECTOR_SCORE: scores}``. ``geom_kind``, the parent
        ids and each column can be one value for every object, or one per object."""
        df = gpd.GeoDataFrame(
            {
                Col.GEOMETRY: list(geometry),
                Col.GEOM_KIND: geom_kind,
                Col.PARENT_IMAGE_ID: parent_image_id,
                Col.PARENT_OBJECT_ID: parent_object_id,
                **(columns or {}),
            },
            geometry=Col.GEOMETRY,
            crs=crs,
        )
        return cls(df, parent_imagery=parent_imagery, parent_objects=parent_objects)

    @classmethod
    def from_file(cls, path_or_gdf, *, parent_imagery=None):
        """Return an objects table with one object per row of a vector file (a GeoPackage, a
        GeoJSON, or anything else geopandas reads) or of a GeoDataFrame, keeping all its columns.
        Each object's kind is the file's ``geom_kind`` if it has that column, or else is found from
        its geometry (see ``infer_geom_kind``).

        With ``parent_imagery``, the objects are linked to the images they are on, through the
        file's ``parent_image_id`` column, or, if it has none, to the only image of
        ``parent_imagery``. Raises a ValueError if a kind is to be found and a geometry isn't a
        point or a polygon, or if ``parent_imagery`` has several images and the file doesn't say
        which one each object is on."""
        # read the file, without changing a GeoDataFrame given
        if isinstance(path_or_gdf, gpd.GeoDataFrame):
            gdf = path_or_gdf.copy()
        else:
            gdf = gpd.read_file(path_or_gdf)
        gdf = gdf.rename_geometry(Col.GEOMETRY) if gdf.geometry.name != Col.GEOMETRY else gdf

        # find their kind, if the file doesn't say
        if Col.GEOM_KIND not in gdf.columns:
            image_crs = parent_imagery.df.crs if parent_imagery is not None else None
            gdf[Col.GEOM_KIND] = infer_geom_kind(gdf.geometry, crs=gdf.crs, image_crs=image_crs)

        # link them to their images
        if Col.PARENT_IMAGE_ID not in gdf.columns:
            if parent_imagery is None:
                gdf[Col.PARENT_IMAGE_ID] = None
            elif len(parent_imagery) == 1:
                gdf[Col.PARENT_IMAGE_ID] = 0  # every object is on the only image
            else:
                raise ValueError(
                    f"The objects need a {Col.PARENT_IMAGE_ID} column to say which of the "
                    f"{len(parent_imagery)} images each one is on"
                )
        if Col.PARENT_OBJECT_ID not in gdf.columns:
            gdf[Col.PARENT_OBJECT_ID] = None
        return cls(gdf, parent_imagery=parent_imagery)

    def get_column(self, column):
        """Return the values of ``column`` for these objects, one per row. If these objects don't
        have the column, return their parents' values, looking recursively one step further back
        in the history until a table has it; an object without a parent then gets a missing value.
        Raises a KeyError if no table in the history has the column."""
        if column in self.df.columns:
            return self.df[column]
        if self.parent_objects is None:
            raise KeyError(f"'{column}' is neither in these objects nor in their history")
        parent_values = pd.Series(self.parent_objects.get_column(column).to_numpy())
        values = self.df[Col.PARENT_OBJECT_ID].map(parent_values)
        return values.rename(column)

    def has_column(self, column):
        """Return whether these objects, or recursively the objects in their history, have
        ``column`` with at least one value that isn't missing."""
        if column in self.df.columns:
            return super().has_column(column)
        return self.parent_objects is not None and self.parent_objects.has_column(column)

    def get_parent_imagery(self):
        """Return the imagery table these objects were found in: their own ``parent_imagery``, or,
        if they have none, recursively the one of the objects in their history (aggregated objects
        find their tiles through the detections they came from). Returns None if no table in the
        history has one."""
        if self.parent_imagery is not None or self.parent_objects is None:
            return self.parent_imagery
        return self.parent_objects.get_parent_imagery()

    def schema(self):
        """Return what these objects offer, as a Schema (see ``Table.schema``), with the columns
        and imagery found in their history (see ``has_column`` and ``get_parent_imagery``)."""
        schema = super().schema()

        # the imagery they were found in
        imagery = self.get_parent_imagery()
        if imagery is not None:
            schema.links.add("parent_imagery")
            schema.on = type(imagery)

        # the columns of their history
        if self.parent_objects is not None:
            schema.links.add("parent_objects")
            history = self.parent_objects.schema().columns
            schema.columns |= {column for column in history if self.has_column(column)}
        return schema

    def get_geometry_in_image_coords(self, pixels=False):
        """Return each object's geometry in its image's CRS, or in its image's pixels if
        ``pixels`` is True. Raises a ValueError if an object has no image."""
        # check every object has an image
        if self.parent_imagery is None or self.df[Col.PARENT_IMAGE_ID].isna().any():
            raise ValueError("Every object needs its image: parent_imagery and a parent_image_id")

        # convert the objects of each image with its georef
        geometry = self.df.geometry.to_numpy().copy()
        for image_id, rows in self.df.groupby(Col.PARENT_IMAGE_ID).indices.items():
            georef = self.parent_imagery.df[Col.GEOREF].iloc[image_id]
            part = geometry[rows]
            if self.has_crs:
                part = gpd.GeoSeries(part, crs=self.df.crs).to_crs(georef["crs"]).to_numpy()
                geometry[rows] = crs_to_pixel(part, georef) if pixels else part
            else:
                geometry[rows] = part if pixels else pixel_to_crs(part, georef)
        return gpd.GeoSeries(geometry, index=self.df.index)

    def group_by_disk_path(self):
        """Return the objects as ``(path, gdf)`` pairs, one per file their pixels are read from
        (see ``Imagery.get_disk_paths``), each ``gdf`` in its images' CRS. Raises a ValueError if
        an object has no image, or no file."""
        # put the objects in their images' CRS
        gdf = gpd.GeoDataFrame(
            self.df[[Col.OBJECT_ID, Col.PARENT_IMAGE_ID]],
            geometry=self.get_geometry_in_image_coords(),
        )

        # find disk paths
        disk_paths = pd.Series(self.parent_imagery.get_disk_paths().to_numpy())
        paths = gdf[Col.PARENT_IMAGE_ID].map(disk_paths)
        if paths.isna().any():
            raise ValueError("Some objects have no file on disk above their image")

        # group by disk path
        georefs = self.parent_imagery.df[Col.GEOREF]
        groups = []
        for path, group in gdf.groupby(paths, sort=False):
            first_image_id = group[Col.PARENT_IMAGE_ID].iloc[0]
            crs = georefs.iloc[first_image_id]["crs"]  # all images of one file share its CRS
            groups.append((path, group.set_crs(crs)))
        return groups
