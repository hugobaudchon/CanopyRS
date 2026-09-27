"""The objects table: boxes, masks and points, each found in an image."""

import geopandas as gpd
import pandas as pd

from canopyrs1.core.constants import Col, GeomKind
from canopyrs1.core.tables.table import Table


class Objects(Table):
    """Holds a set of boxes, masks or points, one per row, with their geometry and kind (see
    GeomKind).

    Each object can have two parents:

    - ``parent_imagery``: the imagery table of the images the objects were found in, through
      ``parent_image_id``;
    - ``parent_objects``: the objects table they were made from, through ``parent_object_id``
      (a mask's box, a kept box's detection). Following it back gives an object's history.

    A column, or the imagery, that these objects don't have is looked for in their history (see
    ``get_column`` and ``get_parent_imagery``): a late step can read a score made several steps
    before.
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

    def get_column(self, column):
        """Return the values of ``column`` for these objects, one per row. If these objects don't
        have the column, return their parents' values, looking one step further back in the
        history until a table has it; an object without a parent then gets a missing value. Raises
        a KeyError if no table in the history has the column."""
        if column in self.df.columns:
            return self.df[column]
        if self.parent_objects is None:
            raise KeyError(f"'{column}' is neither in these objects nor in their history")
        parent_values = pd.Series(self.parent_objects.get_column(column).to_numpy())
        values = self.df[Col.PARENT_OBJECT_ID].map(parent_values)
        return values.rename(column)

    def has_column(self, column):
        """Return whether these objects, or the objects in their history, have ``column`` with at
        least one value that isn't missing."""
        if column in self.df.columns:
            return super().has_column(column)
        return self.parent_objects is not None and self.parent_objects.has_column(column)

    def get_parent_imagery(self):
        """Return the imagery table these objects were found in: their own ``parent_imagery``, or,
        if they have none, the one of the objects in their history (aggregated objects find their
        tiles through the detections they came from). Returns None if no table in the history has
        one."""
        if self.parent_imagery is not None or self.parent_objects is None:
            return self.parent_imagery
        return self.parent_objects.get_parent_imagery()
