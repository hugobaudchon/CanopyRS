"""The base of every table: a (Geo)DataFrame with numbered rows and links to parent tables."""

from canopyrs1.core.tables.contracts import Schema


class Table:
    """A (Geo)DataFrame whose rows are numbered 0 to n - 1 in ``id_column``.

    A table can be linked to parent tables, the tables its rows came from. Each subclass sets its
    parents in its ``__init__``, as plain attributes, and checks their ids with
    ``_check_parent_ids``.

    A table's rows are never removed or reordered: a step that keeps some rows makes a new table,
    whose rows point to the rows they came from. So a row's id is always its position, and a parent
    id is the parent row's position in the parent table.
    """

    id_column = None

    def __init__(self, df):
        """Wrap ``df``, numbering its rows in ``id_column``."""
        df[self.id_column] = range(len(df))
        self.df = df

    def _check_parent_ids(self, column, parent):
        """Raise a ValueError if an id in ``column`` is outside ``parent``, the table these ids
        point to. Does nothing if ``parent`` is None."""
        if parent is None:
            return
        ids = self.df[column].dropna()
        if len(ids) and (ids.min() < 0 or ids.max() >= len(parent)):
            raise ValueError(
                f"{type(self).__name__}.{column} has ids outside its parent "
                f"{type(parent).__name__}, which has {len(parent)} rows"
            )

    def has_column(self, column):
        """Return whether this table has ``column`` with at least one value that isn't missing (an
        empty table only needs to have the column)."""
        return column in self.df.columns and (self.df.empty or self.df[column].notna().any())

    @property
    def has_crs(self):
        """Whether this table's geometries are in a CRS. Tables without geometry, or with
        geometries in pixel coordinates, have none."""
        return getattr(self.df, "crs", None) is not None

    def schema(self):
        """Return what this table offers, as a Schema: its columns with values (see
        ``has_column``) and whether it is in a CRS. Subclasses add their links."""
        columns = {column for column in self.df.columns if self.has_column(column)}
        return Schema(columns=columns, has_crs=self.has_crs)

    def __len__(self):
        return len(self.df)

    def __repr__(self):
        return f"{type(self).__name__}({len(self.df)} rows)"
