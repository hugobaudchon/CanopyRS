"""The base of every table: a (Geo)DataFrame with numbered rows and links to parent tables."""

from collections.abc import Sequence

import pandas as pd

from canopyrs1.core.tables.contracts import Schema


class Table:
    """A (Geo)DataFrame whose rows are numbered 0 to n - 1 in ``id_column``.

    A table can be linked to parent tables, the tables its rows came from. Each subclass sets its
    parents in its ``__init__``, as plain attributes, and checks their ids with
    ``_check_parent_ids``.

    A table's rows are never removed or reordered: a step that keeps some rows makes a new table,
    whose rows point to the rows they came from (see each table's ``select``). So a row's id is
    always its position, and a parent id is the parent row's position in the parent table.
    """

    id_column = None

    def __init__(self, df: pd.DataFrame):
        """Wrap ``df``, numbering its rows in ``id_column``."""
        df[self.id_column] = range(len(df))
        self.df = df

    def _check_parent_ids(self, column: str, parent: "Table | None") -> None:
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

    def _select_rows(self, ids: Sequence[int], columns: dict[str, object] | None) -> pd.DataFrame:
        """Return a copy of the rows ``ids``, with ``columns`` set: one value for every row, or
        one per row. Raises a ValueError if a column has another number of values."""
        rows = self.df.iloc[list(ids)].reset_index(drop=True)
        for column, values in (columns or {}).items():
            if pd.api.types.is_list_like(values) and len(values) != len(rows):
                raise ValueError(
                    f"The column {column!r} has {len(values)} values for {len(rows)} rows"
                )
            rows[column] = values
        return rows

    def has_column(self, column: str) -> bool:
        """Return whether this table has ``column`` with at least one value that isn't missing (an
        empty table only needs to have the column)."""
        return column in self.df.columns and (self.df.empty or self.df[column].notna().any())

    @property
    def has_crs(self) -> bool:
        """Whether this table's geometries are in a CRS. Tables without geometry, or with
        geometries in pixel coordinates, have none."""
        return getattr(self.df, "crs", None) is not None

    def schema(self) -> Schema:
        """Return what this table offers, as a Schema: its columns with values (see
        ``has_column``) and whether it is in a CRS. Subclasses add their links."""
        columns = {column for column in self.df.columns if self.has_column(column)}
        return Schema(columns=columns, has_crs=self.has_crs)

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self.df)} rows)"
