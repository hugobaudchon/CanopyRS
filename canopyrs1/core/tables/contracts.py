"""What a pipeline step needs from its tables (a Need), and what a table offers (a Schema)."""

from dataclasses import dataclass, field


@dataclass
class Schema:
    """What a table offers: its columns with values, the parents it is linked to
    ("parent_imagery", "parent_objects"), whether its geometry is in a CRS, and for objects the
    type of imagery they were found in (``on``). A field left at None is unknown, and isn't
    checked."""

    columns: set = field(default_factory=set)
    links: set = field(default_factory=set)
    has_crs: bool | None = None
    on: type | None = None


@dataclass(frozen=True)
class Need:
    """What a step needs from one of its tables, or promises for one it makes: a table of
    ``data_type`` with every column in ``columns`` and every parent in ``links``, and, if given,
    geometry in a CRS or not (``has_crs``), and objects found in imagery of the type ``on`` (one
    type, or a tuple of the types allowed)."""

    data_type: type
    columns: tuple = ()
    links: tuple = ()
    has_crs: bool | None = None
    on: type | tuple | None = None

    def check(self, schema):
        """Return why ``schema`` doesn't meet this need, as a list of short sentences, one per
        problem ("Objects must be found in Crops, not in Tiles"). Returns an empty list if it
        does."""
        name = self.data_type.__name__
        errors = []

        missing_links = [link for link in self.links if link not in schema.links]
        if missing_links:
            errors.append(f"{name} must be linked to their {' and '.join(missing_links)}")

        missing_columns = [column for column in self.columns if column not in schema.columns]
        if missing_columns:
            noun = "column" if len(missing_columns) == 1 else "columns"
            errors.append(f"{name} must have the {noun} {', '.join(missing_columns)}")

        if self.has_crs is not None and schema.has_crs is not None:
            if schema.has_crs != self.has_crs:
                where = "a CRS" if self.has_crs else "pixel coordinates"
                errors.append(f"{name} must have their geometry in {where}")

        if self.on is not None and schema.on is not None:
            if not issubclass(schema.on, self.on):
                allowed = self.on if isinstance(self.on, tuple) else (self.on,)
                names = " or ".join(t.__name__ for t in allowed)
                errors.append(f"{name} must be found in {names}, not in {schema.on.__name__}")

        return errors
