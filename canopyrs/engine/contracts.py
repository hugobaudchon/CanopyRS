"""Data contracts: how a component declares what it needs and what it makes.

See ``canopyrs/engine/README.md`` for the model in five sentences.

A component declares ``requires`` (a tuple of contract entries) and ``produces`` (a contract entry, or
a tuple). An entry is a data-class *type* (no extra constraints), a ``Need`` (a type plus required
columns / links / CRS-ness / what the objects live on / modalities), or a ``one_of`` over alternatives.

Every check runs on a ``Schema`` — a plain description of what a table exposes. Live tables render
themselves as one via ``.schema()``; the pipeline's static ``validate`` threads declared Schemas
forward. Contracts check *schema-level* properties only (column presence, links, crs-ness, the
linked-imagery type, modality sets) — never data values like timestamps or paths; the attribute
checks are tri-state, skipped when the schema can't say.

Input matching is one rule: a requirement binds the **newest available table of its type**, then the
full check runs on that one candidate — any mismatch is an error, never a silent fallback to an older
table. Roles are types (``Sources`` / ``Tiles`` / ``Crops``), so e.g. a tilerizer's ``Need(Sources)``
finds the raster no matter how many tiles were produced after it.
"""


class Requirement:
    """A ``requires`` entry the pipeline resolves against the available data. ``resolve(get)`` returns
    ``(descriptor, '')`` for the chosen input, or ``(None, error)``. ``get`` maps a data type to the
    *list* of available descriptors, oldest -> newest — live tables at runtime, ``Schema``s in static
    ``validate`` — or an empty sequence. ``Need`` is the basic spec; ``one_of`` builds an OR
    (alternatives may even span different types)."""

    def resolve(self, get):
        raise NotImplementedError

    def needs(self):
        """This requirement's Needs, in declaration order (a ``one_of`` lists its alternatives)."""
        raise NotImplementedError

    @classmethod
    def coerce(cls, spec):
        """A ``requires`` / ``produces`` entry as a Requirement: a bare type becomes a no-constraint
        ``Need``; an existing Requirement passes through."""
        return spec if isinstance(spec, Requirement) else Need(spec)


def as_requirements(spec):
    """Normalize a ``requires`` / ``produces`` declaration (a type, a Requirement, or a tuple of those)
    into a list of Requirements."""
    items = spec if isinstance(spec, tuple) else (spec,)
    return [Requirement.coerce(item) for item in items]


class Need(Requirement):
    """A precondition on one component input, declared in a component's ``requires``. The input must be
    an instance of ``data_type`` and additionally:
      - expose every column in ``columns`` (present with usable values; for Objects, resolvable through
        the prev_objects ancestry);
      - have every relation in ``links`` hydrated (e.g. ``"imagery"`` -> ``objects.imagery`` is set);
      - if ``crs`` is given, hold its geometry in CRS coords (``True``) or tile-pixel coords (``False``);
      - if ``on`` is given (Objects only), live on imagery of that type (or one of them, when a tuple
        is given) — e.g. ``on=Crops`` means each object points at its own crop, so a classifier can
        refuse objects sitting on whole tiles; the aggregator accepts ``on=(Tiles, Crops)``;
      - if ``modalities`` is given, hold at least one row of a supported modality (set intersection).

    A bare type in ``requires`` is the no-extra-constraints case. The pipeline binds each input to the
    newest available instance of its type and checks the Need against it — a mismatch is an error,
    never a silent fallback to an older instance."""

    def __init__(self, data_type, columns=(), links=(), crs=None, on=None, modalities=None):
        self.data_type = data_type
        self.columns = tuple(columns)
        self.links = tuple(links)
        self.crs = crs
        self.on = on
        self.modalities = tuple(modalities) if modalities is not None else None

    def check(self, schema) -> str:
        """An error message if ``schema`` (a ``Schema``) violates this need, else ''. Attribute checks
        (crs / on / modalities) are tri-state: a schema value of None means "can't say" — skip."""
        name = self.data_type.__name__
        for link in self.links:
            if not schema.has_link(link):
                return f"{name}.{link} must be linked"
        for col in self.columns:
            if not schema.provides(col):
                return f"{name} must expose column '{col}' (incl. its ancestry)"
        if self.crs is not None and schema.crs_set is not None and self.crs != schema.crs_set:
            return (f"{name} geometry must be in CRS coords" if self.crs
                    else f"{name} geometry must be in tile-pixel coords (no CRS)")
        if self.on is not None and schema.on is not None:
            allowed = self.on if isinstance(self.on, tuple) else (self.on,)
            if schema.on not in allowed:
                names = "/".join(t.__name__ for t in allowed)
                return f"{name} must live on {names} (these live on {schema.on.__name__})"
        if self.modalities is not None and schema.modalities is not None \
                and not set(self.modalities) & set(schema.modalities):
            return f"{name} must hold a modality in {sorted(self.modalities)} (got {sorted(schema.modalities)})"
        return ""

    def resolve(self, get):
        candidates = list(get(self.data_type) or ())
        if not candidates:
            return None, f"requires {self.data_type.__name__}, but none is available"
        newest = candidates[-1]
        err = self.check(newest.schema())
        if err:
            return None, f"requires {self.data_type.__name__} but the newest doesn't satisfy it: {err}"
        return newest, ""

    def needs(self):
        return (self,)


class AnyOf(Requirement):
    """Satisfied by the FIRST alternative Need the available data meets — alternatives are tried in
    order and may span different types (e.g. classify per-object crops if Objects-on-Crops are
    present, else bare Crops). Built via ``one_of``."""

    def __init__(self, alternatives):
        assert alternatives, "one_of needs at least one alternative"
        self.alternatives = tuple(alternatives)

    def resolve(self, get):
        msgs = []
        for alt in self.alternatives:
            desc, err = alt.resolve(get)
            if desc is not None:
                return desc, ""
            msgs.append(err)
        return None, " OR ".join(msgs)

    def needs(self):
        return self.alternatives


def one_of(*alternatives) -> AnyOf:
    """An OR over requirement alternatives, tried in order (they may span different data types)."""
    return AnyOf(alternatives)


__all__ = ["Requirement", "Need", "AnyOf", "one_of", "Schema", "as_requirements"]


class Schema:
    """What a table exposes — its (ancestry-reachable) columns, links, CRS-ness, the imagery type its
    objects live on (``on``, Objects only), and modality set. The single currency of every contract
    check: live tables render themselves into one via ``Table.schema()``, and the pipeline's static
    ``validate()`` threads declared Schemas forward. ``crs`` / ``on`` / ``modalities`` may be None =
    undeclared (checks skip)."""

    def __init__(self, columns=(), links=(), crs=None, on=None, modalities=None):
        self.columns = set(columns)
        self.links = set(links)
        self.crs_set = crs
        self.on = on
        self.modalities = set(modalities) if modalities is not None else None

    def schema(self) -> "Schema":
        """Itself — so input matching treats live tables and declared Schemas identically."""
        return self

    def provides(self, col) -> bool:
        return col in self.columns

    def has_link(self, name) -> bool:
        return name in self.links
