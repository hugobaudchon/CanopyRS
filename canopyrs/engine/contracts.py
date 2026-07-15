"""Data contracts: how a component declares what it needs and what it makes.

A component declares ``requires`` (a tuple of contract entries) and ``produces`` (a contract entry, or
a tuple). An entry is a data-class *type* (no extra constraints), a ``Need`` (a type plus required
columns / links / CRS-ness), or a ``one_of`` over alternatives. The pipeline checks each ``requires``
against the available data before ``run`` and each ``produces`` against the output after — the same
``Need.check`` serving both.

The contract system is pure duck-typing: a ``Need`` only ever calls ``provides`` / ``has_link`` /
``crs_set`` on whatever it's checking — a live ``Table`` at runtime, a ``Schema`` during static
``validate``. So this module knows nothing about the concrete table classes; it imports nothing from
``data``.
"""


class Requirement:
    """A ``requires`` entry the pipeline resolves against the available data. ``resolve(get)`` returns
    ``(descriptor, '')`` for the chosen input, or ``(None, error)``. ``get`` maps a data type to the
    available descriptor — a table instance at runtime, a ``Schema`` in static ``validate`` — or None.
    ``Need`` is the basic spec; ``one_of`` builds an OR (alternatives may even span different types)."""

    def resolve(self, get):
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
      - have every relation in ``links`` hydrated (e.g. ``"tiles"`` -> ``objects.tiles`` is set);
      - if ``crs`` is given, hold its geometry in CRS coords (``True``) or tile-pixel coords (``False``).

    A bare type in ``requires`` is the no-extra-constraints case. The pipeline threads inputs by
    ``data_type`` and checks the input before ``run``, so a component can assume valid inputs."""

    def __init__(self, data_type, columns=(), links=(), crs=None):
        self.data_type = data_type
        self.columns = tuple(columns)
        self.links = tuple(links)
        self.crs = crs

    def check(self, desc) -> str:
        """An error message if ``desc`` violates this need, else ''. ``desc`` is anything implementing
        the descriptor interface (``provides`` / ``has_link`` / ``crs_set``) — a real table instance
        (runtime) or a simulated ``Schema`` (static ``validate``)."""
        name = self.data_type.__name__
        for link in self.links:
            if not desc.has_link(link):
                return f"{name}.{link} must be linked"
        for col in self.columns:
            if not desc.provides(col):
                return f"{name} must expose column '{col}' (incl. its ancestry)"
        # crs_set may be None on a Schema whose producer didn't declare CRS-ness — then we can't verify.
        if self.crs is not None and desc.crs_set is not None and self.crs != desc.crs_set:
            return (f"{name} geometry must be in CRS coords" if self.crs
                    else f"{name} geometry must be in tile-pixel coords (no CRS)")
        return ""

    def resolve(self, get):
        desc = get(self.data_type)
        if desc is None:
            return None, f"requires {self.data_type.__name__}, but none is available"
        err = self.check(desc)
        return (None, err) if err else (desc, "")


class AnyOf(Requirement):
    """Satisfied by the FIRST alternative Need the available data meets — alternatives are tried in
    order and may span different types (e.g. classify per-object crops if Objects-with-tiles are
    present, else whole tiles; or read a tile from a pre-cut path else its source). Built via ``one_of``."""

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


def one_of(*alternatives) -> AnyOf:
    """An OR over requirement alternatives, tried in order (they may span different data types)."""
    return AnyOf(alternatives)


__all__ = ["Requirement", "Need", "AnyOf", "one_of", "Schema", "as_requirements"]


class Schema:
    """A declared description of what a table exposes — its (ancestry-reachable) columns, hydrated
    links, and CRS-ness — without a real instance. The pipeline's ``validate()`` threads each
    component's ``produces`` forward as Schemas to check requirements before running. Implements the
    same descriptor interface (``provides`` / ``has_link`` / ``crs_set``) a real table does, so one
    ``Need.check`` serves both static and runtime."""

    def __init__(self, columns=(), links=(), crs=None):
        self.columns = set(columns)
        self.links = set(links)
        self.crs_set = crs   # True / False / None (None = the producer didn't declare CRS-ness)

    def provides(self, col) -> bool:
        return col in self.columns

    def has_link(self, name) -> bool:
        return name in self.links
