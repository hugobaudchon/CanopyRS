"""Checking a pipeline's wiring before running it: which tables each component would find, and
whether they meet its needs."""

from collections.abc import Sequence
from dataclasses import dataclass

from canopyrs1.core.tables.contracts import Need, Schema
from canopyrs1.core.tables.objects import Objects
from canopyrs1.core.tables.table import Table


@dataclass
class SimulatedStep:
    """One step of a simulated run: the ``component``, and the newest table of each type just
    before and just after it (dicts of {table type: Schema})."""

    component: object
    before: dict[type, Schema]
    after: dict[type, Schema]


def _made_schema(need: Need, before: dict[type, Schema]) -> Schema:
    """Return the Schema of the table a component promises in ``need`` (one of its
    ``produces``), ``before`` holding the newest table of each type before the component."""
    if isinstance(need.on, tuple):
        raise ValueError(
            f"A component can't promise {need.data_type.__name__} found in one of several types: "
            f"a tuple in on= is only for requires"
        )
    schema = Schema(
        columns=set(need.columns),
        links=set(need.links),
        has_crs=need.has_crs,
        on=need.on,
    )

    # objects made from objects keep their history, as at run time
    history = before.get(Objects)
    if need.data_type is Objects and "parent_objects" in need.links and history is not None:
        if "parent_imagery" not in need.links:
            schema.on = history.on
        schema.columns |= history.columns
        schema.links |= history.links
    return schema


def simulate(inputs: Sequence[Table], components: Sequence[object]) -> list[SimulatedStep]:
    """Return what running ``components`` on the ``inputs`` tables would look like, without
    running anything: one SimulatedStep per component. Each component reads the newest table of
    each type its ``requires`` asks for, and its ``produces`` describe the tables it makes. Objects
    linked to ``parent_objects`` were made from the newest Objects before them, so they also get
    those objects' columns, links and imagery.

    Raises a ValueError at the first component whose needs aren't met, listing its problems one
    per line. Later components aren't checked, as their problems would often be caused by it."""
    # start from the inputs
    available = {type(table): table.schema() for table in inputs}
    simulated = []

    # iterate over components
    for number, component in enumerate(components, start=1):
        before = dict(available)

        # check what it needs
        errors = []
        for need in component.requires:
            schema = before.get(need.data_type)
            if schema is None:
                errors.append(f"There are no {need.data_type.__name__} before this step")
            else:
                errors += need.check(schema)

        # stop at the first component whose needs aren't met
        if errors:
            problems = "\n".join(f"  - {error}" for error in errors)
            name = type(component).__name__
            raise ValueError(f"Step {number}, {name}, can't run:\n{problems}")

        # describe what it makes
        for need in component.produces:
            available[need.data_type] = _made_schema(need, before)
        simulated.append(SimulatedStep(component, before, dict(available)))
    return simulated
