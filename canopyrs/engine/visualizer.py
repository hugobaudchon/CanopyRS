"""
Pipeline flow visualization.

The pipeline has typed tables, and a component's contract is ``Need(type, columns, links, crs, on)``
— not a flat column set. So the chart has **one stacked section per type present** (SOURCES / TILES /
CROPS / OBJECTS); within each, a row per **link** (``→parent`` / ``→imagery`` / ``→prev_objects``), a
**crs** row (tri-state ✓/✗/?), then a row per **column**. Each cell is marked available / produced /
required / required+produced / missing / passthrough per component — read off the *same*
``Pipeline.thread_schemas`` simulation that ``validate`` uses, so the chart never re-derives "what's
available when". A cell is a plain ``(role, char)`` pair — ``("both", (req_char, prod_char))`` for
the required+produced split — and color + width are applied once, at render time.
"""

import sys

from canopyrs.engine.contracts import as_requirements
from canopyrs.engine.data import Crops, Objects, Sources, Tiles


def _stdout_supports_unicode() -> bool:
    try:
        encoding = getattr(sys.stdout, 'encoding', None) or 'ascii'
        '▬─│┼·✓✗'.encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


_USE_UNICODE = _stdout_supports_unicode()
_BLOCK = '▬' if _USE_UNICODE else '#'
_DOT = '·' if _USE_UNICODE else '.'
_YES = '✓' if _USE_UNICODE else 'y'
_NO = '✗' if _USE_UNICODE else 'n'
_UNK = '?'

_RESET = "\033[0m"
_GREEN, _BLUE, _YELLOW, _RED, _GRAY = "\033[92m", "\033[94m", "\033[93m", "\033[91m", "\033[90m"
_ROLE_COLOR = {"available": _GREEN, "produced": _BLUE, "required": _YELLOW,
               "missing": _RED, "passthrough": _GRAY}

EMPTY = ("empty", " ")


def _glyph(value):
    """The tri-state crs glyph: True -> CRS, False -> pixel, None -> undeclared."""
    return _YES if value is True else _NO if value is False else _UNK


def _render_cell(cell, width):
    """A cell as its colored, width-filled string. ``("both", (req_char, prod_char))`` renders as a
    yellow/blue split (requires-value then produces-value)."""
    role, char = cell
    if role == "empty":
        return " " * width
    if role == "both":
        req_char, prod_char = char
        half = width // 2
        return f"{_YELLOW}{req_char * half}{_BLUE}{prod_char * (width - half)}{_RESET}"
    return f"{_ROLE_COLOR[role]}{char * width}{_RESET}"


class _Step:
    """Per-column state for one pipeline step (the seed, or one component). ``before`` / ``after``
    are ``{data_type: Schema}`` snapshots from ``Pipeline.thread_schemas``."""

    def __init__(self, label, before, after, produced_types, req, miss, seed=False):
        self.label = label
        self.before = before
        self.after = after
        self.produced_types = produced_types
        self.req = req       # {"cols"/"links"/"crs": {data_type: set|value}}
        self.miss = miss
        self.seed = seed


class PipelineFlowVisualizer:
    """Visualize the typed data flow through a pipeline.

    Legend (colored blocks): green = input · blue = produced · yellow = required · yellow+blue =
    required+produced · red = MISSING · gray · = passthrough. crs row: ✓ = CRS, ✗ = tile-pixel,
    ? = undeclared.
    """

    SECTIONS = (Sources, Tiles, Crops, Objects)

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def print(self) -> None:
        try:
            self._render(self._track())
        except UnicodeEncodeError:
            pass  # terminals that can't display the chart (e.g. Windows subprocess workers)

    # --- tracking ------------------------------------------------------------
    def _track(self):
        steps = []
        for component, before, after in self.pipeline.thread_schemas():
            if component is None:
                steps.append(_Step("input", {}, after, set(after),
                                   self._empty_marks(), self._empty_marks(), seed=True))
            else:
                req, miss = self._requirement_marks(component, before)
                produced_types = {need.data_type for need in as_requirements(component.produces)}
                steps.append(_Step(component.label, before, after, produced_types, req, miss))

        labels = [step.label for step in steps]
        sections = []
        for data_type in self.SECTIONS:
            rows = []
            for name in data_type.fks:                                  # link rows: →parent / →imagery / ...
                cells = [self._link_cell(data_type, name, step) for step in steps]
                if any(cell != EMPTY for cell in cells):
                    rows.append((f"→{name}", cells))
            crs_cells = [self._crs_cell(data_type, step) for step in steps]
            if any(cell != EMPTY for cell in crs_cells):
                rows.append(("crs", crs_cells))
            for col in sorted(self._shown_columns(data_type, steps)):   # column rows
                rows.append((col, [self._col_cell(data_type, col, step) for step in steps]))
            if rows:
                sections.append((data_type.__name__.upper(), rows))
        return {"labels": labels, "sections": sections}

    def _shown_columns(self, data_type, steps):
        shown = set()
        for step in steps:
            if data_type in step.after:
                shown |= step.after[data_type].columns
            shown |= step.req["cols"].get(data_type, set())
            shown |= step.miss["cols"].get(data_type, set())
        return shown

    @staticmethod
    def _empty_marks():
        return {"cols": {}, "links": {}, "crs": {}}

    def _requirement_marks(self, component, before):
        req, miss = self._empty_marks(), self._empty_marks()
        for need in as_requirements(component.requires):
            desc, _ = need.resolve(before.get(need.data_type))
            self._add_need(req if desc is not None else miss, need)
        return req, miss

    @staticmethod
    def _add_need(acc, need):
        acc["cols"].setdefault(need.data_type, set()).update(need.columns)
        acc["links"].setdefault(need.data_type, set()).update(need.links)
        if need.crs is not None:
            acc["crs"][need.data_type] = need.crs

    # --- per-cell roles --------------------------------------------------------
    def _col_cell(self, data_type, col, step):
        avail_before = data_type in step.before and col in step.before[data_type].columns
        in_after = data_type in step.after and col in step.after[data_type].columns
        produced = data_type in step.produced_types and in_after and not avail_before
        required = col in step.req["cols"].get(data_type, set())
        missing = col in step.miss["cols"].get(data_type, set())
        return self._block(step.seed, in_after, avail_before, produced, required, missing)

    def _link_cell(self, data_type, name, step):
        avail_before = data_type in step.before and name in step.before[data_type].links
        in_after = data_type in step.after and name in step.after[data_type].links
        produced = data_type in step.produced_types and in_after and not avail_before
        required = name in step.req["links"].get(data_type, set())
        missing = name in step.miss["links"].get(data_type, set())
        return self._block(step.seed, in_after, avail_before, produced, required, missing)

    @staticmethod
    def _block(seed, in_after, avail_before, produced, required, missing):
        if seed:
            return ("available", _BLOCK) if in_after else EMPTY
        if missing:
            return ("missing", _BLOCK)
        if required and produced:
            return ("both", (_BLOCK, _BLOCK))
        if required:
            return ("required", _BLOCK)
        if produced:
            return ("produced", _BLOCK)
        if avail_before:
            return ("passthrough", _DOT)
        return EMPTY

    def _crs_cell(self, data_type, step):
        before_crs = step.before[data_type].crs_set if data_type in step.before else None
        after_crs = step.after[data_type].crs_set if data_type in step.after else None
        if step.seed:
            return ("available", _glyph(after_crs)) if (data_type in step.after and after_crs is not None) else EMPTY
        produced = (data_type in step.produced_types and after_crs is not None
                    and (data_type not in step.before or before_crs != after_crs))
        required = data_type in step.req["crs"]
        if data_type in step.miss["crs"]:
            return ("missing", _glyph(step.miss["crs"][data_type]))
        if required and produced:
            return ("both", (_glyph(step.req["crs"][data_type]), _glyph(after_crs)))
        if required:
            return ("required", _glyph(step.req["crs"][data_type]))
        if produced:
            return ("produced", _glyph(after_crs))
        if data_type in step.before and before_crs is not None:
            return ("passthrough", _glyph(before_crs))
        return EMPTY

    # --- rendering -----------------------------------------------------------
    def _render(self, tracked):
        labels = tracked["labels"]
        sections = tracked["sections"]

        v_bar = '│' if _USE_UNICODE else '|'
        h_bar = '─' if _USE_UNICODE else '-'
        cross = '┼' if _USE_UNICODE else '+'

        keys = [key for _, rows in sections for key, _ in rows] + [name for name, _ in sections]
        row_label_width = max(max((len(k) for k in keys), default=10), 10)
        col_widths = [max(len(label), 3) for label in labels]

        def line(label, cells):
            return f"{label.ljust(row_label_width)}  {v_bar} " + f' {v_bar} '.join(
                _render_cell(cell, width) for cell, width in zip(cells, col_widths))

        header = f"{'DATA'.ljust(row_label_width)}  {v_bar} " + f' {v_bar} '.join(
            label.center(width) for label, width in zip(labels, col_widths))
        separator = (h_bar * (row_label_width + 2) + f'{cross}{h_bar}'
                     + f'{h_bar}{cross}{h_bar}'.join(h_bar * width for width in col_widths) + h_bar)

        print("\n" + "=" * len(header))
        print("PIPELINE FLOW CHART")
        print(f"Legend: {_render_cell(('available', _BLOCK), 1)}=input  "
              f"{_render_cell(('produced', _BLOCK), 1)}=produced  "
              f"{_render_cell(('required', _BLOCK), 1)}=required  "
              f"{_render_cell(('both', (_BLOCK, _BLOCK)), 2)}=required+produced  "
              f"{_render_cell(('missing', _BLOCK), 1)}=MISSING!  "
              f"{_render_cell(('passthrough', _DOT), 1)}=passthrough   "
              f"crs: {_YES}=CRS {_NO}=pixel {_UNK}=undeclared")
        print("=" * len(header))
        print(header)
        print(separator)

        blank = [EMPTY] * len(labels)
        for name, rows in sections:
            print(line(name, blank))
            for key, cells in rows:
                print(line(key, cells))
            print(separator)
        print("=" * len(header) + "\n")
