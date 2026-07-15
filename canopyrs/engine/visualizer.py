"""
Pipeline flow visualization.

The pipeline has two typed tables, and a component's contract is ``Need(type, columns, links, crs,
kind)`` — not a flat column set. So the chart has **two stacked sections** (IMAGERY / OBJECTS); within
each, a row per **link** (``→parent`` / ``→imagery`` / ``→prev_objects``), a **kind** row (s=source,
t=tile) for imagery, a **crs** row (tri-state ✓/✗/?), then a row per **column**. Each cell is marked
available / produced / required / required+produced / missing / passthrough per component — read off
the *same* ``Pipeline.thread_schemas`` simulation that ``validate`` uses (per-type schema *lists*;
cells display the newest schema of each type), so the chart never re-derives "what's available when".
"""

import re
import sys

from canopyrs.engine.constants import ImageKind
from canopyrs.engine.contracts import Requirement, AnyOf, as_requirements
from canopyrs.engine.data import Imagery, Objects


_ANSI_ESCAPE_RE = re.compile(r'\033\[[0-9;]*m')


def _stdout_supports_unicode() -> bool:
    try:
        encoding = getattr(sys.stdout, 'encoding', None) or 'ascii'
        '▬─│┼·✓✗'.encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


class _Colors:
    RESET = "\033[0m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    GRAY = "\033[90m"


_USE_UNICODE = _stdout_supports_unicode()


class _Symbols:
    _BLOCK = '▬' if _USE_UNICODE else '#'
    _DOT = '·' if _USE_UNICODE else '.'
    _YES = '✓' if _USE_UNICODE else 'y'
    _NO = '✗' if _USE_UNICODE else 'n'
    _UNK = '?'
    AVAILABLE = f"{_Colors.GREEN}{_BLOCK}{_Colors.RESET}"    # available at input (seed)
    PRODUCED = f"{_Colors.BLUE}{_BLOCK}{_Colors.RESET}"      # produced by this component
    REQUIRED = f"{_Colors.YELLOW}{_BLOCK}{_Colors.RESET}"    # required and available (consumed)
    REQ_AND_PROD = f"{_Colors.YELLOW}{_BLOCK}{_Colors.BLUE}{_BLOCK}{_Colors.RESET}"  # required + produced
    MISSING = f"{_Colors.RED}{_BLOCK}{_Colors.RESET}"        # required but MISSING
    PASSTHROUGH = f"{_Colors.GRAY}{_DOT}{_Colors.RESET}"     # passthrough (still available, unused here)
    EMPTY = " "                                               # not yet available

    _ROLE_COLOR = {"available": _Colors.GREEN, "produced": _Colors.BLUE, "required": _Colors.YELLOW,
                   "missing": _Colors.RED, "passthrough": _Colors.GRAY}

    @classmethod
    def _glyph(cls, value):
        return cls._YES if value is True else cls._NO if value is False else cls._UNK

    @classmethod
    def crs(cls, role, value, req_value=None):
        """A single crs cell — a ✓/✗/? glyph encoding the CRS-ness, colored by role. ``both`` shows the
        required value then the produced value as a two-color split (e.g. yellow ✗ + blue ✓ = requires
        pixel, produces CRS)."""
        if role == "both":
            return f"{_Colors.YELLOW}{cls._glyph(req_value)}{_Colors.BLUE}{cls._glyph(value)}{_Colors.RESET}"
        color = cls._ROLE_COLOR.get(role)
        return f"{color}{cls._glyph(value)}{_Colors.RESET}" if color else cls.EMPTY

    @classmethod
    def tag(cls, role, letter, req_letter=None):
        """A single letter cell (e.g. the imagery kind: s=source, t=tile), colored by role — same
        rendering rules as the crs glyphs."""
        if role == "both":
            return f"{_Colors.YELLOW}{req_letter or cls._UNK}{_Colors.BLUE}{letter or cls._UNK}{_Colors.RESET}"
        color = cls._ROLE_COLOR.get(role)
        return f"{color}{letter or cls._UNK}{_Colors.RESET}" if color else cls.EMPTY


def _fill_width(s: str, width: int) -> str:
    """Fill `width` by repeating the visible symbol character, preserving ANSI color wrapping."""
    visible_char = _ANSI_ESCAPE_RE.sub('', s)
    if not visible_char or visible_char == ' ':
        return ' ' * width
    if len(visible_char) == 2:                              # two-color split (REQ_AND_PROD / crs both)
        half1 = width // 2
        half2 = width - half1
        parts = re.findall(r'(\033\[[0-9;]*m)(.+?)(?=\033)', s)
        if len(parts) == 2:
            return parts[0][0] + parts[0][1] * half1 + parts[1][0] + parts[1][1] * half2 + _Colors.RESET
    match = re.match(r'(\033\[[0-9;]*m)?(.+?)(\033\[[0-9;]*m)?$', s)
    if match:
        prefix = match.group(1) or ''
        char = match.group(2)
        suffix = match.group(3) or ''
        return prefix + (char * width) + suffix
    return s.center(width)


class _Step:
    """Per-column state for one pipeline step (the seed, or one component)."""

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

    Legend (colored blocks): green ▬ = input · blue ▬ = produced · yellow ▬ = required · yellow+blue =
    required+produced · red ▬ = MISSING · gray · = passthrough. crs row: ✓ = CRS, ✗ = tile-pixel,
    ? = undeclared. kind row: s = source, t = tile.
    """

    SECTIONS = (Imagery, Objects)
    KIND_LETTER = {ImageKind.SOURCE: "s", ImageKind.TILE: "t"}

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def print(self) -> None:
        try:
            self._render(self._track())
        except UnicodeEncodeError:
            pass  # terminals that can't display the chart (e.g. Windows subprocess workers)

    # --- tracking ------------------------------------------------------------
    @staticmethod
    def _newest_view(available):
        """The newest schema per type — cells display the newest instance of each type; requirement
        matching (which may pick an older one) runs on the full lists."""
        return {data_type: schemas[-1] for data_type, schemas in available.items() if schemas}

    def _track(self):
        steps = []
        for component, before, after in self.pipeline.thread_schemas():
            if component is None:
                after_view = self._newest_view(after)
                steps.append(_Step("input", {}, after_view, set(after_view),
                                   self._empty_marks(), self._empty_marks(), seed=True))
            else:
                req, miss = self._requirement_marks(component, before)
                produced_types = {need.data_type for need in as_requirements(component.produces)}
                steps.append(_Step(component.label, self._newest_view(before), self._newest_view(after),
                                   produced_types, req, miss))

        labels = [step.label for step in steps]
        sections = []
        for data_type in self.SECTIONS:
            rows = []
            for name in data_type.fks:                                  # link rows: →parent / →imagery / ...
                cells = [self._link_cell(data_type, name, step) for step in steps]
                if any(cell != _Symbols.EMPTY for cell in cells):
                    rows.append((f"→{name}", cells))
            if data_type is Imagery:
                kind_cells = [self._kind_cell(data_type, step) for step in steps]
                if any(cell != _Symbols.EMPTY for cell in kind_cells):
                    rows.append(("kind", kind_cells))
            crs_cells = [self._crs_cell(data_type, step) for step in steps]
            if any(cell != _Symbols.EMPTY for cell in crs_cells):
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
        return {"cols": {}, "links": {}, "crs": {}, "kind": {}}

    def _requirement_marks(self, component, before):
        get = lambda t: before.get(t, ())   # noqa: E731 — resolve over the full per-type schema lists
        req, miss = self._empty_marks(), self._empty_marks()
        for entry in component.requires:
            spec = Requirement.coerce(entry)
            desc, _ = spec.resolve(get)
            if desc is not None:
                self._add_need(req, self._chosen_need(spec, get))
            else:
                for need in self._alternatives(spec):
                    self._add_need(miss, need)
        return req, miss

    @staticmethod
    def _alternatives(spec):
        return list(spec.alternatives) if isinstance(spec, AnyOf) else [spec]

    def _chosen_need(self, spec, get):
        """The Need the available data actually binds (for a one_of, the first satisfiable alternative)."""
        if not isinstance(spec, AnyOf):
            return spec
        return next(need for need in spec.alternatives if need.resolve(get)[0] is not None)

    @staticmethod
    def _add_need(acc, need):
        acc["cols"].setdefault(need.data_type, set()).update(need.columns)
        acc["links"].setdefault(need.data_type, set()).update(need.links)
        if need.crs is not None:
            acc["crs"][need.data_type] = need.crs
        if need.kind is not None:
            acc["kind"][need.data_type] = need.kind

    # --- per-cell symbols ----------------------------------------------------
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
            return _Symbols.AVAILABLE if in_after else _Symbols.EMPTY
        if missing:
            return _Symbols.MISSING
        if required and produced:
            return _Symbols.REQ_AND_PROD
        if required:
            return _Symbols.REQUIRED
        if produced:
            return _Symbols.PRODUCED
        if avail_before:
            return _Symbols.PASSTHROUGH
        return _Symbols.EMPTY

    def _kind_cell(self, data_type, step):
        before_kind = step.before[data_type].kind if data_type in step.before else None
        after_kind = step.after[data_type].kind if data_type in step.after else None
        letter = self.KIND_LETTER.get(after_kind)
        if step.seed:
            return _Symbols.tag("available", letter) if (data_type in step.after and after_kind) else _Symbols.EMPTY
        produced = (data_type in step.produced_types and after_kind is not None
                    and (data_type not in step.before or before_kind != after_kind))
        required = data_type in step.req["kind"]
        if data_type in step.miss["kind"]:
            return _Symbols.tag("missing", self.KIND_LETTER.get(step.miss["kind"][data_type]))
        if required and produced:
            return _Symbols.tag("both", letter, req_letter=self.KIND_LETTER.get(step.req["kind"][data_type]))
        if required:
            return _Symbols.tag("required", self.KIND_LETTER.get(step.req["kind"][data_type]))
        if produced:
            return _Symbols.tag("produced", letter)
        if data_type in step.before and before_kind is not None:
            return _Symbols.tag("passthrough", self.KIND_LETTER.get(before_kind))
        return _Symbols.EMPTY

    def _crs_cell(self, data_type, step):
        before_crs = step.before[data_type].crs_set if data_type in step.before else None
        after_crs = step.after[data_type].crs_set if data_type in step.after else None
        if step.seed:
            return _Symbols.crs("available", after_crs) if (data_type in step.after and after_crs is not None) else _Symbols.EMPTY
        produced = (data_type in step.produced_types and after_crs is not None
                    and (data_type not in step.before or before_crs != after_crs))
        required = data_type in step.req["crs"]
        if data_type in step.miss["crs"]:
            return _Symbols.crs("missing", step.miss["crs"][data_type])
        if required and produced:
            return _Symbols.crs("both", after_crs, req_value=step.req["crs"][data_type])
        if required:
            return _Symbols.crs("required", step.req["crs"][data_type])
        if produced:
            return _Symbols.crs("produced", after_crs)
        if data_type in step.before and before_crs is not None:
            return _Symbols.crs("passthrough", before_crs)
        return _Symbols.EMPTY

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
                _fill_width(cell, width) for cell, width in zip(cells, col_widths))

        header = f"{'DATA'.ljust(row_label_width)}  {v_bar} " + f' {v_bar} '.join(
            label.center(width) for label, width in zip(labels, col_widths))
        separator = (h_bar * (row_label_width + 2) + f'{cross}{h_bar}'
                     + f'{h_bar}{cross}{h_bar}'.join(h_bar * width for width in col_widths) + h_bar)

        print("\n" + "=" * len(header))
        print("PIPELINE FLOW CHART")
        print(f"Legend: {_Symbols.AVAILABLE}=input  {_Symbols.PRODUCED}=produced  "
              f"{_Symbols.REQUIRED}=required  {_Symbols.REQ_AND_PROD}=required+produced  "
              f"{_Symbols.MISSING}=MISSING!  {_Symbols.PASSTHROUGH}=passthrough   "
              f"crs: {_Symbols._YES}=CRS {_Symbols._NO}=pixel {_Symbols._UNK}=undeclared   "
              f"kind: s=source t=tile")
        print("=" * len(header))
        print(header)
        print(separator)

        blank = [_Symbols.EMPTY] * len(labels)
        for name, rows in sections:
            print(line(name, blank))
            for key, cells in rows:
                print(line(key, cells))
            print(separator)
        print("=" * len(header) + "\n")
