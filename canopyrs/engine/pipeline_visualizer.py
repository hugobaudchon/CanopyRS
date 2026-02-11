"""
Pipeline flow visualization.

Provides a visual representation of state/column flow through pipeline components.
Separated from Pipeline class to keep execution logic focused.
"""

import re
import sys
from typing import List, Set, Dict, Any

from canopyrs.engine.components.base import BaseComponent


# Regex to strip ANSI escape codes for width calculation
_ANSI_ESCAPE_RE = re.compile(r'\033\[[0-9;]*m')


def _stdout_supports_unicode() -> bool:
    """Check if stdout can encode Unicode box-drawing and block characters."""
    try:
        encoding = getattr(sys.stdout, 'encoding', None) or 'ascii'
        '▬─│┼·'.encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def _fill_width(s: str, width: int) -> str:
    """Fill the width by repeating the symbol character."""
    # Extract the visible character from the ANSI-wrapped string
    visible_char = _ANSI_ESCAPE_RE.sub('', s)
    if not visible_char or visible_char == ' ':
        return ' ' * width
    # Extract color codes (everything before and after the visible char)
    match = re.match(r'(\033\[[0-9;]*m)?(.+?)(\033\[[0-9;]*m)?$', s)
    if match:
        prefix = match.group(1) or ''
        char = match.group(2)
        suffix = match.group(3) or ''
        return prefix + (char * width) + suffix
    return s.center(width)


# ANSI color codes for terminal output
class _Colors:
    """ANSI escape codes for colored terminal output."""
    RESET = "\033[0m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    GRAY = "\033[90m"


# Colored block symbols for flow visualization (with ASCII fallback for Windows cp1252)
_USE_UNICODE = _stdout_supports_unicode()


class _Symbols:
    """Colored block symbols for the flow chart."""
    _BLOCK = '▬' if _USE_UNICODE else '#'
    _DOT = '·' if _USE_UNICODE else '.'
    AVAILABLE = f"{_Colors.GREEN}{_BLOCK}{_Colors.RESET}"    # Available at input
    PRODUCED = f"{_Colors.BLUE}{_BLOCK}{_Colors.RESET}"      # Produced by component
    REQUIRED = f"{_Colors.YELLOW}{_BLOCK}{_Colors.RESET}"    # Required and available
    MISSING = f"{_Colors.RED}{_BLOCK}{_Colors.RESET}"        # Required but MISSING
    PASSTHROUGH = f"{_Colors.GRAY}{_DOT}{_Colors.RESET}"     # Passthrough
    EMPTY = " "                                               # Not yet available


class PipelineFlowVisualizer:
    """
    Visualizes data flow through pipeline components.

    Shows what each component requires, produces, and what passes through,
    helping users understand and debug pipeline configurations.

    Legend (colored blocks):
        Green  ▬ = available at input
        Blue   ▬ = produced by this component
        Yellow ▬ = required and available (consumed)
        Red    ▬ = required but MISSING (error!)
        Gray   · = passthrough
        (empty)  = not yet available
    """

    def __init__(
        self,
        components: List[BaseComponent],
        initial_state_keys: Set[str],
        initial_columns: Set[str],
    ):
        """
        Initialize visualizer.

        Args:
            components: List of pipeline components
            initial_state_keys: State keys available at pipeline start
            initial_columns: GDF columns available at pipeline start
        """
        self.components = components
        self.initial_state_keys = initial_state_keys
        self.initial_columns = initial_columns

    def print(self) -> None:
        """Print the flow chart to stdout."""
        flow_tracker = self._track_flow()
        try:
            self._render(flow_tracker)
        except UnicodeEncodeError:
            pass  # Skip on terminals that can't display the chart (e.g. Windows subprocess workers)

    def _track_flow(self) -> Dict[str, Any]:
        """
        Track state and column flow through all components.

        Returns:
            Dictionary with 'components', 'state_flow', and 'column_flow'
        """
        available_state = set(self.initial_state_keys)
        available_columns = set(self.initial_columns)

        # Initialize flow tracker
        flow_tracker = {
            'components': ['input'],
            'state_flow': {key: [_Symbols.AVAILABLE] for key in available_state},
            'column_flow': {col: [_Symbols.AVAILABLE] for col in available_columns},
        }

        # Simulate running through each component
        for i, component in enumerate(self.components):
            component_label = f"{i}_{component.name}"
            flow_tracker['components'].append(component_label)

            # Track state flow
            self._track_state_requirements(flow_tracker, component, available_state)
            self._track_state_productions(flow_tracker, component, available_state)

            # Track column flow
            self._track_column_requirements(flow_tracker, component, available_columns)
            self._track_column_productions(flow_tracker, component, available_columns)

            # Update available state/columns with what this component produces
            available_state = available_state | component.produces_state
            available_columns = available_columns | component.produces_columns

        return flow_tracker

    def _track_state_requirements(
        self,
        flow_tracker: Dict[str, Any],
        component: BaseComponent,
        available_state: Set[str]
    ) -> None:
        """Track state requirements for a component."""
        for key in component.requires_state:
            if key not in flow_tracker['state_flow']:
                flow_tracker['state_flow'][key] = [_Symbols.EMPTY] * len(flow_tracker['components'])
            # Pad to current length
            self._pad_flow_list(flow_tracker['state_flow'][key], len(flow_tracker['components']))
            # Mark as missing or required
            flow_tracker['state_flow'][key][-1] = _Symbols.MISSING if key not in available_state else _Symbols.REQUIRED

    def _track_state_productions(
        self,
        flow_tracker: Dict[str, Any],
        component: BaseComponent,
        available_state: Set[str]
    ) -> None:
        """Track state productions for a component."""
        for key in component.produces_state:
            if key not in flow_tracker['state_flow']:
                flow_tracker['state_flow'][key] = [_Symbols.EMPTY] * len(flow_tracker['components'])
            self._pad_flow_list(flow_tracker['state_flow'][key], len(flow_tracker['components']))
            # Mark as produced unless already marked as required/missing
            if flow_tracker['state_flow'][key][-1] not in (_Symbols.MISSING, _Symbols.REQUIRED):
                flow_tracker['state_flow'][key][-1] = _Symbols.PRODUCED

    def _track_column_requirements(
        self,
        flow_tracker: Dict[str, Any],
        component: BaseComponent,
        available_columns: Set[str]
    ) -> None:
        """Track column requirements for a component."""
        for col in component.requires_columns:
            if col not in flow_tracker['column_flow']:
                flow_tracker['column_flow'][col] = [_Symbols.EMPTY] * len(flow_tracker['components'])
            self._pad_flow_list(flow_tracker['column_flow'][col], len(flow_tracker['components']))
            # Mark as missing or required
            flow_tracker['column_flow'][col][-1] = _Symbols.MISSING if col not in available_columns else _Symbols.REQUIRED

    def _track_column_productions(
        self,
        flow_tracker: Dict[str, Any],
        component: BaseComponent,
        available_columns: Set[str]
    ) -> None:
        """Track column productions for a component."""
        for col in component.produces_columns:
            if col not in flow_tracker['column_flow']:
                flow_tracker['column_flow'][col] = [_Symbols.EMPTY] * len(flow_tracker['components'])
            self._pad_flow_list(flow_tracker['column_flow'][col], len(flow_tracker['components']))
            # Mark as produced unless already marked as required/missing
            if flow_tracker['column_flow'][col][-1] not in (_Symbols.MISSING, _Symbols.REQUIRED):
                flow_tracker['column_flow'][col][-1] = _Symbols.PRODUCED

    def _pad_flow_list(self, flow_list: List[str], target_length: int) -> None:
        """Pad flow list to target length, using passthrough marker if previously available."""
        while len(flow_list) < target_length:
            prev_val = flow_list[-1] if flow_list else _Symbols.EMPTY
            # After available, produced, required, or passthrough, the item is still available
            flow_list.append(
                _Symbols.PASSTHROUGH if prev_val in (
                    _Symbols.AVAILABLE, _Symbols.PASSTHROUGH, _Symbols.PRODUCED, _Symbols.REQUIRED
                )
                else _Symbols.EMPTY
            )

    def _render(self, flow_tracker: Dict[str, Any]) -> None:
        """Render and print the flow chart from tracked data."""
        components = flow_tracker['components']
        state_flow = flow_tracker['state_flow']
        column_flow = flow_tracker['column_flow']

        # Box-drawing characters (with ASCII fallback)
        v_bar = '│' if _USE_UNICODE else '|'
        h_bar = '─' if _USE_UNICODE else '-'
        cross = '┼' if _USE_UNICODE else '+'

        # Calculate column widths
        row_label_width = max(
            max((len(k) for k in state_flow.keys()), default=10),
            max((len(k) for k in column_flow.keys()), default=10),
            10
        )
        col_widths = [max(len(c), 3) for c in components]

        # Build header
        header = f"{' ' * row_label_width}  {v_bar} " + f' {v_bar} '.join(
            c.center(w) for c, w in zip(components, col_widths)
        )
        separator = h_bar * (row_label_width + 2) + f'{cross}{h_bar}' + f'{h_bar}{cross}{h_bar}'.join(h_bar * w for w in col_widths) + h_bar

        print("\n" + "=" * len(header))
        print("PIPELINE FLOW CHART")
        print(f"Legend: {_Symbols.AVAILABLE}=input  {_Symbols.PRODUCED}=produced  {_Symbols.REQUIRED}=required  {_Symbols.MISSING}=MISSING!  {_Symbols.PASSTHROUGH}=passthrough")
        print("=" * len(header))
        print(header)
        print(separator)

        # Print state rows
        if state_flow:
            print(f"{'STATE KEYS'.ljust(row_label_width)}  {v_bar} " + f' {v_bar} '.join(' ' * w for w in col_widths))
            for key in sorted(state_flow.keys()):
                values = state_flow[key]
                self._pad_flow_list(values, len(components))
                row = f"{key.ljust(row_label_width)}  {v_bar} " + f' {v_bar} '.join(
                    _fill_width(v, w) for v, w in zip(values, col_widths)
                )
                print(row)
            print(separator)

        # Print column rows
        if column_flow:
            print(f"{'GDF COLUMNS'.ljust(row_label_width)}  {v_bar} " + f' {v_bar} '.join(' ' * w for w in col_widths))
            for col in sorted(column_flow.keys()):
                values = column_flow[col]
                self._pad_flow_list(values, len(components))
                row = f"{col.ljust(row_label_width)}  {v_bar} " + f' {v_bar} '.join(
                    _fill_width(v, w) for v, w in zip(values, col_widths)
                )
                print(row)

        print("=" * len(header) + "\n")
