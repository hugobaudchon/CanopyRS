"""
Base component class with validation support.

Design principles:
1. Class constants for static requirements (visible at class level)
2. Instance attributes for config-dependent requirements (set in __init__)
3. Single __call__() method that returns ComponentResult
4. Pipeline handles all I/O (saving gpkg, COCO generation, state updates)
5. Validation happens before pipeline runs, and at runtime via decorator
6. Helpful hint messages for missing requirements
"""

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Set, Optional, List, Dict, Any, Callable, TypeVar

import geopandas as gpd

from canopyrs.engine.constants import StateKey
from canopyrs.engine.data_state import DataState

# Type variable for the decorator
T = TypeVar('T')


# =============================================================================
# Validation Error
# =============================================================================

class ComponentValidationError(Exception):
    """Raised when component validation fails."""
    pass


# =============================================================================
# Runtime Validation Decorator
# =============================================================================

def validate_requirements(method: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator that validates data_state has required state keys and GDF columns at runtime.

    Checks the component's requires_state and requires_columns against
    the actual data_state before executing the component.

    Usage:
        @validate_requirements
        def __call__(self, data_state: DataState) -> ComponentResult:
            ...
    """
    @functools.wraps(method)
    def wrapper(self: 'BaseComponent', data_state: DataState) -> 'ComponentResult':
        # Check state keys (must exist and not be None)
        missing_state = set()
        for key in self.requires_state:
            if not hasattr(data_state, key) or getattr(data_state, key) is None:
                missing_state.add(key)

        if missing_state:
            hints = []
            for k in sorted(missing_state):
                hint = self.state_hints.get(k, "")
                hints.append(f"  - {k}" + (f" ({hint})" if hint else ""))
            raise ComponentValidationError(
                f"[Runtime] Component '{self.name}' missing required state:\n" + "\n".join(hints)
            )

        # Check GDF columns (if component requires columns)
        if self.requires_columns:
            if data_state.infer_gdf is None:
                raise ComponentValidationError(
                    f"[Runtime] Component '{self.name}' requires GDF columns {self.requires_columns} "
                    f"but infer_gdf is None"
                )

            available_cols = set(data_state.infer_gdf.columns)
            missing_cols = self.requires_columns - available_cols

            if missing_cols:
                hints = []
                for c in sorted(missing_cols):
                    hint = self.column_hints.get(c, "")
                    hints.append(f"  - {c}" + (f" ({hint})" if hint else ""))
                raise ComponentValidationError(
                    f"[Runtime] Component '{self.name}' missing required GDF columns:\n" + "\n".join(hints)
                )

        return method(self, data_state)
    return wrapper


# =============================================================================
# Component Result
# =============================================================================

@dataclass
class ComponentResult:
    """
    Output from component __call__().
    Pipeline uses this to update DataState and handle I/O.
    """
    # Core output data
    gdf: Optional[gpd.GeoDataFrame] = None
    produced_columns: Set[str] = field(default_factory=set)
    objects_are_new: bool = True  # If True, GDF replaces existing; if False, merge into existing (if able to)

    # State updates (for non-GDF state like tiles_path)
    state_updates: Dict[str, Any] = field(default_factory=dict)

    # I/O flags - Pipeline handles actual saving
    save_gpkg: bool = False
    gpkg_name_suffix: str = "notaggregated"  # e.g., "notaggregated", "aggregated"

    save_coco: bool = False
    coco_scores_column: Optional[str] = None
    coco_categories_column: Optional[str] = None

    # Files already written by the component itself (e.g. geodataset internals).
    # Pipeline will register these in component_output_files without re-saving.
    # Keys are file_type strings (e.g. 'coco', 'gpkg', 'pre_aggregated_gpkg').
    output_files: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Base Component
# =============================================================================

class BaseComponent(ABC):
    """
    Base class for all pipeline components.

    Components have a single interface:
    - __call__(data_state) -> ComponentResult

    Pipeline handles:
    - State updates (based on ComponentResult)
    - File I/O (saving gpkg, COCO generation)
    - Output registration

    Requirements are declared as class/instance attributes:
    - requires_state: Set of StateKey values needed
    - requires_columns: Set of Col values needed in GDF
    - produces_state: Set of StateKey values produced
    - produces_columns: Set of Col values produced

    Optional hints for better error messages:
    - state_hints: Dict mapping state key to helpful message
    - column_hints: Dict mapping column name to helpful message
    """

    name: str

    # Default empty requirements (subclasses override)
    requires_state: Set[str] = set()
    requires_columns: Set[str] = set()
    produces_state: Set[str] = set()
    produces_columns: Set[str] = set()

    # Optional hints for validation errors (subclasses can override)
    state_hints: Dict[str, str] = {}
    column_hints: Dict[str, str] = {}

    def __init__(
        self,
        config,
        parent_output_path: str = None,
        component_id: int = None
    ):
        self.config = config
        self.parent_output_path = parent_output_path
        self.component_id = component_id
        self.output_path = None  # Set by Pipeline before calling

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate(
        self,
        available_state: Set[str] = None,
        available_columns: Set[str] = None,
        raise_on_error: bool = True,
    ) -> List[str]:
        """
        Validate that this component can run with the given inputs.

        Used by Pipeline before running to catch config errors early.

        Args:
            available_state: Set of available state keys
            available_columns: Set of available GDF column names
            raise_on_error: If True, raise ComponentValidationError

        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        available_state = available_state or set()
        available_columns = available_columns or set()

        # Check state requirements
        missing_state = self.requires_state - available_state
        for key in missing_state:
            hint = self.state_hints.get(key, "")
            msg = f"Missing required state: '{key}'"
            if hint:
                msg += f"\n    -> Hint: {hint}"
            errors.append(msg)

        # Check column requirements
        missing_columns = self.requires_columns - available_columns
        for col in missing_columns:
            hint = self.column_hints.get(col, "")
            msg = f"Missing required column: '{col}'"
            if hint:
                msg += f"\n    -> Hint: {hint}"
            errors.append(msg)

        if errors and raise_on_error:
            error_msg = f"Component '{self.name}' validation failed:\n"
            error_msg += "\n".join(f"  * {e}" for e in errors)
            raise ComponentValidationError(error_msg)

        return errors

    # -------------------------------------------------------------------------
    # Abstract Method - Single Interface
    # -------------------------------------------------------------------------

    @abstractmethod
    def __call__(self, data_state: DataState) -> ComponentResult:
        """
        Run the component's core logic.

        Args:
            data_state: Current pipeline state

        Returns:
            ComponentResult with outputs and I/O instructions
        """
        pass

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def describe(self) -> str:
        """Return a human-readable description of this component's requirements."""
        lines = [
            f"Component: {self.name}",
            f"  Requires state: {self.requires_state or 'none'}",
            f"  Requires columns: {self.requires_columns or 'none'}",
            f"  Produces state: {self.produces_state or 'none'}",
            f"  Produces columns: {self.produces_columns or 'none'}",
        ]
        return "\n".join(lines)
