"""
Persistence for pipeline resume / initialize.

A run's resumable state lives under ``{output}/_state/`` in two files:
  - ``status.jsonl`` : one row per component {component_id, name, config_hash, status, side_status}.
    ``status`` tracks synchronous compute (todo/running/done/error); ``side_status`` the async side
    outputs (none/pending/done/error). Fully done iff status==done and side_status in {none, done}.
  - ``state.json``   : {component_id: {fields, gdf_ref, config_hash}}, where ``fields`` are the
    persisted DataState fields and ``gdf_ref`` recovers the gdf from the gpkg already written.

``RunState`` owns both files (and the lock guarding them): the pipeline drives mark_running /
mark_done / mark_error / finalize, while ``RunState.open`` + ``load`` read a previous run back.
"""

import ast
import hashlib
import json
import os
import threading
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np

from canopyrs.engine.constants import Col
from canopyrs.engine.data_state import DataState
from canopyrs.engine.utils import get_component_folder_name

STATE_DIRNAME = "_state"
STATUS_FILENAME = "status.jsonl"
STATE_FILENAME = "state.json"

# Container-typed cell values gpkg can't store natively (it stringifies them) -> decode on load.
_LIST_LIKE = (list, tuple, set, dict, np.ndarray)

TODO, RUNNING, DONE, ERROR = "todo", "running", "done", "error"
SIDE_NONE, SIDE_PENDING, SIDE_DONE, SIDE_ERROR = "none", "pending", "done", "error"


def config_hash(component_config) -> str:
    """Stable short hash of a Pydantic component config, to detect config drift on resume."""
    payload = json.dumps(component_config.model_dump(mode="json"), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _encode(value):
    """Make a DataState field value JSON-safe (Path/set markers, recursively)."""
    if isinstance(value, Path):
        return {"__path__": str(value)}
    if isinstance(value, set):
        return {"__set__": [_encode(v) for v in value]}
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    return value


def _decode(value):
    if isinstance(value, dict):
        if "__path__" in value:
            return Path(value["__path__"])
        if "__set__" in value:
            return {_decode(v) for v in value["__set__"]}
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def _decode_cell(v):
    """Parse a single gpkg-stringified container value back to a Python object."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (ValueError, TypeError):
            try:
                return ast.literal_eval(v)
            except (ValueError, SyntaxError):
                return v
    return v


def detect_list_columns(gdf) -> List[str]:
    """Columns holding container values (inspected in-memory); their names are stored in gdf_ref."""
    cols = []
    for col in gdf.columns:
        if col == Col.GEOMETRY:
            continue
        nonnull = gdf[col][gdf[col].notna()]
        if len(nonnull) and isinstance(nonnull.iloc[0], _LIST_LIKE):
            cols.append(col)
    return cols


def gdf_ref(data_state: DataState) -> Optional[dict]:
    """Reference to recover the current gdf on resume: the latest gpkg + its list columns (or None).

    Uses the latest gpkg across all components: gdf-modifying components always save one, so it
    reflects the current infer_gdf even past a component that carried it forward without saving.
    """
    gdf = data_state.infer_gdf
    if gdf is None or len(gdf) == 0:
        return None
    best_id, best_path = -1, None
    for key, files in data_state.component_output_files.items():
        cid = int(key.split("_")[0])
        for file_type in ("gpkg", "pre_aggregated_gpkg"):
            if file_type in files and cid > best_id:
                best_id, best_path = cid, files[file_type]
    if best_path is None:
        return None
    return {"gpkg_path": str(best_path), "list_columns": detect_list_columns(gdf)}


class Snapshot:
    """One component's restored state: decoded ``fields`` + a lazily-loaded gdf via gdf_ref."""

    def __init__(self, fields_dict: Dict, gdf_ref_dict: Optional[Dict]):
        self.fields = fields_dict
        self._gdf_ref = gdf_ref_dict

    @property
    def gdf(self) -> Optional[gpd.GeoDataFrame]:
        if not self._gdf_ref:
            return None
        gdf = gpd.read_file(self._gdf_ref["gpkg_path"])
        for col in self._gdf_ref.get("list_columns", []):  # gpkg stored these as strings
            if col in gdf.columns:
                gdf[col] = gdf[col].apply(_decode_cell)
        return gdf


def resolve_done_prefix(rows: List[Dict]) -> List[int]:
    """The contiguous prefix of component ids that are fully done (the resumable prefix)."""
    done = []
    for row in sorted(rows, key=lambda r: r["component_id"]):
        if row.get("status") == DONE and row.get("side_status") in (SIDE_NONE, SIDE_DONE):
            done.append(row["component_id"])
        else:
            break
    return done


class RunState:
    """
    The run's resumable state (the ``_state/`` directory). Construct for the active run to record
    progress; use ``RunState.open`` to read a previous run for resume/initialize.
    """

    def __init__(self, output_path, components, resume_done_ids=None):
        self._setup(output_path)
        self.components = components
        self.hashes = {i: config_hash(c.config) for i, c in enumerate(components)}
        self.resuming = resume_done_ids is not None
        self.done_ids = set(resume_done_ids or ())
        if not self.resuming:  # fresh run: every component starts as todo
            self._status = {
                i: {"component_id": i, "name": c.name, "config_hash": self.hashes[i],
                    "status": TODO, "side_status": SIDE_NONE}
                for i, c in enumerate(components)
            }
            self._flush_status()

    @classmethod
    def open(cls, run_dir) -> "RunState":
        """Read-only handle on a previous run's _state (no writes, no components)."""
        if not (Path(run_dir) / STATE_DIRNAME / STATUS_FILENAME).exists():
            raise FileNotFoundError(f"No '{STATE_DIRNAME}/' in {run_dir}; it has no resumable state.")
        self = cls.__new__(cls)
        self._setup(run_dir)
        self.components, self.hashes, self.resuming, self.done_ids = None, None, False, set()
        return self

    # progress recording (active run)
    def is_done(self, component_id: int) -> bool:
        return component_id in self.done_ids

    def mark_running(self, component_id: int) -> None:
        self._set(component_id, status=RUNNING)

    def mark_error(self, component_id: int) -> None:
        self._set(component_id, status=ERROR)

    def mark_done(self, component, data_state: DataState) -> None:
        """Snapshot state, mark compute done, and track async side outputs (non-blocking)."""
        cid = component.component_id
        self._snapshot(cid, data_state)
        self._set(cid, status=DONE)
        sides = [sp for sp in data_state.side_processes
                 if isinstance(sp, tuple) and len(sp) > 2 and sp[2].get("component_id") == cid]
        if not sides:
            self._set(cid, side_status=SIDE_NONE)
            return
        with self.lock:
            self._pending[cid] = len(sides)
        self._set(cid, side_status=SIDE_PENDING)
        for state_key, future, reg_info in sides:
            future.add_done_callback(self._side_callback(cid, component.name, state_key, reg_info))

    def finalize(self) -> None:
        """On a successful run, flip any still-pending side statuses to done (callbacks have run)."""
        for cid, row in list(self._status.items()):
            if row.get("status") == DONE and row.get("side_status") == SIDE_PENDING:
                self._set(cid, side_status=SIDE_DONE)

    # reading a previous run (resume / initialize)
    def done_prefix(self) -> List[int]:
        return resolve_done_prefix(list(self._status.values()))

    def config_hash_of(self, component_id: int) -> Optional[str]:
        entry = self._state.get(str(component_id))
        return entry.get("config_hash") if entry else None

    def load(self, component_id: int) -> Snapshot:
        entry = self._state[str(component_id)]
        return Snapshot({k: _decode(v) for k, v in entry["fields"].items()}, entry.get("gdf_ref"))

    # internals
    def _setup(self, run_dir) -> None:
        self.state_dir = Path(run_dir) / STATE_DIRNAME
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self._pending: dict = {}
        status_path = self.state_dir / STATUS_FILENAME
        self._status = {}
        if status_path.exists():
            for line in status_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._status[row["component_id"]] = row
        state_path = self.state_dir / STATE_FILENAME
        self._state = json.loads(state_path.read_text()) if state_path.exists() else {}

    def _set(self, component_id: int, *, status: str = None, side_status: str = None) -> None:
        with self.lock:
            row = self._status[component_id]
            if status is not None:
                row["status"] = status
            if side_status is not None:
                row["side_status"] = side_status
            self._flush_status()

    def _snapshot(self, component_id: int, data_state: DataState) -> None:
        with self.lock:
            persisted = {}
            for f in fields(DataState):
                if f.metadata.get("persist") not in ("input", "registry"):
                    continue
                value = getattr(data_state, f.name)
                if isinstance(value, gpd.GeoDataFrame):
                    continue  # gdf recovered via gdf_ref, never serialized here
                persisted[f.name] = _encode(value)
            self._state[str(component_id)] = {
                "fields": persisted, "gdf_ref": gdf_ref(data_state),
                "config_hash": self.hashes[component_id],
            }
            self._flush_state()

    def _side_callback(self, cid: int, name: str, state_key: Optional[str], reg_info: dict):
        file_type, expected = reg_info.get("file_type"), reg_info.get("expected_path")

        def _cb(future):
            try:
                result = future.result()
                self._register_side_output(cid, name, state_key, file_type,
                                           result if result is not None else expected)
                with self.lock:
                    self._pending[cid] = self._pending.get(cid, 1) - 1
                    remaining = self._pending[cid]
                if remaining <= 0:
                    self._set(cid, side_status=SIDE_DONE)
            except Exception:
                self._set(cid, side_status=SIDE_ERROR)

        return _cb

    def _register_side_output(self, cid, name, state_key, file_type, path) -> None:
        """Patch a component's snapshot once an async side output (e.g. COCO) is written."""
        with self.lock:
            entry = self._state.get(str(cid))
            if entry is None:
                return
            path = str(path)
            if state_key:  # scalar read by downstream components
                entry["fields"][state_key] = _encode(path)
            key = get_component_folder_name(cid, name)
            entry["fields"].setdefault("component_output_files", {}).setdefault(key, {})[file_type] = \
                _encode(Path(path))
            self._flush_state()

    def _flush_status(self) -> None:
        text = "".join(json.dumps(self._status[k]) + "\n" for k in sorted(self._status))
        _atomic_write(self.state_dir / STATUS_FILENAME, text)

    def _flush_state(self) -> None:
        _atomic_write(self.state_dir / STATE_FILENAME, json.dumps(self._state, indent=2))
