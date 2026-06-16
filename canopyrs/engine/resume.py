"""
Resume / initialize orchestration: restore a previous run's state into a fresh DataState.

These functions read a previous run via ``RunState.open`` and apply its last fully-done snapshot.
The active run's progress recording lives on ``RunState`` (see persistence.py); the pipeline owns
one and drives it.
"""

import shutil
from pathlib import Path
from typing import List, Set

from canopyrs.engine.data_state import DataState
from canopyrs.engine.utils import get_component_folder_name
from canopyrs.engine.persistence import (
    RunState, config_hash, STATE_DIRNAME, STATUS_FILENAME, STATE_FILENAME,
)


class ResumeError(RuntimeError):
    """Raised when a run cannot be resumed / initialized from a previous output folder."""


def seed_from_run(src_run: Path, data_state: DataState, *, restore_registries: bool) -> List[int]:
    """Restore the last fully-done component's snapshot from a previous run into data_state."""
    prev = RunState.open(src_run)
    done = prev.done_prefix()
    if not done:
        raise ResumeError(f"No fully-done component in {src_run}; nothing to restore.")
    data_state.apply_snapshot(prev.load(done[-1]), restore_registries=restore_registries)
    return done


def prepare_resume(resume_from: Path, output_path: Path, components, data_state) -> Set[int]:
    """Restore state, refuse on config drift, and (cross-folder) symlink + copy the _state files."""
    prev = RunState.open(resume_from)
    done = prev.done_prefix()
    for cid in done:
        if cid >= len(components) or prev.config_hash_of(cid) != config_hash(components[cid].config):
            raise ResumeError(
                f"Config drift at component {cid}: the resumed pipeline differs from the previous "
                f"run. Re-run from scratch, or use initialize_from to seed a new-config pipeline."
            )
    if done:
        data_state.apply_snapshot(prev.load(done[-1]), restore_registries=True)
    if Path(output_path).resolve() != Path(resume_from).resolve():
        _relink(Path(resume_from), Path(output_path), done, components)
    return set(done)


def _relink(resume_from: Path, output_path: Path, done, components) -> None:
    """
    Cross-folder resume: symlink the done component folders into the new output and copy the
    _state files across. Restored paths keep pointing at resume_from (read-only inputs read in
    place), so resume_from must remain available; no path rewriting is needed.
    """
    out_state = output_path / STATE_DIRNAME
    out_state.mkdir(parents=True, exist_ok=True)
    for cid in done:
        folder = get_component_folder_name(cid, components[cid].name)
        src, dst = resume_from / folder, output_path / folder
        if dst.is_symlink():
            dst.unlink()
        elif dst.exists():
            continue  # a real dir is already present; leave it
        if src.exists():
            dst.symlink_to(src.resolve(), target_is_directory=True)
    for name in (STATUS_FILENAME, STATE_FILENAME):
        shutil.copy(resume_from / STATE_DIRNAME / name, out_state / name)
