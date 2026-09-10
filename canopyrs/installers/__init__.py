"""`canopyrs setup` target registry: name/alias resolution, dependency chains, and the runner.

Each target is a module here (see common.py for the unit contract). `run_targets` resolves
aliases, expands dependency chains in order (detrex -> detectron2 first), dedupes, and runs
each unit verify-first: an already-working target is skipped, so re-runs are cheap and the
whole command is idempotent.
"""

from canopyrs.installers import deepforest, detectron2, detrex, mmdet, rfdetr, sam, sam3
from canopyrs.installers.common import SetupError

TARGETS = {m.NAME: m for m in (deepforest, detectron2, detrex, mmdet, rfdetr, sam, sam3)}

_ALIASES = {alias: m.NAME for m in TARGETS.values() for alias in m.ALIASES}

# Things users will reasonably type that are already satisfied by core — answered kindly.
CORE_BUILTINS = {
    "sam2": "sam2 is part of CanopyRS core — already available.",
    "core": "CanopyRS core is what you `pip install` — nothing extra to set up.",
    "torch": "torch is a core dependency — already available.",
}


def resolve(names):
    """User-typed names -> ordered, deduped list of target modules (dependencies first).
    Returns (modules, notes) where notes are messages for core-builtin names."""
    modules, notes, seen = [], [], set()

    def add(name, requested_by=None):
        canonical = _ALIASES.get(name, name)
        if canonical in seen:
            return
        if canonical in CORE_BUILTINS:
            notes.append(CORE_BUILTINS[canonical])
            return
        if canonical not in TARGETS:
            known = sorted(set(TARGETS) | set(_ALIASES) | set(CORE_BUILTINS))
            raise SetupError(f"Unknown setup target '{name}'. Known targets: {', '.join(known)}")
        seen.add(canonical)
        for dep in TARGETS[canonical].REQUIRES:
            add(dep, requested_by=canonical)
        modules.append(TARGETS[canonical])

    for n in names:
        add(n.lower())
    return modules, notes


def run_targets(names, force=False) -> int:
    """Set up the named targets (and their dependency chains). Returns a process exit code."""
    try:
        modules, notes = resolve(names)
    except SetupError as e:
        print(f"ERROR: {e}")
        return 2
    for note in notes:
        print(note)

    failed = []
    for mod in modules:
        ok, detail = mod.check()
        if ok and not force:
            print(f"[{mod.NAME}] already OK — skipping ({detail})")
            continue
        print(f"[{mod.NAME}] setting up... ({detail})")
        try:
            mod.install()
        except SetupError as e:
            print(f"[{mod.NAME}] FAILED:\n{e}")
            failed.append(mod.NAME)
            continue
        ok, detail = mod.check()
        print(f"[{mod.NAME}] {'OK' if ok else 'STILL BROKEN'} — {detail}")
        if not ok:
            failed.append(mod.NAME)

    if failed:
        print(f"\nSetup incomplete: {', '.join(failed)} failed. Run `canopyrs doctor` for a full report.")
        return 1
    return 0
