"""Installer: DeepForest (plain PyPI package from Weecology)."""

from canopyrs.installers.common import importable, install_extra

NAME = "deepforest"
ALIASES = ()
REQUIRES = ()
EXTRA = "deepforest"


def check():
    if importable("deepforest"):
        return False, "deepforest not installed"
    import deepforest
    return True, f"deepforest {getattr(deepforest, '__version__', '?')} OK"


def install():
    install_extra(EXTRA)


if __name__ == "__main__":
    from canopyrs.installers import run_targets
    raise SystemExit(run_targets([NAME]))
