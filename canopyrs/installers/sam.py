"""Setup target: legacy SAM1 (segment-anything-py). SAM2 is part of core — nothing to set up."""

from canopyrs.installers.common import importable, install_extra

NAME = "sam"
ALIASES = ("sam1", "segment-anything", "segment_anything")
REQUIRES = ()
EXTRA = "sam"


def check():
    if importable("segment_anything"):
        return False, "segment-anything-py not installed"
    return True, "segment-anything (SAM1) OK"


def install():
    install_extra(EXTRA)


if __name__ == "__main__":
    from canopyrs.installers import run_targets
    raise SystemExit(run_targets([NAME]))
