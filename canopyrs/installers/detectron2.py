"""Setup target: detectron2 (compiled, from the detrex/detectron2 nested submodule)."""

from canopyrs.installers.common import (cuda_build_env, cuda_build_preflight, ensure_submodule,
                                           importable, install_extra, pip, require_repo_root)

NAME = "detectron2"
ALIASES = ("d2", "faster_rcnn", "mask_rcnn", "retinanet", "detectree2")
REQUIRES = ()
EXTRA = "detectron2"


def check():
    if importable("torch"):
        return False, "torch not installed (install CanopyRS core first)"
    if importable("detectron2"):
        return False, "detectron2 not installed"
    try:
        import torch  # noqa: F401  (must load before the extension)
        from detectron2 import _C  # noqa: F401
    except ImportError as e:
        return False, f"detectron2 installed but its compiled extension is broken: {e}"
    return True, "detectron2 + compiled extension OK"


def install():
    root = require_repo_root(NAME)
    ensure_submodule(root, "detrex")           # detectron2 sources live inside the detrex submodule
    install_extra(EXTRA)
    cuda_build_preflight()
    pip("install", "--no-build-isolation", "-e", str(root / "detrex" / "detectron2"),
        env=cuda_build_env())


if __name__ == "__main__":
    from canopyrs.installers import run_targets
    raise SystemExit(run_targets([NAME]))
