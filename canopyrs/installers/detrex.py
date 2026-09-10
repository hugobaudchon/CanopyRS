"""Setup target: detrex (compiled deformable-attention ops; depends on detectron2)."""

from canopyrs.installers.common import (SetupError, cuda_build_env, cuda_build_preflight,
                                           ensure_submodule, importable, pip, require_repo_root)

NAME = "detrex"
ALIASES = ("dino", "dino_detrex", "maskdino", "mask2former", "mask2former_detrex")
REQUIRES = ("detectron2",)
EXTRA = "detrex"


def check():
    if importable("detrex"):
        return False, "detrex not installed"
    try:
        # The full verification, shared with pipeline preflight: _C present, op exists, and a
        # functional forward on CUDA tensors when a GPU is visible (a CPU-only build passes
        # the import checks and only fails when the op is actually called — issue #36).
        from canopyrs.engine.models.preflight import detrex_ops_preflight
        detrex_ops_preflight()
    except ValueError as e:   # MissingExtraError
        return False, str(e).splitlines()[0]
    import torch
    gpu = " (GPU op verified)" if torch.cuda.is_available() else " (no GPU visible — op presence only)"
    return True, "detrex + compiled CUDA ops OK" + gpu


def install():
    root = require_repo_root(NAME)
    ensure_submodule(root, "detrex")
    cuda_build_preflight()
    pip("install", "--no-build-isolation", "-e", str(root / "detrex"), env=cuda_build_env())
    # invalidate the preflight cache so check() re-verifies the fresh build
    from canopyrs.engine.models import preflight
    preflight._DETREX_OPS_VERIFIED = False


if __name__ == "__main__":
    from canopyrs.installers import run_targets
    raise SystemExit(run_targets([NAME]))
