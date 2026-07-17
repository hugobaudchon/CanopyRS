"""Setup target: mmdet stack for RSPrompter (mmcv compiled from source + the RSPrompter fork).

Validated 2026-07-16 against torch 2.7.1 / CUDA 12.6 / transformers 5.14:
  - mmcv 2.2.0 compiles cleanly from source (no prebuilt wheels exist for this torch);
  - the RSPrompter clone pip-installs as the `mmdet` package (its setup.py name is 'mmdet',
    vendored mmdet 3.2.0 + mmpretrain + the rsprompter models), carrying the fork's
    transformers>=5 compat fixes — see RSPROMPTER_NOTES.md;
  - full model build + forward pass verified on GPU.

The mmcv compile takes ~30 min. RSPrompter must be cloned next to the CanopyRS root
(the hugobaudchon/mmdet_rsprompter fork once pushed, or KyanChen/RSPrompter + fork patches).
"""

from canopyrs.installers.common import (SetupError, cuda_build_env, cuda_build_preflight,
                                           importable, install_extra, pip, require_repo_root)

NAME = "mmdet"
ALIASES = ("rsprompter", "mmcv", "mmengine")
REQUIRES = ()
EXTRA = "mmdet"   # pure-python halo; mmcv itself is compiled below (no wheels for our torch)


def _rsprompter_dir():
    root = require_repo_root(NAME)
    return root / "RSPrompter"


def check():
    missing = importable("mmengine", "mmdet", "mmcv")
    if missing:
        return False, f"mmdet stack not installed (missing {missing})"
    try:
        import mmcv._ext  # noqa: F401  — the mmcv equivalent of detrex's _C check
    except ImportError as e:
        return False, f"mmcv installed but its compiled extension is broken: {e}"
    try:
        from mmdet.registry import MODELS
        import mmdet.rsprompter  # noqa: F401
        if "RSPrompterAnchor" not in MODELS._module_dict:
            return False, "mmdet installed but RSPrompter models not registered (stock mmdet, not the fork?)"
    except ImportError as e:
        return False, f"mmdet present but RSPrompter models unavailable: {e}"
    return True, "mmcv (compiled) + mmdet fork with RSPrompter models OK"


def install():
    rsp = _rsprompter_dir()
    if not rsp.exists():
        raise SetupError(
            f"RSPrompter clone not found at {rsp}. It provides the mmdet package (vendored, "
            "with the RSPrompter models and transformers>=5 fixes).\nClone it there first — "
            "see RSPROMPTER_NOTES.md — then re-run `canopyrs setup mmdet`."
        )
    install_extra(EXTRA)
    # mmcv from source against the env's torch — ~30 min compile; MMCV_WITH_OPS builds the
    # CUDA ops, FORCE_CUDA turns a missing-nvcc fallback into a loud error.
    cuda_build_preflight()
    pip("install", "--no-binary", "mmcv", "--no-build-isolation", "mmcv==2.2.0",
        env={**cuda_build_env(), "MMCV_WITH_OPS": "1"})
    # The fork's setup.py imports torch (hence --no-build-isolation) and its requirements
    # list is the full mmdet dev set (hence --no-deps; _STACK_DEPS covers the runtime).
    pip("install", "-e", str(rsp), "--no-deps", "--no-build-isolation")


if __name__ == "__main__":
    from canopyrs.installers import run_targets
    raise SystemExit(run_targets([NAME]))
