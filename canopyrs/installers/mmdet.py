"""Installer: mmdet stack for RSPrompter (mmcv compiled from source + the mmdet_rsprompter fork).

Validated 2026-07-16 against torch 2.7.1 / CUDA 12.6 / transformers 5.14:
  - mmcv 2.2.0 compiles cleanly from source (no prebuilt wheels exist for this torch);
  - the fork (github.com/hugobaudchon/mmdet_rsprompter) installs AS the `mmdet` package
    (vendored mmdet 3.2.0 + mmpretrain + the rsprompter models, transformers>=5 fixes) —
    see RSPROMPTER_NOTES.md;
  - full model build + training + inference forward pass verified on GPU.

The mmcv compile takes ~30 min. Two install paths:
  - a local fork checkout exists next to the CanopyRS root -> editable install (developer
    path; deliberately NOT via the [mmdet] extra, whose git URL would replace the editable
    install with a static site-packages copy and silently detach local edits);
  - no checkout -> the [mmdet] extra pulls the fork straight from GitHub.
"""

from canopyrs.installers.common import (cuda_build_env, cuda_build_preflight,
                                        importable, install_extra, pip, repo_root)

NAME = "mmdet"
ALIASES = ("rsprompter", "mmcv", "mmengine")
REQUIRES = ()
EXTRA = "mmdet"   # includes the fork as a git dep; mmcv itself is compiled below

# The extra minus the fork's git URL — used on the editable (developer) path.
_HALO_DEPS = ("mmengine", "importlib_metadata", "modelindex", "rich", "terminaltables",
              "peft>=0.10", "einops")


def _local_fork_checkout():
    root = repo_root()
    if root is None:
        return None
    for name in ("RSPrompter", "mmdet_rsprompter"):
        if (root / name / "setup.py").exists():
            return root / name
    return None


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
    checkout = _local_fork_checkout()
    if checkout is not None:
        print(f"[mmdet] local fork checkout at {checkout} — installing editable (developer path)")
        pip("install", *_HALO_DEPS)
        # setup.py imports torch (hence --no-build-isolation); --no-deps because its
        # requirements list is the full mmdet dev set (_HALO_DEPS covers the runtime).
        pip("install", "-e", str(checkout), "--no-deps", "--no-build-isolation")
    else:
        install_extra(EXTRA)   # pulls the fork from GitHub via the extra's git dep
    # mmcv from source against the env's torch — ~30 min compile; MMCV_WITH_OPS builds the
    # CUDA ops, FORCE_CUDA turns a missing-nvcc fallback into a loud error.
    cuda_build_preflight()
    pip("install", "--no-binary", "mmcv", "--no-build-isolation", "mmcv==2.2.0",
        env={**cuda_build_env(), "MMCV_WITH_OPS": "1"})


if __name__ == "__main__":
    from canopyrs.installers import run_targets
    raise SystemExit(run_targets([NAME]))
