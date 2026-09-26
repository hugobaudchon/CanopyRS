"""Shared plumbing for `canopyrs setup` target units.

Each target is one module in this package (detectron2.py, detrex.py, ...) exposing:
  NAME: str                    canonical target name
  ALIASES: tuple[str, ...]     other names users may type (model names, extra names)
  REQUIRES: tuple[str, ...]    targets that must be set up first (dependency chain)
  check() -> (bool, str)       pure inspection: is it installed AND working? (never mutates)
  install() -> None            the mutating part; raises SetupError with the fix on failure

Units are plain Python (not shell scripts) so the same code runs on Linux and Windows, and so
chained units share process state — a child shell can't export env back to its parent.
Every unit is standalone-runnable: `python -m canopyrs.installers.detrex`.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


class SetupError(RuntimeError):
    """A setup step failed; the message must contain the fix."""


def run(cmd, *, env=None, check=True):
    """Run a command, streaming output. Returns the CompletedProcess."""
    print(f"$ {' '.join(str(c) for c in cmd)}")
    merged = {**os.environ, **(env or {})}
    result = subprocess.run([str(c) for c in cmd], env=merged)
    if check and result.returncode != 0:
        raise SetupError(f"Command failed (exit {result.returncode}): {' '.join(str(c) for c in cmd)}")
    return result


def pip(*args, env=None):
    """pip in THIS interpreter's environment — never a bare `pip` from PATH."""
    if importlib.util.find_spec("pip") is None:
        raise SetupError(
            "This environment has no pip (uv-managed?). Install the packages with your "
            "environment manager instead — see the extras in pyproject.toml."
        )
    return run([sys.executable, "-m", "pip", *args], env=env)


def repo_root() -> Path | None:
    """The CanopyRS source checkout root, or None for a site-packages (wheel) install."""
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").exists() and (root / "canopyrs").is_dir():
        return root
    return None


def require_repo_root(target: str) -> Path:
    root = repo_root()
    if root is None:
        raise SetupError(
            f"'{target}' builds from source and needs a CanopyRS git checkout — this install "
            "lives in site-packages.\nClone the repo and install editable first:\n"
            "  git clone https://github.com/hugobaudchon/CanopyRS.git && cd CanopyRS\n"
            "  pip install -e . && canopyrs setup " + target
        )
    return root


def install_extra(extra: str):
    """Install this project's pip extra — editable from a checkout, from PyPI otherwise."""
    root = repo_root()
    if root is not None:
        pip("install", "-e", f"{root}[{extra}]")
    else:
        pip("install", f"CanopyRS[{extra}]")


def importable(*packages: str) -> list:
    """The subset of ``packages`` that is NOT importable."""
    return [p for p in packages if importlib.util.find_spec(p) is None]


def ensure_submodule(root: Path, path: str):
    """State-aware submodule handling — never destructive.

    Only initializes a missing submodule; an intentionally diverged checkout is left alone
    (we develop in our detrex fork — blindly running `git submodule update` would silently
    rewind it to the pinned commit).
    """
    if not (root / ".git").exists():
        raise SetupError(
            f"{root} is not a git checkout, so the '{path}' submodule can't be initialized. "
            "detrex/detectron2 are source-only: clone the repository."
        )
    status = subprocess.run(["git", "-C", str(root), "submodule", "status", "--recursive"],
                            capture_output=True, text=True).stdout
    for line in status.splitlines():
        if not line.strip():
            continue
        prefix, rest = line[0], line[1:].split()
        sub_path = rest[1]
        if not (sub_path == path or sub_path.startswith(path + "/")):
            continue
        if prefix == "-":
            run(["git", "-C", str(root), "submodule", "update", "--init", sub_path])
        elif prefix == "+":
            print(f"NOTE: submodule '{sub_path}' checkout differs from the pinned commit — "
                  "building what's checked out. Run `git submodule update` yourself if that "
                  "isn't intentional.")


def cuda_build_preflight():
    """Hard requirements for compiling CUDA extensions; fail loudly with the fix (issue #36)."""
    import torch

    problems = []
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        problems.append("`nvcc` not found on PATH.")
    else:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
        torch_cuda = torch.version.cuda or "none (CPU-only torch!)"
        if torch.version.cuda and f"release {'.'.join(torch.version.cuda.split('.')[:2])}" not in out:
            problems.append(
                f"nvcc version does not match torch's CUDA ({torch_cuda}). Compiled ops may "
                "be broken — install a matching CUDA toolkit."
            )
    if not os.environ.get("CUDA_HOME") and nvcc is not None:
        # derivable from nvcc location; set it for the build
        os.environ["CUDA_HOME"] = str(Path(nvcc).resolve().parents[1])
        print(f"CUDA_HOME not set — derived {os.environ['CUDA_HOME']} from nvcc location.")
    elif not os.environ.get("CUDA_HOME"):
        problems.append("CUDA_HOME is not set.")

    if problems:
        raise SetupError(
            "Cannot compile CUDA extensions:\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\n\nOn a cluster without a matching CUDA module, install the toolkit into the "
            "conda env (then re-run):\n"
            "  conda install -c nvidia/label/cuda-12.6.0 cuda-nvcc cuda-cudart-dev cuda-libraries-dev\n"
            "  export CUDA_HOME=$CONDA_PREFIX"
        )


def cuda_build_env() -> dict:
    """Env for FORCE_CUDA builds. FORCE_CUDA=1 turns the silent CPU-only fallback into a loud
    build error; the arch list covers common cluster GPUs so one build runs everywhere."""
    return {
        "FORCE_CUDA": "1",
        "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST", "7.5;8.0;8.6;8.9;9.0"),
        "MAX_JOBS": os.environ.get("MAX_JOBS", "8"),
    }
