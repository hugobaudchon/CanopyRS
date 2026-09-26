# Plan: optional framework installs (`pip install CanopyRS[...]`)

Status: draft for v0.6.0-dev. Related: [issue #36](https://github.com/hugobaudchon/CanopyRS/issues/36)
(silent CPU-only detrex build on HPC).

## Goal

Split CanopyRS dependencies into a light **core** plus optional **framework layers** that models
plug into, so users install only the stacks they need:

```
core (always)          torch, torchvision, transformers, geodataset, GDAL/rasterio/geopandas, ...
│
├── [detectron2]       detectron2 + detrex (git submodule, compiled CUDA ops)
│     ├── dino (detrex)
│     ├── mask2former (detrex projects)
│     ├── maskdino (detrex projects — no need for the standalone IDEA repo)
│     ├── faster_rcnn / retinanet / mask_rcnn (detectron2)
│     └── detectree2
│
├── [mmdet]            mmengine + the mmdet_rsprompter fork (installs AS `mmdet`, other
│     └── rsprompter     mmdet models usable later); mmcv compiled by `canopyrs setup mmdet`
│
├── [rfdetr]           rfdetr (plain PyPI package from Roboflow)
│     └── rf-detr detector
│
├── [sam3]             transformers floor bump (Sam3Tracker*) — GATED HF model, needs
│     └── sam3         Meta/HuggingFace access approval + HF token; extra also serves as
│                      the documented opt-in for that approval flow
│
├── [sam]              segment-anything-py (legacy SAM1)
│     └── sam segmenter
│
└── sam2               stays in CORE (sam2==0.4.1) — ungated, the default segmenter of the
                       preset pipelines
```

The runtime side already supports this: `canopyrs.engine.models.auto_import_models()` catches
`ImportError` per model file, so wrappers for uninstalled frameworks simply don't register.
Heavy imports must stay lazy (module-level for the wrapper file is fine — it's the unit that
gets skipped; RSPrompter already defers everything to `__init__`).

## Why frameworks-as-extras and not per-model extras

- Models share the framework almost entirely (dino/mask2former/maskdino are all detrex configs;
  rsprompter_anchor/query are the same mmdet stack). Per-model extras would be aliases.
- The expensive/fragile part (CUDA extension builds, mmcv wheel matching) is per-framework,
  not per-model. Verification (see below) is also per-framework.

## pyproject sketch

```toml
[project.optional-dependencies]
# Pure-python companions only — the compiled parts need the scripted install below.
detectron2 = [
    "fvcore", "omegaconf>=2.1", "hydra-core>=1.1", "black",   # detectron2/detrex runtime deps
    "fairscale", "timm",                                        # detrex projects (maskdino, dino)
    "peft>=0.10",                                               # LoRA-style finetunes
]
mmdet = [
    # ONE extra for framework AND model: the fork IS the mmdet provider (its package name is
    # literally 'mmdet' — vendored framework + rsprompter models + transformers>=5 fixes), and
    # stock PyPI mmdet is unusable anyway (version guard rejects our mmcv). Its setup.py's
    # vestigial torch import is guarded so pip can build it in an isolated env.
    # PENDING: activates once the fork is pushed to GitHub.
    # "mmdet @ git+https://github.com/hugobaudchon/mmdet_rsprompter",
    "mmengine", "importlib_metadata", "modelindex", "rich", "terminaltables",
    "peft>=0.10", "einops",
    # mmcv itself CANNOT be listed here: no wheels for our torch pin — compiled from source by
    # `canopyrs setup mmdet` (validated against torch 2.7.1, ~30 min).
]
rfdetr = ["rfdetr>=1.0"]
sam3 = ["transformers>=4.57"]   # floor that ships Sam3TrackerModel/Sam3TrackerProcessor — verify exact version
sam = ["segment-anything-py==1.0.1"]   # legacy SAM1, demoted from core
all = ["CanopyRS[detectron2,mmdet,rfdetr,sam,sam3]"]   # union of the above (self-referential extra)
docs = [ ... unchanged ... ]
```

"Install everything" is then: `pip install -e .` (core → provides the `canopyrs` command)
+ `canopyrs setup detrex mmdet rfdetr sam sam3` + the SAM3 HF access request — with
`canopyrs doctor` as the final green-across-the-board check.

Setup is modular: one Python unit per dependency under `canopyrs/installers/`
(`detectron2.py`, `detrex.py`, `mmdet.py`, ...), each exposing `preflight()/install()/verify()`
and runnable standalone (`python -m canopyrs.installers.detrex`). Units declare dependencies
(`detrex.requires = ("detectron2",)`), and the CLI resolves the chain: `canopyrs setup detrex`
runs detectron2's unit then detrex's, verify-first so satisfied units are skipped. Units are
`.py`, NOT `.sh` — Windows has no bash, and Python is the one interpreter a Python package can
assume; shell commands become subprocess calls with explicit env handling, which also lets
chained units share state (a child shell can't export back to its parent).

Splitting detectron2 from detrex as targets is deliberate: detectron2-only users
(faster_rcnn/mask_rcnn/detectree2) install without compiling detrex's deformable-attn ops.
Targets use user vocabulary with aliases so the extra name and target name never fight
(model names like `dino`/`maskdino`/`rsprompter` resolve to their framework);
asking for something already in core (e.g. `canopyrs setup sam2`) reports "part of core —
already available" instead of erroring. Setup always invokes pip as `sys.executable -m pip`
(never bare `pip`), and detects pip-less (uv-managed) envs with a clear message. Setup is install-mode aware: editable
checkout → editable submodule build; PyPI wheel with no source tree → clear "needs a source
checkout" error.

Core keeps: torch==2.7.1, torchvision, transformers>=4.47, sam2==0.4.1, geodataset, geo
stack, faster-coco-eval, wandb, etc. Core alone must be able to run: tilerizing, aggregation,
the default SAM2 pipelines, benchmarks on precomputed results.

## What extras can't do (and the fix): scripted installs + verification

`pip install CanopyRS[detectron2]` can only pull metadata-clean wheels. The compiled pieces
need environment guarantees, so each framework gets an install script + a verifier.

### `[detectron2]` — the issue #36 problem

detectron2/detrex build CUDA ops at pip-install time from the git submodule. If `nvcc` or
`CUDA_HOME` is absent, **detrex's setup.py silently falls back to a CPU-only extension**;
the install "succeeds" and inference crashes at the first `ms_deform_attn_forward` call.

Install contract — implemented as **`canopyrs setup detrex`** (console entry point shipped
with core; replaces a standalone `scripts/install_detrex.sh`). It doesn't *check* that the
extra was installed — it installs it (pip via subprocess): every step is idempotent, so after
`pip install -e .` (which provides the `canopyrs` command) one command covers everything (the
bare extra stays useful for declarative setups — Docker, uv, requirements files):

0. **Self-heal prerequisites — state-aware, never destructive.** A blind
   `git submodule update --init --recursive` would silently rewind a developer's intentional
   detrex checkout to the pinned commit (and it's our own fork — we develop in it). Instead,
   branch on the `git submodule status --recursive` prefix per path:
   - `-` (uninitialized) → `git submodule update --init` for that path only (fresh clone case);
   - `+` (diverged from pin) → leave it, warn: "detrex checkout differs from pinned commit —
     building what's checked out; run `git submodule update` yourself if unintentional";
   - in sync → nothing; not a git repo → clear error (detrex is source-only, clone the repo).
   Then `pip install -e ".[detectron2]"` — a fast no-op when already satisfied, and it
   guarantees torch is importable before the `--no-build-isolation` build that needs it.
1. **Pre-flight (hard fail, don't warn):**
   ```bash
   nvcc --version          # must exist; major.minor should match torch.version.cuda
   python -c "import torch; print(torch.version.cuda)"
   test -n "$CUDA_HOME"
   ```
   Common conda fix when the cluster has no matching CUDA module (from issue #36):
   ```bash
   conda install -c nvidia/label/cuda-12.6.0 cuda-nvcc cuda-cudart-dev cuda-libraries-dev
   export CUDA_HOME=$CONDA_PREFIX
   ```
2. **Build** with explicit arch coverage so one build works across heterogeneous clusters:
   ```bash
   export FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
   pip install --no-build-isolation -e ./detrex/detectron2 -e ./detrex
   ```
   `FORCE_CUDA=1` flips the silent fallback into a loud build error when nvcc is missing —
   this is the single most important line for #36.
3. **Post-install verification** (also part of `canopyrs doctor`, below) — correct import path
   is top-level `detrex._C`, after torch; and when a GPU is visible, run the functional op test
   since `hasattr` passes on CPU-only builds too:
   ```bash
   python -c "
   import torch
   from detrex import _C
   assert hasattr(_C, 'ms_deform_attn_forward')
   print('detrex CUDA op present')"
   # + functional MultiScaleDeformableAttention forward on CUDA tensors if a GPU is visible
   ```

pip has no post-install hooks, so `pip install "CanopyRS[detectron2]"` alone can't run these.
Automation lives in the two layers we control:
- `canopyrs setup detrex` runs extra-install → pre-flight → FORCE_CUDA build → post-verify as
  one atomic step, exiting non-zero on any failure (usable in CI / sbatch setup jobs);
- a **construction-time `preflight()` hook** (below) catches installs that bypassed the script.

### `preflight()` — fail at pipeline construction, not 2h into a run

The deformable-attn op is only *called* when the detector/segmenter component runs — after
tilerizing, which is exactly the issue #36 pain. Wrappers may expose an optional classmethod
`preflight(config)`; `Detector`/`Segmenter` components call it where they already resolve the
model class from the registry (component `__init__` = `Pipeline.from_config` time). detrex
models implement it as: `detrex._C` importable + functional CUDA op forward when
`torch.cuda.is_available()`. Milliseconds of cost, and a clean actionable error ("detrex
compiled without GPU support — rebuild with `canopyrs setup detrex`") before any real work.
Two known failure modes it distinguishes:
- `_C` missing entirely → detrex already substitutes a dummy class raising "detrex is not
  compiled successfully" at instantiation (loud, but only at component run time — preflight
  moves it to construction);
- `_C` built CPU-only (the issue #36 case) → only the functional CUDA call catches it.

### `[mmdet]`

- **RESOLVED 2026-07-16:** mmcv 2.2.0 compiles from source against torch 2.7.1/cu126
  (`--no-binary mmcv --no-build-isolation` + `MMCV_WITH_OPS=1 FORCE_CUDA=1`, ~30 min), and the
  RSPrompter clone pip-installs as the `mmdet` package (its setup.py name IS 'mmdet', vendoring
  mmdet+mmpretrain+the rsprompter models). `canopyrs setup mmdet` automates the whole recipe;
  full model build + GPU forward pass verified in the unified env. No separate env needed.
- **RSPrompter packaging:** the current integration `sys.path`-hacks into an untracked clone of
  KyanChen/RSPrompter with uncommitted local modifications (see RSPROMPTER_NOTES.md). To make
  `[mmdet]` real, either:
  - **(preferred)** push the modified fork to `github.com/hugobaudchon/RSPrompter` and pin it as
    a git dep of the extra (or a submodule like detrex), or
  - vendor just `mmdet/rsprompter/` model code + the selvamask augmentations into
    `canopyrs/engine/models/segmenter/rsprompter/`.
- **Second open question:** RSPrompter was trained/run with transformers==4.38.1 + peft==0.8.2;
  core now requires transformers>=4.47. Its `models.py` imports `transformers.models.sam`
  internals — must be smoke-tested against the new pin before declaring compatibility.

### `[rfdetr]`

Plain PyPI package, no compiled extensions of our own. Only concern: it pins its own
torch/torchvision ranges — check they admit torch 2.7.1 at implementation time.

### `[sam3]`

The dependency side is trivial (a transformers floor that ships `Sam3TrackerModel` /
`Sam3TrackerProcessor`), but the extra earns its place for the **access gate**:
`facebook/sam3` is a gated HuggingFace model requiring Meta approval on the model page plus an
authenticated `HF_TOKEN`. The extra is the natural hook to document that flow, and
`canopyrs doctor` should distinguish the three failure modes: transformers too old /
not logged in to HF / access not granted. sam2 stays core so every default install has a
working segmenter without any approval step; legacy SAM1 moves to a tiny `[sam]` extra.

## Lazy imports & clean errors (no API breakage)

The pipeline must never crash because a framework a run doesn't use is missing, and a missing
framework a run *does* use must fail fast with "install this extra". The existing architecture
already provides the skeleton:

- wrapper modules keep their natural top-level imports; `auto_import_models()` catches
  `ImportError`, so uninstalled frameworks just don't register (pipelines that don't use them
  are untouched);
- `Segmenter`/`Detector` components resolve the model class from the registry at **pipeline
  construction** but only **instantiate** it in `run()` — so bad names fail before tilerizing.

Three small changes (~40 lines) complete it:

1. `auto_import_models()` records failures in `IMPORT_FAILURES: dict[module, Exception]`
   instead of `print`ing.
2. `Registry.get` on a lookup miss appends actionable hints, mapping the missing package to
   its extra via `PACKAGE_TO_EXTRA = {"mmdet"/"mmcv"/"mmengine": "mmdet", "detectron2"/"detrex":
   "detectron2", "rfdetr": "rfdetr", "segment_anything": "sam"}` and the recorded
   `ModuleNotFoundError.name` — no per-model manifest to maintain. Raise
   `MissingExtraError(ValueError)` so existing `except ValueError` callers keep working:

   ```
   'rsprompter_anchor' not in segmenter registry. Available: [...]
   Some models are unavailable because their extra isn't installed:
     - segmenter.rsprompter_infer: missing 'mmdet' → pip install "CanopyRS[mmdet]"
   ```

3. Wrappers whose framework imports are deferred into `__init__` (rsprompter — it registers
   fine without mmdet) guard with one line:
   `require_extra("mmdet", packages=("mmengine", "mmdet", "mmcv"))` — a tiny
   `importlib.util.find_spec` helper raising the same `MissingExtraError`.

**Aggregate reporting:** `Pipeline` construction collects `MissingExtraError`s across ALL
components before raising, so a pipeline missing detrex AND sam3 reports both in one message
ending with a single combined command (`canopyrs setup detrex sam3`) — no fix-one-rerun-hit-
the-next loop. Only `MissingExtraError` is collected; any other construction failure raises
immediately (broad collection would mask real bugs). `preflight()` failures use the same type,
so "not installed" and "installed but broken" land in the same combined report. ~10 lines in
the existing component loop — cheap precisely because everything funnels through one exception
type at one choke point.

Rejected as over-engineering: placeholder/proxy classes in the registry and `__getattr__`
lazy-import shims — they break `isinstance` and class-attribute access (`REQUIRES_BOX_PROMPT`)
that components read before instantiation. `IMPORT_FAILURES` also becomes the data source for
`canopyrs doctor` below.

## `canopyrs doctor` — one command to answer "what can I run?"

New small module (`canopyrs/diagnostics.py`, exposed as `python -m canopyrs.doctor` or a
console script). For each framework, report installed / missing / **broken**, and directly
address #36's "silent until 2h later" failure mode:

- torch: version, `torch.cuda.is_available()`, `torch.version.cuda`, visible GPUs
- detectron2/detrex: import `detrex._C` (note: `from detrex import _C`, after torch — the
  issue's `detrex.layers._C` path doesn't exist in our fork) and check `ms_deform_attn_forward`.
  **`hasattr` alone is NOT GPU-build proof** — CPU-only builds still export the symbol and only
  raise "Not compiled with GPU support" when called. When a GPU is visible, doctor must run a
  tiny functional forward through `MultiScaleDeformableAttention` on CUDA tensors (validated
  2026-07-16 on the canopyrs3 build); warn if the built arch list doesn't cover the local GPU.
- mmdet: `mmcv._ext` imports (the mmcv equivalent of the detrex check); mmengine/mmdet versions
- rfdetr / sam2: import + version
- registry summary: which detector/segmenter/classifier names actually registered
- exit non-zero if a *requested* framework (arg, e.g. `--expect detectron2,mmdet`) is broken —
  so sbatch scripts can fail fast **before** hours of tile preprocessing

Also swap `auto_import_models()`'s bare `print("Failed to import ...")` for a captured record
(module → exception) that doctor can display; keep stdout quiet on normal imports.

## Docs changes (installation guide)

- Restructure install page: core → pick your frameworks (extras table: extra / models unlocked /
  extra install step / verify command).
- Add the pre-flight + `FORCE_CUDA=1` + verify snippets from issue #36 (credit the reporter).
- State plainly: *build detrex/mmcv on a node with nvcc matching `torch.version.cuda`; a login
  node without nvcc will produce a broken install unless FORCE_CUDA=1 makes it fail loudly.*
- Close #36 once the doctor command + docs land.

## Migration steps

1. [ ] Add `[sam3]` extra (transformers floor + gated-access docs) and `[sam]` extra (demote
       segment-anything-py from core); keep sam2 in core; verify auto-import degradation keeps
       core installs clean when the sam3/sam wrappers can't load.
2. [ ] Add `[detectron2]`, `[rfdetr]` extras + `canopyrs setup` entry point (detrex target:
       pre-flight,
       `FORCE_CUDA=1`, post-verify.
3. [ ] Lazy-import error path: `IMPORT_FAILURES` record + `MissingExtraError` hints in
       `Registry.get` + `require_extra()` guard in rsprompter `__init__` + optional
       `preflight(config)` classmethod called by Detector/Segmenter component `__init__`
       (detrex impl: `_C` import + functional CUDA op test).
4. [ ] `canopyrs doctor` (reads `IMPORT_FAILURES`); wire `--expect` for sbatch fail-fast.
5. [ ] Decide RSPrompter packaging (fork-as-submodule vs vendoring) and validate mmcv against
       torch 2.7.1 → then `[rsprompter]` extra + setup target added (target renamed from mmdet; alias kept).
6. [ ] Port rf-detr wrapper (`DETECTOR_REGISTRY.register('rf_detr')`) once `[rfdetr]` exists.
7. [ ] CI job (weekly + on pyproject change): build `core` and `core+detectron2` on a CPU
       runner — CUDA *compilation* needs nvcc but no GPU, so FORCE_CUDA build + `_C` presence
       check run fine; only the functional forward is GPU-gated. Keeps extras from rotting.
8. [ ] Installation docs rewrite; close #36.

## Sanity checklist for "does it make sense?"

- ✔ detectron2/detrex and mmdet as base-framework extras: yes — models are thin configs/wrappers
  on top of them; the hard problems (builds, wheels) are per-framework.
- ✔ sam2 in core (and only sam2): yes — ungated, self-contained, gives every install a working
  default segmenter; legacy SAM1 is a `[sam]` extra for anyone still comparing against it.
- ✔ sam3 as extra: yes — gated HF model (needs Meta approval + HF token); the extra doubles as
  the documented opt-in for the approval flow.
- ⚠ pip extras alone can't deliver the compiled frameworks — every compiled extra ships with a
  scripted install + doctor verification; the extra covers the pure-python halo.
- ✔ `[mmdet]` (resolved): mmcv source-builds against torch 2.7.1 and the RSPrompter fork
  installs as the mmdet provider — `canopyrs setup mmdet` automates it. Remaining chore: push
  the modified RSPrompter clone to a fork repo so setup can clone it automatically.
