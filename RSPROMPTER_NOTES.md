# RSPrompter — Setup, Training, and Integration Notes

Reconstructed from shell history and repo state (2026-07-16) since this wasn't otherwise documented.

## What RSPrompter is here

`RSPrompter/` (repo root, untracked in CanopyRS git) is a direct clone of the upstream repo:

```
git clone git@github.com:KyanChen/RSPrompter.git
```

Checked out on the `release` branch. It has its own nested `.git` — it's **not** a git submodule of CanopyRS, just a plain untracked directory sitting next to it. It is not committed anywhere, so a fresh clone of CanopyRS will not include it.

On top of upstream, local modifications exist in the working tree (uncommitted in the RSPrompter clone itself):

- Modified: `configs/rsprompter/_base_/rsprompter_anchor.py`, `rsprompter_anchor-whu.py`, `mmdet/rsprompter/models.py`, `mmdet/models/roi_heads/mask_heads/fcn_mask_head.py`, a couple of hook `__init__.py` files
- Added: `debug_augments.py`, `debug_hf_sam_weights.py`, `mmdet/rsprompter/selvamask_augmentations.py`, `mmdet/rsprompter/models_debug.py`, `mmdet/evaluation/metrics/coco_metric_debug.py`, `mmdet/engine/hooks/debug_prediction_hook.py`
- Many experiment config variants under `configs/rsprompter/` (seeds, `_NEW`, `_selvamask_*`, `_TEST_WHU`, etc.)

## Environment

Everything runs in a **single shared conda env: `canopyrs2`** — not a separate env from detectron2/detrex.

Build-up order:

1. `conda create -n canopyrs2 python=3.10`
2. `conda install -c conda-forge gdal=3.6.2`
3. `python -m pip install -e .` (CanopyRS itself)
4. `python -m pip install --no-build-isolation -e ./detrex/detectron2 -e ./detrex` — installs detectron2 + detrex
5. RSPrompter deps added on top of the same env:
   - `pip install -U transformers==4.38.1 wandb==0.16.3 einops pycocotools shapely scipy terminaltables importlib peft==0.8.2 mat4py==0.6.0 mpi4py`
   - `pip install -U openmim`
   - `mim install mmcv==2.1.0`, later bumped via `RSPrompter/install_mccv_mila_cluster.sh` to `mim install mmcv==2.2.0`
   - Sanity check: `python -c "import mmcv; import mmcv._ext"`

So `canopyrs2` contains detectron2, detrex, mmdet/mmcv/mmengine/mmpretrain, transformers, and peft all together. The older `canopyrs` env (no "2") is used for baseline detector/maskrcnn training (see `quick_test.sh`) and does not have detrex or RSPrompter deps.

## Data prep

- Main dataset: SelvaMask, pre-tilerized and tarred to `$SCRATCH/data/extracted_segmentation_dataset.tar`.
- SAM backbone pretrained weights downloaded via `RSPrompter/tools/rsprompter/download_hf_sam_pretrain_ckpt.py` into `~/scratch/models/rsprompter/sam_cache/` — both `sam_vit_base` and `sam_vit_huge` variants.
- A side experiment used the WHU aerial building dataset (`~/scratch/data/rsprompter_whu`), converted to COCO format with `RSPrompter/tools/rsprompter/whu2coco.py` (`rsprompter_anchor-selvamask_TEST_WHU.py` config) — a sanity-check track, not the main line.

## Training

Driver script: `experiments/segmenter/sbatch_train_rsprompter.sh`. It:

- Extracts the tarred SelvaMask dataset to node-local SLURM storage (`$SLURM_TMPDIR/data`)
- Exports `RSPROMPTER_DATA_ROOT` / `RSPROMPTER_NUM_GPUS` / `RSPROMPTER_NUM_WORKERS` env vars, read by the mmdet configs
- `cd`s into `RSPrompter/` and runs MMDetection's own `tools/train.py` directly (via `torchrun` for multi-GPU) — training uses RSPrompter's/MMDetection's native training loop, nothing from CanopyRS's own training pipeline
- Submitted repeatedly via `sbatch experiments/segmenter/sbatch_train_rsprompter.sh <config_path>`, iterating through config revisions: `rsprompter_anchor-selvamask.py` → `...-NEW.py` → `..._TEST_WHU.py` → final `...-NEW-666-2666_3` / `_4` families
- Also debugged locally on an interactive node via `CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh <config> 2`

### Final variants (used for evaluation)

Both are `RSPrompterAnchor`, 300 epochs, `EpochBasedTrainLoop`, CosineAnnealing LR after 50-iter linear warmup, checkpointed on best `coco/segm_mAP`. Checkpoints: `~/scratch/training/segmenter_rsprompter/<run_id>/best_coco_segm_mAP_epoch_*.pth`.

| Variant | Config base | SAM backbone | Seeds used |
|---|---|---|---|
| `rsprompter_anchor_v3` | `rsprompter_anchor-selvamask-NEW-666-2666_3` | ViT-Base | 1, 33 (seed 42 excluded — "had issues") |
| `rsprompter_anchor_v4` | `rsprompter_anchor-selvamask-NEW-666-2666_4` | ViT-Huge | 1, 33, 42 |

## Evaluation

`experiments/segmenter/evaluate/test_rsprompter.py <exp_name>`:

- Finds each seed's best checkpoint, wraps it as `SegmenterConfig(model='rsprompter_anchor', ...)`
- Tunes NMS IoU/score thresholds on the `SelvaMask` validation set
- Benchmarks on the `SelvaMask` test set via `SegmenterBenchmarker`
- Averages metrics across seeds (`compute_mean_std_metric_tables`)

## CanopyRS integration (inference only)

`canopyrs/engine/models/segmenter/rsprompter_infer.py` is a thin inference wrapper — training/architecture live entirely in the external RSPrompter clone, nothing is ported into detrex's model zoo. At runtime it:

1. Locates `RSPrompter/` next to the CanopyRS root
2. Adds it (and `RSPrompter/mmdet`) to `sys.path`, monkey-patches `sys.modules` so `mmdet.rsprompter` resolves to RSPrompter's custom package, and registers a few custom models into mmengine's global registry
3. Loads the MMDetection config + checkpoint directly from `RSPrompter/configs/rsprompter/...` and builds/runs the model using MMDetection's own `MODELS.build` / `load_checkpoint` / `model.predict` APIs

Because `RSPrompter/` is untracked, reproducing this setup elsewhere requires re-cloning it, re-applying the local modifications listed above, and re-downloading the SAM checkpoints.

## Update 2026-07-16 — unified env (canopyrs3, torch 2.7.1) validated

RSPrompter now runs in the same env as detectron2/detrex/SAM (no separate env needed):

- **mmcv 2.2.0 compiles from source** against torch 2.7.1 + CUDA 12.6 (`pip install --no-binary
  mmcv --no-build-isolation mmcv==2.2.0` with `MMCV_WITH_OPS=1 FORCE_CUDA=1`, ~30 min).
- **The clone pip-installs as the `mmdet` package**: its `setup.py` has `name='mmdet'` and
  `find_packages()` picks up both vendored `mmdet` (with `mmdet/rsprompter/`) and `mmpretrain`.
  `pip install -e ./RSPrompter --no-deps --no-build-isolation` replaces the old `sys.path`
  hacks. Runtime deps installed separately: mmengine, importlib_metadata, modelindex, rich,
  terminaltables, peft, einops.
- All of this is automated as **`canopyrs setup mmdet`** (canopyrs/installers/mmdet.py).
- Verified end-to-end: model build (117M params) + inference forward pass on GPU.

### Additional fork modifications (transformers>=5 compat), 2026-07-16

- `mmpretrain/models/__init__.py`: wrap `from .multimodal import *` in try/except — BLIP/BERT
  wrappers break against transformers 5 and RSPrompter never uses them.
- `mmdet/rsprompter/models.py`: `SamPromptEncoder(full_sam_config)` (v5 takes the full
  SamConfig; the `shared_patch_embedding` arg is gone).
- `mmdet/rsprompter/models.py`: both `self.mask_decoder(...)` calls — v5 dropped
  `output_attentions` and returns `(masks, iou)` (2-tuple, no attentions).

### Gotchas discovered

- The scratch purge deleted the SAM weight caches (`~/scratch/models/rsprompter/sam_cache/*`
  were empty dirs). Re-download: `SamModel.from_pretrained('facebook/sam-vit-base')` then
  `torch.save(m.state_dict(), .../pytorch_model.bin)` — transformers 5 ignores
  `safe_serialization=False`, so write the .bin via torch.save directly.
- The wall of mmengine "unexpected key" warnings at model load is normal (each submodule loads
  its slice of the full SamModel state dict).
- TODO: push this modified clone to a fork repo (e.g. hugobaudchon/mmdet_rsprompter) so
  `canopyrs setup mmdet` can clone it instead of requiring manual patch re-application.

### Fork-prep cleanup done 2026-07-16

- Deleted: debug output dirs (`debug_vis*/`, `debug_loss_vis*/`, ~31 MB of PNGs), empty
  `work_dirs/`/`.dist_test/`, dead code (`mmdet/rsprompter/models_debug.py` — stale copy of
  models.py, `debug_augments.py`, `debug_hf_sam_weights.py`); gitignored the debug dirs.
- `setup.py`: guarded the vestigial torch/cpp_extension import (`ext_modules` is empty) so pip
  can build the package in an isolated env — required for CanopyRS's `[rsprompter]` extra to
  reference the fork as a git dependency.
- Still present, intentionally: experiment configs beyond the final `_3`/`_4` families
  (`_TEST_WHU`, `_2`, `_5*/_6*/_7*`, `-NEW-666-1777`) — prune before/after forking as desired.
- Push checklist: commit everything on a branch named e.g. `canopyrs` (keep `release` pristine
  for upstream diffs), push to the fork, then activate the git dep in CanopyRS's
  `[rsprompter]` extra (pyproject.toml, currently commented out).
