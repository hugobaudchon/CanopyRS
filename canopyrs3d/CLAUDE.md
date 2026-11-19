# canopyRS3D

Multi-modal tree crown instance analysis at the **AS** sites (asnortheast, asnorthnorth, asnorthwestcent): comparing, and eventually jointly modelling, tree crowns seen from above by two independent modalities.

## Layout

```
canopyRS3D/                    outer working directory
└── CanopyRS/                  clone of hugobaudchon/CanopyRS, own git history
    ├── canopyrs/              upstream 2D pipeline package
    ├── detrex/                submodule (+ nested detectron2)
    └── canopyrs3d/            THIS PROJECT — everything below is ours
        ├── CLAUDE.md          this file
        ├── plan.md            evaluation methodology (the authority)
        ├── .gitignore         keeps data/ and output artefacts out of git
        ├── data/              local only, not in git
        ├── output/            regenerable artefacts, not in git
        └── scripts/
```

`canopyrs3d/` is deliberately a sibling of the upstream `canopyrs/` package, so that merging the two codebases later is a no-move operation. All paths in this file and in `plan.md` are relative to `canopyrs3d/`.

## Data

**The data is local only and deliberately not committed** — `hugobaudchon/CanopyRS` is a public repo, and these annotations are not published. `canopyrs3d/.gitignore` excludes `data/`. A fresh clone will not have it.

| Path | What it is | Role |
|---|---|---|
| `data/lidar_masks/<site>_plot_<p>_annotated_<split>_crowns_gt.gpkg` | **Visible** top-view crown polygons from hand-annotated airborne LiDAR, **12 plots**: asnortheast 203, 206, 207, 208 (val), 209, 210 (val), 256, 257 and asnorthnorth 153, 161, 170, 264 (val). Top 1 m of canopy, clipped by taller neighbours, never overlapping. Layer `crowns`. | **ground truth** |
| `data/lidar_masks_full/<site>_plot_<p>_annotated_<split>_crowns_full_gt.gpkg` | **Full** crown footprints of the same 12 plots: every tree point, no clip, so crowns overlap and overtopped trees are included. Not disjoint — do not point `eval_crown_overlap.py` at it. See `plan.md` §16. | reference only |
| `data/lidar_pred_masks/parcel_<p>_test_crowns.gpkg`, `data/lidar_pred_masks_full/parcel_<p>_test_crowns_full.gpkg` | Visible / full crowns from ForestMamba **instance predictions** (`instance_pred`, `--leaf-classes nonground`) on 35 unannotated parcels (136, 144–152, 154–160, 162–169, 171–174, 212, 213, 215, 216, 220, 263), from `ForestMamba/work_dirs/inference/norm_plots_AS_forestmamba_chm_r12_asnorth9_treepaste_ep150_georef/`. Already UTM, point-identical to `norm_plots_AS/parcel_<p>.laz`. | LiDAR predictions |
| `data/rgb_masks/20250318_asnortheast_50mm_p1_rgb_sf1p0_inferfinal.gpkg` | CanopyRS final output on the 1.1 cm RGB orthomosaic `../../rgb_ortho/20250318_asnortheast_50mm_p1_rgb.cog.tif` (0.045 m/px inference; run in `../../output_CanopyRS/`, which also holds the asnorthnorth and asnorthwestcent outputs). Identity column is `canopyrs_object_id`. | **predictions** |

All are EPSG:32617 (UTM 17N, metres).

**GT masks (regenerated 2026-11-18/19).** Source clouds: `ForestMamba/work_dirs/gt_georef/utm/*.ply`, except plot 257, taken directly from `ForestMamba/data/ForAINetV2/train_val_data/asnortheast_plot_257_annotated_train.ply` (already UTM — the `utm/` copy had its offset added twice and sits ~640 km off; do not use it). Plot 208 is the trainval re-annotation (51 trees), not the older `data/groundtruths` version. Extracted with `ForestMamba/tools/extract_crown_hulls.py --gt --leaf-classes nonground` (plot 257 labels its tree points class 1, not 2), defaults otherwise.

**Georeferencing, not RGB alignment.** `data/lidar_masks*/` are in the **original LiDAR frame**: each plot was checked against its annotated LAS in `../../../annotated_plots_AS/<p>/` (per-tree files or `parcel_<p>.laz`). Seven plots matched exactly; 203 (−0.4, −1.2 cm), 207 (+0.2, +2.1 cm), 209 (+0.3, 0 cm) and 210 (+0.1, −0.4 cm) were corrected by those translations, recorded per feature in `georef_dx_m` / `georef_dy_m`. **No RGB-ortho shift is applied** — the user chose not to align the masks to the imagery. `data/lidar_masks_unshifted/` and `data/lidar_masks_full_unshifted/` hold the same masks before the georef corrections. Do not run `scripts/apply_plot_shifts.py` on these layers: it would overwrite them with ortho-shifted copies.

Backups: `data/_previous_2026-11-18/` (the earlier 5-plot GT and, under `ortho_shifted_2026-11-18/`, the ortho-shifted 12-plot layers whose plot 257 is wrong); `data/_previous_2026-11-10/` (older GT and the 0.07 m/px predictions).

`output/` holds regenerable evaluation artefacts. Nothing in it is an input.

## `CanopyRS/` — vendored source

`CanopyRS/` is a full clone of <https://github.com/hugobaudchon/CanopyRS> (branch `main`), kept as its own git repository with history intact so the two codebases can be **merged together later**.

It is the source of the RGB predictions and the model to be fine-tuned. Treat the upstream directories (`canopyrs/`, `detrex/`, `docs/`, `tests/`, …) as upstream code: pull from `origin` rather than editing in place. New work goes in `canopyrs3d/`.

**Git note.** Work on a `canopyrs3d/*` branch, never commit to `main` — `main` tracks upstream. The repo is **public**, so anything committed here is published: keep data, imagery and annotations out (that is what `canopyrs3d/.gitignore` is for).

The `4 TB` volume is mounted root-owned, so git needs ownership exceptions. They are registered in the user's global config for the clone and both submodules:

```
git config --global --add safe.directory '<path>/CanopyRS'
git config --global --add safe.directory '<path>/CanopyRS/detrex'
git config --global --add safe.directory '<path>/CanopyRS/detrex/detectron2'
```

Install steps live in `../docs/getting-started/installation.md`.

## Where this is going

`canopyRS3D` is to be implemented **with CanopyRS** — the same algorithm that produced the RGB predictions here. The goal is to **fine-tune that model on multi-modal data, including LiDAR**.

The LiDAR-vs-RGB overlap evaluation is the diagnostic that scopes that work: it says where the RGB-only model breaks down and what LiDAR supervision would have to fix. It is not a one-off report card.

## The governing caveat — read before adding any metric

**The LiDAR annotation is deliberately incomplete.** Some trees were left out because they were too uncertain to annotate.

So an RGB mask with no LiDAR counterpart is **not** a false positive — most likely it is a real tree the annotator skipped. Consequences that any future work here must respect:

- Evaluation is **recall-oriented and ground-truth-anchored**. Never penalise the RGB side for finding trees the LiDAR annotation omitted.
- **No precision, F1, or panoptic PQ/RQ as headline metrics** — every false-positive term punishes unannotated trees.
- Unmatched RGB masks get counted and exported for review, never scored.

## Methodology

`plan.md` is the authority on evaluation methodology — many-to-many instance linking, coverage sufficiency, and the mIoU family. It also records probe results measured against the real data (co-registration, chaining behaviour, pixel-staircase bias, linking-criterion validation), so re-deriving them is unnecessary.

The implementation it specifies is a single script, `scripts/eval_crown_overlap.py`.

## Environment

Run with the existing `canopyrs` conda env:

```
/home/hugobaudchon/anaconda3/envs/canopyrs/bin/python
```

It has geopandas 1.0.1, shapely 2.0.1, rasterio, scipy, and `geodataset` 0.6.2.

## Related repos on this machine

Paths relative to `/media/hugobaudchon/4 TB/2026_AS_lidar/`.

| Repo | Relevance |
|---|---|
| `ForestMamba/` | 3D tree instance segmentation on airborne LiDAR; generated the GT crown polygons via `tools/extract_crown_hulls.py`. `tools/eval_predictions.py` holds the 3D instance metrics this project's 2D metrics are meant to stay comparable with. |
| `annotated_plots_AS/` | The hand-annotated `.las` plots the ground truth derives from. |
| `tree-xtractor/` | Per-instance biometrics from segmented point clouds; the cleanest packaging/testing template in this ecosystem. |
| `~/treesight/CanopyRS` | An older working clone of CanopyRS, ~120 commits behind the one vendored here. Prefer this one. |
| `geodataset` (pip, `hugobaudchon/geodataset`) | Shared geo utilities: tiling, aggregation/NMS, mask↔polygon conversion, file-name conventions. |
