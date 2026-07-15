# Swap Pipeline v1 → v3 (promote v3 to the canonical engine)

## Context

The v3 relational pipeline (`canopyrs/engine/v3/`) is complete and is the keeper. v1
(`canopyrs/engine/pipeline.py` + `components/` + `data_state.py` + `persistence.py` + `resume.py` +
`pipeline_visualizer.py` + `constants.py`) and the abandoned v2 (`canopyrs/engine/v2/`) should go. This
plan rewires the three things still on v1 — the `infer.py` CLI, the benchmarker, and the test suite —
then promotes v3 to be *the* engine pipeline.

**Decisions (confirmed):** (1) **delete v1 & v2, promote v3 to canonical** (`engine/v3/*` → `engine/*`);
(2) **full resume + initialize_from parity** in the CLI; (3) **port AOI** to the v3 tilerizer.

**Key enablers already true:** v1's `PipelineConfig.components_configs` is already a `List[(kind, config)]`
— exactly v3's `from_config` steps — so the preset YAMLs (`canopyrs/config/pipelines/preset_*.yaml`) and
`PipelineConfig.from_yaml` carry over unchanged. The benchmark's evaluators are path-based
(COCO/GPKG) and pipeline-agnostic — untouched.

**The one genuinely new capability:** the benchmarker and `infer -t` seed from a **pre-cut tiles folder**
(georeferenced GeoTIFF tiles), not a raster. v3 must gain a way to seed a `Tiles` table from disk.

---

## Phase A — New v3 capabilities (additive, done in `engine/v3/` first, before the move)

### A1. Seed a `Tiles` table from a tiles folder — `Tiles.from_tiles_dir(path, bands=RGB)` (data.py)
Glob the tile rasters in `path`; for each, `rasterio.open` → build `TILE_METADATA` via
`tilemeta.window_meta(src, full_window)` (transform/crs/size), set `tile_path` to the file, `source_id`
absent (no sources link needed — the detector's `one_of(Need(Tiles, columns=(TILE_PATH,)))` branch and the
loader's `tile_path` read path already cover this). Tiles cut by geodataset are georeferenced GeoTIFFs, so
`tilemeta` recovers everything; the aggregator georeferences from `TILE_METADATA` as usual. (Verify the
tiles dir holds GeoTIFFs; if a geodataset tiles COCO is the only georef source, read it instead.)

### A2. Generalize pipeline seeding — `sources` and/or `tiles` (pipeline.py)
`Pipeline.__init__(components, sources=None, tiles=None, output_dir=None, aoi=None)`: build the seed
tables it's given — `Sources.from_paths(sources)` and/or `Tiles.from_tiles_dir(tiles)` (a path) or a
prebuilt `Tiles`. Store all seeds in a list; `run()` stores each; `thread_schemas()`/`validate()` seed
`{type: table.schema()}` from every seed (re-generalize the seed dict that was simplified to Sources-only).
`from_config(steps, sources=None, tiles=None, output_dir=None, aoi=None, resume_from=None,
initialize_from=None)`. When `tiles` is given and the first component is a `Tilerizer`, drop it (mirror
`infer.py`'s existing "tiles provided → no tilerizer" logic) — or let the detector consume the seeded Tiles.

### A3. AOI port (tilerizer.py + pipeline.py)
Run-level AOI (not a preset field), mirroring v1 where it came from `io_config`. `Pipeline(aoi=...)` builds a
geodataset `AOIConfig` via the existing `utils.parse_tilerizer_aoi_config(aoi_config, aoi_type, aois)` and
passes it to each `Tilerizer` at construction (`from_config` instantiates components, so it hands the AOI
to tilerizers there — config belongs in `__init__`, not injected attributes; `out_dir` stays the lone
pipeline-assigned attribute). v3 `Tilerizer.__init__(config, aois_config=None)` passes `aois_config` into
`RasterTilerizer` / `LabeledRasterTilerizer` / `RasterPolygonTilerizer` (they already accept it, per v1
`components/tilerizer.py:222,241,259`).

### A4. Objects seeding + initialize_from / resume_from parity (pipeline.py)
- **Objects seed** (also unblocks `input_gpkg`/`input_coco`): `Objects.from_gpkg(path)` /
  `Objects.from_coco(path, tiles)` factories so a run can start from prior detections/segmentations.
- **resume_from** (same folder): `run(resume=True)` (already exists). **Cross-folder** (`-o NEW`): copy the
  prior run dir into `NEW` (manifest + done component folders), then `run(resume=True)`.
- **initialize_from**: `Pipeline.from_dir(prior)` reconstructs the tables; seed the *new* pipeline with the
  prior run's latest `Tiles` + `Objects` (pass them as seeds in A2). New components restart at id 0.

**Verify A:** `py_compile`/`pyflakes`; extend `test_v3_detector_pipeline.py` with a `tiles=`-seeded run and
an AOI run; cluster smoke (GPU) for the tiles-folder + AOI paths.

---

## Phase B — Promote v3 to canonical engine (mechanical move + import rewrite + deletions)

### B1. Pre-move: cut the last v1 dependency
Inline `config_hash` (currently `persistence.py:42`, `model_dump(mode="json")` + sha) into `v3/store.py`
so v3 no longer imports `engine.persistence`.

### B2. Move `engine/v3/*` → `engine/*` (prefer `git mv` to keep history)
`pipeline.py, data.py, contracts.py, store.py, visualizer.py, loader.py, tilemeta.py, constants.py` →
`engine/`; `v3/components/` → `engine/components/`; fold `v3/__init__.py` re-exports into `engine/__init__.py`.

### B3. Rewrite imports
Global, scoped to the moved files + `test_v3_detector_pipeline.py`:
`canopyrs.engine.v3.<x>` → `canopyrs.engine.<x>`, and `from canopyrs.engine.v3 import store` →
`from canopyrs.engine import store`. (sed: `s/canopyrs\.engine\.v3\b/canopyrs.engine/g`, then audit.)

### B4. Delete v1 + v2
**Delete:** `engine/pipeline.py` (v1), `engine/components/` (v1 — replaced by the moved v3 dir; move v3 in
*after* deleting), `engine/pipeline_visualizer.py`, `engine/resume.py`, `engine/persistence.py`,
`engine/constants.py` (v1 — replaced by moved v3 constants), the whole `engine/v2/`, and the root
`test_v2_pipeline.py` / `test_v2_segmenter_pipeline.py`.
**Retain (the lone v1 remnant):** `engine/data_state.py` — still imported by
`data/classification/preprocessed_datasets.py` on the classifier path (Phase C TODO). `engine/utils.py`,
`engine/raster_validation.py`, `engine/models/`, `engine/config_parsers/` stay.

**Verify B:** `py_compile` the whole `canopyrs/engine` tree; `pyflakes`; `grep -rn "engine.v3\|engine.v2"`
returns nothing (outside intentionally-retained spots); `grep -rn "engine.pipeline_visualizer\|engine.resume\|engine.persistence"` returns nothing.

---

## Phase C — Rewire the v1-Pipeline callers to the promoted v3 API

### C1. `infer.py` (the CLI) — keep every flag
Map the existing arg handling to v3:
- `-i imagery -o out` → `Pipeline.from_config(config.components_configs, sources=imagery, output_dir=out, aoi=…)`.
- `-t tiles -o out` → `... tiles=tiles, ...` (the existing "drop leading tilerizer" guard stays).
- `-aoi` → builds the AOIConfig (A3).
- `--resume_from` / `--initialize_from` → A4 paths.
- `pipeline.run()` (v3 auto-writes `out/final.gpkg`). Drop the v1 `InferIOConfig` plumbing that no longer
  applies (`infer_gdf_columns_to_pass`, product-name-based output naming — v3 names by `{id}_{name}/` +
  `final.gpkg`). `input_gpkg`/`input_coco` → Objects seed (A4).

### C2. Benchmarker — re-point the single choke point `base_benchmarker._infer_single_product`
It funnels every benchmarker (`base_benchmarker.py:62-111`). Rewrite it to:
```
pipe = Pipeline.from_config(pipeline_config.components_configs, tiles=product_tiles_path,
                            output_dir=output_folder)            # + objects seed if input_gpkg/coco
pipe.run()
model_coco = pipe.export("coco", end_at=<index of component_name>)   # tile-level eval (tiles on disk -> OK)
agg_gpkg   = pipe.export("gpkg", end_at=<last aggregator index>)     # raster-level eval (CRS)
```
Replaces `pipeline.data_state.get_output_file(...)`. Find component indices by scanning `pipe.components`
for `name == component_name` / `"aggregator"` (the loop already exists). Maps cleanly **except** the v1
`pre_aggregated_gpkg` (return #2): v1 writes the detector/segmenter detections as a **tile-pixel** GPKG and
the optimizer re-imports it via `input_gpkg`. v3 won't (its `export("gpkg")` requires CRS) — but v3 has a
**more natural** path: the detector run already persists its Objects+Tiles as geoparquet, so #2 becomes
the model component's `output_dir` (or the run dir), not a gpkg.

`find_optimal_detector_aggregator.py` (aggregator-only grid search over cached detections) re-points to
v3's reload: run the model **once** → for each `(nms_threshold, score_threshold)`, `Pipeline.from_dir(run)`
→ seed a `[('aggregator', cfg)]` pipeline with the reloaded latest `Tiles` + `Objects` (in tile-pixel
coords, score columns intact — exactly the aggregator's `requires`) → `run()` → `export("gpkg")` → eval.
This drops v1's pixel-coords-in-GPKG round-trip entirely (cleaner). So pipeline seeding must accept
**in-memory** `Tiles`/`Objects` instances, not just paths (extend A2/A4).

**Care item:** make `export("coco")` benchmark-faithful — categories/scores must align with the truth COCO
so AP/AR stay comparable. v3 reuses the same geodataset `COCOGenerator.from_gdf` (`utils.generate_coco`)
v1 used, but confirm the `categories_column`/`coco_categories_list` mapping matches what the truth COCO and
`CocoEvaluator` expect (v3 currently defaults `coco_categories_list=None`). Evaluators
(`evaluator.py`, metric tests) are otherwise untouched.

### C3. `train_sam` COCO-eval (`train_sam2.py`, `train_sam3.py`, `dataset.py`)
Same shape as C2: `Pipeline.from_config(pipeline_config.components_configs, tiles=…, output_dir=…)` →
`run()` → `export("coco")` for the eval COCO.

### C4. Classifier benchmarker — **TODO, ask the user at implementation**
`benchmark/classifier/{benchmark.py,evaluator.py}` and `data/classification/preprocessed_datasets.py`
(the `DataState`-based per-object-tile preprocessing) need Objects/prompt seeding from prior
segmentations — design-dependent. For this PR: remove their v1 `Pipeline` imports, make
`ClassifierBenchmarker.benchmark()` raise `NotImplementedError("classifier benchmark not yet ported to
v3 — TODO")`, and keep `benchmark/__init__.py` importable (so Detector/Segmenter benchmarkers still load).
`data_state.py` is retained so `data/classification` keeps importing. **At implementation, ask the user
how the classifier benchmark should seed objects (from a segmentation COCO/GPKG, or re-run a segmenter).**

**Verify C:** `py_compile`/`pyflakes` on `infer.py`, `benchmark/`, `train_sam/`; `grep` shows no
`engine.pipeline`/`get_output_file`/`data_state` outside the retained classifier island; a cluster run of
`infer.py -i raster -o out` and `infer.py -t tiles -o out` (+ `-aoi`, `--resume_from`, `--initialize_from`);
a small detector benchmark on one product.

---

## Phase D — Migrate the test suite (v1 → v3)

Run config: `pytest -m "not slow"` (unit, no GPU) / `pytest` (incl. `@slow`). Real raster fixture
`assets/20240130_zf2tower_m3m_rgb_test_crop.tif` + synthetic raster/labels in `tests/conftest.py` stay.

- **Obsolete → delete:** `tests/engine/test_pipeline.py` (v1 gdf-merge rules — no v3 analog),
  `tests/engine/test_data_state.py` (v1 `StateKey`↔`DataState`), `tests/engine/components/test_base.py`
  (v1 `BaseComponent.validate`).
- **Keep as-is (pipeline-agnostic):** `tests/engine/benchmark/test_evaluator.py`,
  `test_benchmarker.py`, `test_alignment_unit.py`.
- **Rewrite to v3:** `tests/engine/test_pipeline_integration.py` → `Pipeline.from_config(steps, sources=…)`,
  assert `pipe.latest(Objects)` non-empty + `final.gpkg`/component COCO exist (keep `@slow`).
  `components/test_aggregator.py` + `test_classifier.py` → assert the v3 `requires`/`produces` `Need`s
  (e.g. aggregator's `requires` carries the weighted score columns; classifier's `one_of` shape).
- **New v3 unit tests (no GPU — the high-value additions):** under `tests/engine/`
  - `test_contracts.py`: `Need.check` / `one_of` / `Schema`; `validate()` rejects bad wiring (detector
    before tilerizer, aggregator missing a weighted score).
  - `test_data_model.py`: ancestry — `Objects.provides` / `column` / `linked` across a hand-built
    `prev_objects` chain (latest-wins); FK validation in `Table._link`.
  - `test_persistence.py`: `save_table`/`load_df` round-trip; `from_dir` re-links FKs (no dangling assert);
    `resume` skips an unchanged config prefix and reruns after a config change.
  - `test_export.py`: `export("gpkg")` widens ancestry columns + latest-wins with two aggregators;
    `export("coco")` raises when tiles aren't on disk.
  - Replace v1 `DataState`/gdf fixtures in `tests/conftest.py`/`tests/engine/conftest.py` with v3 builders
    (small `Sources`/`Tiles`/`Objects`); add a tiny on-disk tiles fixture (from the synthetic raster) for
    `Tiles.from_tiles_dir`.

---

## Critical files

- **Rewire:** `infer.py`; `canopyrs/engine/benchmark/base/{base_benchmarker.py,find_optimal_detector_aggregator.py}`;
  `canopyrs/engine/models/segmenter/train_sam/{train_sam2.py,train_sam3.py,dataset.py}`.
- **Promote/move:** all of `canopyrs/engine/v3/` → `canopyrs/engine/`.
- **Delete:** `canopyrs/engine/{pipeline.py,pipeline_visualizer.py,resume.py,persistence.py,constants.py}`,
  `canopyrs/engine/components/` (v1), `canopyrs/engine/v2/`, root `test_v2_*.py`.
- **Retain (classifier TODO):** `canopyrs/engine/data_state.py`, `canopyrs/data/classification/preprocessed_datasets.py`,
  `canopyrs/engine/benchmark/classifier/` (stubbed entrypoint).
- **Reuse:** `engine/utils.py` (`green_print`, `generate_coco`, `parse_tilerizer_aoi_config`,
  `merge_coco_jsons`, `init_spawn_method`), `engine/config_parsers/{pipeline.py,infer_io.py}` (unchanged),
  `engine/models/registry.py` (`*_REGISTRY`), the preset YAMLs.

## Verification (agent can't run project deps — syntax + grep + cluster)

1. `python3 -m py_compile` + `pyflakes` on the whole `canopyrs/engine` tree, `infer.py`, tests.
2. `grep -rn "engine\.v3\|engine\.v2\|engine\.pipeline_visualizer\|engine\.resume\|engine\.persistence\|get_output_file\|StateKey"` → only the retained classifier island remains.
3. `pytest -m "not slow"` (the new v3 unit tests + rewritten contract/component tests) — runnable without GPU.
4. Cluster (GPU): `infer.py -i raster -o out`; `infer.py -t tiles -o out`; `-aoi`; `--resume_from`;
   `--initialize_from`; one detector benchmark product; `pytest` (slow integration).
5. `git diff --stat` sanity: v1/v2 gone, v3 content now under `engine/`, callers re-pointed.

## Open question deferred to implementation
The **classifier benchmarker** (Phase C4): how it should seed objects in v3 (from a segmentation
COCO/GPKG vs. re-running a segmenter) — ask the user before building it.
