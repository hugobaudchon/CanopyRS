# Pipeline

The `Pipeline` runs components in order, threading typed data between them. It validates the wiring up
front, saves every step, and can reload or resume a run.

## Data model

Instead of one mutable state object, the pipeline keeps typed tables and hands each component the
newest instance of the type it asks for, checked against its declared needs:

| Table | What it holds |
|---|---|
| `Sources` | the input scenes (orthomosaics, rasters) — roots of the imagery tree |
| `Tiles` | model-ready input frames — grid tiles cut from a source, or a seeded folder |
| `Crops` | per-object views, one crop per object (made by the polygon tilerizer) |
| `Objects` | detected/segmented things (boxes or masks), each found in an image |

The three imagery roles share one structure (an `Imagery` base) and form one containment tree: a row
is either **materialized** (it has a file `path`) or a **window into its parent** (`parent_id` — a
tile in its raster, a crop in its tile). Reading is one rule: resolve a region to its nearest
materialized ancestor and read the region's window from that file.

Tables are never mutated or merged. Each component appends new tables, and because objects keep a link
to the object they came from (`prev_object_id`), a late component can reach a value produced several
steps back by walking the ancestry — the aggregator, for example, pulls a `detector_score` that a
classifier never carried forward.

## Contracts

Every component declares what it `requires` and what it `produces`: a data type, the columns and links
it needs, whether geometry is georeferenced or in tile-pixel coordinates, what imagery its objects
live on (`on=Crops` — so a classifier statically refuses objects sitting on whole tiles), and
optionally the modalities it supports. The pipeline checks each requirement **before** running a
component and checks the output **after** — a misconfigured pipeline fails at construction, not three
components later.

Each component receives the *newest* table of the **type** it asks for — a second tilerizer's
`Need(Sources)` finds the seed raster no matter how many tiles exist, and the aggregator's
`Need(Tiles)` is never shadowed by crops. Everything else must hold on that table: a mismatch is an
error, never a silent fallback to an older table.

## Flow chart

Constructing a pipeline prints a colored chart of the typed tables showing, per component, which
columns and links are available, produced, required, or missing, and whether geometry is in CRS or
pixel coordinates. It's the fastest way to catch a wiring error before any inference runs.

## Running

```python
from canopyrs.engine.pipeline import Pipeline
from canopyrs.engine.config_parsers import PipelineConfig

config = PipelineConfig.from_yaml('canopyrs/config/pipelines/preset_det_single_S_fasterrcnn_r50.yaml')

pipe = Pipeline.from_config(config.components_configs, sources='raster.tif', output_dir='./out')
pipe.run()
```

Seed a run from a raster (`sources=`), a folder of pre-cut images (`tiles=` — typed automatically:
tiles for a detector run, crops for a classifier-only run), or prior detections (`objects=`). With `output_dir` set, each component's tables are saved as parquet under `{id}_{name}/`,
a `run.json` **run record** (written by the pipeline, never hand-edited) describes the run, and the
final georeferenced result is written to `out/final.gpkg`. Inspect results in memory with
`pipe.latest(Objects)`.

## Reload, resume, export

| Call | What it does |
|---|---|
| `Pipeline.from_dir(run)` | reload a finished run's tables for inspection or re-export |
| `pipe.run(resume=True)` | skip the leading components already done (unchanged config + outputs on disk) and continue |
| `pipe.export("gpkg", ...)` / `pipe.export("coco", ...)` | write a GeoPackage or COCO for the Objects at a chosen step |

## Standalone components

To run a single component, build a one-step pipeline — see [Standalone Usage](standalone.md).
