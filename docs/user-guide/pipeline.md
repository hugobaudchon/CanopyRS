# Pipeline

The `Pipeline` runs components in order, threading typed data between them. It validates the wiring up front, saves every step, and can reload or resume a run.

## Data model

Instead of one mutable state object, the pipeline keeps three typed tables and hands the latest of each to whichever component needs it:

| Table | What it holds |
|---|---|
| `Sources` | the input files — a raster, a list, or per-modality/timestamp descriptors |
| `Tiles` | spatial footprints, each read from a Source (or a pre-cut tile on disk) |
| `Objects` | detected/segmented things (boxes or masks), born in a Tile |

Tables are never mutated or merged. Each component appends new tables, and relationships are foreign-key columns (`source_id`, `tile_id`, `prev_object_id`). Because objects keep a link to the object they came from, a late component can reach a value produced several steps back by walking the `prev_objects` ancestry — the aggregator, for example, pulls a `detector_score` that a classifier never carried forward.

## Contracts

Every component declares what it `requires` and what it `produces`: a data type, the columns and links it needs, and whether geometry is georeferenced or in tile-pixel coordinates. The pipeline checks each requirement against the available tables **before** running a component, and checks the output **after** — so a misconfigured pipeline fails at construction, not three components later.

## Flow chart

Constructing a pipeline prints a colored chart of the three tables showing, per component, which columns and links are available, produced, required, or missing, and whether geometry is in CRS or pixel coordinates. It's the fastest way to catch a wiring error before any inference runs.

## Running

```python
from canopyrs.engine.pipeline import Pipeline
from canopyrs.engine.config_parsers import PipelineConfig

config = PipelineConfig.from_yaml('canopyrs/config/pipelines/preset_det_single_S_fasterrcnn_r50.yaml')

pipe = Pipeline.from_config(config.components_configs, sources='raster.tif', output_dir='./out')
pipe.run()
```

Seed a run from a raster (`sources=`), a folder of pre-cut tiles (`tiles=`), or prior detections (`objects=`). With `output_dir` set, each component's tables are saved as parquet under `{id}_{name}/`, a `pipeline.json` manifest records the run, and the final georeferenced result is written to `out/final.gpkg`. Inspect results in memory with `pipe.latest(Objects)`.

## Reload, resume, export

| Call | What it does |
|---|---|
| `Pipeline.from_dir(run)` | reload a finished run's tables for inspection or re-export |
| `pipe.run(resume=True)` | skip the leading components already done (unchanged config + outputs on disk) and continue |
| `pipe.export("gpkg", ...)` / `pipe.export("coco", ...)` | write a GeoPackage or COCO for the Objects at a chosen step |

## Standalone components

To run a single component, build a one-step pipeline — see [Standalone Usage](standalone.md).
