# Quick explanation of the engine

1. **Two tables hold everything.** `Imagery` is every image the pipeline knows about — the input
   rasters, the tiles cut from them, the per-object crops. Each row can point to the image it was cut
   from (`parent_id`), so imagery forms a family tree. `Objects` are the detections/segmentations:
   each row points to the image it was found in (`image_id`) and, when it was derived from an earlier
   object (an aggregated box keeping one detection), to that object (`prev_object_id`).

2. **Components declare what they need and what they make.** A component never names its predecessor.
   Instead it declares the *shape* of its input and output (a `Need`): which table type, which columns
   and links, whether geometry is in pixel or map coordinates (`crs`), and whether the imagery must be
   a whole scene or tiles (`kind`). The pipeline checks these declarations before and after every
   component runs, so a mis-wired pipeline fails immediately with a clear message.

3. **Each component receives the newest table of the right kind.** Several imagery tables can exist
   at once (the raster, then the tiles cut from it). The pipeline picks the newest one whose `kind`
   matches the component's declaration — so a component asking for a source raster still finds it
   after tiles were produced — and then checks everything else the component declared against that
   one table. A mismatch is an error; the pipeline never falls back to an older table. When a
   component is given anything other than the newest table, the pipeline prints a line saying so.

4. **Loading pixels is one rule.** An imagery row either has its own file on disk (`path`) or is a
   window into its parent. To load it, the loader walks up the parents to the nearest row that has a
   file and reads the window from that file. This is how tiles (and crops) can exist without ever
   being written to disk.

5. **Old values stay reachable — nothing is copied forward.** Tables are never edited or merged; every
   component appends new ones. When a late component needs a value from an earlier step (the
   aggregator needing `detector_score` after a classifier ran), it follows the `prev_object_id` chain
   back until it finds it (`Objects.column`).

## The two trees

```
Imagery (parent_id: what was I cut from?)   Objects (prev_object_id: what did I come from?)

  raster (kind=source, has a file)            detection ---> aggregated ---> classified
    └── grid tile (window or file)                |
          └── crop (window or file)               +--image_id--> the Imagery row it was found in
```

## The concepts

**Imagery** — a table of images. A row can be a whole input raster, a grid tile, or a crop around a
single object. Every row is georeferenced (`metadata`), points to the image it was cut from
(`parent_id`), and carries a `kind` ("source" scene or model-ready "tile"). A row either exists as a
file on disk (`path`) or is read as a window from its parent image.

**Objects** — a table of things found in images: one row per box or mask (`geom_kind`). Every object
points to the image it was found in (`image_id`) and, when it was derived from an earlier object (an
aggregated box keeping one detection, a classified crop), to that object (`prev_object_id`). This
chain keeps earlier values (like a `detector_score`) reachable later without copying them forward.

**Component** — one processing step: tilerizer, detector, segmenter, aggregator or classifier. It is
built from a config, takes tables as input (Imagery and/or Objects), and returns new tables. It never
knows which component ran before it.

**Need** — how a component declares its input requirements: "Objects, in pixel coordinates, carrying a
`detector_score`, linked to their imagery". A component declares Needs for what it consumes
(`requires`) and for what it returns (`produces`). `one_of(...)` lists acceptable alternatives, tried
in order.

**Schema** — the description a Need is checked against: which columns a table exposes, which links it
has, pixel or map coordinates, source or tile, which modalities. At runtime, each table produces its
own Schema; at construction, declared Schemas are threaded through the whole pipeline, so a mis-wired
pipeline fails before any compute.

**Pipeline** — the runner. It takes as input seeds and ordered components, and then iteratively runs
each component sequentially. It is responsible for handing each component the newest table of the
kind it asks for (checked against its Needs), checking what comes back against the promises, and
storing everything.

**Seeds** — the tables a run starts from, built from the inputs instead of produced by a component: a
raster (`sources=`), a folder of pre-cut tiles (`tiles=`), or prior detections from a GPKG (`objects=`).

**Run record** — a file (`run.json`) written by the pipeline at the end of a run. It records the
component order, each config's hash, and what each step produced. It is what makes a run resumable
(`resume=True`), reloadable (`from_dir`) and exportable (`export("gpkg" | "coco")`).

## Which file owns which concept

| File | Owns |
|---|---|
| `constants.py` | column names (`Col`) and value vocab (`ImageKind`, `GeomKind`, `Modality`) |
| `data.py` | the two tables, FK validation, both ancestry walks (`column`/`linked`, `resolved_paths`) |
| `contracts.py` | `Need` / `one_of` / `Schema`: the declarations and how they're checked and matched |
| `pipeline.py` | running, the construction-time wiring check (`thread_schemas`), resume, reload (`from_dir`), export |
| `store.py` | parquet persistence, the run record (`run.json`), seed persistence, GPKG/COCO writers |
| `loader.py` | torch Dataset/DataLoader loading images (own file, or window into `read_path`) |
| `tilemeta.py` | the serializable per-image georeferencing dict + pixel<->CRS transforms |
| `visualizer.py` | the construction-time flow chart (rendered from `thread_schemas`) |
| `components/` | the five components; each declares its `Need`s in `__init__` and implements `run()` |

## A run on disk

```
out/
  run.json            # the run record: component order, config hashes, what each produced
  _seed/              # the seed tables (so a reload can relink), index-prefixed
  0_tilerizer/imagery.parquet
  1_detector/objects.parquet
  ...
  final.gpkg          # latest georeferenced Objects, widened with their ancestry columns
```

`Pipeline.from_dir(out)` reloads it; `run(resume=True)` skips the already-done prefix;
`export("gpkg"|"coco", end_at=...)` writes results for any step.
