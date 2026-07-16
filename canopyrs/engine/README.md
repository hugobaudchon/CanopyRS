# Quick explanation of the engine

1. **Four tables hold everything.** `Sources` are the input rasters. `Tiles` are the model-ready
   frames cut from them (or seeded from a folder). `Crops` are per-object views, one crop per object.
   The three share one structure (`Imagery`) and each row can point to the image it was cut from
   (`parent_id`), so imagery forms a family tree across tables. `Objects` are the
   detections/segmentations: each row points to the image it was found in (`image_id`) and, when it
   was derived from an earlier object (an aggregated box keeping one detection), to that object
   (`prev_object_id`).

2. **Components declare what they need and what they make.** A component never names its predecessor.
   Instead it declares the *type* and shape of its input and output (a `Need`): which table type,
   which columns and links, whether geometry is in pixel or map coordinates (`crs`), and what imagery
   its objects live on (`on` — a classifier takes objects on crops, never on whole tiles). The
   pipeline checks these declarations before and after every component runs, so a mis-wired pipeline
   fails immediately with a clear message.

3. **Each component receives the newest table of the type it asks for.** Several tables can exist at
   once (the raster, the tiles, the crops). Because roles are types, there is nothing to guess: a
   tilerizer asking for `Sources` gets the raster, the aggregator asking for `Tiles` is never handed
   crops. The one table matching is then checked against everything else the component declared — a
   mismatch is an error, never a silent fallback to an older table.

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

  Sources: raster (has a file)                detection ---> carried to crop ---> classified
    └── Tiles: grid tile (window or file)         |
          └── Crops: crop (window or file)        +--image_id--> the imagery row it was found in
```

## The concepts

**Sources / Tiles / Crops** — the three imagery roles, one table type each: whole input rasters,
model-ready input frames, and per-object crops. Every row is georeferenced (`metadata`), points to the
image it was cut from (`parent_id` — the parent may live in another imagery table), and either exists
as a file on disk (`path`) or is read as a window from its parent image. The shared behavior (the
tree, reading, building) lives in their common base, `Imagery`.

**Objects** — a table of things found in images: one row per box or mask (`geom_kind`). Every object
points to the image it was found in (`image_id`) and, when it was derived from an earlier object (an
aggregated box keeping one detection, a classified crop), to that object (`prev_object_id`). This
chain keeps earlier values (like a `detector_score`) reachable later without copying them forward.

**Component** — one processing step: tilerizer, detector, segmenter, aggregator or classifier. It is
built from a config, takes tables as input, and returns new tables. It never knows which component ran
before it.

**Need** — how a component declares its input requirements: "Objects, in pixel coordinates, carrying a
`detector_score`, linked to their imagery, living on Crops". A component declares Needs for what it
consumes (`requires`) and for what it returns (`produces`).

**Schema** — the description a Need is checked against: which columns a table exposes, which links it
has, pixel or map coordinates, what imagery its objects live on, which modalities. At runtime, each
table produces its own Schema; at construction, declared Schemas are threaded through the whole
pipeline, so a mis-wired pipeline fails before any compute.

**Pipeline** — the runner. It takes as input seeds and ordered components, and then iteratively runs
each component sequentially. It is responsible for handing each component the newest table of the
type it asks for (checked against its Needs), checking what comes back against the promises, and
storing everything.

**Seeds** — the tables a run starts from, built from the inputs instead of produced by a component: a
raster (`sources=`), a folder of pre-cut images (`tiles=`), or prior detections from a GPKG
(`objects=`). A seeded folder is typed by what the components ask for — Tiles for a detector run,
Crops for a classifier-only run — and a classifier-only run over bare crops gets one derived Object
per crop ("1 image = 1 class"), so the user never labels anything by hand.

**Run record** — a file (`run.json`) written by the pipeline at the end of a run. It records the
component order, each config's hash, and what each step produced. It is what makes a run resumable
(`resume=True`), reloadable (`from_dir`) and exportable (`export("gpkg" | "coco")`).

## Which file owns which concept

| File | Owns |
|---|---|
| `constants.py` | column names (`Col`) and value vocab (`GeomKind`, `Modality`) |
| `data.py` | the tables (`Sources`/`Tiles`/`Crops`/`Objects`), FK validation, both ancestry walks (`column`/`linked`, `resolved_paths`) |
| `contracts.py` | `Need` / `one_of` / `Schema`: the declarations and how they're checked and matched |
| `pipeline.py` | running, the construction-time wiring check (`thread_schemas`), the seed rules, resume, reload (`from_dir`), export |
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
  0_tilerizer/tiles.parquet
  1_detector/objects.parquet
  ...
  final.gpkg          # latest georeferenced Objects, widened with their ancestry columns
```

`Pipeline.from_dir(out)` reloads it; `run(resume=True)` skips the already-done prefix;
`export("gpkg"|"coco", end_at=...)` writes results for any step.
