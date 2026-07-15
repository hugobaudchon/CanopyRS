# Quick explanation of the engine

1. **Two tables hold everything.** `Imagery` is every image the pipeline knows about — the input
   rasters, the tiles cut from them, the per-object crops. Each row can point to the image it was cut
   from (`parent_id`), so imagery forms a family tree. `Objects` are the detections/segmentations:
   each row points to the image it was found in (`image_id`) and, when it was derived from an earlier
   object (an aggregated box keeping one detection), to that object (`prev_object_id`).

2. **Components say what they need and what they make.** A component never names its predecessor.
   Instead it declares the *shape* of its input and output (a `Need`): which table type, which columns
   and links, whether geometry is in pixel or map coordinates (`crs`), and whether the imagery must be
   a whole scene or tiles (`kind`). The pipeline checks these declarations before and after every
   component runs, so a mis-wired pipeline fails immediately with a clear message.

3. **Each component receives the newest table that matches its needs.** Several imagery tables can
   exist at once (the raster, then the tiles cut from it). The pipeline walks them newest to oldest
   and hands over the first one satisfying the component's declaration — so a component asking for a
   source raster still finds it after tiles were produced. When a component is given anything other
   than the newest table, the pipeline prints a line saying so.

4. **Loading pixels is one rule.** An imagery row either has its own file on disk (`path`) or is just
   a window into its parent. To load it, walk up the parents to the nearest row that has a file and
   read the window from that file. This is how tiles (and crops) can exist without ever being written
   to disk.

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
