# canopyrs1

The geometry, raster, AOI, tiling, aggregation and COCO code comes from
[geodataset](https://github.com/hugobaudchon/geodataset), published together with CanopyRS.

## Layout

From the lowest layer to the highest. Two rules:
- `core/` never imports torch, a model, or anything outside `core/`.
- Otherwise, a module only imports modules listed above it (checked by a test).

| Module | What it holds |
|---|---|
| **`core/`** | **The building blocks, usable without the pipeline or any model** |
| `core/constants.py` | Column names (`Col`) and the values some columns take (`GeomKind`, `Modality`) |
| `core/naming.py` | File naming conventions. They are a public contract: names must not change |
| `core/geometry/` | Georeferencing, CRS, shape cleanup, masks, COCO segmentation. No files |
| `core/raster/` | Opening rasters without reading pixels, reading and writing windows |
| `core/tables/` | The data model (images and objects) and the contracts between pipeline steps |
| `core/io/` | Files → tables, tables → files |
| `core/aoi/` | Areas of interest, and splitting tiles into folds |
| `core/tiling/` | Tiles, crops, labels on tiles, writing tiles to disk |
| `core/aggregation/` | Merging predictions from overlapping tiles |
| `core/dataset_creation/` | Creating training datasets |
| `config_definitions/` | The settings of each pipeline step |
| `config_presets/` | Ready-made YAML settings |
| `models/` | Models for inference, and the loader that feeds them images |
| `pipeline/` | Running the steps, saving, resuming and exporting a run |
| `public_datasets/` | The public datasets CanopyRS uses |
| `benchmark/` | Evaluating predictions |
| `training/` | Training, one trainer per framework |
| `installers/`, `tools/`, `doctor.py`, `cli.py` | Command-line entry points |

## Words used in the code

| Word | Meaning |
|---|---|
| **image** | One row of `Sources`, `Tiles` or `Crops` |
| **on disk** | The image has its own file (`path`) |
| **window** | An image with no file of its own; its pixels are read from a parent's file |
| **object** | One row of `Objects`: a box, a mask or a point |
| **link** | A pointer from one table to another (`imagery`, `parent`, `prev_objects`) |
| **history** | The chain of objects an object was derived from (`prev_object_id`) |
