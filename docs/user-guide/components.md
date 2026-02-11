# Components

Each component in the pipeline is responsible for a single stage of processing. All components share the same interface: they receive a `DataState`, do their work, and return a `ComponentResult`.

All components can also be run individually — see [Standalone Usage](standalone.md).

## Tilerizer

Splits a large orthomosaic into smaller, overlapping tiles suitable for model inference.

**Tile types:**

| Type | Description |
|---|---|
| `tile` | Unlabeled regular-grid tiles (input to detector or segmenter) |
| `tile_labeled` | Labeled tiles with COCO annotations (input to prompted segmenter) |
| `polygon` | Per-polygon tiles (input to classifier) |

You can select the type of tilerizer you want using the `tile_type` parameter in your tilerizer config.

**Requires:** `imagery_path` (+ `infer_gdf` for `tile_labeled` and `polygon`)

**Produces:** `tiles_path`, optionally `infer_coco_path`

---

## Detector

Runs object detection on image tiles, producing bounding box predictions with confidence scores.

**Requires:** `tiles_path`

**Produces:** `infer_gdf` (geometry, object_id, tile_path, score, class), `infer_coco_path`

---

## Segmenter

Refines bounding box detections into instance segmentation masks. Can operate in prompted mode (using detector boxes) or unprompted mode.

**Requires:** `tiles_path`, optionally `infer_coco_path` (prompted mode)

**Produces:** `infer_gdf` with updated mask geometries, `infer_coco_path`

---

## Aggregator

Merges overlapping detections from tiled inference using non-maximum suppression (NMS). Produces the final per-tree polygons.

**Requires:** `infer_gdf` with geometry, object_id, tile_path, and detector_score and/or segmenter_score columns

**Produces:** aggregated `infer_gdf` with `aggregator_score`

---

## Classifier

Classifies each detected/segmented tree into categories (e.g. species).

**Requires:** `tiles_path`, `infer_coco_path`

**Produces:** classification scores and predictions in `infer_gdf`

---

## Runtime validation

Every component is decorated with `@validate_requirements`. At runtime, before the component logic executes, the decorator checks that all required state keys and GDF columns are present — and raises a clear error with hints if anything is missing.
