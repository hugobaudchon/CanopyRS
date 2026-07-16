# Components

Each component handles a single stage. A component declares the typed data it **requires** and what it
**produces**; the pipeline resolves the right tables in and checks them before and after the component
runs (see [Pipeline](pipeline.md)).

All components can also be run on their own — see [Standalone Usage](standalone.md).

## Tilerizer

Splits a source scene into tiles.

**Tile types** (`tile_type`):

| Type | Description |
|---|---|
| `tile` | regular-grid tiles (input to a detector or segmenter) |
| `labeled` | grid tiles with input objects re-tiled onto them |
| `polygon` | one crop per input object (input to a classifier) |

**Requires:** `Imagery` (kind=source) for `tile` and `labeled` (plus the input `Objects` for
`labeled`). `polygon` needs only `Objects` linked to their imagery — each crop is cut from the
object's own image file, whether that is a source raster or an on-disk tile.

**Produces:** `Imagery` (kind=tile, children of the image they were cut from) — plus the carried
`Objects` for `labeled` and `polygon`

---

## Detector

Runs object detection on tiles, one box per detection.

**Requires:** `Imagery` (kind=tile)

**Produces:** `Objects` (boxes, tile-pixel coords) with `detector_score`, `detector_class`

---

## Segmenter

Produces instance masks — prompted by input objects (e.g. SAM) or automatically over each tile.

**Requires:** `Objects` (prompted) or `Imagery` kind=tile (automatic)

**Produces:** `Objects` (masks, tile-pixel coords) with `segmenter_score`

---

## Aggregator

Merges overlapping detections across tiles with non-maximum suppression (NMS), georeferencing them to
raster coordinates.

**Requires:** `Objects` carrying the weighted score column(s) and with their `imagery` linked

**Produces:** georeferenced `Objects` with `aggregator_score`

---

## Classifier

Classifies each object.

**Requires:** per-object crop `Objects` (preferred) or `Imagery` kind=tile

**Produces:** `Objects` with `classifier_class`, `classifier_score`, `classifier_scores` (and
`classifier_class_name` if `class_names` is set)

---

## Validation

A component's inputs are checked against its declared `requires` before it runs, and its output against
its `produces` after — so a wiring error surfaces early, at the offending component, with a clear message.
