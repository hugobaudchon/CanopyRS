# Quickstart

Run tree detection on a single orthomosaic in a few steps.

## Using a preset configuration

CanopyRS ships with preset pipelines. The fastest way to get started is to use one of them directly via `infer.py`.

**Single raster input:**

```bash
python infer.py \                                                               # TODO fix the command
    --config default_detection_single_S_low \
    --input /path/to/orthomosaic.tif \
    --output /path/to/output/
```

**Folder of already tiled GeoTIFFs:**

```bash
python infer.py \                                                               # TODO fix the command
    --config default_detection_single_S_medium \
    --input /path/to/folder_of_tifs/ \
    --output /path/to/output/
```

## Understanding the output

The output folder will contain one subfolder per component that ran, containing output files such as:

- **GeoPackage (`.gpkg`)** — detected tree polygons with scores
- **COCO JSON** — annotations in COCO format (used internally between components)

The final aggregated GeoPackage is in the aggregator's output folder.

## Choosing the right preset

| Use case | Preset |
|---|---|
| Quick test, low memory | `default_detection_single_S_low` |
| Good balance of speed and quality | `default_detection_single_S_medium` |
| Best detection quality | `default_detection_multi_NQOS_best` |
| Detection + segmentation (small trees) | `default_segmentation_multi_NQOS_best_S` |
| Detection + segmentation (large trees) | `default_segmentation_multi_NQOS_best_L` |

See [Presets](../user-guide/presets.md) for full details.
