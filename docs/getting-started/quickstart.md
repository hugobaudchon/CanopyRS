# Quickstart

Run tree detection on a single orthomosaic in a few steps.

## Using a preset configuration

CanopyRS ships with preset pipelines. The fastest way to get started is to use one of them directly via `infer.py`.

**Single raster/orthomosaic input (`-i`):**

```bash
python infer.py -c default_detection_multi_NQOS_best -i /path/to/raster.tif -o /path/to/output/
```

**Folder of already tiled geo-referenced images (`-t`):**

```bash
python infer.py -c default_detection_multi_NQOS_best -t /path/to/tiles/folder -o /path/to/output/
```

## Command-line arguments

| Argument | Description |
|---|---|
| `-c` | Config name (folder name under `canopyrs/config/`) |
| `-i` | Input path to a single raster/orthomosaic |
| `-t` | Input path to a folder of geo-referenced .tif tiles |
| `-o` | Output path |

## Understanding the output

The output folder will contain one subfolder per component that ran, containing output files such as:

- **GeoPackage (`.gpkg`)** — detected tree polygons with scores
- **COCO JSON** — annotations in COCO format (used internally between components)

The final aggregated GeoPackage is in the aggregator's output folder.

## Choosing the right preset

| Task | Config name | Description |
|---|---|---|
| Detection (Instance) | `default_detection_multi_NQOS_best` | Best model from our paper: DINO + Swin L-384 trained on multi-resolution datasets including SelvaBox. Best quality, ~10 GB GPU memory. |
| Detection (Instance) | `default_detection_single_S_medium` | Single resolution (6 cm/px) DINO + ResNet-50 model. Medium quality but faster and lower memory footprint. |
| Detection (Instance) | `default_detection_single_S_low` | Single resolution (10 cm/px) Faster R-CNN + ResNet-50 model. Faster with lower memory footprint. |
| Segmentation (Instance) | `default_segmentation_multi_NQOS_best` | Same as detection best, but with SAM2 chained after detection for instance segmentation. ~10 GB GPU memory. |
| Segmentation (Instance) | `default_segmentation_multi_NQOS_best_S` | **(Recommended)** Optimized for smaller trees (up to ~15m), using lower score threshold, smaller tiles, and higher GSD (4cm/px). |
| Segmentation (Instance) | `default_segmentation_multi_NQOS_best_L` | **(Recommended)** Optimized for larger trees (up to ~60m), using lower score threshold, larger tiles, and lower GSD (7cm/px). |

See [Presets](../user-guide/presets.md) for full details.
