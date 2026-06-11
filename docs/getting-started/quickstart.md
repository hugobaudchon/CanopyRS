# Quickstart

Run tree detection on a single orthomosaic in a few steps.

!!! tip "Prefer a hosted environment?"

    Our [Google Colab tutorial](https://colab.research.google.com/drive/1UIJiLVpyeg8_-GUKDrqPKuT3RL8QMPdo) runs the full detect + segment pipeline on a sample orthomosaic on a free T4 GPU and visualizes the results — no local install needed. Its setup is adapted for Colab, and you can reuse its cells in your own notebook to run inference on your own data. Colab is slower and can't handle large orthomosaics, so it's best for testing or small datasets; for production use, install CanopyRS locally.

## Sample raster

A small test raster is included in the repository at `assets/20240130_zf2tower_m3m_rgb_test_crop.tif`. You can use it to try the commands below without needing your own data.

## Using a preset configuration

CanopyRS ships with preset pipelines. The fastest way to get started is to use one of them directly via `infer.py`.

**Single raster/orthomosaic input (`-i`):**

```bash
python infer.py -c <CONFIG_NAME> -i <PATH_TO_TIF> -o <PATH_TO_OUTPUT_FOLDER>
```

**Folder of already tiled geo-referenced images (`-t`):**

```bash
python infer.py -c <CONFIG_NAME> -t <PATH_TO_TILES_FOLDER> -o <PATH_TO_OUTPUT_FOLDER>
```

## Command-line arguments

| Argument | Description |
|---|---|
| `-c` | Config name (folder name under `canopyrs/config/`, see [Presets](../user-guide/presets.md) for a list of predefined configs.) |
| `-i` | Input path to a single raster/orthomosaic |
| `-t` | Input path to a folder of geo-referenced .tif tiles |
| `-o` | Output path |

## Understanding the output

The output folder will contain one subfolder per component that ran, containing output files such as:

- **GeoPackage (`.gpkg`)** — for example predicted tree polygons with scores
- **COCO JSON** — predictions in COCO format (used internally between components, can also be used to visualize per-tile predictions, see TODO)

If the chosen pipeline configuration produced a GeoDataFrame containing polygon results, it will be present at the root of your output folder.

## Choosing the right preset

See [Presets](../user-guide/presets.md) for full details.
