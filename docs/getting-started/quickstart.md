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

The output folder contains one `{id}_{name}/` subfolder per component that ran, each holding that step's tables as GeoParquet, plus a `run.json` run record describing the run (so it can be reloaded or resumed).

If the pipeline produced georeferenced polygons, the final result is written to **`final.gpkg`** at the root of the output folder. You can also export any step's Objects as a **GeoPackage** or **COCO** file on demand via `pipeline.export(...)`.

## Choosing the right preset

See [Presets](../user-guide/presets.md) for full details.
