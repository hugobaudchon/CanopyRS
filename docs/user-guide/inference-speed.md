# Inference Speed

Inference time splits between **reading pixels** off your orthomosaic and **running the models** on them. If a run feels slower than expected, the four points below are where most of the time usually goes.

## 1. Store the orthomosaic as a tiled GeoTIFF (COG)

During model inference, CanopyRS never loads the whole orthomosaic: it reads one small window per tile, many times over. How your raster stores its pixels decides how much work each of those reads costs.

- **Tiled** (blocks of e.g. 256×256, what a COG uses): a window read touches only the blocks it overlaps.
- **Striped** (whole image rows, the default of several export tools): a window read has to decompress full-width rows, so reading a 1024×1024 tile can pull tens of megabytes you don't need.

Check what you have:

```bash
gdalinfo my_ortho.tif | grep -i "Block="
```

`Block=256x256` is tiled and you're fine. Something like `Block=20000x1` is striped — convert it once:

```bash
gdal_translate my_ortho.tif my_ortho_cog.tif -of COG -co COMPRESS=DEFLATE -co BIGTIFF=IF_SAFER
```

Note that the conversion can take a little while and take more disk space, but is usually worth it if you plan to use the orthomosaic multiple times.

## 2. Download remote orthomosaics to a local SSD first

A pipeline reads the same orthomosaic several times — each tilerizer step reads it, and every component that needs pixels (detector, segmenter, classifier) reads its windows again. Over HTTP or S3, each of those becomes a network round trip, and a striped raster (see above) multiplies the amount of data pulled.

So if your data lives in the cloud, **copy it locally before running**, ideally onto a local SSD/NVMe disk. Many small window reads are exactly the access pattern local disks are good at and remote storage is bad at.

!!! tip "On a compute cluster"

    Copy the orthomosaic to the node's local scratch disk (e.g. `$SLURM_TMPDIR`) at the start of your job, and point `-i` at that copy.

## 3. Use smaller models

The biggest lever on GPU time is which models you run. A ResNet-50 detector is several times faster than a Swin-L one, at some cost in quality — see [Presets](presets.md) for ready-made configurations from *Best* to *Fastest*, and [Model Zoo](model-zoo.md) to pick per-stage models for a custom pipeline.

## 4. Coarsen the tilerizer's ground resolution or overlap

Everything downstream scales with the **number of tiles**, and two tilerizer settings control it (see [Configuration](configuration.md)):

- **`ground_resolution`** (m/px) — the resolution the orthomosaic is resampled to before tiling. Tile count scales with its square: going from 4.5 cm to 7 cm cuts it by ~60%.
- **`tile_overlap`** (fraction of `tile_size`) — how much neighbouring tiles share. Going from 0.75 to 0.5 gives 4× fewer tiles.

Combined, that's the difference between the SAM 3 `_quality` and `_fast` presets: same 1777 px tiles, ~10× fewer of them.

Both come at a cost:

- A **coarser ground resolution** leaves small crowns with too few pixels to detect. Models are also trained within a range of resolutions, so straying far from the value a preset ships with may degrade quality.
- **Too little overlap** loses large crowns. The aggregator also drops any prediction touching a tile's edge band, so the usable part of a tile is only `tile_size × (1 - 2 × edge_band_buffer_percentage)` wide. Put together, a tree is guaranteed to land entirely inside some tile's usable area only if its diameter is at most:

    ```
    tile_size × (tile_overlap - 2 × edge_band_buffer_percentage)
    ```

    For the `_quality` preset — 1777 px, overlap 0.75, edge band 0.05 — that's 1777 × 0.65 ≈ 1155 px, or ~52 m at 4.5 cm/px. Anything bigger can be cut or edge-dropped in every tile it appears in, and get missed or split in two. Note that this also means `tile_overlap` must stay above `2 × edge_band_buffer_percentage`, or the usable areas stop covering the orthomosaic at all.

If you're unsure, start from the preset closest to your trees ([Presets](presets.md)) and change one value at a time.
