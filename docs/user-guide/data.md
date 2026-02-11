# Data

In order to train or benchmark models, you will need data. In addition to SelvaBox, we provide several other pre-processed datasets.

## Available datasets

| Dataset | Size | Link |
|---|---|---|
| **SelvaBox** | ~31.5 GB | [HuggingFace](https://huggingface.co/datasets/CanopyRS/SelvaBox) |
| **SelvaMask** | ~3.3 GB | [HuggingFace](https://huggingface.co/datasets/CanopyRS/SelvaMask) |
| **Detectree2** | ~1.5 GB | [HuggingFace](https://huggingface.co/datasets/CanopyRS/Detectree2) |
| **NeonTreeEvaluation** | ~3.3 GB | [HuggingFace](https://huggingface.co/datasets/CanopyRS/NeonTreeEvaluation) |
| **OAM-TCD** | ~32.2 GB | [HuggingFace](https://huggingface.co/datasets/CanopyRS/OAM-TCD) |
| **BCI50ha** | ~27.0 GB | [HuggingFace](https://huggingface.co/datasets/CanopyRS/BCI50ha) |
| **QuebecTrees** | ~6.0 GB | [HuggingFace](https://huggingface.co/datasets/CanopyRS/QuebecTrees) |

## Downloading datasets

To download and extract datasets automatically for use with our benchmark or training scripts, we provide a download tool.

For example, to download SelvaBox and Detectree2 datasets:

=== "Linux / macOS"

    ```bash
    python -m canopyrs.tools.detection.download_datasets \
      -d SelvaBox Detectree2 \
      -o <DATA_ROOT>
    ```

=== "Windows (PowerShell)"

    ```powershell
    python -m canopyrs.tools.detection.download_datasets `
      -d SelvaBox Detectree2 `
      -o <DATA_ROOT>
    ```

After extraction, the datasets will be in COCO format (the same as geodataset's tilerizers output).

## Data structure

Your `<DATA_ROOT>` folder will contain one or more 'locations' folders, each containing individual 'rasters' folders. These contain .json COCO annotations and tiles for minimum one fold (train, valid, test...).

For SelvaBox and Detectree2 datasets, the structure looks like this:

```
<DATA_ROOT>
├── brazil_zf2                         (-> Brazil location of SelvaBox)
│   ├── 20240130_zf2quad_m3m_rgb       (-> one of the Brazil location rasters)
│   │   ├── tiles/
│   │   │  ├── valid/
│   │   │  │  ├── 20240130_zf2quad_m3m_rgb_tile_valid_1777_gr0p045_0_6216.tif
│   │   │  │  ├── ...
│   │   ├──  20240130_zf2quad_m3m_rgb_coco_gr0p045_valid.json
│   │   └──  ...
│   ├── 20240130_zf2tower_m3m_rgb
│   ├── 20240130_zf2transectew_m3m_rgb
│   └── 20240131_zf2campirana_m3m_rgb
├── ecuador_tiputini                   (-> Ecuador location of SelvaBox)
│   ├── ...
├── malaysia_detectree2                (-> Malaysia location of Detectree2)
│   ├── ...
└── panama_aguasalud                   (-> Panama location of SelvaBox)
```

Each additional dataset will add one or more locations folders.

## Example location folders by dataset

- **SelvaBox**: `brazil_zf2`, `ecuador_tiputini`, `panama_aguasalud`
- **Detectree2**: `malaysia_detectree2`
