import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from geodataset.aoi import AOIFromPackageConfig
from rasterio.enums import ColorInterp
from tqdm import tqdm

from canopyrs.data.detection.raw_datasets.base_dataset import BasePublicZipDataset
from canopyrs.data.detection.tilerize import tilerize_with_overlap


parent_folder = Path(__file__).parent


class Detectree2Dataset(BasePublicZipDataset):
    zip_url = "https://zenodo.org/api/records/8136161/files-archive"
    name = "malaysia_detectree2"
    annotation_type = "mask"
    aois = {
        "Dan_2014_RGB_project_to_CHM": {
            "test": parent_folder / "aois" / "detectree2_aoi_test.gpkg",
            "valid": parent_folder / "aois" / "detectree2_aoi_valid.gpkg"
        },
        "Sep_MA14_21_orthomosaic_20141023_reprojected_full_res": {
            "test": parent_folder / "aois" / "detectree2_aoi_test.gpkg",
            "valid": parent_folder / "aois" / "detectree2_aoi_valid.gpkg"
        }
    }

    categories = None

    def _parse(self, path: str or Path):
        path = Path(path)

        assert path.exists(), f"Path {path} does not exist. Make sure you called the download method first."

        # Remove files
        shutil.rmtree(path / 'dan_dfs')
        shutil.rmtree(path / 'sep_east')
        shutil.rmtree(path / 'sep_west')

        # Read and write GeoDataFrames
        sepilok_east = gpd.read_file(path / 'SepilokEast.gpkg')
        sepilok_west = gpd.read_file(path / 'SepilokWest.gpkg')
        sepilok_west = sepilok_west.to_crs(sepilok_east.crs)

        # merge annotations for Sepilok
        sepilok = gpd.overlay(sepilok_east, sepilok_west, how='union')
        sepilok.to_file(str(path / 'Sep_MA14_21_orthomosaic_20141023_reprojected_full_res_labels.gpkg'), driver='GPKG')

        (path / 'Danum.gpkg').rename(path / 'Dan_2014_RGB_project_to_CHM_labels.gpkg')

        (path / 'SepilokEast.gpkg').unlink()
        (path / 'SepilokWest.gpkg').unlink()

        # fix color headers and data type from float32 to uint8
        for raster_name in self.aois:
            p = path / f"{raster_name}.tif"
            if p.exists():
                float32_to_uint8_inplace(p)

        print('Detectree2 dataset has been successfully parsed.')

    def tilerize(self,
                 raw_path: str or Path,
                 output_path: str or Path,
                 folds: set[str],
                 ground_resolution: float = None,
                 scale_factor: float = None,
                 tile_size: int = 1024,
                 tile_overlap: float = 0.5,
                 **kwargs):

        raw_path = Path(raw_path)
        output_path = Path(output_path) / self.name
        output_path.mkdir(parents=True, exist_ok=True)

        if not raw_path.name == self.name:
            raw_path = raw_path / self.name

        assert raw_path.exists(), (f"Path {raw_path} does not exist."
                                   f" Make sure you called the download AND parse methods first.")

        aois = self.aois.copy()
        for raster_name in aois:
            # prepare aois
            for raster_aoi in aois[raster_name]:
                aois[raster_name][raster_aoi] = gpd.read_file(f'{aois[raster_name][raster_aoi]}')
                if raster_aoi not in folds:
                    del aois[raster_name][raster_aoi]

            # tilerize
            aois_config = AOIFromPackageConfig(aois[raster_name])

            tilerize_with_overlap(
                raster_path=raw_path / f"{raster_name}.tif",
                labels=raw_path / f"{raster_name}_labels.gpkg",
                main_label_category_column_name=None,
                coco_categories_list=None,
                aois_config=aois_config,
                output_path=output_path,
                ground_resolution=ground_resolution,
                scale_factor=scale_factor,
                tile_size=tile_size,
                tile_overlap=tile_overlap
            )


def float32_to_uint8_inplace(path: Path):
    with rasterio.open(path) as src:
        if src.dtypes[0] != "float32":
            return

        profile = src.profile.copy()
        profile.update(
            dtype="uint8",
            nodata=0,
            PHOTOMETRIC="RGB",
            ALPHA="YES",
            compress="lzw"
        )

        tmp = path.with_suffix(".uint8.tif")
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.colorinterp = (  # band roles written once
                ColorInterp.red,
                ColorInterp.green,
                ColorInterp.blue,
                ColorInterp.alpha,
            )

            for _, win in tqdm(src.block_windows(), desc="Converting to uint8"):
                block = src.read(window=win)
                block = np.clip(block / 256, 0, 255).astype(np.uint8)
                dst.write(block, window=win)

        tmp.replace(path)
