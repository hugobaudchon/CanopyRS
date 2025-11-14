import json
import shutil
from pathlib import Path

import geopandas as gpd
from geodataset.aoi import AOIFromPackageConfig

from canopyrs.data.detection.raw_datasets.base_dataset import BasePublicZipDataset
from canopyrs.data.detection.tilerize import tilerize_with_overlap


parent_folder = Path(__file__).parent


class BCI50haDataset(BasePublicZipDataset):
    zip_url = "https://smithsonian.figshare.com/ndownloader/articles/24784053/versions/2"
    name = "panama_bci50ha"
    annotation_type = "mask"
    aois = {
        "BCI_50ha_2020_08_01_crownmap": {
            "test": parent_folder / "aois" / "20200801_aoi_test_without_holes.gpkg"
        },
        "BCI_50ha_2022_09_29_crownmap": {
            "test": parent_folder / "aois" / "20220929_aoi_test_without_holes.gpkg"
        }
    }

    categories = parent_folder / 'panama_bci_trees_categories.json'

    def _parse(self, path: str or Path):
        path = Path(path)

        assert path.exists(), f"Path {path} does not exist. Make sure you called the download method first."

        # Remove files
        (path / 'Data_description.pdf').unlink()
        (path / 'variables_description.csv').unlink()
        (path / 'BCI_50ha_2020_08_01_crownmap_improved.png').unlink()
        (path / 'BCI_50ha_2020_08_01_crownmap_raw.png').unlink()
        (path / 'BCI_50ha_2022_09_29_crownmap_improved.png').unlink()
        (path / 'BCI_50ha_2022_09_29_crownmap_raw.png').unlink()

        # Rename files
        ((path / 'BCI_50ha_2022_09_29_crownmap_raw/BCI_50ha_2022_09_29_crownmap_raw/BCI_50ha_2022_09_29_global.tif')
        .rename(path / 'BCI_50ha_2022_09_29_crownmap_raw.tif'))
        ((path / 'BCI_50ha_2020_08_01_crownmap_raw/BCI_50ha_2020_08_01_crownmap_raw/BCI_50ha_2020_08_01_global.tif')
         .rename(path / 'BCI_50ha_2020_08_01_crownmap_raw.tif'))

        # Read and write GeoDataFrames
        gdf1 = gpd.read_file(path / 'BCI_50ha_2020_08_01_crownmap_improved/BCI_50ha_2020_08_01_crownmap_improved/BCI_50ha_2020_08_01_crownmap_improved.shp')
        gdf1.to_file(path / 'BCI_50ha_2020_08_01_crownmap_improved.gpkg', driver='GPKG')

        gdf2 = gpd.read_file(path / 'BCI_50ha_2022_09_29_crownmap_improved/BCI_50ha_2022_09_29_crownmap_improved/BCI_50ha_2022_09_29_crownmap_improved.shp')
        gdf2.to_file(path / 'BCI_50ha_2022_09_29_crownmap_improved.gpkg', driver='GPKG')

        # Remove directories
        shutil.rmtree(path / 'BCI_50ha_2020_08_01_crownmap_improved')
        shutil.rmtree(path / 'BCI_50ha_2020_08_01_crownmap_raw')
        shutil.rmtree(path / 'BCI_50ha_2022_09_29_crownmap_improved')
        shutil.rmtree(path / 'BCI_50ha_2022_09_29_crownmap_raw')

        print('BCI50ha dataset has been successfully parsed.')

    def tilerize(self,
                 raw_path: str or Path,
                 output_path: str or Path,
                 folds: set[str],
                 ground_resolution: float = None,
                 scale_factor: float = None,
                 tile_size: int = 1024,
                 tile_overlap: float = 0.5,
                 binary_category: bool = True,
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
            labels = raw_path / f"{raster_name}_improved.gpkg"
            if 'Latin' in gpd.read_file(labels).columns:
                main_label_category_column_name = 'Latin'
            else:
                main_label_category_column_name = 'latin'

            aois_config = AOIFromPackageConfig(aois[raster_name])

            tilerize_with_overlap(
                raster_path=raw_path / f"{raster_name}_raw.tif",
                labels=raw_path / f"{raster_name}_improved.gpkg",
                main_label_category_column_name=main_label_category_column_name if not binary_category else None,
                coco_categories_list=json.load(open(self.categories, 'rb'))['categories'] if not binary_category else None,
                aois_config=aois_config,
                output_path=output_path,
                ground_resolution=ground_resolution,
                scale_factor=scale_factor,
                tile_size=tile_size,
                tile_overlap=tile_overlap
            )
