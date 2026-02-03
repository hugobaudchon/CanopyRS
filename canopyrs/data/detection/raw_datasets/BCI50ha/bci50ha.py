import json
import shutil
from pathlib import Path

import geopandas as gpd
import requests
from geodataset.aoi import AOIFromPackageConfig

from canopyrs.data.detection.raw_datasets.base_dataset import BasePublicZipDataset
from canopyrs.data.detection.raw_datasets.utils import download_with_progress, uncompress_with_progress
from canopyrs.data.detection.tilerize import tilerize_with_overlap


parent_folder = Path(__file__).parent


class BCI50haDataset(BasePublicZipDataset):
    zip_url = None  # Not used - we download individual files via API
    article_id = "24784053"
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

    categories = parent_folder / 'categories' / 'panama_bci_trees_categories.json'

    def download(self, output_path: str or Path):
        """Download BCI50ha dataset using Figshare API to get individual file URLs."""
        output_path = Path(output_path) / self.name
        output_path.mkdir(parents=True, exist_ok=True)

        # Get file metadata from Figshare API
        api_url = f"https://api.figshare.com/v2/articles/{self.article_id}"
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()

        # Download the zip files we need (both 2020 and 2022 datasets)
        files_to_download = [
            'BCI_50ha_2020_08_01_crownmap_raw.zip',
            'BCI_50ha_2020_08_01_crownmap_improved.zip',
            'BCI_50ha_2022_09_29_crownmap_raw.zip',
            'BCI_50ha_2022_09_29_crownmap_improved.zip',
            'variables_description.csv',
            'Data_description.pdf',
        ]

        for file_info in data.get('files', []):
            file_name = file_info['name']
            if file_name in files_to_download:
                print(f"Downloading {file_name}...")
                download_url = file_info['download_url']
                local_file = output_path / file_name
                download_with_progress(download_url, str(local_file))

                # Extract zip files
                if file_name.endswith('.zip'):
                    print(f"Extracting {file_name}...")
                    uncompress_with_progress(str(local_file), str(output_path))

        self._parse(output_path)

    def _parse(self, path: str or Path):
        path = Path(path)

        assert path.exists(), f"Path {path} does not exist. Make sure you called the download method first."

        # Remove files (missing_ok=True in case they weren't downloaded)
        (path / 'Data_description.pdf').unlink(missing_ok=True)
        (path / 'variables_description.csv').unlink(missing_ok=True)
        (path / 'BCI_50ha_2020_08_01_crownmap_improved.png').unlink(missing_ok=True)
        (path / 'BCI_50ha_2020_08_01_crownmap_raw.png').unlink(missing_ok=True)
        (path / 'BCI_50ha_2022_09_29_crownmap_improved.png').unlink(missing_ok=True)
        (path / 'BCI_50ha_2022_09_29_crownmap_raw.png').unlink(missing_ok=True)

        # Find and rename 2020 raw files (search in nested dirs)
        raw_2020_files = list(path.glob('**/BCI_50ha_2020_08_01_global.tif'))
        if raw_2020_files:
            raw_2020_files[0].rename(path / 'BCI_50ha_2020_08_01_crownmap_raw.tif')

        # Find and rename 2022 raw files (if they exist)
        raw_2022_files = list(path.glob('**/BCI_50ha_2022_09_29_global.tif'))
        if raw_2022_files:
            raw_2022_files[0].rename(path / 'BCI_50ha_2022_09_29_crownmap_raw.tif')

        # Find and convert 2020 improved shapefiles to gpkg
        improved_2020_shps = list(path.glob('**/BCI_50ha_2020_08_01_crownmap_improved.shp'))
        if improved_2020_shps:
            gdf1 = gpd.read_file(improved_2020_shps[0])
            gdf1.to_file(path / 'BCI_50ha_2020_08_01_crownmap_improved.gpkg', driver='GPKG')

        # Find and convert 2022 improved shapefiles to gpkg (if exists)
        improved_2022_shps = list(path.glob('**/BCI_50ha_2022_09_29_crownmap_improved.shp'))
        if improved_2022_shps:
            gdf2 = gpd.read_file(improved_2022_shps[0])
            gdf2.to_file(path / 'BCI_50ha_2022_09_29_crownmap_improved.gpkg', driver='GPKG')

        # Remove directories (if they exist)
        for dir_name in [
            'BCI_50ha_2020_08_01_crownmap_improved',
            'BCI_50ha_2020_08_01_crownmap_raw',
            'BCI_50ha_2022_09_29_crownmap_improved',
            'BCI_50ha_2022_09_29_crownmap_raw'
        ]:
            dir_path = path / dir_name
            if dir_path.exists():
                shutil.rmtree(dir_path)

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
