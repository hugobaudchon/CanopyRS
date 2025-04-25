import json
import shutil
from pathlib import Path

import geopandas as gpd
from geodataset.aoi import AOIFromPackageConfig
from geodataset.utils import create_coco_folds

from dataset.detection.raw_datasets.base_dataset import BasePublicZipDataset
from dataset.detection.tilerize import tilerize_no_overlap, combine_gdfs, tilerize_with_overlap


parent_folder = Path(__file__).parent


class BCI50haDataset(BasePublicZipDataset):
    zip_url = "https://smithsonian.figshare.com/ndownloader/articles/24784053/versions/2"
    name = "panama_bci50ha"
    annotation_type = "mask"
    aois = {
        "BCI_50ha_2020_08_01_crownmap": {
            "train": parent_folder / "20200801_aoi_train.gpkg",
            "valid": parent_folder / "20200801_aoi_valid.gpkg",
            "test": parent_folder / "20200801_aoi_test.gpkg"
        },
        "BCI_50ha_2022_09_29_crownmap": {
            "train": parent_folder / "20220929_aoi_train.gpkg",
            "valid": parent_folder / "20220929_aoi_valid.gpkg",
            "test": parent_folder / "20220929_aoi_test.gpkg"
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
                 cross_validation: bool,
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

        aois = self.aois
        for raster_name in aois:
            for raster_aoi in aois[raster_name]:
                aois[raster_name][raster_aoi] = gpd.read_file(f'{aois[raster_name][raster_aoi]}')
                if raster_aoi not in folds:
                    del aois[raster_name][raster_aoi]

            if cross_validation:
                assert 'train' in folds and 'valid' in folds, "For cross-validation, 'train' and 'valid' folds must be passed."
                # First, combining train and valid aois into a single one as we will tile them together
                # and then split the COCO into cross validation folds.
                if 'train' in aois[raster_name] and 'valid' in aois[raster_name]:
                    aois[raster_name]['train'] = combine_gdfs(aois[raster_name]['train'], aois[raster_name]['valid'])
                    del aois[raster_name]['valid']
                elif 'valid' in aois[raster_name]:
                    aois[raster_name]['train'] = aois[raster_name]['valid']
                    del aois[raster_name]['valid']

                aois_config = AOIFromPackageConfig(aois[raster_name])

                labels = raw_path / f"{raster_name}_improved.gpkg"
                if 'Latin' in gpd.read_file(labels).columns:
                    main_label_category_column_name = 'Latin'
                else:
                    main_label_category_column_name = 'latin'

                tilerize_with_overlap(
                    raster_path=raw_path / f"{raster_name}_raw.tif",
                    labels=labels,
                    main_label_category_column_name=main_label_category_column_name,
                    coco_categories_list=json.load(open(self.categories, 'rb'))['categories'],
                    aois_config=aois_config,
                    output_path=output_path
                )
            else:
                labels = raw_path / f"{raster_name}_improved.gpkg"
                if 'Latin' in gpd.read_file(labels).columns:
                    main_label_category_column_name = 'Latin'
                else:
                    main_label_category_column_name = 'latin'

                aois_config = AOIFromPackageConfig(aois[raster_name])

                tilerize_with_overlap(
                    raster_path=raw_path / f"{raster_name}_raw.tif",
                    labels=raw_path / f"{raster_name}_improved.gpkg",
                    main_label_category_column_name=main_label_category_column_name,
                    coco_categories_list=json.load(open(self.categories, 'rb'))['categories'],
                    aois_config=aois_config,
                    output_path=output_path,
                    ground_resolution=ground_resolution,
                    scale_factor=scale_factor,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap
                )
