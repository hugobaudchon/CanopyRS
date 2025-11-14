import shutil
from pathlib import Path

import rasterio
import pandas as pd
import geopandas as gpd
from geodataset.aggregator import Aggregator
from geodataset.aoi import AOIFromPackageConfig
from rasterio.transform import xy
from shapely.geometry.geo import box

from canopyrs.data.detection.raw_datasets.base_dataset import BasePublicZipDataset
from canopyrs.data.detection.tilerize import tilerize_with_overlap


parent_folder = Path(__file__).parent


class ReforesTreeDataset(BasePublicZipDataset):
    zip_url = "https://zenodo.org/records/6813783/files/reforesTree.zip?download=1"
    name = 'ecuador_reforestree'
    annotation_type = "box"
    aois = {
        "all": {
            "train": parent_folder / "aoi_train.gpkg",
            "valid": parent_folder / "aoi_valid.gpkg",
            "test": parent_folder / "aoi_test.gpkg"
        },
        "Carlos Vera Arteaga RGB": {
            "train": parent_folder / "aoi_train.gpkg",
        },
        "Carlos Vera Guevara RGB": {
            "test": parent_folder / "aoi_test.gpkg"
        },
        "Flora Pluas RGB": {
            "train": parent_folder / "aoi_train.gpkg",
        },
        "Leonor Aspiazu RGB": {
            "valid": parent_folder / "aoi_valid.gpkg",
        },
        "Manuel Macias RGB": {
            "train": parent_folder / "aoi_train.gpkg",
        },
        "Nestor Macias RGB": {
            "train": parent_folder / "aoi_train.gpkg",
        }
    }

    categories = None

    def _parse(self, path: str or Path):
        path = Path(path)

        shutil.rmtree(path / 'model')
        shutil.rmtree(path / 'tiles')
        shutil.rmtree(path / 'mapping')
        (path / 'field_data.csv').unlink()
        (path / 'wwf_ecuador/RGB Orthomosaics/Manuel Macias RGB_padded.tif').unlink()

        annotations_df = pd.read_csv(path / "annotations/cleaned/clean_annotations.csv")
        tif_files = [f for f in (path / 'wwf_ecuador/RGB Orthomosaics').iterdir() if f.suffix == '.tif']
        annotations_df['geometry'] = annotations_df[['Xmin', 'Ymin', 'Xmax', 'Ymax']].values.tolist()
        annotations_df['geometry'] = annotations_df['geometry'].apply(lambda x: box(*x))
        annotations_gdf = gpd.GeoDataFrame(annotations_df, geometry='geometry')
        annotations_gdf.drop(columns=['Xmin', 'Ymin', 'Xmax', 'Ymax',
                                      'xmin', 'ymin', 'xmax', 'ymax',
                                      'lon', 'lat', 'X', 'Y',
                                      'tile_index',
                                      'tile_xmin', 'tile_ymin', 'tile_xmax', 'tile_ymax'
                                      ], inplace=True)

        print('Adjusting CRS of ReforesTree rasters...')
        for tif_file in tif_files:
            print(f'Processing {tif_file.name}')
            raster_name_prefix = tif_file.stem
            annotations = annotations_gdf[annotations_gdf['img_name'].str.startswith(raster_name_prefix, na=False)]
            new_tif_path = path / tif_file.name
            tif_file.rename(new_tif_path)

            with rasterio.open(new_tif_path) as dataset:
                # the annotations have many duplicate geometries, so we need to remove them
                annotations = transform_annotations_with_geometry(annotations, dataset)
                annotations['img_path'] = new_tif_path
                aggregator_output_path = path / f'{raster_name_prefix}_annotations_box_aggregator_temp.gpkg'
                aggregator = Aggregator.from_gdf(
                    output_path=aggregator_output_path,
                    gdf=annotations,
                    tiles_paths_column='img_path',
                    polygons_column='geometry',
                    scores_column='score',
                    nms_threshold=0.7,
                    nms_algorithm='iou'
                )
                final_annotations = aggregator.polygons_gdf
                final_annotations.drop(
                    columns=['img_name', 'img_path'],
                    inplace=True
                )
                aggregator_output_path.unlink()
                final_annotations.to_file(str(path / f'{raster_name_prefix}_annotations_box.gpkg'), driver='GPKG')

        shutil.rmtree(path / 'annotations')
        shutil.rmtree(path / 'wwf_ecuador')

        print('ReforesTree dataset has been successfully parsed.')

    def tilerize(self,
                 raw_path: str,
                 output_path: str,
                 folds: set[str],
                 ground_resolution: float = None,
                 scale_factor: float = None,
                 tile_size: int = 1024,
                 tile_overlap: float = 0.5,
                 binary_category: bool = True,
                 **kwargs):

        if binary_category is False:
            raise ValueError("Binary category is not supported for ReforesTree dataset as the dataset doesn't specify tree species."
                             " Please set binary_category to False.")

        raw_path = Path(raw_path)
        output_path = Path(output_path) / self.name
        output_path.mkdir(parents=True, exist_ok=True)

        if not raw_path.name == self.name:
            raw_path = raw_path / self.name

        assert raw_path.exists(), (f"Path {raw_path} does not exist."
                                   f" Make sure you called the download AND parse methods first.")

        tifs = [f for f in raw_path.iterdir() if f.suffix == '.tif']

        for tif in tifs:
            aois = self.aois[tif.stem]
            aois = {k: v for k, v in aois.items() if k in folds}
            if not aois:
                continue
            aois_config = AOIFromPackageConfig(aois)

            tilerize_with_overlap(
                raster_path=tif,
                labels=tif.parent / f"{tif.stem}_annotations_box.gpkg",
                main_label_category_column_name=None,
                coco_categories_list=None,
                aois_config=aois_config,
                output_path=output_path,
                ground_resolution=ground_resolution,
                scale_factor=scale_factor,
                tile_size=tile_size,
                tile_overlap=tile_overlap
            )


def transform_annotations_with_geometry(annotations, dataset):
    # Ensure we're working on a copy to avoid SettingWithCopyWarning
    annotations = annotations.copy()

    # Step 1: Transform geometries from pixel coordinates to spatial coordinates in the raster's CRS
    transform = dataset.transform
    crs = dataset.crs

    def pixel_to_spatial(geom):
        if geom.is_empty:  # Handle empty geometries gracefully
            return geom
        # Extract bounding box (assume geom is a Shapely box)
        minx, miny, maxx, maxy = geom.bounds
        # Transform pixel coordinates to spatial CRS
        topleft = xy(transform, miny, minx, offset="ul")
        bottomright = xy(transform, maxy, maxx, offset="lr")
        return box(topleft[0], topleft[1], bottomright[0], bottomright[1])

    # Apply the transformation
    annotations["geometry"] = annotations["geometry"].apply(pixel_to_spatial)

    # Step 2: Set CRS for the transformed geometries
    annotations.set_crs(crs, inplace=True)

    return annotations

