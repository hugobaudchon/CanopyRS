import json
import shutil

from pathlib import Path
from xml.etree import ElementTree as ET

import rasterio
from geodataset.aoi import AOIFromPackageConfig
from rasterio.transform import xy
from shapely.geometry import box
import geopandas as gpd

from geodataset.utils import CocoNameConvention, COCOGenerator, TileNameConvention

from canopyrs.data.detection.raw_datasets.base_dataset import BasePublicZipDataset
from canopyrs.data.detection.tilerize import tilerize_with_overlap


parent_folder = Path(__file__).parent


class NeonTreeEvaluationDataset(BasePublicZipDataset):
    zip_url = "https://zenodo.org/api/records/5914554/files-archive"
    name = "unitedstates_neon"
    annotation_type = "box"

    aois_folder = parent_folder / 'aois'

    categories = None

    def _parse(self, path: str or Path):
        path = Path(path)

        # evaluation / test data
        test_path = path / 'test'
        test_path.mkdir(exist_ok=True)

        (path / 'evaluation' / 'evaluation' / 'RGB' / 'benchmark_annotations.csv').rename(test_path / 'benchmark_annotations.csv')
        (path / 'evaluation' / 'evaluation' / 'RGB').rename(test_path / 'RGB')

        # train data
        train_path = path / 'training'

        # removing rasters without CRS
        (train_path / 'RGB' / '2019_DSNY_5_452000_3113000_image_crop.tif').unlink()
        (train_path / 'RGB' / '2019_YELL_2_541000_4977000_image_crop.tif').unlink()
        (train_path / 'RGB' / '2019_YELL_2_528000_4978000_image_crop2.tif').unlink()

        tifs = (train_path / 'RGB').iterdir()
        tifs = [t.stem for t in tifs]

        # loop on xml files in annotations folder and move them if they are for training
        annotations_path = path / 'annotations' / 'annotations'
        for file in annotations_path.iterdir():
            if file.suffix == '.xml' and file.stem in tifs:
                file.rename(train_path / 'RGB' / file.name)

        shutil.rmtree(path / 'annotations')
        shutil.rmtree(path / 'evaluation')
        shutil.rmtree(path / 'training' / 'CHM')
        shutil.rmtree(path / 'training' / 'Hyperspectral')
        shutil.rmtree(path / 'training' / 'LiDAR')

        (path / 'training' / 'RGB').rename(path / 'train')
        shutil.rmtree(path / 'training')

        # convert xml to gpkg
        for file in (path / 'train').iterdir():
            if file.suffix == '.xml':
                xml_to_gpkg(path / 'train' / file)
                # remove the xml file once converted
                (path / 'train' / file).unlink()

        # convert test annotation csv to coco
        csv_to_coco(Path(test_path / 'benchmark_annotations.csv'))

        print('NeonTreeEvaluation dataset has been successfully parsed.')

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

        if binary_category is False:
            raise ValueError("Binary category is not supported for NeonTreeEvaluation dataset as the dataset doesn't specify tree species."
                             " Please set binary_category to False.")

        raw_path = Path(raw_path)
        output_path = Path(output_path) / self.name
        output_path.mkdir(parents=True, exist_ok=True)

        if not raw_path.name == self.name:
            raw_path = raw_path / self.name

        assert raw_path.exists(), (f"Path {raw_path} does not exist."
                                   f" Make sure you called the download AND parse methods first.")

        # Also move the test data already tilerized if 'test' is in folds (but in a separate sub-folder)
        if 'test' in folds:
            # Paths
            test_out = output_path / 'NeonTreeEvaluation_Test'
            test_out.mkdir(exist_ok=False)
            src_coco = raw_path / 'test' / 'NeonTreeEvaluation_coco_sf1p0_test.json'
            dest_coco = test_out / 'NeonTreeEvaluation_coco_sf1p0_test.json'
            tiles_src = raw_path / 'test' / 'RGB'
            tiles_dest = test_out / 'tiles' / 'test'
            tiles_dest.mkdir(parents=True, exist_ok=True)

            # 1) Load and modify COCO JSON
            with open(src_coco, 'r') as f:
                coco = json.load(f)

            rename_map = {}
            for img in coco.get('images', []):
                orig_name = img['file_name']

                new_name = TileNameConvention().create_name(
                    product_name=orig_name.replace('.tif', ''),
                    aoi='test',
                    scale_factor=1.0,
                    col=0,
                    row=0
                )

                rename_map[orig_name] = new_name
                img['file_name'] = new_name

            # 2) Write out the modified COCO
            with open(dest_coco, 'w') as f:
                json.dump(coco, f, indent=2)

            # 3) Copy & rename only the files listed in the original COCO
            for orig_name, new_name in rename_map.items():
                src_file = tiles_src / orig_name
                if not src_file.exists():
                    print(f"Warning: source tile not found: {src_file}")
                    continue
                dest_file = tiles_dest / new_name
                shutil.copy2(src_file, dest_file)

        train_files = (raw_path / 'train').iterdir()
        tif_names = [f.stem for f in train_files if f.suffix == '.tif']
        for tif_name in tif_names:
            aois = {}
            if 'train' in folds:
                aois['train'] = self.aois_folder / f"{tif_name}_aoi_train.gpkg"
            if 'valid' in folds:
                aois['valid'] = self.aois_folder / f"{tif_name}_aoi_valid.gpkg"

            aois_config = AOIFromPackageConfig(aois)

            try:
                tilerize_with_overlap(
                    raster_path=raw_path / 'train' / f"{tif_name}.tif",
                    labels=raw_path / 'train' / f"{tif_name}.gpkg",
                    main_label_category_column_name=None,
                    coco_categories_list=None,
                    aois_config=aois_config,
                    output_path=output_path,
                    ground_resolution=ground_resolution,
                    scale_factor=scale_factor,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap
                )
            except Exception:
                print(f"Error tilerizing {tif_name}.tif. Skipping this file.")
                print(f"This is probably due to no annotations being found in one of the AOIs.")
                continue


def xml_to_gpkg(xml_path: str or Path):
    # Derive file stem and TIFF file path
    xml_path = Path(xml_path)
    file_stem = xml_path.stem
    xml_dir = xml_path.parent
    tif_path = xml_dir / f"{file_stem}.tif"

    if not tif_path.exists():
        raise FileNotFoundError(f"TIFF file not found: {tif_path}")

    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Open the corresponding TIFF file to get transform and CRS
    with rasterio.open(tif_path) as dataset:
        transform = dataset.transform
        raster_crs = dataset.crs

    # Extract bounding boxes and transform them
    geoms = []
    attributes = []

    for obj in root.findall("object"):
        # Extract bounding box coordinates
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # Transform pixel coordinates to CRS
        top_left = xy(transform, ymin, xmin, offset="ul")
        bottom_right = xy(transform, ymax, xmax, offset="lr")
        geom = box(top_left[0], bottom_right[1], bottom_right[0], top_left[1])

        # Store geometry and attributes
        geoms.append(geom)
        attributes.append({"name": obj.find("name").text})

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(attributes, geometry=geoms, crs=raster_crs)

    # Save to GPKG
    gpkg_path = xml_dir / f"{file_stem}.gpkg"
    gdf.to_file(str(gpkg_path), driver="GPKG")

    print(f"GeoPackage saved to: {gpkg_path}")


def csv_to_coco(csv_path: Path):
    gdf = gpd.read_file(csv_path)
    gdf['geometry'] = gdf.apply(lambda x: box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)

    images_gdfs = gdf.groupby('image_path')

    coco_name = CocoNameConvention().create_name(
        product_name='NeonTreeEvaluation',
        fold='test',
        scale_factor=1
    )

    COCOGenerator(
        output_path=csv_path.parent / coco_name,
        description='NeonTreeEvaluation',
        tiles_paths=[csv_path.parent / 'RGB' / img_path for img_path in images_gdfs.groups.keys()],
        polygons=[gdf.geometry.tolist() for (path, gdf) in images_gdfs],
        scores=None,
        categories=None,
        other_attributes=None,
        use_rle_for_labels=True,
        n_workers=4,
        coco_categories_list=None
    ).generate_coco()
