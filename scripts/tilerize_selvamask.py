import sys
from pathlib import Path

from geodataset.aoi import AOIFromPackageConfig

from data.detection.tilerize import tilerize_with_overlap


if __name__ == "__main__":
    ground_resolution = 0.045

    train_tile_size = 3555
    train_overlap = 0.5

    eval_tile_size = 1777
    eval_overlap = 0.75


    raw_data_dir = Path("../bcitile")
    output_data_dir = Path("../tilerized/selvamask")
    
    # get list of .tif files in a directory
    tif_files = list(raw_data_dir.glob("*.tif"))

    # remove the _rgb suffix at end of file stems
    tif_files = [p.name.removesuffix(".cog.tif").removesuffix(".tif").removesuffix("_rgb") for p in tif_files]
    print(tif_files)
    
    for tif_name in tif_files:
        labels_gpkg_path = raw_data_dir / f"{tif_name}_labels_masks.gpkg"
        train_gpkg_aoi_path = raw_data_dir / f"{tif_name}_rgb_aoi_train.gpkg"
        valid_gpkg_aoi_path = raw_data_dir / f"{tif_name}_rgb_aoi_valid.gpkg"
        test_gpkg_aoi_path = raw_data_dir / f"{tif_name}_rgb_aoi_test.gpkg"

        # train
        train_aois_config = AOIFromPackageConfig(
            aois={
                'train': train_gpkg_aoi_path
            }
        )
        tilerize_with_overlap(
            raster_path=raw_data_dir / f"{tif_name}_rgb.cog.tif",
            labels=labels_gpkg_path,
            main_label_category_column_name=None,
            coco_categories_list=None,
            aois_config=train_aois_config,
            output_path=output_data_dir,
            ground_resolution=ground_resolution,
            scale_factor=None,
            tile_size=train_tile_size,
            tile_overlap=train_overlap
        )

        # valid and test
        valid_test_aois_config = AOIFromPackageConfig(
            aois={
                'valid': valid_gpkg_aoi_path,
                'test': test_gpkg_aoi_path
            }
        )
        tilerize_with_overlap(
            raster_path=raw_data_dir / f"{tif_name}_rgb.cog.tif",
            labels=labels_gpkg_path,
            main_label_category_column_name=None,
            coco_categories_list=None,
            aois_config=valid_test_aois_config,
            output_path=output_data_dir,
            ground_resolution=ground_resolution,
            scale_factor=None,
            tile_size=eval_tile_size,
            tile_overlap=eval_overlap
        )

    print(tif_files)