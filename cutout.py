# load image, cut out 1800x1800 square, start pipeline on it
import geopandas as gpd
from geodataset.aoi import AOIFromPackageConfig, AOIConfig
from geodataset.tilerize import RasterTilerizer
import rasterio
from rasterio.mask import mask


def cut_test_image(input_path:str, output_path:str, aoi_gdf:gpd.GeoDataFrame, size:int):
    # Convert AOI to JSON geometry
    aoi_geom = [feature["geometry"] for feature in aoi_gdf.__geo_interface__["features"]]
    with rasterio.open(input_path) as src:
        aoi_gdf = aoi_gdf.to_crs(src.crs)
        # Clip raster without loading entire image
        out_image, out_transform = mask(src, aoi_geom, crop=True)
        out_meta = src.meta.copy()

        # Update metadata
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
    return out_image, out_meta

    # tilerizer = RasterTilerizer(input_path, output_path, size, 0.0, global_aoi=aoi_path, aois_config=aoi_config, ignore_black_white_alpha_tiles_threshold=0.95)
    # tilerizer.generate_tiles()


if __name__=="__main__":
    input_paths = ['../montreal_forest_data/quebec_trees_dataset_2021-05-28/2021-05-28/zone1/2021-05-28-sbl-z1-rgb-cog.tif',
                   '../montreal_forest_data/quebec_trees_dataset_2021-06-17/2021-06-17/zone1/2021-06-17-sbl-z1-rgb-cog.tif',
                   '../montreal_forest_data/quebec_trees_dataset_2021-07-21/2021-07-21/zone1/2021-07-21-sbl-z1-rgb-cog.tif',
                   ]
    aoi_path='../montreal_forest_data/nice_cut/AOI_nice_cut2.geojson'
    aoi_gdf = gpd.read_file(aoi_path)
    output_path = '../montreal_forest_data/nice_cut'

    output_names = ['may', 'june', 'july']
    for input_path, output_name in zip(input_paths, output_names):
        out_im, out_meta = cut_test_image(input_path, output_path, aoi_gdf, 3500)
        # Save clipped raster
        with rasterio.open(output_path + '/{}_0.tif'.format(output_name), "w", **out_meta) as dest:
            dest.write(out_im)