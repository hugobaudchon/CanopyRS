# load image, cut out 1800x1800 square, start pipeline on it
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
import numpy as np

def warp_raster(source_path:str, target_path:str):
    # Open the source raster you want to warp
    with rasterio.open(source_path) as src:
        src_data = src.read()
        src_transform = src.transform
        src_crs = src.crs
        num_bands = src.count  # Number of bands

    # Open the target raster whose grid/resolution you want to match
    with rasterio.open(target_path) as target:
        dst_transform = target.transform
        dst_crs = target.crs
        dst_shape = (target.height, target.width)
        # dst_shape = target.shape

    # Prepare an empty array for the reprojected data
    dst_data = np.empty((num_bands,*dst_shape), dtype=src_data.dtype)
    # Reproject the source raster to match the target raster
    reproject(
        source=src_data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_resolution=None,  # Not needed since we use the target transform and shape
        resampling=Resampling.nearest  # You can also use bilinear, cubic, etc.
    )
    # Optionally save the result
    out_meta = target.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": dst_shape[0],
        "width": dst_shape[1],
        "transform": dst_transform,
        "crs": dst_crs,
        "count": num_bands
    })
    return dst_data, out_meta


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
    # input_paths = ['../montreal_forest_data/quebec_trees_dataset_2021-05-28/2021-05-28/zone1/2021-05-28-sbl-z1-rgb-cog.tif',
    #                '../montreal_forest_data/quebec_trees_dataset_2021-06-17/2021-06-17/zone1/2021-06-17-sbl-z1-rgb-cog.tif',
    #                '../montreal_forest_data/quebec_trees_dataset_2021-07-21/2021-07-21/zone1/2021-07-21-sbl-z1-rgb-cog.tif',
    #                ]
    from_large_and_warp = False
    input_paths = []
    dates = ['05-28', '06-17', '07-21', '08-18', '09-02', '10-07']
    for date in dates:
        if from_large_and_warp:
            input_paths.append(f'../montreal_forest_data/quebec_trees_dataset_2021-{date}/2021-{date}/zone1/2021-{date}-sbl-z1-rgb-cog.tif')
        else:
            input_paths.append(f'../montreal_forest_data/nice_cut/small_warped/{date}_1.tif')
    if from_large_and_warp:
        aoi_path = '../montreal_forest_data/nice_cut/AOI_nice_cut2.geojson'
        output_path = '../montreal_forest_data/nice_cut'
    else:
        aoi_path='../montreal_forest_data/nice_cut/AOI_nice_cut3_tiny.geojson'
        output_path = '../montreal_forest_data/nice_cut/tiny'
    aoi_gdf = gpd.read_file(aoi_path)


    # output_names = ['may', 'june', 'july']
    for input_path, output_name in zip(input_paths, dates):
        out_im, out_meta = cut_test_image(input_path, output_path, aoi_gdf, 1800)
        # Save clipped raster
        with rasterio.open(output_path + '/{}_0.tif'.format(output_name), "w", **out_meta) as dest:
            dest.write(out_im)

    if from_large_and_warp:
        print("cutting finished. now warping to match first raster")
        for output_name in dates[1:]:
            out_im, out_meta = warp_raster(output_path + '/{}_0.tif'.format(output_name), output_path + '/{}_0.tif'.format(dates[0]))
            with rasterio.open(output_path + '/{}_1.tif'.format(output_name), "w", **out_meta) as dest:
                dest.write(out_im)
