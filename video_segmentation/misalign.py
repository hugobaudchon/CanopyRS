import os

import geopandas as gpd
import rasterio
from rasterio import Affine

def misalign(input_file, distance, rand_dir = False):
    # Open the source GeoTIFF
    with rasterio.open(input_file) as src:
        profile = src.profile.copy()
        data = src.read()

        # Get the current transform
        transform = src.transform

        # Shift distance meters to the right (x-axis shift)
        new_transform = Affine(transform.a, transform.b, transform.c + distance,
                               transform.d, transform.e, transform.f)

        # Update the transform in the profile
        profile.update(transform=new_transform)
    return data, profile


if __name__ == "__main__":
    input_folder = '../../montreal_forest_data/nice_cut/small_warped/'
    distance = 2
    input_list = [input_file for input_file in os.listdir(input_folder)]
    input_list.sort()
    for input_file in input_list:
        if input_file.endswith('.tif'):
            input_file_path = os.path.join(input_folder, input_file)
            data, profile = misalign(input_file_path, distance)
            distance += 2
            with rasterio.open(f'../../montreal_forest_data/nice_cut/misalign/{input_file}', 'w', **profile) as dst:
                dst.write(data)