import json
import os

import rasterio
from rasterio.mask import mask

from shapely.geometry import shape
import pystac

def fetch_geodata(url, aoi_geojson, out_dir):
    # Load AOI
    with open(aoi_geojson) as f:
        aoi = json.load(f)
        aoi_geom = [shape(aoi['features'][0]['geometry'])]

    catalog = pystac.Catalog.from_file(url)
    # print(catalog.describe())
    os.makedirs(out_dir, exist_ok=True)
    for item in catalog.get_items(recursive=True):
        for asset_key, asset in item.assets.items():
            if asset.media_type == pystac.MediaType.COG:  # Check for Cloud Optimized GeoTIFF
                print(f"Processing {asset.href}")

                with rasterio.open(asset.href) as src:
                    out_image, out_transform = mask(src, aoi_geom, crop=True)
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform
                    })

                    out_path = os.path.join(out_dir, f"{item.id}_{asset_key}.tif")
                    with rasterio.open(out_path, "w", **out_meta) as dest:
                        dest.write(out_image)

                    print(f"Saved: {out_path}")
                    quit()


if __name__ == '__main__':


    # fetch_geodata('http://www.lab.lefolab.stac-assets.umontreal.ca:8888/assets/', '../montreal_forest_data/nice_cut/AOI_nice_cut2.geojson', '../montreal_forest_data/high_res/')
    fetch_geodata(
        'http://www.lab.lefolab.stac.umontreal.ca/stac-fastapi-pgstac/api/v1/pgstac/#',
        # 'http://www.lab.lefolab.stac-assets.umontreal.ca/assets/',
        '../montreal_forest_data/nice_cut/AOI_nice_cut2.geojson',
        '../montreal_forest_data/high_res/'
    )