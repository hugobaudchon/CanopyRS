import json
import os

import rasterio
from rasterio.mask import mask

from shapely.geometry import shape
import pystac
from pyproj import Transformer

import requests
import geopandas as gpd

def transform_aoi_to_wgs84(aoi_gdf, from_crs):
    """Transforms AOI geometry from a given CRS to EPSG:4326 (WGS 84)"""
    # aoi_gdf = gpd.read_file(aoi_geojson)
    aoi_gdf2 = aoi_gdf.to_crs("EPSG:4326")
    # with open(aoi_geojson) as f:
    #     aoi = json.load(f)
    return aoi_gdf2

def fetch_geodata(url, aoi_geojson, out_dir):
    # Load AOI
    # with open(aoi_geojson) as f:
    #     aoi = json.load(f)
    #     aoi_transformed = transform_aoi_to_wgs84(aoi_geojson, aoi['crs']['properties']['name'])
    #     # aoi_geom = [shape(aoi['features'][0]['geometry'])]

    aoi_gdf = gpd.read_file(aoi_geojson)
    aoi_transformed = transform_aoi_to_wgs84(aoi_gdf, aoi_gdf.crs)


    search_payload = {
        # "datetime": f"{start_date}/{end_date}",
        "bbox": aoi_transformed.total_bounds.tolist(),
        "limit": 2000
        # Increase limit as needed
    }
    response = requests.post(
        f"{url}/search",
        json=search_payload,
        # Disable proxies for this request
        proxies={"http": None, "https": None}
    )
    response.raise_for_status()
    data = response.json()
    # Filter results for collections with the keyword in their name
    keyword = 'sbl'
    filtered_collections = []
    for feature in data.get("features", []):
        collection_name = feature.get("collection", "").lower()
        if keyword in collection_name:
            filtered_collections.append(feature)
    # print(catalog.describe())

    os.makedirs(out_dir, exist_ok=True)
    for collection in filtered_collections:
        for asset_key, asset_info in collection.get("assets", {}).items():
            asset_type = asset_info.get("type", "")
            asset_href = asset_info.get("href", "")
            mime_type = asset_info.get("type", "")
            # Check if the asset is a COG
            if ((
                    "cloud-optimized" in mime_type.lower()
                    or "cog" in asset_key.lower()
            ) and "lowres" not in asset_key.lower()
                    and "overview" not in asset_key.lower()
                    and "rgb" in asset_key.lower() #no dsm
                    and "z3" in asset_key.lower()): #only z3 for deadtrees zone
                with rasterio.open(asset_href) as src:
                    aoi_geom = [feature["geometry"] for feature in aoi_gdf.__geo_interface__["features"]]
                    out_image, out_transform = mask(src, aoi_geom, crop=True)
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform
                    })

                    out_path = os.path.join(out_dir, f"{collection['id']}_{asset_key}.tif".replace('/', '_'))
                    with rasterio.open(out_path, "w", **out_meta) as dest:
                        dest.write(out_image)

                    print(f"Saved: {out_path}")
                    # quit()



if __name__ == '__main__':


    # fetch_geodata('http://www.lab.lefolab.stac-assets.umontreal.ca:8888/assets/', '../montreal_forest_data/nice_cut/AOI_nice_cut2.geojson', '../montreal_forest_data/high_res/')
    fetch_geodata(
        'http://www.lab.lefolab.stac.umontreal.ca/stac-fastapi-pgstac/api/v1/pgstac',
        # 'http://www.lab.lefolab.stac-assets.umontreal.ca/assets/',
        '../montreal_forest_data/nice_cut/AOI_deadtrees2.geojson',
        '/run/media/beerend/LALIB_SSD_2/berend/deadtrees1/'
        # '../montreal_forest_data/pystac_download'
    )