import os
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import rasterio
from datasets import load_dataset
from geodataset.utils import CocoNameConvention, TileNameConvention
from rasterio import Affine
from geodataset.dataset import SegmentationLabeledRasterCocoDataset
from tqdm import tqdm


class HFDatasetTools:
    @staticmethod
    def process_coco_datasets_for_huggingface(root_paths: list[str], include_segmentations: bool, output_path: str):
        """
        Process multiple COCO datasets across different splits and save as a proper Hugging Face dataset

        Args:
            root_paths (list): Root paths to the tilerized datasets
            include_segmentations (bool): Whether to include segmentation annotations or not
            output_path (str): Path to save the processed dataset

        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        for fold in ['train', 'valid', 'test']:
            try:
                dataset = SegmentationLabeledRasterCocoDataset(
                    fold=fold,
                    root_path=root_paths
                )
            except Exception as e:
                warnings.warn(f"Error loading dataset for fold {fold}: {e}")
                continue

            fold_path = output_path / fold
            fold_path.mkdir(parents=True, exist_ok=True)

            tiles_metadata = []
            for (index, tile_data) in tqdm(dataset.tiles.items(), desc=f"Processing {fold} fold", total=len(dataset.tiles)):
                tile_path = Path(tile_data['path'])
                del tile_data['path']

                tile_data['fold'] = fold
                tile_data['raster_name'] = str(tile_path.parent.parent.parent.name)
                tile_data['location'] = str(tile_path.parent.parent.parent.parent.name)
                tile_data['file_name'] = tile_data['name']      # for huggingface convention
                tile_data['tile_name'] = tile_data['name']      # for extracting needs later, since tile_data['file_name'] isn't kept by huggingface when pushing to the hub
                del tile_data['name']
                tile_data['annotations'] = {
                     'bbox': [
                        [float(coord) for coord in x['bbox']]
                        for x in tile_data['labels']
                    ],
                    'segmentation': [x['segmentation'] for x in tile_data['labels']] if include_segmentations else None,
                    'area': [
                        float(x['area'])
                        for x in tile_data['labels']
                    ],
                    'iscrowd': [x['iscrowd'] for x in tile_data['labels']],
                    # 'is_rle_format': [x['is_rle_format'] for x in tile_data['labels']] if include_segmentations else None,
                    'category': ['tree',] * len(tile_data['labels']),
                }
                del tile_data['labels']
                tiles_metadata.append(tile_data)

                with rasterio.open(tile_path) as src:
                    metadata = {
                        'crs': src.crs.to_string() if src.crs is not None else None,
                        'transform': list(src.transform),
                        'bounds': list(src.bounds),
                        'width': src.width,
                        'height': src.height,
                        'count': src.count,
                        'dtypes': src.dtypes,
                        'nodata': float(src.nodata) if src.nodata is not None else 0.0
                    }

                    # Add metadata to tile_data
                    tile_data['tile_metadata'] = metadata

                    # Copy file to the new location in huggingface directory structure
                    out_meta = src.meta.copy()
                    out_meta.update({
                        'compress': 'deflate'   # add compression so that it takes less space on huggingface
                    })
                    with rasterio.open(fold_path / tile_path.name, 'w', **out_meta) as dst:
                        for band in range(1, src.count + 1):
                            dst.write(src.read(band), band)

            # write tiles_metadata to .jsonl
            jsonl_path = fold_path / 'metadata.jsonl'
            with jsonl_path.open('w') as f:
                for item in tiles_metadata:
                    f.write(json.dumps(item) + '\n')

    @staticmethod
    def push_to_hub(processed_coco_dataset_path: str, repo_name: str):
        """
        Push the dataset to Hugging Face Hub in .parquet format

        Args:
            processed_coco_dataset_path (str): Path to the processed COCO dataset (using the method process_coco_datasets_for_huggingface)
            repo_name (str): Name of the repository on Hugging Face Hub (you must be logged-in first, see https://huggingface.co/docs/huggingface_hub/en/quick-start)

        """
        dataset = load_dataset("imagefolder", data_dir=processed_coco_dataset_path)
        dataset.push_to_hub(repo_name)

    @staticmethod
    def download_and_extract_huggingface_dataset(hf_dataset_name: str,
                                                 root_output_path: str,
                                                 cleanup_hf_cache: bool = True,
                                                 cleanup_hf_cache_temp_dir: str = './temp_hf_cache'):
        """
        Download a Hugging Face dataset and extract it to disk in the original directory structure,
        assuming that the `image` field in each record is a decoded PIL Image.
        Output structure:

            <root_output_path>/<hf_dataset_name>/
              └── <location>/
                    └── <raster_name>/
                          ├── coco_<fold>_<uuid>.json     # Aggregated metadata for that raster and fold
                          └── tiles/
                                └── <fold>/               # Contains tile image files (.tif)

        Expects each record to have:
          - 'location': top-level directory name
          - 'raster_name': the raster folder name
          - 'fold': split designation (e.g., 'train', 'valid', or 'test')
          - 'tile_name': unique tile file name
          - 'image': a PIL Image (decoded already)
          - 'tile_metadata': dictionary with keys like 'crs', 'transform', 'bounds', 'width', 'height', etc.
          - Optionally, top-level 'crs' and 'transform' can override those in tile_metadata.
        """
        # Download the dataset from HF Hub.
        if cleanup_hf_cache:
            dataset = load_dataset(hf_dataset_name, cache_dir=cleanup_hf_cache_temp_dir)
        else:
            dataset = load_dataset(hf_dataset_name)
        out_base = Path(root_output_path)
        out_base.mkdir(parents=True, exist_ok=True)

        # Group dataset records by (location, raster_name, fold)
        groups = {}
        for split_name, ds in dataset.items():
            ds_meta = ds.remove_columns("image")  # Avoid loading full image data during grouping.
            for i, record in enumerate(ds_meta):
                try:
                    loc = record['location']
                    raster = record['raster_name']
                    fold = record['fold']
                    tile_name = record['tile_name']
                except KeyError as e:
                    raise KeyError(f"Expected key missing in record: {e}")

                key = (loc, raster, fold)
                # Store a tuple of (split_name, index) so we can retrieve the full record later.
                groups.setdefault(key, []).append((split_name, i))

        # Process each group (each raster in a given fold)
        for (loc, raster, fold), indices in groups.items():
            # Create output directories.
            raster_dir = out_base / loc / raster
            tiles_dir = raster_dir / "tiles" / fold
            tiles_dir.mkdir(parents=True, exist_ok=True)

            tile_id = 1
            annotation_id = 1
            coco_tile_metadata = []
            coco_annotations_metadata = []
            # Loop over each tile/image record in the group.
            for (split, idx) in tqdm(indices, desc=f"Extracting {loc} || {raster} || {fold}", total=len(indices)):
                ds_full = dataset[split]
                rec = ds_full[idx]
                dest_image_path = tiles_dir / rec['tile_name']

                data = np.array(rec.get('image'))
                if data.ndim == 3:
                    data = np.moveaxis(data, -1, 0)

                rec_meta = rec.get('tile_metadata')
                meta = {
                    'crs': rec_meta.get('crs'),
                    'bounds': list(rec_meta.get('bounds')),
                    'transform': Affine(*rec_meta.get('transform')),
                    'width': rec_meta.get('width'),
                    'height': rec_meta.get('height'),
                    'count': rec_meta.get('count'),
                    'dtype': data.dtype,
                    'nodata': rec_meta.get('nodata'),
                    'driver': "GTiff",
                    'compress': "deflate"
                }

                try:
                    with rasterio.open(dest_image_path, "w", **meta) as dst:
                        dst.write(data)
                except Exception as e:
                    print(f"Error writing TIFF {dest_image_path}: {e}")

                coco_tile_metadata.append({
                    'id': tile_id,
                    'file_name': rec['tile_name'],
                    'width': rec_meta['width'],
                    'height': rec_meta['height']
                })

                for j in range(len(rec['annotations']['bbox'])):
                    coco_annotations_metadata.append(
                        {
                            'id': annotation_id,
                            'bbox': rec['annotations']['bbox'][j],
                            'segmentation': rec['annotations']['segmentation'][j] if rec['annotations']['segmentation'] else None,
                            'area': rec['annotations']['area'][j],
                            'iscrowd': rec['annotations']['iscrowd'][j],
                            # 'is_rle_format': rec['annotations']['is_rle_format'][j] if rec['annotations']['is_rle_format'] else None,
                            'category_id': 1,  # Assuming a single category for all annotations
                            'image_id': tile_id
                        }
                    )
                    annotation_id += 1

                tile_id += 1
                annotation_id += len(rec['annotations']['bbox'])

            _, scale_factor, ground_resolution, _, _, _ = TileNameConvention.parse_name(coco_tile_metadata[0]['file_name'])

            coco_name = CocoNameConvention.create_name(
                product_name=raster,
                fold=fold,
                scale_factor=scale_factor,
                ground_resolution=ground_resolution
            )
            coco_path = raster_dir / coco_name
            coco = {
                'info': {
                    'description': f"localisation={loc} raster_name={raster} fold={fold}",
                },
                'licenses': ['cc-by-nc-4.0'],
                'images': coco_tile_metadata,
                'annotations': coco_annotations_metadata,
                'categories': [{
                    'id': 1,
                    'name': 'tree',
                    'supercategory': None
                }]
            }
            with coco_path.open('w') as f:
                json.dump(coco, f, indent=3)

        # Remove the temporary HF cache directory if cleanup is enabled.
        if cleanup_hf_cache:
            dataset.cleanup_cache_files()
            try:
                if os.path.exists(cleanup_hf_cache_temp_dir):
                    shutil.rmtree(cleanup_hf_cache_temp_dir)
                    print(f"Successfully removed temporary HF cache directory: {cleanup_hf_cache_temp_dir}")
                else:
                    print(f"No temporary HF cache directory found at: {cleanup_hf_cache_temp_dir}")
            except Exception as e:
                print(f"Warning: Unable to cleanup temporary HF cache directory. {e}")


if __name__ == "__main__":
    hf_dataset_name = "CanopyRSAdmin/BCI50ha"

    include_segmentations = True
    output_folder = '/home/hugo/Documents/CanopyRS/huggingface_datasets/bci50ha'
    root_paths = [
        # '/home/hugo/Documents/CanopyRS/data/tilerized/selva_box/tilerized_3555_0p5_0p045_None',
        # '/home/hugo/Documents/CanopyRS/data/tilerized/selva_box/tilerized_1777_0p5_0p045_None',
        # '/home/hugo/Documents/CanopyRS/data/tilerized/selva_box/tilerized_1777_0p75_0p045_None'

        # '/home/hugo/Documents/CanopyRS/data/tilerized/neon_trees/tilerized_400_0p5_None_1p0',
        # '/home/hugo/Documents/CanopyRS/data/tilerized/neon_trees/tilerized_1200_0p5_None_1p0'

        '/home/hugo/Documents/CanopyRS/data/tilerized/bci50ha/tilerized_1777_0p75_0p045_None'

        # '/home/hugo/Documents/CanopyRS/data/tilerized/quebec_trees/tilerized_1666_0p5_0p03_None',
        # '/home/hugo/Documents/CanopyRS/data/tilerized/quebec_trees/tilerized_3333_0p5_0p03_None'

        # '/home/hugo/Documents/CanopyRS/data/tilerized/oam_tcd/tilerized_1024_0p5_None_1p0'
    ]

    HFDatasetTools.process_coco_datasets_for_huggingface(root_paths, include_segmentations, output_folder)
    HFDatasetTools.push_to_hub(output_folder, hf_dataset_name)

    # HFDatasetTools.download_and_extract_huggingface_dataset(hf_dataset_name, "/home/hugo/Documents/CanopyRS/extracted_datasets")
