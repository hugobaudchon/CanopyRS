import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List, Union, Optional

import cv2
import numpy as np
import rasterio
from PIL import Image, UnidentifiedImageError
from geodataset.utils import CocoNameConvention, coco_rle_segmentation_to_mask, \
    TileNameConvention
from rasterio.windows import Window
from tqdm import tqdm
from pycocotools import mask as mask_utils

from canopyrs.data.detection.raw_datasets.base_dataset import BasePublicZipDataset


"""
DISCLAMER: this script was iteratively built using ChatGPT, and is thus pretty inefficient/unnecessarily complex.
But the output was manually verified and is good.
# TODO: refactor this script and clean up.
"""


class OamTcdDataset(BasePublicZipDataset):
    zip_url = "https://zenodo.org/api/records/11617167/files-archive"
    name = "global_oamtcd"
    annotation_type = "mask"
    aois = None  # They already provide metadata for the aoi of each image in the dataset
    categories = None

    def _parse(self, path: str or Path):
        path = Path(path)

        shutil.rmtree(os.path.join(path, 'images-nc'))
        shutil.rmtree(os.path.join(path, 'images-sa'))
        shutil.rmtree(os.path.join(path, 'masks'))
        shutil.rmtree(os.path.join(path, 'train-nc'))

        # they made a mistake and interchanged the names of the folders for nc & sa masks and images
        (path / 'masks-nc').rename(path / 'images-nc')
        (path / 'masks-sa').rename(path / 'images-sa')

    def tilerize(self,
                 raw_path: str or Path,
                 output_path: str or Path,
                 remove_tree_group_annotations: bool,
                 binary_category: bool = True,
                 use_tc_sa_licensed_data: bool = False,
                 test_and_valid_cut_image_size: int = 1024,
                 test_and_valid_cut_image_overlap: float = 0.5,
                 **kwargs):

        if binary_category is False:
            raise ValueError("Binary category is not supported for OAM-TCD dataset as the dataset doesn't specify tree species."
                             " Please set binary_category to False.")

        raw_path = Path(raw_path)
        output_path = Path(output_path) / self.name
        output_path.mkdir(parents=True, exist_ok=True)

        if not raw_path.name == self.name:
            raw_path = raw_path / self.name

        assert raw_path.exists(), (f"Path {raw_path} does not exist."
                                   f" Make sure you called the download AND parse methods first.")

        # Copy the test data already tilerized
        print("Copying test data...")
        (output_path / 'oam_tcd').mkdir(exist_ok=True)
        if use_tc_sa_licensed_data:
            (output_path / 'oam_tcd_nc').mkdir(exist_ok=True)
            (output_path / 'oam_tcd_sa').mkdir(exist_ok=True)

        # Copy annotations
        print("Copying test annotations...")
        coco_convention = CocoNameConvention()
        coco_name_test = coco_convention.create_name(product_name='oam_tcd', fold='test')

        shutil.copy2(raw_path / 'test/dataset/holdout' / 'test.json',
                     output_path / 'oam_tcd' / coco_name_test)

        if use_tc_sa_licensed_data:
            coco_name_test_nc = coco_convention.create_name(product_name='oam_tcd_nc', fold='test')
            shutil.copy2(raw_path / 'test-nc/dataset-nc/holdout' / 'test.json',
                         output_path / 'oam_tcd_nc' / coco_name_test_nc)
            coco_name_test_sa = coco_convention.create_name(product_name='oam_tcd_sa', fold='test')
            shutil.copy2(raw_path / 'test-sa/dataset-sa/holdout' / 'test.json',
                         output_path / 'oam_tcd_sa' / coco_name_test_sa)

        # Copy images
        print("Copying images...")
        shutil.copytree(raw_path / 'images/dataset/holdout' / 'images',
                        output_path / 'oam_tcd' / 'tiles' / 'all', dirs_exist_ok=True)

        if use_tc_sa_licensed_data:
            shutil.copytree(raw_path / 'images-nc/dataset-nc/holdout' / 'images',
                            output_path / 'oam_tcd_nc' / 'tiles' / 'all', dirs_exist_ok=True)
            shutil.copytree(raw_path / 'images-sa/dataset-sa/holdout' / 'images',
                            output_path / 'oam_tcd_sa' / 'tiles' / 'all', dirs_exist_ok=True)

        # Getting metadata for 5-fold cross validation
        metadata = []
        with open(raw_path / 'meta' / 'train_meta.json', 'r') as f:
            for line in f:
                try:
                    json_object = json.loads(line.strip())
                    metadata.append(json_object)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON object on line {len(metadata) + 1}: {str(e)}")

        predefined_image_folds = {m['image_id']: m['validation_fold'] for m in metadata}

        # use the last fold (id=4) as validation
        with open(raw_path / 'train/dataset/holdout' / 'train.json', 'r') as f:
            train_coco = json.load(f)

        new_train_coco = {
            'licenses': train_coco['licenses'],
            'categories': train_coco['categories'],
            'images': [img for img in train_coco['images'] if predefined_image_folds[img['id']] != 4],
            'annotations': [ann for ann in train_coco['annotations'] if predefined_image_folds[ann['image_id']] != 4]
        }
        new_valid_coco = {
            'licenses': train_coco['licenses'],
            'categories': train_coco['categories'],
            'images': [img for img in train_coco['images'] if predefined_image_folds[img['id']] == 4],
            'annotations': [ann for ann in train_coco['annotations'] if predefined_image_folds[ann['image_id']] == 4]
        }

        train_fold_coco_name = coco_convention.create_name(product_name='oam_tcd', fold='train')
        valid_fold_coco_name = coco_convention.create_name(product_name='oam_tcd', fold='valid')

        # Save the train and valid COCO JSON files for the current fold
        print("Saving train and valid COCO JSON files...")
        with open(output_path / 'oam_tcd' / train_fold_coco_name, 'w') as f:
            json.dump(new_train_coco, f, ensure_ascii=False, indent=2)
        with open(output_path / 'oam_tcd' / valid_fold_coco_name, 'w') as f:
            json.dump(new_valid_coco, f, ensure_ascii=False, indent=2)

        if remove_tree_group_annotations:
            # Remove annotations for the 'canopy' category, as they are not single tree instances and would contradict other datasets
            print("Removing 'canopy' category annotations and blacking out associated pixels, and removing corrupted images...")
        else:
            print("Removing corrupted images...")

        # Moving tiles to separates /tiles/fold folders
        print("Distributing tiles into train/valid/test...")
        base_tiles = output_path / 'oam_tcd' / 'tiles'
        all_tiles = base_tiles / 'all'

        # helper list of (fold_name, coco_filename)
        splits = [
            ('train', train_fold_coco_name),
            ('valid', valid_fold_coco_name),
            ('test', coco_name_test)
        ]

        for split_name, coco_fname in splits:
            split_folder = base_tiles / split_name
            split_folder.mkdir(parents=True, exist_ok=True)

            # load the JSON for this split
            coco_split = json.load((output_path / 'oam_tcd' / coco_fname).open('r'))

            # move each listed image
            for img in coco_split['images']:
                src = all_tiles / img['file_name']
                dst = split_folder / img['file_name']
                if src.exists():
                    shutil.move(str(src), str(dst))
                else:
                    print(f"Warning: expected tile {src.name} for split '{split_name}' not found")

        # Remove the now empty all folder
        all_tiles.rmdir()

        # Train
        remove_corrupted_images_and_remove_category(
            coco_path=output_path / 'oam_tcd' / train_fold_coco_name,
            tiles_folder=output_path / 'oam_tcd' / 'tiles' / 'train',
            remove_category_name='canopy' if remove_tree_group_annotations else None,
            remove_empty_tiles=True
        )

        # Valid
        remove_corrupted_images_and_remove_category(
            coco_path=output_path / 'oam_tcd' / valid_fold_coco_name,
            tiles_folder=output_path / 'oam_tcd' / 'tiles' / 'valid',
            remove_category_name='canopy' if remove_tree_group_annotations else None,
            remove_empty_tiles=True
        )

        # Test
        remove_corrupted_images_and_remove_category(
            coco_path=output_path / 'oam_tcd' / coco_name_test,
            tiles_folder=output_path / 'oam_tcd' / 'tiles' / 'test',
            remove_category_name='canopy' if remove_tree_group_annotations else None,
            remove_empty_tiles=True
        )

        if use_tc_sa_licensed_data:
            # Test-nc
            remove_corrupted_images_and_remove_category(
                coco_path=output_path / 'oam_tcd_nc' / coco_name_test_nc,
                tiles_folder=output_path / 'oam_tcd_nc' / 'tiles' / 'all',
                remove_category_name='canopy' if remove_tree_group_annotations else None,
                remove_empty_tiles=True
            )

            # Test-sa
            remove_corrupted_images_and_remove_category(
                coco_path=output_path / 'oam_tcd_sa' / coco_name_test_sa,
                tiles_folder=output_path / 'oam_tcd_sa' / 'tiles' / 'all',
                remove_category_name='canopy' if remove_tree_group_annotations else None,
                remove_empty_tiles=True
            )

        # Cut valid and test images to the specified size (parameter test_and_valid_cut_image_size) and overlap (parameter test_and_valid_cut_image_overlap)
        _cut_and_filter_coco(
            coco_in_path=output_path / 'oam_tcd' / valid_fold_coco_name,
            tiles_in=output_path / 'oam_tcd' / 'tiles' / 'valid',
            tile_size=test_and_valid_cut_image_size,
            overlap=test_and_valid_cut_image_overlap,
            fold_name='valid'
        )

        _cut_and_filter_coco(
            coco_in_path=output_path / 'oam_tcd' / coco_name_test,
            tiles_in=output_path / 'oam_tcd' / 'tiles' / 'test',
            tile_size=test_and_valid_cut_image_size,
            overlap=test_and_valid_cut_image_overlap,
            fold_name='test'
        )

        if use_tc_sa_licensed_data:
            _cut_and_filter_coco(
                coco_in_path=output_path / 'oam_tcd_nc' / coco_name_test_nc,
                tiles_in=output_path / 'oam_tcd_nc' / 'tiles' / 'all',
                tile_size=test_and_valid_cut_image_size,
                overlap=test_and_valid_cut_image_overlap,
                fold_name='test'
            )

            _cut_and_filter_coco(
                coco_in_path=output_path / 'oam_tcd_sa' / coco_name_test_sa,
                tiles_in=output_path / 'oam_tcd_sa' / 'tiles' / 'all',
                tile_size=test_and_valid_cut_image_size,
                overlap=test_and_valid_cut_image_overlap,
                fold_name='test'
            )

        # rename tiles to follow convention for train folder
        train_json_path = output_path / 'oam_tcd' / train_fold_coco_name
        # load it
        with open(train_json_path, 'r') as f:
            coco_data = json.load(f)

        # iterate over every image entry
        for img in coco_data['images']:
            old_fn = img['file_name']
            new_fn = TileNameConvention().create_name(
                product_name=old_fn.replace('.tif', ''),
                aoi='train',
                scale_factor=1.0,
                col=0,
                row=0
            )

            src = output_path / 'oam_tcd' / 'tiles' / 'train' / old_fn
            dst = output_path / 'oam_tcd' / 'tiles' / 'train' / new_fn

            if src.exists():
                src.rename(dst)  # rename file on disk
                img['file_name'] = new_fn  # update JSON entry
            else:
                print(f"Warning: tile '{old_fn}' not found for renaming")

        # write the updated COCO JSON back out
        with open(train_json_path, 'w') as f:
            json.dump(coco_data, f, ensure_ascii=False, indent=2)

        # Done
        print("Done.")


def decode_rle(rle: Dict[str, Union[List[int], List[List[int]]]]) -> np.ndarray:
    """
    Decode RLE (Run Length Encoding) to binary mask.

    Args:
        rle: Dictionary containing 'counts' and 'size' for RLE encoding

    Returns:
        np.ndarray: Binary mask with correct orientation
    """
    if isinstance(rle['counts'], str):
        raise ValueError("Compressed RLE string format not supported")

    height, width = rle['size']
    mask = np.zeros(width * height, dtype=np.uint8)
    current_position = 0
    value = 0  # Start with 0

    for count in rle['counts']:
        mask[current_position:current_position + count] = value
        current_position += count
        value = 1 - value  # Toggle between 0 and 1

    # Reshape and transpose the mask to align coordinates correctly
    return mask.reshape(height, width).T


def is_image_corrupt(image_path: str) -> bool:
    """
    Check if an image file is corrupted.

    Args:
        image_path (str): Path to the image file

    Returns:
        bool: True if image is corrupted, False otherwise
    """
    # Check if file is too small (likely corrupted)
    if os.path.getsize(image_path) < 100:  # 100 bytes threshold
        return True

    # Try to open and verify the image
    try:
        with Image.open(image_path) as img:
            img.verify()
            if len(img.getbands()) < 3:
                return True

        return False
    except (UnidentifiedImageError, OSError, IOError):
        return True


def remove_corrupted_images_and_remove_category(
        coco_path: str or Path,
        tiles_folder: str or Path,
        remove_category_name: Optional[str] = None,
        remove_empty_tiles: bool = False
) -> None:
    """
    Process COCO annotations by removing specific categories and updating images in place.
    Also removes corrupted images from both filesystem and COCO annotations.

    Args:
        coco_path (str): Path to the COCO JSON file
        tiles_folder (str): Path to the folder containing tile images
        remove_category_name (Optional[str]): Name of the category to remove. If None, no category is removed
        remove_empty_tiles (bool): Whether to remove tiles with no annotations
    """
    # Load COCO JSON
    with open(coco_path, 'r') as f:
        coco_data = json.load(f)

    # Find category ID if category name is provided
    category_id = None
    if remove_category_name is not None:
        for category in coco_data['categories']:
            if category['name'] == remove_category_name:
                category_id = category['id']
                break

        if category_id is None:
            raise ValueError(f"Category '{remove_category_name}' not found in COCO file")

    # Create a mapping of image_id to file_name
    image_map = {img['id']: img for img in coco_data['images']}

    # Group annotations by image_id
    image_annotations: Dict[int, List[Dict]] = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)

    # Process each image and its annotations
    new_images = []
    new_annotations = []
    processed_image_ids = set()

    for img_id, annotations in tqdm(image_annotations.items(), desc=f"Processing images for coco {Path(coco_path).name}"):
        if img_id in processed_image_ids:
            continue

        processed_image_ids.add(img_id)
        img_info = image_map[img_id]
        file_name = img_info['file_name']
        img_path = os.path.join(tiles_folder, file_name)

        # Check if image exists and is not corrupted
        if not os.path.exists(img_path):
            print(f"\nWarning: Image {file_name} not found in tiles folder")
            continue

        if is_image_corrupt(img_path):
            print(f"\nWarning: Corrupted image found and will be removed: {file_name}")
            os.remove(img_path)
            continue

        # If no category to remove, just check for corruption and keep valid images
        if category_id is None:
            new_images.append(img_info)
            new_annotations.extend(annotations)
            continue

        # Filter out annotations of the specified category
        remaining_annotations = [ann for ann in annotations if ann['category_id'] != category_id]

        # If there was annotations originally, and no annotations remain and remove_empty_tiles is True, remove the image as its probably just mostly black pixels due to removing tree groups
        if not remaining_annotations and remove_empty_tiles:
            if os.path.exists(img_path):
                os.remove(img_path)
                print(f"\nWarning: Image {file_name} has no remaining annotations and will be removed.")
            continue

        try:
            with rasterio.open(img_path) as src:
                profile = src.profile.copy()
                # Read as (bands, H, W)
                data = src.read()
            # Convert to H×W×C for mask application
            img_array = np.transpose(data, (1, 2, 0))

            # Black out pixels for removed annotations
            removed_annotations = [ann for ann in annotations if ann['category_id'] == category_id]
            for ann in removed_annotations:
                segmentation = ann['segmentation']
                mask = np.zeros(img_array.shape[:2], dtype=np.uint8)

                # Handle different segmentation formats
                if isinstance(segmentation, dict):
                    # RLE format
                    mask = decode_rle(segmentation)
                else:
                    # Polygon format
                    for seg in segmentation:
                        if len(seg) > 0:  # Check if segment is not empty
                            points = np.array(seg).reshape(-1, 2)
                            points = np.round(points).astype(np.int32)
                            cv2.fillPoly(mask, [points], 1)

                # Apply black mask to image
                img_array[mask == 1] = 0

            # 3) Write back using Rasterio (lossless)
            # Convert back to (bands, H, W)
            new_data = np.transpose(img_array, (2, 0, 1))

            profile.update({
                'compress': 'deflate',  # lossless ZIP
                'predictor': 2,  # horizontal differencing for better compression
                'photometric': 'RGB',  # override YCbCr so Deflate is allowed
            })
            with rasterio.open(img_path, 'w', **profile) as dst:
                dst.write(new_data)

        except Exception as e:
            print(f"Error processing image {file_name}: {str(e)}")
            if os.path.exists(img_path):
                os.remove(img_path)
            continue

        # Keep track of remaining annotations and images
        if remaining_annotations or not remove_empty_tiles:
            new_annotations.extend(remaining_annotations)
            new_images.append(img_info)

    # Update COCO data
    coco_data['images'] = new_images
    coco_data['annotations'] = new_annotations

    # Fix categories IDs if a category was removed
    if remove_category_name is not None and category_id is not None:
        # Remove the specified category from the categories list
        coco_data['categories'] = [cat for cat in coco_data['categories'] if cat['id'] != category_id]

        # Decrement category IDs that are higher than the removed category ID
        for cat in coco_data['categories']:
            if cat['id'] > category_id:
                cat['id'] -= 1

        # Update category IDs in annotations accordingly
        for ann in coco_data['annotations']:
            if ann['category_id'] > category_id:
                ann['category_id'] -= 1

    #normalize all segmentations to compressed RLE strings
    image_map = {img['id']: img for img in coco_data['images']}

    for ann in tqdm(coco_data['annotations'], desc='Normalizing segmentations'):
        img_info = image_map[ann['image_id']]
        H, W = img_info['height'], img_info['width']
        seg = ann['segmentation']

        # 1) Decode into a full H×W mask:
        if isinstance(seg, dict):
            counts = seg['counts']
            # uncompressed list-of-int RLE
            if isinstance(counts, list):
                mask = decode_rle(seg)
            else:
                # compressed RLE (str or bytes)
                if isinstance(counts, str):
                    counts = counts.encode('utf-8')
                mask = mask_utils.decode({'counts': counts, 'size': [H, W]})
        else:
            # polygon list
            mask = np.zeros((H, W), dtype=np.uint8)
            for poly in seg:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [pts], 1)

        # 2) Re-encode to compressed RLE string form:
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle['counts'] = rle['counts'].decode('utf-8')
        rle['size'] = [H, W]
        ann['segmentation'] = rle

    # Save updated COCO JSON back to original location
    with open(coco_path, 'w') as f:
        json.dump(coco_data, f, indent=2)


def _cut_and_filter_coco(
    coco_in_path: Path,
    tiles_in: Path,
    tile_size: int,
    overlap: float,
    fold_name: str
):
    """
    Cut each large tile into smaller overlapping tiles, remove tiles that are
    >80% black or entirely white pixels, filter annotations by ≥40% overlap,
    display each resulting tile with its kept annotations, and finally replace
    the original tiles folder and COCO JSON with the new ones.
    """
    # compute step size
    step = int(tile_size * (1 - overlap))
    assert step > 0, "overlap must be < 1.0"

    # prepare temporary output directory
    tiles_out = tiles_in.parent / (tiles_in.name + '_cropped')
    if tiles_out.exists():
        shutil.rmtree(tiles_out)
    tiles_out.mkdir(parents=True)

    # load COCO
    coco = json.loads(coco_in_path.read_text())
    anns_by_image = {}
    for ann in coco['annotations']:
        anns_by_image.setdefault(ann['image_id'], []).append(ann)

    new_images = []
    new_annotations = []
    next_img_id = 1
    next_ann_id = 1

    for img in tqdm(coco['images'], desc=f"Tiling {coco_in_path.name}"):
        orig_w, orig_h = img['width'], img['height']
        img_annotations = anns_by_image.get(img['id'], [])

        # how many tiles in each direction
        nx = math.ceil((orig_w - tile_size) / step) + 1
        ny = math.ceil((orig_h - tile_size) / step) + 1

        for iy in range(ny):
            for ix in range(nx):
                x0 = min(ix * step, orig_w - tile_size)
                y0 = min(iy * step, orig_h - tile_size)
                x1, y1 = x0 + tile_size, y0 + tile_size

                # read the tile window losslessly
                with rasterio.open(tiles_in / img['file_name']) as src:
                    profile = src.profile.copy()
                    window = Window(x0, y0, tile_size, tile_size)
                    data = src.read(window=window)  # shape: (bands, H, W)
                    transform = src.window_transform(window)

                tile_anns = []
                for ann in img_annotations:
                    # decode the full‐image mask once
                    full_mask = coco_rle_segmentation_to_mask(ann['segmentation'])
                    orig_area = int(full_mask.sum())
                    if orig_area == 0:
                        continue

                    # crop mask to tile
                    tile_mask = full_mask[y0:y1, x0:x1]
                    inter_area = int(tile_mask.sum())

                    # require ≥40% of the true object mask
                    if inter_area / orig_area < 0.4:
                        continue

                    # build new annotation
                    new_ann = {
                        **{k: ann[k] for k in ('iscrowd', 'category_id')},
                        'id': next_ann_id,
                        'image_id': next_img_id,
                        # bbox = tight bounds of the tiled mask
                        'bbox': (lambda m: [
                            int(np.where(np.any(m, axis=0))[0][0]),
                            int(np.where(np.any(m, axis=1))[0][0]),
                            int(np.ptp(np.where(np.any(m, axis=0)))),
                            int(np.ptp(np.where(np.any(m, axis=1))))
                        ])(tile_mask),
                        'area': inter_area
                    }

                    # encode the cropped mask back to RLE
                    r = mask_utils.encode(np.asfortranarray(tile_mask))
                    r['counts'] = r['counts'].decode('utf-8')
                    r['size'] = [tile_size, tile_size]
                    new_ann['segmentation'] = r

                    new_annotations.append(new_ann)
                    tile_anns.append(new_ann)
                    next_ann_id += 1

                # skip if no annotations remain
                if len(tile_anns) == 0:
                    print(f"Skipping tile {img['file_name']} at ({x0}, {y0}) due to no annotations")
                    continue

                # convert to H×W×C for pixel checks
                arr = np.transpose(data, (1, 2, 0))
                total_pixels = arr.shape[0] * arr.shape[1]
                black = np.all(arr == 0, axis=2).sum()
                white = np.all(arr == 255, axis=2).sum()

                # skip if >80% black or all white
                if black / total_pixels > 0.8 or white == total_pixels:
                    print(
                        f"Skipping tile due to excessive black or white pixels "
                        f"(black={black/total_pixels:.2%}, white={white/total_pixels:.2%})")
                    continue

                # save new tile as GeoTIFF with Deflate compression
                new_name = TileNameConvention().create_name(
                    product_name=img['file_name'].replace('.tif', ''),
                    aoi=fold_name,
                    scale_factor=1.0,
                    col=ix,
                    row=iy
                )

                profile.update({
                    'height': tile_size,
                    'width': tile_size,
                    'transform': transform,
                    'photometric': 'RGB',
                    'compress': 'deflate',
                    'predictor': 2,
                })
                with rasterio.open(tiles_out / new_name, 'w', **profile) as dst:
                    dst.write(data)

                # register new image
                new_images.append({
                    **{k: img[k] for k in ('license', 'coco_url') if k in img},
                    'id': next_img_id,
                    'width': tile_size,
                    'height': tile_size,
                    'file_name': new_name
                })

                next_img_id += 1

    # write new COCO JSON
    new_coco = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': coco['categories'],
        'licenses': coco.get('licenses', [])
    }
    temp_json = tiles_out / 'annotations.json'
    temp_json.write_text(json.dumps(new_coco, indent=2))

    # first replace the COCO JSON
    shutil.move(str(temp_json), str(coco_in_path))

    # then replace the tiles folder
    shutil.rmtree(tiles_in)
    shutil.move(str(tiles_out), str(tiles_in))
