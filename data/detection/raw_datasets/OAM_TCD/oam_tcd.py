import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Union, Optional

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from geodataset.utils import CocoNameConvention, create_coco_folds
from pycocotools import mask as maskUtils
from tqdm import tqdm

from dataset.detection.raw_datasets.base_dataset import BasePublicZipDataset


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
                 cross_validation: bool,
                 ground_resolution: float = None,
                 scale_factor: float = None,
                 tile_size: int = 1024,
                 tile_overlap: float = 0.5):

        raw_path = Path(raw_path)
        output_path = Path(output_path) / self.name
        output_path.mkdir(parents=True, exist_ok=True)

        if not raw_path.name == self.name:
            raw_path = raw_path / self.name

        assert raw_path.exists(), (f"Path {raw_path} does not exist."
                                   f" Make sure you called the download AND parse methods first.")

        # Copy the test data already tilerized
        (output_path / 'oam_tcd').mkdir(exist_ok=True)
        (output_path / 'oam_tcd_nc').mkdir(exist_ok=True)
        (output_path / 'oam_tcd_sa').mkdir(exist_ok=True)

        # Copy annotations
        coco_convention = CocoNameConvention()
        coco_name_test = coco_convention.create_name(product_name='oam_tcd', fold='test')
        shutil.copy2(raw_path / 'test/dataset/holdout' / 'test.json',
                     output_path / 'oam_tcd' / coco_name_test)
        coco_name_test_nc = coco_convention.create_name(product_name='oam_tcd_nc', fold='test')
        shutil.copy2(raw_path / 'test-nc/dataset-nc/holdout' / 'test.json',
                     output_path / 'oam_tcd_nc' / coco_name_test_nc)
        coco_name_test_sa = coco_convention.create_name(product_name='oam_tcd_sa', fold='test')
        shutil.copy2(raw_path / 'test-sa/dataset-sa/holdout' / 'test.json',
                     output_path / 'oam_tcd_sa' / coco_name_test_sa)

        # Copy images
        shutil.copytree(raw_path / 'images/dataset/holdout' / 'images',
                        output_path / 'oam_tcd' / 'tiles' / 'all', dirs_exist_ok=True)
        shutil.copytree(raw_path / 'images-nc/dataset-nc/holdout' / 'images',
                        output_path / 'oam_tcd_nc' / 'tiles' / 'all', dirs_exist_ok=True)
        shutil.copytree(raw_path / 'images-sa/dataset-sa/holdout' / 'images',
                        output_path / 'oam_tcd_sa' / 'tiles' / 'all', dirs_exist_ok=True)

        # Now taking care of training annotations
        coco_name_train = coco_convention.create_name(product_name='oam_tcd', fold='train')

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

        if cross_validation:
            shutil.copy2(raw_path / 'train/dataset/holdout' / 'train.json',
                         output_path / 'oam_tcd' / coco_name_train)
            create_coco_folds(
                output_path / 'oam_tcd' / coco_name_train,
                output_path / 'oam_tcd',
                5,
                predefined_image_folds=predefined_image_folds
            )
        else:
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
            with open(output_path / 'oam_tcd' / train_fold_coco_name, 'w') as f:
                json.dump(new_train_coco, f, ensure_ascii=False, indent=2)
            with open(output_path / 'oam_tcd' / valid_fold_coco_name, 'w') as f:
                json.dump(new_valid_coco, f, ensure_ascii=False, indent=2)

            if remove_tree_group_annotations:
                # Remove annotations for the 'canopy' category, as they are not single tree instances and would contradict other datasets
                print("Removing 'canopy' category annotations and blacking out associated pixels, and removing corrupted images...")
            else:
                print("Removing corrupted images...")
            # Train
            remove_corrupted_images_and_remove_category(
                coco_path=output_path / 'oam_tcd' / train_fold_coco_name,
                tiles_folder=output_path / 'oam_tcd' / 'tiles' / 'all',
                remove_category_name='canopy' if remove_tree_group_annotations else None,
                remove_empty_tiles=True
            )

            # Valid
            remove_corrupted_images_and_remove_category(
                coco_path=output_path / 'oam_tcd' / valid_fold_coco_name,
                tiles_folder=output_path / 'oam_tcd' / 'tiles' / 'all',
                remove_category_name='canopy' if remove_tree_group_annotations else None,
                remove_empty_tiles=True
            )

            # Test
            remove_corrupted_images_and_remove_category(
                coco_path=output_path / 'oam_tcd' / coco_name_test,
                tiles_folder=output_path / 'oam_tcd' / 'tiles' / 'all',
                remove_category_name='canopy' if remove_tree_group_annotations else None,
                remove_empty_tiles=True
            )

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

        # If no annotations remain and remove_empty_tiles is True, remove the image
        if not remaining_annotations and remove_empty_tiles:
            if os.path.exists(img_path):
                os.remove(img_path)
            continue

        try:
            with Image.open(img_path) as img:
                original_format = img.format
                original_mode = img.mode
                img_array = np.array(img)

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

            # Convert back to PIL Image
            processed_img = Image.fromarray(img_array)

            # Ensure the image is in the correct mode
            if processed_img.mode != original_mode:
                processed_img = processed_img.convert(original_mode)

            # Prepare save parameters for JPEG compression within TIFF
            save_kwargs = {
                'compression': 'jpeg',
                'quality': 85,
                'subsampling': 0,  # 4:4:4 subsampling for best quality
                'dpi': img.info.get('dpi', (300, 300)),  # Preserve original DPI if available
            }

            # Save processed image with JPEG compression
            processed_img.save(img_path, format=original_format, **save_kwargs)

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

    # Save updated COCO JSON back to original location
    with open(coco_path, 'w') as f:
        json.dump(coco_data, f, indent=2)