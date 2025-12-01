import json
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from geodataset.utils import decode_coco_segmentation, polygon_to_coco_coordinates_segmentation


def is_rle_format(segmentation):
    """Check if segmentation is in RLE format."""
    return isinstance(segmentation, dict) and "counts" in segmentation


def _convert_one_annotation(task):
    """
    Worker function: convert a single annotation if it's RLE.

    task: (file_idx, ann_idx, ann_dict)
    returns: (file_idx, ann_idx, new_segmentation) or None
    """
    file_idx, ann_idx, ann = task
    seg = ann.get("segmentation")
    if not seg or not is_rle_format(seg):
        return None

    polygon = decode_coco_segmentation(ann, "polygon")
    new_seg = polygon_to_coco_coordinates_segmentation(polygon)
    return file_idx, ann_idx, new_seg


def convert_all_coco_files(root_dir, num_workers=None):
    """Recursively find and convert all RLE segmentations to polygons in COCO JSON files."""
    root_dir = Path(root_dir)
    json_files = list(root_dir.rglob("*.json"))

    if not json_files:
        print("No JSON files found.")
        return

    print(f"Found {len(json_files)} JSON files")

    # Load all JSONs and collect tasks per-annotation
    coco_data_list = []
    tasks = []

    for file_idx, json_path in enumerate(json_files):
        with open(json_path, "r") as f:
            coco_data = json.load(f)
        coco_data_list.append((json_path, coco_data))

        anns = coco_data.get("annotations", [])
        for ann_idx, ann in enumerate(anns):
            if "segmentation" in ann and is_rle_format(ann["segmentation"]):
                tasks.append((file_idx, ann_idx, ann))

    if not tasks:
        print("No RLE segmentations found. Nothing to do.")
        return

    print(f"Found {len(tasks)} RLE annotations to convert")

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    print(f"Using {num_workers} workers\n")

    num_converted = 0
    with Pool(num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(_convert_one_annotation, tasks),
            total=len(tasks),
            desc="Converting annotations",
        ):
            if result is None:
                continue
            file_idx, ann_idx, new_seg = result
            json_path, coco_data = coco_data_list[file_idx]
            coco_data["annotations"][ann_idx]["segmentation"] = new_seg
            num_converted += 1

    # Write back all JSON files
    for json_path, coco_data in coco_data_list:
        with open(json_path, "w") as f:
            json.dump(coco_data, f)

    print(f"\nDone! Converted {num_converted} annotations across {len(json_files)} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert RLE segmentations to polygon coordinates in COCO JSON files"
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="Root directory to search for JSON files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count - 1)",
    )

    args = parser.parse_args()
    convert_all_coco_files(args.root_dir, args.workers)
