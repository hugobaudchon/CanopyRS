#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import copy

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


def has_valid_segmentation(seg):
    """
    Return True if a segmentation is considered valid (non-empty).
    Handles both polygon coords and RLE dicts.
    """
    if not seg:
        return False

    # RLE dict (shouldn't normally be present after conversion, but treat as valid)
    if isinstance(seg, dict):
        return True

    # Polygon coordinates
    if isinstance(seg, list):
        if len(seg) == 0:
            return False

        # Could be flat list [x0,y0,...] or list-of-polygons [[...], ...]
        if all(isinstance(x, (int, float)) for x in seg):
            return len(seg) >= 6  # at least 3 points
        else:
            for poly in seg:
                if isinstance(poly, (list, tuple)) and len(poly) >= 6:
                    return True
            return False

    return False


def convert_all_coco_files(root_dir, num_workers=None, min_area=None):
    """
    Recursively find and convert all RLE segmentations to polygons in COCO JSON files.

    If min_area is not None, annotations with area < min_area are removed
    from each JSON file. After processing each file, the number of removed
    annotations is logged as: "removed X / Y".

    For files that are "segmentation COCO" (i.e., not all annotations are
    box-only), after conversion and min-area filtering, annotations whose
    segmentation is empty/invalid are removed. The original and converted
    segmentations are logged for these removed annotations.
    """
    root_dir = Path(root_dir)
    json_files = list(root_dir.rglob("*.json"))

    if not json_files:
        print("No JSON files found.")
        return

    print(f"Found {len(json_files)} JSON files")

    # Load all JSONs and collect tasks per-annotation
    coco_data_list = []  # list of (json_path, coco_data, is_segmentation_coco)
    tasks = []

    for file_idx, json_path in enumerate(json_files):
        with open(json_path, "r") as f:
            coco_data = json.load(f)

        anns = coco_data.get("annotations", [])

        # Determine if this file is "segmentation COCO"
        # i.e., NOT all annotations are pure boxes with no segmentation
        is_segmentation_coco = False
        for ann_idx, ann in enumerate(anns):
            seg = ann.get("segmentation")
            # Store original segmentation in memory for later logging
            ann["_orig_segmentation"] = copy.deepcopy(seg)

            if seg:  # any non-empty segmentation means it's not pure box-only
                is_segmentation_coco = True

            if seg and is_rle_format(seg):
                tasks.append((file_idx, ann_idx, ann))

        coco_data_list.append((json_path, coco_data, is_segmentation_coco))

    # ---- RLE → polygon conversion ----
    if tasks:
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
                json_path, coco_data, is_segmentation_coco = coco_data_list[file_idx]
                coco_data["annotations"][ann_idx]["segmentation"] = new_seg
                num_converted += 1

        print(f"\nConverted {num_converted} RLE annotations across {len(json_files)} files.")
    else:
        print("No RLE segmentations found. Skipping conversion step.")

    # ---- Area filtering (min_area) ----
    total_removed_area = 0
    if min_area is not None:
        print(f"\nFiltering annotations with area < {min_area} ...")

    new_coco_data_list = []
    for (json_path, coco_data, is_segmentation_coco) in coco_data_list:
        anns = coco_data.get("annotations", [])
        before = len(anns)

        if min_area is not None and before > 0:
            filtered = []
            for ann in anns:
                area_val = ann.get("area", 0.0)
                try:
                    area_val = float(area_val)
                except (TypeError, ValueError):
                    area_val = 0.0

                if area_val >= min_area:
                    filtered.append(ann)

            removed = before - len(filtered)
            total_removed_area += removed
            if removed > 0:
                print(
                    f"File '{json_path}': removed {removed} / {before} "
                    f"annotations with area < {min_area}"
                )
            coco_data["annotations"] = filtered

        new_coco_data_list.append((json_path, coco_data, is_segmentation_coco))

    coco_data_list = new_coco_data_list

    # ---- Segmentation cleanup for segmentation COCO files ----
    total_removed_seg = 0
    print("\nCleaning up annotations with invalid/empty segmentation in segmentation COCO files...")
    for (json_path, coco_data, is_segmentation_coco) in coco_data_list:
        if not is_segmentation_coco:
            # Detection-only (bbox-only) COCO: nothing to do here
            continue

        anns = coco_data.get("annotations", [])
        before = len(anns)
        if before == 0:
            continue

        filtered = []
        removed_here = 0

        for ann in anns:
            seg = ann.get("segmentation")
            if not has_valid_segmentation(seg):
                removed_here += 1
                total_removed_seg += 1

                orig_seg = ann.get("_orig_segmentation", None)
                ann_id = ann.get("id", None)

                print(
                    f"[WARN] Removing annotation with invalid/empty segmentation in "
                    f"file '{json_path}', ann_id={ann_id}"
                )
                print("Original segmentation:")
                try:
                    print(f"{json.dumps(orig_seg)}")
                except Exception:
                    print(f"{orig_seg}")
                print("Converted segmentation:")
                try:
                    print(f"{json.dumps(seg)}")
                except Exception:
                    print(f"{seg}")

                continue

            filtered.append(ann)

        if removed_here > 0:
            print(
                f"File '{json_path}': removed {removed_here} / {before} "
                f"annotations with invalid/empty segmentation"
            )

        coco_data["annotations"] = filtered

    # ---- Strip helper keys and write back ----
    for json_path, coco_data, _ in coco_data_list:
        anns = coco_data.get("annotations", [])
        for ann in anns:
            if "_orig_segmentation" in ann:
                del ann["_orig_segmentation"]

        with open(json_path, "w") as f:
            json.dump(coco_data, f)

    if min_area is not None:
        print(f"\nTotal annotations removed by area filter across all files: {total_removed_area}")
    print(f"Total annotations removed due to invalid/empty segmentation: {total_removed_seg}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert RLE segmentations to polygon coordinates in COCO JSON files, "
            "optionally removing annotations below a minimum area, and for "
            "segmentation COCO files, dropping annotations with invalid/empty segmentation."
        )
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
    parser.add_argument(
        "--min-area",
        type=float,
        default=10.0,
        help="If set, annotations with area < MIN_AREA are removed from each COCO JSON.",
    )

    args = parser.parse_args()
    convert_all_coco_files(args.root_dir, args.workers, min_area=args.min_area)
