"""Every segmentation in the SelvaMask COCO files decodes to a valid polygon inside its image."""

import json

import pytest
from shapely.geometry import box

from canopyrs1.core.geometry.coco_segmentation import decode_segmentation

pytestmark = pytest.mark.integration


def _annotations(dataset_root):
    """Yield (image, annotation) for every annotation of every SelvaMask COCO file."""
    paths = sorted((dataset_root / "selvamask").glob("*/*.json"))
    assert paths, f"No COCO file found under {dataset_root / 'selvamask'}"
    for path in paths:
        coco = json.loads(path.read_text())
        images = {image["id"]: image for image in coco["images"]}
        for annotation in coco["annotations"]:
            yield images[annotation["image_id"]], annotation


def test_every_segmentation_decodes(selvamask_dataset):
    problems, n = [], -1
    for n, (image, annotation) in enumerate(_annotations(selvamask_dataset)):
        segmentation, height, width = annotation["segmentation"], image["height"], image["width"]
        where = f"annotation {annotation['id']} of {image['file_name']}"
        polygon = decode_segmentation(segmentation)
        if not polygon.is_valid or polygon.is_empty:
            problems.append(f"{where}: not a valid, non-empty polygon")
            continue
        if not box(0, 0, width, height).covers(polygon):
            problems.append(f"{where}: reaches outside its {width} x {height} image")
        if not decode_segmentation(segmentation, "box").equals(box(*polygon.bounds)):
            problems.append(f"{where}: its box isn't the polygon's bounds")
        if n % 50 == 0:                                   # masks are slow: check a sample
            mask = decode_segmentation(segmentation, "mask", height=height, width=width)
            if mask.shape != (height, width) or not mask.any():
                problems.append(f"{where}: bad mask {mask.shape}, {mask.sum()} pixels")
    assert n >= 0 and not problems, "\n".join(problems[:20])
