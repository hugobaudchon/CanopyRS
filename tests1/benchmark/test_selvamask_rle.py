"""On the SelvaMask masks, the fast RLE encoding gives exactly the RLE of the whole mask."""

import json

import pytest

from canopyrs1.core.geometry.coco_segmentation import decode_segmentation, encode_segmentation

pytestmark = pytest.mark.integration


def test_fast_rle_is_the_same_on_selvamask(selvamask_dataset):
    paths = sorted((selvamask_dataset / "selvamask").glob("*/*.json"))
    assert paths, f"No COCO file found under {selvamask_dataset / 'selvamask'}"
    different, n = [], 0
    for path in paths:
        coco = json.loads(path.read_text())
        images = {image["id"]: image for image in coco["images"]}
        for annotation in coco["annotations"][::20]:  # the whole mask is slow: a sample
            image = images[annotation["image_id"]]
            polygon = decode_segmentation(annotation["segmentation"])
            size = {"height": image["height"], "width": image["width"]}
            fast = encode_segmentation(polygon, rle=True, **size)
            if fast != encode_segmentation(polygon, rle=True, fast=False, **size):
                different.append(f"annotation {annotation['id']} of {path.name}")
            n += 1
    assert n > 0 and not different, "\n".join(different[:20])
