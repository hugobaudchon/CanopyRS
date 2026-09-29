"""The SelvaMask COCO files, read and written again, keep their images, categories and
annotations, with their bbox and area recomputed from their segmentations."""

import json
import warnings

import pytest

from canopyrs1.core.constants import Col
from canopyrs1.core.geometry.coco_segmentation import decode_segmentation
from canopyrs1.core.io.coco import read_coco, write_coco

pytestmark = pytest.mark.integration


def _coco_files(dataset_root):
    """Return (COCO file, its tiles folder) for every SelvaMask COCO file."""
    paths = sorted((dataset_root / "selvamask").glob("*/*.json"))
    assert paths, f"No COCO file found under {dataset_root / 'selvamask'}"
    return [(path, path.parent / "tiles" / path.stem.rsplit("_", 1)[1]) for path in paths]


def _read_and_write(path, tiles_dir, out, categories):
    """Read the COCO file at ``path``, write it again at ``out``, and return the new file."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # SelvaMask's own bbox and area: see test below
        _, objects = read_coco(path, tiles_dir)
    write_coco(objects, out, categories=categories, category_column=Col.CATEGORY_NAME)
    return json.loads(out.read_text())


def test_read_then_written_again(selvamask_dataset, tmp_path):
    for path, tiles_dir in _coco_files(selvamask_dataset):
        original = json.loads(path.read_text())
        once = _read_and_write(path, tiles_dir, tmp_path / "once.json", original["categories"])

        # The same images and categories, and the same annotations in the same order, renumbered.
        assert once["images"] == original["images"], path.name
        assert once["categories"] == original["categories"]
        assert len(once["annotations"]) == len(original["annotations"])
        image_ids = [a["image_id"] for a in original["annotations"]]
        assert [a["image_id"] for a in once["annotations"]] == image_ids
        for new, old in zip(once["annotations"], original["annotations"]):
            assert new["segmentation"] == old["segmentation"], (path.name, old["id"])
            assert (new["category_id"], new["iscrowd"]) == (old["category_id"], old["iscrowd"])
            # bbox and area are those of the segmentation, not the file's.
            polygon = decode_segmentation(new["segmentation"])
            minx, miny, maxx, maxy = polygon.bounds
            assert new["bbox"] == [minx, miny, maxx - minx, maxy - miny]
            assert new["area"] == polygon.area

        # Read and written a second time: exactly the first file.
        twice = _read_and_write(
            tmp_path / "once.json",
            tiles_dir,
            tmp_path / "twice.json",
            original["categories"],
        )
        assert twice == once


def test_selvamask_s_own_boxes_and_areas_are_reported(selvamask_dataset):
    # Its segmentations were traced half a pixel inside the crowns, its boxes and areas weren't.
    path, tiles_dir = _coco_files(selvamask_dataset)[0]
    with pytest.warns(UserWarning, match="aren't those of their segmentation: the areas differ"):
        read_coco(path, tiles_dir)
