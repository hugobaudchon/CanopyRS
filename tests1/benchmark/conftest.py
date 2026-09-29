"""
Fixtures for the tests on real benchmark data.

The SelvaMask valid and test folds (~3.3 GB) are downloaded on first use and cached in
~/.cache/canopyrs_test_data/benchmarks/, shared with every run afterwards.
"""

from pathlib import Path

import pytest

BENCHMARK_DATA = Path.home() / ".cache" / "canopyrs_test_data" / "benchmarks"


@pytest.fixture(scope="session")
def selvamask_dataset():
    """Return the folder holding the SelvaMask valid and test folds, downloading them first if
    they aren't cached yet. Inside it, selvamask/<raster>/ holds one COCO file per fold and the
    tiles in tiles/<fold>/."""
    from canopyrs.data.detection.preprocessed_datasets import DATASET_REGISTRY

    root = BENCHMARK_DATA / "datasets"
    location = root / "selvamask"
    if any(location.glob("**/test/*.tif")) and any(location.glob("**/valid/*.tif")):
        return root

    root.mkdir(parents=True, exist_ok=True)
    dataset = DATASET_REGISTRY["SelvaMask"]()
    dataset.download_and_extract(root_output_path=str(root), folds=["valid", "test"])
    dataset.verify_dataset(root_output_path=str(root), folds=["valid", "test"])
    return root


SELVAMASK_GPKGS = [
    "20240131_zf2block4_ms_m3m_labels_masks.gpkg",
    "20240613_tbsnewsite2_m3e_labels_masks.gpkg",
    "20241122_bcifairchildn_m3m_rgb_labels_masks.gpkg",
]


@pytest.fixture(scope="session")
def selvamask_gpkgs():
    """Return the paths of the three SelvaMask ground-truth GeoPackages (the tree crowns of each
    raster), downloading them into the Hugging Face cache first if they aren't there yet."""
    from huggingface_hub import hf_hub_download

    return [
        Path(
            hf_hub_download(
                repo_id="CanopyRS/SelvaMask",
                repo_type="dataset",
                revision="gpkg",
                filename=filename,
            )
        )
        for filename in SELVAMASK_GPKGS
    ]
