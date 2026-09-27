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
