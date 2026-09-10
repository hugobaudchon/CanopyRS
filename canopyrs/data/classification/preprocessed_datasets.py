"""Classification dataset registry — separate from the detection one: a classification dataset is
folders of per-object crops with a label per image, not tiled rasters.

Each dataset yields, per fold, one entry per product: ``(location, product_name, crops_dir, truth)``
where ``crops_dir`` is a folder of pre-cut crop images and ``truth`` is a ``{file_name: class}``
mapping or a CSV path with ``file_name`` and ``class`` columns. The ClassifierBenchmarker runs a
classifier-only pipeline over each product and scores it against the truth.
"""

from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Type, Union


class BaseClassifierPreprocessedDataset:
    dataset_name: str = None

    def verify_dataset(self, root_output_path: Union[str, Path], folds: List[str]) -> None:
        """Check (and download if supported) the dataset under ``root_output_path`` for ``folds``."""
        raise NotImplementedError

    def iter_fold(self, root_output_path: Union[str, Path], fold: str) -> Iterator[Tuple[
        str,                      # location
        str,                      # product_name
        Path,                     # crops_dir: folder of pre-cut crop images
        Union[dict, str, Path],   # truth: {file_name: class} or a CSV path (file_name, class)
    ]]:
        raise NotImplementedError


DATASET_REGISTRY: Dict[str, Type[BaseClassifierPreprocessedDataset]] = {}


def register_dataset(
        cls: Type[BaseClassifierPreprocessedDataset]
) -> Type[BaseClassifierPreprocessedDataset]:
    name = getattr(cls, "dataset_name", None)
    if not name:
        raise ValueError(f"{cls.__name__} must define a dataset_name")
    DATASET_REGISTRY[name] = cls
    return cls

# Concrete datasets register themselves here (@register_dataset) as they are curated.
