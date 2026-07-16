"""Pure-classification benchmark (1 image = 1 class): run a classifier-only pipeline over folders of
pre-cut crops and score the per-image predictions against a truth mapping.

Two entry points: ``benchmark`` over registered classification datasets (multiple locations/products,
per-product + pooled per-dataset metrics, like the detection benchmarkers), and
``benchmark_single_run`` for one crops folder + truth mapping.

For class-aware detection/segmentation benchmarking (classifying every detection), use the
``classifier_config`` option of the Detector/Segmenter benchmarkers instead.
"""

import json
from pathlib import Path

import pandas as pd

from canopyrs.data.classification.preprocessed_datasets import DATASET_REGISTRY
from canopyrs.engine.benchmark.classifier.evaluator import classification_metrics
from canopyrs.engine.config_parsers import ClassifierConfig
from canopyrs.engine.constants import Col
from canopyrs.engine.data import Objects
from canopyrs.engine.pipeline import Pipeline


class ClassifierBenchmarker:
    """Type-2 benchmark: one crop image, one class. No COCO, no IoU — accuracy, per-class P/R/F1
    and a confusion matrix from {image file name: class}."""

    def __init__(self, output_folder: str | Path, fold_name: str = 'test',
                 raw_data_root: str | Path = None):
        """``fold_name`` / ``raw_data_root`` are only needed for the dataset-based ``benchmark``."""
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.fold_name = fold_name
        self.raw_data_root = Path(raw_data_root) if raw_data_root is not None else None
        assert fold_name in ['test', 'valid'], f'Fold {fold_name} not supported. Supported folds are "test" and "valid".'

    def benchmark(self,
                  classifier_config: ClassifierConfig,
                  dataset_names: str | list[str]) -> pd.DataFrame:
        """Run the classifier over every product of the given registered classification dataset(s),
        recording per-product metrics and pooled per-dataset metrics (``average_over_rasters`` rows,
        computed over the union of all products' images). Writes ``classification_metrics.csv`` and
        returns it as a DataFrame."""
        assert self.raw_data_root is not None, "dataset benchmarking needs raw_data_root"
        names = [dataset_names] if isinstance(dataset_names, str) else list(dataset_names)
        rows = []
        for dataset_name in names:
            assert dataset_name in DATASET_REGISTRY, \
                f'Dataset {dataset_name} not supported. Supported: {list(DATASET_REGISTRY)}.'
            dataset = DATASET_REGISTRY[dataset_name]()
            dataset.verify_dataset(self.raw_data_root, [self.fold_name])

            pooled_truth, pooled_preds = {}, {}
            for location, product_name, crops_dir, truth in dataset.iter_fold(self.raw_data_root, self.fold_name):
                truth_map, preds, metrics = self._run_product(
                    classifier_config, crops_dir, truth,
                    self.output_folder / self.fold_name / product_name)
                rows.append({'location': location, 'product_name': product_name, **self._flat(metrics)})
                # pool across products, prefixing to keep same-named crop files apart
                pooled_truth.update({f"{product_name}/{k}": v for k, v in truth_map.items()})
                pooled_preds.update({f"{product_name}/{k}": v for k, v in preds.items()})

            if pooled_truth:
                pooled = classification_metrics(pooled_truth, pooled_preds)
                rows.append({'location': dataset_name, 'product_name': 'average_over_rasters',
                             **self._flat(pooled)})

        metrics_df = pd.DataFrame(rows)
        metrics_file = self.output_folder / self.fold_name / 'classification_metrics.csv'
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Classification metrics saved to {metrics_file}")
        return metrics_df

    def benchmark_single_run(self,
                             classifier_config: ClassifierConfig,
                             crops_folder: str | Path,
                             truth: dict | str | Path,
                             run_name: str = 'run') -> dict:
        """Classify every image in ``crops_folder`` and score against ``truth`` — a
        ``{file_name: class}`` dict, or a CSV path with ``file_name`` and ``class`` columns.
        Classes are compared as given: use class names in ``truth`` when the config sets
        ``class_names``, class indices otherwise. Returns the metrics dict."""
        _, _, metrics = self._run_product(classifier_config, crops_folder, truth,
                                          self.output_folder / run_name)
        return metrics

    def _run_product(self, classifier_config, crops_folder, truth, output_folder):
        """One classifier-only pipeline over one crops folder: returns (truth_map, preds, metrics)
        and writes the product's ``classification_metrics.json``."""
        truth_map = self._load_truth(truth)
        pipe = Pipeline.from_config(
            [('classifier', classifier_config)],
            tiles=str(crops_folder),
            output_dir=str(output_folder),
        )
        pipe.run(verbose=False)
        preds = self._predictions_by_file(pipe, use_names=bool(classifier_config.class_names))

        metrics = classification_metrics(truth_map, preds)
        metrics_file = Path(output_folder) / 'classification_metrics.json'
        metrics_file.write_text(json.dumps(metrics, indent=2, default=str))
        return truth_map, preds, metrics

    @staticmethod
    def _load_truth(truth) -> dict:
        if isinstance(truth, dict):
            return truth
        df = pd.read_csv(truth)
        if not {'file_name', 'class'} <= set(df.columns):
            raise ValueError(f"{truth} must have 'file_name' and 'class' columns")
        return dict(zip(df['file_name'], df['class']))

    @staticmethod
    def _predictions_by_file(pipe: Pipeline, use_names: bool) -> dict:
        """{crop file name: predicted class} from the run's classified Objects (one per image)."""
        objects = pipe.latest(Objects)
        imagery = objects.linked("imagery")
        image_id = objects.column(Col.IMAGE_ID)   # via the lineage (the derived one-object-per-crop)
        paths = image_id.map(imagery.df.set_index(Col.IMAGE_ID)[Col.PATH])
        col = Col.CLASSIFIER_CLASS_NAME if use_names else Col.CLASSIFIER_CLASS
        return {Path(p).name: c for p, c in zip(paths, objects.df[col])}

    @staticmethod
    def _flat(metrics: dict) -> dict:
        """Metrics as CSV-safe cells: scalars kept, nested per-class/confusion JSON-stringified."""
        return {key: (json.dumps(value, default=str) if isinstance(value, dict) else value)
                for key, value in metrics.items()}
