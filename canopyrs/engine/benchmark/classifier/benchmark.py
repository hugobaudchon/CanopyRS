from pathlib import Path
from typing import Optional, Union, List

import pandas as pd

from canopyrs.engine.benchmark.base.base_benchmarker import BaseBenchmarker
from canopyrs.engine.benchmark.classifier.evaluator import ClassifierCocoEvaluator
from canopyrs.engine.config_parsers import ClassifierConfig, PipelineConfig, TilerizerConfig
from canopyrs.engine.utils import merge_coco_jsons
from canopyrs.data.classification.preprocessed_datasets import DATASET_REGISTRY

PathLike = Union[str, Path]


class ClassifierBenchmarker(BaseBenchmarker):
    def __init__(self,
                 output_folder: PathLike,
                 fold_name: str,
                 raw_data_root: PathLike,
                 eval_iou_threshold: Union[float, List[float]] = 0.75):
        super().__init__(
            output_folder=output_folder,
            fold_name=fold_name,
            raw_data_root=raw_data_root,
            eval_iou_threshold=eval_iou_threshold,
        )

    def _get_preprocessed_datasets(
        self,
        dataset_names: Union[str, List[str]],
        pipeline_outputs_root: Optional[PathLike] = None,
    ):
        datasets = {}
        dataset_names_list = (
            [dataset_names]
            if isinstance(dataset_names, str)
            else list(dataset_names)
        )
        resolved_root = (
            Path(pipeline_outputs_root)
            if pipeline_outputs_root is not None
            else None
        )
        for dataset_name in dataset_names_list:
            assert dataset_name in DATASET_REGISTRY, (
                f'Dataset {dataset_name} not supported. '
                f'Supported: {DATASET_REGISTRY.keys()}.'
            )
            datasets[dataset_name] = DATASET_REGISTRY[dataset_name]()

            if (
                resolved_root is not None
                and hasattr(datasets[dataset_name],
                            'pipeline_outputs_root')
            ):
                datasets[dataset_name].pipeline_outputs_root = (
                    resolved_root
                )

            datasets[dataset_name].verify_dataset(
                root_output_path=self.raw_data_root,
                folds=[self.fold_name],
            )
        return datasets

    @staticmethod
    def _flatten_instance_segmentation_results(results: dict) -> dict:
        flat = {}
        for section_name, section in results.items():
            if not isinstance(section, dict):
                continue
            for k, v in section.items():
                flat[f"{section_name}_{k}"] = v
        return flat

    def _infer_classifier_single_product(
            self,
            product_name: str,
            product_tiles_path: Optional[PathLike],
            classifier_config: ClassifierConfig,
            input_gpkg: Optional[PathLike],
            input_coco: Optional[PathLike],
            tilerizer_config: Optional[TilerizerConfig] = None,
            input_imagery: Optional[PathLike] = None,
            output_folder: Optional[PathLike] = None,
    ):
        if tilerizer_config is not None:
            pipeline_config = PipelineConfig(
                components_configs=[
                    ('tilerizer', tilerizer_config),
                    ('classifier', classifier_config),
                ]
            )
        else:
            pipeline_config = PipelineConfig(
                components_configs=[
                    ('classifier', classifier_config),
                ]
            )

        model_coco, model_gpkg, _, tilerizer_coco = (
            self._infer_single_product(
                product_name=product_name,
                product_tiles_path=product_tiles_path,
                pipeline_config=pipeline_config,
                component_name='classifier',
                input_gpkg=input_gpkg,
                input_coco=input_coco,
                input_imagery=input_imagery,
                output_folder=output_folder,
            )
        )
        return model_coco, model_gpkg, tilerizer_coco

    def benchmark_single_run(
            self,
            run_name: str,
            product_tiles_path: Optional[PathLike],
            classifier_config: ClassifierConfig,
            truth_coco_path: PathLike,
            input_gpkg: Optional[PathLike],
            input_coco: Optional[PathLike],
            tilerizer_config: Optional[TilerizerConfig] = None,
            input_imagery: Optional[PathLike] = None,
    ):
        evaluator = ClassifierCocoEvaluator(verbose=True)

        run_output_folder = self.output_folder / self.fold_name / run_name
        run_output_folder.mkdir(parents=True, exist_ok=True)

        preds_coco_path, _, tilerizer_coco = (
            self._infer_classifier_single_product(
                product_name=run_name,
                product_tiles_path=product_tiles_path,
                classifier_config=classifier_config,
                input_gpkg=input_gpkg,
                input_coco=input_coco,
                tilerizer_config=tilerizer_config,
                input_imagery=input_imagery,
                output_folder=run_output_folder,
            )
        )

        # When tilerizer ran, its output COCO (with GT labels)
        # serves as ground truth for alignment.
        effective_truth = (
            str(tilerizer_coco)
            if tilerizer_coco is not None
            else str(truth_coco_path)
        )

        results = evaluator.tile_level(
            preds_coco_path=str(preds_coco_path),
            truth_coco_path=effective_truth,
            evaluate_class_agnostic=True,
        )

        record = self._flatten_instance_segmentation_results(results)
        record['run_name'] = run_name

        metrics_df = pd.DataFrame([record])
        metrics_file = run_output_folder / "tile_level_metrics.csv"
        metrics_df.to_csv(metrics_file, index=False)
        return metrics_df

    def _iter_fold_classifier(self, dataset, fold: str):
        if not hasattr(dataset, 'iter_fold_classifier'):
            raise NotImplementedError(
                'Dataset must implement iter_fold_classifier() '
                'for classifier benchmarking.'
            )
        return dataset.iter_fold_classifier(self.raw_data_root, fold=fold)

    def _resolve_tilerizer_config(
            self,
            classifier_config: ClassifierConfig,
    ) -> Optional[TilerizerConfig]:
        if classifier_config.tilerizer_config_path is not None:
            return TilerizerConfig.from_yaml(
                path=classifier_config.tilerizer_config_path,
            )
        return None

    def benchmark(
            self,
            classifier_config: ClassifierConfig,
            dataset_names: Union[str, List[str]],
    ):
        tilerizer_config = self._resolve_tilerizer_config(
            classifier_config,
        )

        classification_df = self._benchmark_classification_only(
            classifier_config=classifier_config,
            dataset_names=dataset_names,
            pipeline_outputs_root=(
                classifier_config.pipeline_outputs_root
            ),
            tilerizer_config=tilerizer_config,
            input_imagery=classifier_config.input_imagery,
        )

        instance_df = self._benchmark_instance_segmentation(
            classifier_config=classifier_config,
            dataset_names=dataset_names,
            pipeline_outputs_root=(
                classifier_config.pipeline_outputs_root
            ),
            tilerizer_config=tilerizer_config,
            input_imagery=classifier_config.input_imagery,
        )

        return classification_df, instance_df

    def _benchmark_classification_only(
            self,
            classifier_config: ClassifierConfig,
            dataset_names: Union[str, List[str]],
            pipeline_outputs_root: Optional[PathLike] = None,
            tilerizer_config: Optional[TilerizerConfig] = None,
            input_imagery: Optional[PathLike] = None,
    ):
        datasets = self._get_preprocessed_datasets(
            dataset_names,
            pipeline_outputs_root=pipeline_outputs_root,
        )
        evaluator = ClassifierCocoEvaluator(verbose=True)

        all_metrics = []
        for dataset_name, dataset in datasets.items():
            dataset_preds_cocos = []
            dataset_truths_cocos = []

            for (
                location,
                product_name,
                tiles_path,
                input_gpkg,
                input_coco,
                truths_coco,
            ) in self._iter_fold_classifier(dataset, fold=self.fold_name):
                product_output_folder = (
                    self.output_folder
                    / self.fold_name
                    / 'tile_predictions'
                    / product_name
                )
                preds_coco_json, _, tilerizer_coco = (
                    self._infer_classifier_single_product(
                        product_name=product_name,
                        product_tiles_path=tiles_path,
                        classifier_config=classifier_config,
                        input_gpkg=input_gpkg,
                        input_coco=input_coco,
                        tilerizer_config=tilerizer_config,
                        input_imagery=input_imagery,
                        output_folder=product_output_folder,
                    )
                )

                effective_truth = (
                    str(tilerizer_coco)
                    if tilerizer_coco is not None
                    else str(truths_coco)
                )
                metrics = evaluator.classification_only(
                    preds_coco_path=str(preds_coco_json),
                    truth_coco_path=effective_truth,
                )
                metrics['location'] = location
                metrics['product_name'] = product_name
                all_metrics.append(metrics)

                dataset_preds_cocos.append(str(preds_coco_json))
                dataset_truths_cocos.append(effective_truth)

            merged_preds_coco_path = (
                self.output_folder
                / self.fold_name
                / f"{dataset_name}_merged_preds_coco.json"
            )
            merged_truths_coco_path = (
                self.output_folder
                / self.fold_name
                / f"{dataset_name}_merged_truths_coco.json"
            )
            merge_coco_jsons(dataset_preds_cocos, merged_preds_coco_path)
            merge_coco_jsons(dataset_truths_cocos, merged_truths_coco_path)

            dataset_metrics = evaluator.classification_only(
                preds_coco_path=str(merged_preds_coco_path),
                truth_coco_path=str(merged_truths_coco_path),
            )
            dataset_metrics['location'] = dataset_name
            dataset_metrics['product_name'] = 'average_over_rasters'
            all_metrics.append(dataset_metrics)

        metrics_file = (
            self.output_folder
            / self.fold_name
            / 'classification_only_metrics.csv'
        )
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(metrics_file, index=False)
        return metrics_df

    def _benchmark_instance_segmentation(
            self,
            classifier_config: ClassifierConfig,
            dataset_names: Union[str, List[str]],
            pipeline_outputs_root: Optional[PathLike] = None,
            tilerizer_config: Optional[TilerizerConfig] = None,
            input_imagery: Optional[PathLike] = None,
    ):
        datasets = self._get_preprocessed_datasets(
            dataset_names,
            pipeline_outputs_root=pipeline_outputs_root,
        )
        evaluator = ClassifierCocoEvaluator(verbose=True)

        all_metrics = []
        for dataset_name, dataset in datasets.items():
            dataset_preds_cocos = []
            dataset_truths_cocos = []

            for (location, product_name, tiles_path, input_gpkg, input_coco,
                 truths_coco) in self._iter_fold_classifier(dataset, fold='test'):
                preds_coco_json, _, tilerizer_coco = (
                    self._infer_classifier_single_product(
                        product_name=product_name,
                        product_tiles_path=tiles_path,
                        classifier_config=classifier_config,
                        input_gpkg=input_gpkg,
                        input_coco=input_coco,
                        tilerizer_config=tilerizer_config,
                        input_imagery=input_imagery,
                        output_folder=(
                            self.output_folder
                            / self.fold_name
                            / 'tile_predictions'
                            / product_name
                        ),
                    )
                )

                effective_truth = (
                    str(tilerizer_coco)
                    if tilerizer_coco is not None
                    else str(truths_coco)
                )
                results = evaluator.tile_level(
                    preds_coco_path=str(preds_coco_json),
                    truth_coco_path=effective_truth,
                    evaluate_class_agnostic=True,
                )

                record = self._flatten_instance_segmentation_results(results)
                record['location'] = location
                record['product_name'] = product_name
                all_metrics.append(record)

                dataset_preds_cocos.append(str(preds_coco_json))
                dataset_truths_cocos.append(effective_truth)

            merged_preds_coco_path = self.output_folder / self.fold_name / f"{dataset_name}_merged_preds_coco.json"
            merged_truths_coco_path = self.output_folder / self.fold_name / f"{dataset_name}_merged_truths_coco.json"
            merge_coco_jsons(dataset_preds_cocos, merged_preds_coco_path)
            merge_coco_jsons(dataset_truths_cocos, merged_truths_coco_path)

            dataset_results = evaluator.tile_level(
                preds_coco_path=str(merged_preds_coco_path),
                truth_coco_path=str(merged_truths_coco_path),
                evaluate_class_agnostic=True,
            )
            dataset_record = self._flatten_instance_segmentation_results(dataset_results)
            dataset_record['location'] = dataset_name
            dataset_record['product_name'] = 'average_over_rasters'
            all_metrics.append(dataset_record)

        metrics_file = (
            self.output_folder
            / self.fold_name
            / 'instance_segmentation_metrics.csv'
        )
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(metrics_file, index=False)
        return metrics_df
