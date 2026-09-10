from canopyrs.engine.benchmark.base import BaseBenchmarker
from canopyrs.engine.config_parsers import (AggregatorConfig, ClassifierConfig, DetectorConfig,
                                            PipelineConfig, TilerizerConfig)


class DetectorBenchmarker(BaseBenchmarker):
    """
    Run detector (and optional aggregator) benchmarks at tile and raster level, and search NMS/aggregation params.
    Extends BaseBenchmarker with detection-specific functionality.
    
    All common functionality is inherited from BaseBenchmarker, with public methods
    that have detector-specific signatures.
    """
    
    def find_optimal_nms_iou_threshold(self,
                                       detector_config: DetectorConfig,
                                       base_aggregator_config: AggregatorConfig,
                                       dataset_names: list[str],
                                       nms_iou_thresholds: list[float],
                                       nms_score_thresholds: list[float],
                                       eval_at_ground_resolution: float = 0.045,
                                       n_workers: int = 6,
                                       classifier_config: ClassifierConfig = None,
                                       classifier_crop_tilerizer_config: TilerizerConfig = None):
        """
        Find the optimal NMS IoU threshold for the detector by evaluating different thresholds on the validation set.

        Args:
            detector_config: Configuration for the detector
            base_aggregator_config: Base aggregator configuration (its nms_threshold/score_threshold will be overwritten by the search)
            dataset_names: List of dataset names to evaluate on
            nms_iou_thresholds: List of NMS IoU thresholds to try
            nms_score_thresholds: List of score thresholds to try
            eval_at_ground_resolution: Ground resolution for evaluation
            n_workers: Number of parallel workers
            classifier_config: Optional — classify every detection before the searched aggregation,
                so the search optimizes the same pipeline benchmark() will run (required together
                with classifier_crop_tilerizer_config, a tile_type='polygon' tilerizer)
            classifier_crop_tilerizer_config: Polygon tilerizer config for the classifier's crops

        Returns:
            AggregatorConfig: Optimal aggregator config with best nms_threshold and score_threshold set
        """
        steps = self._with_classifier_if_provided([('detector', detector_config)],
                                      classifier_config, classifier_crop_tilerizer_config)
        pipeline_config = PipelineConfig(components_configs=steps)

        return self._find_optimal_nms_iou_threshold(
            pipeline_config=pipeline_config,
            component_name='detector',
            iou_type='bbox',
            aggregator_config=base_aggregator_config,
            dataset_names=dataset_names,
            nms_iou_thresholds=nms_iou_thresholds,
            nms_score_thresholds=nms_score_thresholds,
            eval_at_ground_resolution=eval_at_ground_resolution,
            n_workers=n_workers
        )
    
    def benchmark(self,
                  detector_config: DetectorConfig,
                  aggregator_config: AggregatorConfig,
                  dataset_names: str | list[str],
                  classifier_config: ClassifierConfig = None,
                  classifier_crop_tilerizer_config: TilerizerConfig = None):
        """
        Runs the detector on the entire test dataset, recording both tile-level and raster-level metrics for each
        individual product and also aggregated for each dataset (which are made of 1 or more products).

        Args:
            detector_config: Configuration for the detector
            aggregator_config: Configuration for the aggregator
            dataset_names: Dataset name(s) to benchmark on
            classifier_config: Optional — classify every detection before aggregation
                (polygon tilerizer + classifier inserted before the aggregator), adding class-aware
                tile-level metrics; survivors carry their class into the raster-level GPKG.
            classifier_crop_tilerizer_config: Polygon tilerizer config for the classifier's crops
                (required with classifier_config).
        """
        model_components = self._with_classifier_if_provided([('detector', detector_config)],
                                                 classifier_config, classifier_crop_tilerizer_config)

        return self._benchmark(
            model_components=model_components,
            aggregator_config=aggregator_config,
            component_name='detector',
            iou_type='bbox',
            dataset_names=dataset_names
        )


