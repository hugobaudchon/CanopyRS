from pathlib import Path

from canopyrs.engine.benchmark.base.base_benchmarker import BaseBenchmarker
from canopyrs.engine.config_parsers import SegmenterConfig, DetectorConfig, AggregatorConfig, PipelineConfig


class SegmenterBenchmarker(BaseBenchmarker):
    """
    Run segmenter benchmarks at tile and raster level.
    Extends BaseBenchmarker with segmentation-specific functionality.
    
    All common functionality is inherited from BaseBenchmarker, with public methods
    that have segmenter-specific signatures.
    """
    
    def find_optimal_nms_iou_threshold(self,
                                       segmenter_config: SegmenterConfig,
                                       base_aggregator_config: AggregatorConfig,
                                       dataset_names: list[str],
                                       nms_iou_thresholds: list[float],
                                       nms_score_thresholds: list[float],
                                       eval_at_ground_resolution: float = 0.045,
                                       n_workers: int = 6,
                                       prompter_detector_config: DetectorConfig = None):
        """
        Find the optimal NMS IoU threshold for the segmenter by evaluating different thresholds on the validation set.
        
        Args:
            segmenter_config: Configuration for the segmenter
            base_aggregator_config: Base aggregator configuration (its nms_threshold/score_threshold will be overwritten by the search)
            dataset_names: List of dataset names to evaluate on
            nms_iou_thresholds: List of NMS IoU thresholds to try
            nms_score_thresholds: List of score thresholds to try
            eval_at_ground_resolution: Ground resolution for evaluation
            n_workers: Number of parallel workers
            prompter_detector_config: Optional detector config to chain before segmenter (default None)
        
        Returns:
            AggregatorConfig: Optimal aggregator config with best nms_threshold and score_threshold set
        """
        # Build pipeline config with optional detector chained before segmenter
        if prompter_detector_config is not None:
            pipeline_config = PipelineConfig(components_configs=[
                ('detector', prompter_detector_config),
                ('segmenter', segmenter_config)
            ])
        else:
            pipeline_config = PipelineConfig(components_configs=[
                ('segmenter', segmenter_config)
            ])

        return self._find_optimal_nms_iou_threshold(
            pipeline_config=pipeline_config,
            component_name='segmenter',
            iou_type='segm',
            aggregator_config=base_aggregator_config,
            dataset_names=dataset_names,
            nms_iou_thresholds=nms_iou_thresholds,
            nms_score_thresholds=nms_score_thresholds,
            eval_at_ground_resolution=eval_at_ground_resolution,
            n_workers=n_workers
        )
    
    def benchmark(self,
                  segmenter_config: SegmenterConfig,
                  aggregator_config: AggregatorConfig,
                  dataset_names: str | list[str],
                  prompter_detector_config: DetectorConfig = None):
        """
        Runs the segmenter on the entire test dataset, recording both tile-level and raster-level metrics for each
        individual product and also aggregated for each dataset (which are made of 1 or more products).
        
        Args:
            segmenter_config: Configuration for the segmenter
            aggregator_config: Configuration for the aggregator
            dataset_names: Dataset name(s) to benchmark on
            prompter_detector_config: Optional detector config to chain before segmenter (default None)
        """
        # Build pipeline config with optional detector chained before segmenter
        if prompter_detector_config is not None:
            pipeline_config = PipelineConfig(components_configs=[
                ('detector', prompter_detector_config),
                ('segmenter', segmenter_config),
                ('aggregator', aggregator_config)
            ])
        else:
            pipeline_config = PipelineConfig(components_configs=[
                ('segmenter', segmenter_config),
                ('aggregator', aggregator_config)
            ])
        
        return self._benchmark(
            pipeline_config_with_aggregator=pipeline_config,
            component_name='segmenter',
            iou_type='segm',
            dataset_names=dataset_names
        )
