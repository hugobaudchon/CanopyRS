from pathlib import Path

from canopyrs.engine.benchmark.base_benchmarker import BaseBenchmarker
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
                                       dataset_names: list[str],
                                       nms_iou_thresholds: list[float],
                                       nms_score_thresholds: list[float],
                                       eval_at_ground_resolution: float = 0.045,
                                       n_workers: int = 6,
                                       nms_algorithm: str = 'ioa_disambiguate',
                                       prompter_detector_config: DetectorConfig = None,
                                       detector_score_weight: float = 0.5,
                                       segmenter_score_weight: float = 0.5,
                                       edge_band_buffer_percentage: float = 0.05,
                                       best_geom_keep_area_ratio: float = 0.5):
        """
        Find the optimal NMS IoU threshold for the segmenter by evaluating different thresholds on the validation set.
        
        Args:
            segmenter_config: Configuration for the segmenter
            dataset_names: List of dataset names to evaluate on
            nms_iou_thresholds: List of NMS IoU thresholds to try
            nms_score_thresholds: List of score thresholds to try
            eval_at_ground_resolution: Ground resolution for evaluation
            n_workers: Number of parallel workers
            nms_algorithm: NMS algorithm to use ('ioa_disambiguate' by default for segmentation)
            prompter_detector_config: Optional detector config to chain before segmenter (default None)
            detector_score_weight: Weight for detector score (default 0.5, used when prompter_detector_config is provided)
            segmenter_score_weight: Weight for segmenter score (default 1.0)
            edge_band_buffer_percentage: Edge band buffer percentage (default 0.05)
            best_geom_keep_area_ratio: Area ratio threshold (default 0.5)
        
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
        
        # Build aggregator config with segmenter-specific defaults
        aggregator_config = AggregatorConfig(
            nms_algorithm=nms_algorithm,
            detector_score_weight=detector_score_weight,
            segmenter_score_weight=segmenter_score_weight,
            edge_band_buffer_percentage=edge_band_buffer_percentage,
            best_geom_keep_area_ratio=best_geom_keep_area_ratio,
        )
        
        return self._find_optimal_nms_iou_threshold(
            pipeline_config=pipeline_config,
            component_name='segmenter',
            iou_type='segm',
            aggregator_config=aggregator_config,
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
