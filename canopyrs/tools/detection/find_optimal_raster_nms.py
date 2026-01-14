#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from canopyrs.engine.benchmark.detector.benchmark import DetectorBenchmarker
from canopyrs.engine.config_parsers import DetectorConfig, AggregatorConfig

def parse_args():
    parser = argparse.ArgumentParser(
        description="Search optimal NMS IoU & score thresholds on the validation fold"
    )
    parser.add_argument(
        "-c", "--detector_config",
        type=Path,
        required=True,
        help="Path to your detector YAML config"
    )
    parser.add_argument(
        "-d", "--datasets",
        nargs="+",
        required=True,
        help="List of dataset names to include in the search"
    )
    parser.add_argument(
        "-r", "--data_root",
        type=Path,
        default=Path.home() / "Documents" / "CanopyRS" / "extracted_datasets",
        help="Root folder where datasets are extracted"
    )
    parser.add_argument(
        "-o", "--output_folder",
        type=Path,
        required=True,
        help="Where to write intermediate outputs and logs"
    )
    parser.add_argument(
        "--iou_thresholds",
        nargs="+",
        type=float,
        default=None,
        help="Custom IoU thresholds (e.g. 0.1 0.2 …). Defaults to 0.05–1.00 step 0.05"
    )
    parser.add_argument(
        "--score_thresholds",
        nargs="+",
        type=float,
        default=None,
        help="Custom score thresholds (e.g. 0.1 0.2 …). Defaults to 0.05–1.00 step 0.05"
    )
    parser.add_argument(
        "--fold_name",
        type=str,
        default="valid",
        help="Which fold to benchmark on (default: valid)"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=6,
        help="Number of workers for parallel NMS processing on CPU."
    )
    parser.add_argument(
        "--eval_iou_threshold",
        type=str,
        default="0.75",
        help="IoU threshold for raster metrics (e.g., 0.75 for RF1_75 or '50:95' for RF1_50:95)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Handle IoU threshold parsing (single or 50:95 sweep)
    if str(args.eval_iou_threshold).lower() == "50:95":
        eval_iou_threshold = [round(t, 2) for t in [0.50 + 0.05 * i for i in range(10)]]
    else:
        try:
            # Allow comma-separated list for convenience
            if "," in str(args.eval_iou_threshold):
                eval_iou_threshold = [float(x) for x in str(args.eval_iou_threshold).split(",")]
            else:
                eval_iou_threshold = float(args.eval_iou_threshold)
        except ValueError:
            raise ValueError("Invalid eval_iou_threshold. Use a float, comma-separated floats, or '50:95'.")

    # build default threshold grids if none provided
    if args.iou_thresholds is None:
        args.iou_thresholds = [i / 20 for i in range(1, 21)]
    if args.score_thresholds is None:
        args.score_thresholds = [i / 20 for i in range(1, 21)]

    cfg = DetectorConfig.from_yaml(str(args.detector_config))
    out = args.output_folder.resolve()
    out.mkdir(parents=True, exist_ok=True)

    bench = DetectorBenchmarker(
        output_folder=str(out),
        fold_name=args.fold_name,
        raw_data_root=str(args.data_root.resolve()),
        eval_iou_threshold=eval_iou_threshold,
    )

    aggregator_config = AggregatorConfig(
        nms_algorithm='iou',
    )

    try:
        optimal_aggregator_config = bench.find_optimal_nms_iou_threshold(
            detector_config=cfg,
            base_aggregator_config=aggregator_config,
            dataset_names=args.datasets,
            nms_iou_thresholds=args.iou_thresholds,
            nms_score_thresholds=args.score_thresholds,
            n_workers=args.n_workers
        )
        print(f"Best NMS IoU Threshold: {optimal_aggregator_config.nms_threshold}")
        print(f"Best Score Threshold:  {optimal_aggregator_config.score_threshold}")
        
        # Save the optimal aggregator config to a YAML file
        config_output_path = out / "optimal_aggregator_config.yaml"
        with open(config_output_path, 'w') as f:
            yaml.dump(optimal_aggregator_config.model_dump(), f, default_flow_style=False, sort_keys=False)
        print(f"\nOptimal aggregator config saved to: {config_output_path}")
    except Exception as e:
        print(f"Error during NMS search: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
