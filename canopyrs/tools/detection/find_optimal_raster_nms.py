#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np

from canopyrs.engine.benchmark.detector.benchmark import DetectorBenchmarker
from canopyrs.engine.config_parsers import DetectorConfig

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
    return parser.parse_args()


def main():
    args = parse_args()

    # build default threshold grids if none provided
    if args.iou_thresholds is None:
        args.iou_thresholds = [round(x, 1) for x in np.arange(0.05, 1.05, 0.05)]
    if args.score_thresholds is None:
        args.score_thresholds = [round(x, 2) for x in np.arange(0.05, 1.05, 0.05)]

    cfg = DetectorConfig.from_yaml(str(args.detector_config))
    out = args.output_folder.resolve()
    out.mkdir(parents=True, exist_ok=True)

    bench = DetectorBenchmarker(
        output_folder=str(out),
        fold_name=args.fold_name,
        raw_data_root=str(args.data_root.resolve()),
    )

    try:
        best_iou, best_score = bench.find_optimal_nms_iou_threshold(
            detector_config=cfg,
            dataset_names=args.datasets,
            nms_iou_thresholds=args.iou_thresholds,
            nms_score_thresholds=args.score_thresholds,
            n_workers=args.n_workers
        )
        print(f"Best NMS IoU Threshold: {best_iou}")
        print(f"Best Score Threshold:  {best_score}")
    except Exception as e:
        print(f"Error during NMS search: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
