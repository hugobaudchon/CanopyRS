#!/usr/bin/env python3
import argparse
from pathlib import Path

from canopyrs.engine.benchmark.detector.benchmark import DetectorBenchmarker
from canopyrs.engine.config_parsers import DetectorConfig, AggregatorConfig

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run final detector benchmark on a given fold (e.g. test)"
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
        help="List of dataset names to benchmark"
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
        help="Where to write benchmark results"
    )
    parser.add_argument(
        "--fold_name",
        type=str,
        default="test",
        help="Which fold to benchmark on (default: test)"
    )
    parser.add_argument(
        "--nms_threshold",
        type=float,
        default=None,
        help="Use this NMS IoU threshold (skip search tool if provided)"
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=None,
        help="Use this score threshold (skip search tool if provided)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = DetectorConfig.from_yaml(str(args.detector_config))
    out = args.output_folder.resolve()
    out.mkdir(parents=True, exist_ok=True)

    bench = DetectorBenchmarker(
        output_folder=str(out),
        fold_name=args.fold_name,
        raw_data_root=str(args.data_root.resolve()),
    )

    agg_cfg = None
    if args.nms_threshold is not None and args.score_threshold is not None:
        agg_cfg = AggregatorConfig(
            nms_threshold=args.nms_threshold,
            score_threshold=args.score_threshold,
            nms_algorithm='iou',
        )

    bench.benchmark(
        detector_config=cfg,
        aggregator_config=agg_cfg,
        dataset_names=args.datasets,
    )
    print(f"Benchmark completed. Results in {out}")


if __name__ == "__main__":
    main()
