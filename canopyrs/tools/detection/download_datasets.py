import argparse
import sys
from pathlib import Path

from canopyrs.data.detection.preprocessed_datasets import DATASET_REGISTRY


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and verify one or more datasets from DATASET_REGISTRY"
    )
    parser.add_argument(
        "-d", "--datasets",
        nargs="+",
        required=True,
        choices=sorted(DATASET_REGISTRY.keys()),
        help="Name(s) of dataset(s) to download"
    )
    parser.add_argument(
        "-o", "--output_root",
        type=Path,
        default=Path.home() / "Documents" / "CanopyRS" / "extracted_datasets",
        help="Root output directory"
    )
    parser.add_argument(
        "-f", "--folds",
        nargs="+",
        default=["train", "valid", "test"],
        help="Fold names to download (default: train valid test)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root_output = args.output_root.resolve()
    root_output.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        print(f"\nProcessing dataset '{name}'")
        DatasetClass = DATASET_REGISTRY.get(name)
        if DatasetClass is None:
            print(f"Unknown dataset '{name}'", file=sys.stderr)
            continue

        ds = DatasetClass()
        try:
            print(f"Downloading & extracting to {root_output}")
            ds.download_and_extract(root_output_path=str(root_output), folds=args.folds)

            print(f"Verifying dataset integrity...")
            ds.verify_dataset(root_output_path=str(root_output), folds=args.folds)

            print(f"Dataset '{name}' downloaded and verified successfully.")
        except Exception as e:
            print(f"Error with '{name}': {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
