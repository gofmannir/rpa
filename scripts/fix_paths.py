#!/usr/bin/env python3
"""Fix paths in dataset_split.json for remote VM execution.

Replaces local macOS paths with VM paths.
"""

import argparse
import json
import sys
from pathlib import Path

# Path mappings
LOCAL_DATASET_PATH = "/Users/nirgofman/Desktop/running-pattern/final_dataset"
REMOTE_DATASET_PATH = "/data/final_dataset"


def fix_paths(input_file: Path, output_file: Path | None = None) -> None:
    """Replace local paths with remote VM paths.

    Args:
        input_file: Path to dataset_split.json
        output_file: Output path (defaults to overwriting input)
    """
    if output_file is None:
        output_file = input_file

    print(f"Reading: {input_file}")

    with input_file.open() as f:
        data = json.load(f)

    # Count replacements
    replacements = 0

    # Fix paths in train, val, test lists
    for split in ["train", "val", "test"]:
        if split in data:
            for i, path in enumerate(data[split]):
                if LOCAL_DATASET_PATH in path:
                    data[split][i] = path.replace(LOCAL_DATASET_PATH, REMOTE_DATASET_PATH)
                    replacements += 1

    print(f"Replaced {replacements} paths")
    print(f"  FROM: {LOCAL_DATASET_PATH}")
    print(f"  TO:   {REMOTE_DATASET_PATH}")

    with output_file.open("w") as f:
        json.dump(data, f, indent=2)

    print(f"Written: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix paths in dataset_split.json for VM")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset_split.json"),
        help="Input JSON file (default: dataset_split.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file (default: overwrite input)",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Run on remote VM (use /app/rpa paths)",
    )

    args = parser.parse_args()

    # Adjust input path if running remotely
    if args.remote:
        args.input = Path("/app/rpa/dataset_split.json")
        args.output = args.input

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    fix_paths(args.input, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
