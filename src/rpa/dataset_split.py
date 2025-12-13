"""Dataset splitting module with runner-based stratification.

Splits dataset by runner ID (not randomly) to prevent data leakage.
Outputs a JSON file with train/val/test paths - does NOT move any files.

The Golden Rule: Never put the same runner in both train and test sets,
or the model will memorize physical appearance instead of learning biomechanics.
"""

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from loguru import logger
from pydantic import BaseModel, Field

from rpa.dataset_stats import DatasetStats, VideoMetadata, compute_dataset_stats

# Constants
RATIO_TOLERANCE = 0.001
MIN_RUNNERS_FOR_VAL = 3
MIN_RUNNERS_FOR_TWO_SPLITS = 2


class SplitConfig(BaseModel):
    """Configuration for dataset splitting."""

    train_ratio: Annotated[float, Field(ge=0.0, le=1.0)] = 0.7
    val_ratio: Annotated[float, Field(ge=0.0, le=1.0)] = 0.15
    test_ratio: Annotated[float, Field(ge=0.0, le=1.0)] = 0.15
    seed: int = 42
    stratify_by_label: bool = True  # Try to balance labels across splits

    def __post_init__(self) -> None:
        """Validate ratios sum to 1.0."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > RATIO_TOLERANCE:
            msg = f"Ratios must sum to 1.0, got {total}"
            raise ValueError(msg)


@dataclass
class SplitResult:
    """Result of dataset splitting."""

    train_paths: list[str]
    val_paths: list[str]
    test_paths: list[str]

    train_runners: list[str]
    val_runners: list[str]
    test_runners: list[str]

    # Statistics per split
    train_label_counts: dict[int, int]
    val_label_counts: dict[int, int]
    test_label_counts: dict[int, int]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "train": self.train_paths,
            "val": self.val_paths,
            "test": self.test_paths,
            "metadata": {
                "train_runners": self.train_runners,
                "val_runners": self.val_runners,
                "test_runners": self.test_runners,
                "train_label_counts": self.train_label_counts,
                "val_label_counts": self.val_label_counts,
                "test_label_counts": self.test_label_counts,
                "total_train": len(self.train_paths),
                "total_val": len(self.val_paths),
                "total_test": len(self.test_paths),
            },
        }

    def save_json(self, output_path: Path) -> None:
        """Save split to JSON file."""
        with output_path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved split to {path}", path=output_path)


def _group_runners_by_label(stats: DatasetStats) -> dict[int, list[str]]:
    """Group runners by their primary label.

    Each runner should have only one label (their foot strike pattern).
    If a runner has multiple labels, use the most frequent one.

    Returns:
        Dict mapping label -> list of runner IDs
    """
    label_to_runners: dict[int, list[str]] = defaultdict(list)

    for runner in stats.unique_runners:
        runner_labels = stats.label_by_runner[runner]
        # Get the dominant label for this runner
        primary_label = runner_labels.most_common(1)[0][0]
        label_to_runners[primary_label].append(runner)

    return dict(label_to_runners)


def _split_runners_for_label(
    runners: list[str],
    config: SplitConfig,
) -> tuple[list[str], list[str], list[str]]:
    """Split runners for a single label into train/val/test.

    Ensures at least 1 runner per split when possible.

    Returns:
        Tuple of (train_runners, val_runners, test_runners)
    """
    n = len(runners)
    n_train = max(1, int(n * config.train_ratio))
    n_val = max(1, int(n * config.val_ratio)) if n > MIN_RUNNERS_FOR_TWO_SPLITS else 0

    # Ensure at least 1 runner per split if we have enough runners
    if n >= MIN_RUNNERS_FOR_VAL:
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            n_train = n - n_val - n_test
    else:
        n_test = 0
        if n == MIN_RUNNERS_FOR_TWO_SPLITS:
            n_train, n_val = 1, 1
        else:
            n_train, n_val = 1, 0

    train = runners[:n_train]
    val = runners[n_train : n_train + n_val]
    test = runners[n_train + n_val :]

    return train, val, test


def split_by_runner(
    stats: DatasetStats,
    config: SplitConfig | None = None,
) -> SplitResult:
    """Split dataset by runner ID with optional label stratification.

    Args:
        stats: Computed dataset statistics
        config: Split configuration (ratios, seed, etc.)

    Returns:
        SplitResult with train/val/test paths and metadata
    """
    if config is None:
        config = SplitConfig()

    random.seed(config.seed)

    # Group runners by their label for stratified splitting
    label_to_runners = _group_runners_by_label(stats)

    train_runners: list[str] = []
    val_runners: list[str] = []
    test_runners: list[str] = []

    if config.stratify_by_label:
        # Stratified split: ensure each label is represented in all splits
        for label, runners in label_to_runners.items():
            random.shuffle(runners)
            train, val, test = _split_runners_for_label(runners, config)

            train_runners.extend(train)
            val_runners.extend(val)
            test_runners.extend(test)

            logger.debug(
                "Label {l}: {t} train, {v} val, {te} test runners",
                l=label,
                t=len(train),
                v=len(val),
                te=len(test),
            )
    else:
        # Simple split without stratification
        all_runners = list(stats.unique_runners)
        random.shuffle(all_runners)

        n = len(all_runners)
        n_train = int(n * config.train_ratio)
        n_val = int(n * config.val_ratio)

        train_runners = all_runners[:n_train]
        val_runners = all_runners[n_train : n_train + n_val]
        test_runners = all_runners[n_train + n_val :]

    # Convert runner sets to file paths
    runner_to_files: dict[str, list[str]] = defaultdict(list)
    for meta in stats.metadata:
        runner_to_files[meta.runner_id].append(str(meta.path))

    train_paths = [f for r in train_runners for f in runner_to_files[r]]
    val_paths = [f for r in val_runners for f in runner_to_files[r]]
    test_paths = [f for r in test_runners for f in runner_to_files[r]]

    # Compute label counts per split
    def count_labels(paths: list[str], metadata: list[VideoMetadata]) -> dict[int, int]:
        path_set = set(paths)
        counts: dict[int, int] = defaultdict(int)
        for m in metadata:
            if str(m.path) in path_set:
                counts[m.label] += 1
        return dict(counts)

    train_labels = count_labels(train_paths, stats.metadata)
    val_labels = count_labels(val_paths, stats.metadata)
    test_labels = count_labels(test_paths, stats.metadata)

    return SplitResult(
        train_paths=sorted(train_paths),
        val_paths=sorted(val_paths),
        test_paths=sorted(test_paths),
        train_runners=sorted(train_runners),
        val_runners=sorted(val_runners),
        test_runners=sorted(test_runners),
        train_label_counts=train_labels,
        val_label_counts=val_labels,
        test_label_counts=test_labels,
    )


def print_split_summary(result: SplitResult) -> None:
    """Print a summary of the split result."""
    logger.info("=" * 60)
    logger.info("DATASET SPLIT SUMMARY")
    logger.info("=" * 60)

    total = len(result.train_paths) + len(result.val_paths) + len(result.test_paths)

    logger.info("Split by Runner ID (no data leakage)")
    logger.info("")
    logger.info(
        "Train: {n} clips ({pct:.1f}%) from {r} runners",
        n=len(result.train_paths),
        pct=len(result.train_paths) / total * 100,
        r=len(result.train_runners),
    )
    logger.info(
        "Val:   {n} clips ({pct:.1f}%) from {r} runners",
        n=len(result.val_paths),
        pct=len(result.val_paths) / total * 100,
        r=len(result.val_runners),
    )
    logger.info(
        "Test:  {n} clips ({pct:.1f}%) from {r} runners",
        n=len(result.test_paths),
        pct=len(result.test_paths) / total * 100,
        r=len(result.test_runners),
    )

    logger.info("")
    logger.info("Train runners: {r}", r=result.train_runners)
    logger.info("Val runners:   {r}", r=result.val_runners)
    logger.info("Test runners:  {r}", r=result.test_runners)

    logger.info("")
    logger.info("Label distribution per split:")
    all_labels = sorted(
        set(result.train_label_counts.keys())
        | set(result.val_label_counts.keys())
        | set(result.test_label_counts.keys())
    )
    for label in all_labels:
        train_n = result.train_label_counts.get(label, 0)
        val_n = result.val_label_counts.get(label, 0)
        test_n = result.test_label_counts.get(label, 0)
        logger.info(
            "  Label {l}: train={t}, val={v}, test={te}",
            l=label,
            t=train_n,
            v=val_n,
            te=test_n,
        )


@dataclass
class SplitParams:
    """Parameters for split_dataset function."""

    dataset_dir: Path
    output_json: Path
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    stratify: bool = True


def split_dataset(params: SplitParams) -> SplitResult:
    """Main function to split a dataset and save to JSON.

    Args:
        params: Split parameters

    Returns:
        SplitResult with all split information
    """
    logger.info("Scanning dataset: {path}", path=params.dataset_dir)
    stats = compute_dataset_stats(params.dataset_dir)

    if stats.parsed_files == 0:
        msg = f"No valid video files found in {params.dataset_dir}"
        raise ValueError(msg)

    logger.info(
        "Found {n} clips from {r} runners",
        n=stats.parsed_files,
        r=len(stats.unique_runners),
    )

    config = SplitConfig(
        train_ratio=params.train_ratio,
        val_ratio=params.val_ratio,
        test_ratio=params.test_ratio,
        seed=params.seed,
        stratify_by_label=params.stratify,
    )

    result = split_by_runner(stats, config)
    print_split_summary(result)

    result.save_json(params.output_json)
    return result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Split dataset by runner ID and save to JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing video files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction for training set",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction for validation set",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction for test set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable label stratification",
    )

    args = parser.parse_args()

    # Validate ratios
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > RATIO_TOLERANCE:
        parser.error(f"Ratios must sum to 1.0, got {total}")

    params = SplitParams(
        dataset_dir=args.input_dir,
        output_json=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify=not args.no_stratify,
    )
    split_dataset(params)


if __name__ == "__main__":
    main()
