"""Training script for VideoMAE fine-tuning on running pattern classification.

Loads dataset splits from JSON file and trains a VideoMAE model for binary classification.

Usage:
    uv run python -m rpa.train --split-json dataset_split.json --output-dir trained_model

    # With custom hyperparameters
    uv run python -m rpa.train --split-json dataset_split.json --output-dir trained_model \
        --epochs 10 --batch-size 8 --lr 5e-5
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from transformers import VideoMAEForVideoClassification

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# VideoMAE requirements
NUM_FRAMES = 16
FRAME_SIZE = 224

# Training defaults
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 5e-5
LOG_INTERVAL = 10


@dataclass
class TrainConfig:
    """Training configuration."""

    split_json: Path
    output_dir: Path
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LR
    num_workers: int = 0
    label_remap: dict[int, int] | None = None


class VideoDataset(Dataset):
    """Dataset for loading video clips from JSON split file.

    Extracts labels from filenames using pattern: *_{label}.mp4
    Supports label remapping for binary classification.
    """

    def __init__(
        self,
        video_paths: list[str],
        label_remap: dict[int, int] | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            video_paths: List of video file paths
            label_remap: Optional dict to remap labels (e.g., {2: 0})
        """
        self.video_paths = [Path(p) for p in video_paths]
        self.label_remap = label_remap or {}
        self.labels: list[int] = []

        # Extract labels from filenames
        self._extract_labels()

        # Build label index mapping
        unique_labels = sorted(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(unique_labels)

        logger.info(
            "Dataset: {n} videos, {c} classes, labels={labels}",
            n=len(self.video_paths),
            c=self.num_classes,
            labels=unique_labels,
        )

    def _extract_labels(self) -> None:
        """Extract labels from filenames."""
        # Pattern: filename ends with _XXX.mp4 where XXX is the label
        label_pattern = re.compile(r"_(\d+)\.mp4$")

        for path in self.video_paths:
            match = label_pattern.search(path.name)
            if match:
                label = int(match.group(1))
                # Apply remapping
                label = self.label_remap.get(label, label)
                self.labels.append(label)
            else:
                logger.warning("Could not extract label from: {name}", name=path.name)
                self.labels.append(0)  # Default fallback

    def _load_video_frames(self, video_path: Path) -> np.ndarray | None:
        """Load and preprocess video frames.

        Args:
            video_path: Path to video file

        Returns:
            Array of shape (NUM_FRAMES, FRAME_SIZE, FRAME_SIZE, 3) normalized
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("Failed to open video: {path}", path=video_path)
            return None

        frames: list[np.ndarray] = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(
                    frame_rgb, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_LINEAR
                )
                frames.append(frame_resized)
        finally:
            cap.release()

        if not frames:
            logger.warning("No frames extracted from: {path}", path=video_path)
            return None

        # Temporal sampling
        frames_array = self._temporal_sample(frames)

        # Normalize
        frames_normalized = frames_array.astype(np.float32) / 255.0
        frames_normalized = (frames_normalized - IMAGENET_MEAN) / IMAGENET_STD

        result: np.ndarray = frames_normalized.astype(np.float32)
        return result

    def _temporal_sample(self, frames: list[np.ndarray]) -> np.ndarray:
        """Sample exactly NUM_FRAMES from the video.

        Strategy:
        - If video > NUM_FRAMES: uniform sampling
        - If video < NUM_FRAMES: loop/repeat frames
        """
        n_frames = len(frames)

        if n_frames == NUM_FRAMES:
            return np.stack(frames)

        if n_frames > NUM_FRAMES:
            # Uniform sampling
            sample_indices = np.linspace(0, n_frames - 1, NUM_FRAMES, dtype=int)
            return np.stack([frames[i] for i in sample_indices])

        # Loop frames
        loop_indices = [i % n_frames for i in range(NUM_FRAMES)]
        return np.stack([frames[i] for i in loop_indices])

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | str]:
        """Get a video sample."""
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        label_idx = self.label_to_idx[label]

        try:
            frames = self._load_video_frames(video_path)
            if frames is None:
                frames = np.zeros((NUM_FRAMES, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.float32)
        except Exception as e:
            logger.warning("Error loading {path}: {err}", path=video_path, err=e)
            frames = np.zeros((NUM_FRAMES, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.float32)

        # Convert from (T, H, W, C) to (T, C, H, W) for VideoMAE
        frames_tensor: torch.Tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)

        return {
            "pixel_values": frames_tensor,
            "label": label_idx,
            "filename": video_path.name,
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    """Custom collate function to batch video samples."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    filenames = [item["filename"] for item in batch]

    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "filenames": filenames,
    }


def load_split_data(
    split_json: Path,
    label_remap: dict[int, int] | None = None,
) -> tuple[VideoDataset, VideoDataset, VideoDataset]:
    """Load train/val/test datasets from split JSON.

    Args:
        split_json: Path to dataset_split.json
        label_remap: Optional label remapping dict

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    with split_json.open() as f:
        split_data = json.load(f)

    train_paths = split_data["train"]
    val_paths = split_data["val"]
    test_paths = split_data["test"]

    logger.info(
        "Loading splits: train={t}, val={v}, test={te}",
        t=len(train_paths),
        v=len(val_paths),
        te=len(test_paths),
    )

    train_dataset = VideoDataset(train_paths, label_remap)
    val_dataset = VideoDataset(val_paths, label_remap)
    test_dataset = VideoDataset(test_paths, label_remap)

    return train_dataset, val_dataset, test_dataset


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: {device}", device=device)
    return device


def train_epoch(
    model: VideoMAEForVideoClassification,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """Train for one epoch.

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = outputs.logits.argmax(dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % LOG_INTERVAL == 0:
            logger.info(
                "Epoch {e} | Batch {b}/{total} | Loss: {loss:.4f}",
                e=epoch,
                b=batch_idx + 1,
                total=len(dataloader),
                loss=loss.item(),
            )

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total * 100
    return avg_loss, accuracy


def evaluate(
    model: VideoMAEForVideoClassification,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on a dataset.

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)

            total_loss += outputs.loss.item()
            predictions = outputs.logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    accuracy = correct / total * 100 if total > 0 else 0.0
    return avg_loss, accuracy


@dataclass
class CheckpointInfo:
    """Information for saving a checkpoint."""

    output_dir: Path
    epoch: int
    val_accuracy: float
    is_best: bool = False


def save_checkpoint(
    model: VideoMAEForVideoClassification,
    dataset: VideoDataset,
    info: CheckpointInfo,
) -> None:
    """Save model checkpoint."""
    info.output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    checkpoint_dir = info.output_dir / f"checkpoint-epoch-{info.epoch}"
    model.save_pretrained(checkpoint_dir)

    # Save label mapping
    label_mapping = {
        "idx_to_label": dataset.idx_to_label,
        "label_to_idx": dataset.label_to_idx,
        "num_classes": dataset.num_classes,
    }
    with (checkpoint_dir / "label_mapping.json").open("w") as f:
        json.dump(label_mapping, f, indent=2)

    logger.info(
        "Saved checkpoint to {path} (val_acc={acc:.1f}%)",
        path=checkpoint_dir,
        acc=info.val_accuracy,
    )

    # Save best model
    if info.is_best:
        best_dir = info.output_dir / "best_model"
        model.save_pretrained(best_dir)
        with (best_dir / "label_mapping.json").open("w") as f:
            json.dump(label_mapping, f, indent=2)
        logger.info("Saved best model to {path}", path=best_dir)


def train(config: TrainConfig) -> None:
    """Main training function."""
    device = get_device()

    # Load datasets
    train_dataset, val_dataset, test_dataset = load_split_data(
        config.split_json, config.label_remap
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )

    # Load model
    logger.info("Loading VideoMAE model...")
    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics",
        num_labels=train_dataset.num_classes,
        ignore_mismatched_sizes=True,
    )
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Training loop
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 60)
    logger.info("Epochs: {n}", n=config.epochs)
    logger.info("Batch size: {n}", n=config.batch_size)
    logger.info("Learning rate: {lr}", lr=config.learning_rate)
    logger.info("Train samples: {n}", n=len(train_dataset))
    logger.info("Val samples: {n}", n=len(val_dataset))
    logger.info("Test samples: {n}", n=len(test_dataset))
    logger.info("Num classes: {n}", n=train_dataset.num_classes)
    logger.info("=" * 60)

    best_val_accuracy = 0.0

    for epoch in range(1, config.epochs + 1):
        logger.info("=" * 60)
        logger.info("EPOCH {e}/{total}", e=epoch, total=config.epochs)
        logger.info("=" * 60)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, epoch)
        logger.info(
            "Train - Loss: {loss:.4f}, Accuracy: {acc:.1f}%",
            loss=train_loss,
            acc=train_acc,
        )

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, device)
        logger.info(
            "Val   - Loss: {loss:.4f}, Accuracy: {acc:.1f}%",
            loss=val_loss,
            acc=val_acc,
        )

        # Save checkpoint
        is_best = val_acc > best_val_accuracy
        if is_best:
            best_val_accuracy = val_acc

        checkpoint_info = CheckpointInfo(
            output_dir=config.output_dir,
            epoch=epoch,
            val_accuracy=val_acc,
            is_best=is_best,
        )
        save_checkpoint(model, train_dataset, checkpoint_info)

    # Final evaluation on test set
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("=" * 60)

    test_loss, test_acc = evaluate(model, test_loader, device)
    logger.info(
        "Test  - Loss: {loss:.4f}, Accuracy: {acc:.1f}%",
        loss=test_loss,
        acc=test_acc,
    )

    logger.success("Training complete! Best val accuracy: {acc:.1f}%", acc=best_val_accuracy)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train VideoMAE for running pattern classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        required=True,
        help="Path to dataset split JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data loader workers",
    )
    parser.add_argument(
        "--remap-labels",
        type=str,
        help="Remap labels, e.g., '2:0' to map label 2 to 0",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Validate split JSON
    if not args.split_json.exists():
        logger.error("Split JSON not found: {path}", path=args.split_json)
        sys.exit(1)

    # Parse label remapping
    label_remap: dict[int, int] | None = None
    if args.remap_labels:
        label_remap = {}
        for mapping in args.remap_labels.split(","):
            parts = mapping.strip().split(":")
            if len(parts) == 2:  # noqa: PLR2004
                label_remap[int(parts[0])] = int(parts[1])
        logger.info("Label remapping: {remap}", remap=label_remap)

    config = TrainConfig(
        split_json=args.split_json,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        label_remap=label_remap,
    )

    train(config)


if __name__ == "__main__":
    main()
