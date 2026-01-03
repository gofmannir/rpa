"""Inference script for running predictions on video files using a trained VideoMAE model.

Usage:
    uv run python -m rpa.inference --model_dir /path/to/trained_model --video /path/to/video.mp4

    # Run on multiple videos
    uv run python -m rpa.inference --model_dir /path/to/trained_model --video vid1.mp4 vid2.mp4 vid3.mp4

    # Run on all videos in a directory
    uv run python -m rpa.inference --model_dir /path/to/trained_model --video_dir /path/to/clips/
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from transformers import VideoMAEForVideoClassification

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# VideoMAE requirements
NUM_FRAMES = 16
FRAME_SIZE = 224


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert RGB frame to grayscale (replicated to 3 channels).

    Matches the grayscale conversion used in training/augmentation pipeline.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return np.stack([gray, gray, gray], axis=-1)


def load_model(model_dir: Path) -> tuple[VideoMAEForVideoClassification, dict[int, int], torch.device]:
    """Load trained model and label mapping from disk.

    Args:
        model_dir: Directory containing saved model files

    Returns:
        Tuple of (model, idx_to_label mapping, device)
    """
    # Device setup: prefer CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: {device}", device=device)

    # Load model
    logger.info("Loading model from: {path}", path=model_dir)
    model = VideoMAEForVideoClassification.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()

    # Load label mapping
    label_path = model_dir / "label_mapping.json"
    if not label_path.exists():
        logger.warning("Label mapping not found, using default indices")
        num_labels = model.config.num_labels
        idx_to_label = {i: i for i in range(num_labels)}
    else:
        with label_path.open() as f:
            mapping = json.load(f)
        # Convert string keys back to int
        idx_to_label = {int(k): v for k, v in mapping["idx_to_label"].items()}

    logger.info("Model loaded with {n} classes", n=len(idx_to_label))
    return model, idx_to_label, device


def load_video(video_path: Path) -> torch.Tensor | None:
    """Load and preprocess a video file for inference.

    Args:
        video_path: Path to video file

    Returns:
        Tensor of shape (1, T, C, H, W) ready for model, or None if loading fails
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Failed to open video: {path}", path=video_path)
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
            # Convert to grayscale (3-channel) to match training pipeline
            frame_gray = to_grayscale(frame_resized)
            frames.append(frame_gray)
    finally:
        cap.release()

    if not frames:
        logger.error("No frames extracted from video")
        return None

    # Temporal sampling
    n_frames = len(frames)
    if n_frames == NUM_FRAMES:
        frames_array = np.stack(frames)
    elif n_frames > NUM_FRAMES:
        start = (n_frames - NUM_FRAMES) // 2
        frames_array = np.stack(frames[start : start + NUM_FRAMES])
    else:
        indices = [i % n_frames for i in range(NUM_FRAMES)]
        frames_array = np.stack([frames[i] for i in indices])

    # Normalize
    frames_normalized = frames_array.astype(np.float32) / 255.0
    frames_normalized = (frames_normalized - IMAGENET_MEAN) / IMAGENET_STD

    # Convert to tensor: (T, H, W, C) -> (1, T, C, H, W)
    frames_tensor = torch.from_numpy(frames_normalized.astype(np.float32)).permute(0, 3, 1, 2)
    return frames_tensor.unsqueeze(0)


def predict(
    model: VideoMAEForVideoClassification,
    video_path: Path,
    idx_to_label: dict[int, int],
    device: torch.device,
) -> dict[str, int | float | str | dict[int, float]] | None:
    """Run prediction on a single video.

    Args:
        model: Trained model
        video_path: Path to video file
        idx_to_label: Index to label mapping
        device: Torch device

    Returns:
        Dictionary with prediction results, or None if failed
    """
    pixel_values = load_video(video_path)
    if pixel_values is None:
        return None

    pixel_values = pixel_values.to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_idx = int(outputs.logits.argmax(dim=-1).item())
        confidence = float(probs[0, pred_idx].item()) * 100

    pred_label = idx_to_label[pred_idx]
    all_probs = {idx_to_label[i]: float(p) * 100 for i, p in enumerate(probs[0].tolist())}

    return {
        "video": video_path.name,
        "predicted_label": pred_label,
        "confidence": confidence,
        "all_probabilities": all_probs,
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference on video files using a trained VideoMAE model"
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Path to directory containing trained model",
    )
    parser.add_argument(
        "--video",
        type=Path,
        nargs="+",
        help="Path(s) to video file(s) for prediction",
    )
    parser.add_argument(
        "--video_dir",
        type=Path,
        help="Directory containing video files for batch prediction",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional: Save predictions to JSON file",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Validate model directory
    if not args.model_dir.exists():
        logger.error("Model directory does not exist: {path}", path=args.model_dir)
        sys.exit(1)

    # Collect video files
    video_files: list[Path] = []
    if args.video:
        video_files.extend(args.video)
    if args.video_dir:
        if args.video_dir.exists():
            video_files.extend(sorted(args.video_dir.glob("*.mp4")))
        else:
            logger.error("Video directory does not exist: {path}", path=args.video_dir)
            sys.exit(1)

    if not video_files:
        logger.error("No video files specified. Use --video or --video_dir")
        sys.exit(1)

    # Load model
    model, idx_to_label, device = load_model(args.model_dir)

    # Run predictions
    results: list[dict] = []
    logger.info("Running predictions on {n} videos...", n=len(video_files))

    for video_path in video_files:
        if not video_path.exists():
            logger.warning("Video not found: {path}", path=video_path)
            continue

        result = predict(model, video_path, idx_to_label, device)
        if result:
            results.append(result)
            logger.info(
                "{video} -> Label: {label} (Confidence: {conf:.1f}%)",
                video=result["video"],
                label=result["predicted_label"],
                conf=result["confidence"],
            )

    # Print summary
    logger.info("=" * 60)
    logger.info("PREDICTION SUMMARY")
    logger.info("=" * 60)

    header = f"{'Video':<45} | {'Label':>6} | {'Conf':>7}"
    separator = "-" * len(header)
    logger.info(separator)
    logger.info(header)
    logger.info(separator)

    for r in results:
        logger.info(
            "{video:<45} | {label:>6} | {conf:>6.1f}%",
            video=r["video"][:45],
            label=r["predicted_label"],
            conf=r["confidence"],
        )

    logger.info(separator)

    # Save to JSON if requested
    if args.output:
        with args.output.open("w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to: {path}", path=args.output)


if __name__ == "__main__":
    main()
