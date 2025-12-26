#!/bin/bash
set -e

# Configuration
PROJECT_ID="nir-personal"
ZONE="us-central1-a"
INSTANCE_NAME="rpa-training-gpu"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOCAL_MODEL_DIR="$PROJECT_DIR/trained_model"

echo "=== RPA Training - Download Model ==="
echo "Downloading trained model from VM..."
echo ""

# Create local directory
mkdir -p "$LOCAL_MODEL_DIR"

# Download the best model
echo "Downloading best model..."
gcloud compute scp --recurse \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    "$INSTANCE_NAME":/app/rpa/trained_model/best_model \
    "$LOCAL_MODEL_DIR/"

# Also download the latest checkpoint
echo ""
echo "Downloading latest checkpoint..."
gcloud compute scp --recurse \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    "$INSTANCE_NAME":/app/rpa/trained_model/checkpoint-* \
    "$LOCAL_MODEL_DIR/" 2>/dev/null || echo "No checkpoints found"

echo ""
echo "=== Download complete ==="
echo "Model saved to: $LOCAL_MODEL_DIR"
echo ""
echo "To run inference:"
echo "  uv run python -m rpa.inference --model_dir $LOCAL_MODEL_DIR/best_model --video <video_path>"
