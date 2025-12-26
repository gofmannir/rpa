#!/bin/bash
set -e

# Configuration
PROJECT_ID="nir-personal"
ZONE="us-central1-a"
INSTANCE_NAME="rpa-training-gpu"

# Training parameters
EPOCHS="${1:-10}"
BATCH_SIZE="${2:-8}"
LR="${3:-5e-5}"

echo "=== RPA Training - Remote Execution ==="
echo "Instance: $INSTANCE_NAME"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo ""

# First, fix paths in dataset_split.json on the VM
echo "Fixing paths in dataset_split.json..."
gcloud compute ssh "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    --command="cd /app/rpa && python3 scripts/fix_paths.py --remote"

# Run training
echo ""
echo "Starting training..."
gcloud compute ssh "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    --command="
        cd /app/rpa && \
        export PATH=\"\$HOME/.local/bin:\$PATH\" && \
        echo 'Checking GPU...' && \
        nvidia-smi && \
        echo '' && \
        echo 'Syncing dependencies...' && \
        uv sync && \
        echo '' && \
        echo 'Starting training...' && \
        uv run python -m rpa.train \
            --split-json dataset_split.json \
            --output-dir trained_model \
            --remap-labels 2:0 \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --lr $LR
    "

echo ""
echo "=== Training complete ==="
echo "Model saved to: /app/rpa/trained_model"
echo "Run ./scripts/download_model.sh to download the trained model"
