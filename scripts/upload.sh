#!/bin/bash
set -e

# Configuration
PROJECT_ID="nir-personal"
ZONE="us-central1-a"
INSTANCE_NAME="rpa-training-gpu"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATASET_DIR="$(dirname "$PROJECT_DIR")/final_dataset"

echo "=== RPA Training - File Upload ==="
echo "Project dir: $PROJECT_DIR"
echo "Dataset dir: $DATASET_DIR"
echo "Target VM: $INSTANCE_NAME"
echo ""

# Check if dataset exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory not found: $DATASET_DIR"
    exit 1
fi

# Wait for VM to be ready
echo "Waiting for VM to be ready..."
gcloud compute ssh "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    --command="echo 'VM is ready'" \
    -- -o ConnectTimeout=60

# Upload dataset (this will take a while for 1GB)
echo ""
echo "Uploading dataset (~1GB, this may take a few minutes)..."
gcloud compute scp --recurse \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    --compress \
    "$DATASET_DIR" \
    "$INSTANCE_NAME":/data/

# Upload code (excluding .venv, __pycache__, etc.)
echo ""
echo "Uploading code..."
rsync -avz --progress \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='.mypy_cache' \
    --exclude='.ruff_cache' \
    --exclude='.git' \
    --exclude='*.pyc' \
    --exclude='terraform/.terraform' \
    --exclude='terraform/*.tfstate*' \
    --exclude='trained_model' \
    -e "gcloud compute ssh --zone=$ZONE --project=$PROJECT_ID -- " \
    "$PROJECT_DIR/" \
    "$INSTANCE_NAME":/app/rpa/

echo ""
echo "=== Upload complete ==="
echo "Dataset uploaded to: /data/final_dataset"
echo "Code uploaded to: /app/rpa"
