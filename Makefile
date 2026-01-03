PROJ_VENV=$(CURDIR)/.venv

MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-builtin-variables

.PHONY: help
help:
	@echo "Available targets:"
	@$(MAKE) -pRrq -f $(MAKEFILE_LIST) : 2>/dev/null |\
	  awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}'|\
	  egrep -v -e '^[^[:alnum:]]' -e '^$@' |\
	  sort |\
	  awk '{print "  " $$0}'

$(PROJ_VENV):
	uv venv "$@"

.PHONY: init clean
init: $(PROJ_VENV)
	uv sync

clean:
	rm -rf "$(PROJ_VENV)"

.PHONY: check check-py
check-py:
	uv run ruff check
	uv run ruff format --check
	uv run mypy .
check: check-py

.PHONY: format
format:
	uv run ruff format

.PHONY: fix
fix: format
	uv run ruff check --fix

.PHONY: train
train:
	uv run python -m rpa.train \
		--split-json dataset_split.json \
		--output-dir trained_model \
		--remap-labels 2:0 \
		--epochs 5 \
		--batch-size 8

# Train with pre-augmented dataset (no runtime augmentation)
.PHONY: train-augmented
train-augmented:
	uv run python -m rpa.train \
		--split-json dataset_split_augmented.json \
		--output-dir trained_model_augmented \
		--remap-labels 2:0 \
		--epochs 5 \
		--batch-size 8 \
		--no-augmentation

# Augment dataset on GCS (run on VM for better bandwidth)
GCS_BUCKET=gs://rpa-dataset-nirgofman
.PHONY: augment-gcs
augment-gcs:
	uv run python -m rpa.augment \
		--input $(GCS_BUCKET)/raw/ \
		--output $(GCS_BUCKET)/augmented/ \
		--versions 25 \
		--workers 4 \
		--checkpoint $(GCS_BUCKET)/augmented/.checkpoint.json

# GCP VM name
VM_NAME=rpa-training-gpu
VM_ZONE=us-central1-a

# SSH to GCP VM
.PHONY: ssh-vm
ssh-vm:
	gcloud compute ssh $(VM_NAME) --zone=$(VM_ZONE)

# Sync code to VM (via git push)
.PHONY: sync-vm
sync-vm:
	git push vm main --force

# Download trained model from VM
.PHONY: download-model
download-model:
	gcloud compute scp --recurse \
		$(VM_NAME):/home/nirgofman/rpa/trained_model_gcs/best_model \
		./trained_model_gcs/best_model \
		--zone=$(VM_ZONE)

# Run inference on videos
# Usage: make inference VIDEO=/path/to/video.mp4
#        make inference VIDEO_DIR=/path/to/clips/
MODEL_DIR=./trained_model_gcs/best_model
.PHONY: inference
inference:
ifdef VIDEO_DIR
	uv run python -m rpa.inference --model_dir $(MODEL_DIR) --video_dir $(VIDEO_DIR)
else ifdef VIDEO
	uv run python -m rpa.inference --model_dir $(MODEL_DIR) --video $(VIDEO)
else
	@echo "Usage: make inference VIDEO=/path/to/video.mp4"
	@echo "       make inference VIDEO_DIR=/path/to/clips/"
endif

# Augment local directory (for testing)
.PHONY: augment-local
augment-local:
	uv run python -m rpa.augment \
		--input /tmp/test_videos/ \
		--output /tmp/test_augmented/ \
		--versions 5 \
		--workers 2

# Create GCS-aware split with augmented train data
.PHONY: create-gcs-split
create-gcs-split:
	uv run python -m rpa.create_augmented_split \
		--original-split dataset_split.json \
		--bucket $(GCS_BUCKET) \
		--output dataset_split_gcs.json \
		--versions 25

# Train on GCS data (augmented train, raw val/test)
.PHONY: train-gcs
train-gcs:
	uv run python -m rpa.train \
		--split-json dataset_split_gcs.json \
		--output-dir trained_model_gcs \
		--remap-labels 2:0 \
		--epochs 10 \
		--batch-size 8 \
		--no-augmentation

# File browser for GCS bucket (requires Docker)
.PHONY: filestash
filestash:
	@echo "Starting Filestash file browser..."
	@echo "Open http://localhost:8334 in your browser"
	@echo "Bucket: $(GCS_BUCKET)"
	docker run --rm -d \
		-p 8334:8334 \
		--name filestash \
		-v $(HOME)/rpa-gcs-key.json:/app/data/state/config/gcs-key.json:ro \
		-e APPLICATION_URL=http://localhost:8334 \
		machines/filestash
	@echo "Container started. Configure GCS backend in admin panel."

.PHONY: filestash-stop
filestash-stop:
	docker stop filestash 2>/dev/null || true

# Terraform targets for GCS service account
.PHONY: tf-init
tf-init:
	cd terraform && terraform init

.PHONY: tf-plan
tf-plan:
	cd terraform && terraform plan

.PHONY: tf-apply
tf-apply:
	cd terraform && terraform apply
	@echo "Service account key saved to: terraform/rpa-gcs-key.json"

.PHONY: tf-destroy
tf-destroy:
	cd terraform && terraform destroy
