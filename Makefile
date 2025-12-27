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

# Sync code to VM
.PHONY: sync-vm
sync-vm:
	gcloud compute scp --recurse --zone=$(VM_ZONE) src/rpa/ $(VM_NAME):~/rpa/src/rpa/

# Augment local directory (for testing)
.PHONY: augment-local
augment-local:
	uv run python -m rpa.augment \
		--input /tmp/test_videos/ \
		--output /tmp/test_augmented/ \
		--versions 5 \
		--workers 2
