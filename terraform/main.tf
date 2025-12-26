terraform {
  required_version = ">= 1.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# VM instance with GPU
resource "google_compute_instance" "gpu_vm" {
  name         = var.instance_name
  machine_type = var.machine_type
  zone         = var.zone

  # Spot/Preemptible instance for cost savings
  scheduling {
    preemptible                 = true
    automatic_restart           = false
    on_host_maintenance         = "TERMINATE"
    provisioning_model          = "SPOT"
    instance_termination_action = "STOP"
  }

  boot_disk {
    initialize_params {
      image = "deeplearning-platform-release/pytorch-2-7-cu128-ubuntu-2204-nvidia-570"
      size  = var.disk_size_gb
      type  = "pd-ssd"
    }
  }

  network_interface {
    network = "default"
    access_config {
      # Ephemeral public IP
    }
  }

  guest_accelerator {
    type  = var.gpu_type
    count = var.gpu_count
  }

  metadata = {
    install-nvidia-driver = "True"
  }

  metadata_startup_script = file("${path.module}/startup.sh")

  # Allow SSH
  tags = ["ssh-enabled"]

  # Service account with default scopes
  service_account {
    scopes = ["cloud-platform"]
  }
}

# Firewall rule for SSH (if not already exists)
resource "google_compute_firewall" "ssh" {
  name    = "allow-ssh-rpa-training"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["ssh-enabled"]
}
