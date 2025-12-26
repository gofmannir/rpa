variable "project_id" {
  description = "GCP project ID"
  type        = string
  default     = "nir-personal"
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "machine_type" {
  description = "VM machine type"
  type        = string
  default     = "n1-standard-4"
}

variable "gpu_type" {
  description = "GPU accelerator type"
  type        = string
  default     = "nvidia-tesla-t4"
}

variable "gpu_count" {
  description = "Number of GPUs"
  type        = number
  default     = 1
}

variable "disk_size_gb" {
  description = "Boot disk size in GB"
  type        = number
  default     = 100
}

variable "instance_name" {
  description = "Name of the VM instance"
  type        = string
  default     = "rpa-training-gpu"
}
