#!/bin/bash
set -e

# Log startup script execution
exec > >(tee /var/log/startup-script.log) 2>&1
echo "Starting VM setup at $(date)"

# Create directories for data and app
mkdir -p /data /app
chmod 777 /data /app

# The Deep Learning VM image already has:
# - NVIDIA drivers
# - CUDA
# - Python 3.10+
# - PyTorch

# Install uv package manager
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH for all users
echo 'export PATH="$HOME/.local/bin:$PATH"' >> /etc/profile.d/uv.sh
chmod +x /etc/profile.d/uv.sh

# Install Python 3.13 via uv (for the current user, will be done on first login)
cat > /etc/profile.d/setup-python.sh << 'EOF'
# One-time Python 3.13 setup
if [ ! -f ~/.python_setup_done ]; then
    echo "Setting up Python 3.13..."
    export PATH="$HOME/.local/bin:$PATH"
    uv python install 3.13 2>/dev/null || true
    touch ~/.python_setup_done
fi
EOF
chmod +x /etc/profile.d/setup-python.sh

# Verify GPU is available
echo "Checking GPU..."
nvidia-smi || echo "Warning: nvidia-smi failed, GPU might not be ready yet"

echo "VM setup complete at $(date)"
echo "To check GPU status: nvidia-smi"
echo "Data directory: /data"
echo "App directory: /app"
