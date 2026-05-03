#!/bin/bash
# LeHome Setup Script for Principia VM
# Built based on Artifacts/installation_guide.md 
# Author: Antigravity

set -e # Exit on error

# --- Configuration (Principia VM: large deps on /data; see Artifacts/Instance_adaptation/principia_vm_troubleshooting.md) ---
WORKSPACE_DIR="/data/lehome_workspace"
REPO_DIR="${WORKSPACE_DIR}/lehome-challenge"
UV_BIN_DIR="/data/.local/bin"

echo "🚀 Starting LeHome Challenge Setup..."

# Phase 1: Storage and Environment Setup
echo "📂 [1/6] Setting up storage and environment variables..."
sudo mkdir -p "$WORKSPACE_DIR" /data/huggingface_cache /data/uv_cache "$UV_BIN_DIR"
sudo chown -R principia:principia "$WORKSPACE_DIR" /data/huggingface_cache /data/uv_cache "$UV_BIN_DIR"

# Update .bashrc idempotently
update_bashrc() {
    local line="$1"
    if ! grep -qF "$line" ~/.bashrc; then
        echo "$line" >> ~/.bashrc
        echo "   ✅ Added to .bashrc: $line"
    fi
}

update_bashrc 'export HF_HOME="/data/huggingface_cache"'
update_bashrc 'export UV_CACHE_DIR="/data/uv_cache"'
update_bashrc 'export __GLX_VENDOR_LIBRARY_NAME=nvidia'
update_bashrc 'export PATH="/data/.local/bin:$PATH"'

# Export for current session
export HF_HOME="/data/huggingface_cache"
export UV_CACHE_DIR="/data/uv_cache"
export PATH="${UV_BIN_DIR}:${PATH}"

# Phase 2: System Dependencies
echo "📦 [2/6] Installing system dependencies (requires sudo)..."
# sudo apt update && \
sudo apt install -y \
    libglu1-mesa libgl1 libegl1 libxrandr2 \
    libxinerama1 libxcursor1 libxi6 libxext6 libx11-6 \
    zip psmisc \
    ffmpeg  # required by torchcodec video backend
echo "   ✅ System dependencies installed."

# Phase 3: uv Installation
echo "⚡ [3/6] Installing 'uv' package manager..."
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_BIN_DIR" sh
echo "   ✅ uv installed at $UV_BIN_DIR/uv"

# Phase 4: Repository Cloning
echo "🔗 [4/6] Cloning repositories..."
sudo mkdir -p "$WORKSPACE_DIR"
sudo chown -R principia:principia "$WORKSPACE_DIR"

cd "$WORKSPACE_DIR"

if [ ! -d "lehome-challenge" ]; then
    git clone https://github.com/lehome-official/lehome-challenge.git
else
    echo "   ⚠️  'lehome-challenge' already exists, skipping clone."
fi

cd "$REPO_DIR"
mkdir -p third_party && cd third_party

if [ ! -d "IsaacLab" ]; then
    git clone https://github.com/lehome-official/IsaacLab.git
else
    echo "   ⚠️  'IsaacLab' already exists, skipping clone."
fi
cd ..

# Phase 5: Dependency Installation
echo "🛠️ [5/6] Installing Python dependencies (this may take a few minutes)..."

# 5.1 Enforce lockfile with uv sync
# Note: This creates the .venv on /data (inside lehome-challenge)
uv sync

# 5.2 Manual restore steps for IsaacLab & LeHome (required after uv sync)
echo "   🔧 Restoring IsaacLab and LeHome internal packages..."
# We use 'uv pip' inside the venv to ensure these are not wiped by a future sync
# but are correctly pointed to our dev folders.
uv pip install setuptools flatdict pyarrow prettytable

# Use absolute paths for sub-packages
uv pip install -e "$REPO_DIR/third_party/IsaacLab/source/isaaclab" --no-build-isolation
uv pip install -e "$REPO_DIR/third_party/IsaacLab/source/isaaclab_assets" --no-build-isolation
uv pip install -e "$REPO_DIR/third_party/IsaacLab/source/isaaclab_tasks" --no-build-isolation
uv pip install -e "$REPO_DIR/source/lehome"

# Additional DP dependencies that might be missing from base sync
uv pip install omegaconf h5py diffusers draccus opencv-python-headless einops imageio[ffmpeg]

echo "   ✅ Python environment prepared."

# Phase 6: Dataset Downloads
echo "📊 [6/6] Downloading LeHome datasets..."

# Ensure HF login
if ! "$REPO_DIR/.venv/bin/huggingface-cli" whoami &>/dev/null; then
    echo "   ⚠️  Not authenticated with Hugging Face. Initiating login..."
    "$REPO_DIR/.venv/bin/huggingface-cli" login --token HF_TOKEN_IDE
fi

# Use the absolute path to huggingface-cli in the venv
HF_CLI="$REPO_DIR/.venv/bin/huggingface-cli"

echo "   📥 Downloading simulation assets..."
$HF_CLI download lehome/asset_challenge --repo-type dataset --local-dir Assets

echo "   📥 Downloading merged demonstration data..."
$HF_CLI download lehome/dataset_challenge_merged --repo-type dataset --local-dir Datasets/example

echo "🎉 Setup Complete!"
echo "--------------------------------------------------------"
echo "Next Steps:"
echo "1. Run 'source ~/.bashrc' to refresh your variables."
echo "2. Use './run_train_optimized.sh' to start training."
echo "--------------------------------------------------------"
