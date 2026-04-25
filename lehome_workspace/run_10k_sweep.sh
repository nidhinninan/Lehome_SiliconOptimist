#!/bin/bash
# run_10k_sweep.sh — Sequential 10k-step micro-sweep across vision backbones.
#
# BUG-7 FIXES applied:
#   · CLI: Uses absolute $VENV_PYTHON with custom bootstrapper (fixes discovery bugs)
#   · Config loading: --config_path=<yaml>  (draccus flag), not --config=
#   · eval_freq is a TOP-LEVEL TrainPipelineConfig field: --eval_freq=N (not --eval.eval_freq)
#   · steps is TOP-LEVEL: --steps=N (already correct, preserved)
#   · Sequential Resumption: Skips existing runs if 'last' checkpoint is found.
#
# Assumptions:
#   · Script is run from the repo root: LeHome-trial/
#   · Script handles its own BYOP pip-installs into the venv at runtime.
#   · Dataset path in each YAML is set correctly for your local VM.
#
# Usage:
#   cd /data/lehome_workspace
#   chmod +x run_10k_sweep.sh
#   ./run_10k_sweep.sh

set -euo pipefail

# Set absolute path for robust execution
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Phase 0: Shared Memory (shm) Fix ──────────────────────────────────────────
# Check if /dev/shm is too small (LeRobot images are large).
# Expansion requires sudo. If sudo fails, we fallback to --num_workers=0.
SHM_REMOUNTED=false
if [ -d "/dev/shm" ]; then
    SHM_SIZE=$(df -Pk /dev/shm | grep -v Filesystem | awk '{print $2}')
    if [ "$SHM_SIZE" -lt 2097152 ]; then
        echo "⚠️  Shared memory (/dev/shm) is less than 2GB ($((SHM_SIZE/1024))MB)."
        echo "   Attempting to remount to 2GB..."
        if sudo mount -o remount,size=2G /dev/shm 2>/dev/null; then
            echo "   ✅ Shared memory expanded to 2GB."
            SHM_REMOUNTED=true
        else
            echo "   ❌ Sudo mount failed. Scripts will fallback to --num_workers=0 if needed."
        fi
    else
        echo "✅ Shared memory is sufficient ($((SHM_SIZE/1024))MB)."
        SHM_REMOUNTED=true
    fi
fi

# ── Phase 1: Install BYOP packages ──────────────────────────────────────────
# Use absolute path to venv to bypass global aliases on Principia VM.
VENV_PYTHON="$WORKSPACE_DIR/lehome-challenge/.venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ ERROR: Virtual environment not found at $VENV_PYTHON"
    echo "   Please run 'uv sync' in $WORKSPACE_DIR/lehome-challenge first."
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Installing BYOP packages"
echo "════════════════════════════════════════════════════════════"
pip install -q -e "$WORKSPACE_DIR/lerobot_policy_dino"
pip install -q -e "$WORKSPACE_DIR/lerobot_policy_clip"
echo "✔ BYOP packages installed."

# ── Phase 2: Sweep ──────────────────────────────────────────────────────────
CONFIGS_DIR="$WORKSPACE_DIR/configs"
STEPS=10000
EVAL_FREQ=10000  # Evaluate once at the very end of training

run_sweep_run() {
    local NAME=$1
    local CONFIG=$2

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  🚀 SWEEP RUN: $NAME"
    echo "  Config: $CONFIG"
    echo "════════════════════════════════════════════════════════════"

    # ⏭️  SKIP CHECK: If the 'last' checkpoint exists, assume the run is done.
    # We check both relative paths and the absolute /data/ path on the VM.
    local CHECK_REL="outputs/sweep_10k/$NAME/checkpoints/last"
    local CHECK_ABS="/data/outputs/sweep_10k/$NAME/checkpoints/last"

    if [ -d "$CHECK_REL" ] || [ -d "$CHECK_ABS" ]; then
        echo "   ⏭️  SKIPPING: Backbone '$NAME' is already trained."
        echo "   Checkpoint found at: ${CHECK_REL} (or ${CHECK_ABS})"
        return 0
    fi

    local LOG_FILE="$WORKSPACE_DIR/lehome-challenge/logs/readout/$NAME-$EVAL_FREQ.log"
    mkdir -p "$(dirname "$LOG_FILE")"

    # Fallback to --num_workers=0 if we couldn't expand shm.
    local WORKERS=4
    if [ "$SHM_REMOUNTED" = false ]; then
        WORKERS=0
        echo "   (Falling back to --num_workers=0 due to shm limits)"
    fi

    # Use the custom bootstrapper to ensure BYOP plugins are registered.
    "$VENV_PYTHON" "$WORKSPACE_DIR/lerobot_train_with_plugins.py" \
        --config_path="$CONFIG" \
        --steps="$STEPS" \
        --eval_freq="$EVAL_FREQ" \
        # 🔗 LINKED DEPENDENCY: This must match the image_transforms.enable 
        # setting in the associated .yaml configuration files.
        --dataset.image_transforms.enable=false \
        --num_workers="$WORKERS" \
        2>&1 | tee "$LOG_FILE"

    echo "  ✔ Run '$NAME' completed."
}

# Run A: ResNet18 — ImageNet-pretrained baseline
run_sweep_run "ResNet18_ImageNet" "$CONFIGS_DIR/sweep_resnet18.yaml"

# Run B: DINOv2-Small — self-supervised ViT backbone (frozen)
run_sweep_run "DINOv2_Small" "$CONFIGS_DIR/sweep_dino.yaml"

# Run C: CLIP-ViT-B/16 — vision-language ViT backbone (frozen)
run_sweep_run "CLIP_Base" "$CONFIGS_DIR/sweep_clip.yaml"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✅ ALL SWEEP RUNS COMPLETED"
echo "  Results saved to: outputs/sweep_10k/"
echo "  Compare final eval metrics across the 3 run directories."
echo "════════════════════════════════════════════════════════════"
