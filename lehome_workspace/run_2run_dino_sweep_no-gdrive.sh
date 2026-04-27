#!/bin/bash
# run_2run_dino_sweep_no-gdrive.sh — Two-run sweep: MAP+registers DINOv2 first, then baseline DINOv2.
# No rclone / Google Drive offload. For optional Drive sync + `.complete` skip marker, use
# `run_2run_dino_sweep.sh` instead.
#
# W&B (VM): place `.env` under this workspace with WANDB_API_UCMO=<your key>.
#   This script loads it and sets WANDB_API_KEY for Weights & Biases.
#   Top-level job_name + wandb.project in each YAML control run name / project.
#
# Usage:
#   cd /path/to/lehome_workspace
#   chmod +x run_2run_dino_sweep_no-gdrive.sh
#   ./run_2run_dino_sweep_no-gdrive.sh
#
# CLI matches LeRobot v0.4.3 / lerobot_train_with_plugins.py (see run_10k_sweep.sh).

set -euo pipefail

WORKSPACE_DIR=$(pwd)
# WORKSPACE_DIR=$(pwd)

# Optional: load VM .env (WANDB_API_UCMO -> WANDB_API_KEY)
if [ -f "$WORKSPACE_DIR/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    . "$WORKSPACE_DIR/.env"
    set +a
fi
if [ -n "${WANDB_API_UCMO:-}" ]; then
    export WANDB_API_KEY="${WANDB_API_UCMO}"
fi

SHM_REMOUNTED=false
if [ -d "/dev/shm" ]; then
    SHM_SIZE=$(df -Pk /dev/shm | grep -v Filesystem | awk '{print $2}')
    if [ "$SHM_SIZE" -lt 2097152 ]; then
        echo "⚠️  Shared memory (/dev/shm) is less than 2GB ($((SHM_SIZE/1024))MB)."
        if sudo mount -o remount,size=2G /dev/shm 2>/dev/null; then
            echo "   ✅ Shared memory expanded to 2GB."
            SHM_REMOUNTED=true
        else
            echo "   ❌ Sudo mount failed. Falling back to --num_workers=0 if needed."
        fi
    else
        echo "✅ Shared memory is sufficient ($((SHM_SIZE/1024))MB)."
        SHM_REMOUNTED=true
    fi
fi

VENV_PYTHON="$WORKSPACE_DIR/lehome-challenge/.venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ ERROR: Virtual environment not found at $VENV_PYTHON"
    echo "   Run 'uv sync' in $WORKSPACE_DIR/lehome-challenge first."
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Installing DINO BYOP package"
echo "════════════════════════════════════════════════════════════"
if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$VENV_PYTHON" -q -e "$WORKSPACE_DIR/lerobot_policy_dino"
else
    "$VENV_PYTHON" -m pip install -q -e "$WORKSPACE_DIR/lerobot_policy_dino"
fi
echo "✔ lerobot_policy_dino installed."

CONFIGS_DIR="$WORKSPACE_DIR/configs"
STEPS=10000
EVAL_FREQ=10000

run_sweep_run() {
    local NAME=$1
    local CONFIG=$2

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  SWEEP RUN: $NAME"
    echo "  Config: $CONFIG"
    echo "════════════════════════════════════════════════════════════"

    local CHECK_REL="outputs/sweep_10k/$NAME/checkpoints/last"
    local CHECK_ABS="/root/data/lehome_workspace/outputs/sweep_10k/$NAME/checkpoints/last"

    if [ -d "$CHECK_REL" ] || [ -d "$CHECK_ABS" ]; then
        echo "   SKIPPING: checkpoint exists at ${CHECK_REL} or ${CHECK_ABS}"
        return 0
    fi

    local LOG_FILE="$WORKSPACE_DIR/lehome-challenge/logs/readout/$NAME-$EVAL_FREQ.log"
    mkdir -p "$(dirname "$LOG_FILE")"

    local WORKERS=12
    if [ "$SHM_REMOUNTED" = false ]; then
        WORKERS=0
        echo "   (Falling back to --num_workers=0 due to shm limits)"
    fi

    "$VENV_PYTHON" "$WORKSPACE_DIR/lerobot_train_with_plugins.py" \
        --config_path="$CONFIG" \
        --steps="$STEPS" \
        --eval_freq="$EVAL_FREQ" \
        --dataset.image_transforms.enable=false \
        #--dataset.video_backend=pyav \ # Installed FFmpeg
        --num_workers="$WORKERS" \
        2>&1 | tee "$LOG_FILE"

    echo "  Done: $NAME"
}

# Order: MAP+registers first, then original baseline
run_sweep_run "DINOv2_MAP_Registers" "$CONFIGS_DIR/sweep_dino_map_registers.yaml"
run_sweep_run "DINOv2_Baseline" "$CONFIGS_DIR/sweep_dino_baseline.yaml"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Two-run DINO sweep finished."
echo "  Check outputs/sweep_10k/DINOv2_MAP_Registers and DINOv2_Baseline"
echo "  W&B project: lehome_challenge (run names from job_name in YAML)"
echo "════════════════════════════════════════════════════════════"
