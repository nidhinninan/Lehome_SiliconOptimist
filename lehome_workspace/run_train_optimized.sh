#!/bin/bash
# LeHome Optimized Training Script
# Author: Antigravity

# --- CONFIGURATION HUB ---
# Change this variable to switch between garment types:
# Options: dp_top_short, dp_top_long, dp_pant_short, dp_pant_long
CONFIG_NAME="dp_top_short"

# Set to 'true' if you want to resume from the last checkpoint
RESUME="false" 
# -------------------------

# Paths (Absolute as per best practices)
WORKSPACE_BASE="/data/lehome_workspace/lehome-challenge"
VENV_PYTHON="$WORKSPACE_BASE/.venv/bin/python"
LOG_DIR="$WORKSPACE_BASE/logs/training"

echo "🏹 Preparing training for: $CONFIG_NAME"

# 1. Performance Tweak: Expand Shared Memory
# Prevents "Bus error" in DataLoader workers
echo "🧠 Expanding /dev/shm..."
sudo mount -o remount,size=2G /dev/shm || \
    echo "   ⚠️  Could not expand /dev/shm. If training crashes, set WORKERS=0 below."

# 2. Argument Mapping
case $CONFIG_NAME in
    "dp_top_short")
        DATASET="Datasets/example/top_short_merged"
        OUTPUT="outputs/train/dp_top_short"
        ;;
    "dp_pant_short")
        DATASET="Datasets/example/pant_short_merged"
        OUTPUT="outputs/train/dp_pant_short"
        ;;
    "dp_pant_long")
        DATASET="Datasets/example/pant_long_merged"
        OUTPUT="outputs/train/dp_pant_long"
        ;;
    "dp_top_long")
        DATASET="Datasets/example/top_long_merged"
        OUTPUT="outputs/train/dp_top_long"
        ;;
    *)
        echo "❌ Error: Invalid CONFIG_NAME '$CONFIG_NAME'"
        exit 1
        ;;
esac

# 3. Resume Logic
RESUME_FLAG=""
if [ "$RESUME" = "true" ]; then
    # Point to the config in the 'last' checkpoint
    CONFIG_OVERRIDE="--config_path=$OUTPUT/checkpoints/last/pretrained_model/train_config.json"
    RESUME_FLAG="--resume=true"
    echo "🔄 Mode: RESUMING from last checkpoint"
else
    # Start fresh using base config
    CONFIG_OVERRIDE="--config_path=configs/train_dp.yaml"
    echo "✨ Mode: NEW training run"
fi

# Ensure log directory exists
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M)
LOG_FILE="$LOG_DIR/${CONFIG_NAME}_${TIMESTAMP}.log"

echo "📂 Log File: $LOG_FILE"

# 4. Execution
cd "$WORKSPACE_BASE"

# Using absolute path to bypass Principia aliases
$VENV_PYTHON -m lerobot.scripts.lerobot_train \
    $CONFIG_OVERRIDE \
    $RESUME_FLAG \
    --dataset.root="$DATASET" \
    --output_dir="$OUTPUT" \
    --num_workers 4 \
    2>&1 | tee "$LOG_FILE"

echo "✅ Training session ended. Check logs at $LOG_FILE"
