#!/bin/bash
# run_train_dino_dp_top_short_150k.sh — 150k MAP+registers DINOv2 training on dp_top_short.
#
# Uses existing configs/sweep_dino_map_registers.yaml (unchanged for 10k sweep); fresh runs
# override steps, dataset root, output_dir, save/eval/log/job_name via CLI (draccus precedence).
#
# Based on run_train_optimized.sh (RESUME hub, dp_top_short paths) and run_2run_dino_sweep.sh
# (WORKSPACE_DIR, shm, BYOP install, lerobot_train_with_plugins.py, W&B .env, thread caps).
#
# W&B (VM): place `.env` under lehome_workspace with WANDB_API_UCMO=<key> (exported as WANDB_API_KEY).
#
# Optional:
#   SKIP_IF_LAST_EXISTS=1  If checkpoints/last exists and RESUME!=true, exit 0 (no-op).
#   OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS  Override defaults (default 1 each).
#   If torchcodec decode fails, re-run once with: --dataset.video_backend=pyav (add to EXTRA_TRAIN_ARGS).
#   ENABLE_RCLONE_CHECKPOINT_SYNC=1  Optional: upload checkpoints to Drive via rclone.
#     - During training: moves only step_*/ (keeps checkpoints/last local for resume).
#     - After training: copies checkpoints/last to Drive, and moves any remaining step_*/.
#   RCLONE_REMOTE=gdrive             rclone remote name (default gdrive).
#   RCLONE_MIN_AGE=15m               min file age before moving step_* (default 15m).
#   RCLONE_SLEEP_SEC=1800            seconds between rclone passes (default 1800).
#   WANDB_PROJECT=lehome_challenge   used only for Drive path grouping (default lehome_challenge).
#
# Usage:
#   cd /path/to/lehome_workspace
#   chmod +x run_train_dino_dp_top_short_150k.sh
#   RESUME=false ./run_train_dino_dp_top_short_150k.sh
#   RESUME=true  ./run_train_dino_dp_top_short_150k.sh
#
# VM verification (torch/lerobot) is not run in the local dev workspace; run on the GPU VM.

set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHALLENGE_DIR="$WORKSPACE_DIR/lehome-challenge"
VENV_PYTHON="$CHALLENGE_DIR/.venv/bin/python"
CONFIG_YAML="$WORKSPACE_DIR/configs/sweep_dino_map_registers.yaml"

# --- CONFIGURATION HUB (match plan / train_dp cadence) ---
RESUME="${RESUME:-false}"
STEPS="${STEPS:-150000}"
SAVE_FREQ="${SAVE_FREQ:-50000}"
EVAL_FREQ="${EVAL_FREQ:-20000}"
LOG_FREQ="${LOG_FREQ:-1000}"
JOB_NAME="${JOB_NAME:-DINOv2_MAP_Registers_dp_top_short_150k_phase1}"
DATASET="${DATASET:-Datasets/example/top_short_merged}"
OUTPUT="${OUTPUT:-outputs/train/dp_top_short_dino_map_registers_150k}"
SKIP_IF_LAST_EXISTS="${SKIP_IF_LAST_EXISTS:-0}"
# Balanced training-time eval for monitoring learning progress without OOM:
#   - eval every 20k steps (signal at 20/40/60/80/100/120/140k)
#   - Very light: 4 episodes × batch_size=2 to minimize GPU spike
#   - Full 50-episode eval should be done separately with parallel_eval.sh
EVAL_N_EPISODES="${EVAL_N_EPISODES:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
EVAL_USE_ASYNC_ENVS="${EVAL_USE_ASYNC_ENVS:-false}"
# Extra args appended verbatim (e.g. --dataset.video_backend=pyav)
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
# -------------------------

if [ ! -f "$VENV_PYTHON" ]; then
    echo "ERROR: Virtual environment not found at $VENV_PYTHON"
    echo "  Run 'uv sync' in $CHALLENGE_DIR first."
    exit 1
fi

if [ ! -f "$CONFIG_YAML" ]; then
    echo "ERROR: Config not found: $CONFIG_YAML"
    exit 1
fi

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
        echo "WARNING: /dev/shm < 2GB ($((SHM_SIZE / 1024))MB). Attempting remount..."
        if sudo mount -o remount,size=2G /dev/shm 2>/dev/null; then
            echo "  Shared memory expanded to 2GB."
            SHM_REMOUNTED=true
        else
            echo "  Sudo remount failed; will use num_workers=0 if needed."
        fi
    else
        echo "OK: /dev/shm sufficient ($((SHM_SIZE / 1024))MB)."
        SHM_REMOUNTED=true
    fi
fi

WORKERS=12
if [ "$SHM_REMOUNTED" = false ]; then
    WORKERS=0
    echo "  Falling back to --num_workers=0 (shm limits)."
fi

echo ""
echo "================================================================"
echo "  Installing lerobot_policy_dino (editable)"
echo "================================================================"
if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$VENV_PYTHON" -q -e "$WORKSPACE_DIR/lerobot_policy_dino"
else
    "$VENV_PYTHON" -m pip install -q -e "$WORKSPACE_DIR/lerobot_policy_dino"
fi
echo "  lerobot_policy_dino installed."

LOG_DIR="$CHALLENGE_DIR/logs/readout"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/dino_map_dp_top_short_150k_phase1_${TIMESTAMP}.log"

LAST_REL="$OUTPUT/checkpoints/last"
LAST_ABS_ROOT="/root/data/lehome_workspace/lehome-challenge/$OUTPUT/checkpoints/last"
LAST_ABS_DATA="/data/lehome_workspace/lehome-challenge/$OUTPUT/checkpoints/last"

cd "$CHALLENGE_DIR"

if [ "$RESUME" = "true" ]; then
    echo "Mode: RESUME from $OUTPUT/checkpoints/last/pretrained_model/train_config.json"
    if [ ! -f "$OUTPUT/checkpoints/last/pretrained_model/train_config.json" ]; then
        echo "ERROR: Resume requested but checkpoint missing:"
        echo "  $OUTPUT/checkpoints/last/pretrained_model/train_config.json"
        exit 1
    fi
else
    echo "Mode: NEW run — config $CONFIG_YAML + CLI overrides"
    echo "  dataset.root=$DATASET  output_dir=$OUTPUT  steps=$STEPS  job_name=$JOB_NAME"
    if [ "$SKIP_IF_LAST_EXISTS" = "1" ]; then
        if [ -d "$LAST_REL" ] || [ -d "$LAST_ABS_ROOT" ] || [ -d "$LAST_ABS_DATA" ]; then
            echo "SKIP_IF_LAST_EXISTS=1: checkpoint exists (last/). Exiting 0."
            exit 0
        fi
    fi
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# Optional rclone checkpoint offload (keeps checkpoints/last local for resume)
RCLONE_REMOTE_NAME="${RCLONE_REMOTE:-gdrive}"
RCLONE_MIN_AGE="${RCLONE_MIN_AGE:-15m}"
RCLONE_SLEEP_SEC="${RCLONE_SLEEP_SEC:-1800}"
WANDB_PROJECT="${WANDB_PROJECT:-lehome_challenge}"
RUN_TAG="${JOB_NAME}__$(date +%Y%m%d_%H%M%S)__$(hostname -s)"
RCLONE_BASE="LeHome/models/${WANDB_PROJECT}/${JOB_NAME}/${RUN_TAG}"
RCLONE_DST="${RCLONE_REMOTE_NAME}:${RCLONE_BASE}/"
RCLONE_LOG="$CHALLENGE_DIR/logs/readout/dino_map_dp_top_short_150k_phase1_rclone_${TIMESTAMP}.log"
RCLONE_BG_PID=""

if [ "${ENABLE_RCLONE_CHECKPOINT_SYNC:-0}" = 1 ]; then
    if ! command -v rclone >/dev/null 2>&1; then
        echo "ERROR: ENABLE_RCLONE_CHECKPOINT_SYNC=1 but rclone not found in PATH"
        exit 1
    fi
    echo "rclone: enabled"
    echo "  dst=${RCLONE_DST}checkpoints"
    echo "  during training: move step_*/ and numeric checkpoint dirs only (keep last/ local)"
    echo "  min-age=${RCLONE_MIN_AGE} sleep=${RCLONE_SLEEP_SEC}s log=${RCLONE_LOG}"
    (
        while true; do
            SRC_DIR="$OUTPUT/checkpoints"
            if [ -d "$SRC_DIR" ]; then
                # LeRobot checkpoint dirs may be named "step_*/" or numeric like "030000/".
                rclone move "$SRC_DIR" "${RCLONE_DST}checkpoints" \
                    --filter '+ step_*/**' \
                    --filter '+ [0-9]*/**' \
                    --filter '- **' \
                    --min-age "$RCLONE_MIN_AGE" \
                    --delete-empty-src-dirs \
                    --log-file "$RCLONE_LOG" --log-level INFO \
                    || true
            fi
            sleep "$RCLONE_SLEEP_SEC"
        done
    ) &
    RCLONE_BG_PID=$!
else
    echo "rclone: disabled (set ENABLE_RCLONE_CHECKPOINT_SYNC=1 to upload to Drive)"
fi

# Ensure background rclone is stopped and final sync is performed.
cleanup_rclone() {
    if [ -n "${RCLONE_BG_PID:-}" ]; then
        kill "$RCLONE_BG_PID" 2>/dev/null || true
        wait "$RCLONE_BG_PID" 2>/dev/null || true
        RCLONE_BG_PID=""

        echo "rclone: final sync"
        if [ -d "$OUTPUT/checkpoints" ]; then
            # 1) Copy last/ (keep it local for phase-2 resume)
            if [ -d "$OUTPUT/checkpoints/last" ]; then
                rclone copy "$OUTPUT/checkpoints/last" "${RCLONE_DST}checkpoints/last" \
                    --log-file "$RCLONE_LOG" --log-level INFO \
                    || true
            fi
            # 2) Move any remaining step_* and numeric checkpoints to Drive
            rclone move "$OUTPUT/checkpoints" "${RCLONE_DST}checkpoints" \
                --filter '+ step_*/**' \
                --filter '+ [0-9]*/**' \
                --filter '- **' \
                --delete-empty-src-dirs \
                --log-file "$RCLONE_LOG" --log-level INFO \
                || true
        fi

        # Keep only last/ locally (best-effort pruning)
        if [ -d "$OUTPUT/checkpoints" ]; then
            find "$OUTPUT/checkpoints" -maxdepth 1 -type d \( -name 'step_*' -o -name '[0-9]*' \) -exec rm -rf {} + 2>/dev/null || true
        fi
    fi
}
trap cleanup_rclone EXIT

# shellcheck disable=SC2086
if [ "$RESUME" = "true" ]; then
    echo "Log: $LOG_FILE"
    "$VENV_PYTHON" "$WORKSPACE_DIR/lerobot_train_with_plugins.py" \
        --config_path="$OUTPUT/checkpoints/last/pretrained_model/train_config.json" \
        --resume=true \
        $EXTRA_TRAIN_ARGS \
        2>&1 | tee "$LOG_FILE"
else
    echo "Log: $LOG_FILE"
    "$VENV_PYTHON" "$WORKSPACE_DIR/lerobot_train_with_plugins.py" \
        --config_path="$CONFIG_YAML" \
        --steps="$STEPS" \
        --save_freq="$SAVE_FREQ" \
        --eval_freq="$EVAL_FREQ" \
        --eval.n_episodes="$EVAL_N_EPISODES" \
        --eval.batch_size="$EVAL_BATCH_SIZE" \
        --eval.use_async_envs="$EVAL_USE_ASYNC_ENVS" \
        --log_freq="$LOG_FREQ" \
        --job_name="$JOB_NAME" \
        --dataset.root="$DATASET" \
        --output_dir="$OUTPUT" \
        --dataset.image_transforms.enable=false \
        --num_workers="$WORKERS" \
        $EXTRA_TRAIN_ARGS \
        2>&1 | tee "$LOG_FILE"
fi

echo "Training session ended. Log: $LOG_FILE"
