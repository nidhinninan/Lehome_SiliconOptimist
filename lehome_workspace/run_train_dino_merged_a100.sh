#!/bin/bash
# run_train_dino_merged_a100.sh — DINOv2 MAP+registers training on merged 4-garment dataset for A100-class VMs.
#
# Principia VM paths (/data/lehome_workspace): set LEHOME_VM_PROFILE=principia so HF_HOME / UV_CACHE_DIR live on
# /data (see principia_vm_troubleshooting.md). Tier presets below are for A100-class GPUs — unchanged when you run
# from Principia; only storage PATH exports differ. For RTX 4070 Ti 16GB training, use run_train_dino_dp_top_short_150k.sh.
#
# Wraps the same pipeline as run_train_dino_dp_top_short_150k.sh (sweep_dino_map_registers.yaml,
# lerobot_train_with_plugins.py, BYOP install, rclone, thread caps) with plan presets:
#   A100_TIER=conservative|normal|aggressive|smoke_cons|smoke_norm|smoke_aggr   (default: normal)
#
# Defaults apply only when RESUME!=true and the corresponding env var is unset (bash : "${VAR:=...}").
#
# Optional:
#   SHM_TARGET=16G   passed to sudo mount -o remount,size=... when /dev/shm is small (default 16G).
#   Same SKIP_IF_LAST_EXISTS, EXTRA_TRAIN_ARGS, rclone, W&B vars as the 150k launcher.
#   INCLUDE_LEGACY_VM_LAST_PATHS=1  Also treat /root/data/lehome_workspace/.../last as existing (non-Principia VMs).
#
# Usage (from lehome_workspace on VM):
#   chmod +x run_train_dino_merged_a100.sh
#   LEHOME_VM_PROFILE=principia A100_TIER=normal RESUME=false ./run_train_dino_merged_a100.sh
#   A100_TIER=normal RESUME=false ./run_train_dino_merged_a100.sh
#   A100_TIER=smoke_norm RESUME=false ./run_train_dino_merged_a100.sh
#   RESUME=true OUTPUT=outputs/train/... ./run_train_dino_merged_a100.sh
#
# VM verification (torch/lerobot) is not run in the local dev workspace; run on the GPU VM.

set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Principia VM: keep Hugging Face + uv caches off the root (/) disk ---
if [[ "${LEHOME_VM_PROFILE:-}" == "principia" ]] || [[ "$WORKSPACE_DIR" == /data/lehome_workspace ]] || [[ "$WORKSPACE_DIR" == /data/lehome_workspace/* ]]; then
    export UV_CACHE_DIR="${UV_CACHE_DIR:-/data/uv_cache}"
    export HF_HOME="${HF_HOME:-/data/huggingface_cache}"
    mkdir -p "$UV_CACHE_DIR" "$HF_HOME" 2>/dev/null || true
    if [[ -d /data/.local/bin ]] && [[ ":$PATH:" != *:/data/.local/bin:* ]]; then
        export PATH="/data/.local/bin:$PATH"
    fi
fi
CHALLENGE_DIR="$WORKSPACE_DIR/lehome-challenge"
VENV_PYTHON="$CHALLENGE_DIR/.venv/bin/python"
CONFIG_YAML="$WORKSPACE_DIR/configs/sweep_dino_map_registers.yaml"

RESUME="${RESUME:-false}"
SHM_TARGET="${SHM_TARGET:-16G}"

# --- A100 tier presets (only when not resuming; user can override any var before invoking) ---
if [ "$RESUME" != "true" ]; then
    case "${A100_TIER:-normal}" in
        conservative)
            : "${STEPS:=400000}"
            : "${SAVE_FREQ:=50000}"
            : "${EVAL_FREQ:=20000}"
            : "${EVAL_N_EPISODES:=4}"
            : "${EVAL_BATCH_SIZE:=2}"
            : "${JOB_NAME:=DINOv2_MAP8_Registers_merged_A100_400k_Conservative}"
            : "${OUTPUT:=outputs/train/dp_merged_dino_map8_registers_a100_400k_cons}"
            : "${EXTRA_TRAIN_ARGS:=--batch_size=16 --num_workers=12 --policy.map_num_queries=8}"
            ;;
        normal)
            : "${STEPS:=400000}"
            : "${SAVE_FREQ:=50000}"
            : "${EVAL_FREQ:=20000}"
            : "${EVAL_N_EPISODES:=8}"
            : "${EVAL_BATCH_SIZE:=2}"
            : "${JOB_NAME:=DINOv2_MAP8_Registers_merged_A100_400k_Normal}"
            : "${OUTPUT:=outputs/train/dp_merged_dino_map8_registers_a100_400k_norm}"
            : "${EXTRA_TRAIN_ARGS:=--batch_size=32 --num_workers=12 --policy.map_num_queries=8}"
            ;;
        aggressive)
            : "${STEPS:=400000}"
            : "${SAVE_FREQ:=50000}"
            : "${EVAL_FREQ:=25000}"
            : "${EVAL_N_EPISODES:=8}"
            : "${EVAL_BATCH_SIZE:=4}"
            : "${JOB_NAME:=DINOv2_MAP8_Registers_merged_A100_400k_Aggressive}"
            : "${OUTPUT:=outputs/train/dp_merged_dino_map8_registers_a100_400k_aggr}"
            : "${EXTRA_TRAIN_ARGS:=--batch_size=48 --num_workers=16 --policy.map_num_queries=8}"
            ;;
        smoke_cons)
            : "${STEPS:=5000}"
            : "${SAVE_FREQ:=5000}"
            : "${EVAL_FREQ:=1000}"
            : "${EVAL_N_EPISODES:=4}"
            : "${EVAL_BATCH_SIZE:=2}"
            : "${JOB_NAME:=SMOKE_DINOv2_MAP8_Registers_merged_A100_cons}"
            : "${OUTPUT:=outputs/train/smoke_dp_merged_dino_map8_registers_a100_cons}"
            : "${EXTRA_TRAIN_ARGS:=--batch_size=16 --num_workers=12 --policy.map_num_queries=8}"
            ;;
        smoke_norm)
            : "${STEPS:=5000}"
            : "${SAVE_FREQ:=5000}"
            : "${EVAL_FREQ:=1000}"
            : "${EVAL_N_EPISODES:=4}"
            : "${EVAL_BATCH_SIZE:=2}"
            : "${JOB_NAME:=SMOKE_DINOv2_MAP8_Registers_merged_A100_norm}"
            : "${OUTPUT:=outputs/train/smoke_dp_merged_dino_map8_registers_a100_norm}"
            : "${EXTRA_TRAIN_ARGS:=--batch_size=32 --num_workers=12 --policy.map_num_queries=8}"
            ;;
        smoke_aggr)
            : "${STEPS:=5000}"
            : "${SAVE_FREQ:=5000}"
            : "${EVAL_FREQ:=2000}"
            : "${EVAL_N_EPISODES:=4}"
            : "${EVAL_BATCH_SIZE:=2}"
            : "${JOB_NAME:=SMOKE_DINOv2_MAP8_Registers_merged_A100_aggr}"
            : "${OUTPUT:=outputs/train/smoke_dp_merged_dino_map8_registers_a100_aggr}"
            : "${EXTRA_TRAIN_ARGS:=--batch_size=48 --num_workers=16 --policy.map_num_queries=8}"
            ;;
        *)
            echo "ERROR: Unknown A100_TIER='${A100_TIER:-}'. Use: conservative|normal|aggressive|smoke_cons|smoke_norm|smoke_aggr"
            exit 1
            ;;
    esac
fi

# --- CONFIGURATION HUB (merged dataset default) ---
: "${STEPS:=400000}"
: "${SAVE_FREQ:=50000}"
: "${EVAL_FREQ:=20000}"
: "${LOG_FREQ:=1000}"
: "${JOB_NAME:=DINOv2_MAP8_Registers_merged_A100}"
: "${DATASET:=Datasets/example/dataset_challenge_merged}"
: "${OUTPUT:=outputs/train/dp_merged_dino_map8_registers_a100}"
SKIP_IF_LAST_EXISTS="${SKIP_IF_LAST_EXISTS:-0}"
: "${EVAL_N_EPISODES:=8}"
: "${EVAL_BATCH_SIZE:=2}"
EVAL_USE_ASYNC_ENVS="${EVAL_USE_ASYNC_ENVS:-false}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "ERROR: Virtual environment not found at $VENV_PYTHON"
    echo "  Run 'uv sync' in $CHALLENGE_DIR first."
    exit 1
fi

if [ ! -f "$CONFIG_YAML" ]; then
    echo "ERROR: Config not found: $CONFIG_YAML"
    exit 1
fi

if [ -f "$WORKSPACE_DIR/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    . "$WORKSPACE_DIR/.env"
    set +a
fi
if [ -n "${WANDB_API_UCMO:-}" ]; then
    export WANDB_API_KEY="${WANDB_API_UCMO}"
fi

# /dev/shm: prefer larger remount for A100 + many workers (plan: up to ~16G when possible)
# df -Pk "Size" is in 1K-blocks (1024-byte units); 16 GiB => 16 * 1024 * 1024 blocks
SHM_REMOUNTED=false
SHM_MIN_KB=$((16 * 1024 * 1024))
if [ -d "/dev/shm" ]; then
    SHM_SIZE=$(df -Pk /dev/shm | grep -v Filesystem | awk '{print $2}')
    if [ "$SHM_SIZE" -lt "$SHM_MIN_KB" ]; then
        echo "WARNING: /dev/shm < ${SHM_TARGET} target ($((SHM_SIZE / 1024))MB). Attempting remount to ${SHM_TARGET}..."
        if sudo mount -o remount,size="${SHM_TARGET}" /dev/shm 2>/dev/null; then
            echo "  Shared memory remounted to ${SHM_TARGET}."
            SHM_REMOUNTED=true
        elif sudo mount -o remount,size=2G /dev/shm 2>/dev/null; then
            echo "  Fallback: remounted /dev/shm to 2G."
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
echo "  A100 merged run: A100_TIER=${A100_TIER:-normal}  RESUME=$RESUME"
echo "  dataset=$DATASET  output=$OUTPUT  steps=$STEPS  job_name=$JOB_NAME"
echo "================================================================"
echo ""
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
LOG_FILE="$LOG_DIR/dino_map_merged_a100_${TIMESTAMP}.log"

LAST_REL="$OUTPUT/checkpoints/last"
# Canonical + alternate absolute layouts (Principia uses /data/lehome_workspace; legacy cloud images may use /root/data/...)
LAST_CANONICAL="$CHALLENGE_DIR/$OUTPUT/checkpoints/last"
LAST_ABS_DATA="/data/lehome_workspace/lehome-challenge/$OUTPUT/checkpoints/last"
LAST_ABS_LEGACY_ROOT="/root/data/lehome_workspace/lehome-challenge/$OUTPUT/checkpoints/last"

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
        _skip_last=0
        [ -d "$LAST_REL" ] && _skip_last=1
        [ -d "$LAST_CANONICAL" ] && _skip_last=1
        [ -d "$LAST_ABS_DATA" ] && _skip_last=1
        if [ "${INCLUDE_LEGACY_VM_LAST_PATHS:-0}" = "1" ] && [ -d "$LAST_ABS_LEGACY_ROOT" ]; then _skip_last=1; fi
        if [ "$_skip_last" = "1" ]; then
            echo "SKIP_IF_LAST_EXISTS=1: checkpoint exists (last/). Exiting 0."
            exit 0
        fi
    fi
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

RCLONE_REMOTE_NAME="${RCLONE_REMOTE:-gdrive}"
RCLONE_MIN_AGE="${RCLONE_MIN_AGE:-15m}"
RCLONE_SLEEP_SEC="${RCLONE_SLEEP_SEC:-1800}"
WANDB_PROJECT="${WANDB_PROJECT:-lehome_challenge}"
RUN_TAG="${JOB_NAME}__$(date +%Y%m%d_%H%M%S)__$(hostname -s)"
RCLONE_BASE="LeHome/models/${WANDB_PROJECT}/${JOB_NAME}/${RUN_TAG}"
RCLONE_DST="${RCLONE_REMOTE_NAME}:${RCLONE_BASE}/"
RCLONE_LOG="$CHALLENGE_DIR/logs/readout/dino_map_merged_a100_rclone_${TIMESTAMP}.log"
RCLONE_BG_PID=""

if [ "${ENABLE_RCLONE_CHECKPOINT_SYNC:-0}" = 1 ]; then
    if ! command -v rclone >/dev/null 2>&1; then
        echo "ERROR: ENABLE_RCLONE_CHECKPOINT_SYNC=1 but rclone not found in PATH"
        exit 1
    fi
    echo "rclone: enabled"
    (
        while true; do
            SRC_DIR="$OUTPUT/checkpoints"
            if [ -d "$SRC_DIR" ]; then
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

cleanup_rclone() {
    if [ -n "${RCLONE_BG_PID:-}" ]; then
        kill "$RCLONE_BG_PID" 2>/dev/null || true
        wait "$RCLONE_BG_PID" 2>/dev/null || true
        RCLONE_BG_PID=""
        echo "rclone: final sync"
        if [ -d "$OUTPUT/checkpoints" ]; then
            if [ -d "$OUTPUT/checkpoints/last" ]; then
                rclone copy "$OUTPUT/checkpoints/last" "${RCLONE_DST}checkpoints/last" \
                    --log-file "$RCLONE_LOG" --log-level INFO \
                    || true
            fi
            rclone move "$OUTPUT/checkpoints" "${RCLONE_DST}checkpoints" \
                --filter '+ step_*/**' \
                --filter '+ [0-9]*/**' \
                --filter '- **' \
                --delete-empty-src-dirs \
                --log-file "$RCLONE_LOG" --log-level INFO \
                || true
        fi
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
