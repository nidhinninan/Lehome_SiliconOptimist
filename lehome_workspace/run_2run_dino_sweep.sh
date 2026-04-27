#!/bin/bash
# run_2run_dino_sweep.sh — Two-run DINO sweep + optional rclone upload to Google Drive.
#
# Same sweep order as run_2run_dino_sweep_no-gdrive.sh (MAP+registers, then baseline), with:
#   - Robust WORKSPACE_DIR (script location, not cwd).
#   - Optional background rclone: moves only step_* checkpoints during training; final
#     flush moves last/ and remainder after training exits.
#   - Skip/resume: skips if checkpoints/last exists OR output_dir/.complete exists (so
#     runs are not restarted after checkpoints were moved off-VM).
#
# W&B (VM): place `.env` under this workspace with WANDB_API_UCMO=<your key>.
#
# Google Drive (VM): configure rclone remote named `gdrive` (see rclone config docs).
#   Set ENABLE_RCLONE_CHECKPOINT_SYNC=1 to enable uploads.
#
# Environment (optional):
#   ENABLE_RCLONE_CHECKPOINT_SYNC=0|1   default 0 — must set 1 to upload/move.
#   RCLONE_REMOTE=gdrive                  default gdrive — remote name only (no colon).
#   RCLONE_MIN_AGE=15m                    min file age before moving step_* (mtime-based).
#   RCLONE_SLEEP_SEC=600                  seconds between background rclone passes.
#
# Usage:
#   cd /path/to/lehome_workspace
#   chmod +x run_2run_dino_sweep.sh
#   ENABLE_RCLONE_CHECKPOINT_SYNC=1 ./run_2run_dino_sweep.sh
#
# CLI matches LeRobot v0.4.3 / lerobot_train_with_plugins.py (see run_10k_sweep.sh).

set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

# --- YAML helpers (top-level keys only; sufficient for sweep YAMLs) ------------
yaml_scalar() {
    # yaml_scalar "job_name" file.yaml -> value after colon
    local key=$1
    local file=$2
    grep -E "^[[:space:]]*${key}:" "$file" | head -1 | sed -E "s/^[[:space:]]*${key}:[[:space:]]*//; s/^['\"]//; s/['\"]$//"
}

yaml_wandb_project() {
    local file=$1
    # Indented `project:` under wandb (these sweep YAMLs have no other nested `project:`).
    grep -E "^[[:space:]]+project:" "$file" | head -1 \
        | sed -E 's/^[[:space:]]+project:[[:space:]]*//; s/^['\''"]//; s/['\''"]$//'
}

run_sweep_run() {
    local NAME=$1
    local CONFIG=$2

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  SWEEP RUN: $NAME"
    echo "  Config: $CONFIG"
    echo "════════════════════════════════════════════════════════════"

    local OUTPUT_DIR
    OUTPUT_DIR=$(yaml_scalar "output_dir" "$CONFIG")
    if [ -z "$OUTPUT_DIR" ]; then
        echo "❌ ERROR: Could not parse output_dir from $CONFIG"
        return 1
    fi

    local JOB_NAME WANDB_PROJECT
    JOB_NAME=$(yaml_scalar "job_name" "$CONFIG")
    WANDB_PROJECT=$(yaml_wandb_project "$CONFIG")
    if [ -z "$JOB_NAME" ] || [ -z "$WANDB_PROJECT" ]; then
        echo "❌ ERROR: Could not parse job_name or wandb.project from $CONFIG"
        return 1
    fi

    local CONFIG_STEM
    CONFIG_STEM=$(basename "$CONFIG" .yaml)

    local CHECK_REL="outputs/sweep_10k/$NAME/checkpoints/last"
    local CHECK_ABS="/root/data/lehome_workspace/outputs/sweep_10k/$NAME/checkpoints/last"
    local CHECKPOINTS_REL="outputs/sweep_10k/$NAME/checkpoints"
    local CHECKPOINTS_ABS="/root/data/lehome_workspace/outputs/sweep_10k/$NAME/checkpoints"
    local COMPLETE_FILE="$OUTPUT_DIR/.complete"

    if [ -f "$COMPLETE_FILE" ]; then
        echo "   SKIPPING: completion marker exists: $COMPLETE_FILE"
        return 0
    fi
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

    local RUN_TAG="${JOB_NAME}__${CONFIG_STEM}__$(date +%Y%m%d_%H%M%S)__$(hostname -s)"
    local RCLONE_REMOTE_NAME="${RCLONE_REMOTE:-gdrive}"
    local RCLONE_MIN_AGE="${RCLONE_MIN_AGE:-15m}"
    local RCLONE_SLEEP_SEC="${RCLONE_SLEEP_SEC:-600}"
    local RCLONE_BASE="LeHome/models/${WANDB_PROJECT}/${JOB_NAME}/${CONFIG_STEM}__${RUN_TAG}"
    local RCLONE_DST="${RCLONE_REMOTE_NAME}:${RCLONE_BASE}/"
    local RCLONE_LOG="$WORKSPACE_DIR/lehome-challenge/logs/readout/$NAME-rclone.log"

    local RCLONE_BG_PID=""
    if [ "${ENABLE_RCLONE_CHECKPOINT_SYNC:-0}" = 1 ]; then
        if ! command -v rclone >/dev/null 2>&1; then
            echo "❌ ERROR: ENABLE_RCLONE_CHECKPOINT_SYNC=1 but rclone not found in PATH"
            return 1
        fi
        echo "   rclone: background move of step_* only -> ${RCLONE_DST}checkpoints"
        echo "   rclone: min-age=${RCLONE_MIN_AGE} sleep=${RCLONE_SLEEP_SEC}s log=$RCLONE_LOG"
        (
            while true; do
                if [ -d "$CHECKPOINTS_ABS" ] || [ -d "$WORKSPACE_DIR/$CHECKPOINTS_REL" ]; then
                    LOOP_SRC="$CHECKPOINTS_ABS"
                    [ -d "$LOOP_SRC" ] || LOOP_SRC="$WORKSPACE_DIR/$CHECKPOINTS_REL"
                    rclone move "$LOOP_SRC" "${RCLONE_DST}checkpoints" \
                        --filter '+ step_*/**' \
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
        echo "   rclone: disabled (set ENABLE_RCLONE_CHECKPOINT_SYNC=1 to upload to Drive)"
    fi

    "$VENV_PYTHON" "$WORKSPACE_DIR/lerobot_train_with_plugins.py" \
        --config_path="$CONFIG" \
        --steps="$STEPS" \
        --eval_freq="$EVAL_FREQ" \
        --dataset.image_transforms.enable=false \
        #--dataset.video_backend=pyav \
        --num_workers="$WORKERS" \
        2>&1 | tee "$LOG_FILE"

    if [ -n "$RCLONE_BG_PID" ]; then
        kill "$RCLONE_BG_PID" 2>/dev/null || true
        wait "$RCLONE_BG_PID" 2>/dev/null || true
        echo "   rclone: final flush (including last/) -> ${RCLONE_DST}checkpoints"
        if [ -d "$CHECKPOINTS_ABS" ] || [ -d "$WORKSPACE_DIR/$CHECKPOINTS_REL" ]; then
            local CHECK_SRC="$CHECKPOINTS_ABS"
            [ -d "$CHECK_SRC" ] || CHECK_SRC="$WORKSPACE_DIR/$CHECKPOINTS_REL"
            rclone move "$CHECK_SRC" "${RCLONE_DST}checkpoints" \
                --delete-empty-src-dirs \
                --log-file "$RCLONE_LOG" --log-level INFO
        fi
        mkdir -p "$OUTPUT_DIR"
        touch "$COMPLETE_FILE"
        echo "   rclone: wrote completion marker $COMPLETE_FILE"
    else
        mkdir -p "$OUTPUT_DIR"
        touch "$COMPLETE_FILE"
        echo "   Wrote completion marker $COMPLETE_FILE (no rclone this run)"
    fi

    echo "  Done: $NAME"
}

# Order: MAP+registers first, then original baseline
run_sweep_run "DINOv2_MAP_Registers" "$CONFIGS_DIR/sweep_dino_map_registers.yaml"
run_sweep_run "DINOv2_Baseline" "$CONFIGS_DIR/sweep_dino_baseline.yaml"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Two-run DINO sweep finished."
echo "  Local outputs: outputs/sweep_10k/DINOv2_MAP_Registers and DINOv2_Baseline"
echo "  Drive (if enabled): gdrive:LeHome/models/<project>/<job_name>/..."
echo "  W&B project: lehome_challenge (run names from job_name in YAML)"
echo "════════════════════════════════════════════════════════════"
