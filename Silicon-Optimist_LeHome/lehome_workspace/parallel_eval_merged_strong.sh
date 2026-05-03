#!/bin/bash
# parallel_eval_merged_strong.sh — stronger parallel eval for a single LeRobot checkpoint (merged DINO / DP).
#
# Principia VM: default CHALLENGE_DIR is next to this script; if the workspace lives at
# /data/lehome_workspace, use LEHOME_VM_PROFILE=principia or pass CHALLENGE_DIR=/data/lehome_workspace/lehome-challenge.
# HF transformers cache can use HF_HOME=/data/huggingface_cache (exported when Principia profile is detected).
#
# Based on repo parallel_eval.sh pattern; uses official flags from docs/policy_eval.md and README.
#
# Env (override as needed):
#   POLICY_PATH   Path to .../checkpoints/last/pretrained_model (under lehome-challenge)
#   NUM_EPISODES  Episodes per garment category (default 20)
#   STEP_HZ       0 = max sim speed (default 0)
#   CHALLENGE_DIR Explicit lehome-challenge root (default: $WORKSPACE_DIR/lehome-challenge; Principia often /data/lehome_workspace/lehome-challenge)
#
# Dataset roots: per-garment merged splits (metadata for LeRobot eval), same pattern as root parallel_eval.sh.
#
# Usage (on Principia VM — paths/cache only; default POLICY_PATH is still the A100 merged checkpoint unless you override):
#   LEHOME_VM_PROFILE=principia ./parallel_eval_merged_strong.sh
#   POLICY_PATH=outputs/train/<your_run>/checkpoints/last/pretrained_model ./parallel_eval_merged_strong.sh
#
# Usage (on VM, from lehome_workspace):
#   cd /data/lehome_workspace/lehome-challenge   # or your CHALLENGE path
#   POLICY_PATH=outputs/train/dp_merged_dino_map8_registers_a100_400k_norm/checkpoints/last/pretrained_model \
#     ../parallel_eval_merged_strong.sh
#
# Or from lehome_workspace with absolute challenge dir:
#   CHALLENGE_DIR=/data/lehome_workspace/lehome-challenge bash parallel_eval_merged_strong.sh

set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${LEHOME_VM_PROFILE:-}" == "principia" ]] || [[ "$WORKSPACE_DIR" == /data/lehome_workspace ]] || [[ "$WORKSPACE_DIR" == /data/lehome_workspace/* ]]; then
    export HF_HOME="${HF_HOME:-/data/huggingface_cache}"
    mkdir -p "$HF_HOME" 2>/dev/null || true
    if [[ -d /data/.local/bin ]] && [[ ":$PATH:" != *:/data/.local/bin:* ]]; then
        export PATH="/data/.local/bin:$PATH"
    fi
fi

if [[ "${LEHOME_VM_PROFILE:-}" == "principia" ]] && [[ -z "${CHALLENGE_DIR:-}" ]] && [[ -d "/data/lehome_workspace/lehome-challenge" ]]; then
    CHALLENGE_DIR="/data/lehome_workspace/lehome-challenge"
fi
CHALLENGE_DIR="${CHALLENGE_DIR:-$WORKSPACE_DIR/lehome-challenge}"
# Default checkpoint: A100 merged training artifact (override POLICY_PATH for your run)
POLICY_PATH="${POLICY_PATH:-outputs/train/dp_merged_dino_map8_registers_a100_400k_norm/checkpoints/last/pretrained_model}"
NUM_EPISODES="${NUM_EPISODES:-20}"
STEP_HZ="${STEP_HZ:-0}"

cd "$CHALLENGE_DIR"
# shellcheck disable=SC1091
source .venv/bin/activate

mkdir -p outputs

echo "Policy: $POLICY_PATH"
echo "Episodes per garment: $NUM_EPISODES  step_hz=$STEP_HZ"
echo "Logs under $CHALLENGE_DIR/outputs/eval_merged_strong_*.log"

launch() {
    local gtype="$1"
    local droot="$2"
    local logf="$3"
    python -m scripts.eval \
        --policy_type lerobot \
        --policy_path "$POLICY_PATH" \
        --dataset_root "$droot" \
        --garment_type "$gtype" \
        --num_episodes "$NUM_EPISODES" \
        --step_hz "$STEP_HZ" \
        --headless \
        --device cpu > "$logf" 2>&1 &
}

echo "Launching Evaluation: Top Long..."
launch top_long Datasets/example/top_long_merged outputs/eval_merged_strong_top_long.log

echo "Launching Evaluation: Top Short..."
launch top_short Datasets/example/top_short_merged outputs/eval_merged_strong_top_short.log

echo "Launching Evaluation: Pant Long..."
launch pant_long Datasets/example/pant_long_merged outputs/eval_merged_strong_pant_long.log

echo "Launching Evaluation: Pant Short..."
launch pant_short Datasets/example/pant_short_merged outputs/eval_merged_strong_pant_short.log

echo "All 4 evaluations launched in background. Monitor: htop / nvidia-smi"
wait
echo "All parallel evaluations complete."
