#!/bin/bash
# LeHome Challenge: Parallel Evaluation Script
# This script bypasses the serial bottleneck in evaluation.py by launching
# 4 separate background processes, one for each garment type.
# Ensure you are running this from the lehome-challenge root directory.

echo "================================================="
echo "Starting Parallel Evaluation for 4 Garment Types"
echo "================================================="

# Activating the virtual environment
source .venv/bin/activate

# 1. Top Long
echo "[STARTING] Top Long Evaluation..."
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/dp_top_long/checkpoints/last/pretrained_model \
    --garment_type top_long \
    --dataset_root Datasets/example/top_long_merged \
    --num_episodes 50 \
    --device cpu \
    --headless > outputs/eval_top_long_parallel.log 2>&1 &
TOP_LONG_PID=$!

# 2. Top Short
echo "[STARTING] Top Short Evaluation..."
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/dp_top_short/checkpoints/last/pretrained_model \
    --garment_type top_short \
    --dataset_root Datasets/example/top_short_merged \
    --num_episodes 50 \
    --device cpu \
    --headless > outputs/eval_top_short_parallel.log 2>&1 &
TOP_SHORT_PID=$!

# 3. Pant Long
echo "[STARTING] Pant Long Evaluation..."
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/dp_pant_long/checkpoints/last/pretrained_model \
    --garment_type pant_long \
    --dataset_root Datasets/example/pant_long_merged \
    --num_episodes 50 \
    --device cpu \
    --headless > outputs/eval_pant_long_parallel.log 2>&1 &
PANT_LONG_PID=$!

# 4. Pant Short
echo "[STARTING] Pant Short Evaluation..."
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/dp_pant_short/checkpoints/last/pretrained_model \
    --garment_type pant_short \
    --dataset_root Datasets/example/pant_short_merged \
    --num_episodes 50 \
    --device cpu \
    --headless > outputs/eval_pant_short_parallel.log 2>&1 &
PANT_SHORT_PID=$!

echo "================================================="
echo "All 4 evaluations launched in the background!"
echo "You can monitor GPU usage with 'nvidia-smi'."
echo "To check progress, tail the log files:"
echo "  tail -f outputs/eval_top_long_parallel.log"
echo "================================================="

# Wait for all background processes to finish
wait $TOP_LONG_PID
wait $TOP_SHORT_PID
wait $PANT_LONG_PID
wait $PANT_SHORT_PID

echo "================================================="
echo "ALL EVALUATIONS COMPLETED!"
echo "================================================="
