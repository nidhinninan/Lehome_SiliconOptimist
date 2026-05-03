#!/bin/bash
# This script runs evaluations for all 4 garment types concurrently.
# By passing `--step_hz 0`, the simulation runs at maximum possible speed instead of sleeping.

WORKSPACE_DIR="/data/lehome_workspace/lehome-challenge"
cd $WORKSPACE_DIR
source .venv/bin/activate

# Create outputs dir for logs if it doesn't exist
mkdir -p outputs

echo "Launching Evaluation: Top Long..."
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/dp_top_long/checkpoints/last/pretrained_model \
    --garment_type top_long \
    --num_episodes 5 \
    --step_hz 0 \
    --headless \
    --device cpu > outputs/eval_top_long.log 2>&1 &

echo "Launching Evaluation: Top Short..."
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/dp_top_short/checkpoints/last/pretrained_model \
    --garment_type top_short \
    --num_episodes 5 \
    --step_hz 0 \
    --headless \
    --device cpu > outputs/eval_top_short.log 2>&1 &

echo "Launching Evaluation: Pant Long..."
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/dp_pant_long/checkpoints/last/pretrained_model \
    --garment_type pant_long \
    --num_episodes 5 \
    --step_hz 0 \
    --headless \
    --device cpu > outputs/eval_pant_long.log 2>&1 &

echo "Launching Evaluation: Pant Short..."
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/dp_pant_short/checkpoints/last/pretrained_model \
    --garment_type pant_short \
    --num_episodes 5 \
    --step_hz 0 \
    --headless \
    --device cpu > outputs/eval_pant_short.log 2>&1 &

echo "✅ All 4 evaluations have been launched in the background!"
echo "Use 'htop' or 'nvidia-smi' to monitor execution."
echo "Logs are being written locally to outputs/eval_*.log"

# Wait for all background processes to finish
wait
echo "🎉 All parallel evaluations are complete!"
