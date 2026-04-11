#!/bin/bash
# script to resume training with augmentations enabled

CHECKPOINT_PATH=$1
if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Usage: ./resume_with_augmentations.sh <checkpoint_path>"
    echo "Example: ./resume_with_augmentations.sh outputs/train/dp_top_short/checkpoints/last"
    exit 1
fi

echo "Resuming training from $CHECKPOINT_PATH with image_transforms enabled..."

# Note: We use the absolute python path to bypass Principia VM aliases 
# (as documented in Section 6.0 of installation_guide.md).
/data/lehome_workspace/lehome-challenge/.venv/bin/python \
    -m lerobot.scripts.lerobot_train \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --resume true \
    --dataset.image_transforms.enable true \
    --num_workers 4

echo "Training resumed with augmentations and 4 CPU workers."
