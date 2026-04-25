#!/bin/bash

# LeHome Unified Visualization & Inspection Script
# Designed for both VM (IsaacSim enabled) and Local Machine (Rerun only)

MODE=$1
DATASET_ID=${2:-"top_long"}
ROOT_DIR=${3:-"Datasets/lehome_merged"}

if [ -z "$MODE" ]; then
    echo "Usage: ./visualize_dataset.sh [mode] [dataset_id] [root_dir]"
    echo "Modes: viz, replay, inspect"
    echo "Example: ./visualize_dataset.sh viz top_long"
    exit 1
fi

# Detect environment
HAS_ISAAC=false
if python -c "import isaacsim" &> /dev/null; then
    HAS_ISAAC=true
fi

case $MODE in
    "viz")
        echo "Launching Rerun viewer for $DATASET_ID..."
        # root maps to the folder CONTAINING the dataset for the viz tool
        # repo-id is the folder name itself
        lerobot-dataset-viz --root "$ROOT_DIR" --repo-id "$DATASET_ID" --display-compressed-images true
        ;;

    "replay")
        if [ "$HAS_ISAAC" = false ]; then
            echo "Error: IsaacSim not found in this environment. 'replay' mode is only available on the VM."
            exit 1
        fi
        echo "Replaying $DATASET_ID in simulation..."
        python -m scripts.dataset replay --dataset_id "$DATASET_ID"
        ;;

    "inspect")
        echo "Inspecting metadata for $DATASET_ID..."
        # Try native Hackathon script first
        if python -m scripts.dataset inspect --dataset_id "$DATASET_ID" &> /dev/null; then
            python -m scripts.dataset inspect --dataset_id "$DATASET_ID"
        else
            echo "Falling back to lerobot-edit-dataset info..."
            lerobot-edit-dataset --root "$ROOT_DIR/$DATASET_ID" --operation.type info
        fi
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Available modes: viz, replay, inspect"
        exit 1
        ;;
esac
