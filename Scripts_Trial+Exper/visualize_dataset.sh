#!/bin/bash

# visualize_dataset.sh
# Script to inspect and visualize local LeRobot datasets on the LeHome VM.
# Incorporates fixes from the dataset-viz audit report.

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -r, --repo-id  Dataset repo ID (e.g., top_long_merged)"
    echo "  -d, --root     Dataset root parent dir (e.g., /data/lehome_workspace/lehome-challenge/Datasets/example)"
    echo "  -e, --episode  Episode index to view (default: 0)"
    echo "  -i, --inspect  Run inspection mode instead of UI visualization"
    echo "  -h, --help     Show this help message"
    exit 1
}

# Default VM paths (adjust if running locally)
REPO_ID="top_long_merged"
ROOT_DIR="/data/lehome_workspace/lehome-challenge/Datasets/example"
EPISODE_IDX=0
INSPECT_ONLY=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -r|--repo-id) REPO_ID="$2"; shift ;;
        -d|--root) ROOT_DIR="$2"; shift ;;
        -e|--episode) EPISODE_IDX="$2"; shift ;;
        -i|--inspect) INSPECT_ONLY=1 ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

FULL_PATH="${ROOT_DIR}/${REPO_ID}"

if [ ! -d "$FULL_PATH" ]; then
    echo "ERROR: Dataset not found at $FULL_PATH"
    echo "Make sure the dataset is downloaded. To download a dataset, run:"
    echo "  pip install \"huggingface-hub[cli,hf-transfer]\""
    echo "  hf download lehome/dataset_challenge_merged --repo-type dataset --local-dir Datasets/example"
    exit 1
fi

echo "=========================================="
echo " Dataset root: $ROOT_DIR"
echo " Repo ID:      $REPO_ID"
echo " Full Path:    $FULL_PATH"
echo "=========================================="

if [ "$INSPECT_ONLY" -eq 1 ]; then
    echo "==> Running Inspection Mode..."
    # Check if LeHome specific scripts are available (on the VM)
    if python -c "import scripts.dataset" 2>/dev/null; then
        echo "Using LeHome dataset inspect script for rich metadata..."
        python -m scripts.dataset inspect \
            --dataset_root "$FULL_PATH" \
            --show_frames 5 \
            --show_stats
    else
        echo "LeHome scripts not found, falling back to lerobot inspection..."
        # Notice how --root is passed directly to the dataset directory for lerobot-edit-dataset
        lerobot-edit-dataset \
            --repo_id "$REPO_ID" \
            --root "$FULL_PATH" \
            --operation.type info \
            --operation.show_features true
    fi
    exit 0
fi

echo "==> Running UI Visualization Mode (Episode: $EPISODE_IDX)..."

# Ensure lerobot visualization supports local mode (0.4.2+)
lerobot-dataset-viz --help 2>&1 | grep -q "\--mode" || {
    echo "WARNING: --mode flag might not be available in your current lerobot version."
    echo "This is required to view local datasets. If it fails, ensure you have lerobot installed properly."
}

# Notice how --root is passed to the parent directory for lerobot-dataset-viz
lerobot-dataset-viz \
    --repo-id "$REPO_ID" \
    --root "$ROOT_DIR" \
    --mode local \
    --episode-index "$EPISODE_IDX"
