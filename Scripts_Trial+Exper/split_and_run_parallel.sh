#!/bin/bash
# Automatically splits a target garment list and evaluates chunks in 4 parallel shells.

WORKSPACE_DIR="/data/lehome_workspace/lehome-challenge"
cd $WORKSPACE_DIR
source .venv/bin/activate

# 1. Define the specific Garment Type to split
GARMENT_TYPE=$1
if [ -z "$GARMENT_TYPE" ]; then
    echo "Usage: ./split_and_run_parallel.sh <garment_type>"
    echo "Example: ./split_and_run_parallel.sh pant_long"
    exit 1
fi

case $GARMENT_TYPE in
    top_short) DIR_NAME="Top_Short" ;;
    top_long) DIR_NAME="Top_Long" ;;
    pant_short) DIR_NAME="Pant_Short" ;;
    pant_long) DIR_NAME="Pant_Long" ;;
    *) echo "Error: Unknown garment type: $GARMENT_TYPE. Use top_short, top_long, pant_short, or pant_long."; exit 1 ;;
esac

LIST_FILE="Assets/objects/Challenge_Garment/Release/${DIR_NAME}/${DIR_NAME}.txt"

echo "Checking if $LIST_FILE exists..."
if [ ! -f "$LIST_FILE" ]; then
    echo "Error: $LIST_FILE does not exist!"
    exit 1
fi

mkdir -p outputs

# 2. Clean up any previous splits and generate 4 new chunks 
# (using standard prefix split_... to avoid clutter)
rm -f Assets/objects/Challenge_Garment/Release/${DIR_NAME}/split_${GARMENT_TYPE}_*.txt

echo "Splitting $LIST_FILE into 4 chunks..."
# split uses l/4 to split by lines into exactly 4 chunks cleanly without breaking names
split -n l/4 -d --additional-suffix=.txt "$LIST_FILE" "Assets/objects/Challenge_Garment/Release/${DIR_NAME}/split_${GARMENT_TYPE}_"

# 3. Launch the Evaluation Process for each Chunk
NUM=0
for SPLIT_FILE in Assets/objects/Challenge_Garment/Release/${DIR_NAME}/split_${GARMENT_TYPE}_*.txt; do
    echo "Launching shell $NUM for chunk: $SPLIT_FILE"
    
    python -m scripts.eval \
        --policy_type lerobot \
        --policy_path outputs/train/dp_${GARMENT_TYPE}/checkpoints/last/pretrained_model \
        --garment_type $GARMENT_TYPE \
        --custom_list_path "$SPLIT_FILE" \
        --num_episodes 5 \
        --step_hz 0 \
        --headless \
        --device cpu > outputs/eval_split_${DIR_NAME}_${NUM}.log 2>&1 &
        
    NUM=$((NUM+1))
done

echo "✅ All 4 parallel evaluations for $GARMENT_TYPE have been launched!"
echo "Use 'htop' to monitor cpu utilization across the 4 python instances."
wait
echo "🎉 Done! The evaluations have completed."
