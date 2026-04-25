#!/bin/bash
# LeHome Workspace Sync Script
# Author: Antigravity
# Purpose: Sync the VM workspace from a downloaded updated_files zip.

set -e

WORKSPACE_DIR="/data/lehome_workspace"
STAGING_DIR="$WORKSPACE_DIR/updated_files"
DOWNLOADS_DIR="/home/principia/Downloads"

echo "🔄 Starting Workspace Sync..."

# 1. Dependency Check
echo "📦 [1/6] Checking system dependencies..."
for cmd in unzip rsync; do
    if ! command -v $cmd &> /dev/null; then
        echo "   Installing $cmd..."
        sudo apt update && sudo apt install -y $cmd
    fi
done

# 2. Locate Latest Zip
echo "🔍 [2/6] Locating latest updated_file_*.zip..."
ZIP_FILE=$(ls -t "$DOWNLOADS_DIR"/updated_file_*.zip 2>/dev/null | head -n 1)

if [ -z "$ZIP_FILE" ]; then
    echo "❌ ERROR: No updated_file_*.zip found in $DOWNLOADS_DIR"
    exit 1
fi
echo "   Found: $ZIP_FILE"

# 3. Extraction (Directly to workspace root, creates updated_files/ staging area)
echo "📂 [3/6] Extracting to $WORKSPACE_DIR..."
# Per your feedback, unzipping directly into the workspace root is correct 
# because the zip itself contains the 'updated_files' folder. 
unzip -o "$ZIP_FILE" -d "$WORKSPACE_DIR"

# 4. Rsync Sync to Workspace root
echo "🚀 [4/6] Syncing files from staging to root..."
# We sync from the staging area created by the unzip command.
rsync -avh --progress "$WORKSPACE_DIR/updated_files/" "$WORKSPACE_DIR/"

# 5. Post-Sync: Environment Refresh (Insights from 'Syncing LeHome Workspace Files')
echo "🛠️ [5/6] Refreshing Python environment..."
REPO_DIR="$WORKSPACE_DIR/lehome-challenge"
VENV_PYTHON="$REPO_DIR/.venv/bin/python"

if [ -f "$VENV_PYTHON" ]; then
    # Re-install BYOP packages if they exist in the sync
    if [ -d "$WORKSPACE_DIR/lerobot_policy_dino" ]; then
        echo "   Re-installing DINOv2 BYOP..."
        uv pip install -e "$WORKSPACE_DIR/lerobot_policy_dino" --python "$VENV_PYTHON"
    fi
    if [ -d "$WORKSPACE_DIR/lerobot_policy_clip" ]; then
        echo "   Re-installing CLIP BYOP..."
        uv pip install -e "$WORKSPACE_DIR/lerobot_policy_clip" --python "$VENV_PYTHON"
    fi
else
    echo "   ⚠️  Virtual environment not found at $VENV_PYTHON. Skipping pip installs."
fi

# 6. Audit & Logging
echo "📝 [6/6] Updating LeHome Change Log..."
LOG_FILE="$WORKSPACE_DIR/lehome_change_log.md"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

if [ -f "$LOG_FILE" ]; then
    # Find the LOG_START marker or just append
    if grep -q "<!-- LOG_START -->" "$LOG_FILE"; then
        sed -i "/<!-- LOG_START -->/a \
\
### $TIMESTAMP — Workspace Sync Applied\
**Description**:\
- Automatically synced latest files from $(basename "$ZIP_FILE") using sync_workspace.sh.\
- Applied rsync from staging area to root.\
- Refreshed BYOP package links in the venv." "$LOG_FILE"
    else
        echo -e "\n### $TIMESTAMP — Workspace Sync Applied\n- Synced from $(basename "$ZIP_FILE")" >> "$LOG_FILE"
    fi
    echo "   ✅ Change log updated."
else
    echo "   ⚠️  Change log not found at $LOG_FILE."
fi

echo "🎉 Workspace Sync Complete!"
echo "--------------------------------------------------"
echo "Staging area preserved at: $STAGING_DIR"
echo "Current environment: $(ls -lh $WORKSPACE_DIR | grep drwx | head -n 3)"
echo "--------------------------------------------------"
