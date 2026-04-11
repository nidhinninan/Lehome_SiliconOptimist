#!/bin/bash
# Initialize conda properly for scripting
source ~/anaconda3/etc/profile.d/conda.sh || { echo "Conda not found at ~/anaconda3. Please check your installation path."; exit 1; }

echo "Creating 'lehome' conda environment with Python 3.11..."
# Create environment with Python 3.11 to match VM requirements
conda create -n lehome python=3.11 -y
conda activate lehome

echo "Installing ffmpeg..."
# Install system dependencies with explicitly required codecs for LeRobot
conda install ffmpeg -c conda-forge -y

echo "Installing LeRobot v0.4.3..."
# Install matched version of LeRobot
pip install lerobot==0.4.3

echo "Installing Hugging Face CLI for dataset management..."
# Install HF CLI for downloading datasets
pip install "huggingface-hub[cli,hf-transfer]"

echo "--------------------------------------------------------"
echo "Local environment setup complete!"
echo "To activate:  conda activate lehome"
echo "To download a dataset chunk (e.g. top_long) to local machine:"
echo "huggingface-cli download lehome/dataset_challenge_merged --repo-type dataset --include \"data/top_long/*\" --local-dir Datasets/lehome_merged"
echo "--------------------------------------------------------"
