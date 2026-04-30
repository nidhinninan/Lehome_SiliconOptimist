import os
import argparse
import wandb
from pathlib import Path

def download_artifact(project: str, artifact_name: str, download_dir: str):
    """
    Downloads a W&B artifact locally so it can be baked into a Docker image.
    This avoids putting your W&B API key inside the Dockerfile.
    """
    print(f"Downloading artifact {artifact_name} (project arg: {project})...")
    
    # Initialize a W&B API client
    api = wandb.Api()
    
    # Fetch the artifact.
    #
    # W&B accepts either:
    # - "project/artifact:alias" or "entity/project/artifact:alias"
    # - "project/artifact:version" or "entity/project/artifact:version"
    #
    # We support two input styles:
    # 1) Fully-qualified ref passed via --artifact (contains at least one "/" and a ":")
    # 2) A short name passed via --artifact plus a --project prefix to form "project/<artifact>"
    #
    # Docs: https://docs.wandb.ai/models/ref/python/public-api/api
    if ("/" in artifact_name) and (":" in artifact_name):
        artifact_path = artifact_name
    else:
        artifact_path = f"{project}/{artifact_name}"
    try:
        artifact = api.artifact(artifact_path)
        
        # Download the artifact contents
        path = artifact.download(root=download_dir)
        print(f"✅ Successfully downloaded artifact to: {path}")
    except Exception as e:
        print(f"❌ Error downloading artifact: {e}")
        print("Please ensure your W&B API key is set via `wandb login` or the WANDB_API_KEY environment variable.")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download W&B model artifact for Docker build")
    parser.add_argument("--project", type=str, default="lehome_challenge", help="W&B Project Name")
    parser.add_argument("--artifact", type=str, required=True, help="Artifact name (e.g., 'model-run123:v0' or 'model-run123:latest')")
    parser.add_argument("--out", type=str, default="pretrained_model", help="Output directory to save the checkpoint")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.out, exist_ok=True)
    download_artifact(args.project, args.artifact, args.out)
