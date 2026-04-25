#!/usr/bin/env python3
"""
Sanity check script to visually audit `frame_idx=0` images from LeRobot Datasets.
This script extracts 15-20 random instances from each garment class and saves them,
allowing you to quickly verify that the first frame guarantees the garment is fully laid out.

Example:
  python scripts/garment_classifier/sanity_check_first_frames.py \
    --output_dir ./garment_classifier_data/sanity_check \
    --top_long_repo top_long_merged --top_long_root /path/to/top_long_merged \
    --top_short_repo top_short_merged --top_short_root /path/to/top_short_merged \
    --pant_long_repo pant_long_merged --pant_long_root /path/to/pant_long_merged \
    --pant_short_repo pant_short_merged --pant_short_root /path/to/pant_short_merged
"""

import argparse
import random
import os
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as e:
    raise SystemExit(
        "lerobot is required. Install with: pip install lerobot\n" f"Original error: {e}"
    ) from e

CLASS_ORDER = ("top_long", "top_short", "pant_long", "pant_short")
IMAGE_KEYS = ("observation.images.top_rgb", "observation.images.top")

def _scalar(row: dict, key: str) -> int:
    v = row[key]
    if isinstance(v, (list, tuple)):
        return int(v[0])
    return int(v)

def _tensor_to_uint8_hwc(img) -> np.ndarray:
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0 + 1e-6:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    return arr

def extract_sanity_frames(
    repo_id: str,
    root: Path,
    class_name: str,
    output_dir: Path,
    frame_idx: int,
    num_samples: int,
    rng: random.Random,
):
    print(f"Loading '{class_name}' dataset from {repo_id}...")
    ds = LeRobotDataset(repo_id, root=root)
    meta = ds.meta
    episodes = meta.episodes
    if episodes is None:
        print(f"WARNING: No episodes metadata found for {class_name}.")
        return

    n_ep_total = len(episodes)
    sample_size = min(n_ep_total, num_samples)
    sampled_indices = rng.sample(range(n_ep_total), sample_size)

    out_folder = output_dir / class_name
    out_folder.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for ep_idx in sampled_indices:
        ep = episodes[ep_idx]
        length = _scalar(ep, "length")
        
        if frame_idx >= length:
            print(f"Skipping episode {ep_idx} in {class_name}: Length ({length}) <= frame_idx ({frame_idx})")
            continue
            
        global_idx = _scalar(ep, "dataset_from_index") + frame_idx
        sample = ds[global_idx]
        
        # Find exactly the top RGB camera view
        img_key = None
        for k in IMAGE_KEYS:
            if k in sample:
                img_key = k
                break
                
        if not img_key:
            print(f"Skipping episode {ep_idx} in {class_name}: Missing keys {IMAGE_KEYS}")
            continue

        hwc = _tensor_to_uint8_hwc(sample[img_key])
        out_path = out_folder / f"ep{ep_idx:04d}_frame{frame_idx}.png"
        Image.fromarray(hwc, mode="RGB").save(out_path)
        saved_count += 1

    print(f"Extracted {saved_count}/{sample_size} valid sanity frames for {class_name}.\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", type=Path, required=True, help="Folder to save sanity check images.")
    p.add_argument("--frame_idx", type=int, default=0, help="Frame index to check (default is 0).")
    p.add_argument("--samples_per_class", type=int, default=20, help="Number of episodes to visually inspect.")
    p.add_argument("--seed", type=int, default=42)
    
    for c in CLASS_ORDER:
        p.add_argument(f"--{c}_repo", type=str, required=True, help=f"Dataset repo_id for {c}")
        p.add_argument(f"--{c}_root", type=Path, required=True, help=f"Local dataset root for {c}")

    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"Starting Sanity Check: Sampling {args.samples_per_class} episodes per dataset for Frame IDX: {args.frame_idx}\n")
    for c in CLASS_ORDER:
        repo = getattr(args, f"{c}_repo")
        root = getattr(args, f"{c}_root")
        extract_sanity_frames(
            repo_id=repo,
            root=root,
            class_name=c,
            output_dir=args.output_dir,
            frame_idx=args.frame_idx,
            num_samples=args.samples_per_class,
            rng=rng,
        )

    print(f"Sanity Check Complete! Please review images visually in: {args.output_dir.absolute()}")

if __name__ == "__main__":
    main()
