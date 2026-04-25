#!/usr/bin/env python3
"""
Export one frame per episode from four LeRobot garment datasets into ImageFolder layout
for training a 4-way garment classifier.

Uses LeRobot v3 episode metadata (DeepWiki / upstream): global index =
    dataset_from_index + frame_idx
with bounds check frame_idx < length.

Requires: pip install lerobot torch torchvision pillow

Example:
  python scripts/garment_classifier/export_garment_classifier_dataset.py \\
    --output_dir ./garment_classifier_data \\
    --top_long_repo top_long_merged --top_long_root /path/to/top_long_merged \\
    --top_short_repo top_short_merged --top_short_root /path/to/top_short_merged \\
    --pant_long_repo pant_long_merged --pant_long_root /path/to/pant_long_merged \\
    --pant_short_repo pant_short_merged --pant_short_root /path/to/pant_short_merged
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "lerobot is required. Install with: pip install lerobot\n" f"Original error: {e}"
    ) from e

# Stable class order (must match train script / router checkpoint mapping).
CLASS_ORDER = ("top_long", "top_short", "pant_long", "pant_short")

# Primary LeHome feature key; eval env may expose the same or an alias (handled in router).
IMAGE_KEY_PRIMARY = "observation.images.top_rgb"
IMAGE_KEY_ALIASES = (
    "observation.images.top",
    "observation.image.top_rgb",
)


def _scalar(row: dict[str, Any], key: str) -> int:
    """Episode parquet rows may store ints or single-element lists."""
    v = row[key]
    if isinstance(v, (list, tuple)):
        return int(v[0])
    return int(v)


def _tensor_to_uint8_hwc(img: Any) -> np.ndarray:
    """Convert LeRobot image tensor to uint8 HWC RGB."""
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


def _pick_image_key(sample: dict[str, Any]) -> str:
    if IMAGE_KEY_PRIMARY in sample:
        return IMAGE_KEY_PRIMARY
    for k in IMAGE_KEY_ALIASES:
        if k in sample:
            return k
    raise KeyError(
        f"No image key in sample. Tried {IMAGE_KEY_PRIMARY} and {IMAGE_KEY_ALIASES}. "
        f"Keys: {sorted(sample.keys())[:40]}..."
    )


def export_class(
    repo_id: str,
    root: Path,
    class_name: str,
    output_dir: Path,
    frame_idx: int,
    val_ratio: float,
    rng: random.Random,
    max_episodes: int | None,
) -> tuple[int, int, int]:
    """
    Returns (saved_train, saved_val, skipped_short).
    """
    ds = LeRobotDataset(repo_id, root=root)
    meta = ds.meta
    if hasattr(meta, "ensure_readable"):
        meta.ensure_readable()
    episodes = meta.episodes
    if episodes is None:
        raise RuntimeError(f"No episodes metadata for {repo_id} @ {root}")

    n_ep_total = len(episodes)
    n_ep = min(n_ep_total, max_episodes) if max_episodes is not None else n_ep_total

    shuffled = list(range(n_ep))
    rng.shuffle(shuffled)
    n_val = int(round(n_ep * val_ratio))
    val_episodes = set(shuffled[:n_val])

    saved_train = saved_val = skipped = 0

    for ep_idx in range(n_ep):
        ep = episodes[ep_idx]
        length = _scalar(ep, "length")
        if frame_idx >= length:
            skipped += 1
            continue
        global_idx = _scalar(ep, "dataset_from_index") + frame_idx
        sample = ds[global_idx]
        key = _pick_image_key(sample)
        hwc = _tensor_to_uint8_hwc(sample[key])

        split = "val" if ep_idx in val_episodes else "train"
        out_sub = output_dir / split / class_name
        out_sub.mkdir(parents=True, exist_ok=True)
        out_path = out_sub / f"{class_name}_ep{ep_idx:05d}_f{frame_idx}.png"
        Image.fromarray(hwc, mode="RGB").save(out_path)

        if split == "val":
            saved_val += 1
        else:
            saved_train += 1

    return saved_train, saved_val, skipped


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--frame_idx", type=int, default=0, help="Intra-episode frame index (default 0).")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_episodes", type=int, default=None, help="Cap episodes per class (debug).")
    for c in CLASS_ORDER:
        p.add_argument(
            f"--{c}_repo",
            type=str,
            required=True,
            help=f"repo_id label for LeRobotDataset when loading {c}",
        )
        p.add_argument(
            f"--{c}_root",
            type=Path,
            required=True,
            help=f"Local root path for {c} merged dataset",
        )

    args = p.parse_args()
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    label_map = {name: i for i, name in enumerate(CLASS_ORDER)}
    (args.output_dir / "label_map.json").write_text(json.dumps(label_map, indent=2) + "\n")

    summary: dict[str, Any] = {
        "frame_idx": args.frame_idx,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "classes": {},
    }

    for c in CLASS_ORDER:
        repo = getattr(args, f"{c}_repo")
        root = getattr(args, f"{c}_root")
        tr, va, sk = export_class(
            repo_id=repo,
            root=root,
            class_name=c,
            output_dir=args.output_dir,
            frame_idx=args.frame_idx,
            val_ratio=args.val_ratio,
            rng=rng,
            max_episodes=args.max_episodes,
        )
        summary["classes"][c] = {"train": tr, "val": va, "skipped_short_episode": sk}

    (args.output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
