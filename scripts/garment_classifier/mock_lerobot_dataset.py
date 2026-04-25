#!/usr/bin/env python3
import os
from pathlib import Path
import torch
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def create_mock_dataset(name, root):
    features = {
        "observation.images.top_rgb": {
            "dtype": "video",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["j1", "j2", "j3", "j4", "j5", "j6"],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["a1", "a2", "a3", "a4", "a5", "a6"],
        },
    }
    
    dataset = LeRobotDataset.create(
        repo_id=name,
        root=root,
        fps=30,
        features=features,
        use_videos=False, # Use images for mock simplicity
    )
    
    # Add 2 episodes
    for ep in range(2):
        for frame in range(5):
            obs = {
                "observation.images.top_rgb": torch.randint(0, 256, (224, 224, 3), dtype=torch.uint8),
                "observation.state": torch.randn(6),
                "action": torch.randn(6),
                "task": f"mock task {name}",
            }
            dataset.add_frame(obs)
        dataset.save_episode()
    dataset.finalize()
    print(f"Mock dataset {name} created at {root}")

if __name__ == "__main__":
    base_root = Path("mock_datasets")
    for garment in ["top_long", "top_short", "pant_long", "pant_short"]:
        create_mock_dataset(f"{garment}_merged", base_root / f"{garment}_merged")
