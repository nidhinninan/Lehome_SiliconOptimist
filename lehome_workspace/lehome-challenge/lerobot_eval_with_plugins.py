#!/usr/bin/env python
"""
lerobot_eval_with_plugins.py

A wrapper for LeHome evaluation that explicitly imports third-party LeRobot plugins.
This prevents config decode errors like:
  "Couldn't find a choice class for 'dino_diffusion' in PreTrainedConfig"
when the plugin package isn't imported (and thus doesn't register its config types).
"""

import sys
from pathlib import Path


def _register_plugins() -> None:
    # Plugin repos live next to lehome-challenge in /data/lehome_workspace/
    workspace_dir = Path(__file__).resolve().parents[1]

    # Ensure we import the *real* packages (not empty namespace folders).
    sys.path.insert(0, str(workspace_dir / "lerobot_policy_dino" / "src"))
    sys.path.insert(0, str(workspace_dir / "lerobot_policy_clip" / "src"))

    for pkg in ("lerobot_policy_dino", "lerobot_policy_clip"):
        try:
            mod = __import__(pkg)
            if getattr(mod, "__file__", None) is None:
                raise ImportError("Imported an empty namespace package.")
            print(f"✅ Registered plugin: {pkg}")
        except ImportError as e:
            print(f"⚠️  Warning: {pkg} not found or broken ({e})")


def main() -> None:
    _register_plugins()

    # Import after registration so config subclasses are available during decode.
    from scripts.eval import main as eval_main

    eval_main()


if __name__ == "__main__":
    main()

