#!/usr/bin/env python
"""
lerobot_train_with_plugins.py

A wrapper for lerobot-train that explicitly imports third-party plugins.
This bypasses discovery issues where package name normalization (underscores vs hyphens)
prevents LeRobot from automatically finding and registering custom policies.
"""
import sys
from pathlib import Path

# Fix namespace collisions: When running from the workspace root, Python detects
# the uninitialized folder 'lerobot_policy_dino' as a blank namespace package.
# We must inject the src/ directories into sys.path to ensure the *actual* plugin
# code (the __init__.py) is imported.
WORKSPACE_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(WORKSPACE_DIR / "lerobot_policy_dino" / "src"))
sys.path.insert(0, str(WORKSPACE_DIR / "lerobot_policy_clip" / "src"))

# 1. Explicitly import plugins to trigger @register_subclass decorators
try:
    import lerobot_policy_dino
    if getattr(lerobot_policy_dino, '__file__', None) is None:
        raise ImportError("Imported an empty namespace package.")
    print("✅ Registered DINOv2 Plugin")
except ImportError as e:
    print(f"⚠️  Warning: lerobot_policy_dino not found or broken. DINOv2 sweep may fail. ({e})")

try:
    import lerobot_policy_clip
    if getattr(lerobot_policy_clip, '__file__', None) is None:
        raise ImportError("Imported an empty namespace package.")
    print("✅ Registered CLIP Plugin")
except ImportError as e:
    print(f"⚠️  Warning: lerobot_policy_clip not found or broken. CLIP sweep may fail. ({e})")

# 2. Import and run the standard LeRobot training script
from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    main()
