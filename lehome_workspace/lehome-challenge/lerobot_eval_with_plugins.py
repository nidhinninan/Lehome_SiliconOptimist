#!/usr/bin/env python
"""
lerobot_eval_with_plugins.py

A wrapper for LeHome evaluation that explicitly imports the DINO LeRobot plugin.
This prevents config decode errors like:
  "Couldn't find a choice class for 'dino_diffusion' in PreTrainedConfig"
when the plugin package isn't imported (and thus doesn't register its config types).

Only ``lerobot_policy_dino`` is required here. If you evaluate a CLIP-based custom
policy (e.g. ``clip_diffusion``), install/import ``lerobot_policy_clip`` separately
or extend this wrapper.

Environment:
  LEHOME_PLUGIN_WORKSPACE — Absolute path to the directory that contains
    ``lerobot_policy_dino/`` (with ``src/``).
    Defaults to the parent of ``lehome-challenge`` (same layout as
    ``/data/lehome_workspace``).
"""

import os
import sys
from pathlib import Path


def _plugin_workspace_dir() -> Path:
    """Directory containing sibling plugin repos (lerobot_policy_*)."""
    raw = os.environ.get("LEHOME_PLUGIN_WORKSPACE", "").strip()
    if raw:
        p = Path(raw).expanduser().resolve()
        if not p.is_dir():
            raise RuntimeError(
                f"LEHOME_PLUGIN_WORKSPACE is not a directory: {p}"
            )
        return p
    return Path(__file__).resolve().parents[1]


def _register_plugins() -> None:
    workspace_dir = _plugin_workspace_dir()
    dino_src = workspace_dir / "lerobot_policy_dino" / "src"

    if not dino_src.is_dir():
        raise RuntimeError(
            f"Plugin src directory missing: {dino_src}\n"
            f"Expected layout under workspace {workspace_dir}:\n"
            "  lerobot_policy_dino/src/ …\n"
            "Set LEHOME_PLUGIN_WORKSPACE to the parent of that repo if needed."
        )

    sys.path.insert(0, str(dino_src))

    pkg = "lerobot_policy_dino"
    try:
        mod = __import__(pkg)
        if getattr(mod, "__file__", None) is None:
            raise ImportError("Imported an empty namespace package.")
        print(f"✅ Registered plugin: {pkg}", flush=True)
    except ImportError as e:
        raise RuntimeError(
            f"Failed to import LeRobot plugin '{pkg}' (required by this wrapper). "
            f"Fix the install or LEHOME_PLUGIN_WORKSPACE. Underlying error: {e}"
        ) from e


def main() -> None:
    _register_plugins()

    # Import after registration so config subclasses are available during decode.
    from scripts.eval import main as eval_main

    eval_main()


if __name__ == "__main__":
    main()
