#!/bin/bash
# Apply host computer updates for LeHome Challenge Evaluation.
# Idempotent: safe to re-run.
# Run from the root of a fresh clone of https://github.com/lehome-official/lehome-challenge.git
# BEFORE the first `uv sync`.

set -euo pipefail

if [ ! -f pyproject.toml ] || [ ! -d scripts ]; then
    echo "ERROR: must be run from the lehome-challenge repository root (pyproject.toml + scripts/ expected)."
    exit 1
fi

echo "==> Applying Silicon Optimists host-side eval patches..."

# 1. pyproject.toml — pin warp-lang for IsaacLab/Isaac Sim 5.1 compatibility.
if grep -q "warp-lang" pyproject.toml; then
    echo "    [skip] warp-lang already pinned in pyproject.toml"
else
    if grep -q '"numpy==1.26.0",' pyproject.toml; then
        sed -i '/"numpy==1.26.0",/a \    "warp-lang==1.11.1",' pyproject.toml
    elif grep -q "override-dependencies" pyproject.toml; then
        # Fallback: insert after the override-dependencies opening line.
        sed -i '/override-dependencies *= *\[/a \    "warp-lang==1.11.1",' pyproject.toml
    else
        echo "ERROR: could not locate insertion point for warp-lang in pyproject.toml."
        echo "       Add '\"warp-lang==1.11.1\",' to override-dependencies manually."
        exit 1
    fi
    grep -q '"warp-lang==1.11.1"' pyproject.toml || { echo "ERROR: sed patch did not take effect."; exit 1; }
    echo "    [ok] added warp-lang==1.11.1 to pyproject.toml"
fi

mkdir -p scripts/eval_policy scripts/utils
[ -f scripts/__init__.py ] || touch scripts/__init__.py

echo "==> Writing lerobot_eval_with_plugins.py (only used for --policy_type lerobot, NOT for docker mode)..."
cat << 'INNER_EOF' > lerobot_eval_with_plugins.py
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
INNER_EOF

echo "==> Overwriting scripts/eval_policy/lerobot_policy.py with hardened adapter..."
cat << 'INNER_EOF' > scripts/eval_policy/lerobot_policy.py
import torch
import numpy as np
from typing import Dict, Any, Optional, Set
from torch import Tensor
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.processor.core import TransitionKey

from lehome.utils.logger import get_logger
from scripts.utils.eval_utils import preprocess_observation
from .base_policy import BasePolicy
from .registry import PolicyRegistry

logger = get_logger(__name__)

def _validate_pretrained_model_dir(policy_path: str) -> None:
    p = Path(policy_path)
    if not p.exists():
        raise FileNotFoundError(f"--policy_path does not exist: {policy_path}")
    if not p.is_dir():
        raise NotADirectoryError(f"--policy_path must be a directory: {policy_path}")

    cfg = p / "config.json"
    if not cfg.exists():
        raise FileNotFoundError(
            "LeRobot pretrained model is missing config.json. "
            f"Expected: {cfg} (did you point --policy_path at the correct pretrained_model/ folder?)"
        )

    # Most LeRobot pretrained models store weights as model.safetensors; keep this as a helpful hint.
    weights_ok = any((p / name).exists() for name in ("model.safetensors", "pytorch_model.bin"))
    if not weights_ok:
        raise FileNotFoundError(
            "LeRobot pretrained model appears to be missing weights "
            "(expected model.safetensors or pytorch_model.bin) under "
            f"{policy_path}"
        )


@PolicyRegistry.register("lerobot")
class LeRobotPolicy(BasePolicy):
    """
    Adapter class for official LeRobot policies (ACT, Diffusion, SmolVLA, etc.).
    
    This class handles:
    1. Loading policy weights and configurations.
    2. Filtering observation keys to match policy requirements.
    3. Preprocessing raw numpy observations into tensors (including image normalization).
    4. Running inference.
    5. Postprocessing actions (un-normalization).
    """

    def __init__(
        self, 
        policy_path: str, 
        dataset_root: str, 
        task_description: str, 
        device: str = "cuda",
        task_name: Optional[str] = None,
    ):
        """
        Initialize the LeRobot policy.

        Args:
            policy_path: Path to the pretrained model checkpoint.
            dataset_root: Path to the dataset root (used for metadata).
            task_description: Text description of the task (for VLA models).
            device: Device to run the model on ('cpu' or 'cuda').
            task_name: Isaac Lab task id (e.g. ``--task``); used to infer bimanual action dim when metadata is incomplete.
        """
        super().__init__()
        self.device = torch.device(device)
        self.task_description = task_description
        self.task_name = task_name or ""
        
        logger.info(f"Loading LeRobot policy from: {policy_path}")

        _validate_pretrained_model_dir(policy_path)
        
        # 1. Load Metadata
        meta = LeRobotDatasetMetadata(repo_id="lehome", root=dataset_root)
        
        # 2. Load Policy Config
        policy_cfg = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=[])
        policy_cfg.pretrained_path = policy_path
        
        # 3. Filter Metadata (Logic from original create_il_policy)
        # Identify features required by the policy
        self.input_features: Optional[Set[str]] = None
        if hasattr(policy_cfg, "input_features"):
            self.input_features = set(policy_cfg.input_features.keys())
            self._filter_metadata(meta, self.input_features)

        # 4. Create Policy
        self.policy = make_policy(policy_cfg, ds_meta=meta)
        self.policy.eval()
        self.policy.to(self.device)
        
        # 5. Create Processors
        preprocessor_overrides = {
            "device_processor": {"device": str(self.device)},
        }
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=policy_path,
            preprocessor_overrides=preprocessor_overrides,
        )
        
        # 6. Infer Action Dimension (Logic from original run_evaluation_loop)
        self.action_dim = self._infer_action_dim(meta)
        logger.info(f"LeRobotPolicy initialized. Action dim: {self.action_dim}")

    def reset(self):
        """Reset the internal state of the policy."""
        self.policy.reset()

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate action from observation.

        Args:
            observation: Dictionary of numpy arrays (raw environment output).

        Returns:
            action: Numpy array of action values (un-normalized).
        """
        # 1. Filter observations (keep only what the policy needs)
        if self.input_features:
            observation = self._filter_observations(observation, self.input_features)

        # 2. Preprocess (Numpy -> Tensor Batch, Normalize, etc.)
        batch_obs = self._process_observation(observation)
        
        # 3. Inference
        with torch.inference_mode():
            batch_action = self.policy.select_action(batch_obs)
            
        # 4. Postprocess (Un-normalize)
        if self.postprocessor:
            batch_action = self.postprocessor(batch_action)
            
        # 5. Convert to Numpy (Remove batch dimension)
        return batch_action.squeeze(0).cpu().numpy()

    # --------------------------------------------------------------------------
    # Internal Helper Methods
    # --------------------------------------------------------------------------

    def _filter_metadata(self, meta: LeRobotDatasetMetadata, expected_keys: Set[str]):
        """Remove extra features from metadata that are not required by the policy."""
        dataset_features = set(meta.features.keys())
        
        # Find features in dataset but not needed by policy
        extra_features = dataset_features - expected_keys
        
        # Filter out system features (these are OK to have extra)
        system_features = {
            "timestamp", "frame_index", "episode_index", 
            "index", "task_index", "next.done"
        }
        extra_features = extra_features - system_features

        # Remove extra observation features from metadata to prevent validation errors
        for feature in extra_features:
            if feature.startswith("observation."):
                del meta.features[feature]

    def _infer_action_dim(self, meta: LeRobotDatasetMetadata) -> int:
        """Infer action dimension from metadata or task name / description heuristic."""
        action_dim = None
        
        # Try metadata 'action' shape
        if meta and hasattr(meta, "features") and "action" in meta.features:
            action_shape = meta.features["action"].get("shape", [])
            if action_shape and len(action_shape) > 0:
                action_dim = action_shape[0]
                
        # Try metadata 'observation.state' shape (fallback)
        if (action_dim is None and meta and hasattr(meta, "features") 
            and "observation.state" in meta.features):
            state_shape = meta.features["observation.state"].get("shape", [])
            if state_shape and len(state_shape) > 0:
                action_dim = state_shape[0]
                
        # Final fallback based on task id + description (heuristic for bimanual)
        if action_dim is None:
            hint = f"{self.task_name} {self.task_description}"
            if "Bi" in hint or "bi" in hint.lower():
                action_dim = 12  # Dual-arm
            else:
                action_dim = 6   # Single-arm
        
        return action_dim

    def _filter_observations(self, obs_dict: Dict[str, Any], policy_input_features: Set[str]) -> Dict[str, Any]:
        """Filter observation dictionary to only include features expected by policy."""
        filtered = {}
        for key, value in obs_dict.items():
            # Keep all non-observation keys (like internal env state if any)
            if not key.startswith("observation."):
                filtered[key] = value
            # Keep observation features that policy expects
            elif key in policy_input_features:
                filtered[key] = value
        return filtered

    def _prepare_for_preprocessor(self, observation_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare observation dictionary for LeRobot preprocessor pipeline."""
        obs_for_preproc = {}
        for key, value in observation_dict.items():
            if not key.startswith("observation."):
                continue

            if isinstance(value, np.ndarray):
                value_tensor = torch.from_numpy(value).float()
                if value.ndim == 3 and value.shape[-1] == 3:  # Image: (H, W, C)
                    # (H, W, C) -> (C, H, W), [0, 1] normalization
                    value_tensor = value_tensor.permute(2, 0, 1).to(self.device) / 255.0
                    obs_for_preproc[key] = value_tensor.unsqueeze(0)  # Add batch dim
                else:
                    obs_for_preproc[key] = value_tensor.unsqueeze(0)  # Add batch dim
            else:
                obs_for_preproc[key] = value

        # Create transition format with complementary_data for VLA models
        dummy_action = torch.zeros(1, self.action_dim, dtype=torch.float32, device=self.device)
        transition = {
            TransitionKey.OBSERVATION: obs_for_preproc,
            TransitionKey.ACTION: dummy_action,
            TransitionKey.COMPLEMENTARY_DATA: {"task": self.task_description},
        }
        return transition

    def _process_observation(self, observation_dict: Dict[str, Any]) -> Dict[str, Tensor]:
        """Process observation using the LeRobot preprocessor or manual fallback."""
        if self.preprocessor is not None:
            transition = self._prepare_for_preprocessor(observation_dict)
            transformed_transition = self.preprocessor._forward(transition)
            return self.preprocessor.to_output(transformed_transition)
        else:
            # Fallback to manual preprocessing (moved to utils in Step A)
            return preprocess_observation(
                observation_dict, self.device, self.task_description
            )INNER_EOF

echo "==> Overwriting scripts/utils/evaluation.py (passes task_name into the lerobot adapter)..."
cat << 'INNER_EOF' > scripts/utils/evaluation.py
import os
import argparse
import gymnasium as gym
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg

from scripts.eval_policy import PolicyRegistry
from scripts.eval_policy.base_policy import BasePolicy

from scripts.utils.eval_utils import (
    convert_ee_pose_to_joints,
    save_videos_from_observations,
    calculate_and_print_metrics,
)

from lehome.utils.record import (
    RateLimiter,
    get_next_experiment_path_with_gap,
    append_episode_initial_pose,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from .common import stabilize_garment_after_reset
from lehome.utils.logger import get_logger

logger = get_logger(__name__)


def run_evaluation_loop(
    env: DirectRLEnv,
    policy: BasePolicy,
    args: argparse.Namespace,
    ee_solver: Optional[Any] = None,
    is_bimanual: bool = False,
    garment_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Core evaluation loop.
    Refactored to be agnostic of specific model implementations.
    """

    # --- Dataset Recording Setup (Optional) ---
    eval_dataset = None
    json_path = None
    episode_index = 0
    if args.save_datasets:
        features = None
        if args.dataset_root and Path(args.dataset_root).exists():
            source_dataset = LeRobotDataset(repo_id="collected_dataset", root=Path(args.dataset_root))
            features = dict(source_dataset.meta.features)
            fps = source_dataset.fps
        else:
            fps = 30  # Default FPS if no source dataset is provided
            action_names = [
                "shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex", "wrist_roll", "gripper",
            ]
            if is_bimanual:
                left_names = [f"left_{n}" for n in action_names]
                right_names = [f"right_{n}" for n in action_names]
                joint_names = left_names + right_names
            else:
                joint_names = action_names
            dim = len(joint_names)
            features = {
                "observation.state": {
                    "dtype": "float32",
                    "shape": (dim,),
                    "names": joint_names,
                },
                "action": {
                    "dtype": "float32",
                    "shape": (dim,),
                    "names": joint_names,
                },
            }
            image_keys = ["top_rgb", "left_rgb", "right_rgb"] if is_bimanual else ["top_rgb", "wrist_rgb"]
            for key in image_keys:
                features[f"observation.images.{key}"] = {
                    "dtype": "video",
                    "shape": (480, 640, 3),
                    "names": ["height", "width", "channels"],
                }
        root_path = Path(args.eval_dataset_path)
        eval_dataset = LeRobotDataset.create(
            repo_id="lehome_eval",
            fps=fps,
            root=get_next_experiment_path_with_gap(root_path),
            use_videos=True,
            image_writer_threads=8,
            image_writer_processes=0,
            features=features,
        )
        json_path = eval_dataset.root / "meta" / "garment_info.json"

    all_episode_metrics = []
    logger.info(f"Starting evaluation: {args.num_episodes} episodes")
    rate_limiter = RateLimiter(args.step_hz) if args.step_hz > 0 else None

    for i in range(args.num_episodes):
        # 1. Reset Environment & Policy
        env.reset()
        policy.reset()
        stabilize_garment_after_reset(env, args)

        # 2. Initial Observation (Numpy)
        object_initial_pose = env.get_all_pose() if args.save_datasets else None
        observation_dict = env._get_observations()

        # Prepare for video recording
        episode_frames = (
            {k: [] for k in observation_dict.keys() if "images" in k}
            if args.save_video
            else {}
        )

        episode_return = 0.0
        episode_length = 0
        extra_steps = 0
        success_flag = False
        success = torch.tensor(False)

        for st in range(args.max_steps):
            if rate_limiter:
                rate_limiter.sleep(env)

            # 3. Policy Inference (The core abstraction)
            # Input: Numpy Dict -> Output: Numpy Array
            action_np = policy.select_action(observation_dict)

            # 4. Prepare Action for Environment (Tensor)
            # Convert numpy action to tensor for Isaac Lab
            action = torch.from_numpy(action_np).float().to(args.device).unsqueeze(0)

            # 5. Inverse Kinematics (Optional Helper Logic)
            # If policy outputs EE pose but env needs joints
            if args.use_ee_pose and ee_solver is not None:
                current_joints = (
                    torch.from_numpy(observation_dict["observation.state"])
                    .float()
                    .to(args.device)
                )
                action = convert_ee_pose_to_joints(
                    ee_pose_action=action.squeeze(0),
                    current_joints=current_joints,
                    solver=ee_solver,
                    is_bimanual=is_bimanual,
                    state_unit="rad",
                    device=args.device,
                ).unsqueeze(0)

            # 6. Step Environment
            env.step(action)

            # Check success first
            if not success_flag:
                success = env._get_success()
                if success.item():
                    success_flag = True
                    extra_steps = 50  # Run a bit longer after success to settle

            # Get reward from environment (Isaac Lab stores rewards internally)
            reward_value = env._get_rewards()
            if isinstance(reward_value, torch.Tensor):
                reward = reward_value.item()
            else:
                reward = float(reward_value)

            # Accumulate reward for all steps (including post-success steps)
            episode_return += reward
            # Only count length before success (for consistency with episode termination)
            if not success_flag:
                episode_length += 1

            # Update Observation
            observation_dict = env._get_observations()

            # Recording
            if args.save_datasets:
                frame = {
                    k: v
                    for k, v in observation_dict.items()
                    if k != "observation.top_depth"
                }
                frame["task"] = args.task_description
                eval_dataset.add_frame(frame)

            if args.save_video:
                for key, val in observation_dict.items():
                    if "images" in key:
                        episode_frames[key].append(val.copy())

            if success_flag:
                extra_steps -= 1
                if extra_steps <= 0:
                    break

        # --- End of Episode Handling ---
        is_success = success.item() if success_flag else False

        # Save Datasets
        if args.save_datasets:
            if success_flag:
                eval_dataset.save_episode()
                append_episode_initial_pose(
                    json_path,
                    episode_index,
                    object_initial_pose,
                    garment_name=garment_name,
                )
                episode_index += 1
            else:
                eval_dataset.clear_episode_buffer()

        # Save Videos (Using generic util)
        if args.save_video:
            save_videos_from_observations(
                episode_frames,
                success=success if success_flag else torch.tensor(False),
                save_dir=args.video_dir,
                episode_idx=i,
                garment_name=garment_name,
            )

        # Log Metrics
        all_episode_metrics.append(
            {"return": episode_return, "length": episode_length, "success": is_success}
        )
        logger.info(
            f"Episode {i + 1}/{args.num_episodes}: Return={episode_return:.2f}, Length={episode_length}, Success={is_success}"
        )

    return all_episode_metrics


def eval(args: argparse.Namespace, simulation_app: Any) -> None:
    """
    Main entry point for evaluation logic.
    """
    # 1. Environment Configuration
    env_cfg = parse_env_cfg(args.task, device=args.device)
    env_cfg.sim.use_fabric = False
    if args.use_random_seed:
        env_cfg.use_random_seed = True
    else:
        env_cfg.use_random_seed = False
        env_cfg.seed = args.seed
        # Propagate seed to sim config if structure exists
        if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "seed"):
            env_cfg.sim.seed = args.seed

    env_cfg.garment_cfg_base_path = args.garment_cfg_base_path
    env_cfg.particle_cfg_path = args.particle_cfg_path

    # 2. Initialize Policy (Using the Policy Registry)
    # This replaces create_il_policy, make_pre_post_processors, etc.
    logger.info(f"Initializing Policy Type: {args.policy_type}")

    # Check if policy is registered
    if not PolicyRegistry.is_registered(args.policy_type):
        available_policies = PolicyRegistry.list_policies()
        raise ValueError(
            f"Policy type '{args.policy_type}' not found in registry. "
            f"Available policies: {', '.join(available_policies)}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_bimanual = "Bi" in args.task or "bi" in args.task.lower()

    # Create policy instance from registry with appropriate arguments
    # Different policies may require different initialization arguments
    policy_kwargs = {
        "device": device,
    }

    if args.policy_type == "lerobot":
        # LeRobot policy requires policy_path and dataset_root
        if not args.policy_path:
            raise ValueError("--policy_path is required for lerobot policy type")
        if not args.dataset_root:
            raise ValueError("--dataset_root is required for lerobot policy type")
        policy_kwargs.update(
            {
                "policy_path": args.policy_path,
                "dataset_root": args.dataset_root,
                "task_description": args.task_description,
                "task_name": args.task,
            }
        )
    elif args.policy_type == "classifier_router":
        if not args.policy_path:
            raise ValueError(
                "--policy_path must point to a classifier router JSON config "
                "when policy_type is classifier_router"
            )
        policy_kwargs["router_config_path"] = args.policy_path
    else:
        # For custom policies, pass policy_path as model_path if provided
        if args.policy_path:
            policy_kwargs["model_path"] = args.policy_path

    # Create policy from registry
    policy = PolicyRegistry.create(args.policy_type, **policy_kwargs)
    logger.info(f"Policy '{args.policy_type}' loaded successfully")

    # 3. Initialize IK Solver (If needed)
    ee_solver = None
    if args.use_ee_pose:
        from lehome.utils import RobotKinematics

        urdf_path = args.ee_urdf_path  # Assuming path is handled or add check logic
        joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]
        ee_solver = RobotKinematics(
            str(urdf_path),
            target_frame_name="gripper_frame_link",
            joint_names=joint_names,
        )
        logger.info(f"IK solver loaded.")

    # 4. Load Evaluation List
    # Only loads from 'Release' directory based on garment_type
    eval_list = []  # List of (name, stage)

    # Evaluate a specific category based on garment_type
    if hasattr(args, "custom_list_path") and args.custom_list_path is not None:
        eval_list_path = args.custom_list_path
    elif args.garment_type == "custom":
        # For 'custom' type, we load from the root Release_test_list.txt
        eval_list_path = os.path.join(
            args.garment_cfg_base_path, "Release", "Release_test_list.txt"
        )
    else:
        # Map argument to specific sub-category directory
        type_map = {
            "top_long": "Top_Long",
            "top_short": "Top_Short",
            "pant_long": "Pant_Long",
            "pant_short": "Pant_Short",
        }
        file_prefix = type_map.get(args.garment_type, "Top_Long")
        # Path: Assets/objects/Challenge_Garment/Release/Top_Long/Top_Long.txt
        eval_list_path = os.path.join(
            args.garment_cfg_base_path, "Release", file_prefix, f"{file_prefix}.txt"
        )

    logger.info(
        f"Loading evaluation list for category '{args.garment_type}' from: {eval_list_path}"
    )

    if not os.path.exists(eval_list_path):
        raise FileNotFoundError(f"Evaluation list not found: {eval_list_path}")

    with open(eval_list_path, "r") as f:
        names = [line.strip() for line in f.readlines() if line.strip()]
        for name in names:
            eval_list.append((name, "Release"))

    logger.info(f"Loaded {len(eval_list)} garments for category: {args.garment_type}")

    if not eval_list:
        raise ValueError(
            f"No garments found to evaluate for category '{args.garment_type}'."
        )

    # 5. Main Evaluation Loops
    all_garment_metrics = []

    # Init Env with first garment
    first_name, first_stage = eval_list[0]
    env_cfg.garment_name = first_name
    env_cfg.garment_version = first_stage
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.initialize_obs()

    try:
        for garment_idx, (garment_name, garment_stage) in enumerate(eval_list):
            logger.info(
                f"Evaluating: {garment_name} ({garment_stage}) ({garment_idx+1}/{len(eval_list)})"
            )

            # Switch Garment Logic
            if garment_idx > 0:
                if hasattr(env, "switch_garment"):
                    env.switch_garment(garment_name, garment_stage)
                    env.reset()
                    policy.reset()
                else:
                    env.close()
                    env_cfg.garment_name = garment_name
                    env_cfg.garment_version = garment_stage
                    env = gym.make(args.task, cfg=env_cfg).unwrapped
                    env.initialize_obs()
                    policy.reset()

            # Run Loop
            metrics = run_evaluation_loop(
                env=env,
                policy=policy,
                args=args,
                ee_solver=ee_solver,
                is_bimanual=is_bimanual,
                garment_name=garment_name,
            )

            all_garment_metrics.append(
                {"garment_name": garment_name, "metrics": metrics}
            )

    finally:
        env.close()

    # Print summary across all garments
    logger.info("=" * 60)
    logger.info("Overall Summary")
    logger.info("=" * 60)

    if all_garment_metrics:
        # Aggregate all episode metrics
        all_episodes = []
        for garment_data in all_garment_metrics:
            for episode_metric in garment_data["metrics"]:
                episode_metric["garment_name"] = garment_data["garment_name"]
                all_episodes.append(episode_metric)

        # Print overall metrics
        calculate_and_print_metrics(all_episodes)

        # Print per-garment summary
        logger.info("=" * 60)
        logger.info("Per-Garment Summary")
        logger.info("=" * 60)
        for garment_data in all_garment_metrics:
            garment_name = garment_data["garment_name"]
            metrics = garment_data["metrics"]
            success_count = sum(1 for m in metrics if m["success"])
            success_rate = success_count / len(metrics) if metrics else 0.0
            avg_return = np.mean([m["return"] for m in metrics]) if metrics else 0.0
            logger.info(
                f"  {garment_name}: Success Rate = {success_rate:.2%}, Avg Return = {avg_return:.2f}"
            )
    else:
        logger.info("No metrics collected (all evaluations failed)")

    logger.info("=" * 60)
    logger.info("Evaluation completed successfully")
    logger.info("=" * 60)
INNER_EOF

echo
echo "==> Host-side eval patches applied successfully."
echo "    Next steps:"
echo "      1) huggingface-cli login         (required for the lehome/asset_challenge dataset)"
echo "      2) uv sync                       (installs warp-lang 1.11.1 along with the rest)"
echo "      3) Continue from Step 3 in README_SUBMISSION.md (clone IsaacLab, install LeHome, etc.)"
echo "      4) For evaluation use: python -m scripts.eval --policy_type docker --docker_url http://localhost:808X ..."
echo "         (do NOT use lerobot_eval_with_plugins.py for docker-mode evaluation)"
