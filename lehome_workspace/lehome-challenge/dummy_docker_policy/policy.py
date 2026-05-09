"""
LeHome Challenge — DINOv2 MAP+Registers Docker Policy.

Loads a DinoDiffusionPolicy checkpoint (model.safetensors) and serves it
over the BasePolicyServer HTTP protocol.

FIXED VERSION: Includes LeRobot normalization/unnormalization pipeline.
"""
import torch
import numpy as np
import os
import logging
from typing import Dict, List, Optional, Any
from safetensors.torch import load_file
from pathlib import Path

from server import BasePolicyServer

# Register the "dino_diffusion" policy type with the lerobot factory.
import lerobot_policy_dino  # noqa: F401 - side-effect import
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.processor.core import TransitionKey

_IMAGE_KEYS = [
    "observation.images.top_rgb",
    "observation.images.left_rgb",
    "observation.images.right_rgb",
]
_STATE_KEY = "observation.state"

class LeRobotDockerPolicy(BasePolicyServer):
    def __init__(self, pretrained_model_path: str = "pretrained_model", meta_path: str = "meta"):
        """
        Loads the saved PyTorch/LeRobot checkpoint and initializes the pre/post processors.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Loading DinoDiffusion policy on {self.device}")

        # 1. Load Metadata (for normalization stats)
        # We assume the 'meta' folder is copied into the container root or provided path.
        if os.path.exists(meta_path):
            logging.info(f"Loading dataset metadata from {meta_path}")
            # LeRobotDatasetMetadata expects the parent directory of 'meta' as root
            meta = LeRobotDatasetMetadata(repo_id="lehome", root=str(Path(meta_path).parent))
        else:
            logging.warning(f"Metadata path {meta_path} not found. Normalization might be incorrect!")
            meta = None

        # 2. Load Policy Config
        config_path = os.path.join(pretrained_model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found at {config_path}")
            
        policy_cfg = PreTrainedConfig.from_pretrained(pretrained_model_path, cli_overrides=[])
        policy_cfg.pretrained_path = pretrained_model_path

        # 3. Create Policy
        self.policy = make_policy(policy_cfg, ds_meta=meta)
        
        # Load weights
        weights_path = os.path.join(pretrained_model_path, "model.safetensors")
        if not os.path.exists(weights_path):
             weights_path = os.path.join(pretrained_model_path, "pytorch_model.bin")
             
        state_dict = load_file(weights_path, device=str(self.device))
        missing, unexpected = self.policy.load_state_dict(state_dict, strict=False)
        if missing:
            logging.warning(f"Missing keys when loading checkpoint: {missing}")
        if unexpected:
            logging.warning(f"Unexpected keys when loading checkpoint: {unexpected}")

        self.policy.to(self.device)
        self.policy.eval()

        # 4. Create Processors (Normalization/Un-normalization)
        preprocessor_overrides = {
            "device_processor": {"device": str(self.device)},
        }
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=pretrained_model_path,
            preprocessor_overrides=preprocessor_overrides,
        )
        
        logging.info("Policy and processors loaded successfully.")

    def reset(self):
        """Called at the start of each episode — clears the obs/action queues."""
        self.policy.reset()

    def infer(self, observation: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """
        Converts numpy observations to tensors, runs through preprocessor,
        select_action(), and then postprocessor.
        """
        # 1. Prepare for preprocessor (Numpy -> Tensor Batch)
        obs_for_preproc = {}
        for key, value in observation.items():
            if not key.startswith("observation."):
                continue
            
            # Convert to tensor and add batch dim
            val_tensor = torch.from_numpy(value).float()
            
            if "images" in key:
                # (H, W, 3) -> (1, 3, H, W), [0, 1] normalization
                val_tensor = val_tensor.permute(2, 0, 1).to(self.device) / 255.0
                obs_for_preproc[key] = val_tensor.unsqueeze(0)
            else:
                # (D,) -> (1, D)
                obs_for_preproc[key] = val_tensor.unsqueeze(0).to(self.device)

        # 2. Run Preprocessor (Normalization)
        if self.preprocessor:
            # Construct transition dict expected by LeRobot preprocessor
            # (Note: task description is not used by DinoDiffusion but required by some VLA logic)
            dummy_action = torch.zeros(1, 12, dtype=torch.float32, device=self.device)
            transition = {
                TransitionKey.OBSERVATION: obs_for_preproc,
                TransitionKey.ACTION: dummy_action,
            }
            transformed_transition = self.preprocessor._forward(transition)
            batch_obs = self.preprocessor.to_output(transformed_transition)
        else:
            batch_obs = obs_for_preproc

        # 3. Inference
        with torch.no_grad():
            batch_action = self.policy.select_action(batch_obs)

        # 4. Postprocessor (Un-normalization)
        if self.postprocessor:
            batch_action = self.postprocessor(batch_action)

        # 5. Return as list of one (12,) numpy array
        return [batch_action.squeeze(0).cpu().numpy()]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Start the server with our policy logic
    # We assume 'pretrained_model' and 'meta' are in the current working directory (/app)
    policy_server = LeRobotDockerPolicy(
        pretrained_model_path="/app/pretrained_model",
        meta_path="/app/meta"
    )
    policy_server.run(host="0.0.0.0", port=8080)
