"""

LeHome Challenge — DINOv2 MAP+Registers Docker Policy.

Loads a DinoDiffusionPolicy checkpoint (model.safetensors) and serves it
over the BasePolicyServer HTTP protocol.
"""
import torch
import numpy as np
import os
import logging
from typing import Dict, List
from safetensors.torch import load_file

from server import BasePolicyServer

# Register the "dino_diffusion" policy type with the lerobot factory.
import lerobot_policy_dino  # noqa: F401 - side-effect import
from lerobot_policy_dino import DinoDiffusionPolicy, DinoDiffusionConfig
from lerobot.configs.types import FeatureType, PolicyFeature

_IMAGE_KEYS = [
    "observation.images.top_rgb",
    "observation.images.left_rgb",
    "observation.images.right_rgb",
]
_STATE_KEY = "observation.state"

class LeRobotDockerPolicy(BasePolicyServer):
    def __init__(self, pretrained_model_path: str = "pretrained_model"):
        """
        Loads the saved PyTorch/LeRobot checkpoint.
        """
        # 1. Decide device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Loading DinoDiffusion policy on {self.device}")

        # 2. Load your single generic model OR your 4 specialized models + DinoV2 router.
        config = DinoDiffusionConfig(
            vision_backbone="facebook/dinov2-with-registers-small",
            spatial_pooling="map",
            map_num_queries=8,
            use_registers=True,
            num_register_tokens=4,
            device=str(self.device),
            input_features={
                _STATE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(12,)),
                "observation.images.top_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
                "observation.images.left_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
                "observation.images.right_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            },
            output_features={
                "action": PolicyFeature(type=FeatureType.ACTION, shape=(12,)),
            },
        )

        self.pretrained_model_path = pretrained_model_path
        self.policy = DinoDiffusionPolicy(config)

        weights_path = os.path.join(pretrained_model_path, "model.safetensors")
        state_dict = load_file(weights_path, device=str(self.device))
        missing, unexpected = self.policy.load_state_dict(state_dict, strict=False)
        if missing:
            logging.warning(f"Missing keys when loading checkpoint: {missing}")
        if unexpected:
            logging.warning(f"Unexpected keys when loading checkpoint: {unexpected}")

        self.policy.to(self.device)
        self.policy.eval()
        logging.info("Policy loaded successfully.")

    def reset(self):
        """Called at the start of each episode — clears the obs/action queues."""
        self.policy.reset()

    def infer(self, observation: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """
        Converts numpy observations to tensors, runs select_action(), and returns
        a single action as a list of one numpy array.
        """
        obs_tensor = {}
        
        # Images: (H, W, C) uint8 -> (1, C, H, W) float32 in [0, 1]
        for key in _IMAGE_KEYS:
            img = torch.from_numpy(observation[key]).permute(2, 0, 1).float() / 255.0
            obs_tensor[key] = img.unsqueeze(0).to(self.device)

        # State: (12,) float32 -> (1, 12)
        state = torch.from_numpy(observation[_STATE_KEY]).float()
        obs_tensor[_STATE_KEY] = state.unsqueeze(0).to(self.device)

        with torch.no_grad():
            # select_action() manages the internal obs/action queues and returns
            # one action tensor of shape (1, action_dim).
            action = self.policy.select_action(obs_tensor)

        # action shape: (1, 12) -> return as list of one (12,) numpy array
        return [action.squeeze(0).cpu().numpy()]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Start the server with our policy logic
    policy_server = LeRobotDockerPolicy()
    policy_server.run(host="0.0.0.0", port=8080)
