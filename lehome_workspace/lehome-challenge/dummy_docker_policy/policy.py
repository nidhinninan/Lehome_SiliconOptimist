"""
LeHome Challenge — Your Docker Policy.
This integrates LeRobot checkpoint loading and provides the HTTP endpoint.
"""
import torch
import numpy as np
import logging
from typing import Dict, List
from server import BasePolicyServer

# Note: Adjust these if using the custom BYOP package
# from lerobot.common.policies.factory import make_policy

class LeRobotDockerPolicy(BasePolicyServer):
    def __init__(self, pretrained_model_path: str = "pretrained_model"):
        """
        Loads the saved PyTorch/LeRobot checkpoint. 
        """
        # 1. Decide device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Loading model to device: {self.device}")

        # 2. Load your single generic model OR your 4 specialized models + DinoV2 router.
        # For standard LeRobot checkpoint:
        # self.policy = make_policy(pretrained_model_path, device=self.device)
        # self.policy.eval()

        # Example pseudo-code for the Router approach:
        # self.router_classifier = load_dino_linear_probe("path/to/probe.pt", device=self.device)
        # self.policies = {
        #     "top_long": make_policy("pretrained_model/top_long", device=self.device),
        #     "pant_short": make_policy("pretrained_model/pant_short", device=self.device),
        #     # ... etc
        # }
        
        self.pretrained_model_path = pretrained_model_path

    def reset(self):
        """Called at the start of each episode."""
        logging.info("Episode reset. Clearing any policy buffers if necessary.")
        # If your policy is stateful (e.g. RNN/LSTM based or maintains action chunk state), reset it here.
        # if hasattr(self.policy, 'reset'):
        #     self.policy.reset()

    def infer(self, observation: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """
        Receives NumPy arrays, converts to Torch Tensors, runs inference, and returns actions.
        """
        # Convert observation dict from numpy arrays to Torch Tensors and move to device
        # Note: the Docker protocol sends specific keys like 'observation.images.top_rgb'
        obs_tensor = {}
        for k, v in observation.items():
            tensor_v = torch.from_numpy(v).to(self.device).unsqueeze(0) # Add batch dim
            # Handle image channels if necessary (e.g., LeRobot typically expects (B, C, H, W) in [0, 1] range)
            if 'images' in k and tensor_v.ndim == 4:
                # Transpose from (B, H, W, C) to (B, C, H, W)
                tensor_v = tensor_v.permute(0, 3, 1, 2)
                # Normalize from [0, 255] uint8 to [0, 1] float32 if needed by your model
                if tensor_v.dtype == torch.uint8:
                    tensor_v = tensor_v.float() / 255.0
            obs_tensor[k] = tensor_v

        # --- ROUTER APPROACH EXAMPLE ---
        # 1. Use top_rgb image to classify garment type
        # predicted_class = self.router_classifier(obs_tensor["observation.images.top_rgb"])
        # 2. Select the specialized policy
        # active_policy = self.policies[predicted_class]
        
        # --- SINGLE POLICY APPROACH ---
        # active_policy = self.policy

        # --- INFERENCE ---
        with torch.no_grad():
            # Output of active_policy(obs_tensor) is usually a dict or tensor of shape (B, chunk_size, action_dim)
            # action_pred = active_policy(obs_tensor)
            
            # Placeholder dummy action return (remove this in your actual code)
            ACTION_DIM = 12
            CHUNK_SIZE = 10
            action_pred = torch.zeros((1, CHUNK_SIZE, ACTION_DIM), device=self.device)
            # --------------------------------------------------------------------------

        # Remove batch dimension: shape (chunk_size, action_dim)
        action_pred = action_pred.squeeze(0)
        
        # Convert back to list of NumPy arrays to satisfy the HTTP server protocol
        action_list = [a.cpu().numpy() for a in action_pred]
        return action_list

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Start the server with our policy logic
    policy_server = LeRobotDockerPolicy()
    policy_server.run(host="0.0.0.0", port=8080)
