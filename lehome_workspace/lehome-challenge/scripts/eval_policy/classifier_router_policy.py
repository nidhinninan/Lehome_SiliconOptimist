"""
Routes to one of four LeRobot DP experts using a frozen ResNet18 garment classifier.

Configuration: pass ``router_config_path`` (JSON) via evaluation harness when
``policy_type`` is ``classifier_router`` (see scripts/utils/evaluation.py).

JSON schema::
    {
      "dataset_root": "/path/to/merged_dataset_for_metadata",
      "task_description": "fold the garment on the table",
      "classifier_checkpoint": "/path/to/classifier_best.pt",
      "expert_checkpoints": {
        "top_long": "/path/to/checkpoint_top_long",
        "top_short": "...",
        "pant_long": "...",
        "pant_short": "..."
      },
      "min_confidence": 0.0
    }

The checkpoint must be produced by ``train_garment_classifier.py`` (includes
``model_state_dict`` and ``class_names``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from .base_policy import BasePolicy
from .lerobot_policy import LeRobotPolicy
from .registry import PolicyRegistry


IMAGE_KEYS = (
    "observation.images.top_rgb",
    "observation.images.top",
)


def _find_top_image(observation: Dict[str, np.ndarray]) -> np.ndarray:
    for k in IMAGE_KEYS:
        if k in observation:
            return observation[k]
    for key in observation:
        if "images" in key and "top" in key.lower() and "depth" not in key.lower():
            return observation[key]
    raise KeyError(
        f"No top RGB image in observation. Tried {IMAGE_KEYS}. "
        f"Have: {[x for x in observation if 'image' in x.lower()]}"
    )


@PolicyRegistry.register("classifier_router")
class ClassifierRouterPolicy(BasePolicy):
    def __init__(
        self,
        router_config_path: str,
        device: str = "cuda",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        cfg_path = Path(router_config_path)
        if not cfg_path.is_file():
            raise FileNotFoundError(f"router_config_path not found: {cfg_path}")
        self._cfg = json.loads(cfg_path.read_text())

        self._dataset_root = self._cfg["dataset_root"]
        self._task_description = self._cfg.get(
            "task_description", "fold the garment on the table"
        )
        self._min_confidence = float(self._cfg.get("min_confidence", 0.0))
        experts = self._cfg["expert_checkpoints"]
        if not isinstance(experts, dict) or len(experts) < 1:
            raise ValueError("expert_checkpoints must be a non-empty dict")

        ckpt_path = Path(self._cfg["classifier_checkpoint"])
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"classifier_checkpoint not found: {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=self.device)
        class_names: list[str] = ckpt["class_names"]
        num_classes = len(class_names)

        self._classifier = models.resnet18(weights=None)
        self._classifier.fc = nn.Linear(self._classifier.fc.in_features, num_classes)
        self._classifier.load_state_dict(ckpt["model_state_dict"])
        self._classifier.to(self.device)
        self._classifier.eval()

        self._idx_to_name = {i: class_names[i] for i in range(num_classes)}
        self._expert_paths = {k: str(Path(v)) for k, v in experts.items()}
        for name in class_names:
            if name not in self._expert_paths:
                raise KeyError(
                    f"expert_checkpoints missing key '{name}'. Have: {list(self._expert_paths)}"
                )

        self._imagenet_eval = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self._lazy_load = self._cfg.get("lazy_load", True)
        self._active_expert: Optional[LeRobotPolicy] = None
        
        self._loaded_experts: dict[int, LeRobotPolicy] = {}
        if not self._lazy_load:
            # Eagerly load all experts into memory
            for i in range(num_classes):
                self._loaded_experts[i] = self._make_expert(i)

    def reset(self) -> None:
        if self._lazy_load and self._active_expert is not None:
            # Free memory for the previous lazy loaded expert
            del self._active_expert
            torch.cuda.empty_cache()
            
        self._active_expert = None
        
        # Reset any held experts
        for expert in self._loaded_experts.values():
            expert.reset()

    def _classify(self, observation: Dict[str, np.ndarray]) -> int:
        img = _find_top_image(observation)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        from PIL import Image

        pil = Image.fromarray(img, mode="RGB")
        x = self._imagenet_eval(pil).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self._classifier(x)
            prob = torch.softmax(logits, dim=1)
            conf, pred = prob.max(dim=1)
        if conf.item() < self._min_confidence:
            # fall back to argmax anyway if min_confidence is 0; otherwise still argmax
            pass
        return int(pred.item())

    def _make_expert(self, class_idx: int) -> LeRobotPolicy:
        name = self._idx_to_name[class_idx]
        path = self._expert_paths[name]
        expert = LeRobotPolicy(
            policy_path=path,
            dataset_root=self._dataset_root,
            task_description=self._task_description,
            device=str(self.device),
        )
        return expert

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        if self._active_expert is None:
            pred_idx = self._classify(observation)
            
            if self._lazy_load:
                self._active_expert = self._make_expert(pred_idx)
                self._active_expert.reset()
            else:
                # Use the pre-loaded expert
                self._active_expert = self._loaded_experts[pred_idx]
                
        return self._active_expert.select_action(observation)
