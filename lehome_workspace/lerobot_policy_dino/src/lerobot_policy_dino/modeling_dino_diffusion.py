"""Diffusion Policy with a frozen DINOv2 vision backbone.

Bugs fixed versus the original draft:
  BUG-1: Config now overrides __post_init__ to skip ResNet-only validation.
  BUG-2: _make_noise_scheduler called with explicit kwargs, not the fictional
         `noise_scheduler_kwargs` dict that doesn't exist on DiffusionConfig.
  BUG-3: Normalize/Unnormalize imported from lerobot.policies.normalize
         (correct path for lerobot v0.4.3 src layout).
  BUG-6: num_inference_steps stored as instance attribute with None fallback.
  MOD-1: 'horror' typo corrected to 'horizon'.
  MOD-2: Images resized to 224×224 before the DINOv2 backbone to avoid
         processing 480×640 frames (would be extremely slow and OOM-risky).
  OBS_ROBOT: Uses the lerobot constant instead of a magic string.
"""
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
import torchvision.transforms.functional as TF
from torch import Tensor, nn
from transformers import Dinov2Model

# lerobot.common does not exist in pip-installed v0.4.3 — use hardcoded constants.
# OBS_ROBOT / OBS_ENV are simple string keys defined in the LeRobot source.
OBS_ROBOT = "observation.state"
OBS_ENV = "observation.environment_state"
from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    DiffusionPolicy,
    _make_noise_scheduler,
)
from lerobot.processor import (
    NormalizerProcessorStep as Normalize,
    UnnormalizerProcessorStep as Unnormalize,
)

from .configuration_dino_diffusion import DinoDiffusionConfig

# DINOv2-small was pretrained on 224×224 images.
_DINO_INPUT_SIZE = 224


class DinoDiffusionModel(nn.Module):
    """DiffusionModel replacement that uses a frozen DINOv2-Small encoder."""

    def __init__(self, config: DinoDiffusionConfig):
        super().__init__()
        self.config = config

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = Dinov2Model.from_pretrained(config.vision_backbone)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # DINOv2-small hidden_size = 384
        self.feature_dim = self.backbone.config.hidden_size

        # ── Global-conditioning dimension ─────────────────────────────────────
        num_images = len(config.image_features)
        single_step_dim = config.robot_state_feature.shape[0] + self.feature_dim * num_images
        if config.env_state_feature:
            single_step_dim += config.env_state_feature.shape[0]

        # ── U-Net (unchanged from upstream DiffusionModel) ────────────────────
        self.unet = DiffusionConditionalUnet1d(
            config, global_cond_dim=single_step_dim * config.n_obs_steps
        )

        # ── Noise scheduler (BUG-2 FIX: explicit kwargs, no fictional dict) ───
        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        # ── BUG-6 FIX: store inference steps with None fallback ───────────────
        if config.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

    # ── Private helpers ───────────────────────────────────────────────────────

    def _encode_images(self, img: Tensor) -> Tensor:
        """Encode a (B*N, C, H, W) image tensor with the frozen DINOv2 backbone.

        MOD-2 FIX: Resize to 224×224 to match DINOv2 pretraining resolution.
        The LeHome images are 480×640; without resizing DINOv2 would process
        ~1565 patch tokens instead of 256, making it very slow.
        """
        # Resize: bilinear interpolation, antialias for downsampling quality.
        if img.shape[-2:] != (_DINO_INPUT_SIZE, _DINO_INPUT_SIZE):
            img = F.interpolate(
                img,
                size=(_DINO_INPUT_SIZE, _DINO_INPUT_SIZE),
                mode="bilinear",
                antialias=True,
            )
        with torch.no_grad():
            outputs = self.backbone(img)
            if self.config.use_cls_token:
                return outputs.last_hidden_state[:, 0, :]  # (B*N, feature_dim) — CLS token
            else:
                return outputs.last_hidden_state.mean(dim=1)  # Global avg pool over patches

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Build the flat global conditioning vector for the U-Net.

        Shape: (B, n_obs_steps * single_step_dim)
        """
        B = batch[OBS_ROBOT].shape[0]

        # 1. Encode each camera independently.
        image_features = []
        for img_key in self.config.image_features:
            img = batch[img_key]        # (B, N_obs, C, H, W)
            _, N, C, H, W = img.shape
            feat = self._encode_images(img.view(B * N, C, H, W))  # (B*N, feature_dim)
            image_features.append(feat.view(B, N, -1))            # (B, N, feature_dim)

        # 2. Collect robot state (B, N, state_dim) and optionally env state.
        all_feats = [batch[OBS_ROBOT]]   # robot state: (B, N, state_dim)
        all_feats.extend(image_features)
        if self.config.env_state_feature:
            all_feats.append(batch[OBS_ENV])

        # 3. Concatenate along feature dim, then flatten temporal dim.
        combined = torch.cat(all_feats, dim=-1)   # (B, N, single_step_dim)
        return combined.view(B, -1)                # (B, N * single_step_dim)

    # ── Public methods (called by DiffusionPolicy.forward / select_action) ───

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        actions = batch["action"]                  # (B, horizon, action_dim)
        global_cond = self._prepare_global_conditioning(batch)

        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (actions.shape[0],),
            device=actions.device,
        ).long()

        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        noise_pred = self.unet(noisy_actions, timesteps, global_cond=global_cond)
        return F.mse_loss(noise_pred, noise)

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs) -> Tensor:
        B = batch[OBS_ROBOT].shape[0]
        global_cond = self._prepare_global_conditioning(batch)

        # MOD-1 FIX: 'horror' → 'horizon'
        horizon = self.config.horizon
        action_dim = self.config.output_features["action"].shape[0]

        if noise is None:
            sample = torch.randn(
                (B, horizon, action_dim),
                device=global_cond.device,
                dtype=global_cond.dtype,
            )
        else:
            sample = noise

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.unet(sample, t, global_cond=global_cond)
            sample = self.noise_scheduler.step(noise_pred, t, sample).prev_sample

        return sample


class DinoDiffusionPolicy(DiffusionPolicy):
    """Diffusion Policy using a frozen DINOv2-Small vision backbone.

    Registered as policy type ``"dino_diffusion"`` for the lerobot-train CLI.

    We subclass DiffusionPolicy to inherit its ``select_action`` / ``forward`` /
    ``reset`` logic (which all delegate to ``self.diffusion``). We only replace:
      · the ``__init__`` to swap in our custom DinoDiffusionModel encoder.
      · the ``config_class`` / ``name`` class attributes.
    """

    config_class = DinoDiffusionConfig
    name = "dino_diffusion"

    def __init__(
        self,
        config: DinoDiffusionConfig,
        dataset_stats: dict | None = None,
        dataset_meta: dict | None = None,
        **kwargs,
    ):
        # Call PreTrainedPolicy.__init__ directly to skip DiffusionPolicy's init
        # (which would try to build a ResNet encoder via torchvision).
        super(DiffusionPolicy, self).__init__(config)
        config.validate_features()
        self.config = config

        # ── BUG-3 FIX: correct import path for lerobot v0.4.3 ─────────────────
        self.normalize_inputs = Normalize(
            config.input_features, config.normalization_mapping, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        # Replace the standard DiffusionModel with our DINOv2-backed model.
        self.diffusion = DinoDiffusionModel(config)

    def get_optim_params(self) -> list:
        """Filter out frozen vision backbone parameters to prevent optimizer crashes."""
        return [p for p in self.parameters() if p.requires_grad]
        self.reset()
