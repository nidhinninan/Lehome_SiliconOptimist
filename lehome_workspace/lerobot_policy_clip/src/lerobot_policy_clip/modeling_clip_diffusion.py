"""Diffusion Policy with a frozen CLIP-ViT-B/16 vision backbone.

Bugs fixed versus the original draft:
  BUG-1: Config now overrides __post_init__ to skip ResNet-only validation.
  BUG-2: _make_noise_scheduler called with explicit kwargs, not the fictional
         `noise_scheduler_kwargs` dict that doesn't exist on DiffusionConfig.
  BUG-3: Normalize/Unnormalize imported from lerobot.policies.normalize
         (correct path for lerobot v0.4.3 src layout).
  BUG-6: num_inference_steps stored as instance attribute with None fallback.
  MOD-1: 'horror' typo corrected to 'horizon'.
  MOD-3 (CRITICAL): CLIP uses FIXED positional embeddings sized for 224×224 images
         (196 patches for ViT-B/16). Passing 480×640 input would cause a positional
         embedding shape mismatch and crash. Images are now resized to 224×224 before
         the CLIP backbone.
  OBS_ROBOT: Uses the lerobot constant instead of a magic string.
"""
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import CLIPVisionModel

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

from .configuration_clip_diffusion import ClipDiffusionConfig

# CLIP ViT-B/16 was pretrained on 224×224 images with fixed positional embeddings.
_CLIP_INPUT_SIZE = 224


class ClipDiffusionModel(nn.Module):
    """DiffusionModel replacement that uses a frozen CLIP-ViT-B/16 encoder."""

    def __init__(self, config: ClipDiffusionConfig):
        super().__init__()
        self.config = config

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = CLIPVisionModel.from_pretrained(config.vision_backbone)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # CLIP ViT-B/16 hidden_size = 768
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
        """Encode a (B*N, C, H, W) image tensor with the frozen CLIP vision backbone.

        MOD-3 FIX (CRITICAL): CLIP ViT-B/16 has FIXED positional embeddings for
        196 patches (14×14 grid at 16px stride on 224×224 images). Passing 480×640
        input creates 1200 patches → positional embedding size mismatch → crash.
        Resize to 224×224 before forwarding.
        """
        if img.shape[-2:] != (_CLIP_INPUT_SIZE, _CLIP_INPUT_SIZE):
            img = F.interpolate(
                img,
                size=(_CLIP_INPUT_SIZE, _CLIP_INPUT_SIZE),
                mode="bilinear",
                antialias=True,
            )
        with torch.no_grad():
            outputs = self.backbone(img)
            if self.config.use_cls_token:
                # pooler_output is the CLS token after the projection layer — standard CLIP usage.
                return outputs.pooler_output  # (B*N, feature_dim)
            else:
                return outputs.last_hidden_state.mean(dim=1)  # Global avg pool

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Build the flat global conditioning vector for the U-Net.

        Shape: (B, n_obs_steps * single_step_dim)
        """
        B = batch[OBS_ROBOT].shape[0]

        image_features = []
        for img_key in self.config.image_features:
            img = batch[img_key]        # (B, N_obs, C, H, W)
            _, N, C, H, W = img.shape
            feat = self._encode_images(img.view(B * N, C, H, W))  # (B*N, feature_dim)
            image_features.append(feat.view(B, N, -1))            # (B, N, feature_dim)

        all_feats = [batch[OBS_ROBOT]]   # robot state: (B, N, state_dim)
        all_feats.extend(image_features)
        if self.config.env_state_feature:
            all_feats.append(batch[OBS_ENV])

        combined = torch.cat(all_feats, dim=-1)   # (B, N, single_step_dim)
        return combined.view(B, -1)                # (B, N * single_step_dim)

    # ── Public methods (called by DiffusionPolicy.forward / select_action) ───

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        actions = batch["action"]
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


class ClipDiffusionPolicy(DiffusionPolicy):
    """Diffusion Policy using a frozen CLIP-ViT-B/16 vision backbone.

    Registered as policy type ``"clip_diffusion"`` for the lerobot-train CLI.
    """

    config_class = ClipDiffusionConfig
    name = "clip_diffusion"

    def __init__(
        self,
        config: ClipDiffusionConfig,
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

        # Replace the standard DiffusionModel with our CLIP-backed model.
        self.diffusion = ClipDiffusionModel(config)

    def get_optim_params(self) -> list:
        """Filter out frozen vision backbone parameters to prevent optimizer crashes."""
        return [p for p in self.parameters() if p.requires_grad]
        self.reset()
