from dataclasses import dataclass, field
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


@PreTrainedConfig.register_subclass("dino_diffusion")
@dataclass
class DinoDiffusionConfig(DiffusionConfig):
    """Configuration for Diffusion Policy with a frozen DINOv2-Small vision backbone.

    Inherits all DiffusionConfig parameters. Key overrides:
        vision_backbone: HuggingFace model ID for the DINOv2 backbone.
        use_cls_token: If True, use the CLS token as the image embedding.
                       If False, use mean-pooled patch tokens.

    BUG-1 FIX: Override __post_init__ to skip DiffusionConfig's ResNet-only
    validation, while still running the other checks (prediction_type,
    noise_scheduler_type, horizon/downsample compatibility).
    """
    # DINOv2-specific overrides
    vision_backbone: str = "facebook/dinov2-small"
    pretrained_backbone_weights: str | None = None  # Unused — HF weights loaded directly
    use_cls_token: bool = True

    # Disable the crop pipeline — DINOv2 handles its own resize internally
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = False

    def __post_init__(self):
        # Skip DiffusionConfig.__post_init__ (which asserts vision_backbone.startswith("resnet"))
        # and call PreTrainedConfig.__post_init__ directly for base-class validation only.
        PreTrainedConfig.__post_init__(self)

        # Replicate the other non-ResNet checks from DiffusionConfig.__post_init__:
        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. "
                f"Got {self.prediction_type}."
            )
        supported_noise_schedulers = ["DDPM", "DDIM"]
        if self.noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`noise_scheduler_type` must be one of {supported_noise_schedulers}. "
                f"Got {self.noise_scheduler_type}."
            )
        # Check horizon / U-Net downsampling compatibility.
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                f"The horizon ({self.horizon}) must be divisible by the U-Net downsampling "
                f"factor ({downsampling_factor})."
            )
