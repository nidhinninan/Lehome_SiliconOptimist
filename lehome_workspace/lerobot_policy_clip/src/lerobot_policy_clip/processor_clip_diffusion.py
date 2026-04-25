from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors
from .configuration_clip_diffusion import ClipDiffusionConfig

def make_clip_diffusion_pre_post_processors(config: ClipDiffusionConfig, dataset_stats: dict | None = None):
    return make_diffusion_pre_post_processors(config, dataset_stats)
