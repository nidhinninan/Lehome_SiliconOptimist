from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors
from .configuration_dino_diffusion import DinoDiffusionConfig

def make_dino_diffusion_pre_post_processors(config: DinoDiffusionConfig, dataset_stats: dict | None = None):
    # DINO and standard Diffusion use the same pre/post-processing pipeline
    # (Renaming, Normalization, Batch Dimension, Device movement)
    return make_diffusion_pre_post_processors(config, dataset_stats)
