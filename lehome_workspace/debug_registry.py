import sys

print("Before BYOP imports:", sys.modules.keys() if 'lerobot.configs.policies' in sys.modules else 'Not loaded yet')

import lerobot.configs.policies
print("Initial choices registry:", lerobot.configs.policies.PreTrainedConfig.get_choices())

import lerobot_policy_dino
print("After Dino import choices:", lerobot.configs.policies.PreTrainedConfig.get_choices())

import lerobot_policy_clip
print("After Clip import choices:", lerobot.configs.policies.PreTrainedConfig.get_choices())

from lerobot.scripts.lerobot_train import main
print("Before main choices:", lerobot.configs.policies.PreTrainedConfig.get_choices())

print("This is a debug file to expose the draccus context")
