---
name: lerobot-dataset-manager
description: Manage, view, and edit LeRobot datasets on Hugging Face or locally. Make sure to use this skill whenever the user explicitly or implicitly mentions viewing LeRobot datasets, inspecting episodes, managing datasets, modifying hugging face datasets, splitting datasets, dropping features/cameras from a dataset, deleting episodes, converting image datasets to video format, or anything related to data modification within the LeRobot ecosystem. This is useful for dataset inspection, cleaning, or prep for training!
---

# LeRobot Dataset Manager

A skill dedicated to interacting with the dataset format used in the LeRobot ecosystem (`v3.0`). 
This skill guides you in helping the user to inspect dataset metadata, view episodes, and edit/manage dataset splits, features, and content.

## General Principles

When interacting with tasks related to dataset viewing or management, keep the following in mind:
- **Prioritize Bash Scripts**: For dataset editing or info retrieval tasks, generate and provide clear bash scripts (using `lerobot-edit-dataset`) that the user can execute locally.
- **Provide Markdown Summaries**: When the user requests information about a dataset, help them run the info command and provide the resulting output as a nice, readable markdown summary.
- **Do not run `lerobot-dataset-viz` yourself**: Visualizing requires a UI buffer! When the user needs to visualize or see what an episode looks like, just provide them the exact `lerobot-dataset-viz` command to run in their own terminal. Tell them what arguments to use.
- **Run the `lerobot-edit-dataset info` command for them if possible**: If the user asks for dataset features, you can optionally run the `info` command using `run_command` in the background (if it takes a bit of time) and parse the stdout for them, since that doesn't require a UI. However, if the user explicitly wants you to just give them the scripts, do that instead.

## Available Tooling

LeRobot provides two primary command-line tools for datasets:
1. `lerobot-edit-dataset`: For viewing metadata and modifying datasets.
2. `lerobot-dataset-viz`: For displaying episodes sequentially in a UI window.

### 1. View Dataset Info

To get a quick summary of a dataset (number of episodes, format, features, sizes):

```bash
lerobot-edit-dataset \
    --repo_id lerobot/pusht \
    --operation.type info \
    --operation.show_features true
```
*Note: Replace `lerobot/pusht` with the user's specific dataset path.*

### 2. Viewing / Visualizing Episodes

To visualize a specific episode (or start from 0 and loop), instruct the user to run the following in their terminal:

```bash
lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0
```
**Important:** Do NOT run this using your tool calling! This command opens up an interactive Pygame/Video window. Instead, write this snippet out for the user and tell them: "Run this command in your terminal to see the episode visually."

### 3. Editing Datasets

LeRobot supports specific standard operations for modifying datasets. Generate scripts using these patterns:

**Delete Episodes**
Useful for removing bad demonstrations.
```bash
lerobot-edit-dataset \
    --repo_id lerobot/pusht \
    --operation.type delete_episodes \
    --operation.episode_indices "[0, 2, 5]" \
    --new_repo_id <YOUR_NEW_REPO_ID> # optional
```

**Split Dataset**
Good for creating training and validation splits.
```bash
lerobot-edit-dataset \
    --repo_id lerobot/pusht \
    --operation.type split \
    --operation.splits '{"train": 0.8, "val": 0.2}'
```

**Merge Datasets**
```bash
lerobot-edit-dataset \
    --repo_id lerobot/pusht_merged \
    --operation.type merge \
    --operation.repo_ids "['lerobot/pusht_train', 'lerobot/pusht_val']"
```

**Remove Features (e.g., dropping a camera)**
To remove an unused visual observation.
```bash
lerobot-edit-dataset \
    --repo_id lerobot/pusht \
    --operation.type remove_feature \
    --operation.feature_names "['observation.images.top']"
```

**Convert Image Dataset to Video**
Useful for reducing storage usage.
```bash
lerobot-edit-dataset \
    --repo_id lerobot/pusht_image \
    --new_repo_id lerobot/pusht_video \
    --operation.type convert_image_to_video \
    --push_to_hub true # optional
```

**Modify Tasks (Text Instructions)**
```bash
lerobot-edit-dataset \
    --repo_id lerobot/pusht \
    --operation.type modify_tasks \
    --operation.new_task "Pick up the cube and place it"
```

## Python API

If the user wants to do programmatic iteration over the dataset that isn't supported by the CLI (e.g. customized filtering or data augmentation step), guide them to use the `LeRobotDataset` API directly inside a python script:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# It handles loading directly from Hub or local
dataset = LeRobotDataset("lerobot/pusht")

# Check metadata
print(dataset.meta.info)

# Access a specific frame
frame = dataset[0]
print(frame.keys())
```

## Summary Checklist for the Agent
- If the user asks what the features are, give them the `lerobot-edit-dataset ... info` bash command or run it and parse it for them.
- If they ask to view, give them the `lerobot-dataset-viz` command.
- If they ask to modify, provide the proper `lerobot-edit-dataset` script.
