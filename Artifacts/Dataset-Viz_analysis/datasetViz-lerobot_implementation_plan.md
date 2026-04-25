# Custom LeRobotDataset Implementation Guide

This document outlines how to implement a custom `LeRobotDataset` for multiple camera streams and proprioception, ensuring full compatibility with the latest image augmentation pipeline (v0.1.0/v0.4.3).

## Goal
Implement a dataset that supports:
1. **Multiple Camera Streams**: Visual observations from multiple perspectives (e.g., top, front, wrist).
2. **Proprioception**: Robot joint states or other low-level sensor data.
3. **Augmentation Compatibility**: Ensuring the image augmentation pipeline can automatically detect and transform the camera feeds.

## Proposed Implementation

### 1. Feature Definition
To ensure compatibility with the `image_transforms` pipeline, visual features MUST be prefixed with `observation.images.`. Proprioception should be stored in `observation.state`.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Standard naming for proprioception/state and actions
joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

features = {
    "observation.state": {
        "dtype": "float32",
        "shape": (len(joint_names),),
        "names": joint_names,
    },
    "action": {
        "dtype": "float32",
        "shape": (len(joint_names),),
        "names": joint_names,
    },
    # Image features must use the 'observation.images.<name>' prefix
    "observation.images.top_rgb": {
        "dtype": "video",
        "shape": (480, 640, 3), # (H, W, C)
        "names": ["height", "width", "channels"],
    },
    "observation.images.wrist_rgb": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
}
```

### 2. Dataset Initialization
Use `LeRobotDataset.create` to initialize the dataset with these features.

```python
dataset = LeRobotDataset.create(
    repo_id="your-org/your-dataset", # Or a local path
    root="path/to/dataset/root",
    fps=30, # Recording/Teleop frequency
    features=features,
    use_videos=True, # Recommended for image observations
)
```

### 3. Adding Data (Recording Loop)
During your data collection loop, format the observation dictionary before passing it to `add_frame`.

```python
# In your recording loop:
frame = {
    "observation.state": current_joint_state_tensor,
    "action": target_action_tensor,
    "observation.images.top_rgb": top_camera_frame, # numpy array or tensor
    "observation.images.wrist_rgb": wrist_camera_frame,
    "task": "Pick up the block", # Optional task description
}
dataset.add_frame(frame)

# After the episode:
dataset.save_episode()

# After all recording:
dataset.finalize()
```

### 4. Enabling Augmentations during Training
When training, the `LeRobotDataset` loader will automatically apply transformations if an `ImageTransforms` object is passed to the constructor. This object is typically built from an `ImageTransformsConfig`.

```python
from lerobot.datasets.transforms import ImageTransforms, ImageTransformsConfig

# Configure the augmentation pipeline
transforms_config = ImageTransformsConfig(
    enable=True,
    max_num_transforms=3, # Randomly apply up to 3 transforms
    random_order=True,
)
transforms = ImageTransforms(transforms_config)

# Initialize dataset for training with transforms enabled
dataset = LeRobotDataset(
    repo_id="your-org/your-dataset",
    image_transforms=transforms # This hooks into the pipeline
)
```

> [!IMPORTANT]
> The augmentation pipeline uses `torchvision.transforms.v2`. The `ImageTransforms` class iterates over all keys starting with `observation.images.` and applies the configured transforms (e.g., `ColorJitter`, `SharpnessJitter`) to each image tensor.

### 5. Advanced: Custom Transform Logic (RandomSubsetApply)
If you need more control, you can use `RandomSubsetApply` (added in PR #234) to group specific transforms that should be applied together:

```python
from lerobot.datasets.transforms import RandomSubsetApply
from torchvision.transforms import v2

custom_tf = RandomSubsetApply(
    [v2.ColorJitter(brightness=0.1), v2.GaussianBlur(3)],
    n_subset=1,
    random_order=True
)
```

### 6. Geometric Consistency and Symmetries
As identified in research on augmented LeRobot datasets (e.g., `twarner/lerobot-augmented`), when applying geometric transforms like horizontal flips, you MUST synchronize these with the proprioceptive state if it is orientation-sensitive:

- **Image Flip**: If you horizontally flip camera frames, you should negate orientation-sensitive features in `observation.state` (e.g., negating joint angles for shoulder-pan or Y-axis positions).
- **Offline Augmentation**: For specialized needs, you can pre-process your dataset to create variants (e.g., a "flipped" version of an episode) to quadruple your training data diversity.

## Verification Plan

### Manual Verification
1. **Run Inspection**: Use the `lerobot-inspect` tool or the local `scripts/dataset inspect` script to verify that the dataset includes all cameras and proprioception fields.
   ```bash
   python -m scripts.dataset inspect --dataset_root path/to/dataset
   ```
2. **Check Training Resume**: Use the provided `resume_with_augmentations.sh` script to verify that the training script starts without errors and detects the multiple camera streams.
   ```bash
   ./resume_with_augmentations.sh path/to/checkpoint
   ```
3. **Visualize Augmentations**: (Optional) Use `lerobot-imgtransform-viz` if available to see the augmented images.

## Open Questions
- Is there a specific set of augmentations you want to prioritize (e.g., ColorJitter, RandomCrop)?
- Are your camera streams synchronized at the hardware level, or do we need to account for slight temporal offsets? (LeRobot expects synchronized frames per observation).
