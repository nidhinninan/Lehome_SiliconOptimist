#!/bin/bash

# Base directory on your VM
BASE_DIR="/data/lehome_workspace/lehome-challenge"

EVAL_UTILS="$BASE_DIR/scripts/utils/eval_utils.py"
EVALUATION="$BASE_DIR/scripts/utils/evaluation.py"

echo "Patching eval_utils.py..."
python3 -c "
import os
path = '${EVAL_UTILS}'
if not os.path.exists(path):
    print(f'Error: Could not find {path}')
    exit(1)

with open(path, 'r') as f:
    text = f.read()

# Fix 1: Add garment_name argument and prefix logic
old_func_start = '''    fps: int = 30,
) -> None:
    \"\"\"Save captured frames as MP4 videos.\"\"\"
    if success.item():
        target_dir = os.path.join(save_dir, \"success\")
    else:
        target_dir = os.path.join(save_dir, \"failure\")

    os.makedirs(target_dir, exist_ok=True)

    for key, frames in all_episode_frames.items():'''

new_func_start = '''    fps: int = 30,
    garment_name: Optional[str] = None,
) -> None:
    \"\"\"Save captured frames as MP4 videos.\"\"\"
    if success.item():
        target_dir = os.path.join(save_dir, \"success\")
    else:
        target_dir = os.path.join(save_dir, \"failure\")

    os.makedirs(target_dir, exist_ok=True)

    # Prefix with garment name if provided to avoid overwriting
    prefix = f\"{garment_name}_\" if garment_name else \"\"

    for key, frames in all_episode_frames.items():'''

text = text.replace(old_func_start, new_func_start)

# Fix 2: Add prefix to filename
old_path_def = '''        out_path = os.path.join(
            target_dir, f\"episode{episode_idx}_{key.replace('.', '_')}.mp4\"
        )'''
new_path_def = '''        out_path = os.path.join(
            target_dir, f\"{prefix}episode{episode_idx}_{key.replace('.', '_')}.mp4\"
        )'''

text = text.replace(old_path_def, new_path_def)

with open(path, 'w') as f:
    f.write(text)
"

echo "Patching evaluation.py..."
python3 -c "
import os
path = '${EVALUATION}'
if not os.path.exists(path):
    print(f'Error: Could not find {path}')
    exit(1)

with open(path, 'r') as f:
    text = f.read()

old_call = '''        # Save Videos (Using generic util)
        if args.save_video:
            save_videos_from_observations(
                episode_frames,
                success=success if success_flag else torch.tensor(False),
                save_dir=args.video_dir,
                episode_idx=i,
            )'''

new_call = '''        # Save Videos (Using generic util)
        if args.save_video:
            save_videos_from_observations(
                episode_frames,
                success=success if success_flag else torch.tensor(False),
                save_dir=args.video_dir,
                episode_idx=i,
                garment_name=garment_name,
            )'''

text = text.replace(old_call, new_call)

with open(path, 'w') as f:
    f.write(text)
"

echo "Patching evaluation.py for RateLimiter Speedup..."
python3 -c "
import os
path = '${EVALUATION}'
if not os.path.exists(path):
    print(f'Error: Could not find {path}')
    exit(1)

with open(path, 'r') as f:
    text = f.read()

old_rl = '''    all_episode_metrics = []
    logger.info(f\"Starting evaluation: {args.num_episodes} episodes\")
    rate_limiter = RateLimiter(args.step_hz)'''

new_rl = '''    all_episode_metrics = []
    logger.info(f\"Starting evaluation: {args.num_episodes} episodes\")
    rate_limiter = RateLimiter(args.step_hz) if args.step_hz > 0 else None'''

text = text.replace(old_rl, new_rl)

with open(path, 'w') as f:
    f.write(text)
"

echo "Done! The video overwrite bug and RateLimiter speedup have been patched on your VM."
