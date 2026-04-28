# 10k-Step "Micro-Sweep" Implementation Plan

## User Review Required

> [!WARNING]
> This plan officially Abandons the Karpathy "Time-Constrained" subprocess wrapper in favor of the **10k Fixed-Step** method. 
> 
> As discussed, a fixed-time sweep risks picking a model solely because it performs matrix multiplications faster (compute efficiency). A fixed-step sweep correctly identifies which model has superior underlying visual representations and a higher maximum success rate ceiling (sample efficiency), which is what matters for the 150k target.

## Goal Description
Conduct a high-efficiency comparative analysis of vision backbones (ResNet18, DINOv2-Small, CLIP-B/16) on the LeHome cloth folding task. We will run each for exactly 10,000 steps to evaluate early convergence and representation quality, evaluating against 50 simulation episodes.

## Proposed Changes

We will execute this entirely through standard LeRobot features and the "Bring Your Own Policy" (BYOP) framework (Path A from the `.agent/skills/lehome-lerobot-custom-policy` skill). **No changes to `lerobot_train.py` are needed.**

---

### Phase 1: BYOP Scaffold for DINO and CLIP

Since LeRobot v0.4.3 native `diffusion` policy hardcodes the assumption of a 2D CNN feature map via `torchvision`, we must create standard BYOP packages that intercept the image and encode it using `transformers` ViT models before passing those embeddings to the U-Net.

#### [NEW] `lehome_workspace/lerobot_policy_dino/`
We will create a lightweight BYOP package to support the DINOv2 encoder.
- **`configuration_dino_diffusion.py`**: Subclass of `PreTrainedConfig`, registered as `policy.type = dino_diffusion`.
- **`modeling_dino_diffusion.py`**: A `DinoDiffusionPolicy` class that replaces the standard `ObservationEncoder` with a frozen `Dinov2Model` from Hugging Face `transformers`, extracting the CLS token to condition the Diffusion U-Net.
- **`processor_dino_diffusion.py`**: Defines `make_dino_diffusion_pre_post_processors`.

#### [NEW] `lehome_workspace/lerobot_policy_clip/`
Identical structure to DINO, but wrapping `CLIPVisionModel`. Registered as `policy.type = clip_diffusion`.

---

### Phase 2: The Sweep Orchestrator

#### [NEW] `lehome_workspace/run_10k_sweep.sh`
Instead of Python subprocess managers, we will use a robust bash script that simply sequences the 10k training commands via the official `lerobot-train` CLI.

```bash
#!/bin/bash
# run_10k_sweep.sh

# Common arguments
STEPS=10000
BATCH_SIZE=8
DATASET="dataset_name" # Placeholder, update to target LeHome garment dataset

echo "Starting Run A: ResNet18 (Native DP, ImageNet Pretrained)"
lerobot-train \
    --policy.type=diffusion \
    --policy.vision_backbone=resnet18 \
    --policy.use_group_norm=false \
    --dataset.repo_id=$DATASET \
    --training.offline.steps=$STEPS \
    --eval.eval_freq=$STEPS \
    --eval.batch_size=$BATCH_SIZE \
    --wandb.enable=true \
    --wandb.run_name="Run_A_ResNet18_ImageNet"

# ... followed by Run B (dino_diffusion), Run C (clip_diffusion), and Run D (random init)
```

By setting `--eval.eval_freq=10000` and `--training.offline.steps=10000`, LeRobot will automatically hit exactly one evaluation at the very end of the run and save the checkpoint gracefully.

---

### Phase 3: Documentation Sync

#### [MODIFY] `lehome_workspace/lehome_change_log.md`
- Remove the entry for `time_constrained_sweep.py` (which we are abandoning).
- Log the creation of `lerobot_policy_dino`, `lerobot_policy_clip`, and `run_10k_sweep.sh`.

## Finalized Setup Decisions

1. **Dataset Selection**: We will use a **single garment type** dataset (e.g., `dp_pant_long`) for the sweep. Since your final architecture uses 4 separate DP policies (one per garment), we do not need the sweep to solve "all garments at once." A single garment type is representative of the complex deformation challenges (wrinkles, sleeves/legs) and is much more likely to show meaningful convergence differences within only 10,000 steps than a mixed dataset.
2. **Computational Limits**: The configurations will be executed **sequentially** in a single consolidated bash script `run_10k_sweep.sh`.

## Verification Plan

1. **Dry Run**: We will test `run_10k_sweep.sh` with `--training.offline.steps=10` to ensure all policies (Native ResNet and both BYOPs) compile, successfully encode an image, complete a forward/backward pass, and don't throw Out-Of-Memory (OOM) errors at batch size 8.
2. **Verification**: If ViT-Base models OOM at batch size 8, we will fall back to batch size 4 for those runs in the bash script.
