# DPPO Feasibility Analysis: Using Hackathon Metrics as Reward

## Question 1: Are we training from scratch or fine-tuning?

**Training from scratch.** From the config:
- `pretrained_backbone_weights: None` — ResNet18 vision backbone starts from random weights
- `pretrained_path: None` — No pretrained diffusion model loaded

The model is learning the entire policy (vision encoder + diffusion denoising U-Net) purely from your 250 episodes of demonstration data via Behavioral Cloning (supervised learning on noise prediction loss).

---

## Question 2: Can the hackathon eval metric be used directly as DPPO reward?

**Yes — and remarkably, the hackathon repo already provides exactly the dense reward function you'd need.**

### What the hackathon metric actually is

After reading the full pipeline ([garment_bi_v2.py](file:///home/nidhinninan/Documents/Research/Hackathon/LeHome-trial/lehome_workspace/lehome-challenge/source/lehome/lehome/tasks/bedroom/garment_bi_v2.py#L346-L465) + [success_checker_chanllege.py](file:///home/nidhinninan/Documents/Research/Hackathon/LeHome-trial/lehome_workspace/lehome-challenge/source/lehome/lehome/utils/success_checker_chanllege.py)):

1. **Success = binary.** Specific cloth particle pairs must be within distance thresholds (e.g., for `top-short-sleeve`: 5 conditions on particle distances, some `<=` and some `>=`).
2. **But [_get_rewards()](file:///home/nidhinninan/Documents/Research/Hackathon/LeHome-trial/lehome_workspace/lehome-challenge/source/lehome/lehome/tasks/bedroom/garment_bi_v2.py#346-467) already implements a dense distance-based reward** that returns continuous values in `[0, 0.9]` (reserving 1.0 for full success):
   - **Primary conditions (80% weight):** Folding distances (exponential decay penalty)
   - **Secondary conditions (20% weight):** Shape constraints (gentler curve)
   - Uses geometric mean of avg/min primary rewards to penalize if *any* primary condition is bad

This is a well-shaped, dense reward — exactly what DPPO would need.

### Would it work for DPPO?

| Factor | Assessment |
|---|---|
| **Reward density** | ✅ Dense. The existing [_get_rewards()](file:///home/nidhinninan/Documents/Research/Hackathon/LeHome-trial/lehome_workspace/lehome-challenge/source/lehome/lehome/tasks/bedroom/garment_bi_v2.py#346-467) gives continuous gradient signal, not just 0/1. |
| **Reward alignment** | ✅ Perfectly aligned with hackathon objective — it literally IS the hackathon objective. |
| **Differentiability** | ⚠️ Not needed. DPPO uses policy gradients (PPO), not backprop through the reward. |
| **Computational cost** | ⚠️ The `@step_interval(interval=50)` decorator only checks every 50 steps. For RL, you'd want more frequent reward signals — you'd need to remove or reduce this interval. |
| **Sim speed** | ❌ **This is the real blocker.** DPPO needs thousands of rollouts *during training*. Each rollout requires running the Isaac Lab sim with PhysX cloth simulation. On a single RTX A4000, this would be extremely slow. |

---

## Audit of My Prior DPPO Response

### What I got right ✅
- DPPO adds an RL (PPO) optimization wrapper on top of a pre-trained DP — correct
- It requires a live simulation environment during training — correct
- It requires a reward function — correct, and **the hackathon already provides one**
- 250 episodes is small for BC — correct; the original paper uses larger datasets for complex tasks

### What I got wrong or overstated ❌

> [!WARNING]
> **"writing a dense, accurate reward function for cloth folding (which is notoriously difficult)"**
> 
> This was **misleading in your specific case.** The LeHome hackathon **already ships a dense, well-shaped reward function** in [_get_rewards()](file:///home/nidhinninan/Documents/Research/Hackathon/LeHome-trial/lehome_workspace/lehome-challenge/source/lehome/lehome/tasks/bedroom/garment_bi_v2.py#346-467). You would NOT need to write one from scratch. The "notoriously difficult" claim applies to real-world cloth folding where you don't have privileged particle positions — but in this Isaac Lab sim, particle positions are directly accessible. The hard reward design work is already done.

> [!NOTE]
> The statement about a "Value Network" being needed was correct but could mislead — DPPO's PPO layer handles this, you wouldn't code it from scratch.

---

## Practical Feasibility Assessment

### Could you actually do DPPO for this hackathon?

**Theoretically yes. Practically, it's a very hard lift for these reasons:**

1. **Infrastructure gap:** The DPPO repo (`irom-princeton/dppo`) is built for MuJoCo/D4RL environments. Adapting it to Isaac Lab + PhysX cloth sim is non-trivial engineering.
2. **Sim throughput:** DPPO needs ~1000s of rollouts. Your eval currently takes ~minutes per episode. Even parallelized, you're looking at days of RL training on a single A4000.
3. **Scope:** This is a hackathon, not a PhD thesis. The BC approach at 150K steps is the pragmatic choice.

### What would actually help more with 250 episodes (Cross-Validated)

If the BC policy at 150K steps underperforms, try these in order of impact/effort:

---

#### 1. Enable Image Augmentations — Resident Data Pipeline

> [!NOTE]
> **Can I resume with this? YES.** 
> You do **NOT** have to start from scratch to enable augmentations. Because `image_transforms` are applied on-the-fly at the data-loading stage (before the image reaches the model), they do not change the network's layers or weights. You can keep your current 150K step progress and simply resume with augmentations enabled.

**Exact defaults for the 6 pre-configured transforms:**
| Transform | Default Range | Weight | Type |
| :--- | :--- | :--- | :--- |
| **Brightness** | `[0.8, 1.2]` | 1.0 | ColorJitter |
| **Contrast** | `[0.8, 1.2]` | 1.0 | ColorJitter |
| **Saturation** | `[0.5, 1.5]` | 1.0 | ColorJitter |
| **Hue** | `[-0.05, 0.05]` | 1.0 | ColorJitter |
| **Sharpness** | `[0.5, 1.5]` | 1.0 | SharpnessJitter |
| **Affine** | `deg: [-5, 5], trans: [5%, 5%]` | 1.0 | RandomAffine |

**Config change — simply flip the enable flag:**
```yaml
dataset:
  image_transforms:
    enable: true
```
*Note: `max_num_transforms` defaults to 3, meaning LeRobot will randomly choose up to 3 of these 6 to apply to each image frame during training.*

**How to resume with augmentations (No Scratch Needed):**
You can use the following Hydra override command to enable augmentations while resuming from your checkpoint:
```bash
# Resume script I created for you:
./resume_with_augmentations.sh outputs/train/dp_top_short/checkpoints/last

# OR manually:
lerobot-train \
    --checkpoint_path outputs/train/dp_top_short/checkpoints/last \
    --resume true \
    --dataset.image_transforms.enable true \
    --num_workers 4
```

### Monitoring CPU Usage
To verify if your CPU workers are actually running:
1.  Open a **new terminal**.
2.  Run: `htop`
3.  **Check:** You should see multiple `lerobot-train` processes. If you only see one, the workers are not active.

> [!IMPORTANT]
> **Syntax Note:** LeRobot 0.4.2 uses `draccus`, which requires overrides to be prefixed with `--` (unlike Hydra). Ensure you use `--dataset.image_transforms.enable true`.

**Validation:** LeRobot docs reference a `lerobot-imgtransform-viz` CLI tool for previewing augmentation effects. This appears to be a real entry point in the package. If unavailable, you can visually verify by logging augmented frames during the first few training steps.

---

#### 2. Use Pretrained Vision Backbone — High Impact, Needs Care

The ResNet18 backbone in LeRobot's Diffusion Policy can be initialized with ImageNet weights instead of random weights. This gives the encoder a massive head start in understanding visual features (edges, textures, shapes).

> [!WARNING]
> **CRITICAL: GroupNorm vs BatchNorm Conflict — SOURCE-CODE VERIFIED**
> LeRobot's `DiffusionRgbEncoder.__init__()` ([modeling_diffusion.py](https://github.com/huggingface/lerobot/blob/4dbbcca4/src/lerobot/policies/diffusion/modeling_diffusion.py#L369-L418)) **explicitly raises a `ValueError`** if both `use_group_norm=True` and `pretrained_backbone_weights` are set:
> ```python
> if config.use_group_norm:
>     if config.pretrained_backbone_weights:
>         raise ValueError(
>             "You can't replace BatchNorm in a pretrained model without ruining the weights!"
>         )
> ```
> You **MUST** set `use_group_norm: false` when using pretrained weights. The code will crash immediately otherwise.

**Config change:**
```yaml
policy:
  pretrained_backbone_weights: "ResNet18_Weights.IMAGENET1K_V1"
  use_group_norm: false   # CRITICAL: must be false with pretrained weights
```

> [!CAUTION]
> **You MUST train from scratch.** You cannot resume from your existing 150K checkpoint (which was trained with `use_group_norm: true` / random weights). Switching `use_group_norm` changes the network architecture (GroupNorm layers → BatchNorm layers), making old checkpoint weights incompatible. The shapes won't even match.

**Why this helps:** With 250 episodes, the policy spends a large fraction of its 150K steps just learning to "see" (edge detection, texture recognition). Pretrained weights skip this phase entirely, letting the model focus on learning the folding motion.

**Trade-off:** BatchNorm (used with pretrained weights) is slightly less stable with small batch sizes (your batch_size=8). However, this is generally acceptable for fine-tuning scenarios where the backbone is already well-initialized.

**Recommendation:** Try this AFTER confirming that augmentations alone don't solve the problem. This change requires a full new training run from step 0.

---

#### 3. ✅ Collect More Demonstrations — **VALID BUT CONTEXT-DEPENDENT**

More data always helps, but the value depends on *what kind* of data:
- **Diverse initial configurations** (different cloth positions/orientations) → high value
- **Repetitive demos of the same fold** → diminishing returns
- Since you're training in sim, you could also use the sim's built-in randomization (`texture_randomization`, `light_randomization` in the garment config) to generate visual diversity without recording new demos

---

#### 4. ✅ Tune `drop_n_last_frames` and `n_action_steps` — **ALREADY AT GOOD DEFAULTS**

Your current values are:
- `horizon: 16` — predicts 16 future action steps
- `n_action_steps: 8` — executes the first 8, then re-plans
- `drop_n_last_frames: 7` — drops incomplete sequences near episode ends
- `n_obs_steps: 2` — uses 2 observation frames for context

These follow the standard formula: `drop_n_last_frames = horizon - n_action_steps - n_obs_steps + 1 = 16 - 8 - 2 + 1 = 7` ✅

**When to tune:**
| Symptom | Fix |
|---|---|
| Robot movements are jerky/stuttery | Increase `horizon` (e.g., 24 or 32) |
| Robot doesn't react to cloth state changes | Decrease `n_action_steps` (e.g., 4) |
| Robot moves too slowly during eval | Increase `n_action_steps` |

**Verdict:** Leave these alone unless you observe specific behavioral issues during eval. The defaults are robust for bimanual manipulation.

---

**Summary:** Suggestions 1 and 3 are safe config changes. Pretrained weights (2) require a GroupNorm fix. Action chunking (4) is already correct.
