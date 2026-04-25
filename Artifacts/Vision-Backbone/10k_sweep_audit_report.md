# 10k Micro-Sweep Audit Report

**Date**: 2026-04-11  
**Auditor**: Independent code review against LeRobot v0.4.x API  
**Sources**: DeepWiki (`huggingface/lerobot`), Exa code search (GitHub source), HF Docs (BYOP tutorial)

---

## Verdict: 🔴 NOT EXECUTABLE — 7 Critical Bugs, 3 Moderate Issues

The implementation contains **fatal errors** that will crash at import-time or during the first training step. The bugs fall into three categories: (1) wrong LeRobot API surface assumptions, (2) BYOP naming convention violations, and (3) bash script CLI argument errors.

---

## Critical Bugs (Will Crash)

### BUG-1: `DiffusionConfig.__post_init__` blocks non-ResNet backbones 🔴
**Files**: `configuration_dino_diffusion.py` (L5-13), `configuration_clip_diffusion.py` (L5-13)  
**Evidence**: [configuration_diffusion.py source](https://github.com/liyitenga/lerobot_alohamini/blob/ae2ffc8d/lerobot/common/policies/diffusion/configuration_diffusion.py)

```python
# In DiffusionConfig.__post_init__:
if not self.vision_backbone.startswith("resnet"):
    raise ValueError(
        f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
    )
```

Our configs inherit from `DiffusionConfig` and set `vision_backbone = "facebook/dinov2-small"` and `"openai/clip-vit-base-patch16"`. Since these don't start with `"resnet"`, **`__post_init__` will raise `ValueError` immediately on instantiation**.

**Fix**: Override `__post_init__` in both config classes to skip the ResNet-only validation while still running other validations (prediction_type, noise_scheduler_type, horizon/downsample compat).

---

### BUG-2: `noise_scheduler_kwargs` does not exist on `DiffusionConfig` 🔴
**Files**: `modeling_dino_diffusion.py` (L36-39), `modeling_clip_diffusion.py` (L32-35)

```python
# Our code:
self.noise_scheduler = _make_noise_scheduler(
    config.noise_scheduler_type, 
    **config.noise_scheduler_kwargs   # <-- THIS FIELD DOES NOT EXIST
)
```

**Evidence**: The real `DiffusionModel.__init__` passes individual fields explicitly:
```python
# Actual LeRobot code:
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
```

**Fix**: Replace `**config.noise_scheduler_kwargs` with the explicit field references.

---

### BUG-3: Wrong import path for `Normalize`/`Unnormalize` 🔴
**Files**: `modeling_dino_diffusion.py` (L129), `modeling_clip_diffusion.py` (L95)

```python
# Our code:
from lerobot.policies.normalization import Normalize, Unnormalize
```

**Evidence**: The real import path in `modeling_diffusion.py` is:
```python
from lerobot.common.policies.normalize import Normalize, Unnormalize
```

However, this is the **old** import path (pre-v0.4.x). In v0.4.x+ with the processor migration (PR #1452), normalization is handled by `PolicyProcessorPipeline` and `NormalizerProcessorStep` in `lerobot.processor.normalize_processor`. The `Normalize`/`Unnormalize` nn.Modules still exist for backward compatibility but the path is version-dependent.

**Fix**: For v0.4.3, check whether the module lives at `lerobot.policies.normalize` or `lerobot.common.policies.normalize`. Most likely: `from lerobot.policies.normalize import Normalize, Unnormalize` (the `common` prefix was dropped in the src layout reorganization).

---

### BUG-4: BYOP file naming convention mismatch 🔴
**Files**: All DINO/CLIP files

The LeRobot factory discovers third-party policies via **strict naming conventions** (from `_get_policy_cls_from_policy_name` in `factory.py`):

| Convention | Expected for `dino_diffusion` | Our actual filename |
|---|---|---|
| Config module | `configuration_dino_diffusion.py` | ✅ `configuration_dino_diffusion.py` |
| Modeling module | `modeling_dino_diffusion.py` | ✅ `modeling_dino_diffusion.py` |
| Config class | `DinoDiffusionConfig` | ✅ `DinoDiffusionConfig` |
| Policy class | `DinoDiffusionPolicy` | ✅ `DinoDiffusionPolicy` |
| Processor function | **`make_dino_diffusion_pre_post_processors`** | ✅ `make_dino_diffusion_pre_post_processors` |
| Processor module | **`processor_dino_diffusion.py`** | ✅ `processor_dino_diffusion.py` |

✅ **Naming is actually correct for DINO.** Same for CLIP (`clip_diffusion`). **No bug here** — I initially suspected one but verified it's fine.

However, there **is** a discovery problem: the factory function `_get_policy_cls_from_policy_name()` constructs the module path by looking at the config class's `__module__` attribute and swapping `configuration_` for `modeling_`. Since our config lives in `lerobot_policy_dino.configuration_dino_diffusion`, the factory will look for `lerobot_policy_dino.modeling_dino_diffusion`. This **should work** because `pip install -e` makes the package importable. ✅

---

### BUG-5: `observation.state` batch shape assumption is wrong 🔴
**Files**: `modeling_dino_diffusion.py` (L64), `modeling_clip_diffusion.py` (L54)

```python
all_features = [batch["observation.state"]]  # Assumed shape: (B, N, state_dim)
```

In the standard DiffusionModel, the robot state is accessed via the constant `OBS_ROBOT`:
```python
from lerobot.common.constants import OBS_ENV, OBS_ROBOT
# ...
batch_size, n_obs_steps = batch[OBS_ROBOT].shape[:2]
global_cond_feats = [batch[OBS_ROBOT]]
```

The key `OBS_ROBOT` resolves to `"observation.state"` so the key name is fine, but the **shape** after normalization may be `(B, n_obs_steps, state_dim)` or `(B, state_dim)` depending on whether the data pipeline has already applied delta_timestamps. This needs verification but is likely OK since the image batch also has the same temporal structure.

**Verdict**: Probably OK but fragile — should use `OBS_ROBOT` constant for correctness.

---

### BUG-6: `num_inference_steps` not stored as instance attribute 🔴
**Files**: `modeling_dino_diffusion.py` (L107), `modeling_clip_diffusion.py` (L81)

```python
self.noise_scheduler.set_timesteps(self.config.num_inference_steps)
```

The real `DiffusionModel` stores `self.num_inference_steps` as an instance attribute with a fallback:
```python
if config.num_inference_steps is None:
    self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
else:
    self.num_inference_steps = config.num_inference_steps
```

Our code accesses `self.config.num_inference_steps` directly, which could be `None` by default, causing the scheduler to fail.

**Fix**: Add the same fallback logic.

---

### BUG-7: Sweep script uses wrong CLI arguments 🔴
**File**: `run_10k_sweep.sh`

| Our CLI arg | Correct arg | Issue |
|---|---|---|
| `--config=$BASE_CONFIG` | `--config_path=$BASE_CONFIG` | Wrong flag name — `lerobot-train` uses `--config_path` per draccus |
| `--eval.eval_freq=$EVAL_FREQ` | `--eval_freq=$EVAL_FREQ` | `eval_freq` is a **top-level** field on `TrainPipelineConfig`, not nested under `eval` |
| `python -m lerobot.scripts.train` | `python -m lerobot.scripts.lerobot_train` or `lerobot-train` | The entry point script is `lerobot_train.py`, not `train.py` |

**Evidence**: DeepWiki confirms `eval_freq` and `steps` are top-level fields. The CLI entry point is `lerobot-train` (mapped to `lerobot.scripts.lerobot_train:train`). The config file is loaded via `--config_path`.

**Fix**: Correct all three CLI arguments.

---

## Moderate Issues

### MOD-1: Typo — `horror` variable name
**Files**: `modeling_dino_diffusion.py` (L102), `modeling_clip_diffusion.py` (L78)

```python
horror = self.config.horizon  # Should be `horizon`
```

Functionally correct but embarrassing if someone reads the code.

---

### MOD-2: DINOv2 image size mismatch risk
The LeHome images are `480x640`. DINOv2-Small expects `224x224` by default (its `image_processor` handles resizing). However, since we're passing raw tensors directly to `Dinov2Model`, **there is no automatic resizing**. The ViT will process `480x640` images, creating `(480/14) * (640/14) ≈ 1565` patch tokens instead of the expected 256. This will work (ViTs handle variable-length sequences) but will be **extremely slow and memory-hungry**.

**Fix**: Add a `torchvision.transforms.Resize((224, 224))` before the backbone, or use `Dinov2ForImageClassification`'s built-in preprocessing.

---

### MOD-3: CLIP image preprocessor not applied
Same issue as MOD-2. CLIP ViT-B/16 expects `224x224` images. Without resizing, the 480x640 input will create `(480/16) * (640/16) = 1200` patches vs the expected 196. This **may cause a positional embedding shape mismatch error** since CLIP uses fixed-length positional embeddings (unlike DINOv2 which can handle variable lengths).

**Fix**: This is **likely a crash** for CLIP specifically. Must resize to 224x224 before passing to CLIPVisionModel.

---

## Summary Table

| ID | Severity | File(s) | Will Crash? | Fix Complexity |
|---|---|---|---|---|
| BUG-1 | 🔴 Critical | Both configs | Yes, at import | Override `__post_init__` |
| BUG-2 | 🔴 Critical | Both models | Yes, at init | Replace with explicit kwargs |
| BUG-3 | 🔴 Critical | Both models | Yes, at init | Fix import path |
| BUG-5 | 🟡 Moderate | Both models | Probably not | Use `OBS_ROBOT` constant |
| BUG-6 | 🔴 Critical | Both models | Yes, at inference | Add None fallback |
| BUG-7 | 🔴 Critical | `run_10k_sweep.sh` | Yes, at launch | Fix 3 CLI args |
| MOD-1 | ⚪ Cosmetic | Both models | No | Rename variable |
| MOD-2 | 🟡 Moderate | DINO model | No, but slow | Add resize transform |
| MOD-3 | 🔴 Critical | CLIP model | Yes, pos embed | Add resize transform |
