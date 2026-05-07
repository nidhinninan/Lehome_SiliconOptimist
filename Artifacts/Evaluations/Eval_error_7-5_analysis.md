Now I have a complete picture. Here is the full grounded analysis and explanation.

---

## Project Journey — Grounded Synthesis

### 1. The Overall Arc (What Was Done)

The project is a **LeHome Challenge** submission for garment folding with a bimanual robot in Isaac Sim. The work went through five distinct phases:

**Phase 1 → Baseline Diffusion Policy (ResNet-18, 30K steps)**
A first run with LeRobot's native ResNet-18 backbone ran for only 30K steps. The cosine LR scheduler was set to decay over 30K, so by the time the training ended, the LR had decayed to near zero prematurely. The model reached loss 0.025 but was LR-starved — the model hadn't truly converged, it just ran out of learning rate.

**Phase 2 → Vision Backbone Research (DINOv2 vs CLIP vs ResNet)**
Deep research was done comparing DINOv2, CLIP, and ResNet for cloth-folding visuomotor control. The key conclusion: DINOv2 is the best backbone for deformable cloth tasks because it preserves dense spatial patch features. CLIP is language-semantic, less precise for geometry. The CLS token alone discards spatial info — a MAP (Multi-headed Attention Pooling) head preserves it. The **registers variant** (`dinov2-with-registers-small`) was identified as essential to prevent high-norm artifact tokens from corrupting spatial readout.

**Phase 3 → BYOP Custom Policy (`dino_diffusion`) Implementation**
A completely custom LeRobot "Bring Your Own Policy" (BYOP) plugin was built: `lerobot_policy_dino`. This surgically subclasses `DiffusionPolicy` while swapping in the DINOv2 backbone + MAP head. Multiple crashes were hit and resolved (plugin registry order, namespace package trap, frozen backbone optimizer crash, API signature mismatches). The final architecture uses `facebook/dinov2-with-registers-small`, a frozen backbone, a trainable MAP head (8 queries × 384-dim = 3072-dim per camera), and the standard Diffusion U-Net conditioned on a flattened vision+state vector.

**Phase 4 → 150K Training Run + Augmentation Resume**
A full 150K-step training run was launched on the merged 4-garment dataset. Training went cleanly: loss dropped from 0.303 → 0.022 by 93K steps. At 93K the model entered the "usable zone" (loss < 0.024). The final 57K steps were done as an augmentation resume (`resume_with_augmentations.sh`) — enabling 6 photometric+mild geometric transforms to force visual invariance and reduce overfitting to specific pixel values in the 250-episode dataset.

**Phase 5 → Eval Pipeline Debugging**
Evaluation on Isaac Sim required resolving several environmental issues: EULA blocking, Warp API version incompatibility, plugin registration order for eval (same problem as training), and Vulkan duplicate ICD. The log you're looking at is the **first successful launch** of the eval pipeline with the DINOv2 custom policy.

---

### 2. The Root-Cause of the Crash: `KeyError: 'observation.images.top_rgb'`

This is a **mismatch between how the upstream `DiffusionPolicy.predict_action_chunk` reshapes the batch vs. how `DinoDiffusionModel._prepare_global_conditioning` reads it**.

Here is the precise chain of events:

**Step A — in `DiffusionPolicy.select_action`** (the *inherited* parent method):
```python
# Individual image keys are STACKED into one combined key
batch[OBS_IMAGES] = torch.stack(
    [batch["observation.images.top_rgb"],
     batch["observation.images.left_rgb"],
     batch["observation.images.right_rgb"]],
    dim=-4
)
# This + the state key go into the temporal queue
self._queues = populate_queues(self._queues, batch)
```
The queue tracks `OBS_STATE`, `OBS_IMAGES`, and `ACTION` — **not** individual camera keys.

**Step B — in `predict_action_chunk`** (also the *inherited* parent method):
```python
# CRITICAL: rebuilds batch from queue — only queue keys survive
batch = {k: torch.stack(list(self._queues[k]), dim=1)
         for k in batch if k in self._queues}
actions = self.diffusion.generate_actions(batch, ...)
```
After this line, `batch` **only contains** `OBS_STATE` and `OBS_IMAGES` (the stacked combined tensor). The original per-camera keys like `"observation.images.top_rgb"` are **gone**.

**Step C — in `DinoDiffusionModel._prepare_global_conditioning`** (the *custom* code):
```python
# Tries to look up each camera key separately — but they no longer exist
for img_key in self.config.image_features:   # e.g. "observation.images.top_rgb"
    img = batch[img_key]   # KeyError: 'observation.images.top_rgb'
```

**The standard `DiffusionModel` never hits this** because it uses `batch[OBS_IMAGES]` (the stacked key). The custom `DinoDiffusionModel` was written to use the individual named keys, which exist before the queue but are stripped out by the parent class's queue mechanics.

---

### 3. How to Fix It

The fix belongs in `DinoDiffusionModel._prepare_global_conditioning` in `/data/lehome_workspace/lerobot_policy_dino/src/lerobot_policy_dino/modeling_dino_diffusion.py`.

Instead of looking up `batch[img_key]` for each camera, read from `batch[OBS_IMAGES]` which is shaped `(B, T, N_cams, C, H, W)` and slice each camera by index:

```python
from lerobot.utils.constants import OBS_IMAGES

def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
    B = batch[OBS_ROBOT].shape[0]
    image_features = []

    if OBS_IMAGES in batch:
        # Shape from queue: (B, T, N_cams, C, H, W)
        stacked = batch[OBS_IMAGES]
        n_cams = stacked.shape[2]
        for cam_idx in range(n_cams):
            img = stacked[:, :, cam_idx, :, :, :]  # (B, T, C, H, W)
            _, N, C, H, W = img.shape
            feat = self._encode_images(img.reshape(B * N, C, H, W))
            image_features.append(feat.view(B, N, -1))
    else:
        # Fallback: individual keys (e.g. used during direct batch construction)
        for img_key in self.config.image_features:
            img = batch[img_key]
            _, N, C, H, W = img.shape
            feat = self._encode_images(img.reshape(B * N, C, H, W))
            image_features.append(feat.view(B, N, -1))

    all_feats = [batch[OBS_ROBOT]]
    all_feats.extend(image_features)
    if self.config.env_state_feature:
        all_feats.append(batch[OBS_ENV])

    combined = torch.cat(all_feats, dim=-1)
    return combined.view(B, -1)
```

The camera order will match `self.config.image_features` order as long as that matches the stack order in `select_action` (which iterates over `self.config.image_features` in the same sequence — it does).

---

### 4. Are the DEBUG Logs Normal?

**Yes — almost entirely expected.** Here is what each class of debug output means:

| Log Class | Lines | Normal? | Meaning |
|---|---|---|---|
| `[ext: omni.*]` startup | 1–400 | ✅ Expected | Isaac Sim / Omniverse extension loading. Always present at startup. |
| `DEBUG:AutoNode:Defining data type...` | 200–257 | ✅ Expected | OmniGraph data type registry — fires once per session. |
| `DEBUG:h5py._conv`, `DEBUG:matplotlib` | 378–385 | ✅ Expected | HDF5 converter registration and matplotlib init. Harmless. |
| `DEBUG:draccus.*` (700+ lines) | 416–707 | ⚠️ Verbose but expected | `draccus` (LeRobot's config parser) reflecting on every field of `DinoDiffusionConfig`. Logs twice because the config is constructed twice (first pass for schema introspection, second pass to decode from `config.json`). Harmless; can be suppressed by setting `DEBUG` log level higher. |
| `DEBUG:filelock` | 402–415 | ✅ Expected | HuggingFace dataset cache file lock acquisition/release. Normal. |
| `DEBUG:urllib3` + HuggingFace HEAD/GET | 708–724 | ✅ Expected | DINOv2 backbone weight resolution — checks if local cache is current. Runs once at policy init. |
| `You are using a model of type dinov2_with_registers to instantiate a model of type dinov2.` | L718 | ⚠️ Warning — known | Transformers version mismatch: the HF `Dinov2Model` class doesn't yet have a native `dinov2_with_registers` config subclass in the installed version. The registers model loads anyway (weights are compatible), but the config `model_type` triggers the warning. Functionally OK; the registers variant still works. |
| `PhysX error: attachShape: non-SDF triangle mesh...` | L752 | ⚠️ Expected for cloth | Garment mesh is not an SDF-type collision shape, so PhysX can't simulate it as a fully dynamic rigid body. This is normal for cloth/deformable simulation in Isaac Sim — the cloth uses a particle solver, not rigid-body PhysX. Not a bug. |
| `Warning: Possible version incompatibility omni::fabric::IStageReaderWriter v0.14 vs v0.13` | L753 | ⚠️ Known/harmless | Minor API version mismatch between two Isaac Lab subcomponents. Observed consistently across runs; does not affect the simulation or cameras. |
| `Warp DeprecationWarning: warp.types.warp_type_to_np_dtype` | L759 | ⚠️ Deprecation, non-fatal | Warp internal symbol moved to a private namespace. Still accessible. Pin `warp-lang==1.11.1` (as documented in `eval_isaac_sim_environment_troubleshooting_note.md`) to avoid this. |
| The actual crash `KeyError: 'observation.images.top_rgb'` | L760–791 | ❌ The real bug | Described above. |

**The one actionable non-crash warning** worth tracking is the `dinov2_with_registers → dinov2` type mismatch. It means the installed `transformers` version doesn't have a proper class for the registers variant and is silently loading it as the base `dinov2` config. For current usage this is fine (weights still load), but it could cause subtle behavioral differences if the registers-specific config parameters matter to the backbone behavior.

---

### 5. Key Learnings Preserved in the Artifacts (Summary)

| Learning | Evidence |
|---|---|
| **CLS token alone wastes DINOv2's spatial power** — MAP head is the right readout | `dinov2_spatial_pooling_audit_thread_2026-04-25.md` + Voltron paper |
| **Registers variant is non-negotiable for any spatial pooling** — artifacts corrupt soft-argmax | `DINOv2_CLIP_ResNet_Architectural_Analysis.md`, Darcet ICLR'24 |
| **Freeze the backbone** for small datasets — prevents catastrophic forgetting | `Frozen_backbone.md` |
| **30K step LR starvation** was the failure mode of the first run | `training-DP_progress_analysis_33koutof150k.md` |
| **Augmentation resume at 93K** is the strategy to prevent pixel overfitting | `training-DP_augmentation_resume_strategy_150k.md` |
| **Plugin registration order** must happen before `draccus.parse` | `BYOP_training_crash_log.md` Error 1 & 2 |
| **`predict_action_chunk` strips individual image keys** — custom model must use `OBS_IMAGES` | The bug you just hit (not yet written up) |
| **`--step_hz 0`** removes the real-time limiter and gives 2–5× eval speedup | `eval_speedup_analysis.md` |
| **`--device cpu`** for physics + GPU for inference is the stable eval combo | Same |

Ready to proceed to implementing the fix in `modeling_dino_diffusion.py` whenever you give the go-ahead.

...

`OBS_IMAGES = "observation" + "." + "image" + "s"` = `"observation.images"`. The locally-defined constant is an exact match. Now let me verify the full review.

---

## Review Result: The Fix is Correct and Complete ✅

Here is a precise check of every item I expected against what was applied:

### What Was Changed

**1. `OBS_IMAGES` constant added at module level (line 27)**
```python
OBS_IMAGES = "observation.images"
```
✅ The string value is confirmed correct — matches `lerobot.utils.constants.OBS_IMAGES` exactly (`"observation" + "." + "image" + "s"`). Defining it locally avoids any import path fragility.

**2. `_prepare_global_conditioning` rewritten (lines 171–212)**

All expected elements are present:

| Check | Status |
|---|---|
| Primary path `if OBS_IMAGES in batch:` | ✅ Present |
| `stacked_images = batch[OBS_IMAGES]` with correct shape comment `(B, N_obs, N_cams, C, H, W)` | ✅ Present |
| Camera count via `stacked_images.shape[2]` (index 2 = N_cams axis) | ✅ Correct axis |
| Camera count mismatch validation with helpful error message | ✅ Present (added robustness beyond what I suggested) |
| Camera slice `stacked_images[:, :, cam_idx, :, :, :]` → `(B, N_obs, C, H, W)` | ✅ Correct |
| Uses `reshape` instead of `view` (safe for non-contiguous tensors from einops) | ✅ Present |
| Fallback `else:` path for direct named-key batches | ✅ Present |
| State and env-state concatenation unchanged | ✅ Untouched |

### Shape Sanity Check (Both Paths)

**Eval path** (queue-rebuilt batch in `predict_action_chunk`):
- Each image enters `select_action` as `(1, C, H, W)` after preprocessing
- Parent stacks 3 cameras with `dim=-4` on 4D tensors → dim=1 in output → `(1, N_cams, C, H, W)` stored in queue
- Queue stacks `n_obs_steps=2` entries along `dim=1` → `(1, 2, N_cams, C, H, W)` = `(B, S, N_cams, C, H, W)` ✅
- Fix slices `[:, :, cam_idx, :, :, :]` → `(B, S, C, H, W)` ✅

**Training path** (`compute_loss` via `DiffusionPolicy.forward`):
- Dataset loads images as `(B, n_obs_steps, C, H, W)` = `(B, 2, C, H, W)` (5D)
- Parent's `forward` stacks 3 cameras with `dim=-4` on 5D tensors → dim=2 in output → `(B, 2, N_cams, C, H, W)` = `(B, S, N_cams, C, H, W)` ✅
- Same fix path handles this identically ✅

### One Subtle Confirmation

The parent class `reset()` (inherited by `DinoDiffusionPolicy`) initializes `self._queues[OBS_IMAGES]` using the same string `"observation.images"` from `lerobot.utils.constants`. The locally-defined constant in your file produces the same string, so queue lookup is consistent throughout. `policy.reset()` is also explicitly called in the eval loop before `select_action` is ever hit, so `_queues` is always initialized. ✅

### Nothing Missing

The fix covers everything necessary. You can now re-run the eval command — the crash at `_prepare_global_conditioning` should be gone and the first episode should proceed to actual inference.

...

Correct — the fix is **backward-compatible and training-neutral**. Here is exactly why:

### What the fix changed

`_prepare_global_conditioning` now checks `if OBS_IMAGES in batch` first. That's it.

### What happens during training (now and before)

During training, `DiffusionPolicy.forward` is called. That method (parent class, **untouched**) does:

```python
batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
loss = self.diffusion.compute_loss(batch)
```

So the training batch **always** has `OBS_IMAGES` in it before reaching `_prepare_global_conditioning`. The primary `if OBS_IMAGES in batch:` path fires — the exact same computation as before, just reading from the stacked tensor instead of individual keys. The math, the tensor values, the gradients, the loss — **identical**.

### What was broken before (eval only)

The old code only had the fallback path (individual named keys). Those keys exist in the training batch but **not** in the eval batch (they get stripped by the parent's queue mechanics). So the bug was exclusively an **eval-time key-lookup failure** — it never affected training at all.

### Your checkpoint is fine

The checkpoint was trained with the old code. Since the training path produces mathematically identical results with the new code, you can:
- **Resume training** from the checkpoint — no drift, no incompatibility
- **Eval the checkpoint** — now works where it previously crashed

Nothing about the model weights, the optimizer state, the loss function, or the gradient flow changed. The fix purely adds the `OBS_IMAGES` read path that the eval-time batch requires.