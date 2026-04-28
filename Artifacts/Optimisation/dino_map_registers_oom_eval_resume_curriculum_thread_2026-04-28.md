# DINOv2 MAP+Registers Long-Run Training: OOM, Eval Scheduling, Resume vs Fresh, and Augmentation-as-Curriculum

**Date**: 2026-04-28  
**Scope**: Technical synthesis of a troubleshooting and configuration-design thread around LeRobot **v0.4.3** BYOP training for `policy.type: dino_diffusion` with **`spatial_pooling: map`**, **`use_registers: true`**, and **`facebook/dinov2-with-registers-small`**.  
**Audience**: Engineers reproducing runs on the GPU VM (`lerobot_train_with_plugins.py`, workspace under `lehome_workspace/`).

This document preserves **failure modes**, **VRAM reasoning**, **eval semantics**, **config precedence on resume**, and **curriculum-style augmentation** without hiding assumptions.

---

## 1. Problem statements encountered in the thread

### 1.1 `FileExistsError` on startup (not OOM)

**Symptom**:

```text
FileExistsError: Output directory outputs/train/dp_top_short_dino_map_registers_150k already exists and resume is False.
```

**Mechanism**: LeRobot’s `TrainPipelineConfig.validate()` rejects overwriting an existing `output_dir` when `resume=False`. This protects `checkpoints/`, logs, and W&B run identity from accidental clobbering.

**Resolution patterns**:

- **Continue training** from existing `checkpoints/last`: set **`RESUME=true`** (script uses `train_config.json` under `last/pretrained_model/`).
- **New experiment**: change `--output_dir` / `OUTPUT=`, or remove/rename the old directory (only if you accept data loss).

**Why this mattered**: The operator initially interpreted a different failure as “OOM” because both appeared in adjacent sessions. Separating errors avoids mis-tuning VRAM when the fix is filesystem policy.

---

### 1.2 CUDA OOM mid-run (~9K–10K steps) during `AdamW` fused update

**Symptom** (representative traceback path):

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 74.00 MiB.
GPU 0 ... 15.57 GiB total ... ~46 MiB free
PyTorch allocated ~14.85 GiB; reserved but unallocated ~275 MiB
...
File ".../torch/optim/adam.py", line 666, in _multi_tensor_adam
    device_grads = torch._foreach_add(...)
```

**What this means technically**:

1. **The failure is not “new model parameters appeared at step 10k.”** Optimizer step runs every training iteration; the stack shows **`optimizer.step()`** inside `update_policy`, i.e. the standard training inner loop.

2. **The GPU was already saturated**: ~15.38 GiB of 15.57 GiB in use for this process. The allocator needed **74 MiB** contiguous memory and failed. Small tail allocations failing under near-full GPUs are classic signs of:
   - **Peak usage during a fused multi-tensor Adam kernel** (temporary buffers), and/or
   - **Allocator fragmentation** (adequate total free memory split into unusable chunks).

3. **PyTorch’s hint** (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) targets **CUDA caching allocator fragmentation**, not “train a smaller model.” It is appropriate as a mitigation when failures happen on **optimizer kernels** despite seemingly stable prior steps.

---

## 2. Baseline training characteristics (from logs printed by `lerobot_train.py`)

Representative resolved configuration excerpt:

- **`cfg.steps`**: operator ran **70k** in one session (CLI/env override), not necessarily the script default `150000`.
- **`dataset.num_frames`**: ~76k frames; **`dataset.num_episodes`**: 250.
- **`Effective batch size`**: **8 × 1 = 8** (no gradient accumulation in that run).
- **`num_learnable_params`**: ~778M learnable (with **frozen DINO backbone**, but MAP head + diffusion U-Net remain large).
- **`num_total_params`**: ~800M including frozen backbone weights loaded in VRAM.

**VRAM implication**: Even with AMP (`policy.use_amp: true` in YAML), **Adam states for hundreds of millions of trainable parameters** dominate steady-state footprint alongside activations for diffusion training and multi-camera batches.

---

## 3. Why the 10k MAP sweep could succeed while a long run OOM’d near 10k

### 3.1 Config parity vs behavioral parity

The **same YAML backbone** (`sweep_dino_map_registers.yaml`) implies similar **per-step training memory** if `batch_size`, horizon, observation structure, and AMP usage match.

However, **training length** interacts with **online evaluation scheduling** and other **periodic barriers** that do not exist in a 10k-only run.

### 3.2 Eval frequency coupling (central thesis)

Scripts used in the project set:

- **`eval_freq`** (top-level TrainPipelineConfig field in LeRobot v0.4.3; not nested under `eval.`)

For the **two-run DINO sweep** (`run_2run_dino_sweep.sh`):

- `STEPS=10000`
- `EVAL_FREQ=10000`

So the short run completes **roughly one training span** and may only hit eval in a way that aligns with **end-of-run bookkeeping**—it does **not** repeatedly interleave heavyweight eval with tens of thousands of subsequent training steps.

For the **long-run launcher** (`run_train_dino_dp_top_short_150k.sh`), defaults (as discussed in-thread) historically used **`EVAL_FREQ=10000`** with explicit:

- `--eval.n_episodes=...`
- `--eval.batch_size=...`

Thus a 70k–150k step job triggers **many** eval periods: 10k, 20k, 30k, …

**Hypothesis consistent with observation** (failed right after logged `step:9K`, next boundary 10k):

- The online eval stage (simulation + batched episodes + policy inference path) produces a **VRAM spike** and/or **fragmentation** that leaves the allocator unable to satisfy a **small** extra request during the next fused Adam step.

This is **not proven solely from stdout** without the eval logs, but the **temporal coincidence** with `eval_freq` and the **optimizer-kernel failure mode** strongly point to **eval/training interaction** rather than “mysteriously different instructions at step 10k.”

---

## 4. The Hugging Face warning: `dinov2_with_registers` vs `dinov2`

Observed once at model construction:

```text
You are using a model of type dinov2_with_registers to instantiate a model of type dinov2...
```

**Interpretation**:

- This is a **Transformers architecture / config-type compatibility warning**, not OOM causality.
- Training proceeded past model creation into optimization; therefore it is **orthogonal** to the CUDA allocator failure unless it forced an unintended module path (not indicated here).

---

## 5. Operational mitigation ladder (VRAM + eval scheduling)

Ordered from “least invasive to training semantics” to “more invasive.”

### 5.1 Allocator fragmentation mitigation (environment)

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Rationale**: When failures occur on **`_multi_tensor_adam`** with large reserved-but-unallocated pools, fragmentation mitigation is appropriate.

### 5.2 Reduce online eval pressure (configuration)

Tune **frequency**:

- Increase **`eval_freq`** so eval happens less often (fewer spikes).

Tune **eval workload** (LeRobot `EvalConfig` fields):

- Lower **`eval.n_episodes`**
- Lower **`eval.batch_size`**
- Keep **`eval.use_async_envs=false`** unless you know the async path is safe on your GPU budget (async can increase simultaneous environment load).

**Intent**: preserve a **learning signal** in W&B/console (not just train loss) while avoiding the **first-eval wall** at ~10k on a nearly full GPU.

### 5.3 Reduce training microbatch (configuration)

Lower **`batch_size`** in YAML/CLI.

**Effect**: Usually reduces activation memory per step (sometimes the dominant knob for diffusion training), at the cost of:

- Noisier gradients unless compensated (not always desired in BC), and/or
- Longer wall time to traverse the dataset.

### 5.4 Architecture / readout changes (new run, not resume-compatible without care)

Examples that change tensor shapes:

- Reduce **`policy.map_num_queries`** (MAP head output dimension scales with queries).
- Switch **`spatial_pooling`** away from `map` to `baseline` (major experiment change).

These are **not** “drop-in resume” compatible with a checkpoint trained at a different head geometry.

---

## 6. Repository-side defaults discussed/updated in-thread (for traceability)

The workspace tracking and scripts evolved during the thread; consult current files for ground truth:

- `lehome_workspace/run_train_dino_dp_top_short_150k.sh`  
  - Non-resume branch passes `--dataset.image_transforms.enable=false` explicitly.
  - Resume branch uses `checkpoints/last/pretrained_model/train_config.json`.

- `lehome_workspace/configs/sweep_dino_map_registers.yaml`  
  - May include explicit `eval_freq` / `eval:` stanza depending on iteration (check file).

When mirroring on the VM, use `lehome_workspace/vm_transfer_list.md` as the synchronization source of truth.

---

## 7. Image augmentations (`dataset.image_transforms.enable`): CPU vs GPU, and precedence

### 7.1 Where transforms run

In typical LeRobot datasets, enabling **image transforms** applies **torchvision-style** augmentation in the **data pipeline** (commonly CPU-side in dataloader workers) before tensors are consumed by the policy forward.

**Expected effects**:

- **CPU time / throughput**: `data_s` in logs often increases (more work per sample; more worker load). This can reduce steps/hour even if VRAM is unchanged.
- **GPU memory**: Usually **does not materially change tensor shapes** seen by the model (still 3×H×W feature entries in the schema), so **VRAM** is not the primary cost center compared to **wall-clock**.

**Caveat**: Any pipeline that accidentally materializes extra GPU tensors could change VRAM; the LeHome expectation here is CPU-side transforms with later GPU transfer.

### 7.2 `RESUME=false` branch: CLI override ordering in `run_train_dino_dp_top_short_150k.sh`

Non-resume invocation ends with:

1. `--dataset.image_transforms.enable=false`
2. `$EXTRA_TRAIN_ARGS` appended after

Therefore, if:

```bash
EXTRA_TRAIN_ARGS="--dataset.image_transforms.enable=true"
```

then the final token for that flag is typically **`true`**, assuming the argument parser resolves repeated flags as **last-wins** (the common draccus/argparse convention).

**Verification**: rely on LeRobot’s initial **config dump** line in logs (the `INFO ... {'batch_size': ..., 'dataset': {'image_transforms': {'enable': ...}}}` block). If it still reads `False`, your override did not apply as expected.

### 7.3 `RESUME=true` branch: checkpoint config authority

Resume path uses:

```bash
--config_path="$OUTPUT/checkpoints/last/pretrained_model/train_config.json"
--resume=true
```

**Key point**: In resume mode, the training config is **hydrated from the checkpoint’s saved JSON** first. Practical experience in this project (also noted in `lehome_change_log.md`) is that **some dataset-related toggles may not reliably follow CLI overrides on resume**.

**Reliable resume pattern for toggling augmentations**:

- Edit `train_config.json` under `checkpoints/last/pretrained_model/` so that:
  - `dataset.image_transforms.enable` becomes `true` (and any nested transform policy matches intent)

Then resume.

---

## 8. Curriculum interpretation: “no-aug → aug” via resume

### 8.1 What is preserved on resume

**Weights**: loaded from checkpoint.  
**Optimizer + scheduler state**: typically restored (Adam moments, step counters, etc., depending on LeRobot checkpoint contents).

### 8.2 What changes when enabling transforms mid-stream

The **data distribution** changes from deterministic demonstration pixels to an **augmented distribution** (randomized photometric / geometric jitter within configured ranges).

This is **curriculum-like** in the sense of **sequential training phases**:

- Phase 1: minimize BC loss on near-exact pixels (memorization risk in small datasets).
- Phase 2: force invariances that improve robustness for sim2real-like variability.

### 8.3 What is *not* a “pure” curriculum restart

Because optimizer state carries over, this is **not** equivalent to:

- resetting Adam states,
- resetting learning rate schedule from scratch,
- or starting a new W&B run identity (depending on configuration).

Those may or may not matter; if you want a “hard stage boundary,” you need an explicit training recipe (not covered here).

**Empirical expectation** (from related project notes): loss may **jump** when augmentation turns on; that can be desirable if it reflects increased task difficulty rather than breakage.

---

## 9. Decision matrix (quick reference)

| Goal | Mechanism | Starts from step 0? | Augmentation applies? |
|------|-----------|---------------------|------------------------|
| Continue same run | `RESUME=true` + checkpoint `train_config.json` unchanged | No | Same as phase 1 |
| Continue same weights but enable aug | `RESUME=true` + edit checkpoint `train_config.json` (`image_transforms.enable=true`) | No | Yes, from resume point onward |
| New experiment / clean slate | `RESUME=false`, new/changed `output_dir` | Yes | Controlled by YAML/CLI |

---

## 10. Verification checklist (avoid silent misconfiguration)

1. **OOM**: capture `nvidia-smi` around eval boundaries; correlate timestamps with eval logs.
2. **Transforms**: confirm printed config shows `dataset.image_transforms.enable: True/False` as intended.
3. **Resume**: confirm log states resume path and prints expected `cfg.steps`, `eval_freq`, and dataset root.
4. **Directory guard**: if `resume=False`, ensure `output_dir` absent or intentionally replaced.

---

## 11. References within this repository

- `lehome_workspace/run_train_dino_dp_top_short_150k.sh` — branching resume vs fresh; CLI flags for eval and transforms on fresh runs.
- `lehome_workspace/run_2run_dino_sweep.sh` — short sweep eval cadence (`STEPS=10000`, `EVAL_FREQ=10000`).
- `lehome_workspace/lerobot_policy_dino/src/lerobot_policy_dino/modeling_dino_diffusion.py` — frozen DINO backbone + MAP head path; resize to 224×224 inside encoder (VRAM shaped by patch tokens after resize, not raw 480×640).
- `Artifacts/Training_Update/training-DP_augmentation_resume_strategy_150k.md` — complementary note on augmentation motivation and loss jump expectations.

---

## 12. Epistemic status

Claims about **exact LeRobot internal ordering** (whether eval runs immediately before/after the training step at multiples of `eval_freq`) are **not derived from vendored source in this workspace** because the pip-installed `lerobot` package is not pinned as a git submodule here. The conclusions above are **engineered inference** from:

- stack trace location (`optimizer.step` during training),
- temporal proximity to `eval_freq` boundaries,
- differences between **short** vs **long** runs with periodic eval.

To upgrade this section from inference to proof, capture the training log lines around the first eval event on the VM and/or consult `lerobot==0.4.3` `lerobot_train.py` on the training machine.
