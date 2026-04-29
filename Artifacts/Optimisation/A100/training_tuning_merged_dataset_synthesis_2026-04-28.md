# A100 Training Optimization: Merged Dataset & Resource Tuning

**Date:** 2026-04-28  
**Project:** LeHome Challenge  
**Focus:** DINOv2 + Diffusion Policy Tuning on Merged Garment Datasets (A100)

---

## 1. Executive Summary
Training was transitioned from single-garment datasets (batch 8) to a merged four-type dataset (266k frames) on an A100. Key optimizations include re-calculating step budgets to avoid "overkill" epochs, balancing CPU data loading (`data_s`) against GPU update times (`updt_s`), and resolving system memory instability (OOM) through refined worker counts. A two-phase curriculum (Clean BC -> Augmented BC) was established to maximize convergence and generalization.

---

## 2. Mathematical Grounding: Steps vs. Epochs

The shift from `batch_size: 8` to `32` drastically changed the data exposure per training step.

- **Dataset Size:** 265,798 frames.
- **Effective Batch Size:** 32.
- **Iteration Math:** 1 Epoch ≈ 8,306 steps ($265,798 / 32$).

### Overkill Analysis
Previous recommendations of 400,000 steps were modeled on a 12-pass exposure at batch size 8. Applying the same step count to batch size 32 results in **~48 epochs**, which exceeds the point of diminishing returns for this challenge and consumes ~40 hours of A100 wall-clock time.

**Revised Target:**
- **Total Steps:** 125,000 (≈ 15 Epochs).
- **Rationale:** Hits the "Goldilocks" zone of 12-15 passes over the merged dataset while reducing wall-clock time to ~12.5 hours.

---

## 3. Hardware Equilibrium: The `data_s` vs `updt_s` Dance

Efficient training on high-end compute (A100) requires the CPU data pipeline to hand off batches exactly as the GPU finishes a backward pass.

### Observations from the Logs
- **GPU Update Time (`updt_s`):** Stable at ~0.150s - 0.180s for 32 samples.
- **CPU Data Time (`data_s`):** Fluctuated based on worker counts and OS caching.

| Configuration | `updt_s` | `data_s` | Observation |
| :--- | :--- | :--- | :--- |
| 32 Batch / 12 Workers | 0.196s | 0.166s | Safe; CPU finishes slightly ahead of GPU. |
| 32 Batch / 16 Workers | 0.182s | 0.179s | **Ideal Equilibrium.** Perfect hardware lockstep. |
| 32 Batch / 8 Workers | 0.180s | 0.180s | Stable; minimal context switching overhead. |

### The "Cache Saturation" Effect
During long runs, `data_s` was observed to increase from **0.179s to 0.215s** as the OS Page Cache filled up, forcing workers to fetch video chunks directly from NVMe storage. Total step time remained stable, and the shift was identified as a normal hardware transition rather than a memory leak or throttling.

---

## 4. Stability & Memory (The OOM Killer Incident)

A fatal error (`RuntimeError: DataLoader worker exited unexpectedly`) occurred when running **32 Batch / 16 Workers**. 

**Technical Diagnosis:**
- Each worker holds its own buffer in System RAM / Shared Memory (`/dev/shm`).
- 16 workers × 32 samples × 3 high-res RGB streams exceeded the VM's memory limits.
- The Linux OOM Killer terminated a background process, leading to a crash.

**Mitigation:**
- Reduce `num_workers` to **8** for high-load segments or ensure `/dev/shm` is remounted to at least **16GB-24GB** to handle the tensor traffic.

---

## 5. Curriculum Training Strategy

To handle the complexity of four distinct garment types (merged dataset), a staged augmentation approach was adopted.

### Phase 1: Clean Convergence (Steps 0 - 75,000)
- **Augmentation:** Disabled.
- **Goal:** Establish clean correspondence between visuals and actions for all 4 types (≈ 9 Epochs).
- **Logic:** Early noise can entrench wrong correlations before the base physics are mastered.

### Phase 2: Robustness/Generalization (Steps 75,000 - 125,000)
- **Augmentation:** Enabled (`image_transforms: true`).
- **Goal:** Force the policy to build photometric and geometric invariants (≈ 6 Epochs).
- **Expectation:** Training loss will "bump" up; this is a healthy indicator of the model learning to ignore visual nuisances.

---

## 6. Final Optimal Configuration Profile

```markdown
# Profile: A100 Merged DINO-MAP8
Steps: 125,000
Batch Size: 32
Num Workers: 8-16 (Dependent on /dev/shm)
AMP: True
Video Backend: torchcodec
Curriculum:
  - 0-75k: No Augmentation
  - 75k-125k: Photometric + Mild Geometric Augs
```

### Performance Targets
- **Step Speed:** ~0.35s / step.
- **Throughput:** ~6 mins / 1,000 steps.
- **Total Wall-Clock:** ~12.5 Hours.

---
*End of Synthesis.*
