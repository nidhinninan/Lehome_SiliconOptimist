## controlledSweep_v2 — conversation chain (verbatim)

Date captured: 2026-04-13

---

### USER

I want to brainstorm and finalize the sweep for the vision backbone as detailed by the documentation under the vision backbone folder in the artifact section of the repo. Okay.

### ASSISTANT

Brainstorming and locating the vision-backbone documentation in the repo. Reading the brainstorming skill and exploring the artifact folder.

### ASSISTANT

Reading the vision-backbone artifacts to extract sweep guidance and constraints.

### ASSISTANT

Checking for existing training configs or sweep scripts that should align with the finalized sweep.

### ASSISTANT

Here is a concise pass through context, options, and a **frozen sweep spec** you can treat as final unless you want to change one knob.

## What the artifacts already commit to

From `vision-backbone-research.md` and `vision_backbone_implementation_plan.md`:

- **Goal**: Compare vision representations for Diffusion Policy **without** full 150k-step runs.
- **Micro-sweep**: **10,000 training steps** per run (~1–1.5 h/run on an A4000-class GPU at batch 8, depending on backbone).
- **Readout**: **50-episode sim eval** (Isaac Lab) on the 10k checkpoint; interpret **early success rate** as a proxy for “already useful vision,” not final task performance.
- **Three runs** (A/B/C): ResNet-18 ImageNet baseline vs frozen DINOv2 ViT-S vs frozen CLIP ViT-B/16 — with the doc’s explicit caveat that **C is larger** (~86M) and tests “semantic at scale,” not param-matched fairness.

## Three ways to scope the sweep (trade-offs)

1. **Minimal A/B/C only (artifact-as-written)**
  Fastest (~3–4.5 h train + eval overhead). Clear story: *classic CNN pretrain vs dense spatial ViT vs semantic ViT*. Weakness: you cannot **isolate** “pretrain vs architecture” without an extra control.
2. **A/B/C + Run D: `resnet18` random init**
  Adds one more ~1.5 h train. Stronger science: directly tests the doc’s claim that ImageNet (or strong features) **skips “learning to see”** at 10k. Recommended if you have one extra GPU-night.
3. **Wider backbone grid (e.g. `resnet34`, DINOv2-Base)**
  More runs, messier interpretation (param count + speed + VRAM interact). Only worth it if the minimal sweep shows **no** separation and you need a second axis.

**Recommendation:** Lock in **Approach 1** as the **primary sweep**; treat **Approach 2 (Run D)** as an optional **confirmation** if A/B/C look ambiguous or you care about the pretrain ablation.

---

## Finalized sweep (primary — v1)


| Run   | Backbone          | Implementation                                                      | Pretrain | Trainable vision?     | Notes                                                                                     |
| ----- | ----------------- | ------------------------------------------------------------------- | -------- | --------------------- | ----------------------------------------------------------------------------------------- |
| **A** | `resnet18`        | Native LeRobot `policy.vision_backbone=resnet18`                    | ImageNet | Per default DP config | Set `**use_group_norm=false`** (per artifact; required for torchvision ImageNet weights). |
| **B** | DINOv2 **ViT-S**  | BYOP custom encoder (`Dinov2Model` / equivalent)                    | DINOv2   | **Frozen**            | ~21M params; “spatial / deformable object” hypothesis.                                    |
| **C** | CLIP **ViT-B/16** | BYOP (`CLIPVisionModel`), **CLS** (or fixed pooling) as in artifact | CLIP     | **Frozen**            | ~86M; not param-matched to A/B — intentional contrast “semantic / large.”                 |


**Held constant across A/B/C**

- **Steps**: `10_000` (single checkpoint for eval unless you explicitly add a mid-point).
- **Batch size**: **8** if it fits; if ViT-Base OOMs, doc already says drop to **4** and **note the step-time doubling** in results (do **not** silently mix batch sizes without labeling).
- **Optimizer / LR / horizon / cameras / dataset**: identical policy YAML (or equivalent) except backbone path and encoder implementation.
- **Eval**: **50 episodes**, same sim settings and (ideally) **fixed eval seed(s)** so runs are comparable.
- **Reporting**: step time, VRAM peak, success rate @10k, plus qualitative failure modes if you log them.

**Success criterion (what “wins” at 10k)**
Higher sim success and/or faster apparent improvement than baselines, interpreted as **representation quality for early policy learning**, not final challenge score.

---

## Optional add-on (v1.1)


| Run   | Backbone                    | Purpose                                                                                                      |
| ----- | --------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **D** | `resnet18`, **random init** | Control for **pretraining** vs architecture; validates the “skip learning to see” narrative if A ≫ D at 10k. |


---

## One decision from you

Do you want the **locked primary sweep to stay strictly A/B/C**, or **promote Run D into v1** so pretraining is always controlled? Reply with **“A/B/C only”** or **“A/B/C/D”** and I’ll treat that as the final sweep definition (and we can mirror it into the artifact markdown if you want that edit next).

### USER

A/B/C/D

### ASSISTANT

Here is the **final sweep** with **A/B/C/D** locked in.

## Final vision-backbone micro-sweep (v1)


| Run   | Backbone      | Stack                                 | Pretrain               | Vision trainable        | Role                            |
| ----- | ------------- | ------------------------------------- | ---------------------- | ----------------------- | ------------------------------- |
| **A** | `resnet18`    | Native DP, `vision_backbone=resnet18` | ImageNet               | Default policy behavior | Strong CNN baseline             |
| **B** | DINOv2 ViT-S  | BYOP encoder                          | DINOv2                 | **Frozen**              | Spatial / dense features (~21M) |
| **C** | CLIP ViT-B/16 | BYOP, CLS (or fixed pool)             | CLIP                   | **Frozen**              | Semantic / large (~86M)         |
| **D** | `resnet18`    | Same as A                             | **None / random init** | Default                 | Pretrain control vs A           |


**Shared protocol**

- **Train**: 10_000 steps, same data / policy config except where the row differs.
- **A & D**: `use_group_norm=false` for ImageNet-compatible ResNet when A uses pretrained weights; D uses random init (no ImageNet load).
- **Eval**: 50 Isaac Lab episodes on the 10k checkpoint; same sim settings and fixed eval seeds where possible.
- **Batch**: Prefer 8; if OOM on C, use 4 and **label** the run so timings are not mixed without annotation.

**How to read results**

- **A vs D**: isolates **pretraining** at fixed architecture.
- **A vs B vs C**: compares **representation families** at 10k (B is closer param-wise to “medium” encoders; C is deliberately heavier / semantic).

If you want this copied into `Artifacts/Vision-Backbone/` as the canonical “final sweep” section, say so and we can add it there in one edit.

### USER

name it controlledSweep_v2.

### ASSISTANT

Adding the official name `controlledSweep_v2` to the Vision-Backbone artifacts and aligning the sweep matrix with A/B/C/D.

### ASSISTANT

Updates are in place: the sweep is documented as `**controlledSweep_v2`** in both artifacts.

- `**Artifacts/Vision-Backbone/vision-backbone-research.md**` — Section **§4** now has a `**controlledSweep_v2`** subsection with the full A/B/C/D table, protocol (10k steps, 50-episode eval, batch 8/4), and how to read A vs D vs B/C.
- `**Artifacts/Vision-Backbone/vision_backbone_implementation_plan.md**` — **Phase 3** is retitled and tied to `**controlledSweep_v2`**, includes Run D, and asks that queued runs be tagged with that name in the automation script when you add it.