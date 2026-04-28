# Curriculum-style training, augmentation resume, and eval success — technical synthesis

**Date:** 2026-04-28  
**Purpose:** Preserve deep technical reasoning connecting (1) staged / curriculum-like training (no augmentation → resume with augmentation), (2) frozen DINO + diffusion policy training, (3) LeHome challenge **evaluation contract** and success metrics, and (4) the **concrete augmentation families and ranges** documented in project readouts. This artifact is written so a future session can reconstruct decisions from evidence and code paths, not only conclusions.

---

## 1. Executive summary

- **Staged augmentation** (train or warm-start without dataset-time image transforms, then **resume** with `dataset.image_transforms.enable=true`) is a **plausible and often beneficial** strategy for visuomotor imitation on cloth folding when transforms stay **photometric + mild geometric** and avoid **label-inconsistent** perturbations (e.g. aggressive random crops, flips without action symmetry).
- The augmentation set recorded in `Artifacts/Training_Update/training-DP_augmentation_resume_strategy_150k.md` is **not extreme** for this use case; the main tuning levers are **tightening saturation/sharpness**, optionally **softening affine**, and controlling **how many transforms compose per sample** (`max_num_transforms`, order) if exposed in config.
- **Hackathon eval success** is driven by **binary episode success** aggregated over episodes; the policy receives **only observations** (and prev action where applicable). There is **no garment type id** in the observation stream. Generalization pressure is therefore aligned with **robustness under visual and subvariant variation**—exactly what measured, validity-preserving augmentation targets.
- **Resume** in modern LeRobot-style stacks is **stateful** (weights + optimizer + RNG where saved), so flipping augmentation mid-run is a **deliberate distribution shift**; expect loss to move; judge **success rate and stability**, not loss alone.

---

## 2. Grounding: what “evaluation” means in this repo

### 2.1 Policy interface at eval time

The core loop passes a **NumPy observation dict** from the environment into `policy.select_action` and steps the simulator with the returned joint command. No garment label is injected into that dict by this path.

```104:137:lehome_workspace/lehome-challenge/scripts/utils/evaluation.py
    for i in range(args.num_episodes):
        # 1. Reset Environment & Policy
        env.reset()
        policy.reset()
        stabilize_garment_after_reset(env, args)

        # 2. Initial Observation (Numpy)
        object_initial_pose = env.get_all_pose() if args.save_datasets else None
        observation_dict = env._get_observations()
        ...
        for st in range(args.max_steps):
            ...
            # 3. Policy Inference (The core abstraction)
            # Input: Numpy Dict -> Output: Numpy Array
            action_np = policy.select_action(observation_dict)
            ...
            env.step(action)
            ...
            observation_dict = env._get_observations()
```

**Implication:** Any “router + four experts” submission must choose experts using **only** what is in `observation_dict` (plus internal memory the policy itself maintains across steps). Organizers may choose `--garment_type` for **which universe of garments** to load; that is **not** the same as exposing `garment_type_id` to the policy.

### 2.2 Success rate definition

Episode metrics include a boolean `success`; aggregate reporting uses the **mean of binary success flags** across episodes.

```155:178:lehome_workspace/lehome-challenge/scripts/utils/eval_utils.py
def calculate_and_print_metrics(metrics: List[Dict[str, Any]]) -> None:
    ...
    total_successes = [1 if m["success"] else 0 for m in metrics]
    ...
    success_rate = np.mean(total_successes)
    ...
    logger.info(f"Success Rate: {success_rate:.2%}")
```

Success within an episode is obtained from the environment (`env._get_success()` in the loop above), not from training loss.

### 2.3 Official eval entry and LeRobot policy path

The policy eval guide documents LeRobot evaluation with `pretrained_model`, `dataset_root` for normalization metadata, and notes that `--device cpu` drives sim placement while policy inference may still use CUDA.

```9:21:lehome_workspace/lehome-challenge/docs/policy_eval.md
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/act \
    --garment_type "top_long" \
    --dataset_root Datasets/example/top_long \
    --num_episodes 5 \
    --enable_cameras \
    --device cpu
```

**Connection to augmentation:** Training-time augmentations alter the **pixel distribution** seen during BC; eval uses **unaugmented** sim observations. Bridging that gap is the whole point of augmentation for sim success—not matching training pixels literally.

---

## 3. Documented augmentation suite (project readout)

The following table is **verbatim intent** from `Artifacts/Training_Update/training-DP_augmentation_resume_strategy_150k.md` (verified defaults in that note):

| Transform   | Range                         | Weight |
|------------|-------------------------------|--------|
| Brightness | [0.8, 1.2]                    | 1.0    |
| Contrast   | [0.8, 1.2]                    | 1.0    |
| Saturation | [0.5, 1.5]                    | 1.0    |
| Hue        | [-0.05, 0.05]                 | 1.0    |
| Sharpness  | [0.5, 1.5]                    | 1.0    |
| Affine     | deg ±5°, translation 5%     | 1.0    |

### 3.1 Are these “wrong families”?

**No**, for cloth visuomotor BC:

- **Brightness / contrast / hue** are standard photometric invariances; ranges here are moderate (hue especially small).
- **Affine (small rotation + small translation)** is a mild geometric stress test that still preserves **topology** of the scene (unlike crop or flip).
- **Missing from the list:** random crop, elastic warp, heavy color posterization, mixup/cutmix-style ops—those are the usual high-risk families for **fine geometric** tasks.

### 3.2 Are any ranges “extreme”?

**Mostly no.** The two worth tightening if you see brittle behavior right after resume:

1. **Saturation [0.5, 1.5]** — wider than brightness/contrast; can push fabric color into regimes rare in the camera distribution and slightly distort subtle shading on folds. **Suggested:** `[0.7, 1.3]` or `[0.8, 1.2]` for a conservative merged-dataset run.
2. **Sharpness [0.5, 1.5]** — strong sharpening can create halos and edge artifacts unlike real optics. **Suggested:** `[0.8, 1.2]`.

**Affine ±5°, trans 5%** is reasonable. If loss spikes hard or success drops after flip, **phase down** to ±3° and 2–3% translation before abandoning affine entirely.

### 3.3 Composition (not in the table but critical)

Even mild ops become harsh if **many stack per frame** with random order. If your YAML exposes `max_num_transforms` and `random_order`, a practical schedule is:

- **Phase A (first augmented segment):** `max_num_transforms: 2`, `random_order: false` (if supported).
- **Phase B (after metrics stabilize):** raise to 3 only if CPU and training remain stable.

---

## 4. Staged training: “BC → augmented BC” and why it maps to curriculum thinking

### 4.1 Operational pattern in this workspace

The repo includes a thin wrapper that resumes from a checkpoint directory and enables image transforms:

```1:20:resume_with_augmentations.sh
#!/bin/bash
# script to resume training with augmentations enabled
...
/data/lehome_workspace/lehome-challenge/.venv/bin/python \
    -m lerobot.scripts.lerobot_train \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --resume true \
    --dataset.image_transforms.enable true \
    --num_workers 4
```

**Note:** `num_workers` here is pinned to 4 in this script; on a large VM you may override for throughput, subject to `/dev/shm` and CPU contention (see optimization artifacts). The important point for *this* document is the **boolean flip** of `dataset.image_transforms.enable` at resume.

### 4.2 Why this is not “just superstition”

1. **Optimization landscape:** Early in training, the policy benefits from a **clean** correspondence between images and actions. Adding strong nuisance early can slow convergence or entrench wrong correlations.
2. **Distribution shift at resume:** When augmentation turns on, the effective training distribution changes; **loss may increase** while **eval success** improves or stabilizes under variation—that is not a contradiction (the readout in `training-DP_augmentation_resume_strategy_150k.md` already anticipates a loss bump).
3. **Literature pattern:** Manipulation-focused work often stresses that **naively random** transforms are often **invalid or irrelevant** relative to task structure; methods like corrective or neighborhood-bounded augmentation behave like a **curriculum in transform space**. That supports the intuition: **start faithful, then widen perturbations in a controlled way**—even if your implementation is simpler (off-the-shelf LeRobot transforms).

External anchors collected for reasoning (URLs only; not vendored into the repo):

- LeRobot IL / resume documentation: `https://huggingface.co/docs/lerobot/il_robots` and training example with `--resume=true` in the LeRobot repo examples.
- Discussion of invalid random transforms in manipulation (RSS): `https://www.roboticsproceedings.org/rss18/p031.pdf`
- RoboMimic resume semantics (optimizer state): `https://robomimic.github.io/docs/sources/introduction/getting_started.md.txt`

### 4.3 DeepWiki (LeHome) cross-check (organizer-level framing)

DeepWiki summary for `lehome-official/lehome-challenge` aligns with the code above: observations include state and RGB streams; **garment type id is not passed as an observation**; `--garment_type` is a **script** selector for which assets/episodes to evaluate; success rate is **mean of episode binary successes**. That matches implementing staged augmentation purely in **training data**, without touching eval observation contract.

---

## 5. Interaction with **frozen DINO** + **diffusion policy**

### 5.1 Frozen backbone

When the vision tower is frozen (or mostly frozen), the policy still adapts through the **head**, conditioning MLPs, and diffusion action head. Augmentation then primarily:

- Forces the **downstream** pathway to rely on features that are **stable under nuisance**, not on narrow intensity shortcuts.
- Reduces **co-adaptation** between a trainable encoder and decoder that can overfit pixels together.

### 5.2 Diffusion over action chunks

Diffusion policies model **conditional distributions over action sequences**. Augmentation changes **image conditioning** while actions stay tied to **original** demos (dataset-time transforms in LeRobot do not relabel actions). Therefore:

- **Photometric** ops are usually **safe** (same action remains correct).
- **Affine on image** without equivalent **action-frame** change is only safe when **small enough** that the expert action remains approximately valid in the robot frame (your ±5° / 5% regime is in that spirit).

---

## 6. Connections to merged four-garment training and wall-clock tradeoffs

From prior project reasoning (merged dataset ~266k frames, batch 8 “epoch” mental model):

- Step count maps to **approximate dataset passes**; augmentation increases **CPU** load per step; **eval** frequency and `n_episodes` dominate wall time if set to LeRobot defaults.
- **Staged augmentation** on a merged run remains attractive because you still want a **clean convergence phase** across four visual families, then a **robustness phase**.

There is no single “correct” fraction of steps for phase 1 vs phase 2; treat it as a **hypothesis** validated by **eval success curves** per garment category (external eval sweeps), not training loss alone.

---

## 7. Pitfalls checklist (actionable)

| Risk | Mitigation |
|------|------------|
| Aggressive **crop** removes cloth boundary | Do not add random crop for folding unless validated |
| **Horizontal flip** breaks left/right semantics | Avoid unless task + action symmetry proven |
| Too many stacked transforms | Limit `max_num_transforms`; staged increase |
| **Resume + same LR** after large shift | Consider shorter eval windows after flip; optionally lower LR for first N steps (experiment) |
| Eval too heavy during train | Explicit `eval:` block; light in-train eval + heavy milestone eval |

---

## 8. What this artifact does **not** claim

- It does **not** guarantee that staged augmentation beats **joint training** with mild aug from step 0; that is an empirical question on your merged run.
- It does **not** substitute for measuring **category-wise** success if you care about balanced performance across top/pant long/short.

---

## 9. Conversation context + research inputs

**Local sources:**

- `Artifacts/Training_Update/training-DP_augmentation_resume_strategy_150k.md` — augmentation table, resume narrative, expected loss bump.
- `resume_with_augmentations.sh` — concrete CLI for resume + enable transforms.
- `lehome_workspace/lehome-challenge/docs/policy_eval.md` — eval command shape, `pretrained_model`, dataset_root, device note.
- `lehome_workspace/lehome-challenge/scripts/utils/evaluation.py` — observation-only `select_action` loop.
- `lehome_workspace/lehome-challenge/scripts/utils/eval_utils.py` — success rate aggregation.

**External / tool-gathered (used for reasoning, not copied verbatim):**

- DeepWiki `ask_question` on `lehome-official/lehome-challenge` — observation contents, no garment id to policy, `--garment_type` selection role, success metric framing.
- Exa-backed subagent notes — LeRobot resume with `train_config.json`, checkpoint contents (optimizer/RNG), RoboMimic resume semantics, RSS discussion of validity of random transforms, Diffusion Policy workspace checkpointing.
- RivalSearch `social_search` on a broad “staged augmentation” query returned **noise** (non-robotics “staged” articles); robotics signal came from **targeted papers** and IL docs instead.

---

## 10. Suggested follow-ups (optional)

1. Log **eval success** (and optionally per-garment scripts) at fixed intervals **before** and **after** augmentation flip on a short merged run to quantify benefit.
2. If saturation/sharpness are tightened, record the exact YAML diff in the next training readout for reproducibility.

---

*End of artifact.*
