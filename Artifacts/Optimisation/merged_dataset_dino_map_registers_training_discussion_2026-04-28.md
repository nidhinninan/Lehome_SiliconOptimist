# Merged four-garment dataset + DINO MAP+registers training — discussion artifact

**Date:** 2026-04-28  
**Scope:** Reasoning and technical notes from a Cursor conversation chain covering (1) how many training steps to run on the LeHome **merged** demonstration dataset, (2) augmentation and generalization, (3) `num_workers`, AMP, video backend, and OOM risks, (4) checkpoint / eval cadence for long runs, (5) explicit `eval` blocks vs LeRobot defaults, and (6) external research anchors (DeepWiki, Hugging Face docs/MCP, Exa, RivalSearch, upstream LeRobot code).

This document is written so a future reader can **reconstruct decisions** from reasoning and cited facts, not only final parameter values.

---

## 1. Problem statement

**Goal:** Train `policy.type: dino_diffusion` with **DINOv2 with registers + MAP spatial pooling** (`sweep_dino_map_registers.yaml` profile) on the **larger merged dataset** containing **four garment categories** (tops/pants, long/short), instead of a single category such as `top_short_merged`.

**Constraints and preferences discussed:**

- Keep alignment with prior sweep conclusions: **`policy.use_amp: true`**, **`batch_size: 8`**, **`video_backend: torchcodec`** (with **`pyav` fallback** if decode fails), **`num_workers: 12`** (not 16), **transforms off** for baseline / speed-fair sweeps; augmentation considered for a **later phase** or **resume**.
- Long runs need **sensible `save_freq` / `eval_freq`** without making evaluation dominate wall-clock time.
- Need clarity on **LeRobot’s default `eval` settings** when YAML does not define an `eval:` block.

---

## 2. Dataset facts (project artifacts)

Source: `Artifacts/Dataset-Viz_analysis/dataset_details.md` and related notes.

| Garment category | Sub-folder (example naming) | Episodes | Total frames |
|------------------|-----------------------------|----------|--------------|
| Top Long | `top_long_merged` | 250 | 83,068 |
| Top Short | `top_short_merged` | 250 | 76,066 |
| Pant Long | `pant_long_merged` | 250 | 65,909 |
| Pant Short | `pant_short_merged` | 250 | 40,755 |
| **Total** | | **1,000** | **265,798** |

**Implications for “steps” reasoning:**

- LeRobot training is primarily **step-based** (optimizer updates), not epoch-based. A useful mental model is **effective dataset passes** via approximate **frame-level exposure**:
  - With **`batch_size: 8`**, one training step consumes **8 frames** (with the usual caveats: sequence sampling, padding, `drop_n_last_frames`, etc.).
  - Approximate **frames per step** ≈ 8 ⇒ approximate **steps per full pass** over all frames ≈ `265,798 / 8 ≈ 33,225` steps.

**Worked equivalence (approximate):**

| Total steps | Approx. full-dataset passes (merged, ~266k frames, batch 8) |
|------------|----------------------------------------------------------------|
| 150,000 | ~4.5 |
| 200,000 | ~6.0 |
| 300,000 | ~9.0 |
| 400,000 | ~12.0 |
| 500,000 | ~15.0 |
| 600,000 | ~18.1 |

**Why this matters:** A historical **150k** run on a **single** category (e.g. `top_short` ~76k frames) implies a **much higher** per-frame revisit rate than **150k** on the **merged** set. You cannot compare “150k” across those two settings as the same amount of “data exposure.”

**Hugging Face:** The merged dataset is referenced as `lehome/dataset_challenge_merged` (Hub card; license `apache-2.0` per HF MCP `hub_repo_details` at the time of the conversation).

**Challenge / eval reality:** Docs in the same artifact chain note that evaluation can load **many subvariants** (meshes/textures/physics) **without** exposing a garment **label** to the policy. That pushes toward either **one unified policy** that is visually robust, **or** a **router / classifier** (out of scope for this note unless implemented).

---

## 3. How many steps for a “strong” merged model?

**There is no universal step count** that guarantees sim success rate without task metrics. The discussion converged on **practical bands** using (a) approximate dataset passes above, (b) BC + optional augmented BC practice from project notes, and (c) diminishing returns past the point where validation loss plateaus and eval metrics stop improving.

**Recommended starting plan (offline BC on merged data):**

- **Target total:** **`400,000`** steps as a serious default for a **single** merged run.
- **Minimum worth trying before abandoning architecture/hparams:** on the order of **`300,000`** steps (if metrics are still moving in the right direction).
- **Stretch:** **`500,000`** steps if eval/loss still improves and budget allows.
- **Avoid by default:** **`600,000+`** unless there is clear evidence of continued improvement (time cost + overfitting risk).

**Intuition:** ~**12 passes** over the merged frame pool (at 400k steps) is a reasonable “serious” exposure for a unified policy; ~**4.5 passes** (150k on merged) is closer to “short” in **relative** terms vs the old single-category 150k plan.

---

## 4. Augmentation: do you need it, and for how long?

**Project-specific prior:** `Artifacts/Training_Update/training-DP_augmentation_resume_strategy_150k.md` describes transitioning from pure BC to **augmented BC** by **resuming** from a strong checkpoint and enabling **LeRobot image transforms**, with an **expected loss bump** that is treated as healthy (the model is no longer matching trivial pixel details).

**Why augmentation helps here:** The merged policy must handle **four** folding families and **subvariant** visual/physics diversity. Augmentation targets **photometric** and mild **geometric** invariances (lighting, color, small affine jitter), not “new cloth physics.”

**Suggested schedule (two phases):**

1. **Base fit (no augmentation):** ~**150k–200k** steps with `dataset.image_transforms.enable: false` (or CLI override) so the policy learns clean dynamics and stable features first.
2. **Generalization push (augmentation on):** continue to **~400k total** (i.e. **~200k** augmented steps if phase 1 is 200k), with transforms enabled.

**LeRobot defaults (from upstream `lerobot==0.4.3` `ImageTransformsConfig` / docs):** transforms are **disabled by default**; when enabled, typical defaults include **up to 3 transforms per frame** (`max_num_transforms: 3`) sampled from brightness/contrast/saturation/hue/sharpness/affine-style ops. **Start mild** if CPU load spikes (e.g. `max_num_transforms: 2` first).

**Pitfall:** Aggressive **random crop** or wrong **crop size** can break visuomotor policies. A public LeRobot discussion (issue thread) highlighted “visually blind” or collapsed behavior that was partly attributed to **overly small crops** — relevant as a warning for any **policy-level** `crop_*` settings, not dataset transforms.

---

## 5. System knobs: AMP, workers, video backend, OOM

**`policy.use_amp: true`**

- Motivation: MAP+registers increases **conditioning size** and memory traffic; AMP often reduces **activation footprint** and can improve throughput on modern NVIDIA GPUs.
- Pitfall (from project notes): AMP can speed the **GPU** while the run is still **CPU decode / dataloader** bound; always look at **`data_s` vs `updt_s`** in logs, not GPU TFLOPS alone.

**`num_workers: 12` (prefer over 16 in the reported sweep)**

- Empirical pattern from the user’s 3-run micro-sweep: **`data_s` ~ flat** (~0.022s) when moving from 12 → 16 workers, while **`updt_s` / wall** could look **worse** and **host RAM / disk pressure** increased. So **16** is not a free win.
- Rule: **benchmark** 8 / 12 / 16 on the **same** machine; respect **`/dev/shm`** size (project scripts remount to 2G when possible; else `num_workers=0` fallback).

**Thread caps (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` = 1)**

- Rationale: each DataLoader worker is a process; uncontrolled BLAS threading inside many workers can **oversubscribe** CPU and **hurt** decode latency.

**`dataset.video_backend: torchcodec`**

- Preferred when FFmpeg + `torchcodec` are healthy on the VM (faster decode).
- Operational fallback: **`pyav`** if decode fails (known class of failures in project notes).

**OOM note (MAP):** MAP increases per-image conditioning vs CLS pooling; **do not** raise `batch_size` first on near-full VRAM — prefer AMP, stable pipeline, then cautious batch probes.

---

## 6. Training-time evaluation vs challenge `scripts/eval.py`

**Important distinction:**

- **`lerobot_train` “eval”** (driven by `eval_freq` + `EvalConfig`) is the **framework’s** periodic evaluation path (environment / metrics as implemented in that LeRobot version).
- **`lehome_workspace/lehome-challenge/scripts/eval.py`** is a **separate** Isaac Lab–based evaluation entry for the challenge; it is not automatically what `lerobot_train` calls unless the training stack is wired that way in your installed LeRobot.

**LeRobot v0.4.3 default `EvalConfig` (authoritative):** in `src/lerobot/configs/default.py`:

- `n_episodes: 50`
- `batch_size: 50` (vectorized env batch)
- `use_async_envs: false`

**If your YAML does not set `eval:`**, you inherit those defaults — which can be **very expensive** if `eval_freq` is frequent on a 400k run.

**Transcript / repo note:** `run_2run_dino_sweep.sh` passes **`--eval_freq`** but (in the versions discussed) did **not** pass `--eval.n_episodes` / `--eval.batch_size`, so the sweep used **defaults (50/50/false)**.

---

## 7. “Light but good” eval cadence (practical recommendation)

**Goal:** Catch failures early (divergence, sudden metric collapse) without making eval **dominate** wall time.

**Suggested training-time eval policy:**

- **Frequent light eval:** `eval.n_episodes: 10`, `eval.batch_size: 10` (keeps the default LeRobot **validity constraint** `batch_size <= n_episodes` and avoids instantiating many unused envs).
- **Cadence:** `eval_freq: 10000` (≈40 evals / 400k steps).
- **Milestone heavy eval:** at each **50k** checkpoint (or end of run), run a **larger** eval **manually** or by temporarily increasing `n_episodes` to **32–50** for statistically stronger success estimates.

**Why not 50 every 10k on a 400k run:** cost scales roughly linearly with episodes; **50 × frequent** can add **many hours** depending on sim throughput.

---

# Config sections (scenarios)

Use these as **templates**. Adjust **paths** to your VM layout.

---

## A. Base merged run — 400k steps, saves every 50k, light periodic eval

**Scenario:** Single long offline training job on the **merged** local dataset root; **no augmentation**; MAP+registers; AMP on; workers 12; torchcodec.

```yaml
dataset:
  repo_id: repo_dp
  root: /root/data/lehome_workspace/lehome-challenge/Datasets/example/dataset_challenge_merged
  video_backend: torchcodec
  image_transforms:
    enable: false

policy:
  type: dino_diffusion
  vision_backbone: facebook/dinov2-with-registers-small
  spatial_pooling: map
  map_num_queries: 8
  use_registers: true
  num_register_tokens: 4
  device: cuda
  push_to_hub: false
  use_amp: true
  crop_shape: null
  crop_is_random: false
  input_features:
    observation.state: { type: STATE, shape: [12] }
    observation.images.top_rgb: { type: VISUAL, shape: [3, 480, 640] }
    observation.images.left_rgb: { type: VISUAL, shape: [3, 480, 640] }
    observation.images.right_rgb: { type: VISUAL, shape: [3, 480, 640] }
  output_features:
    action: { type: ACTION, shape: [12] }

eval:
  n_episodes: 10
  batch_size: 10
  use_async_envs: false

output_dir: /root/data/lehome_workspace/outputs/train/dp_merged_dino_map_registers_400k
job_name: DINOv2_MAP_Registers_merged_400k

batch_size: 8
num_workers: 12
steps: 400000
save_freq: 50000
eval_freq: 10000
log_freq: 1000

wandb:
  enable: true
  project: lehome_challenge
```

**CLI / runtime overrides commonly paired:**

- If torchcodec fails: `--dataset.video_backend=pyav` once, fix FFmpeg, revert.
- Thread caps in shell: `export OMP_NUM_THREADS=1` (and MKL/OPENBLAS) before train.

---

## B. Same as A, but `eval` aligned 1:1 with saves (less frequent eval)

**Scenario:** You want **eval only when you checkpoint** (8 evals on a 400k run), minimizing eval overhead.

Set:

```yaml
eval_freq: 50000
save_freq: 50000
```

Keep **`eval.n_episodes`** at **10** or increase to **20–32** if each eval is infrequent enough to afford heavier measurement.

**Trade-off:** You will notice problems **later** (every 50k) → higher risk of **wasted** training time if a run explodes at step 5k.

---

## C. Milestone-heavy eval in YAML (50 episodes) — use sparingly

**Scenario:** You **intentionally** want the **default LeRobot** “full” eval cost **every** `eval_freq` tick (usually **not** recommended for 400k).

```yaml
eval:
  n_episodes: 50
  batch_size: 50
  use_async_envs: false
eval_freq: 50000
```

**Context:** Matches **LeRobot v0.4.3** defaults. Good for **rare** eval; expensive if `eval_freq` is small.

---

## D. Augmentation phase (resume or second leg)

**Scenario:** After a strong no-aug checkpoint, enable transforms for **generalization**; expect **loss bump** (project note).

**Typical override:**

```yaml
dataset:
  image_transforms:
    enable: true
    # optional: max_num_transforms: 2  # milder CPU load than 3
```

**Often implemented** as **resume** from `checkpoints/last` with CLI `--dataset.image_transforms.enable=true` (exact flag style depends on your draccus CLI; project scripts use `--dataset.image_transforms.enable=false` for sweeps).

**Do not** assume augmentation changes **network shape**; it changes **data pipeline statistics**.

---

## E. Match repository sweep file `sweep_dino_map_registers.yaml` (10k micro-sweep / baseline)

**Scenario:** Short architecture or speed experiments; **no** `eval:` block in-repo — inherits **LeRobot defaults** unless overridden on CLI.

**Current repo fields (abridged):** `lehome_workspace/configs/sweep_dino_map_registers.yaml` — `steps: 10000`, `save_freq: 10000`, `log_freq: 500`, `policy.use_amp: true`, `num_workers: 12`, `dataset.image_transforms.enable: false`, `video_backend: torchcodec`.

**Note:** Production long runs should still add **`eval:`** or explicit **`--eval.*`** if you do not want **50/50** evals.

---

## F. Shell launcher pattern (from `run_train_dino_dp_top_short_150k.sh` family)

**Scenario:** You keep YAML mostly stable and pass **steps / output / dataset root / eval_freq** via environment variables.

Typical environment variables:

- `STEPS=400000`
- `SAVE_FREQ=50000`
- `EVAL_FREQ=10000`
- `DATASET=Datasets/example/dataset_challenge_merged`
- `OUTPUT=outputs/train/dp_merged_dino_map_registers_400k`

**Eval sub-keys** are set in YAML or as `--eval.n_episodes=10` style overrides (exact accepted forms depend on the installed LeRobot CLI parser).

---

# Conversation Context + Research

This section records **context gathered at the start** of the conversation chain (tools: DeepWiki MCP, Hugging Face MCP, Exa `web_search_exa` / `web_fetch_exa`, RivalSearchMCP `web_search` / `social_search` / `scientific_research`), plus **local** project artifacts. It is **not** a claim that every external hit was high-signal; some social searches returned **empty** results for niche queries.

## LeRobot / training mechanics (DeepWiki + HF docs)

- **Steps vs epochs:** LeRobot’s training loop is driven by **`steps`**, not epoch count; `TrainPipelineConfig` carries `steps` and `eval_freq` as **top-level** fields; **`eval` is a nested `EvalConfig`**. (DeepWiki `ask_question` on `huggingface/lerobot`.)
- **Image transforms:** Configurable through dataset-side **`ImageTransformsConfig`**, applied during training data loading; distinct from some **policy-side** image preprocessing (resize/crop) in `DiffusionConfig` in general LeRobot.
- **DataLoader:** `num_workers` is passed to PyTorch `DataLoader`; with `num_workers>0`, **`prefetch_factor=2`** is a common default; more workers can help up to a point, then **memory and contention** can dominate.
- **HF dataset card:** `lehome/dataset_challenge_merged` — high-level metadata (license, downloads) via `hub_repo_details`.

## Upstream code anchor (fetched as raw `v0.4.3`)

- **`EvalConfig` defaults** in `lerobot v0.4.3` `src/lerobot/configs/default.py`: `n_episodes=50`, `batch_size=50`, `use_async_envs=false`, with a guardrail error if `batch_size > n_episodes`.
- **`TrainPipelineConfig`** in `src/lerobot/configs/train.py` includes `eval: EvalConfig` and default `eval_freq` (value depends on version; user scripts override with `--eval_freq`).

## Exa / web documentation (selected themes)

- **Diffusion policy training tutorial (robomimic):** horizon knobs (`observation_horizon`, `prediction_horizon`, `action_horizon`) and template JSON patterns — useful as **general** DP context, not LeHome-specific.
- **LeRobot GitHub issue (DP training symptoms):** report of “blind” behavior at low steps and degradation / jitter at higher steps; discussion includes **image crop size** and dataset mismatch themes — useful as a **caution** about **bad preprocessing** being misread as “too many steps.”
- **Hugging Face LeRobot dataset v3 docs (via Exa search results):** documents `ImageTransformsConfig` examples (`enable`, `max_num_transforms`, `random_order`, per-transform weights and kwargs).

## RivalSearchMCP

- **Web search** returned many navigation/pointer results (LeRobot GitHub, docs mirrors, community tutorials) without a single dominant numeric answer for “best steps.”
- **Social search** often returned **0** results for tightly scoped strings (common for niche queries).

## ArXiv / academic search (broad)

- `scientific_research` with a broad imitation-learning query returns many **general** IL papers; not a substitute for **task-specific** success metrics on LeHome.

## Project artifacts explicitly used

- `Artifacts/Dataset-Viz_analysis/dataset_details.md` — frame and episode counts, 3× RGB streams, DP normalization notes.
- `Artifacts/folding_cloth_research_v2.md` — strategy landscape (unified model vs router, VLA references, subvariant randomization challenge).
- `Artifacts/Training_Update/training-DP_augmentation_resume_strategy_150k.md` — resume with augmentations, loss bump narrative.
- `LEARNING-2.md`, `Learning.md` — BYOP pitfalls (optim params, resize), dataloader/AMP interactions, `num_workers` heuristics, eval script notes.
- `Artifacts/Optimisation/sweep/10k_sweep_analysis.md` — DINO vs ResNet wall-clock and loss at 10k.
- `Artifacts/Optimisation/dino_map_registers_utilization_wall_clock_synthesis_2026-04-27.md` — MAP VRAM, `updt_s` vs `data_s`, worker/thread guidance.
- `Artifacts/Optimisation/MAP_head_choice.md` — `map_num_queries: 8` rationale for multi-garment unified policy.

## Agent transcript file referenced by the user

- Transcript id `7243c188-126c-462a-8629-0bce38cb790f` (local Cursor agent-transcripts): contained the **3-run sweep table** (AMP, workers 4/12/16) and the later **grounded** note that **without** an `eval:` block, **LeRobot defaults** apply — matching v0.4.3 `EvalConfig`.

---

## 8. Light eval stage (MAP+registers): grounded technical response (2026-04-28)

This subsection records a **self-contained** analysis of the **very light** training-time eval added for the heavy **DINOv2-with-registers-small + MAP** (`dino_diffusion`) profile—specifically whether **small `eval.n_episodes` / `eval.batch_size`** makes results misleading, whether they still **represent** training progress, and whether the eval is **wasted work**. Facts below are tied to repo files, `lehome_change_log.md`, and external MCP research (DeepWiki `huggingface/lerobot`, RivalSearch topic research); interpretive bullets are labeled as such.

### 8.1 Facts from this repository (configs, launcher, changelog, notes)

**`lehome_workspace/configs/sweep_dino_map_registers.yaml`** (eval block and training eval cadence):

- Training: `batch_size: 8`, `num_workers: 12`, `policy.use_amp: true`, MAP+registers + `spatial_pooling: map`, three 480×640 RGB streams.
- Eval (explicit `eval:` section, chosen for GPU-friendly monitoring): `eval_freq: 20000`, `eval.n_episodes: 4`, `eval.batch_size: 2`, `eval.use_async_envs: false`. Comments in-file state intent: monitor learning while avoiding the **10k-step OOM** boundary on this model class; **full** high-episode eval deferred to `parallel_eval.sh`.

**`lehome_workspace/run_train_dino_dp_top_short_150k.sh`** mirrors the same philosophy via defaults and CLI:

- Default `EVAL_FREQ="${EVAL_FREQ:-20000}"`, `EVAL_N_EPISODES="${EVAL_N_EPISODES:-4}"`, `EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"`, `EVAL_USE_ASYNC_ENVS="${EVAL_USE_ASYNC_ENVS:-false}"`.
- Passes `--eval.n_episodes`, `--eval.batch_size`, `--eval.use_async_envs` on fresh runs; uses `configs/sweep_dino_map_registers.yaml` as `--config_path` with draccus-style overrides for steps, dataset root, output, etc.

**`lehome_change_log.md` (2026-04-28 12:30:00 UTC entry):**

- Motivation: monitor W&B/console **without** repeating OOM at the first **`eval_freq=10000`** boundary on DINOv2 + MAP head.
- Change: **`eval_freq` → 20000**, eval reduced to **`n_episodes=4`, `batch_size=2`**; explicit `eval:` in YAML for documentation.

**`LEARNING-3.md`:** user note that OOM led to reducing eval state; goal is **“vague understanding”** of training progress—aligned with low-episode eval, not definitive success rates.

**Contrast: `parallel_eval.sh` (post-training, challenge `scripts.eval`):** launches **5 episodes per garment type** (×4 garment types), headless CPU, **separate** from `lerobot_train` in-loop eval—i.e. the project already treats **thorough** eval as **offline** from the light training-time path.

### 8.2 LeRobot evaluation mechanics (DeepWiki MCP, `repoName=huggingface/lerobot`)

Grounded summary from `ask_question` on upstream LeRobot (training + eval pipeline):

- **`eval.n_episodes`:** total rollouts for eval; passed into `eval_policy_all` when `eval_freq > 0` and an environment is configured.
- **`eval.batch_size`:** width of **vectorized / parallel** environments during eval (`make_env`); if `0`, can auto-tune (implementation-dependent), bounded by `n_episodes`.
- **`eval_freq`:** training steps between eval calls.
- **Flow:** `eval_policy_all` → rollout loop → policy in **`eval()`** mode → `select_action` → `env.step()`; metrics such as **`pc_success`** depend on the env returning **`info["is_success"]`** (or equivalent) per step/episode.
- **Small `n_episodes`:** explicitly characterized as producing **unreliable / high-variance** success-rate estimates in stochastic settings; **larger** episode counts (e.g. benchmark docs cite **10+** per task for stronger estimates; **1** for smoke tests only) are preferred when the goal is **statistical** confidence, not a quick pulse.

### 8.3 Broader research (RivalSearchMCP `research_topic`)

Two topic runs (small-episode variance in IL/robotics training; best practices for in-training eval with LeRobot / diffusion / ACT) returned multi-source aggregates (DeepWiki pages, HF docs, blog posts, example commands) with **mixed confidence** scores. Consistent themes:

- **High variance** in success rate when episode count is small—especially manipulation / contact-rich sims (garment-like tasks fit this pattern).
- **Training-time** workflows often use **sparse** `eval_freq` and **moderate** in-loop episode counts, reserving **large** `n_episodes` (and/or multi-seed runs) for **standalone** `lerobot-eval` or milestone checks—consistent with this repo’s split between light in-train eval and `parallel_eval.sh`.

### 8.4 Interpretation (explicitly labeled)

These are **reasoned conclusions** from the facts above, not separate empirical measurements from this workspace.

| Question | Observation |
|----------|----------------|
| **Will small eval batch / few episodes “misbehave” in what it shows?** | **Yes, in the sense of high variance:** with **4** episodes, discrete success counts are coarse (0/4, 1/4, …); **`batch_size=2`** only parallelizes two envs—it does **not** substitute for more episodes for statistics. Metrics can swing between eval ticks even when train loss trends smoothly. |
| **Is it at least somewhat representative of “how training is going”?** | **Partially:** repeated eval points at **`eval_freq=20000`** on a **150k** run still yield **trend** information (e.g. success moving off zero, correlating with loss). **Not** a tight estimate of final challenge performance. |
| **Complete waste of time?** | **No**, given documented **OOM** motivation and minimal extra cost vs full **50×50** LeRobot defaults. It buys **sanity checks** (divergence, total failure modes) and W&B visibility; **decisive** numbers remain the domain of heavier eval (`parallel_eval.sh`, larger `n_episodes`, or dedicated `lerobot-eval`). |

### 8.5 Relation to §7 templates in this document

Section **7** suggested **`n_episodes: 10`, `batch_size: 10`, `eval_freq: 10000`** as a generic “light but good” default for long merged runs. The **2026-04-28** MAP+registers **150k** path is **stricter** (fewer episodes, smaller eval batch, less frequent eval) specifically to **fit VRAM** on the **registers + MAP + diffusion** stack after observed failures—**not** because 10/10 was judged inferior in principle. For merged 400k-style runs, reassess episodes/batch once OOM behavior is confirmed on target GPU and LeRobot env wiring.

---

## Change log (this file)

| Item | Value |
|------|--------|
| **Created** | 2026-04-28 |
| **Path** | `Artifacts/Optimisation/merged_dataset_dino_map_registers_training_discussion_2026-04-28.md` |
| **Purpose** | Preserve reasoning, config scenarios, and research context from the conversation chain |
| **Updated** | 2026-04-28 — Appended **§8 Light eval stage (MAP+registers)** (grounded technical response: configs, `run_train_dino_dp_top_short_150k.sh`, changelog, DeepWiki + RivalSearch, interpretation table, tie-in to §7). |

---

*End of artifact.*
