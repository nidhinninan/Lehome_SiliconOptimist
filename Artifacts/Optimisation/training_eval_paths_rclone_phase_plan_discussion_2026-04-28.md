# Training, eval, paths, rclone, and phased-run plan — discussion artifact (2026-04-28)

This document captures a **detailed, technical record** of an internal conversation about LeHome / LeRobot training for the **DINO diffusion MAP+registers** policy: sweep run comparison, production 150k-style settings, **training-time eval** vs **challenge eval**, **dataset and output paths** on the VM, and the **`run_train_dino_dp_top_short_150k.sh`** launcher (including optional rclone and a two-phase no-aug → aug strategy).

**Scope**: facts grounded in this repo, user-provided VM logs/WandB descriptions, and **LeRobot v0.4.3** source URLs where cited. No local execution of `torch`/`lerobot` was performed (workspace policy).

---

## 1. Problem context

### 1.1 Goal

Train a **custom BYOP** policy (`lerobot_policy_dino`, type `dino_diffusion`) with **MAP spatial pooling + register tokens**, on LeHome-style data (three RGB streams at 480×640, 12-D state/action), with attention to:

- **Wall-clock time** vs **training quality** (loss curves are a proxy; real success needs sim/eval).
- **Hardware utilization** (GPU starvation vs dataloader / decode / host memory).
- **Operational** concerns: checkpoint cadence, WandB, optional **Google Drive offload** via `rclone`, and **resume** for a second training phase.

### 1.2 Dataset scale (reference)

From `Artifacts/Dataset-Viz_analysis/dataset_details.md` and `Artifacts/folding_cloth_research_v2.md`:

- Hugging Face dataset id: **`lehome/dataset_challenge_merged`** (challenge packaging).
- Four garment splits (250 episodes each), e.g. **`top_short_merged`** ≈ **76,066** frames; **`top_long_merged`** ≈ **83,068** frames; full merge ≈ **265,798** frames across 1,000 episodes.

The **10k sweep** runs discussed in chat logged **`dataset.num_frames=83068`, `num_episodes=250`** → they were on **`top_long_merged` only**, not the full four-way merge. Scaling wall time to a merged 4-type run is **not linear** without remeasuring (more episodes, more variety, same decode cost per frame).

### 1.3 Subvariant / eval reality

Challenge eval loads **garment subvariants** without a category label. Training-time metrics in LeRobot do not replace **LeHome `scripts.eval` / `parallel_eval.sh`** (typically **50 episodes** per garment type). Folding “capability” is **not** proven by train loss alone.

---

## 2. Three-variant sweep (user runs) — controlled differences and readouts

Three WandB-visible runs (orange / dark green / light green) shared the same **MAP+registers DINO-small** policy profile, **batch 8**, **torchcodec** video backend, **transforms off**, same dataset root for the sweep (**`top_long_merged`** in the pasted configs).

| Run | Color (WandB) | Key diffs | Steps | Notable timing (logs) |
|-----|----------------|-----------|-------|------------------------|
| 1 | Orange | `use_amp: false`, `num_workers: 4` | 10,000 | `updt_s` late ~**0.233–0.241** s; `data_s` ~**0.022** |
| 2 | Dark green | `use_amp: true`, `num_workers: 12` | 5,000 | `updt_s` ~**0.237–0.244** early, stabilizing; `data_s` ~**0.022** |
| 3 | Light green | `use_amp: true`, `num_workers: 16` | 5,000 | `updt_s` often **highest** late (~0.245–0.249); `data_s` still ~**0.022** |

**Reasoning**:

- Runs 2 and 3 used **half the optimizer steps** and a **shorter LR schedule** than run 1, so **final train loss is not comparable** at 5k vs 10k without aligning steps and schedulers.
- **`data_s` ~ flat** across worker counts suggests the bottleneck was **not** trivially “add workers → faster prefetch” in that VM snapshot; run 3 showed **more host RAM / disk pressure** and slightly worse `update_s` in WandB descriptions → **diminishing returns or contention** past 12 workers for this workload.

**Production-oriented recommendation from that discussion** (for long runs on the same machine class):

- Prefer **`use_amp: true`** (matches repo YAML intent for MAP+registers).
- Prefer **`num_workers: 12`** over **16** unless re-profiled on the target VM.
- For **“most training”**, use the **intended step count** (e.g. 150k) with a sane **save/eval** cadence — do not treat the 5k micro-sweep as the final word on capability.

---

## 3. Where training configuration comes from (no magic)

### 3.1 Entrypoint

`lehome_workspace/lerobot_train_with_plugins.py` only registers BYOP plugins then calls upstream:

```text
lerobot.scripts.lerobot_train.main
```

So behavior and defaults are defined by the **installed LeRobot** (project pins **`lerobot==0.4.3`** in challenge docs), not by this wrapper.

### 3.2 Merge order

1. **`--config_path=<file>`** — YAML (e.g. `configs/sweep_dino_map_registers.yaml`) or, on resume, `.../checkpoints/last/pretrained_model/train_config.json`.
2. **CLI overrides** — draccus dotted keys (e.g. `--eval_freq`, `--eval.n_episodes`).
3. **Resume caveat**: LeRobot’s `TrainPipelineConfig` documents that **resuming prefers the checkpoint’s saved config** over ad-hoc CLI for many fields — so **phase-2 aug cannot be toggled by CLI alone** without editing the resumed JSON or starting fresh.

### 3.3 `eval_freq` is top-level

Repo scripts explicitly pass **`--eval_freq=N`**, not `--eval.eval_freq` (see comment in `run_10k_sweep.sh`).

---

## 4. Eval: exact schema, defaults, and “light cadence”

### 4.1 Authoritative `EvalConfig` (LeRobot v0.4.3)

Source: `https://raw.githubusercontent.com/huggingface/lerobot/v0.4.3/src/lerobot/configs/default.py`

Fields (defaults):

- `n_episodes: int = 50`
- `batch_size: int = 50`
- `use_async_envs: bool = False`
- `__post_init__` raises if `batch_size > n_episodes`.

`TrainPipelineConfig` embeds:

- `eval_freq: int = 20_000` (default)
- `eval: EvalConfig = field(default_factory=EvalConfig)`

Source: `https://raw.githubusercontent.com/huggingface/lerobot/v0.4.3/src/lerobot/configs/train.py`

### 4.2 What `run_2run_dino_sweep.sh` actually set for eval

The sweep script **did not** pass `--eval.*` and the sweep YAMLs in-repo **omit** an `eval:` block → training used **LeRobot defaults**: **50 episodes, batch 50, async false**, with **`eval_freq=10000`** from the script’s `EVAL_FREQ` variable.

### 4.3 Light cadence (suggested pattern)

**Purpose**: cheap **monitoring** during training (catch divergence / broken env / policy NaNs) without paying the full cost of 50× parallel eval every N steps.

**Suggested training-time eval** (example used in discussion):

```yaml
eval:
  n_episodes: 10
  batch_size: 10
  use_async_envs: false
```

**CLI equivalent** (fresh runs):

```bash
--eval.n_episodes=10 --eval.batch_size=10 --eval.use_async_envs=false
```

**Still do “real” eval** for challenge metrics via `lehome-challenge/scripts/eval.py` or `parallel_eval.sh` (50 episodes / garment), which is **separate** from `lerobot-train`’s periodic eval.

---

## 5. `run_train_dino_dp_top_short_150k.sh` — design and file locations

### 5.1 Role

Single launcher for a **long MAP+registers** run on **`top_short_merged`** by default, using `configs/sweep_dino_map_registers.yaml` **without editing the YAML**, applying CLI overrides for steps, dataset root, output dir, logging, eval cadence, optional rclone.

### 5.2 Path semantics (relative vs absolute)

The script does:

- `WORKSPACE_DIR` = directory containing the script (`BASH_SOURCE`).
- `CHALLENGE_DIR="$WORKSPACE_DIR/lehome-challenge"`.
- `cd "$CHALLENGE_DIR"` before invoking training.

Default dataset CLI:

```text
--dataset.root=Datasets/example/top_short_merged
```

**This is intentionally relative to `lehome-challenge`**, so on a VM like:

```text
~/data/lehome_workspace/lehome-challenge/Datasets/example/...
```

the effective path is:

```text
~/data/lehome_workspace/lehome-challenge/Datasets/example/top_short_merged
```

**Nothing is “missing”** if `Datasets/example/...` exists under `lehome-challenge`. Absolute `/root/data/...` paths appear in some **YAML templates** for other machines; the 150k script’s relative form matches a **`cd lehome-challenge`** workflow.

### 5.3 Defaults implemented in the script (phase-1 oriented)

| Variable | Default | Scenario / intent |
|----------|---------|-------------------|
| `STEPS` | `150000` | Full long run unless overridden (e.g. **`STEPS=70000`** for first leg of 70k+80k plan). |
| `SAVE_FREQ` | `50000` | User asked for **artifacts at 50k** (and 100k/150k on a 150k run). |
| `EVAL_FREQ` | `10000` | Periodic **training-time** eval for monitoring. |
| `LOG_FREQ` | `1000` | Reasonable WandB/log granularity. |
| `JOB_NAME` | `DINOv2_MAP_Registers_dp_top_short_150k_phase1` | Distinguish phase-1 WandB / Drive folder naming. |
| `DATASET` | `Datasets/example/top_short_merged` | Default single split; override for other garment YAML-relative roots. |
| `OUTPUT` | `outputs/train/dp_top_short_dino_map_registers_150k` | Relative to `lehome-challenge`; holds `checkpoints/`. |
| `EVAL_N_EPISODES` / `EVAL_BATCH_SIZE` / `EVAL_USE_ASYNC_ENVS` | `10` / `10` / `false` | **Light** training-time eval; satisfies `batch_size <= n_episodes`. |

**Fresh run** passes `--eval.*` overrides; **resume** path uses only checkpoint `train_config.json` + `--resume=true` → **eval overrides are not re-applied** on resume unless embedded in that JSON or changed upstream.

### 5.4 Optional rclone (`ENABLE_RCLONE_CHECKPOINT_SYNC=1`)

Pattern aligned with `run_2run_dino_sweep.sh`:

- **During training**: background loop moves **`step_*`** only (min-age + sleep), **keeps `checkpoints/last` local** for resume.
- **On exit**: stop background job; **`rclone copy`** `checkpoints/last` to Drive; **`rclone move`** remaining `step_*`; best-effort **delete local `step_*`** while leaving **`last/`** for phase-2 resume.

**Caveat**: each invocation generates a new `RUN_TAG` (timestamp + hostname) → **new Drive subfolder per run** unless the script is later changed to pin a stable `RCLONE_BASE`.

### 5.5 Resume command shape

Example:

```bash
cd /path/to/lehome_workspace
ENABLE_RCLONE_CHECKPOINT_SYNC=1 RESUME=true ./run_train_dino_dp_top_short_150k.sh
```

**Must** use `RESUME=true` (string), not `RESUME=1`, because the script compares to `"true"`.

**Must** keep **`OUTPUT`** consistent with the run being resumed.

---

## 6. Two-phase plan (70k no-aug + 80k aug) — what is and is not covered

**Stated plan**:

- Phase 1: **70k** steps, **no augmentation** (`dataset.image_transforms.enable=false`).
- Phase 2: **80k** further steps with **augmentation on**, same **`batch_size=8`**, **`use_amp=true`**, **`num_workers=12`**.

**Covered by current launcher for phase 1**:

- Transforms off on fresh runs (`--dataset.image_transforms.enable=false`).
- Light eval + save/eval/log cadence as above.
- Optional rclone with **`last/`** retained locally.

**Not automatically solved**:

- **Default `STEPS` is still 150000** → for a strict 70k phase-1 leg, run with **`STEPS=70000`** explicitly.
- **Augmentation on resume** requires editing **`checkpoints/last/pretrained_model/train_config.json`** (or equivalent) because LeRobot resume loads that config; CLI toggles may be ignored.

---

## 7. Repo scripts cross-reference

| Script | Role |
|--------|------|
| `lehome_workspace/run_train_dino_dp_top_short_150k.sh` | Long-run launcher; shm, workers, BYOP install, optional rclone, phase-1 naming. |
| `lehome_workspace/run_2run_dino_sweep.sh` | Two-run sweep; `EVAL_FREQ=10000`; optional rclone; thread caps. |
| `lehome_workspace/run_10k_sweep.sh` | Documents `eval_freq` top-level CLI pitfall. |
| `lehome_workspace/lehome-challenge/parallel_eval.sh` | **Challenge-style** eval: 50 episodes, often `--device cpu`, **not** `lerobot-train` eval. |

---

## 8. “Am I missing something?” checklist (VM)

1. **Dataset path**: Under `lehome-challenge`, is `Datasets/example/top_short_merged` present and a valid LeRobot dataset root?
2. **CWD**: Script `cd`s to `lehome-challenge`; relative `dataset.root` is resolved there.
3. **Video backend**: YAML may say `torchcodec`; VM may need **`EXTRA_TRAIN_ARGS='--dataset.video_backend=pyav'`** if FFmpeg/torchcodec libs are missing (historical issue logged in `lehome_change_log.md`).
4. **`/dev/shm`**: If remount to 2G fails, script forces **`num_workers=0`** → large throughput hit.
5. **Resume**: `OUTPUT` matches prior run; `checkpoints/last/pretrained_model/train_config.json` exists.
6. **Eval expectations**: Training-time eval ≠ `parallel_eval.sh` success rates.

---

## 9. Config cheat sheet by scenario

### Scenario A — Micro-sweep (10k), MAP+registers, top_long, WandB on

- **Config file**: `lehome_workspace/configs/sweep_dino_map_registers.yaml`
- **Driver**: `run_2run_dino_sweep.sh` (or no-gdrive variant)
- **Typical**: `steps=10000`, `eval_freq=10000`, default **heavy** `eval` (50/50) unless overridden.

### Scenario B — Phase-1 long run (e.g. 70k), top_short, light train-eval, save at 50k

- **Launcher**: `run_train_dino_dp_top_short_150k.sh`
- **Invoke**: `STEPS=70000 ./run_train_dino_dp_top_short_150k.sh` (plus optional rclone env).
- **Train eval**: `eval_freq=10000`, `eval.n_episodes=10`, `eval.batch_size=10`, `use_async_envs=false`.
- **Saves**: `save_freq=50000` → at 70k you get **50k checkpoint + final progression** (exact step checkpoints depend on LeRobot’s save rules at end-of-run; `last/` remains the resume handle).

### Scenario C — Resume same run (continue steps, same output dir)

- **Invoke**: `RESUME=true ./run_train_dino_dp_top_short_150k.sh`
- **Rclone**: Optional `ENABLE_RCLONE_CHECKPOINT_SYNC=1`; note **new Drive subfolder per invocation** unless script is extended.

### Scenario D — “Real” challenge eval after training

- **Use**: `lehome-challenge/parallel_eval.sh` or `python -m scripts.eval ... --num_episodes 50 ...`
- **Not**: expecting `lerobot-train`’s light eval to match submission metrics.

### Scenario E — Phase-2 augmentation after phase-1

- **Requires**: mutating **saved** `train_config.json` under `checkpoints/last/...` (or new run), because **resume loads checkpoint config**; CLI `--dataset.image_transforms.enable=true` may not apply on resume.

---

## 10. Change log pointer (repo workspace)

Edits to `lehome_workspace/run_train_dino_dp_top_short_150k.sh`, `lehome_change_log.md`, and `vm_transfer_list.md` were recorded under **`2026-04-28 02:45:00 UTC`** in `lehome_workspace/lehome_change_log.md` for VM mirroring.

---

## 11. Sources (external)

- LeRobot **v0.4.3** `EvalConfig`: `https://raw.githubusercontent.com/huggingface/lerobot/v0.4.3/src/lerobot/configs/default.py`
- LeRobot **v0.4.3** `TrainPipelineConfig`: `https://raw.githubusercontent.com/huggingface/lerobot/v0.4.3/src/lerobot/configs/train.py`

---

*End of artifact.*
