## Recommended “final 150k” config set (single garment category like `top_short_merged`)

### Core training knobs (based on your 3-run sweep + system charts)
- **`policy.use_amp: true`**  
  - Your AMP runs (2/3) tracked loss similarly to non-AMP early, and AMP is the intended setting in `sweep_dino_map_registers.yaml`.
- **`num_workers: 12` (not 16)**  
  - Run 3 (16 workers) looked *slightly slower* in `train/update_s` and showed *more* host-memory/disk pressure; `data_s` stayed ~flat (~0.022s). So 16 didn’t reduce input stalls and likely increased CPU contention.
- **Keep**: `batch_size: 8`, `video_backend: torchcodec` (with fallback), transforms off, MAP+registers DINO small.

### For the 150k “production” run (your requested artifact cadence)
Use your existing `run_train_dino_dp_top_short_150k.sh`, but set:
- **`STEPS=150000`**
- **`SAVE_FREQ=50000`** (you requested artifacts only at 50k)
- **`LOG_FREQ=1000`** (fine; you can keep 500 if you want more granular timing plots)
- **Thread caps** (already in `run_2run_dino_sweep.sh` and in your 150k script):  
  `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` (important with 12 workers)

### Eval frequency (catch problems early without dominating wall clock)
Because your `eval` in configs is **heavy** (`n_episodes: 50`) and likely IsaacSim-bound, I’d use a two-phase cadence:
- **Early burn-in (first 10–20k steps)**: **`eval_freq = 5000`** *but* reduce eval cost: **`eval.n_episodes = 5–10`**  
- **Main run (after burn-in)**: **`eval_freq = 10000`** with **`eval.n_episodes = 10–20`**  
- **Milestone evaluations (at 50k / 100k / 150k)**: run a **full** eval (**50 episodes**) *once per milestone* (manual or via temporarily overriding `eval.n_episodes=50`).

If you must pick a single always-on setting: **`eval_freq=10000` and `eval.n_episodes=10–20`** is the best balance for early warning vs runtime.

### Rclone settings for your “save only at 50k” plan
From `run_2run_dino_sweep.sh`: background rclone only moves `step_*/**` with `--min-age 15m` and final flush moves everything.
- With checkpoints only at **50k/100k/150k**, background sync frequency can be relaxed:
  - **`RCLONE_SLEEP_SEC=1800`** (30 min) is fine
  - keep **`RCLONE_MIN_AGE=15m`**
- Expect the **final flush** to be the biggest single overhead chunk.

---

## Estimated wall-clock time for a 150k run (same VM, single category)

### Baseline compute estimate from your measured step timings
Using your observed steady-state:
- Typical **per-step wall time** \(\approx updt\_s + data\_s \approx 0.245\text{–}0.250 + 0.022 \approx 0.267\text{–}0.272\) seconds/step

So for **150,000** steps:
- **Core training loop** \(\approx 150000 \times 0.27 \approx 40,500\) seconds \(\approx 11.25\) hours

### Add expected overheads (rough order-of-magnitude)
- **Checkpoint save @ 50k / 100k / 150k**: typically minutes total (depends on filesystem + serialization)
- **Rclone uploads**:
  - background moves at 50k/100k can overlap training if they’re not saturating disk/network
  - **final flush** often adds **~10–40 minutes** depending on bandwidth and how much is left locally
- **Eval overhead** (dominant uncertainty):
  - If you keep `n_episodes=50` frequently, eval can easily add **hours** over 150k.
  - With the suggested lighter eval (10–20 episodes every 10k, full 50 only at milestones), budget roughly **~0.5–2 hours** total.

### Practical estimate you can plan around (single category)
- **Best-guess total**: **~12–14 hours** for 150k on `top_short_merged` (or any single category), with **save every 50k** and **light eval cadence**.
- If you run **50-episode eval every 10k**: expect materially longer (often **+several hours**).

---

## Final “use this” summary (configs)
- **Base YAML**: `lehome_workspace/configs/sweep_dino_map_registers.yaml` (unchanged)
- **CLI overrides for 150k single-category**:
  - `--dataset.root=Datasets/example/top_short_merged` (or other category)
  - `--steps=150000`
  - `--save_freq=50000`
  - `--eval_freq=10000` (plus smaller `eval.n_episodes` if you can override it)
  - `--num_workers=12` (and keep shm remount + thread caps)
  - Optional once-if-needed: `--dataset.video_backend=pyav` if torchcodec decode is flaky

If you want, I can point out exactly which flags to add to `run_train_dino_dp_top_short_150k.sh` to override `eval.n_episodes` / `eval.batch_size` (depends on the exact config schema accepted by your `lerobot_train_with_plugins.py`/draccus).