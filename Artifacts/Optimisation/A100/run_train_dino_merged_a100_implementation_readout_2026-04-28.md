# A100 merged DINO launcher + strong parallel eval — implementation readout

**Date:** 2026-04-28  
**Scope:** Record of what was implemented for the A100 DINO MAP+registers tuning plan (workspace scripts + tracking). Source: post-implementation summary aligned with [`lehome_workspace/run_train_dino_merged_a100.sh`](../../../lehome_workspace/run_train_dino_merged_a100.sh) and [`lehome_workspace/parallel_eval_merged_strong.sh`](../../../lehome_workspace/parallel_eval_merged_strong.sh).

---

## 1. `lehome_workspace/run_train_dino_merged_a100.sh` (A100 launcher)

- Same pipeline as `run_train_dino_dp_top_short_150k.sh`: `sweep_dino_map_registers.yaml`, `lerobot_train_with_plugins.py`, BYOP install, optional rclone, BLAS thread caps, numeric `step_*` rclone filters.
- Default **`DATASET=Datasets/example/dataset_challenge_merged`**.
- **`A100_TIER`** (only when **`RESUME!=true`**) sets plan presets if variables are still unset:
  - `conservative` | `normal` | `aggressive` | `smoke_cons` | `smoke_norm` | `smoke_aggr`  
  (matches the plan’s batch / workers / steps / eval / `OUTPUT` / `JOB_NAME` / `EXTRA_TRAIN_ARGS`).
- **`SHM_TARGET`** (default **`16G`**): if `/dev/shm` is smaller than ~16 GiB in `df -Pk` 1K-blocks, try `sudo mount -o remount,size=$SHM_TARGET`; on failure, try **`2G`**; if still no remount, **`num_workers=0`** as before.
- **Executable:** `chmod +x` applied.

---

## 2. `lehome_workspace/parallel_eval_merged_strong.sh` (stronger eval)

- Four parallel `python -m scripts.eval` runs (`--policy_type lerobot`, `--device cpu`, `--headless`).
- Default **`NUM_EPISODES=20`**, **`STEP_HZ=0`**.
- Per-category **`--dataset_root`**: `Datasets/example/{top_long,top_short,pant_long,pant_short}_merged` (same idea as the old `parallel_eval.sh`, with stronger episode count).
- Override checkpoint: **`POLICY_PATH`** (path under `lehome-challenge` after `cd`), default matches the plan’s normal run output.
- Uses **`CHALLENGE_DIR`** (default `$WORKSPACE_DIR/lehome-challenge`) so it works when the script lives in `lehome_workspace/`.
- **Executable:** `chmod +x` applied.

---

## 3. Tracking

- [`lehome_workspace/vm_transfer_list.md`](../../../lehome_workspace/vm_transfer_list.md) — **Last Updated** + new rows for both scripts.
- [`lehome_workspace/lehome_change_log.md`](../../../lehome_workspace/lehome_change_log.md) — new **2026-04-28 20:00:00 UTC** entry describing behavior and VM usage.

---

## VM usage (short)

```bash
cd /path/to/lehome_workspace
A100_TIER=normal RESUME=false ./run_train_dino_merged_a100.sh
# smoke:
A100_TIER=smoke_norm RESUME=false ./run_train_dino_merged_a100.sh

POLICY_PATH=outputs/train/dp_merged_dino_map8_registers_a100_400k_norm/checkpoints/last/pretrained_model \
  ./parallel_eval_merged_strong.sh
```

**Status:** All related implementation todos for this batch were completed at implementation time.

---

*End of artifact.*
