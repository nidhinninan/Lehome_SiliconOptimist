# LeHome Workspace Change Log

> [!IMPORTANT]
> **VM Transfer List**: All files required for replication on the VM are listed in [vm_transfer_list.md](./vm_transfer_list.md).
> **Updated Files**: ALL the latest iteration of the files can be found in [./updated_files/](./updated_files/) folder.

This document and the `updated_files` folder track all manual and agentic modifications to the `lehome_workspace/` repository clones. It serves as a grounded reference for replicating state on remote VMs.

---
<!-- LOG_START -->

### 2026-04-30 18:00:00 UTC — `dummy_docker_policy/`: VM one-shot Docker build script + upstream `server.py` + requirements template

**Why**: Provide a single bash entrypoint on the GPU VM to (1) optionally download a pinned W&B artifact into `pretrained_model/`, (2) `docker build` against `Dockerfile.submission`, (3) optionally `docker login` + push to HF Spaces registry (`registry.hf.space`). Aligns with `Artifacts/Custom_policy/docker_submission_plan.md` (no secrets in image; eval contract unchanged).

**Files**:
- `lehome_workspace/lehome-challenge/dummy_docker_policy/build_docker_submission.sh` — executable; flags `--wb-project`, `--wb-artifact`, `--skip-download`, `--policy-dir`, `--image-tag`, `--push`, `--hf-image-ref` (preferred), `--hf-user/--hf-space` (fallback), `--env-file`; sources `.env` for `WANDB_API_KEY` / `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`.
- `lehome_workspace/lehome-challenge/dummy_docker_policy/server.py` — copied from upstream `lehome-official/lehome-challenge` `main` `dummy_docker_policy/server.py` (official HTTP contract; do not edit per challenge docs).
- `lehome_workspace/lehome-challenge/dummy_docker_policy/requirements.submission.template` — minimal non-torch deps + commented pins; copy to `requirements.txt` via `--init-requirements` or manually; extend with `lerobot` etc. to match training.

**Also updated**: `download_wandb_model.py` now accepts fully-qualified W&B artifact refs like `entity/project/name:alias` directly via `--artifact` (in addition to the older `--project` + short-name style). This matches W&B documented formats and reduces one-shot failures.

**Existing (unchanged this step)**: `Dockerfile.submission`, `policy.py`.

### 2026-04-29 20:15:00 UTC — `source/lehome/setup.py`: fix editable install without PyPI `toml`

**Why**: `uv pip install -e …/source/lehome` runs setuptools in an isolated build environment. Upstream `setup.py` did `import toml` at module scope; if `[build-system].requires` omits `toml` (or isolation does not install it), metadata preparation fails with `ModuleNotFoundError: No module named 'toml'` on Python 3.12.

**File**: `lehome_workspace/lehome-challenge/source/lehome/setup.py`

**Change**: On Python **3.11+**, read `config/extension.toml` with stdlib **`tomllib`** and binary `open(..., "rb")`. On **3.10**, keep **`toml.load`** (requires `toml` in build isolation — declare in `pyproject.toml` `[build-system].requires` for that case).

**Diff** (conceptual): remove top-level `import toml`; add `sys.version_info` branch with `tomllib.load` vs `toml.load`.

### 2026-04-29 12:00:00 UTC — setup_lehome.sh: Principia `/data` paths

**Why**: Align `setup_lehome.sh` with Principia layout from `principia_vm_troubleshooting.md` / `technical_grounding_architectural_synthesis.md` — workspace and caches on `/data`, not `$HOME/data`.

**File**: `lehome_workspace/setup_lehome.sh`

**Diff summary**:
- `WORKSPACE_DIR=/data/lehome_workspace`, `UV_BIN_DIR=/data/.local/bin`.
- `HF_HOME=/data/huggingface_cache`, `UV_CACHE_DIR=/data/uv_cache`; `.bashrc` and current-session exports use those absolute paths.
- `sudo mkdir` includes `$UV_BIN_DIR`; `chown -R principia:principia` on workspace, cache dirs, and uv bin parent.

### 2026-04-28 23:05:00 UTC — Split: merged launcher = A100 tiers; dp_top_short = 4070 Ti on Principia

**Why**: Correct earlier mix-up — `run_train_dino_merged_a100.sh` must keep **A100** batch/worker presets regardless of Principia `/data` paths (only cache/PATH exports are Principia-specific). **RTX 4070 Ti 16GB** tuning belongs on **`run_train_dino_dp_top_short_150k.sh`** when `TOP_SHORT_GPU_PROFILE=4070ti` (default on Principia `/data` paths). Parallel eval default checkpoint remains the **A100** merged artifact unless `POLICY_PATH` is set.

**Files**:
- `lehome_workspace/run_train_dino_merged_a100.sh`
- `lehome_workspace/run_train_dino_dp_top_short_150k.sh`
- `lehome_workspace/parallel_eval_merged_strong.sh`

**Summary**:
- **Shared (paths)**: When `LEHOME_VM_PROFILE=principia` or `WORKSPACE_DIR` is under `/data/lehome_workspace`, export `UV_CACHE_DIR`, `HF_HOME`, optional `/data/.local/bin` `PATH`.
- **`run_train_dino_merged_a100.sh`**: Removed `principia_*` tiers and path-based default-tier override. **`A100_TIER:-normal`** only: conservative/normal/aggressive/smoke_* — same on Principia as on an A100 box; docs clarify Principia path block does not change GPU presets.
- **`run_train_dino_dp_top_short_150k.sh`**: **`TOP_SHORT_GPU_PROFILE`** default `4070ti`; when `4070ti` and Principia `/data` workspace: default `JOB_NAME`/`OUTPUT` suffix `_4070ti`, append `--batch_size=12 --num_workers=10` if `--batch_size=` not already in `EXTRA_TRAIN_ARGS`, `WORKERS=10` when shm OK. Set **`TOP_SHORT_GPU_PROFILE=a100`** to use legacy names/batches on the same paths.
- **`parallel_eval_merged_strong.sh`**: Single default **`POLICY_PATH`** → `..._a100_400k_norm/...`; Principia still gets `HF_HOME`/PATH and optional `CHALLENGE_DIR`.

### 2026-04-28 20:00:00 UTC — A100 merged DINO training launcher + strong parallel eval script (plan implementation)

**Why**: Execute the A100 DINO MAP+registers tuning plan without altering the existing `run_train_dino_dp_top_short_150k.sh` 4070-oriented launcher.

**Files**:
- `lehome_workspace/run_train_dino_merged_a100.sh` (new)
- `lehome_workspace/parallel_eval_merged_strong.sh` (new)

**Summary**:
- **`run_train_dino_merged_a100.sh`**: Same training entry as `run_train_dino_dp_top_short_150k.sh` (`lerobot_train_with_plugins.py`, `configs/sweep_dino_map_registers.yaml`, BYOP install, rclone optional, thread caps). Defaults `DATASET=Datasets/example/dataset_challenge_merged`. **`A100_TIER`** selects plan presets when `RESUME!=true` and vars are unset: `conservative|normal|aggressive|smoke_cons|smoke_norm|smoke_aggr` (steps/save/eval/output/`EXTRA_TRAIN_ARGS` for batch/workers/map queries). **`SHM_TARGET`** (default `16G`) used when remounting small `/dev/shm`; fallback to `2G` if needed. **`SHM_MIN_KB=$((16*1024*1024))`**: remount attempted when `df -Pk` free 1K-blocks is below ~16 GiB.
- **`parallel_eval_merged_strong.sh`**: Four background `python -m scripts.eval` jobs (`--policy_type lerobot`, `--device cpu`, `--headless`, default **`NUM_EPISODES=20`**), per-garment **`dataset_root`** (`top_long_merged`, …). Override checkpoint with **`POLICY_PATH`** (relative to `lehome-challenge` after `cd`).

**VM**: `chmod +x` both scripts; run launcher from `lehome_workspace`; run eval with `CHALLENGE_DIR` pointing at the VM’s `lehome-challenge` clone (script default: `$WORKSPACE_DIR/lehome-challenge`).

### 2026-04-28 11:43:30 UTC — Minimal fix: rclone filter matches LeRobot numeric checkpoint dirs (`030000/`) not just `step_*/`

(Reverted unrelated additions from 11:35:45 UTC entry below: `is_truthy()` helper, `RCLONE_COPY_LAST_DURING_TRAINING`, per-pass log markers — none were needed to fix the reported bug.)

### 2026-04-28 11:41:29 UTC — Fix: rclone filter matches LeRobot numeric checkpoint dirs (`030000/`) not just `step_*/`

**Why**: On VM, LeRobot saved checkpoints as numeric directories (`030000`, `060000`, …). The launcher’s rclone loop filtered only `step_*/**`, so nothing was moved mid-training (and the final sync also skipped them).

**Files**:
- `lehome_workspace/run_train_dino_dp_top_short_150k.sh`

**Diff summary**:
```diff
 rclone move "$SRC_DIR" "${RCLONE_DST}checkpoints" \
   --filter '+ step_*/**' \
+  --filter '+ [0-9]*/**' \
   --filter '- **' \
   ...

 rclone move "$OUTPUT/checkpoints" "${RCLONE_DST}checkpoints" \
   --filter '+ step_*/**' \
+  --filter '+ [0-9]*/**' \
   --filter '- **' \
   ...

-find ... -name 'step_*' ...
+find ... \( -name 'step_*' -o -name '[0-9]*' \) ...
```

### 2026-04-28 11:35:45 UTC — Fix: make mid-training rclone sync verifiable + enable with truthy flags

**Why**: User suspected checkpoints were not being uploaded mid-training. The launcher *did* spawn a background rclone loop, but it only enabled when `ENABLE_RCLONE_CHECKPOINT_SYNC=1` (not `true/yes`), only moved `step_*` after `--min-age`, and had no obvious per-pass marker in logs.

**Files**:
- `lehome_workspace/run_train_dino_dp_top_short_150k.sh`

**Diff summary**:
```diff
-if [ "${ENABLE_RCLONE_CHECKPOINT_SYNC:-0}" = 1 ]; then
+is_truthy() { case "${1:-}" in 1|true|yes|on|... ) return 0 ;; *) return 1 ;; esac; }
+RCLONE_COPY_LAST_DURING_TRAINING="${RCLONE_COPY_LAST_DURING_TRAINING:-0}"
+if is_truthy "${ENABLE_RCLONE_CHECKPOINT_SYNC:-0}"; then
   ...
   while true; do
     rclone move ... --filter '+ step_*/**' ... --min-age "$RCLONE_MIN_AGE" ...
+    # optional safety: non-destructive copy of checkpoints/last during training
+    rclone copy "$SRC_DIR/last" "${RCLONE_DST}checkpoints/last" --update ...
   done
```

**How to verify on VM**:
```bash
ENABLE_RCLONE_CHECKPOINT_SYNC=true \
RCLONE_COPY_LAST_DURING_TRAINING=true \
RCLONE_SLEEP_SEC=300 \
RCLONE_MIN_AGE=5m \
./run_train_dino_dp_top_short_150k.sh
```
Then watch the rclone log:
`lehome-challenge/logs/readout/dino_map_dp_top_short_150k_phase1_rclone_*.log`

### 2026-04-28 12:30:00 UTC — Balanced training-time eval for DINOv2+MAP 150k run

**Why**: User wants to monitor ongoing learning progress via eval metrics (loss curves, success rate, etc.) in W&B/console while avoiding the OOM that occurred at the first `eval_freq=10000` boundary. 
- Increased `eval_freq` to 20000 (fewer but still informative checkpoints)
- Reduced to very light eval (`n_episodes=4`, `batch_size=2`) to minimize GPU spike on heavy model (DINOv2 registers + MAP head)
- Added explicit `eval:` section to YAML for self-documenting config
- Updated script defaults and comments

**Files**:
- `lehome_workspace/run_train_dino_dp_top_short_150k.sh` (updated defaults + comments)
- `lehome_workspace/configs/sweep_dino_map_registers.yaml` (added eval section)

**Recommended launch**:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  RESUME=true ./run_train_dino_dp_top_short_150k.sh
```

### 2026-04-28 02:45:00 UTC — Update: phase-1 150k launcher (light eval + keep `last/` local while rclone offloads step checkpoints)

**Why**: Make the first phase of the planned 2-stage run (70k no-aug → resume with aug later) cheaper to monitor and safer to resume by keeping `checkpoints/last` locally while still offloading `step_*` checkpoints to Google Drive.

**Files**:
- `lehome_workspace/run_train_dino_dp_top_short_150k.sh` (updated)

**Key behavior changes (diff summary)**:
```diff
 # Defaults now match recommended “production” cadence
-SAVE_FREQ=20000
-EVAL_FREQ=20000
+SAVE_FREQ=50000
+EVAL_FREQ=10000
 JOB_NAME default now includes _phase1

 # Light training-time eval overrides (LeRobot v0.4.3 EvalConfig)
+--eval.n_episodes=10
+--eval.batch_size=10
+--eval.use_async_envs=false

 # Optional rclone support (ENABLE_RCLONE_CHECKPOINT_SYNC=1)
+During training: rclone move only step_*/** (keeps checkpoints/last local)
+After training: rclone copy checkpoints/last to Drive; move any remaining step_*/**
+Best-effort prune of local step_* dirs; leaves checkpoints/last for phase-2 resume
```

**Notes / gotchas**:
- Phase-2 augmentation switch (`dataset.image_transforms.enable=true`) cannot be applied via CLI when resuming because LeRobot resume loads config from `checkpoints/last/pretrained_model/train_config.json`. You must edit that JSON before resuming (or start a fresh run with the desired config).

### 2026-04-27 23:45:00 UTC — DINO MAP+registers 150k launcher (dp_top_short, CLI overrides on sweep YAML)

**Why**: Single long-run script for `dp_top_short` using existing `configs/sweep_dino_map_registers.yaml` without editing it; CLI overrides `steps`, `dataset.root`, `output_dir`, `save_freq`/`eval_freq`/`log_freq`/`job_name`; same BYOP/shm/W&B/thread-cap pattern as `run_2run_dino_sweep.sh`.

**Files**:
- `lehome_workspace/run_train_dino_dp_top_short_150k.sh` (new)

**Behavior summary**:
- `WORKSPACE_DIR` from `BASH_SOURCE`; `cd lehome-challenge`; `uv`/`pip install -e lerobot_policy_dino`; optional `.env` + `WANDB_API_UCMO` → `WANDB_API_KEY`.
- SHM check → `WORKERS=12` or `0`; exports `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS` default `1`.
- Fresh: `lerobot_train_with_plugins.py --config_path=$WORKSPACE_DIR/configs/sweep_dino_map_registers.yaml` + `--steps=150000` (default), `--save_freq=20000`, `--eval_freq=20000`, `--log_freq=1000`, `--job_name=DINOv2_MAP_Registers_dp_top_short_150k`, `--dataset.root=Datasets/example/top_short_merged`, `--output_dir=outputs/train/dp_top_short_dino_map_registers_150k`, `--dataset.image_transforms.enable=false`, `--num_workers=$WORKERS`. Env overrides: `RESUME`, `STEPS`, `SAVE_FREQ`, `EVAL_FREQ`, `LOG_FREQ`, `JOB_NAME`, `DATASET`, `OUTPUT`, `SKIP_IF_LAST_EXISTS`, `EXTRA_TRAIN_ARGS`.
- Resume: `--config_path=$OUTPUT/checkpoints/last/pretrained_model/train_config.json --resume=true` (minimal CLI). Optional skip: `SKIP_IF_LAST_EXISTS=1` when not resuming if `checkpoints/last` exists (also checks `/root/data/...` and `/data/...` absolute variants).
- **VM**: run on GPU VM; local workspace does not execute `lerobot_train`.

### 2026-04-27 22:57:13 UTC — MAP+registers training profile: AMP, torchcodec, num_workers in YAML; thread caps in sweep scripts

**Why**: Adopt the concrete MAP+registers VM profile (`use_amp`, `dataset.video_backend=torchcodec`, `num_workers=12`, `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS` defaults, transforms off).

**Files**:
- `lehome_workspace/configs/sweep_dino_map_registers.yaml`
- `lehome_workspace/run_2run_dino_sweep.sh`
- `lehome_workspace/run_2run_dino_sweep_no-gdrive.sh`

**`configs/sweep_dino_map_registers.yaml` (diff summary)**:
```diff
 dataset:
   repo_id: repo_dp
   root: /root/data/lehome_workspace/lehome-challenge/Datasets/example/top_long_merged
+  video_backend: torchcodec
+  # comment: FFmpeg + torchcodec required; CLI override --dataset.video_backend=pyav if needed
   image_transforms:
     enable: false
 policy:
   ...
-  use_amp: false
+  use_amp: true
 ...
 batch_size: 8
+num_workers: 12
```

**Sweep scripts**: Before `lerobot_train_with_plugins.py`, export `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` with `${VAR:-1}` so the VM can override. Header comment documents MAP+registers throughput rationale.

**Note**: `sweep_dino_baseline.yaml` unchanged. Second sweep job still uses `--dataset.image_transforms.enable=false` from CLI; MAP YAML keeps `image_transforms.enable: false`.

### 2026-04-27 16:13:41 UTC — Rename sweep scripts: trial → default name; original → `no-gdrive`

**Renames (git mv)**:
- `lehome_workspace/run_2run_dino_sweep_trial.sh` → `lehome_workspace/run_2run_dino_sweep.sh` (rclone-capable driver is now the canonical filename).
- `lehome_workspace/run_2run_dino_sweep.sh` → `lehome_workspace/run_2run_dino_sweep_no-gdrive.sh` (previous default two-run sweep without Drive offload).

**Edits**:
- `run_2run_dino_sweep.sh`: header/usage comments use new basename; echo labels drop “(trial)”; docstring cross-ref points to `run_2run_dino_sweep_no-gdrive.sh` for the simpler variant.
- `run_2run_dino_sweep_no-gdrive.sh`: header states no-gdrive role and points to `run_2run_dino_sweep.sh` for optional sync.
- `configs/sweep_dino_baseline.yaml`, `configs/sweep_dino_map_registers.yaml`: YAML comment mentions both launcher names for `--dataset.image_transforms.enable`.
- `To_RUN_onVM_cmds.txt`: Drive section uses `./run_2run_dino_sweep.sh`; notes `./run_2run_dino_sweep_no-gdrive.sh` for no-offload.

**Historical note**: Log entries before this rename that cite **`run_2run_dino_sweep.sh`** for the 2026-04-26 pyav-only fix refer to the file that is now **`run_2run_dino_sweep_no-gdrive.sh`**.

### 2026-04-27 16:20:00 — New: `run_2run_dino_sweep_trial.sh` (rclone → Drive + `.complete` skip marker) *(filename later merged into `run_2run_dino_sweep.sh`; see 2026-04-27 16:13 UTC entry)*

**Files added**:
- `lehome_workspace/run_2run_dino_sweep_trial.sh` *(renamed to `run_2run_dino_sweep.sh` — see newer log entry)*

**Description**:
- Duplicate of the two-run DINO sweep driver with **trial** features: optional **Google Drive offload** via `rclone move`, and a **`output_dir/.complete` marker** so reruns skip after checkpoints were moved off-VM (not only when `checkpoints/last` exists).
- **`WORKSPACE_DIR`** is derived from the script path (`BASH_SOURCE`), matching `run_10k_sweep.sh` robustness (works regardless of `cwd`).
- **Drive layout** (when `ENABLE_RCLONE_CHECKPOINT_SYNC=1`):  
  `{RCLONE_REMOTE}:LeHome/models/{wandb.project}/{job_name}/{config_stem}__{RUN_TAG}/checkpoints/...`  
  where `RUN_TAG="${job_name}__${config_stem}__$(date +%Y%m%d_%H%M%S)__$(hostname -s)"`, and `job_name` / `wandb.project` / `output_dir` are parsed from each YAML with small `grep`/`sed` helpers.
- **During training**: background loop every `RCLONE_SLEEP_SEC` (default 600s) runs  
  `rclone move … --filter '+ step_*/**' --filter '- **' --min-age "${RCLONE_MIN_AGE:-15m}" --delete-empty-src-dirs`  
  so **`checkpoints/last/` is not moved** until training finishes.
- **After training**: stop background job, **final** `rclone move` (no step filter) moves **`last/`** and any remainder; on success, **`touch "$OUTPUT_DIR/.complete"`**. If rclone is disabled, still writes `.complete` after a successful training run so the sweep does not restart blindly.
- **Skip order**: if `"$OUTPUT_DIR/.complete"` exists → skip; else if `outputs/.../checkpoints/last` or `/root/data/lehome_workspace/.../last` exists → skip (same as original).
- **Training CLI**: keeps `--dataset.video_backend=pyav` (VM-safe vs missing FFmpeg libs for torchcodec).
- **Env knobs**: `ENABLE_RCLONE_CHECKPOINT_SYNC` (default 0), `RCLONE_REMOTE` (default `gdrive`), `RCLONE_MIN_AGE`, `RCLONE_SLEEP_SEC`.

**Authoritative source**: same implementation now lives in [`lehome_workspace/run_2run_dino_sweep.sh`](./run_2run_dino_sweep.sh) (~230 lines). Key fragments:

```bash
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RCLONE_BASE="LeHome/models/${WANDB_PROJECT}/${JOB_NAME}/${CONFIG_STEM}__${RUN_TAG}"
RCLONE_DST="${RCLONE_REMOTE_NAME}:${RCLONE_BASE}/"
rclone move "$LOOP_SRC" "${RCLONE_DST}checkpoints" \
  --filter '+ step_*/**' --filter '- **' --min-age "$RCLONE_MIN_AGE" --delete-empty-src-dirs
# after training: rclone move "$CHECK_SRC" "${RCLONE_DST}checkpoints" --delete-empty-src-dirs
# touch "$OUTPUT_DIR/.complete"
```

**Operational note**: To re-run a finished trial for the same YAML/output path, remove `"$output_dir/.complete"` (and any partial local checkpoints) on the VM.

### 2026-04-26 20:05 — Fix: torchcodec FFmpeg missing → switch to pyav video backend

**Problem**: `run_2run_dino_sweep.sh` crashed in the DataLoader with:
```
RuntimeError: Could not load libtorchcodec. FFmpeg shared libraries not found
(libavutil.so.59/58/57 and libavdevice.so.58 all missing).
```
The VM's LeRobot install uses `torchcodec` as the default video backend, which requires system FFmpeg shared libraries that were not installed.

**Fix 1** — `lehome_workspace/run_2run_dino_sweep.sh`:
Added `--dataset.video_backend=pyav` to the training command. `pyav` decodes video frames via the Python `av` package (already installed as a LeRobot dependency) without needing system FFmpeg `.so` files.

**Diff**:
```diff
 "$VENV_PYTHON" "$WORKSPACE_DIR/lerobot_train_with_plugins.py" \
     --config_path="$CONFIG" \
     --steps="$STEPS" \
     --eval_freq="$EVAL_FREQ" \
     --dataset.image_transforms.enable=false \
+    --dataset.video_backend=pyav \
     --num_workers="$WORKERS" \
```

**Fix 2** — `lehome_workspace/setup_lehome.sh`:
Added `ffmpeg` to the `apt install` block so future fresh VM setups will have the system FFmpeg libraries available (allowing `torchcodec` to work as well).

**Diff**:
```diff
 sudo apt install -y \
     libglu1-mesa libgl1 libegl1 libxrandr2 \
     libxinerama1 libxcursor1 libxi6 libxext6 libx11-6 \
-    zip psmisc  # psmisc includes 'fuser'
+    zip psmisc \
+    ffmpeg  # required by torchcodec video backend
```

### 2026-04-26 18:44:41 — Fix: `setup_lehome.sh` paths use `$HOME` (not literal `~`)
**Files modified**:
- `lehome_workspace/setup_lehome.sh`

**Description**:
- `WORKSPACE_DIR`, `REPO_DIR`, and `UV_BIN_DIR` now use `${HOME}/data/...` so `cd`, `mkdir`, and `uv` install see real absolute paths (tilde inside `"..."` assignments stays literal and broke `cd "$REPO_DIR"` on the VM).
- `.bashrc` snippets now append `export HF_HOME="$HOME/data/huggingface_cache"`, `export UV_CACHE_DIR="$HOME/data/uv_cache"`, and `export PATH="$HOME/data/.local/bin:$PATH"` so new shells expand home correctly.
- Session exports use `${HOME}/data/...` and `PATH="${UV_BIN_DIR}:${PATH}"`.

**Diff (configuration block)**:
```bash
# before
WORKSPACE_DIR="~/data/lehome_workspace"
REPO_DIR="$WORKSPACE_DIR/lehome-challenge"
UV_BIN_DIR="~/data/.local/bin"
# after
WORKSPACE_DIR="${HOME}/data/lehome_workspace"
REPO_DIR="${WORKSPACE_DIR}/lehome-challenge"
UV_BIN_DIR="${HOME}/data/.local/bin"
```

**VM note**: If `.bashrc` already contains the old `~/data/...` lines, remove those duplicates after sync so only the `$HOME`-based exports remain.

### 2026-04-26 12:00:00 — DINOv2 MAP + registers variant + 2-run sweep + W&B
**Files modified / added**:
- `lehome_workspace/lerobot_policy_dino/src/lerobot_policy_dino/configuration_dino_diffusion.py`
- `lehome_workspace/lerobot_policy_dino/src/lerobot_policy_dino/modeling_dino_diffusion.py`
- `lehome_workspace/configs/sweep_dino_baseline.yaml` (new)
- `lehome_workspace/configs/sweep_dino_map_registers.yaml` (new)
- `lehome_workspace/run_2run_dino_sweep.sh` (new)

**Description**:
- Added `spatial_pooling` (`baseline` | `map`), `map_num_queries`, `use_registers`, `num_register_tokens` to `DinoDiffusionConfig` with validation: `map` requires registers checkpoint + `use_registers=True`.
- Implemented `MAPHead` (K queries cross-attend to patch tokens; output `reshape` to `(B, K*D)`). `DinoDiffusionModel` sets `head_output_dim` and `single_step_dim` accordingly; backbone stays frozen; MAP head is trainable.
- `_encode_images`: baseline preserves original CLS vs mean-pool; `map` drops CLS + register tokens then applies MAP (backbone under `no_grad`, head outside).
- Removed dead `self.reset()` after `return` in `get_optim_params`.
- New sweep YAMLs: baseline `facebook/dinov2-small` + MAP run `facebook/dinov2-with-registers-small`; both use `wandb.enable: true`, `wandb.project: lehome_challenge`, top-level `job_name` for distinct W&B runs (`DINOv2_Baseline`, `DINOv2_MAP_Registers`).
- New `run_2run_dino_sweep.sh`: installs only `lerobot_policy_dino`, loads optional `$WORKSPACE_DIR/.env` and sets `WANDB_API_KEY=$WANDB_API_UCMO`, runs MAP+registers then baseline (same CLI flags as `run_10k_sweep.sh`).

**Key diff (config validation excerpt)**:
```python
if self.spatial_pooling == "map":
    if not self.use_registers:
        raise ValueError("spatial_pooling='map' requires use_registers=True.")
    if "registers" not in self.vision_backbone.lower():
        raise ValueError("spatial_pooling='map' expects a dinov2-with-registers-* checkpoint.")
```

**Key diff (MAP forward)**:
```python
pooled = self.norm(attn_out + q)
return pooled.reshape(bsz, -1)
```

### 2026-04-12 10:40:00 — Documentation: Linked Dependency Comments
**Files Modified**:
- `lehome_workspace/configs/sweep_dino.yaml`
- `lehome_workspace/configs/sweep_clip.yaml`
- `lehome_workspace/configs/sweep_resnet18.yaml`
- `lehome_workspace/run_10k_sweep.sh`

**Description**:
- Added `🔗 LINKED DEPENDENCY` comments to all sweep-related configuration files and the orchestration script. This clarifies that the `image_transforms.enable` state must be manually synchronized across both the YAML and CLI flags to maintain scientific baseline integrity.

### 2026-04-12 10:29:00 — Alignment: Disabling Sweep Augmentations
- Disabled `image_transforms.enable` in DINO and CLIP sweep configurations.
- Patched `run_10k_sweep.sh` to remove a hardcoded `--dataset.image_transforms.enable=true` override.
- This aligns the entire sweep with the ResNet18 baseline and resolves the CPU-bound data loading bottleneck (`data_s: 0.62`) observed in the VM logs.

### 2026-04-12 14:02:00 — Bug-Fix: Proactive Downstream Pipeline Hotfixes
**Files Modified**:
- `lehome_workspace/lerobot_policy_dino/src/lerobot_policy_dino/modeling_dino_diffusion.py`
- `lehome_workspace/lerobot_policy_clip/src/lerobot_policy_clip/modeling_clip_diffusion.py`

**Description**:
- Audited the downstream training pipeline (LeRobot v0.4.3) using DeepWiki based on initialization failures and applied **two proactive hotfixes** to prevent imminent crashes during the training and validation loops:
  - **Eval Loop Crash (`noise` kwarg)**: LeRobot's `DiffusionPolicy.predict_action_chunk` now injects `noise=noise` during eval validation. Added the `noise: Tensor | None = None, **kwargs` signature to our custom `generate_actions` to prevent `TypeError: unexpected keyword argument`.
  - **Optimizer Crash (`requires_grad=False`)**: LeRobot's default `make_optimizer_and_scheduler` builds parameter groups using the raw output of `policy.get_optim_params()`. Since we subclassed `DiffusionPolicy` which blindly returns all parameters, the frozen vision backbones would have crashed the distributed optimizers. Overrode `get_optim_params()` manually to strictly filter and return only `[p for p in self.parameters() if p.requires_grad]`.

### 2026-04-12 13:54:00 — Bug-Fix: BYOP Policy Initialization Signature
**Files Modified**:
- `lehome_workspace/lerobot_policy_dino/src/lerobot_policy_dino/modeling_dino_diffusion.py`
- `lehome_workspace/lerobot_policy_clip/src/lerobot_policy_clip/modeling_clip_diffusion.py`

**Description**:
- Fixes `TypeError: DinoDiffusionPolicy.__init__() got an unexpected keyword argument 'dataset_meta'`. In recent LeRobot updates, the `make_policy()` factory function started passing `dataset_meta` directly to policy constructors. Updated the `__init__` signatures of both DINO and CLIP BYOP policies to include `dataset_meta: dict | None = None` and `**kwargs` to safely absorb these upstream changes.

### 2026-04-12 13:42:00 — Bug-Fix: BYOP Namespace Collision & Build System
**Files Modified**:
- `lehome_workspace/lerobot_train_with_plugins.py`
- `lehome_workspace/lerobot_policy_dino/pyproject.toml`
- `lehome_workspace/lerobot_policy_clip/pyproject.toml`

**Description**:
- **Build System Fix**: Added `[build-system]` to both `pyproject.toml` files. Without this, `uv` performed broken editable installs, resulting in empty namespace packages instead of linking the `src/` directories.
- **Namespace Injection**: Updated `lerobot_train_with_plugins.py` to directly inject the plugin `src/` directories into `sys.path`. Added safety checks to verify the packages are loaded correctly and crash if they resolve to empty namespace folders.

### 2026-04-12 12:51:00 — Automation: LeHome Setup & Optimized Training
**Files Added**:
- `lehome_workspace/setup_lehome.sh`
- `lehome_workspace/run_train_optimized.sh`

**Description**:
- Created `setup_lehome.sh` to automate the entire Principia VM environment setup (storage redirection, system deps, `uv` installation, and dataset downloads) using absolute paths.
- Created `run_train_optimized.sh` as a configurable launcher for multi-garment training. Fixed the shared memory "Bus Error" bug by adding automatic `/dev/shm` remounting. Enabled timestamped logging and optimized `num_workers`.

### 2026-04-12 07:41:00 — Bug-Fix: Support `uv pip` in Sweep Script
**Files Modified**:
- `lehome_workspace/run_10k_sweep.sh`

**Description**:
- Updated the BYOP package installation logic to detect if `uv` is available. If `uv` is found, it uses `uv pip install --python <venv>` instead of standard `pip`. This fixes the "No module named pip" error encountered on the Principia VM, which manages venvs with `uv` without installing `pip` by default.

### 2026-04-12 07:34:00 — Bug-Fix: `df` command compatibility in Sweep Script
**Files Modified**:
- `lehome_workspace/run_10k_sweep.sh`

**Description**:
- Fixed an `invalid option -- 'K'` error encountered on the Principia VM. The `df` command flag for 1024-byte blocks was corrected from capital `-K` to lowercase `-k` (standard GNU/POSIX flag).

### 2026-04-12 11:37:00 — Bug-Fix: BYOP Plugin Discovery & Sweep Resumption
**Files Modified**:
- `lehome_workspace/run_10k_sweep.sh`
**Files Added**:
- `lehome_workspace/lerobot_train_with_plugins.py`

**Description**:
- **Plugin Registration Fix**: Addressed `KeyError: 'dino_diffusion'` caused by LeRobot's auto-discovery mechanism failing to find packages where underscores were normalized to hyphens. Created `lerobot_train_with_plugins.py` to explicitly import and register BYOP plugins before training.
- **Sequential Resumption**: Patched `run_10k_sweep.sh` to support incremental sweeps. The script now checks for existing checkpoints in both relative and absolute (`/data/outputs/`) paths to ensure compatibility with various VM storage configurations.

### 2026-04-12 00:22:00 — Bug-Fix: Shared Memory Crash & New Normalization API

### 2026-04-12 02:40:00 — Part 1: Garment Classifier Training & Audit
**Files Added**:
- `scripts/garment_classifier/sanity_check_first_frames.py`
- `scripts/garment_classifier/export_garment_classifier_dataset.py`
- `scripts/garment_classifier/train_garment_classifier.py`
- `.cursor/rules/lehome-environment-split.mdc`

**Description**:
- **Sanity Check**: Built `sanity_check_first_frames.py` using `LeRobotDataset` API to sample 15-20 random `frame_idx=0` images per garment class for visual review. This prevents "blind training" on occluded startup states.
- **Extraction**: Built `export_garment_classifier_dataset.py` to convert LeRobot v3 `.parquet/.mp4` datasets into an `ImageFolder` structure (`train/`, `val/`) with class-sorted mapping.
- **Training**: Built `train_garment_classifier.py` for ResNet18 fine-tuning. Upgraded augmentations from standard ImageNet-diversities to robotic-overhead specific ones: `RandomAffine(degrees=15, translate=(0.1, 0.1))` and `GaussianBlur(kernel_size=3)`.
- **Environment Guard**: Created `.cursor/rules/lehome-environment-split.mdc` to explicitly codify that GPU/Library intensive operations belong on the **VM**, ensuring future agents don't attempt local runs of `torch/lerobot`.
- **Lazy Load Config**: Updated the router policy to support `lazy_load` and `min_confidence` toggles in JSON.

### 2026-04-12 00:22:00 — Bug-Fix: Shared Memory Crash & New Normalization API
**Files Modified**:
- `lehome_workspace/run_10k_sweep.sh`
- `lehome_workspace/lerobot_policy_dino/src/lerobot_policy_dino/modeling_dino_diffusion.py`
- `lehome_workspace/lerobot_policy_clip/src/lerobot_policy_clip/modeling_clip_diffusion.py`
- `Artifacts/installation_guide.md`

**Description**:
- Fixed **Fatal Error** (Shared Memory Bus Crash): Added logic to `run_10k_sweep.sh` to detect `/dev/shm` size and attempt to remount it to 2GB via `sudo`. Added a fallback to `num_workers=0` if remounting fails to ensure the training doesn't crash.
- Fixed **ModuleNotFoundError** (Normalization API): LeRobot v0.4.3 refactored `Normalize` and `Unnormalize` from `lerobot.policies.normalize` to `lerobot.processor.normalize_processor`. Updated both CLIP and DINO modeling files to use the new `NormalizerProcessorStep` and `UnnormalizerProcessorStep` classes (aliased for drop-in compatibility).
- Fixed **Environment Leak**: `run_10k_sweep.sh` now uses the absolute path to the project's virtual environment python (`VENV_PYTHON`) for all commands (`pip install`, `lerobot-train`). This bypasses the Principia VM's global scripts which often point to incompatible library versions.
- Updated `installation_guide.md` with explicit warnings about the global `lerobot-train` alias.

### 2026-04-12 00:02:00 — Bug-Fix: `lerobot.common` Import Error
**Files Modified**:
- `lehome_workspace/lerobot_policy_dino/src/lerobot_policy_dino/modeling_dino_diffusion.py`
- `lehome_workspace/lerobot_policy_clip/src/lerobot_policy_clip/modeling_clip_diffusion.py`

**Description**:
- `lerobot.common` does not exist as a subpackage in the pip-installed `lerobot==0.4.3`. That path only exists in the GitHub development layout (src tree). The pip package flattens it.
- Removed `from lerobot.common.constants import OBS_ENV, OBS_ROBOT` from both modeling files.
- Replaced with hardcoded module-level string constants `OBS_ROBOT = "observation.state"` and `OBS_ENV = "observation.environment_state"`, which are the exact values the constants resolve to in the source anyway.
- **Error 2** (dataset `FileNotFoundError`): The `root:` path in the YAML sweep configs is a relative path (`Datasets/example/top_long_merged`). Must be updated to the **absolute path** on the VM before running. User to apply manually.

### 2026-04-11 23:32:00 — Bug-Fix: Sweep Script Absolute Paths
**Files Modified**:
- `lehome_workspace/run_10k_sweep.sh`

**Description**:
- Users running the sweep script from inside `/data/lehome_workspace/` were encountering a `path not found` error because the `pip install` commands hardcoded the `lehome_workspace/` prefix.
- Updated `run_10k_sweep.sh` to dynamically resolve its own parent directory (`WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"`) and use absolute paths for the BYOP pip installs, configs, and logs. This makes the script fully robust regardless of where it is executed from.

### 2026-04-11 20:22:00 — Bug-Fix Pass: BYOP Packages & Sweep Script
**Files Modified**:
- `lehome_workspace/lerobot_policy_dino/src/lerobot_policy_dino/configuration_dino_diffusion.py`
- `lehome_workspace/lerobot_policy_dino/src/lerobot_policy_dino/modeling_dino_diffusion.py`
- `lehome_workspace/lerobot_policy_clip/src/lerobot_policy_clip/configuration_clip_diffusion.py`
- `lehome_workspace/lerobot_policy_clip/src/lerobot_policy_clip/modeling_clip_diffusion.py`
- `lehome_workspace/run_10k_sweep.sh` (Updated: 2026-04-12 15:40:00)
- `lerobot_train_with_plugins.py` (Updated: 2026-04-12 13:42:00)
- `configs/sweep_dino.yaml` (Updated: 2026-04-12 10:40:00)
- `configs/sweep_resnet18.yaml` (Updated: 2026-04-12 10:40:00)
- `configs/sweep_clip.yaml` (Updated: 2026-04-12 10:40:00)

**Files Added**:
- `lehome_workspace/configs/sweep_dino.yaml`
- `lehome_workspace/configs/sweep_clip.yaml`
- `lehome_workspace/configs/sweep_resnet18.yaml`

**Description**:
Independent audit against live LeRobot v0.4.3 API (via DeepWiki + Exa code search) revealed 7 critical bugs. All fixed:
- **BUG-1**: `DiffusionConfig.__post_init__` rejects non-ResNet backbone names. Fixed by overriding `__post_init__` in both config subclasses to call `PreTrainedConfig.__post_init__` directly and replicate only the non-ResNet validation checks.
- **BUG-2**: `config.noise_scheduler_kwargs` does not exist on `DiffusionConfig`. Fixed by replacing `**config.noise_scheduler_kwargs` with explicit kwargs matching the real `DiffusionModel` source.
- **BUG-3**: Wrong `Normalize`/`Unnormalize` import path. Fixed to `from lerobot.policies.normalize import Normalize, Unnormalize`.
- **BUG-6**: `config.num_inference_steps` can be `None` by default. Fixed by storing `self.num_inference_steps` with a None-safe fallback to `num_train_timesteps`.
- **BUG-7 (3 sub-bugs)**: Sweep script used wrong CLI: `python -m lerobot.scripts.train` → `lerobot-train`; `--config=` → `--config_path=`; `--eval.eval_freq=` → `--eval_freq=` (top-level field).
- **Sweep Enhancements**: Added `--dataset.image_transforms.enable=true`, `--num_workers=4`, and extensive logging out to a standardized `logs/` directory using `tee` as requested.
- **MOD-2**: DINOv2 images now resized to 224×224 before backbone to avoid extreme memory use from 1565 patch tokens.
- **MOD-3 (CRITICAL)**: CLIP ViT-B/16 has fixed positional embeddings for 196 patches. 480×640 input caused shape mismatch crash. Images now resized to 224×224 before CLIP backbone.
- **MOD-1**: Typo `horror` → `horizon` corrected in both modeling files.
- Per-backbone YAML configs created (`sweep_dino.yaml`, `sweep_clip.yaml`, `sweep_resnet18.yaml`) so each `lerobot-train --config_path=` call is self-contained.


### 2026-04-11 11:51:00
**Files Added**: 
- `lehome_workspace/time_constrained_sweep.py`

**Description**:
~~Developed a standalone wrapper for `lerobot-train` that implements a wall-clock budget (based on Karpathy's `autoresearch`). It monitors the training subprocess and calculates "Success Rate per GPU Minute".~~ 
*Retracted: Wrapper approach abandoned in favor of internal 10k fixed-step sweep to better predict 150k convergence.*

### 2026-04-11 14:55:00
**Files Added**:
- `lehome_workspace/lerobot_policy_dino/`
- `lehome_workspace/lerobot_policy_clip/`
- `lehome_workspace/run_10k_sweep.sh`
**Files Deleted**:
- `lehome_workspace/time_constrained_sweep.py`

**Description**:
Implemented the "10k Micro-Sweep" infrastructure. Created independent BYOP packages for DINOv2 (Small) and CLIP (ViT-B/16) allowing them to be used directly with `lerobot-train`. Added a sequential bash orchestrator that runs four 10k-step training runs (ResNet18-ImageNet, DINOv2, CLIP, ResNet18-Random) to evaluate representation quality on the A4000.


**Code (time_constrained_sweep.py)**:
```python
import subprocess
import time
import os
import signal
import sys
from pathlib import Path

# ... (full implementation in file) ...
```

**Files Added (Rules)**:
- `.agent/rules/lehome_workspace_tracking.md`

**Description**:
Added workspace-level rule to mandate mirrored logging of all repository modifications to ensure state can be reconstructed by other LLMs on a VM.

### 2026-04-11 15:19:36
**Goal**: Comprehensive grounding of workspace modifications for VM migration.

**1. Core Repository Patches (lehome_workspace/lehome-challenge/)**

**File**: `configs/train_dp.yaml`
**Description**: Disabled AMP (caused CPU bottleneck) and adjusted training budget to 150k steps.
```diff
 policy:
   type: diffusion
+  use_amp: false
   device: cuda
...
 output_dir: outputs/train/dp_top_long
-batch_size: 16
-steps: 30000
-save_freq: 5000
+batch_size: 8
+steps: 150000
+save_freq: 20000
```

**Files**: `scripts/utils/eval_utils.py`, `scripts/utils/evaluation.py`, `scripts/utils/parser.py`, `scripts/eval_policy/__init__.py`
**Description**: Implemented `ClassifierRouterPolicy` support, patched video overwriting BUG in eval, and added `--custom_list_path` for parallel evaluation.
```python
# Key Patch: Prevent video overwriting in eval_utils.py
prefix = f"{garment_name}_" if garment_name else ""
out_path = os.path.join(target_dir, f"{prefix}episode{episode_idx}_{key.replace('.', '_')}.mp4")
```

**2. New Scripts & Utilities**

**File**: `lehome_workspace/lehome-challenge/scripts/eval_policy/classifier_router_policy.py`
**Code**:
```python
# ... (Implementation of ClassifierRouterPolicy using 1st-frame ResNet18 classification) ...
```

**File**: `lehome_workspace/lehome-challenge/parallel_eval.sh`
**Code**:
```bash
# Launches 4 background eval processes using --garment_type
python -m scripts.eval --garment_type top_long ... &
python -m scripts.eval --garment_type top_short ... &
# ...
```

**File**: `lehome_workspace/lehome-challenge/visualize_dataset.sh`
**Code**:
```bash
# Unified tool for viz (Local/Rerun) and replay (VM/IsaacSim)
lerobot-dataset-viz --root "$ROOT_DIR" --repo-id "$DATASET_ID"
```

**3. BYOP Package Pointers**

**Directories**: 
- `lehome_workspace/lerobot_policy_dino/`
- `lehome_workspace/lerobot_policy_clip/`

**Description**: These contain the `pyproject.toml` and `src/` for custom LeRobot policies. They are to be uploaded to the VM separately via SCP/direct transfer. The `run_10k_sweep.sh` handles the `pip install -e` for these packages.
