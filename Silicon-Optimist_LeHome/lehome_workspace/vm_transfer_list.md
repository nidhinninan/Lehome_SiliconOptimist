# LeHome VM Transfer List

**Last Updated**: 2026-04-30 22:10:00 UTC

The following files must be copied to your VM machine to match the current local research state.

## 1. Patched Files (Modified in `lehome-challenge`)
These files should overwrite the existing ones in your `lehome-challenge` clone.

- `source/lehome/setup.py` (Updated: 2026-04-29 20:15:00 UTC — editable install: use stdlib `tomllib` on Python 3.11+ so isolated builds do not require PyPI `toml`; fixes `ModuleNotFoundError: No module named 'toml'` during `uv pip install -e source/lehome`)
- `scripts/eval_policy/__init__.py` (Updated: 2026-04-11 15:19:36)
- `scripts/utils/eval_utils.py` (Updated: 2026-04-11 15:19:36)
- `scripts/utils/evaluation.py` (Updated: 2026-04-11 15:19:36)
- `scripts/utils/parser.py` (Updated: 2026-04-11 15:19:36)
- `configs/train_dp.yaml` (Updated: 2026-04-11 15:19:36)

## 2. New Utilities (Added to `lehome-challenge`)
- `dummy_docker_policy/build_docker_submission.sh` (Updated: 2026-04-30 22:10:00 UTC — auto `requirements.txt` from template + `rsync` stage `lerobot_policy_dino` from `../../lerobot_policy_dino` when missing; `PRETRAINED_PATH` before download; W&B + docker build + optional HF push; `./build_docker_submission.sh --help`)
- `dummy_docker_policy/requirements.txt` (Updated: 2026-04-30 22:10:00 UTC — same pins as `requirements.submission.template` for `COPY requirements.txt` in Docker)
- `dummy_docker_policy/Dockerfile.submission` (Updated: 2026-04-30 22:10:00 UTC — comment: staging via build script / plain build context)
- `dummy_docker_policy/server.py` (Updated: 2026-04-30 18:00:00 UTC — upstream LeHome policy HTTP server; keep unmodified)
- `dummy_docker_policy/requirements.submission.template` (Updated: 2026-04-30 18:00:00 UTC — seed for `requirements.txt`; add `lerobot` pins to match training)
- `dummy_docker_policy/README.md`, `dummy_docker_policy/Dockerfile` (Updated: 2026-04-30 22:10:00 UTC — add-only merge from `lehome-official/lehome-challenge` `main`; official minimal Docker sample; submission flow uses `Dockerfile.submission`)
- `dummy_docker_policy/Dockerfile.submission`, `dummy_docker_policy/download_wandb_model.py`, `dummy_docker_policy/policy.py` — keep in sync with VM for Docker image bake (Updated: 2026-04-30 18:50:00 UTC — Reconstructed DINOv2 pre-cache and BYOP install in Dockerfile; policy.py implementation; W&B artifact ref parsing hardened)
- `.gitignore` (Updated: 2026-04-30 22:10:00 UTC — ignore `dummy_docker_policy/lerobot_policy_dino/` staged copy)
- `scripts/eval_policy/classifier_router_policy.py` (Updated: 2026-04-11 15:19:36)
- `scripts/garment_classifier/sanity_check_first_frames.py` (New: 2026-04-12 02:40:00)
- `scripts/garment_classifier/export_garment_classifier_dataset.py` (New: 2026-04-12 02:40:00)
- `scripts/garment_classifier/train_garment_classifier.py` (New: 2026-04-12 02:40:00)
- `parallel_eval.sh` (Updated: 2026-04-11 15:19:36)
- `visualize_dataset.sh` (Updated: 2026-04-11 15:19:36)

## 3. External Infrastructure (Added to `lehome_workspace/`)
These files reside outside the `lehome-challenge` repository.

- `run_train_dino_merged_a100.sh` (Updated: 2026-04-28 23:05:00 UTC — Principia: `/data` caches + PATH only; **A100** tiers unchanged (`A100_TIER:-normal`); no `principia_*` presets; checkpoint skip + optional `INCLUDE_LEGACY_VM_LAST_PATHS`)
- `parallel_eval_merged_strong.sh` (Updated: 2026-04-28 23:05:00 UTC — Principia: `HF_HOME`, PATH, optional `CHALLENGE_DIR`; default **`POLICY_PATH`** = A100 merged `..._a100_400k_norm/...`)
- `setup_lehome.sh` (Updated: 2026-04-29 12:00:00 UTC — Principia: `WORKSPACE_DIR=/data/lehome_workspace`, `HF_HOME`/`UV_CACHE_DIR`/`PATH` under `/data`, `uv` in `/data/.local/bin`, `chown principia:principia` on workspace + caches + uv bin dir)
- `run_train_optimized.sh` (New: 2026-04-12 12:51:00)
- `run_train_dino_dp_top_short_150k.sh` (Updated: 2026-04-28 23:05:00 UTC — **`TOP_SHORT_GPU_PROFILE`** `4070ti|a100`: on Principia `/data` paths + `4070ti`, default batch 12 / workers 10 / output `_4070ti`; set `a100` for non–4070-sized runs; caches + skip-last + rclone unchanged)
- `configs/sweep_dino_map_registers.yaml` (Updated: 2026-04-28 12:30:00 UTC — added explicit `eval:` section with safe defaults)
- `run_10k_sweep.sh` (Updated: 2026-04-12 15:40:00)
- `run_2run_dino_sweep.sh` (Updated: 2026-04-27 22:57:13 UTC — MAP profile: `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS` default 1 before train; optional rclone + `.complete` skip; `WORKSPACE_DIR` from `BASH_SOURCE`)
- `run_2run_dino_sweep_no-gdrive.sh` (Updated: 2026-04-27 22:57:13 UTC — same thread exports before train; `WORKSPACE_DIR=$(pwd)`; no rclone)
- `configs/sweep_dino_baseline.yaml` (New: 2026-04-26 12:00:00)
- `configs/sweep_dino_map_registers.yaml` (Updated: 2026-04-27 22:57:13 UTC — `policy.use_amp: true`, `dataset.video_backend: torchcodec`, top-level `num_workers: 12`, comments for FFmpeg/torchcodec fallback)
- `.env` (VM): optional; set `WANDB_API_UCMO` then script exports `WANDB_API_KEY` (Updated: 2026-04-26 12:00:00)
- `lerobot_train_with_plugins.py` (Updated: 2026-04-12 13:42:00)
- `configs/sweep_dino.yaml` (Updated: 2026-04-12 10:40:00)
- `configs/sweep_resnet18.yaml` (Updated: 2026-04-12 10:40:00)
- `configs/sweep_clip.yaml` (Updated: 2026-04-12 10:40:00)

### DINOv2 BYOP Package (`lehome_workspace/lerobot_policy_dino/`)
- `pyproject.toml` (Updated: 2026-04-12 13:42:00)
- `src/lerobot_policy_dino/__init__.py` (Updated: 2026-04-11 15:19:36)
- `src/lerobot_policy_dino/modeling_dino_diffusion.py` (Updated: 2026-04-26 12:00:00)
- `src/lerobot_policy_dino/configuration_dino_diffusion.py` (Updated: 2026-04-26 12:00:00)
- `src/lerobot_policy_dino/processor_dino_diffusion.py` (Updated: 2026-04-11 15:19:36)

### CLIP BYOP Package (`lehome_workspace/lerobot_policy_clip/`)
- `pyproject.toml` (Updated: 2026-04-12 13:42:00)
- `src/lerobot_policy_clip/__init__.py` (Updated: 2026-04-11 15:19:36)
- `src/lerobot_policy_clip/modeling_clip_diffusion.py` (Updated: 2026-04-12 14:02:00)
- `src/lerobot_policy_clip/configuration_clip_diffusion.py` (Updated: 2026-04-11 20:22:00)
- `src/lerobot_policy_clip/processor_clip_diffusion.py` (Updated: 2026-04-11 15:19:36)
