# LeHome VM Transfer List

**Last Updated**: 2026-04-28 12:30:00 UTC

The following files must be copied to your VM machine to match the current local research state.

## 1. Patched Files (Modified in `lehome-challenge`)
These files should overwrite the existing ones in your `lehome-challenge` clone.

- `scripts/eval_policy/__init__.py` (Updated: 2026-04-11 15:19:36)
- `scripts/utils/eval_utils.py` (Updated: 2026-04-11 15:19:36)
- `scripts/utils/evaluation.py` (Updated: 2026-04-11 15:19:36)
- `scripts/utils/parser.py` (Updated: 2026-04-11 15:19:36)
- `configs/train_dp.yaml` (Updated: 2026-04-11 15:19:36)

## 2. New Utilities (Added to `lehome-challenge`)
- `scripts/eval_policy/classifier_router_policy.py` (Updated: 2026-04-11 15:19:36)
- `scripts/garment_classifier/sanity_check_first_frames.py` (New: 2026-04-12 02:40:00)
- `scripts/garment_classifier/export_garment_classifier_dataset.py` (New: 2026-04-12 02:40:00)
- `scripts/garment_classifier/train_garment_classifier.py` (New: 2026-04-12 02:40:00)
- `parallel_eval.sh` (Updated: 2026-04-11 15:19:36)
- `visualize_dataset.sh` (Updated: 2026-04-11 15:19:36)

## 3. External Infrastructure (Added to `lehome_workspace/`)
These files reside outside the `lehome-challenge` repository.

- `setup_lehome.sh` (Updated: 2026-04-26 20:05:00 — added `ffmpeg` to apt install for torchcodec support)
- `run_train_optimized.sh` (New: 2026-04-12 12:51:00)
- `run_train_dino_dp_top_short_150k.sh` (Updated: 2026-04-28 12:30:00 UTC — balanced eval: freq=20k, n_episodes=4, batch_size=2 for learning visibility without OOM; added comments)
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
