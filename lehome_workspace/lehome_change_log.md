# LeHome Workspace Change Log

> [!IMPORTANT]
> **VM Transfer List**: All files required for replication on the VM are listed in [vm_transfer_list.md](./vm_transfer_list.md).
> **Updated Files**: ALL the latest iteration of the files can be found in [./updated_files/](./updated_files/) folder.

This document and the `updated_files` folder track all manual and agentic modifications to the `lehome_workspace/` repository clones. It serves as a grounded reference for replicating state on remote VMs.

---
<!-- LOG_START -->

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
