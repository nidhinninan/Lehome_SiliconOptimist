# LeHome Challenge — Principia VM Installation Guide

## Why This Strategy

Your VM has **29GB free on `/`** and **93GB free on `/data`**. The LeHome challenge requires `isaacsim==5.1.0` (~15GB), `torch==2.7.0` (CUDA 12.8), large Hugging Face datasets, and model checkpoints. The official setup uses `uv` to resolve strict custom indices (NVIDIA, PyTorch). We follow it exactly, but redirect **all** caches and workspaces to `/data`.

> [!IMPORTANT]
> Your VM must have a GPU driver supporting **CUDA 12.8** (required by Isaac Sim 5.1.0 and PyTorch 2.7.0). Principia VMs with Isaac Sim 5.1.0-rc.19 pre-installed should already satisfy this.

---

## Phase 0 — Hugging Face Authentication (CRITICAL)

To avoid the **429 Too Many Requests** error when downloading large datasets (especially from cloud IPs), you **must** authenticate with Hugging Face.

1.  **Create an account** at [huggingface.co](https://huggingface.co/join) if you haven't.
2.  **Generate a Token**: Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a new **Read** token.
3.  **Login on the VM**:
    ```bash
    source .venv/bin/activate  # Ensure your environment is active
    huggingface-cli login
    ```
    *Paste your token when prompted (it will be invisible as you type).*

---

## Phase 1 — Storage & Environment Setup

### 1.1 Redirect Caches to `/data`
```bash
mkdir -p /data/lehome_workspace /data/huggingface_cache /data/uv_cache

cat >> ~/.bashrc << 'EOF'
export HF_HOME="/data/huggingface_cache"
export UV_CACHE_DIR="/data/uv_cache"
export __GLX_VENDOR_LIBRARY_NAME=nvidia
EOF
source ~/.bashrc
```

### 1.2 Install Server System Dependencies (headless rendering & utilities)
```bash
sudo apt update && sudo apt install -y \
    libglu1-mesa libgl1 libegl1 libxrandr2 \
    libxinerama1 libxcursor1 libxi6 libxext6 libx11-6 \
    zip
```

### 1.3 Install `uv`
```bash
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/data/.local/bin" sh
echo 'export PATH="/data/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

---

## Phase 2 — Clone & Install

### 2.1 Clone Repositories
```bash
cd /data/lehome_workspace
git clone https://github.com/lehome-official/lehome-challenge.git
cd lehome-challenge

mkdir -p third_party && cd third_party
git clone https://github.com/lehome-official/IsaacLab.git
cd ..
```

### 2.2 Install All Dependencies
```bash
# Downloads isaacsim, torch, lerobot, etc. into .venv — all cached on /data
uv sync
```

### 2.3 Install IsaacLab Packages & LeHome
The `isaaclab.sh` script relies on global Python aliases on the Principia VM, which bypasses your `.venv`. We must manually install its dependencies and sub-packages using `uv` with build-isolation disabled (to prevent `flatdict` compilation failures):

```bash
source .venv/bin/activate

# 1. Install core PyPI build tools to prevent dependency crashes later
uv pip install setuptools flatdict pyarrow prettytable

# 2. Install all IsaacLab sub-packages directly into the `.venv`
uv pip install -e ./third_party/IsaacLab/source/isaaclab --no-build-isolation
uv pip install -e ./third_party/IsaacLab/source/isaaclab_assets --no-build-isolation
uv pip install -e ./third_party/IsaacLab/source/isaaclab_tasks --no-build-isolation

# 3. Install the specific LeHome project codebase
uv pip install -e ./source/lehome
```

---

## Phase 3 — Verify Installation

```bash
source .venv/bin/activate
cd third_party/IsaacLab
./isaaclab.sh -h        # Should print help without errors
cd ../..
```

---

## Phase 4 — Download Datasets

These commands download from Hugging Face into subdirectories of your project (on `/data`).

```bash
cd /data/lehome_workspace/lehome-challenge
source .venv/bin/activate

# Simulation assets (garments, robot URDFs, scenes)
huggingface-cli download lehome/asset_challenge --repo-type dataset --local-dir Assets

# Pre-collected demonstration data (4 garment types, merged)
huggingface-cli download lehome/dataset_challenge_merged --repo-type dataset --local-dir Datasets/example
```

---

## Phase 5 — Train & Evaluate Baselines

### 5.1 Action Chunking with Transformers (ACT)
```bash
cd /data/lehome_workspace/lehome-challenge
source .venv/bin/activate
lerobot-train --config_path=configs/train_act.yaml
```

### 5.2 Diffusion Policy (Parallel Execution — Recommended)
For DP, it is highly recommended to train separate models for each garment type. Use the following overrides for each instance:

```bash
cd /data/lehome_workspace/lehome-challenge
source .venv/bin/activate

# Top Long (uses defaults in configs/train_dp.yaml)
lerobot-train --config_path=configs/train_dp.yaml

# Top Short (manually override output and dataset)
lerobot-train --config_path=configs/train_dp.yaml \
  --dataset.root=Datasets/example/top_short_merged \
  --output_dir=outputs/train/dp_top_short

# Pant Long...
lerobot-train --config_path=configs/train_dp.yaml \
  --dataset.root=Datasets/example/pant_long_merged \
  --output_dir=outputs/train/dp_pant_long

# Pant Short...
lerobot-train --config_path=configs/train_dp.yaml \
  --dataset.root=Datasets/example/pant_short_merged \
  --output_dir=outputs/train/dp_pant_short
```

> [!WARNING]
> **Avoid Global `lerobot-train` Aliases**
> Principia VMs often have a newer version of LeRobot pre-installed in `~/.local/bin/`. Running `lerobot-train` directly may load this incompatible global version (causing `ModuleNotFoundError` for `lerobot.policies.normalize`). 
> **Always use the absolute venv path:** `/data/lehome_workspace/lehome-challenge/.venv/bin/python -m lerobot.scripts.lerobot_train`

> [!WARNING]
> **Troubleshooting `Bus error` (Insufficient Shared Memory)**
> If you get a `DataLoader worker is killed by signal: Bus error` when training starts, your VM container's `/dev/shm` limit is too small (often 64MB default). 
> **Fix:** Run `sudo mount -o remount,size=2G /dev/shm` to expand it to 2GB. If you lack sudo privileges, append `--num_workers=0` to the training command instead.

### 5.2 Evaluate Trained Policies
You can evaluate your models purely for numerical metrics, or render them to MP4 videos to visually see the robot in action!

> [!IMPORTANT]
> **Diffusion Policy (DP) is vision-based. You MUST include `--enable_cameras` even in `--headless` mode.**
> Without this flag, the rendering pipeline is disabled, and your robot will receive blank images, leading to task failure.

> [!TIP]
> **If evaluation success is low:** Consider resuming training with augmentations for the final steps (e.g. 80K → 150K). 
> **CMD:** `/data/lehome_workspace/lehome-challenge/.venv/bin/python -m lerobot.scripts.lerobot_train --config_path=outputs/train/dp_top_long/checkpoints/last/pretrained_model/train_config.json --resume=true --dataset.image_transforms.enable=true` (See Section 6.1 for details).

**Option A: Quick Evaluation (No Video)**
```bash
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/act_top_long/checkpoints/last/pretrained_model \
    --garment_type "top_long" \
    --dataset_root Datasets/example/top_long_merged \
    --num_episodes 1 \
    --headless \
    --enable_cameras \
    --device cpu
```

**Option B: Visual Evaluation (Saves MP4 Videos)**
```bash
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/act_top_long/checkpoints/last/pretrained_model \
    --dataset_root Datasets/example/top_long_merged \
    --garment_type "top_long" \
    --num_episodes 2 \
    --max_steps 600 \
    --save_video \
    --enable_cameras \
    --video_dir outputs/eval_videos/act_top_long \
    --headless \
    --device cpu
```

---

---

## Troubleshooting & Environment Conflicts (Principia VM)

### 1. The "Python Alias" Conflict
**Symptoms:** `ModuleNotFoundError: No module named 'isaaclab'` (or `pyarrow`) even when you think you are "activated."
**The Cause:** Principia VMs use a global alias `python -> ~/python.sh` which ignores your `.venv` and runs the wrong Isaac Sim environment.
**The Fix:** Always use the **absolute path** to your project's python to bypass all aliases:
```bash
/data/lehome_workspace/lehome-challenge/.venv/bin/python -m scripts.eval ...
```

### 2. Missing `pyarrow`, `isaaclab`, or internal extensions
If `uv sync` missed these or `isaaclab` extensions fail to load on startup, you must force-install them. First, install the core build tools to prevent `flatdict` and Parquet errors, then explicitly install the IsaacLab sub-packages so `uv` automatically resolves their downstream dependencies (like `torchvision`, `prettytable`, `numba`, etc.):
```bash
# 1. Fix missing data readers and build-blockers
uv pip install pyarrow setuptools flatdict

# 2. Install the IsaacLab sub-packages with build isolation disabled to allow flatdict to compile
uv pip install -e /data/lehome_workspace/lehome-challenge/third_party/IsaacLab/source/isaaclab --no-build-isolation
uv pip install -e /data/lehome_workspace/lehome-challenge/third_party/IsaacLab/source/isaaclab_assets --no-build-isolation
uv pip install -e /data/lehome_workspace/lehome-challenge/third_party/IsaacLab/source/isaaclab_tasks --no-build-isolation
```

### 3. Missing Dependencies for Diffusion Policy (DP) Evaluation
**Symptoms:** `ModuleNotFoundError: No module named 'omegaconf'` or `No module named 'h5py'` when evaluating a DP checkpoint (ACT may have worked fine).
**The Cause:** The Diffusion Policy code path imports additional libraries (`omegaconf`, `h5py`, `diffusers`, etc.) that ACT never touches. `uv sync` may not have installed all transitive dependencies of `lerobot`.
**The Fix:**
```bash
uv pip install omegaconf h5py diffusers draccus opencv-python-headless einops imageio[ffmpeg]
```

> [!WARNING]
> **Do NOT run `uv sync` after manually installing packages.** `uv sync` enforces a strict lockfile and will **uninstall** anything not in `uv.lock`, including your manually added IsaacLab, LeHome, and DP dependencies. If you accidentally run `uv sync`, re-run the full restore sequence from Section 2.3 above, then this step.

### 4. Sanity Check — Verify All Critical Imports
Run this one-liner to confirm every key module is importable before starting an evaluation:
```bash
python -c "import omegaconf, h5py, diffusers, draccus, einops, cv2, isaaclab, lehome; print('All imports OK')"
```
If it prints `All imports OK`, your environment is ready.

### 5. PhysX `attachShape` Warning During Garment Simulation
**Symptoms:** `PhysX error: attachShape: non-SDF triangle mesh... not supported for non-kinematic PxRigidDynamic instances.`
**Impact:** **None — this is safe to ignore.** It's a known Isaac Sim warning for cloth/garment objects. Garments use soft-body particle simulation, not rigid-body collision meshes, so this warning does not affect evaluation results.

### 6. `sh: 1: zenity: not found` Warning
**Symptoms:** This message appears repeatedly when running evals in `--headless` mode.
**The Cause:** Isaac Sim tries to use the `zenity` tool to display GUI dialogs (like end-user license agreements or errors). Since the VM is headless, `zenity` is not installed.
**The Fix:** **Ignore it.** It is purely cosmetic and does not affect your evaluation speed or results.

---

## Phase 6 — Advanced Execution (Use Absolute Paths)

### 6.0 Essential: Fixing the Video Overwrite Bug
Evaluations on a full garment list (e.g. `top_short_merged`) may overwrite videos because the episode index resets to 0. Run the provided patch script on your VM:
```bash
bash patch_video_bug.sh
```
This ensures videos are prefixed with the garment name (e.g. `Top_Short_1_episode0_...`).

### 6.1 Resuming a Paused Training Run
If you stop the training manually (or it crashes) and you want to resume exactly where it left off, point `--config_path` directly to the `train_config.json` inside your latest checkpoint folder, and add `--resume=true`.

**Standard Resume:**
```bash
/data/lehome_workspace/lehome-challenge/.venv/bin/python \
  -m lerobot.scripts.lerobot_train \
  --config_path=outputs/train/act_top_long/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

**Resume with Image Augmentations (Recommended for final phase):**
To enable augmentations during a resume (e.g., from 80K/93K to 150K):
```bash
/data/lehome_workspace/lehome-challenge/.venv/bin/python \
  -m lerobot.scripts.lerobot_train \
  --config_path=outputs/train/act_top_long/checkpoints/last/pretrained_model/train_config.json \
  --resume=true \
  --dataset.image_transforms.enable=true
```
OR save log as well and set num_workers;
```
/data/lehome_workspace/lehome-challenge/.venv/bin/python   -m lerobot.scripts.lerobot_train   --config_path=outputs/train/dp_top_short/checkpoints/last/pretrained_model/train_config.json   --resume=true   --dataset.image_transforms.enable=true   --num_workers 4   2>&1 | tee /data/lehome_workspace/lehome-challenge/logs/readout/dp_top_short_resume_aug_120k-4_CPUworkers.log

```

### 6.2 Evaluating a Checkpoint & Saving a Video

**ACT Policy:**
```bash
/data/lehome_workspace/lehome-challenge/.venv/bin/python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/act_top_long/checkpoints/last/pretrained_model \
    --dataset_root Datasets/example/top_long_merged \
    --garment_type "top_long" \
    --num_episodes 5 \
    --max_steps 600 \
    --save_video \
    --enable_cameras \
    --video_dir outputs/eval_videos/act_top_long \
    --device cpu
```

**Diffusion Policy (DP):**
```bash
/data/lehome_workspace/lehome-challenge/.venv/bin/python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/dp_top_long/checkpoints/last/pretrained_model \
    --dataset_root Datasets/example/top_long_merged \
    --garment_type "top_long" \
    --num_episodes 5 \
    --max_steps 600 \
    --save_video \
    --enable_cameras \
    --video_dir outputs/eval_videos/dp_top_long \
    --device cpu
```

## Estimated Storage Budget on `/data` (93GB available)

| Item                                     | Estimated Size |
| ---------------------------------------- | -------------- |
| `uv` cache (isaacsim wheel, torch, etc.) | ~15–20 GB      |
| `.venv` (installed packages)             | ~8–12 GB       |
| `Assets/` (simulation assets)            | ~1–5 GB        |
| `Datasets/example/` (demo data)          | ~5–15 GB       |
| Training checkpoints (per policy)        | ~1–3 GB        |
| **Total estimate**                       | **~30–55 GB**  |

> [!NOTE]
> These are estimates. The 93GB on `/data` should be sufficient for the full pipeline. You can reclaim space later by clearing the uv cache with `uv cache clean`.
