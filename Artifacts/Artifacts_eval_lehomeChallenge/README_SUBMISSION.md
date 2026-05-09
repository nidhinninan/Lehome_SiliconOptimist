# LeHome Challenge 2026 — Submission README
### Team: Silicon Optimists | Branch: `principia_instance`

**Repository:** [nidhinninan/Lehome\_SiliconOptimist @ principia\_instance](https://github.com/nidhinninan/Lehome_SiliconOptimist/tree/principia_instance)

> **Note to evaluators:** The `principia_instance` branch is the **latest and active submission branch**. All policies, scripts and artifacts referenced below correspond to this branch.
>
> ⚠️ **CRITICAL UPDATE**: Please read **`EVALUATION_README_FINAL.md`** first. It contains instructions for a **micro-rebuild** of the Docker image to fix a critical normalization bug, and applies necessary host-side patches for Isaac Sim 5.1 compatibility. You must follow the steps in `EVALUATION_README_FINAL.md` to run the evaluation correctly.

---

## Architecture — What We Built

This submission uses a custom **Diffusion Policy (DP)** architecture that replaces LeRobot's default ResNet backbone with a **frozen DINOv2 ViT** (`facebook/dinov2-with-registers-small`). The key design decisions are:

### 1. Frozen DINOv2 as a Static Feature Extractor
The backbone is completely frozen — zero gradient updates pass through it. This provides:
- **Decoupling**: The Diffusion U-Net dedicates 100% of its capacity to learning the physics of cloth manipulation, not visual features.
- **Forgetting prevention**: The small imitation learning dataset does not overwrite DINOv2's general understanding of geometry, folds, and shadows.
- **VRAM efficiency**: ~60–70% of memory freed from gradient storage, enabling larger batch sizes.

### 2. Surgical Inheritance to Bypass LeRobot's ResNet Hardcoding
- `@PreTrainedConfig.register_subclass` registers the new architecture.
- `__post_init__` overrides bypass the ResNet-only validation path.
- `get_optim_params()` is overridden to return only `requires_grad == True` parameters, cleanly excluding the frozen ViT.

### 3. Antialiased Downsampling for 480×640 → 224×224
The LeHome dataset provides 480×640 images but DINOv2 was pre-trained at 224×224. `_encode_images()` applies antialiased bilinear interpolation before the backbone forward pass, keeping patch token counts at 256 per camera.

### 4. Multi-headed Attention Pooling (MAP) Head with Register Token Ablation
- `dinov2-with-registers-small` is used specifically; the 4 register tokens are sliced out before pooling to prevent "scratchpad" artifact tokens from corrupting the spatial readout.
- **K=8 learnable query vectors** cross-attend to the 256 patch tokens, giving spatially-specialized summaries (e.g., per-query focus on gripper, cloth boundary, fold).
- The 8×384 output is flattened per camera into the `global_cond` vector for the Diffusion U-Net.

---

## Docker Image

```
nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell
```

The image is hosted on Docker Hub under a private repository. Access credentials for evaluation are provided below.

---

## Docker Credentials — Quick Reference

| | |
|---|---|
| **Username** | `nninspaceexp` |
| **Access token** | `<REDACTED_DOCKER_PAT>` |
| **Scope** | Read-only (pull only) |
| **Expires** | Jul 30, 2026 |

```bash
# One-liner login + pull (copy-paste ready)
docker login -u nninspaceexp --password-stdin <<< "<REDACTED_DOCKER_PAT>" && \
docker pull nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell
```

### Docker Hub Access (Read-Only Token)

The Docker Hub image requires authentication to pull. A **read-only Personal Access Token (PAT)** is provided for evaluation purposes only:

| Field | Value |
|---|---|
| **Docker Hub username** | `nninspaceexp` |
| **Access token** | `<REDACTED_DOCKER_PAT>` |
| **Token description** | `LeHome_Challenge-Organisers` |
| **Token scope** | `Read-only` — can **pull** the image; cannot push, delete, or modify anything |
| **Expires** | Jul 30, 2026 at 23:59:59 |

**To authenticate and pull:**
```bash
docker login -u nninspaceexp
# When prompted for password, enter:
# <REDACTED_DOCKER_PAT>

# Or non-interactively:
docker login -u nninspaceexp --password-stdin <<< "<REDACTED_DOCKER_PAT>"
docker pull nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell
```

> **Note to evaluators:** This token is scoped to read-only (`pull` access only) and expires Jul 30, 2026. It cannot be used to modify any repository. The token type used is a Docker Hub **Repository-scoped Access Token** set to `Read-only`.

---

## Step-by-Step Evaluation Instructions

These are the exact commands used on our evaluation machine, in the exact order they were run.

### Prerequisites

- Ubuntu 22.04 (tested)
- NVIDIA GPU with ≥24 GB VRAM (tested on RTX 5090)
- CUDA-compatible driver (CUDA 11.8+)
- `uv` package manager: `pip install uv`
- `docker` with NVIDIA Container Toolkit (`--gpus all` support)
- Internet access for HuggingFace asset download

---

### Step 1 — Clone the Hackathon Evaluation Repository

```bash
git clone https://github.com/lehome-official/lehome-challenge.git
cd lehome-challenge
```

---

### Step 2 — Install Python Dependencies

```bash
uv sync
```

This creates a `.venv/` directory and installs all packages defined in `pyproject.toml`.

---

### Step 3 — Clone and Install IsaacLab

```bash
# Clone IsaacLab into the third_party directory
cd third_party
git clone https://github.com/lehome-official/IsaacLab.git
cd ..

# Activate the venv, then install IsaacLab extensions
source .venv/bin/activate
./third_party/IsaacLab/isaaclab.sh -i none
```

When prompted about the NVIDIA EULA, type `yes` and press Enter.

---

### Step 4 — Install the LeHome Package

```bash
# Still with venv active from Step 3
uv pip install -e ./source/lehome
```

---

### Step 5 — Download Simulation Assets from HuggingFace

```bash
hf download lehome/asset_challenge --repo-type dataset --local-dir Assets
```

This downloads all garment meshes, robot USDs, and scene files into the `Assets/` directory. It may take several minutes. You will see progress bars per file.

---

### Step 6 — Install Xvfb (Virtual Display — Required for Headless Servers)

Isaac Sim's `pynput` keyboard listener requires an active X11 display even in `--headless` mode. On a server without a physical display, this causes:
```
ImportError: this platform is not supported: failed to acquire X connection
```

The fix is a one-time system-level install of `xvfb`:
```bash
sudo apt-get update && sudo apt-get install -y xvfb
```

`xvfb-run -a` creates an isolated synthetic X session for each process. The `-a` flag auto-selects a free display port (`:99`, `:100`, etc.), making it safe to run multiple parallel evaluations simultaneously.

---

### Step 7 — Pull the Policy Docker Image

```bash
# Authenticate with the read-only token (copy-paste ready)
docker login -u nninspaceexp --password-stdin <<< "<REDACTED_DOCKER_PAT>"

# Pull the image
docker pull nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell
```

---

### Step 8 — Start the Policy Servers (Docker Containers)

Each Docker container runs the frozen DINOv2 + Diffusion U-Net inference server on GPU. Open two separate terminals:

**Terminal A — Policy Server for Garment Category 1 (port 8081)**
```bash
docker run --rm --gpus all \
    -p 8081:8080 \
    nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell
```

**Terminal B — Policy Server for Garment Category 2 (port 8082)**
```bash
docker run --rm --gpus all \
    -p 8082:8080 \
    nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell
```

**Wait** until both terminals print:
```
Policy server listening on 0.0.0.0:8080
```
before proceeding.

---

### Step 9 — Run Evaluation (Two Garment Types in Parallel)

Return to the `lehome-challenge` directory. Open two more terminals (or use `&` backgrounding). Activate the venv first in each:

```bash
cd /path/to/lehome-challenge
source .venv/bin/activate
```

**Terminal C — Evaluate `top_long` (using policy server on port 8081)**
```bash
xvfb-run -a python -m scripts.eval \
    --policy_type docker \
    --docker_url http://localhost:8081 \
    --garment_type top_long \
    --step_hz 0 \
    --device cpu \
    --headless \
    --enable_cameras \
    > eval_top_long_fast.log 2>&1 &

echo "top_long eval started, PID: $!"
```

**Terminal D — Evaluate `pant_long` (using policy server on port 8082)**
```bash
xvfb-run -a python -m scripts.eval \
    --policy_type docker \
    --docker_url http://localhost:8082 \
    --garment_type pant_long \
    --step_hz 0 \
    --device cpu \
    --headless \
    --enable_cameras \
    > eval_pant_long_fast.log 2>&1 &

echo "pant_long eval started, PID: $!"
```

---

### Step 10 — Evaluate Remaining Categories (After Batch 1 Finishes)

When both Batch 1 evaluations complete, reuse the same two docker containers (they remain running) and fire the remaining garment categories:

**Terminal C (reuse)**
```bash
xvfb-run -a python -m scripts.eval \
    --policy_type docker \
    --docker_url http://localhost:8081 \
    --garment_type top_short \
    --step_hz 0 \
    --device cpu \
    --headless \
    --enable_cameras \
    > eval_top_short_fast.log 2>&1 &
```

**Terminal D (reuse)**
```bash
xvfb-run -a python -m scripts.eval \
    --policy_type docker \
    --docker_url http://localhost:8082 \
    --garment_type pant_short \
    --step_hz 0 \
    --device cpu \
    --headless \
    --enable_cameras \
    > eval_pant_short_fast.log 2>&1 &
```

---

## Key Flag Reference

| Flag | Value | Reason |
|---|---|---|
| `--policy_type docker` | `docker` | Routes inference to the HTTP policy server in the container |
| `--docker_url` | `http://localhost:808x` | Address of the running policy server |
| `--step_hz 0` | `0` | Disables the `RateLimiter` — simulation runs at maximum possible speed |
| `--device cpu` | `cpu` | Keeps PhysX cloth physics on CPU (GPU cloth physics is unstable in Isaac Sim 5.1) |
| `--headless` | flag | Runs without a GUI window |
| `--enable_cameras` | flag | Required — enables camera observations for the vision-based policy |
| `xvfb-run -a` | wrapper | Provides a synthetic X11 display on headless servers |

---

## Parallel Evaluation Capacity

Based on observed resource usage, **exactly 2 simultaneous evaluations** are safe:

| Resource | Per Eval | 2 Evals | Limit |
|---|---|---|---|
| System RAM | ~15 GB | ~30 GB | Machine-dependent |
| GPU VRAM (sim renderer) | ~4.5 GB | ~9 GB | 24 GB+ recommended |
| GPU VRAM (Docker policy) | ~6.6 GB | ~13.2 GB | 24 GB+ recommended |
| **Total VRAM** | **~11 GB** | **~22 GB** | **24 GB minimum** |

A 3rd simultaneous evaluation risks OOM on both RAM and VRAM.

---

## Monitoring Progress

```bash
# Watch live output from top_long evaluation
tail -f eval_top_long_fast.log

# Watch live output from pant_long evaluation
tail -f eval_pant_long_fast.log

# Confirm both eval processes are alive
ps aux | grep scripts.eval | grep -v grep

# Confirm both Docker containers are alive
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'

# Check GPU allocation
nvidia-smi
```

### Expected Log Milestones

| Time | Log Message | Meaning |
|---|---|---|
| ~0s | `Simulation App Starting` | IsaacLab booting |
| ~8s | `app ready` | Sim engine initialized |
| ~9s | `[DockerPolicy] Connected to http://localhost:808x` | Policy server confirmed reachable |
| ~9s | `Loaded 12 garments for category: ...` | Asset list parsed |
| ~30s | `Simulation App Startup Complete` | Full sim ready |
| ongoing | `Episode X/5: Return=..., Length=..., Success=...` | Active evaluation in progress |

### Harmless Warnings to Ignore

```
[Error] [omni.physx.plugin] PhysX error: attachShape: non-SDF triangle mesh...
```
Expected PhysX cloth mesh warning — does not affect evaluation.

```
** (zenity:XXXXX): WARNING **: AT-SPI: Could not obtain desktop path or name
```
Expected xvfb accessibility warning — harmless.

---

## About the Docker Access Token

We created a **Docker Hub Repository-scoped Personal Access Token** with the following properties:

- **Token:** `<REDACTED_DOCKER_PAT>`
- **Description:** `LeHome_Challenge-Organisers`
- **Type:** Repository-scoped PAT (not a full account PAT)
- **Permissions:** `Read-only` — allows `docker pull` only
- **Cannot:** push, delete tags, create repositories, or access account settings
- **Expiry:** Jul 30, 2026 at 23:59:59

This is the minimum viable credential for evaluation. It poses no security risk beyond read access to the image itself.

To generate this type of token yourself for future reference:
> Docker Hub → Account Settings → Security → Personal Access Tokens → Generate New Token → Access permissions: **Read-only** → (optionally) restrict to specific repository

---

*LeHome Challenge 2026 — Silicon Optimists*
*Model: `FullDP-v2_Blackwell` | Branch: `principia_instance`*
