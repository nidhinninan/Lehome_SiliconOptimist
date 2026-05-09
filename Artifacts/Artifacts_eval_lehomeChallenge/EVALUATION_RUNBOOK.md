# LeHome Challenge 2026 — Evaluation Runbook
### Silicon Optimists — `principia_instance` Branch

> **Repository:** [nidhinninan/Lehome\_SiliconOptimist @ principia\_instance](https://github.com/nidhinninan/Lehome_SiliconOptimist/tree/principia_instance)
>
> The **`principia_instance`** branch is the latest and primary branch for this submission. All instructions and model artifacts referenced below correspond to this branch.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [System Requirements](#2-system-requirements)
3. [Pulling and Running the Policy Docker Images](#3-pulling-and-running-the-policy-docker-images)
4. [Host Environment Setup (One-Time)](#4-host-environment-setup-one-time)
5. [Running Evaluation](#5-running-evaluation)
6. [Parallel Evaluation Strategy](#6-parallel-evaluation-strategy)
7. [Monitoring and Interpreting Logs](#7-monitoring-and-interpreting-logs)
8. [Resource Budget at a Glance](#8-resource-budget-at-a-glance)

---

## 1. Architecture Overview

This section documents the key design decisions behind the submitted Diffusion Policy (DP) model — specifically the departure from the standard LeRobot ResNet backbone in favour of a frozen DINOv2 ViT.

### 1.1 The "Frozen Advantage" Strategy

Instead of fine-tuning a vision backbone end-to-end, DINOv2 (`facebook/dinov2-with-registers-small`) was **completely frozen** and used as a static feature extractor. This design choice provided three compounding advantages:

- **Decoupling Vision from Physics.** By keeping visual representations stable throughout training, the downstream Diffusion U-Net could dedicate 100% of its learning capacity to the "physics" of the task — the multi-modal mapping of actions to cloth deformation — without also managing shifting visual features.
- **Preventing Catastrophic Forgetting.** Freezing the backbone prevented the small imitation learning dataset from overwriting DINOv2's generalized understanding of geometries, folds, and surface shadows.
- **VRAM Efficiency.** Excluding the massive ViT backbone from gradient computation freed approximately 60–70% of GPU memory during training, which was reinvested into larger batch sizes and longer training horizons.

### 1.2 Surgical Inheritance and Framework Bypass

LeRobot's core is hardcoded to expect ResNet-style backbones. Integrating a ViT required "surgical inheritance" to bypass internal validations at multiple levels:

- **Registry and Initialization Overrides:** The custom policy used `@PreTrainedConfig.register_subclass` to register the new architecture. `__post_init__` sanity checks were overridden to bypass the ResNet-only path, and `super().__init__()` was routed to leapfrog Torchvision ResNet builders directly to `PreTrainedPolicy`.
- **Optimizer Protection:** `get_optim_params()` was overridden to explicitly return only parameters where `requires_grad == True`, cleanly isolating all frozen DINOv2 layers from optimizer updates.

### 1.3 Resolution and Positional Embedding Handling

The LeHome dataset provides **480×640 images**, while DINOv2 is pre-trained on **224×224**. Unlike CLIP (which hard-crashes on mismatched fixed grids), DINOv2 supports interpolated positional embeddings — but feeding the resulting ~1,565 raw patch tokens per camera into the U-Net would cause OOM errors:

- **Antialiased Downsampling:** The `_encode_images()` function intercepts the tensor pipeline and applies antialiased bilinear interpolation to resize inputs to exactly **224×224** (or **252×252** for an 18×18 patch grid), keeping the patch token count at a manageable 256 tokens per camera.

### 1.4 Advanced Spatial Pooling via MAP Head

The most significant architectural hurdle was preserving geometric information after encoding. A raw `[CLS]` token or mean pool discards the 2D spatial structure that SpatialSoftmax on ResNets preserves. Applying standard SpatialSoftmax directly to ViT patch grids risks locking onto high-norm artifact tokens. The solution was a **Multi-headed Attention Pooling (MAP) Head**:

- **Register Token Ablation:** Using `facebook/dinov2-with-registers-small` specifically, the code explicitly slices out the `[CLS]` token and all 4 register tokens (`[:, 1 + num_register_tokens:, :]`) before pooling, preventing these "scratchpad" tokens from corrupting the spatial readout.
- **Learned Query Vectors:** A 1-block cross-attention transformer with **K=8 learnable query vectors** cross-attends to the 256 pure patch tokens.
- **Spatial Specialization:** The 8 queries act as independent spatial summaries, allowing the network to allocate queries to semantically distinct regions (e.g., gripper position, cloth boundary, fold geometry).
- **U-Net Conditioning:** The resulting **8 × 384**-dimensional output is flattened and concatenated per camera, producing a high-capacity, spatially-resolved `global_cond` vector for the downstream Diffusion U-Net — bridging the gap between dense ViT semantics and the flat conditioning interface required by the DP architecture.

---

## 2. System Requirements

| Resource | Minimum | Used in This Run |
|---|---|---|
| GPU | NVIDIA RTX 3090 (24 GB) | NVIDIA RTX 5090 (32 GB) |
| GPU VRAM | ~22 GB (for 2 parallel evals) | 22.6 GB |
| System RAM | 35 GB | 31.5 GB / 50.5 GB |
| CPU | 16 cores | AMD EPYC 9654 (96 cores) |
| CUDA | 11.8+ | 13.0 |
| OS | Ubuntu 22.04 | Ubuntu 22.04.5 |
| Docker | 24.0+ | Latest |
| Python | 3.11 | 3.11.14 |

---

## 3. Pulling and Running the Policy Docker Images

The submitted policy is served via an HTTP server inside a Docker container. The image is pre-built and contains the frozen DINOv2 + Diffusion U-Net model weights along with a FastAPI inference server listening on port **8080**.

### Pull the Image

```bash
docker pull nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell
```

### Run the Policy Servers

Two containers must be started — one per garment category pair being evaluated in parallel. Each maps a different host port to the container's internal port `8080`.

**Terminal 1 — Policy Server A (port 8081)**
```bash
docker run --rm --gpus all \
    -p 8081:8080 \
    nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell
```

**Terminal 2 — Policy Server B (port 8082)**
```bash
docker run --rm --gpus all \
    -p 8082:8080 \
    nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell
```

Wait until both terminals print:
```
Policy server listening on 0.0.0.0:8080
```
before proceeding to run the evaluation scripts.

> **Note:** `--gpus all` routes NVIDIA GPU access into the container. This is where the DINOv2 + Diffusion U-Net inference runs. The simulation itself runs on the host CPU.

---

## 4. Host Environment Setup (One-Time)

### 4.1 Clone and Install the Evaluation Environment

```bash
# Clone the hackathon evaluation repo
git clone https://github.com/lehome-official/lehome-challenge.git
cd lehome-challenge

# Install Python dependencies via uv
uv sync

# Clone IsaacLab into third_party
cd third_party
git clone https://github.com/lehome-official/IsaacLab.git
cd ..

# Activate the virtual environment and install IsaacLab extensions
source .venv/bin/activate
./third_party/IsaacLab/isaaclab.sh -i none

# Install the local lehome package
uv pip install -e ./source/lehome
```

### 4.2 Download Simulation Assets

```bash
hf download lehome/asset_challenge --repo-type dataset --local-dir Assets
```

### 4.3 Install xvfb (Virtual Display — Critical for Headless Servers)

Isaac Sim requires an X11 display connection to load its `pynput` keyboard listener, even in `--headless` mode. On servers without a physical display, `xvfb` provides a virtual framebuffer:

```bash
sudo apt-get update && sudo apt-get install -y xvfb
```

> **Why this is needed:** Without a display, the `pynput` keyboard backend raises:
> ```
> ImportError: this platform is not supported: ('failed to acquire X connection: Can't connect to display ":0"...)
> ```
> `xvfb-run -a` creates a synthetic, isolated X session. The `-a` flag automatically selects a free display number (`:99`, `:100`, etc.), making parallel invocations safe and non-conflicting.

---

## 5. Running Evaluation

All evaluation commands are run from the `lehome-challenge` directory with the virtual environment active.

### Activate the Environment

```bash
cd /path/to/lehome-challenge
source .venv/bin/activate
```

### Terminal 3 — Evaluate `top_long` (against Policy Server A on port 8081)

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

echo "top_long eval PID: $!"
```

### Terminal 4 — Evaluate `pant_long` (against Policy Server B on port 8082)

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

echo "pant_long eval PID: $!"
```

### Key Flag Explanations

| Flag | Value | Reason |
|---|---|---|
| `--policy_type docker` | `docker` | Routes inference to the HTTP policy server |
| `--docker_url` | `http://localhost:808x` | Address of the running policy container |
| `--step_hz 0` | `0` | **Disables the RateLimiter** — sim runs at max possible speed |
| `--device cpu` | `cpu` | Keeps PhysX cloth physics on CPU for stability |
| `--headless` | flag | Runs without a GUI window |
| `--enable_cameras` | flag | Required for the vision-based Diffusion Policy |
| `xvfb-run -a` | wrapper | Provides a synthetic X11 display on a headless server |

---

## 6. Parallel Evaluation Strategy

### Why Exactly 2 Parallel Evaluations

Running more than 2 evaluations concurrently will crash the host due to RAM exhaustion. Based on observed consumption:

| Resource | Per Eval Instance | 2 Instances | Limit |
|---|---|---|---|
| System RAM | ~15 GB | ~31 GB | 50.5 GB |
| GPU VRAM (sim renderer) | ~4.5 GB | ~9 GB | 32 GB |
| GPU VRAM (Docker policy) | ~6.6 GB | ~13.2 GB | 32 GB |
| **Total VRAM** | **~11 GB** | **~22 GB** | **32 GB** |

A 3rd instance would require ~47 GB of RAM (exceeding the 50.5 GB limit) and ~33 GB of VRAM (exceeding the 32 GB limit).

### Suggested Evaluation Order

Run the 4 garment categories in 2 sequential batches of 2:

**Batch 1 (concurrent):**
```bash
# top_long → port 8081
# pant_long → port 8082
```

**Batch 2 (concurrent, after Batch 1 finishes):**
```bash
# top_short → port 8081
# pant_short → port 8082
```

### Monitoring Progress

```bash
# Watch the top_long log in real time
tail -f eval_top_long_fast.log

# Watch the pant_long log in real time
tail -f eval_pant_long_fast.log

# Check that both eval processes are still running
ps -eo pid,etime,args | grep scripts.eval | grep -v grep

# Verify both Docker containers are alive
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'

# Check GPU allocation
nvidia-smi

# Check RAM / CPU
top -b -n 1 | head -n 15
```

---

## 7. Monitoring and Interpreting Logs

### Expected Log Milestones (per eval process)

| Timestamp | Log Message | Meaning |
|---|---|---|
| `~0s` | `Simulation App Starting` | IsaacLab booting up |
| `~8s` | `app ready` | Sim engine initialized |
| `~9s` | `[DockerPolicy] Connected to http://localhost:808x` | Policy server reachable |
| `~9s` | `Loaded 12 garments for category: ...` | Asset list parsed successfully |
| `~30s` | `Simulation App Startup Complete` | PhysX + renderer fully loaded |
| `ongoing` | `Episode X/5: Return=..., Length=..., Success=...` | Active evaluation underway |

### Warnings You Can Safely Ignore

```
[Error] [omni.physx.plugin] PhysX error: attachShape: non-SDF triangle mesh...
```
Standard PhysX warning for cloth mesh types. Does not affect evaluation.

```
** (zenity:XXXXX): WARNING **: AT-SPI: Could not obtain desktop path or name
** (zenity:XXXXX): WARNING **: atk-bridge: get_device_events_reply: unknown signature
```
Accessibility tool warnings from `xvfb-run`'s virtual display. Expected and harmless.

```
Warning: Possible version incompatibility. Attempting to load omni::fabric...
```
Isaac Sim internal version mismatch warning. Does not affect simulation correctness.

### Signs of a Healthy Run

- Docker containers show low-to-moderate CPU in `docker stats` (bursty spikes during inference calls, idle between steps).
- The two `python` eval processes each show ~110–130% CPU in `top` (utilizing 1+ cores for physics stepping).
- Log file line count increases over time — if it freezes for >30 minutes, consider restarting.

---

## 8. Resource Budget at a Glance

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU: NVIDIA RTX 5090 (32 GB)                 │
├─────────────────────────┬───────────────────────────────────────┤
│  Docker Policy A (8081) │  ~6.6 GB  — DINOv2 + Diffusion U-Net │
│  Docker Policy B (8082) │  ~6.6 GB  — DINOv2 + Diffusion U-Net │
│  IsaacLab Sim A         │  ~4.7 GB  — RTX scene renderer        │
│  IsaacLab Sim B         │  ~4.1 GB  — RTX scene renderer        │
│  OS / Desktop           │  ~0.2 GB                              │
├─────────────────────────┴───────────────────────────────────────┤
│  TOTAL VRAM USED: ~22.2 GB / 32 GB         (10 GB headroom)     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    RAM: 50.5 GB Total                           │
├─────────────────────────┬───────────────────────────────────────┤
│  IsaacLab Sim A         │  ~15.8 GB                             │
│  IsaacLab Sim B         │  ~14.1 GB                             │
│  OS + misc              │  ~3.0 GB                              │
├─────────────────────────┴───────────────────────────────────────┤
│  TOTAL RAM USED: ~32.9 GB / 50.5 GB        (17.6 GB headroom)   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              CPU: AMD EPYC 9654 (96 Logical Cores)              │
├─────────────────────────────────────────────────────────────────┤
│  IsaacLab Sim A + B     │  ~2.5 cores (PhysX cloth + render)    │
│  Available              │  ~93 cores idle                        │
└─────────────────────────────────────────────────────────────────┘
```

---

*Generated during the LeHome Challenge 2026 evaluation run — Silicon Optimists team.*
*Branch: `principia_instance` | Model: `FullDP-v2_Blackwell`*
