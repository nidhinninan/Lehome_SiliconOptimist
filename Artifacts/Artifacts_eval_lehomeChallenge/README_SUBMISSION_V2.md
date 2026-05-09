# LeHome Challenge 2026 — Submission README (V2 — Micro Docker)

### Team: Silicon Optimists | Branch: `principia_instance`

**Repository:** [nidhinninan/Lehome_SiliconOptimist @ principia_instance](https://github.com/nidhinninan/Lehome_SiliconOptimist/tree/principia_instance)

This document is the **evaluator-facing guide** for replicating our results using the **Micro Docker** workflow. The entire fix lives **inside the Docker image** — the host-side setup is the standard hackathon process, unchanged. No host patches or extra scripts are required.

---

## What you need to know first

### Checkpoints are unchanged

- **Same weights as the original submission:** `model.safetensors` is **not** retrained or replaced.
- The Micro Docker change fixes **inference plumbing only**, not the learned policy.

### Where the fix lives — and what is unchanged on the host

The original Docker container had a critical bug in its HTTP server wrapper (`policy.py`): it fed raw, unnormalized observations directly into the model and returned un-denormalized actions to the simulator. This produced 0% success.

The Micro Docker build layers on top of the original submitted image and replaces **only two things inside the container**:

| File in container | What changed | Why it matters |
|-------------------|-------------|----------------|
| `policy.py` | Rewritten to use LeRobot's full `PolicyProcessorPipeline` | Model receives data in the same normalized distribution it was trained on; simulator receives correctly scaled joint angles |
| `meta/` | Dataset metadata / normalization statistics added | Required by the pre/post-processors to compute correct normalization |

**Nothing on the host needs to change.** Evaluators follow the standard hackathon setup steps exactly as documented by the organizers and run the standard `python -m scripts.eval --policy_type docker` command. The only difference from the original submission README is the **image tag** used when starting the Docker containers.

### Image tags

| Tag | Contents |
|-----|----------|
| `nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell` | Original submitted image (base for the rebuild) |
| `nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell_patched-Updated` | Micro-rebuilt image — **use this one for evaluation** |

---

## Credentials

### Docker Hub (read-only pull token)

| Field | Value |
|-------|-------|
| Username | `nninspaceexp` |
| Access token | `<TOKEN-REDACTED>` |
| Scope | Read-only (pull only) |
| Expires | Jul 30, 2026 |

```bash
docker login -u nninspaceexp --password-stdin <<< "<TOKEN-REDACTED>"
```

### Hugging Face (gated `lehome/asset_challenge` dataset)

| Field | Value |
|-------|-------|
| Token | <_USE OWN HF TOKEN_> |

```bash
huggingface-cli login --token <USE_OWN_HF_TOKEN>
```

---

## Prerequisites

- Ubuntu 22.04 (tested)
- NVIDIA GPU with ≥24 GB VRAM (tested on RTX 5090 class)
- CUDA-compatible driver (CUDA 11.8+)
- `uv` — `pip install uv`
- Docker with NVIDIA Container Toolkit (`--gpus all`)
- `sudo apt-get install -y xvfb` (headless X11 for Isaac Sim)
- **No `zenity` on the host.** Omniverse Kit's hang-detector spawns a `zenity` modal ("Kit appears to be hanging — terminate?") under `xvfb` if early Warp / PhysX initialization is slow. Because no human can click it, the eval freezes indefinitely. Either purge or shadow zenity:

  ```bash
  # Option A: remove (preferred on a dedicated eval VM)
  sudo apt-get remove -y zenity

  # Option B: shadow with a no-op (keeps zenity installed but auto-cancels Kit's hang dialog)
  sudo ln -sf /bin/true /usr/local/bin/zenity
  ```

---

## Step-by-step replication

### Step 1 — Clone the official evaluation repository

```bash
git clone https://github.com/lehome-official/lehome-challenge.git
cd lehome-challenge
```

### Step 2 — Install Python dependencies (standard)

```bash
uv sync
source .venv/bin/activate
```

### Step 3 — Clone and install IsaacLab (standard)

```bash
cd third_party
git clone https://github.com/lehome-official/IsaacLab.git
cd ..
echo "yes" | ./third_party/IsaacLab/isaaclab.sh -i none   # accept NVIDIA EULA when prompted

# Fix IsaacLab 5.1.0 warp-lang dependency issue
uv pip install warp-lang==1.11.1
```

### Step 4 — Install the LeHome package (standard)

```bash
uv pip install -e ./source/lehome
```

### Step 5 — Download simulation assets

```bash
hf download lehome/asset_challenge --repo-type dataset --local-dir Assets
```

(Use the HF token above if the CLI is not already authenticated.)

### Step 6 — Build the Micro Docker image

This is the **only step that differs** from the original submission README.

You need four files/folders in the same directory: `Dockerfile.patch`, `policy.py`, `meta/`, and `pretrained_model/`. These are provided in the submission artifact bundle (`dummy_docker_policy/` in our workspace). Navigate to that directory and run:

```bash
# Authenticate to pull the base image
docker login -u nninspaceexp --password-stdin <<< "<TOKEN-REDACTED>"

# Build the patched image
docker build -t nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell_patched-Updated \
    -f Dockerfile.patch .
```

The build pulls `FullDP-v2_Blackwell` as the base layer, then overlays `policy.py`, `meta/`, and `pretrained_model/` (which contains the required `config.json` and pre/post processors).

### Step 7 — Start two policy servers (NEED MINIMUM 24GB+ VRAM)

**Terminal A — port 8081**

```bash
docker run --rm --gpus all -e HF_HOME=/tmp -p 8081:8080 \
    nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell_patched-Updated
```

**Terminal B — port 8082**

```bash
docker run --rm --gpus all -e HF_HOME=/tmp -p 8082:8080 \
    nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell_patched-Updated
```

Wait until both terminals print: `Policy server listening on 0.0.0.0:8080`

### Step 8 — Run evaluation (standard `scripts.eval` command)

Back in the `lehome-challenge` directory with the venv active.

**`top_long` (port 8081)**

```bash
WARP_CACHE_PATH=/data/warp_cache OMNI_KIT_ACCEPT_EULA=yes PYTHONUNBUFFERED=1 \
xvfb-run -a python -u -m scripts.eval \
    --policy_type docker \
    --docker_url http://localhost:8081 \
    --garment_type top_long \
    --step_hz 0 \
    --device cpu \
    --headless \
    --enable_cameras \
    > eval_top_long.log 2>&1 &
```

**`pant_long` (port 8082) — in parallel**

```bash
WARP_CACHE_PATH=/data/warp_cache OMNI_KIT_ACCEPT_EULA=yes PYTHONUNBUFFERED=1 \
xvfb-run -a python -u -m scripts.eval \
    --policy_type docker \
    --docker_url http://localhost:8082 \
    --garment_type pant_long \
    --step_hz 0 \
    --device cpu \
    --headless \
    --enable_cameras \
    > eval_pant_long.log 2>&1 &
```

After those finish, reuse the same containers for `top_short` and `pant_short` by swapping `--garment_type`.

### Flag reference

| Flag | Value | Purpose |
|------|-------|---------|
| `--policy_type docker` | `docker` | Routes inference to the HTTP server inside the container |
| `--docker_url` | `http://localhost:8081` / `8082` | Address of the running policy server |
| `--step_hz 0` | `0` | Disables rate limiter — sim runs at max speed |
| `--device cpu` | `cpu` | PhysX cloth on CPU (stable on Isaac Sim 5.1) |
| `--headless` | flag | No GUI window |
| `--enable_cameras` | flag | Required — enables camera observations |
| `xvfb-run -a` | wrapper | Synthetic X11 display on headless servers |
| `WARP_CACHE_PATH=/data/warp_cache` | env var | Writable Warp kernel cache; prevents Kit "appears to be hanging" zenity popup on first run |
| `OMNI_KIT_ACCEPT_EULA=yes` | env var | Non-interactively accepts the Omniverse Kit EULA on the host |
| `PYTHONUNBUFFERED=1` + `python -u` | env/flag | Streams Python and Kit logs immediately so you can monitor progress in `eval_*.log` |

---

## Resource guide (2 parallel evals)

| Resource | Per eval | 2 evals | Minimum |
|----------|----------|---------|---------|
| GPU VRAM (sim renderer) | ~4.5 GB | ~9 GB | — |
| GPU VRAM (Docker policy) | ~6.6 GB | ~13.2 GB | — |
| **Total VRAM** | **~11 GB** | **~22 GB** | **24 GB** |
| System RAM | ~15 GB | ~30 GB | machine-dependent |

Running a 3rd simultaneous evaluation risks OOM on both RAM and VRAM.

---

## What evaluators should not do

- Do not run `apply_host_updates.sh` — that script was for our internal `--policy_type lerobot` validation path. It is **not needed** for Docker-mode evaluation.
- Do not use `lerobot_eval_with_plugins.py` — same, internal use only.
- Do not bind-mount or otherwise modify the running container; all policy logic and weights are baked into the image.

---

## Related artifacts

| File | Purpose |
|------|---------|
| `MICRO_DOCKER_REBUILD_GUIDE.me` | Short build/push reference |
| `EVALUATION_README_FINAL.md` | Full rationale: normalization bug, host vs container scope |
| `README_SUBMISSION.md` | Original architecture write-up; V2 supersedes it for image tag and credentials |

---

*LeHome Challenge 2026 — Silicon Optimists*
*V2: Micro Docker (`FullDP-v2_Blackwell` → `FullDP-v2_Blackwell_patched-Updated`) | Checkpoints unchanged*
