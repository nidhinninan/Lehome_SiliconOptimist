# LeHome Challenge 2026 — Final Evaluation Instructions
### Team: Silicon Optimists

This document is the **single, authoritative entry point** for evaluators. It supersedes the standalone `EVALUATION_ADAPTATION_README.md` and is intended to be read **in conjunction with `README_SUBMISSION.md`** (which it references where appropriate).

> **Submission integrity statement:** This evaluation uses a **micro-rebuild** of our original Docker submission. The base image and the **model checkpoints remain exactly as submitted before the deadline**. The *only* modifications are highly targeted, non-checkpoint related script patches:
> 1. **Inside the Docker container**: We patched the inference wrapper (`policy.py`) to correctly apply LeRobot's input normalization and output un-normalization pipeline, which was accidentally bypassed in the original submission. We also included the dataset `meta/` folder to provide the normalization statistics.
> 2. **On the host machine**: We apply a few targeted patches (e.g., pinning `warp-lang==1.11.1` for Isaac Sim 5.1 compatibility) to ensure the host environment can communicate with the Docker policy server reliably on modern hardware.

---

## What changed in the Docker Micro-Rebuild (and why)

The original Docker submission had a critical bug in its HTTP server wrapper (`policy.py`) where it fed raw, unnormalized observations directly into the model and returned un-denormalized actions to the simulator. This resulted in a 0% success rate. 

To fix this without touching the model weights, we created a **micro-rebuild** (`Dockerfile.patch`) that layers on top of the original submitted image (`nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell`).

| File | Change | Why it matters |
|---|---|---|
| `policy.py` | Re-written to use LeRobot's `PolicyProcessorPipeline` | Ensures the model receives data in the same normalized distribution it was trained on, and the simulator receives correctly scaled joint angles. |
| `meta/` | Copied dataset metadata folder | Provides the `stats.json` required by the pre/post-processors to perform the normalization. |

**The model checkpoint (`model.safetensors`) and the core policy architecture remain exactly as submitted.**

---

## What changed on the host (and why)

| File | Change | Why it matters for Docker-mode evaluation |
|---|---|---|
| `pyproject.toml` | Pin `warp-lang==1.11.1` in `override-dependencies` | Without this pin, `uv sync` resolves to a `warp-lang` version that is incompatible with Isaac Sim 5.1 / IsaacLab and the simulator fails to boot on Blackwell-class GPUs. **This is the only patch that materially affects Docker-mode results.** |
| `lerobot_eval_with_plugins.py` *(new)* | Plugin-aware wrapper | Only used for our internal `--policy_type lerobot` validation runs. **Not invoked during Docker-mode evaluation.** Provided for reproducibility of our internal numbers. |
| `scripts/eval_policy/lerobot_policy.py` | Stricter pretrained-model validation, fixed action-dim inference | Same — only relevant on the `--policy_type lerobot` path. **Not exercised by `--policy_type docker`.** |
| `scripts/utils/evaluation.py` | Forwards `task_name` to the lerobot adapter | Same — `--policy_type lerobot` only. |
| `scripts/__init__.py` *(new)* | Empty package marker | Ensures `python -m scripts.eval` continues to work even on Python interpreters that disable implicit namespace packages. |

**Note:** The host-side `lerobot_eval_with_plugins.py` and `lerobot_policy.py` patches are only relevant for our internal `--policy_type lerobot` runs. They are **not exercised by `--policy_type docker`**.

---

## Step-by-Step Instructions

### Step 0 — System Prerequisites

Same as `README_SUBMISSION.md` → "Prerequisites": Ubuntu 22.04, NVIDIA GPU ≥ 24 GB VRAM, CUDA 11.8+, `uv`, Docker with NVIDIA Container Toolkit, network access.

### Step 1 — Clone the Official Hackathon Repository

```bash
git clone https://github.com/lehome-official/lehome-challenge.git
cd lehome-challenge
```

### Step 2 — Apply Host-Side Eval Patches (**before** `uv sync`)

Copy `apply_host_updates.sh` (provided alongside this README) into the `lehome-challenge/` root and run it:

```bash
chmod +x apply_host_updates.sh
./apply_host_updates.sh
```

Expected output:
```
==> Applying Silicon Optimists host-side eval patches...
    [ok] added warp-lang==1.11.1 to pyproject.toml
==> Writing lerobot_eval_with_plugins.py ...
==> Overwriting scripts/eval_policy/lerobot_policy.py ...
==> Overwriting scripts/utils/evaluation.py ...
==> Host-side eval patches applied successfully.
```

The script is idempotent and exits non-zero if anything fails — do not proceed if you see an error.

### Step 3 — HuggingFace Login (required before asset download)

The `lehome/asset_challenge` dataset is gated. Authenticate once:

```bash
huggingface-cli login
# paste a HF read token when prompted
```

### Step 4 — Install Python Dependencies

```bash
uv sync
```

This now picks up the patched `pyproject.toml` (including `warp-lang==1.11.1`).

### Step 5 — Build the Micro-Rebuild Docker Image

Instead of just pulling the original image (as described in `README_SUBMISSION.md` Step 7), you must build the micro-rebuild image. We have provided an archive/folder containing:
- `Dockerfile.patch`
- `policy.py` (the fixed inference wrapper)
- `meta/` (the dataset statistics)
- `pretrained_model/` (contains `config.json`, `policy_preprocessor.json`, `policy_postprocessor.json`, and matching `*.safetensors` required by the new `policy.py` to instantiate the LeRobot `PolicyProcessorPipeline`; the base `FullDP-v2_Blackwell` image only contains `model.safetensors`)

Navigate to this folder and build the patched image:

```bash
# Authenticate with the read-only token to pull the base image
docker login -u nninspaceexp --password-stdin <<< "<REDACTED_DOCKER_PAT>"

# Build the micro-rebuild image
docker build -t nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell_patched-Updated -f Dockerfile.patch .
```

### Step 6 — Continue with `README_SUBMISSION.md`

From this point onward, **follow `README_SUBMISSION.md` from Step 3 ("Clone and Install IsaacLab") through Step 10 (parallel evaluation)**, with one critical change:

**When starting the Docker containers (Step 8), use the patched image tag:**
```bash
docker run --rm --gpus all -e HF_HOME=/tmp -p 8081:8080 nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell_patched-Updated
```

> **Important — do NOT swap the evaluation command.** When you reach Step 9 in `README_SUBMISSION.md`, run the official module entry point exactly as written:
>
> ```bash
> xvfb-run -a python -m scripts.eval --policy_type docker --docker_url http://localhost:8081 ...
> ```
>
> The `lerobot_eval_with_plugins.py` wrapper that this script writes into your repo is **only** for our internal `--policy_type lerobot` runs and will fail in a clean evaluator environment because it requires a separate `lerobot_policy_dino/src/` checkout that lives outside the container.

---

## What Evaluators Should NOT Do

- **Do not bind-mount, copy into, or otherwise modify the running Docker container.** All policy logic and weights live inside the image.
- **Do not invoke `lerobot_eval_with_plugins.py` for Docker-mode evaluation** (see warning above).

---

## Quick Sanity Checklist

Before launching evaluation, verify:

- [ ] `grep warp-lang pyproject.toml` returns `"warp-lang==1.11.1",`
- [ ] `huggingface-cli whoami` returns your HF user
- [ ] `docker images | grep FullDP-v2_Blackwell_patched` shows the built image
- [ ] Two policy server containers are listening (`docker ps` shows ports 8081 and 8082)
- [ ] Both containers have printed `Policy server listening on 0.0.0.0:8080`

---

*LeHome Challenge 2026 — Silicon Optimists*
*Image: `FullDP-v2_Blackwell` | Submission branch: `principia_instance` | Host patch source: `VM_update`*
