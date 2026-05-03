# Artifact: Docker policy image for Blackwell (`sm_120`) and LeHome submission path

This document records the **Docker build and runtime path** discussed for the LeHome challenge policy container (`dummy_docker_policy`), including **GPU architecture constraints**, **errors encountered**, **why they occurred**, and **how they were addressed**. It is written for someone who was not in the conversation but needs to reproduce the reasoning.

---

## 1. Problem framing: two different GPU failures

Two distinct classes of issues appeared in terminal logs:

### 1.1 CUDA out of memory (VRAM)

**Symptom:** `torch.OutOfMemoryError: CUDA out of memory` while loading tensors (e.g. `result[k] = f.get_tensor(k)`).

**Meaning:** The workload requested more **device memory** than was free on the GPU at that moment. Typical mitigations are smaller batch size, smaller model, `torch.no_grad()` for inference-only paths, freeing cache, reducing fragmentation (`PYTORCH_CUDA_ALLOC_CONF`), or a GPU with more VRAM.

**Relation to “RTX 5090” question:** A 32 GB card would reduce or eliminate *this* class of failure for many workloads; it does **not** fix driver or wheel **compatibility** problems by itself.

### 1.2 Compute capability mismatch (`sm_120` vs shipped PyTorch kernels)

**Symptom (example):**

```text
NVIDIA GeForce RTX 5060 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
```

**Meaning:**

- NVIDIA **Blackwell** consumer GPUs (e.g. RTX 50-series) report **compute capability 12.0**, denoted **`sm_120`**.
- A given PyTorch wheel is built to include **precompiled CUDA kernels** only for a **finite set** of architectures (here, up through **`sm_90`**).
- If the GPU’s architecture is **not** in that set, PyTorch may warn or fail when executing CUDA ops (often later as “no kernel image is available for execution on the device” in stricter paths).

**Important nuance:** Replacing the GPU with another **Blackwell** SKU (e.g. RTX 5090) **does not remove** this error if the **same PyTorch build** is still used. The fix is always **software alignment**: install a PyTorch (and CUDA user-space stack) built for **Blackwell**, typically tied to **CUDA 12.8+** for this generation, not “more VRAM.”

**Where it showed up in this thread:** The user ran a **prebuilt** image:

`nninspaceexp/lehome_silicon-optimists:FullDP-v1`

That image bundled a PyTorch/CUDA combination that predated or omitted **`sm_120`** support, so the warning appeared **inside the container** even when the host correctly passed through the GPU (`docker run --gpus all`).

---

## 2. Repository layout relevant to the build

| Piece | Role |
|--------|------|
| `dummy_docker_policy/Dockerfile.submission` | **Submission** GPU image: base OS + PyTorch stack + policy deps + weights. |
| `dummy_docker_policy/Dockerfile` | Minimal CPU-oriented image for smoke tests; **not** the Blackwell submission path. |
| `dummy_docker_policy/build_docker_submission.sh` | Orchestrates optional W&B download, stages `lerobot_policy_dino`, runs `docker build -f Dockerfile.submission`. Default Dockerfile is **`Dockerfile.submission`**. |
| `requirements.txt` / `requirements.submission.template` | Python deps for the image context; **torch stack ownership** was intentionally moved to the Dockerfile to avoid conflicting pins. |
| `pretrained_model/` | Weights baked into the image when using `--skip-download`; must be non-empty or the script exits. |

---

## 3. Evolution of the solution (logic flow)

### 3.1 Move the submission image to a Blackwell-capable **base**

**Decision:** Change `Dockerfile.submission` `FROM` from an older CUDA 12.1 PyTorch image (kernels only through `sm_90`) to an official **`pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime`** (or equivalent **CUDA 12.8** runtime tag).

**Reasoning:** The **container** must ship a PyTorch build whose CUDA toolkit and wheel artifacts include **Blackwell** support. Bumping only the host driver is insufficient if the **image** still installs old wheels.

### 3.2 `useradd` failure: UID 1000 already taken

**Symptom:** `useradd: UID 1000 is not unique` during `docker build`.

**Cause:** Newer official PyTorch images often already define a user occupying **UID 1000** (common convention). Creating another user with the same numeric UID violates uniqueness.

**Solution direction:** Make user creation **conditional**:

- If a user named `user` already exists, skip creation.
- Else if UID 1000 is taken, create `user` with **UID 1001** (or reuse policy appropriate to deployment).
- Else create `user` with UID 1000 as before.

**Nuance:** Some platforms (e.g. certain hosted runtimes) assume UID **1000** for file ownership. The conditional approach trades strict UID for **build reliability**; if a downstream system mandates UID 1000, the policy would be “use the base image’s existing account” instead of adding a second user.

### 3.3 PEP 668: `externally-managed-environment` on `pip install`

**Symptom:** `pip install -r requirements.txt` failed with **PEP 668** (“externally-managed-environment”) on Ubuntu 24.04–style bases with system-managed Python.

**Cause:** The distribution marks the interpreter as **externally managed** so `pip` does not mutate the OS Python without an explicit escape hatch.

**Solution:** In a **disposable container** (not a multi-user server), set:

`ENV PIP_BREAK_SYSTEM_PACKAGES=1`

(or pass `--break-system-packages` on each `pip` invocation).

**Reasoning:** Containers are rebuilt from Dockerfile; the risk profile differs from mutating a developer laptop OS Python.

### 3.4 Lerobot pulling a **different** torch than the base image

**Observation from build logs:** After `pip install -e lerobot_policy_dino`, dependency resolution installed **`torch` 2.7.x** with **CUDA 12.6–style** NVIDIA pip metapackages, and **uninstalled** the base image’s **`torch 2.11.0+cu128`** (and related NVIDIA libs).

**Why it matters:**

- **Blackwell** support is tied to having the **right** PyTorch **and** CUDA user-space libraries. An unintended downgrade to **cu126** wheels can **reintroduce** the “no `sm_120` kernels / wrong CUDA major” class of failure even if the `FROM` line says CUDA 12.8.
- **`lerobot==0.4.3`** declares **upper bounds** on `torch` and `torchvision` (e.g. `torch < 2.8`), so blindly upgrading to the newest torch in the image can violate declared compatibility for reproducibility.

**Resolution strategy (final tightened design):**

1. Let `lerobot_policy_dino` install **lerobot** and its dependencies as usual.
2. **Re-pin** `torch`, `torchvision`, and `torchaudio` to **known-good versions** that satisfy **lerobot’s** declared ranges **and** are installed from **`https://download.pytorch.org/whl/cu128`** so the **cu128** wheel line is authoritative.

Concretely (as implemented in-repo): **`torch==2.7.1`**, **`torchvision==0.22.1`**, **`torchaudio==2.7.1`** from the **cu128** index, parameterized via `ARG` for future bumps without editing multiple lines.

**Nuance:** Pip may print **“dependency conflicts”** warnings because installed metadata still describes `lerobot`’s constraints vs the chosen torch line. Those warnings are **informational** unless pip aborts; the intentional choice is **“compatible enough + correct CUDA wheel channel.”** Operational validation remains: **run inference on real Blackwell hardware**.

### 3.5 `docker push`: “tag does not exist”

**Symptom:** Pushing `nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell` failed with **tag does not exist**.

**Cause:** Docker Hub does not create a tag in the void. **`docker push`** uploads a **local** image reference. If **`docker build` failed** before tagging, **no local image** existed with that name:tag, so the CLI reported that the tag “does not exist” (locally).

**Lesson:** Always confirm `docker images | grep FullDP-v2_Blackwell` (or equivalent) **after a successful build**, then push.

### 3.6 Intermittent BuildKit export / snapshot error

**Symptom (observed during one rebuild):** `failed to prepare extraction snapshot ... parent snapshot ... does not exist`.

**Interpretation:** Typically a **BuildKit / graph driver** inconsistency (cache layer corruption, interrupted export, or storage backend glitch), not necessarily a Dockerfile syntax error.

**Mitigation:** Retry the build; if recurrent, `docker builder prune` and/or `--no-cache` for a clean graph.

### 3.7 Shell / script interaction note

At one point a long compound command (`build && push`) was used; a **non-zero** or **parse edge** in the surrounding shell can surface as an error **attributed to a line number** inside the script even when the real failure mode was **ordering** (push before build success) or **environment**. After isolating steps (`build` then `push`) and fixing the Dockerfile, the build completed with **exit code 0**.

### 3.8 `pip install -e` fails building `evdev`: `gcc` missing in runtime base image

**Symptom (from Docker build log):**

```text
Building wheel for evdev (pyproject.toml): finished with status 'error'
...
error: command 'gcc' failed: No such file or directory
ERROR: Could not build wheels for evdev, which is required to install pyproject.toml-based projects
```

**Cause:** The submission image used a **runtime** CUDA/PyTorch base image that does not include a compiler toolchain. `pynput` pulls `evdev`, which builds a native extension and requires `gcc` (and basic build tooling).

**Fix:** Install a minimal build toolchain in `dummy_docker_policy/Dockerfile.submission`, e.g. add:

- `build-essential` (brings in `gcc`, `g++`, `make`, headers tooling)

This change allows `pip install --no-cache-dir -e /app/lerobot_policy_dino` to complete successfully.

---

## 4. Guardrails added in `build_docker_submission.sh`

A **`grep`** gate was added so `requirements.txt` cannot silently pin **`torch` / `torchvision` / `torchaudio`**, which would fight the Dockerfile’s deliberate **cu128** reinstall step and recreate the downgrade/upgrade tug-of-war.

**Reasoning:** Single **source of truth** for the CUDA wheel line reduces “works on my laptop” drift between `requirements.txt` edits and image behavior.

---

## 5. Commands reference (end state)

**Build (weights already in `pretrained_model/`):**

```bash
cd /data/lehome_workspace/lehome-challenge/dummy_docker_policy
./build_docker_submission.sh --skip-download --image-tag nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell
```

**Run (GPU):**

```bash
docker run --rm --gpus all -p 8081:8080 nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell
```

**Push:**

```bash
docker push nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell
```

---

## 6. Checklist for the next person debugging “still broken on RTX 50xx”

1. **Confirm inside the container:** `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"`.
2. **Confirm** `torch` wheels are **`+cu128`** (or the intended CUDA tag), not an accidental **`cu126`** line left by a dependency.
3. **Confirm** host driver is new enough for Blackwell **and** matches container CUDA expectations (container user-space vs host driver are different layers).
4. **If OOM:** separate issue—reduce memory use or use more VRAM; not the same as `sm_120` mismatch.

---

## 7. Artifact metadata

| Field | Value |
|--------|--------|
| Topic | LeHome `dummy_docker_policy` Docker submission, Blackwell (`sm_120`), lerobot 0.4.3, cu128 torch pinning |
| Conversation scope | OOM vs compatibility, FullDP-v1 image behavior, Dockerfile.submission iterations, build/push pitfalls, lerobot/torch interaction |
| On-disk artifact path | `/data/lehome_workspace/artifact/DOCKER_BUILD_BLACKWELL_AND_LEROBOT.md` |

This file is a **narrative and technical record**; it does not replace pinned versions in the repo—always treat **`Dockerfile.submission`** and **`build_docker_submission.sh`** in the repository as the **authoritative** build definition.
