# LeHome Docker Submission Plan (HF Registry)

This document is a **plan only**. It is designed so a low-cost model (or a human following a checklist) can produce a working Docker submission with minimal iteration. All actual execution (downloading checkpoints, building, pushing, testing) is intended to happen on the **GPU VM**, not this local workspace.

## Goal

Create a Docker image that runs an HTTP policy server compatible with the official LeHome evaluation client (`policy_type=docker`), then publish it to **Hugging Face’s Docker registry** so organizers can `docker pull` + run evaluation.

## Contract (what organizers will run)

Organizers will run two processes:

1. Your policy container:

```bash
docker run --rm -p 8080:8080 <your-image>
```

1. The official evaluator (in the official `lehome-challenge` repo):

```bash
python -m scripts.eval --policy_type docker --docker_url http://localhost:8080 --garment_type top_long --headless --device cpu --enable_cameras
```

Your container must expose:

- `POST /reset` → `{"status":"ok"}`
- `POST /infer` → `{"actions":[[12 floats], ...]}`

The request body to `/infer` is the observation dict where images/depth are base64-encoded arrays with `{base64, shape, dtype}`.

## Why CUDA 12.1 in the Docker image is OK even if the VM has CUDA 12.9/13

Docker images include their own CUDA toolkit user-space libraries. The host only provides the NVIDIA driver. NVIDIA drivers are backwards compatible, so a host with a modern driver (supporting CUDA 12.9/13) can run a container built for CUDA 12.1.

## Security rules

- Do **not** put W&B tokens in the Docker image.
- Do **not** give organizers W&B credentials.
- Do **download the checkpoint on the VM first**, then bake weights into the image via `COPY`.

## Implementation choices (keep it simple)

- **Single unified diffusion policy** and **router+4 specialists** are both compatible with the same Docker protocol. The router design is just internal logic inside `policy.py`.
- Do not use depth if you didn’t train with it; ignore `observation.top_depth`.

## Files to base from (official upstream)

Use the official `dummy_docker_policy/` as the template:

- `dummy_docker_policy/server.py` (do not change)
- `dummy_docker_policy/policy.py` (you implement model loading + inference)
- `dummy_docker_policy/Dockerfile` (extend for torch/lerobot + copy weights)

## W&B → local download (on VM)

1. Find your W&B artifact name/version (prefer an explicit version, not “latest”).
2. Download it locally into a folder you will copy into the image, e.g. `pretrained_model/`.
3. Confirm the folder contains the expected LeRobot/DP checkpoint structure (often includes `train_config.json`, model weights, etc.).

## Build (on VM)

Build a CUDA-enabled runtime image (safe default):

- Base image: `pytorch/pytorch:<pytorch_version>-cuda12.1-cudnn8-runtime`
- `COPY pretrained_model/ /app/pretrained_model/`
- Run `python policy.py` to start the server

## Push to Hugging Face Docker registry (HF Spaces)

Hugging Face’s Docker registry is tied to Spaces; the simplest workflow is:

1. Create a Space with **SDK = Docker** (UI step).
2. Log in to the registry:

```bash
docker login registry.hf.space -u <hf_username>
# password: HF access token with write permission to that Space
```

3. Get the **exact image reference** from the Space UI.

In your Space page, use **“Run with Docker”** and copy the provided `docker pull ...` command.

Important notes (HF behavior in practice):
- The registry repo name is often `registry.hf.space/<org>-<space>` (hyphen-joined), and for **long space names** it may be truncated / have a suffix, so it’s not always derivable.
- Tags are commonly `:latest` and `:<commit-sha>`; do not assume arbitrary tags like `:v1` always work.

4. (Optional) Tag + push (only if you are pushing a locally-built image):

```bash
docker tag my-policy:latest <IMAGE_REF_FROM_SPACE_UI>
docker push <IMAGE_REF_FROM_SPACE_UI>
```

## What you provide to organizers

You provide:

1. The image reference:

- Use the exact reference from the Space UI (example format):
  - `registry.hf.space/<org>-<space>:latest`

1. A **fine-grained HF access token** with **read-only** access to just that Space/repo so they can pull.
2. The eval command (and any required flags) they should run in the official repo.

You do **not** provide:

- W&B API keys
- Google Drive access

## Minimal test plan (on VM)

1. Start container:

```bash
docker run --rm -p 8080:8080 <IMAGE_REF_FROM_SPACE_UI>
# e.g. registry.hf.space/<org>-<space>:latest
```

1. Run `scripts.eval` against it from an official `lehome-challenge` clone on the VM:

```bash
python -m scripts.eval --policy_type docker --docker_url http://localhost:8080 --garment_type top_long --headless --device cpu --enable_cameras
```

If this runs end-to-end without contract errors (HTTP timeouts, JSON schema issues, wrong action dims), the submission is structurally correct.