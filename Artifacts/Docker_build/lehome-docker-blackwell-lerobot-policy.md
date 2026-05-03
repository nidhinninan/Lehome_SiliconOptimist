# Artifact: LeHome Docker submission, Blackwell (sm_120), and LeRobot policy compatibility

**Source:** Discussion chain (docker build failure, PyTorch / CUDA / `lerobot`, submission Dockerfile path).  
**Workspace paths referenced:** `lehome-challenge/dummy_docker_policy/`

## 1. Docker build: `useradd: UID 1000 is not unique`

- **Cause:** Base image `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime` already defines a user with UID 1000. A Dockerfile step that unconditionally creates `-u 1000` fails and stops the build (no image/tag produced).
- **Fix pattern:** If `user` already exists, reuse it; else create `user` with UID 1000 only when free, otherwise use 1001 (or equivalent fallback).

## 2. `pip install -e lerobot_policy_dino` and PyTorch downgrades

- Installing `lerobot_policy_dino` pulls **`lerobot==0.4.3`**, which may cause pip to resolve **Torch from PyPI** (often **cu126** wheels), downgrading or replacing the Torch shipped in the CUDA 12.8 base image.
- **Risk for Blackwell:** You need a **PyTorch build whose CUDA wheel matches your GPU stack** (often **cu128** for Blackwell-oriented setups). The downgrade is an **environment consistency** issue, not “CUDA in the base image” by itself.

## 3. Re-pinning Torch to cu128 after `lerobot` install — will it break LeRobot / policy execution?

- **`lerobot==0.4.3` declares** (see installed metadata): `torch>=2.2.1,<2.8.0`, `torchvision>=0.21.0,<0.23.0` (and conditional `torchcodec` on Linux).
- **Re-pinning to Torch 2.11.x** is **outside** that declared range for `lerobot==0.4.3` → **not a supported combo**; runtime may still work but can break at import or inference.
- **Safer approach:** After `lerobot` installs, reinstall **Torch 2.7.x** matching wheels from **`https://download.pytorch.org/whl/cu128`**, with **pinned versions** for `torch`, `torchvision`, and `torchaudio` so pip does not float to an incompatible Torch.

## 4. “Higher CUDA” vs LeRobot

- **Raising the CUDA version in the container base** does not automatically break LeRobot; what matters is the **resolved PyTorch wheel set** and whether it satisfies **`lerobot`’s version constraints**.
- Blackwell support is primarily about **using a PyTorch build appropriate for the GPU** (cu128 channel in this workflow), not only the distro CUDA packages.

## 5. Submission files (`Dockerfile.submission`, `requirements.txt`, `build_docker_submission.sh`)

**Intent:** CUDA 12.8–oriented base + BYOP package + **controlled** cu128 reinstall that stays inside `lerobot==0.4.3`’s Torch upper bound.

- **`Dockerfile.submission`:** Uses a CUDA 12.8 PyTorch runtime base; installs `requirements.txt`, then editable `lerobot_policy_dino`, then **pins** `torch` / `torchvision` / `torchaudio` (e.g. 2.7.1 / 0.22.1 / 2.7.1) from the **cu128** index.
- **`requirements.txt`:** Should **not** pin `torch` / `torchvision` / `torchaudio` — those belong in the Dockerfile step so the cu128 + version pins stay authoritative.
- **`build_docker_submission.sh`:** Can validate that `requirements.txt` does not pin those packages; documents Blackwell-oriented build flow.

## 6. Operational checklist

1. Fix **user creation** in the Dockerfile so UID collisions cannot abort the build.
2. After **`lerobot_policy_dino`**, **reinstall pinned** Torch stack from **cu128** index within **`torch<2.8`** for `lerobot==0.4.3`.
3. Do not duplicate Torch pins in **`requirements.txt`** unless you intentionally override the Dockerfile contract.
4. Rebuild and tag the image; verify with a quick import/runtime smoke test inside the container if needed.

---
*This file is a snapshot of the conversation decisions; adjust version pins if upstream wheels or `lerobot` constraints change.*
