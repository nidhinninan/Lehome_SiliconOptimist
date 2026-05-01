# Docker submission — local replication (VM parity)

**Date:** 2026-04-30  
**Context:** This artifact has three parts: (1) **original prompt context** — Drive sync, VM root cause, fixes, and recommended commands; (2) **independent audit** — findings and score as supplied by the user; (3) **local implementation summary** — what was replicated in this repo (plan execution).

---

## Original prompt context (start of conversation chain)

### Sync from Google Drive (user workflow)

The local tree was first aligned with Drive using `rclone` from `gdrive:LeHome/dummy_docker_policy` into `lehome_workspace/lehome-challenge/dummy_docker_policy`, then `rsync` to sync. That brought over whatever was on Drive (which did not include the full VM-fixed layout by itself).

### Cause (VM Docker failure)

Docker failed because the **build context** did not contain what `Dockerfile.submission` copies in:

- **`lerobot_policy_dino/`** — not under `dummy_docker_policy/` (the package lived at `/data/lehome_workspace/lerobot_policy_dino` on the VM).
- **`requirements.txt`** — only `requirements.submission.template` existed, so a manual `docker build` had nothing to `COPY` as `requirements.txt`.

The later **“tag does not exist”** on `docker push` was because **`docker build` never produced an image**, not because Docker Hub lacked a tag.

### What was done on the VM

- Added **`requirements.txt`** in `dummy_docker_policy/` (same content as the template) so plain `docker build` works.
- **Copied** `/data/lehome_workspace/lerobot_policy_dino` → `dummy_docker_policy/lerobot_policy_dino/` so the tree matches the Dockerfile `COPY`.
- **Updated `build_docker_submission.sh`** so future runs automatically:
  - create `requirements.txt` from `requirements.submission.template` if it is missing;
  - stage `lerobot_policy_dino` from `../../lerobot_policy_dino` (workspace layout) into the policy dir when it is not already there.
- **Gitignored** `dummy_docker_policy/lerobot_policy_dino/` in `lehome-challenge/.gitignore` so the staged copy is not committed by mistake.

### Recommended commands on the GPU VM

After the W&B checkpoint is in `pretrained_model/`:

```bash
cd /data/lehome_workspace/lehome-challenge/dummy_docker_policy
./build_docker_submission.sh \
  --skip-download \
  --image-tag nninspaceexp/lehome_silicon-optimists:FullDP-v1
docker push nninspaceexp/lehome_silicon-optimists:FullDP-v1
```

Or repeat the manual flow: ensure `pretrained_model/` is populated (`download_wandb_model.py` if needed), then `docker build` + `docker push`.

**Note (VM vs missing files):** A separate build attempt failed with **no space left on device** while unpacking the CUDA base image. That is **host disk**, unrelated to missing `COPY` paths. On the GPU VM use `docker system prune` / free disk if needed. The `COPY …/lerobot_policy_dino: not found` class of errors is addressed by staging the package and having `requirements.txt`.

---

## Independent audit report (user-supplied)

**Audit scope:** Review of the original error (`/lerobot_policy_dino` not found + subsequent push failure), all changes made to resolve it, current file state, and whether the solution is production-ready for the LeHome challenge submission. Performed as a fresh, independent review.

### 1. Root cause assessment (confirmed)

- The original failure was correctly diagnosed: the Docker build context was missing `lerobot_policy_dino/` (and `requirements.txt`).
- The follow-on “tag does not exist” error on `docker push` was a **symptom**, not the root cause — no image was built, so nothing could be pushed.
- The question about manually creating tags on Docker Hub was answered correctly: **no** manual tag creation is needed; `docker push` creates the tag on Docker Hub automatically.

### 2. Changes made — quality assessment

**What was done well**

- **`requirements.txt`:** Created correctly from the template with appropriate comments and minimal dependencies. Satisfies the `COPY --chown=user requirements.txt .` step.
- **Package staging:** `lerobot_policy_dino/` was correctly copied from the workspace root (`/data/lehome_workspace/lerobot_policy_dino`) into `dummy_docker_policy/lerobot_policy_dino/`. The `COPY` instruction in the Dockerfile resolves.
- **`build_docker_submission.sh` enhancements:** Robust fallback to create `requirements.txt` if missing; smart staging for `lerobot_policy_dino` with clear errors and path resolution from workspace root; preserved original functionality (`--skip-download`, `--push`, etc.); good separation of concerns and helpful output.
- **`.gitignore` update:** Correctly added `dummy_docker_policy/lerobot_policy_dino/` to prevent accidentally committing the staged copy; well commented.
- **Non-destructive approach:** Changes are additive and improve maintainability.

**Issues identified (independent findings)**

- **Critical: disk space exhaustion (not addressed in code)**  
  The most recent build attempt failed with `ERROR: write .../libtorch_cuda.so: no space left on device`. This is a host-level problem (the PyTorch CUDA base image is large, on the order of multiple GB). The build will continue to fail until disk is cleaned (`docker system prune -a`, remove unused images/containers, check `/var/lib/docker` usage).

- **`requirements.txt` completeness**  
  The file at audit time only contained `numpy` + `opencv-python-headless`. The `lerobot_policy_dino/pyproject.toml` declares dependencies on `lerobot==0.4.3` and `transformers`. While the editable install (`pip install -e ...`) should pull these in, that is fragile for a submission; production submissions should use explicit pins for reproducibility (`lerobot`, `transformers`, `safetensors`, `einops`, etc.). The comment in the file acknowledges this but is easy to miss.

- **Minor documentation debt**  
  The Dockerfile still contained an old comment instructing manual `cp -r ... lerobot_policy_dino` (addressed in **Local implementation summary §6** in this artifact).

- **Build context and `.dockerignore`**  
  No verification was done on whether `.dockerignore` exists and excludes `pretrained_model/` or the staged package (could affect clean builds). The `cp -a` staging approach works but is not idempotent/optimal (always copies everything).

- **Script ordering**  
  The `PRETRAINED_PATH` variable was defined after the new staging block on the VM iteration; while it works, the script is slightly harder to follow than the original layout (local replication moved `PRETRAINED_PATH` earlier).

### 3. Overall verdict

- The file-missing errors are **resolved** at the code/layout level. Staging the package + `requirements.txt` + build script automation is technically sound.
- The setup is **not** fully “good to go” for a successful submission push if **disk space** is exhausted — that remains the blocking operational issue.

**Readiness score (audit):** 7.5/10

**Recommendations (audit, for reference)**

- Address disk space first (`docker system df`, `docker system prune -a --volumes`).
- Consider expanding `requirements.txt` with explicit pins from the training environment.
- Optionally update the outdated comment in `Dockerfile.submission`.
- The engineering decisions around staging and script automation were appropriate and well implemented; the core problem from the terminal log was fixed at the code level.

---

## Local implementation summary (this repo)

### 1. GitHub merge (plan step 0)

- Shallow clone of `https://github.com/lehome-official/lehome-challenge` (`main`), then:
  - `rsync -a --ignore-existing …/dummy_docker_policy/ → lehome_workspace/lehome-challenge/dummy_docker_policy/`
- **Added** upstream-only paths: `lehome_workspace/lehome-challenge/dummy_docker_policy/Dockerfile`, `README.md`.
- **Unchanged** (local wins): `Dockerfile.submission`, `policy.py`, `server.py`, `build_docker_submission.sh`, etc.

---

### 2. `requirements.txt`

- New file: `lehome_workspace/lehome-challenge/dummy_docker_policy/requirements.txt`
- Same pins as `requirements.submission.template` so `COPY requirements.txt` in `Dockerfile.submission` works for a plain `docker build`.

---

### 3. Staged `lerobot_policy_dino/`

- Directory: `lehome_workspace/lehome-challenge/dummy_docker_policy/lerobot_policy_dino/`
- Populated from `lehome_workspace/lerobot_policy_dino/` via `rsync -a`.

---

### 4. `build_docker_submission.sh`

- After `cd` into the policy dir: `PRETRAINED_PATH` is set first.
- **Auto** `requirements.txt` from `requirements.submission.template` if missing.
- **Auto** `rsync` of `../../lerobot_policy_dino` into the policy dir if that folder is missing or empty.
- Removed the fatal “missing requirements.txt” block; `--init-requirements` still works.
- `bash -n` passes; script remains executable.

---

### 5. `.gitignore`

- File: `lehome_workspace/lehome-challenge/.gitignore`
- Entry: `dummy_docker_policy/lerobot_policy_dino/` so the staged tree is not committed.

---

### 6. `Dockerfile.submission`

- Comment above the `COPY lerobot_policy_dino/` line updated to reference the build script / build context instead of a manual `cp -r`.

---

### 7. Workspace tracking

- `lehome_workspace/lehome_change_log.md` — entry **2026-04-30 22:10:00 UTC**.
- `lehome_workspace/vm_transfer_list.md` — **Last Updated** and refreshed lines for touched artifacts.

---

### Notes

- `pretrained_model/` is still required for `--skip-download` (or use W&B download). Docker build was not run in the local agent workspace (LeHome policy: heavy Docker/torch verification on the GPU VM).
- Upstream official `dummy_docker_policy/` on `main` is a minimal sample (`Dockerfile`, `README.md`); submission flow uses `Dockerfile.submission`.

---

### VM command reminder (generic)

After W&B checkpoint is in `pretrained_model/`:

```bash
cd /path/to/lehome_workspace/lehome-challenge/dummy_docker_policy
./build_docker_submission.sh --skip-download --image-tag YOUR_REGISTRY/IMAGE:TAG
docker push YOUR_REGISTRY/IMAGE:TAG
```

If builds fail with **no space left on device** while unpacking the CUDA base image, free disk on the host (`docker system prune`, etc.); that is not fixable in-repo.