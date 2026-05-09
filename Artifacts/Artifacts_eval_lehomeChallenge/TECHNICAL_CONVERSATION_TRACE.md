# Technical conversation trace — LeHome Challenge 2026 (Silicon Optimists)
### Scope
This document is a **technical reconstruction** of the questions, answers, debugging steps, and code changes made during the “Step 8 onward” evaluation workflow for the Silicon Optimists LeHome submission.

It is written to preserve:
- The **Q/A structure** of the chat (what was asked, what was answered).
- The **reasoning chain** (what evidence was observed, what hypotheses were tested, what was concluded).
- The **specific failure modes** and the **minimal fixes** that were adopted.

### Safety / redaction policy
The original chat included credentials/tokens in pasted text and `.env`. Those are **not reproduced here**. Any secrets are replaced with `REDACTED`.

---

## Background context (what the workflow is doing)

### The evaluation architecture
- **Simulator side**: `python -m scripts.eval` inside the `lehome-challenge` repo runs IsaacLab/IsaacSim and produces observations (robot state + camera images).
- **Policy side**: A Docker container exposes an HTTP server:
  - `POST /reset` — reset episode state
  - `POST /infer` — receive observation dict, return action(s)
- **Glue**: `scripts/eval_policy/docker_policy.py` serializes observations, POSTs them to the container, and caches action chunks.

### What makes this submission special
The policy inside the container is a LeRobot diffusion policy variant with:
- A **frozen DINOv2 backbone** (`facebook/dinov2-with-registers-small`)
- A **MAP head** (multi-query attention pooling)
- A diffusion action generator that uses a U-Net conditioned on:
  - robot state history
  - per-camera DINO features

---

## Q/A + Timeline (Step 8 onward)

### Q1 — “EXECUTE FROM STEP 8”
**Question asked**
- User: “EXCDUCTE FROM STEP 8” (start policy servers, then evaluation).

**Answer / action taken**
- Started the policy server containers as specified by Step 8, but used detached mode (`-d`) to avoid needing two interactive terminals.
- Verified they reached readiness: `Policy server listening on 0.0.0.0:8080`.

**Evidence**
- `docker logs` showed the listening line on both containers.

**Note**
- A warning appeared: “model type dinov2_with_registers instantiating dinov2”. This was treated as non-fatal and consistent with prior runs.

---

### Q2 — “next step”
**Question asked**
- User: “next step”

**Answer / action taken**
- Proceeded to Step 9: run evaluations pointing to the two policy servers.

**What broke**
Two independent issues appeared immediately:

1) **IsaacLab “kit flags” passed incorrectly**
- The README used raw Kit flags in the eval command:
  - `--/renderer/multiGpu/enabled=false`
  - `--/renderer/activeGpu=0`
- The `scripts.eval` argparse rejected these as unknown:
  - `eval.py: error: unrecognized arguments: --/renderer/multiGpu/enabled=false --/renderer/activeGpu=0`

2) **`xvfb-run: python: not found`**
- When launching with `xvfb-run`, the process couldn’t resolve `python`.

**Why these happened**
- `scripts.eval` only accepts Kit args via `--kit_args "<...>"` because IsaacLab wires Kit arguments through an explicit CLI flag.
- `xvfb-run` executes in a non-interactive environment; relying on `python` being on PATH is brittle. Using the venv’s interpreter is reliable.

**Corrections adopted**
- Use:
  - `.venv/bin/python -m scripts.eval` (explicit interpreter)
  - `--kit_args "--/renderer/multiGpu/enabled=false --/renderer/activeGpu=0"` (proper Kit arg pass-through)

---

### Q3 — “Continue / evaluation runs but RemoteDisconnected”
**Question asked**
- User: “continue”

**Answer / action taken**
- Restarted eval using `.venv/bin/python` and `--kit_args ...`.

**What broke**
- Both evals reached:
  - policy connection success
  - garment list loaded
- Then failed with:
  - `http.client.RemoteDisconnected: Remote end closed connection without response`

**Root-cause analysis**
This error is a *client symptom*. The actual exception was inside the policy server.

**Evidence**
Docker container logs showed:
- `KeyError: 'observation.images.top_rgb'` thrown during inference, causing the server to crash the request handler and close the connection.

**Interpretation**
- The simulator *was* sending camera images.
- But the policy stack internally could not find the expected per-camera key at the stage it attempted to use it.

---

### Q4 — “Why is the server missing 'observation.images.top_rgb'?”
**Question asked (implicit)**
- User: proceed, fix Step 9 failures.

**Answer / action taken**
Investigated both sides:

1) **Environment observation keys**
- `source/lehome/lehome/tasks/bedroom/garment_bi_v2.py::_get_observations` produces keys:
  - `observation.images.top_rgb`
  - `observation.images.left_rgb`
  - `observation.images.right_rgb`
  - plus state and depth

2) **Eval docker client serialization**
- `scripts/eval_policy/docker_policy.py` serializes any key containing `"images"` or `"depth"` as base64 blobs, so images should be preserved.

3) **Policy server code in the image**
- `/app/policy.py` (inside container) explicitly reads `_IMAGE_KEYS` including `observation.images.top_rgb` and converts them to torch tensors.

So the KeyError could not be explained by the simulator/client/server boundary.

**Actual root cause (deeper)**
The KeyError was not thrown by `policy.py` when reading `observation[...]`; it was thrown inside the LeRobot diffusion model implementation:
- In `/app/lerobot_policy_dino/.../modeling_dino_diffusion.py`
- specifically in `_prepare_global_conditioning`, where it attempted:
  - `img = batch[img_key]`
  - for `img_key in self.config.image_features`

**Why those keys were missing in that internal `batch`**
Upstream LeRobot’s `DiffusionPolicy` does this in `select_action`:
- It stacks per-camera keys into a single `observation.images` tensor (OBS_IMAGES) and populates queues keyed on:
  - `observation.state`
  - `action`
  - `observation.images` (single stacked key)
- In `predict_action_chunk`, it rebuilds a batch only using keys present in `_queues`.

Therefore, if a custom diffusion model expects **per-camera keys** at generation-time, but the parent policy only keeps **OBS_IMAGES**, those per-camera keys will be dropped before generation.

**Conclusion**
- The initial Docker image had an internal mismatch:
  - custom conditioning expected `observation.images.top_rgb` keys
  - but inherited queueing logic only kept `observation.images`

**Fix adopted (queue fix)**
Modify `DinoDiffusionPolicy` (the custom policy class) to override:
- `reset()` — create queues for each `img_key in config.image_features`
- `select_action()` — populate those queues without collapsing into a single key
- `predict_action_chunk()` — stack from those per-camera queues into `(B, N_obs, C, H, W)` tensors per key

This fix does **not** change the trained weights; it changes only inference-time data plumbing.

---

### Q5 — “Start servers, run eval again; now device mismatch”
**Question asked**
- User continued to request evaluation progression and new image build.

**Answer / action taken**
- Built a new image including the queue fix.
- Restarted a single-port evaluation workflow.

**What broke**
Policy server crashed during the diffusion loop with:
- `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`

**Why this happened**
In the custom diffusion implementation, `generate_actions` used:
- `for t in self.noise_scheduler.timesteps:`
- and passed `t` directly into the U-Net:
  - `self.unet(sample, t, global_cond=...)`

In diffusers schedulers, `noise_scheduler.timesteps` is commonly stored on **CPU** (or as Python scalars).
But `sample` and the U-Net parameters are on **CUDA** in the Docker container.

Upstream LeRobot avoids this by converting each step to a CUDA tensor shaped `(B,)`, e.g.:
- `torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device)`

**Fix adopted (timestep device fix)**
In `DinoDiffusionModel.generate_actions`:
- replace the raw `t` argument to the U-Net with `t_batch` on the same device as `sample`.

Again, this is **inference-only** plumbing and does not require retraining.

---

### Q6 — “Do we need GPU for inference? Can we avoid changes?”
**Question asked**
- User: “but don't you need gpu for inference”
- User preference: avoid changing code unless necessary.

**Answer**
- Running on CPU is technically possible but practically too slow for diffusion + DINOv2 (many denoising steps per action).
- Therefore, the minimal code fix was deemed **necessary** to make GPU inference functional.

---

### Q7 — “Exhaustive review of dummy_docker_policy; don’t break trained policy”
**Question asked**
- User: “review entire dummy_docker_policy… exhaustive… identify what needs to be changed… don’t do changes that make trained policy unusable”

**Answer / conclusions**
Key findings from review:

1) **`server.py` is correct and should not be modified**
- It defines the protocol and base64 decoding into numpy arrays.

2) **`policy.py` is a thin adapter**
- Converts numpy arrays to torch tensors
- Calls `policy.select_action`
- Correctly sets device (`cuda` if available)

3) **Normalization is not used at inference**
- The checkpoint (`pretrained_model/model.safetensors`) contained only `diffusion.*` keys and **no normalize/unnormalize buffers**.
- Therefore, introducing normalization at inference would likely diverge from the trained behavior.

4) The only necessary changes were:
- queue fix (per-camera keys preserved through rollout queues)
- timestep device fix (avoid cuda/cpu mismatch during denoising)

Both fixes are **runtime correctness** changes and do not alter the trained parameters.

---

## Code changes adopted (what, where, why)

### 1) Eval-side Kit args handling (host repo)
**Problem**
- Raw Kit flags passed directly to `scripts.eval` are rejected by argparse.
- Additionally, `scripts/utils/common.py` overwrote `args.kit_args`, discarding user-provided `--kit_args`.

**Fix**
- Merge user `--kit_args` with log-level flags inside `launch_app_from_args`.
- This ensures the evaluator can pass Kit settings without patching IsaacLab.

**Why this was required**
- Without it, the “renderer stability flags” cannot be used via the documented CLI.

---

### 2) Queue fix inside BYOP package (`lerobot_policy_dino`)
**Problem**
- Upstream `DiffusionPolicy` queues only `observation.images` (stacked key).
- Custom conditioning iterated per-camera keys, which were missing at generation time → `KeyError`.

**Fix**
- Override queue creation and action selection to store per-camera keys.

**Why safe**
- It changes only how observations are buffered/stacked; it does not modify model weights or feature dimensions.

---

### 3) Timestep device fix inside BYOP package (`lerobot_policy_dino`)
**Problem**
- CPU timesteps fed into CUDA U-Net → `Expected all tensors to be on the same device`.

**Fix**
- Convert each timestep into a CUDA tensor shaped `(B,)` on `sample.device`.
- Mirrors upstream LeRobot diffusion implementation.

**Why safe**
- No parameter changes; only dtype/device correctness for the timestep conditioning input.

---

## Image management: ports, tags, and pushing strategy

### Ports
Policy servers are exposed as container port 8080.
Host-side choices:
- Single evaluation: map `-p 8080:8080`
- Parallel eval: map `-p 8081:8080` and `-p 8082:8080`

### Tag semantics on Docker Hub
- A Docker tag (e.g. `...:FullDP-v2_Blackwell`) is a **mutable pointer** to a manifest digest.
- Pushing again to the same tag **moves the pointer** to the new digest.
- Old digests remain addressable by digest, but most evaluators pull by tag.

**Operational recommendation**
- Use immutable-ish tags for reproducibility (e.g. `...:fixed-queues-v2`) and optionally update the “canonical” tag for convenience.

---

## Final state achieved in this chat

1) Identified two independent inference blockers:
- **per-camera key drop** in inherited diffusion queues
- **CPU timestep** passed into CUDA U-Net

2) Implemented minimal fixes that:
- do not require retraining
- preserve checkpoint compatibility
- enable GPU inference and evaluation

3) Built and pushed updated images, and managed ports/containers cleanly as requested.

---

## Appendix: key runtime error signatures (for future triage)

### A) Client sees RemoteDisconnected
Symptom:
- `http.client.RemoteDisconnected: Remote end closed connection without response`

Likely causes:
- Policy server threw an exception during `/infer` and crashed the handler.

How to confirm:
- `docker logs <container>` and look for Python tracebacks.

### B) KeyError on per-camera image key
Symptom in policy logs:
- `KeyError: 'observation.images.top_rgb'`

Root cause class:
- Observation keys dropped/renamed between the HTTP adapter and the diffusion model’s conditioning builder.

### C) CUDA/CPU mismatch on timestep
Symptom in policy logs:
- `Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`

Root cause class:
- Scheduler timesteps (CPU) passed directly into CUDA model layers.

