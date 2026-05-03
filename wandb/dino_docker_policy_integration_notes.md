## LeHome `dummy_docker_policy` + custom DINOv2 MAP+Registers policy integration notes

This document is a development log of the changes made to:

- `lehome-challenge/dummy_docker_policy/policy.py`
- `lehome-challenge/dummy_docker_policy/Dockerfile.submission`

to run a custom LeRobot BYOP policy package (`/data/lehome_workspace/lerobot_policy_dino`) and a locally-downloaded checkpoint at:

- `lehome-challenge/dummy_docker_policy/pretrained_model/model.safetensors`

The goal was **straightforward, working evaluation inference** (organizers run the container; they bear compute cost).

---

### Context / requirements

- **Custom BYOP package**: `/data/lehome_workspace/lerobot_policy_dino`
  - Registers the policy type `dino_diffusion` via `@PreTrainedConfig.register_subclass("dino_diffusion")`.
- **Observation keys expected by evaluation server** (`server.py`):
  - `observation.images.top_rgb`  (H, W, 3) `uint8`
  - `observation.images.left_rgb` (H, W, 3) `uint8`
  - `observation.images.right_rgb`(H, W, 3) `uint8`
  - `observation.state`           (12,) `float32`
- **Tensor formatting requirement**:
  - Images: `(H, W, C) uint8` → `(1, C, H, W) float32` in `[0, 1]`
  - State: `(12,)` → `(1, 12)`
- **Training config used as ground truth**:
  - `configs/sweep_dino_map_registers.yaml` specifies:
    - DINO backbone: `facebook/dinov2-with-registers-small`
    - `spatial_pooling: map`, `map_num_queries: 8`, `use_registers: true`, `num_register_tokens: 4`
    - input features: 3 cameras + `observation.state`, output feature: `action` (12,)

---

### What was implemented

#### 1) `policy.py`

- Loads the model by:
  - Building a `DinoDiffusionConfig` in code (because the checkpoint directory contains **only** `model.safetensors` and **no** `config.json`).
  - Creating `DinoDiffusionPolicy(config)`.
  - Loading weights with `safetensors.torch.load_file(...)` + `policy.load_state_dict(..., strict=False)`.
- Implements `infer()`:
  - Converts the three camera images + state to a dict of batched tensors.
  - Calls `self.policy.select_action(obs_tensor)` (DiffusionPolicy provides this method).
  - Returns `[action.squeeze(0).cpu().numpy()]` to match the HTTP server contract.
- Implements `reset()`:
  - Calls `self.policy.reset()` to clear the diffusion queues per episode.

#### 2) `Dockerfile.submission`

- Copies the BYOP package into the image and installs it editable:
  - `COPY lerobot_policy_dino/ /app/lerobot_policy_dino/`
  - `RUN pip install -e /app/lerobot_policy_dino`
- Pre-caches the HuggingFace DINO backbone:
  - `RUN python3 -c "from transformers import Dinov2Model; Dinov2Model.from_pretrained('facebook/dinov2-with-registers-small')"`
  - This reduces risk of offline failures at evaluation time.

---

### Errors encountered and the grounded fixes

#### A) `ModuleNotFoundError: lerobot_policy_dino`

**Symptom**
- Running `policy.py` locally failed with:
  - `ModuleNotFoundError: No module named 'lerobot_policy_dino'`

**Root cause**
- The BYOP package must be installed (or copied + installed in Docker).

**Fix**
- Docker: `COPY` the directory and `pip install -e /app/lerobot_policy_dino`.
- Local dev: installed into the project venv using:
  - `uv pip install --python lehome-challenge/.venv -e /data/lehome_workspace/lerobot_policy_dino`

#### B) `FileNotFoundError: pretrained_model/model.safetensors`

**Symptom**
- Running from the wrong working directory caused:
  - `FileNotFoundError: pretrained_model/model.safetensors`

**Root cause**
- `pretrained_model_path="pretrained_model"` is relative; it must be resolved from the `dummy_docker_policy/` working directory (as in Docker).

**Fix**
- Run from `lehome-challenge/dummy_docker_policy/` or pass an absolute path.

#### C) `AttributeError: 'DinoDiffusionPolicy' object has no attribute '_queues'`

**Symptom**
- Calling `select_action()` without resetting first (in a bare test script) caused:
  - `AttributeError: ... no attribute '_queues'`

**Root cause**
- DiffusionPolicy uses internal deques; `reset()` initializes them.

**Fix**
- Ensure `reset()` is called at episode start. In the real server path, `/reset` triggers this.

#### D) `KeyError: 'observation.images.top_rgb'` inside `DinoDiffusionModel._prepare_global_conditioning`

**Symptom**
- After calling `select_action()`, inference crashed with:
  - `KeyError: 'observation.images.top_rgb'`

**Root cause (important)**
- In LeRobot’s upstream `DiffusionPolicy.select_action()`:
  - It **stacks** the individual image keys into a single key (commonly `OBS_IMAGES`) using:
    - `torch.stack([batch[key] for key in self.config.image_features], dim=-4)`
  - Then it fills queues and `predict_action_chunk()` stacks from queues and calls `diffusion.generate_actions(batch)`.
- In this custom package (`lerobot_policy_dino`):
  - `DinoDiffusionModel._prepare_global_conditioning()` expects **individual image keys** from `config.image_features` to be present in the batch:
    - `img = batch[img_key]  # (B, N_obs, C, H, W)`
- Result: once images were consolidated into `OBS_IMAGES`, the per-camera keys were missing at the point where `_prepare_global_conditioning()` looked them up.

**Fix applied (kept simple)**
- A small monkey patch in `policy.py` overrides `self.policy.diffusion._prepare_global_conditioning` to:
  - Detect `OBS_IMAGES` in `batch`
  - Unpack it back into the original per-camera keys listed in `self.policy.config.image_features`
  - Call the original implementation

**Verification**
- With the patch, a minimal CPU test (single step) produced:
  - `Action shape: (12,)`
  - `Success!`

#### E) GPU-specific local errors (not a submission logic bug)

**Symptom**
- On the local machine, CUDA errors appeared due to a GPU/torch capability mismatch (Blackwell `sm_120` vs installed torch build).

**Interpretation**
- This is an **environment mismatch**, not a policy integration bug.
- Organizers will run the container on their own machines; performance/compatibility depends on their host GPU/driver stack, and the base image includes CUDA runtime.

---

### Final state (what to expect)

- `policy.py` now:
  - Loads the MAP+Registers DINO diffusion policy from `model.safetensors`
  - Converts three RGB camera frames + state into the expected tensor dict
  - Calls `select_action()` and returns a 1-step action chunk as `List[np.ndarray]`
  - Applies the **minimal** monkey patch needed so the custom model works with upstream `DiffusionPolicy.select_action()` image stacking behavior
- `Dockerfile.submission` now:
  - Installs the BYOP package inside the image
  - Pre-caches the DINO backbone

---

### Build notes (important for Docker context)

The Docker build context is `lehome-challenge/dummy_docker_policy/`. To build successfully, ensure the BYOP package folder exists inside the build context (before `docker build`):

```bash
cp -r /data/lehome_workspace/lerobot_policy_dino \
      /data/lehome_workspace/lehome-challenge/dummy_docker_policy/lerobot_policy_dino
```

Then build as usual (example):

```bash
cd /data/lehome_workspace/lehome-challenge/dummy_docker_policy
./build_docker_submission.sh --skip-download --init-requirements
```

---

### References used to ground behavior

- **LeHome policy server observation contract**: `lehome-challenge/dummy_docker_policy/server.py`
- **Training config for shapes/architecture**: `configs/sweep_dino_map_registers.yaml`
- **Custom model input expectations**: `lerobot_policy_dino/src/lerobot_policy_dino/modeling_dino_diffusion.py`
- **LeRobot diffusion image stacking behavior**:
  - `lerobot.policies.diffusion.modeling_diffusion.DiffusionPolicy.select_action()`
  - Observed in installed site-packages under `lehome-challenge/.venv/...`

