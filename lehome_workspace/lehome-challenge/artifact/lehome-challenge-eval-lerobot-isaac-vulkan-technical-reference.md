# LeHome challenge — technical reference (eval, LeRobot, Isaac Sim, Vulkan)

**Canonical file:** `artifact/lehome-challenge-eval-lerobot-isaac-vulkan-technical-reference.md`

Consolidated findings, errors, solutions, and reasoning from debugging LeHome evaluation (`scripts.eval`), LeRobot custom policies, Isaac Sim / Omniverse startup, and GPU/Vulkan environment notes.

**Repository / git:** At document creation time, `lehome-challenge` had no `.git`; a local repo was initialized on **`main`** with branch **`docs/artifact-lehome-eval`** for tracking documentation-only work. Nested repos (e.g. `third_party/IsaacLab/.git`) remain independent.

---

## 1. Terminal session summary (`terminals/5.txt`)

| Item | Value |
|------|--------|
| **cwd** | `/data/lehome_workspace/lehome-challenge` |
| **Command pattern** | `WARP_CACHE_PATH=... OMNI_KIT_ACCEPT_EULA=yes PYTHONUNBUFFERED=1 xvfb-run -a .venv/bin/python -u -m scripts.eval ... --policy_type lerobot --policy_path .../dummy_docker_policy/pretrained_model ... --device cpu --headless --enable_cameras --num_episodes 1` |

**Observed sequence:**

1. Long Omniverse / Isaac Sim extension startup (`omni.*`, `isaacsim.*`, `isaaclab.*`).
2. **Failure A:** `config.json not found` under `--policy_path`; then Draccus parsing failed with empty config (`Expected a dict with a 'type' key ... got {}`).
3. User interrupted once (`^C`), cleared `pretrained_model`, **rclone copy** of real checkpoint `pretrained_model` (~2.979 GiB, 7 files).
4. **Failure B:** `Couldn't find a choice class for 'dino_diffusion' in PreTrainedConfig` — config existed but custom policy type was not registered in-process.
5. **Vulkan / GPU warning:** “Multiple Installable Client Drivers (ICDs)” — same physical GPU listed twice; instability risk.

---

## 2. Bug → solution → reasoning matrix

### 2.1 Missing or invalid LeRobot checkpoint directory

| | |
|--|--|
| **Bug** | `--policy_path` pointed at an empty or wrong folder: no `config.json`, or incomplete contents. Symptom: `config.json not found` / `{}` parse error. |
| **Solution** | Point `--policy_path` at the **actual** `pretrained_model/` directory from a real checkpoint (must include `config.json` with `"type": ...`, weights such as `model.safetensors`, processors if required). Populate via `rclone copy` or equivalent. Optional code hardening: validate folder **before** `PreTrainedConfig.from_pretrained`. |
| **Reasoning** | LeRobot loads config via Draccus from `config.json`. Missing file → empty parse → misleading `{}` error. |

### 2.2 Unknown policy type `dino_diffusion` (plugin not loaded)

| | |
|--|--|
| **Bug** | `PreTrainedConfig.from_pretrained` runs **before** custom `@register_subclass` hooks execute. Config says `"type": "dino_diffusion"` but only stock LeRobot types are registered → `KeyError` / “Couldn't find a choice class”. |
| **Solution** | **Option A:** Import plugins at eval entry (`scripts/eval.py`). **Option B:** Import in `scripts/eval_policy/lerobot_policy.py` before `from_pretrained`. **Option C:** Wrapper script (mirrors training). Implemented approach: **`lerobot_eval_with_plugins.py`** — prepend `lerobot_policy_dino/src` and `lerobot_policy_clip/src` to `sys.path`, `import lerobot_policy_dino` and `lerobot_policy_clip`, reject namespace-only imports (`__file__ is None`), then call `scripts.eval.main`. Same pattern as **`/data/lehome_workspace/lerobot_train_with_plugins.py`** for `lerobot_train`. |
| **Reasoning** | Registration is side-effect of import; order matters. Training wrapper already solved discovery vs. normalized package names / blank namespace folders. |

### 2.3 Warp / Isaac Sim (`warp.types.array` / import failures)

| | |
|--|--|
| **Bug** | Incompatible `warp-lang` vs. Isaac Sim bundled Warp expectations → crashes during task/sim imports. |
| **Solution** | Pin **`warp-lang`** to an Isaac-compatible version in **`pyproject.toml`** → `[tool.uv].override-dependencies` includes `warp-lang==1.11.1` (project-specific pin). Re-sync venv with project lock (`uv sync` / install policy). |
| **Reasoning** | Isaac Sim ships/extensions expect a consistent Warp ABI; “latest” Warp often breaks Omniverse extensions. |

### 2.4 Hugging Face download failure during eval (post-plugin fix)

| | |
|--|--|
| **Bug** | After plugins register and config decodes, **`Dinov2Model.from_pretrained(config.vision_backbone)`** pulls `facebook/dinov2-with-registers-small` from Hugging Face. Environment returned **`ProxyError` / `Tunnel connection failed: 403 Forbidden`** — eval aborted during policy construction. |
| **Solution** | Ensure outbound HTTPS to `huggingface.co` works, or set **`HF_HUB_OFFLINE=1`** with models cached under `HF_HOME`/`~/.cache/huggingface`, or vendor backbone weights locally and point `vision_backbone` / cache to local paths; fix corporate proxy env vars (`HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY`). |
| **Reasoning** | Vision backbone is not always fully embedded in the policy checkpoint; Transformers may still resolve hub IDs at init. |

### 2.5 Vulkan / GPU / NVML (Isaac Sim RTX path)

| | |
|--|--|
| **Symptoms** | “Multiple ICDs” duplicate GPU; **`NVML_ERROR_DRIVER_NOT_LOADED`**; “No device could be created”; **`Driver Version: 0`** in some runs (e.g. sandbox/containers without GPU passthrough). |
| **Quick mitigations** | (1) **`export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`** (or `/usr/share/vulkan/icd.d/nvidia_icd.json`) so only one NVIDIA ICD is used. (2) On host: **`nvidia-smi`**, **`ls /dev/nvidia*`** — if missing, fix NVIDIA driver + container runtime (`--gpus all`, `nvidia-container-toolkit`). (3) Isaac Sim **`user.config.json`**: set **`activeGpu`** and disable multi-GPU if duplicates persist. |
| **Reasoning** | Duplicate ICDs enumerate the same GPU twice → Vulkan/RT instability. NVML/driver 0 means no usable NVIDIA stack in that process namespace — not fixable by Vulkan env alone. |

### 2.6 `training_state/` for evaluation

| | |
|--|--|
| **Clarification** | **Not required** for inference-only eval unless resuming training. |

---

## 3. Files and commands reference

- **Training plugin wrapper:** `/data/lehome_workspace/lerobot_train_with_plugins.py` — registers DINO/CLIP plugins then `lerobot.scripts.lerobot_train.main`.
- **Eval plugin wrapper:** `lehome-challenge/lerobot_eval_with_plugins.py` — registers plugins then `scripts.eval.main`.
- **Policy validation (fail-fast):** `scripts/eval_policy/lerobot_policy.py` — checks `config.json` and weight files before `PreTrainedConfig.from_pretrained`.
- **Eval entry:** `scripts/eval.py` — Isaac Lab launcher, imports `lehome.tasks.bedroom`, runs `eval()`.

**Example eval invocation (with plugins):**

```bash
cd /data/lehome_workspace/lehome-challenge
WARP_CACHE_PATH=/data/warp_cache OMNI_KIT_ACCEPT_EULA=yes PYTHONUNBUFFERED=1 \
xvfb-run -a .venv/bin/python -u lerobot_eval_with_plugins.py \
  --policy_type lerobot \
  --policy_path .../pretrained_model \
  --dataset_root ... \
  --garment_type top_long \
  --step_hz 0 \
  --device cpu \
  --headless \
  --enable_cameras \
  --num_episodes 1
```

---

## 4. Discussion thread compression (user-requested follow-ups)

- **Report items (original):** (1) Pin `warp-lang` for Isaac compatibility. (2) Load custom LeRobot policy plugins before config decode — options A/B/C. (3) Correct `--policy_path` layout. (4) Omit `training_state/` for eval.
- **Vulkan “quick solution”:** Force single ICD via `VK_ICD_FILENAMES`; separate issue from **driver/NVML/docker GPU visibility** — fix runtime + drivers if `nvidia-smi` fails.
- **Artifact instructions:** Maintain this document at **`artifact/lehome-challenge-eval-lerobot-isaac-vulkan-technical-reference.md`**; future edits should **append** new dated sections rather than rewriting prior sections.

---

## Appendix: subsection — Session artifact appendum (consolidated chat outcomes)

**Title:** LeHome eval pipeline — errors, fixes, and environment reasoning (May 2026)

This appendix records the resolution narrative in execution order:

1. **Empty checkpoint path** → user populated `pretrained_model` via **rclone**; resolves missing `config.json`.
2. **`dino_diffusion` unknown** → root cause **plugin registration order**; fix **wrapper imports** aligned with `lerobot_train_with_plugins.py`.
3. **Warp** → **project-level pin** `warp-lang==1.11.1` under uv overrides.
4. **Next blocker after plugins** → **Hugging Face hub / proxy** during DINOv2 backbone load; offline/cache/proxy strategy.
5. **Vulkan / duplicate ICD / NVML** → **ICD selection env** vs **driver/container** diagnosis.

---

*End of artifact (initial version). Append future findings below this line.*

---
