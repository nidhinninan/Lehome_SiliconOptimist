# 10k Vision Sweep Reference Log — Errors & Bugs

> **Conversation:** [Implementing Time-Constrained Vision Benchmarking](file:///home/nidhinninan/.gemini/antigravity/brain/7b40dd64-4ec1-4e3a-818a-632da6cb704b/.system_generated/logs/overview.txt)
> **Scope:** Standardized, sequential evaluation of ResNet18, DINOv2, and CLIP backbones within the LeRobot framework.

---

## 🏗️ CLI & Orchestration Bugs

### Error 1 — Static Relative Paths
**Symptom:** Unzip worked, but `pip install` failed with "not a valid requirement" or "path not found" when executed from `/data/`.
**Root Cause:** The initial `run_10k_sweep.sh` used `lehome_workspace/` relative prefixes. When the user moved the script to `/data/lehome_workspace/` on the VM, the relative paths pointed to non-existent folders.
**Fix:** Implemented dynamic absolute path resolution:
```bash
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -e "$WORKSPACE_DIR/lerobot_policy_dino"
```
**Why it works:** BASH_SOURCE ensures the script finds its own parent directory, making it "execution-site independent."

### Error 2 — Global vs. Venv Alias Collision
**Symptom:** Script used the system's global `lerobot-train` which pointed to an outdated Python 3.10 installation instead of the project's Python 3.12 venv.
**Root Cause:** The Principia VM has pre-installed global aliases. Calling `lerobot-train` directly bypassed the local virtual environment.
**Fix:** Switched from CLI commands to directly invoking the venv Python:
```bash
VENV_PYTHON="$WORKSPACE_DIR/lehome-challenge/.venv/bin/python"
"$VENV_PYTHON" "$WORKSPACE_DIR/lerobot_train_with_plugins.py" ...
```

---

## ⚡ Environment & Hardware Failures

### Error 3 — Shared Memory Bus Error
**Symptom:** Training would start and then instantly crash with a "Bus Error" or "SIGBUS" during the first few steps.
**Root Cause:** LeRobot handles high-resolution images. With `num_workers=4`, the default Linux shared memory (`/dev/shm`) was often limited to 64MB or 512MB, which is insufficient for image buffers.
**Fix:** Added a remounting block to the script:
```bash
sudo mount -o remount,size=2G /dev/shm
```
Added a fallback to `num_workers=0` if sudo is unavailable to prevent the crash at the cost of speed.

### Error 4 — `df` Compatibility
**Symptom:** `invalid option -- 'K'` error on the VM.
**Root Cause:** The script used `df -PK`, but the specific Unix flavor on the VM only supported lowercase `-k` for 1024-byte block size.
**Fix:** Changed flag to `df -Pk`.

---

## 🧩 BYOP & Registry Failures

### Error 5 — Registry "Silent Fail" (Namespace Trap)
**Symptom:** Script logs `✅ DINO Plugin Registered`, but then crashes with `KeyError: 'dino_diffusion'`.
**Root Cause:** Python's "Implicit Namespace Packages." Because the folder `lerobot_policy_dino` existed in the CWD, `import lerobot_policy_dino` succeeded by loading the FOLDER, but it never executed the `src/` code or the `@register_subclass` decorator.
**Fix:**
1. Added `[build-system]` to `pyproject.toml` to fix editable installs.
2. In the bootstrapper, used `sys.path.insert(0, ...)` to prioritize the `src/` directory.

### Error 6 — LeRobot v0.4.3 API Mismatch
**Symptom:** Multiple `ImportError` or `ModuleNotFoundError`.
**Root Cause:**
*   `lerobot.common` was flattened in the pip package.
*   `Normalize/Unnormalize` moved to `lerobot.processor`.
**Fix:**
*   Hardcoded `OBS_ROBOT = "observation.state"` in the modeling files.
*   Updated imports to `from lerobot.processor import NormalizerProcessorStep as Normalize`.

---

## 📊 Data Pipeline Issues

### Error 7 — Dataset Mapping Mismatch
**Symptom:** `FileNotFoundError: Datasets/example/top_long_merged/meta/info.json`.
**Root Cause:** The `.yaml` configs had relative `root:` paths. When training started, LeRobot looked relative to the CWD, not the workspace root.
**Fix:** User manually updated YAML `root` to absolute paths on the VM. We added `🔗 LINKED DEPENDENCY` comments to ensure this is checked first.

### Error 8 — Augmentation Bottleneck
**Symptom:** `data_s` (0.6s) was much higher than `updt_s` (0.3s).
**Root Cause:** Image transforms were enabled globally in the script, causing the CPU to struggle with 640x480 transforms every batch.
**Fix:** Disabled `image_transforms.enable` in both YAMLs and the CLI flag to ensure fair benchmarking against the ResNet baseline.

---

## 📈 Success Summary Table

| Issue Category | Root Cause | Solution | Result |
| :--- | :--- | :--- | :--- |
| **Pathing** | Relative paths in shell | Dynamic `$WORKSPACE_DIR` resolution | Executable from any path |
| **Environment** | SHM exhaustion | `sudo mount` remounting + worker fallback | Zero "Bus Error" crashes |
| **Discovery** | Registry registration failure | Bootstrapper for manual import injection | Custom policies load correctly |
| **Compatibility** | LeRobot v0.4.3 refactor | Updated import paths to `lerobot.processor` | API Alignment |
| **Performance** | CPU transform overhead | Disabled transforms for sweep | GPU-saturated training |
