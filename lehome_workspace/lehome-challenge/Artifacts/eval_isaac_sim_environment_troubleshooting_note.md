# LeHome Evaluation: Isaac Sim Environment Troubleshooting Note

**Artifact purpose:** Capture root causes, fixes, and reasoning from troubleshooting LeHome `scripts.eval` runs (Isaac Lab + Isaac Sim + LeRobot) on headless Linux.

**Scope:** Configuration and dependency issues—not application logic bugs in LeHome itself unless noted.

**Environment reference:** `isaacsim==5.1.0`, Isaac Lab dev installs under `third_party/IsaacLab`, Python 3.11 in `.venv`, `uv` for dependency management.

---

## 1. “No output” / empty log while eval appears hung

### Observed behavior

- Redirecting stdout to a file (e.g. `> eval.log` or `tee`) produced an empty or nearly empty log for a long time.
- Alternatively, the process blocked early during startup.

### Root causes

1. **Omniverse Kit EULA gate (blocking)**  
   Importing Isaac Sim pulls in `isaacsim`, which bootstraps Kit and runs `check_eula()` in `isaacsim/kit/kit_app.py`. If neither `OMNI_KIT_ACCEPT_EULA` nor a persistent `EULA_ACCEPTED` marker is satisfied, the runtime calls **`input("Do you accept the EULA? (Yes/No): ")`**.  
   In non-interactive contexts (piped stdin, background jobs, some IDE terminals), nothing answers that prompt, so the process **blocks indefinitely**—often mistaken for a hang.

2. **Stdout buffering when redirected**  
   With stdout attached to a file or pipe, Python may use **block buffering**. Early prints may not flush to disk until the buffer fills or the process exits, so logs look empty while work is happening.

### Classification

- **Bug:** Only in the sense of **misconfiguration / unsuitable environment for interactive EULA**—not a defect in LeHome’s Python code.
- **Operational fix:** Non-interactive acceptance + unbuffered output.

### Recommended invocation pattern

```bash
export OMNI_KIT_ACCEPT_EULA=yes   # or pass once per command prefix
export PYTHONUNBUFFERED=1
# ...
/path/to/.venv/bin/python -u -m scripts.eval ...
```

**Reasoning:** `OMNI_KIT_ACCEPT_EULA=yes` satisfies Kit’s check without `input()`. `PYTHONUNBUFFERED=1` and `python -u` force line-oriented streaming so `tee` and log files reflect progress immediately.

---

## 2. Virtual framebuffer (`xvfb-run`) for headless hosts

### Context

Users running without a real display sometimes wrap eval with:

```bash
xvfb-run -a .venv/bin/python -u -m scripts.eval ...
```

### Classification

**Operational choice**, not a bug fix for IsaacLab itself. Use when the stack or extensions expect an X display and the machine has none; omit when a real display or vendor-supported headless path is sufficient.

---

## 3. `AttributeError: module 'warp.types' has no attribute 'array'` (and related Warp errors)

### Observed behavior

During Simulation App / extension startup, logs showed errors such as:

- `AttributeError: module 'warp.types' has no attribute 'array'`
- `AttributeError: module 'warp' has no attribute 'context'`
- Cascade failures loading `omni.replicator.*`, `isaaclab_assets`, `isaaclab_tasks`, then evaluation failing when importing Isaac Lab stacks.

### Root cause

The **`warp-lang`** PyPI package evolves its public API. Newer releases (e.g. **1.13.x**) reorganized or removed symbols that **Isaac Sim 5.1 / bundled Omniverse extensions** still reference at import time (e.g. type annotations and helpers expecting `wp.types.array`, `wp.context`).

Dependency resolution without an upper bound can install **latest** `warp-lang`, which is **incompatible** with the Isaac Sim 5.1 extension set shipped in the environment.

### Classification

- **Dependency version mismatch** between **pinned Isaac Sim** and **floating `warp-lang`**—a **real integration bug class** for this stack until NVIDIA alignsSim + Warp pins in docs or defaults.

### Solution adopted

1. **Pin `warp-lang`** to a version known to expose the APIs Isaac Sim extensions expect—for this workspace, **`warp-lang==1.11.1`** was validated as compatible (includes deprecation shims such as `warp.types.array` accessible via lazy `__getattr__`).
2. Record the pin in **`pyproject.toml`** under `[tool.uv] override-dependencies`:

```toml
override-dependencies = [
    "packaging==23.0",
    "numpy==1.26.0",
    "warp-lang==1.11.1",
]
```

**Reasoning:** `uv sync` and future installs must not upgrade `warp-lang` past the Sim-compatible line. Overrides keep the resolver deterministic for this repo.

**Follow-up:** After changing pins, run `uv lock` / `uv sync` when network access to configured indexes (PyPI, NVIDIA) is available so `uv.lock` stays consistent.

---

## 4. `PermissionError` writing Warp kernel cache (`/root/.cache/warp`)

### Observed behavior

After Warp initializes (`wp.init()` paths inside Isaac Lab), failure such as:

```text
PermissionError: [Errno 13] Permission denied: '/root/.cache/warp'
```

### Root cause

Warp defaults its kernel cache directory to a user cache path (via `WARP_CACHE_PATH` or system user cache). In restricted containers or hardened root environments, creating or writing under `/root/.cache` may fail.

### Solution

Point cache to a writable location (e.g. on mounted `/data`):

```bash
export WARP_CACHE_PATH=/data/warp_cache   # example; directory must exist and be writable
mkdir -p "$WARP_CACHE_PATH"
```

**Reasoning:** Matches Warp’s documented resolution order; avoids permission issues without changing application code.

---

## 5. `ModuleNotFoundError: No module named 'isaaclab'`

### Observed behavior

Running `python -m scripts.eval` with the **system** interpreter or any Python **outside** the project venv.

### Root cause

`setup_lehome.sh` installs Isaac Lab **editable** into **`lehome-challenge/.venv`**:

```bash
uv pip install -e "$REPO_DIR/third_party/IsaacLab/source/isaaclab" ...
```

That package is **not** on the global/system Python path.

### Classification

**User environment error** (wrong interpreter)—not missing install if setup completed successfully.

### Solution

Always invoke eval with the venv interpreter:

```bash
cd /path/to/lehome-challenge
./.venv/bin/python -m scripts.eval ...
# or: source .venv/bin/activate && python -m scripts.eval ...
```

**Reasoning:** Guarantees `isaaclab`, `isaacsim`, and `lehome` site-packages match the setup script.

---

## 6. Working directory and `python -m scripts.eval`

### Nuance

The package `scripts` must be importable as a top-level module. That requires either:

- Running from the **repository root** (`lehome-challenge`) so `.` is on `sys.path`, or  
- Setting `PYTHONPATH` to that root explicitly.

If the shell’s current working directory is elsewhere (e.g. `/data`), `python -m scripts.eval` can fail with **`No module named 'scripts'`** even when `isaaclab` is installed.

**Mitigation:** `cd` into `lehome-challenge` before `-m scripts.eval`, or document `PYTHONPATH`.

---

## 7. LeRobot policy load failure (`config.json` missing)

### Observed behavior (from logs after Sim started successfully)

```text
config.json not found in .../dummy_docker_policy/pretrained_model
draccus.utils.ParsingError: Expected a dict with a 'type' key for PreTrainedConfig, got {}
```

### Root cause

LeRobot’s `PreTrainedConfig.from_pretrained()` expects a valid **`config.json`** (and related artifacts) under `--policy_path`. The dummy or placeholder directory did not contain that file.

### Classification

**Data / checkpoint layout issue**—not an Isaac Sim or Isaac Lab installation bug.

### Solution

Point `--policy_path` at a real LeRobot export (e.g. under `outputs/train/.../pretrained_model` with `config.json`), or restore the expected files into the policy directory.

---

## 8. GPU / Vulkan / driver messages (informational)

Logs may include:

- **NVML / driver not loaded** in environments without NVIDIA drivers (e.g. some sandboxes).
- **Multiple Vulkan ICDs** warning duplicate GPU enumeration—driver/stack hygiene on multi-GPU hosts.
- **CUDA errors** when no suitable GPU is present but CUDA libraries load.

These affect rendering and PhysX/GPU paths; **CPU-only** or **headless** runs may still proceed with degraded graphics depending on configuration. Treat as **environment capability** issues, distinct from the Python import/Warp/EULA problems above.

---

## 9. Summary table

| Symptom | Likely cause | Mitigation |
|--------|----------------|------------|
| Empty log, no progress | EULA `input()` block + buffering | `OMNI_KIT_ACCEPT_EULA=yes`, `PYTHONUNBUFFERED=1`, `python -u` |
| `warp.types` / `warp.context` AttributeError | `warp-lang` too new for Isaac Sim 5.1 | Pin `warp-lang==1.11.1` (uv override) |
| Permission on `/root/.cache/warp` | Warp cache not writable | `WARP_CACHE_PATH` to writable dir |
| `No module named 'isaaclab'` | Wrong Python (not `.venv`) | Use `.venv/bin/python` or activate venv |
| `No module named 'scripts'` | Wrong cwd / PYTHONPATH | Run from repo root or set `PYTHONPATH` |
| `config.json not found` | Invalid `--policy_path` for LeRobot | Use complete pretrained checkpoint dir |

---

## 10. References (in-tree)

- Setup automation: `setup_lehome.sh` (uv sync, editable Isaac Lab + LeHome installs).
- Eval entrypoint: `scripts/eval.py` → `isaaclab.app.AppLauncher`, then task import.
- Omniverse EULA check implementation: `isaacsim/kit/kit_app.py` (`OMNI_KIT_ACCEPT_EULA`, `EULA_ACCEPTED`).
- Warp cache resolution: `warp/config.py` (`WARP_CACHE_PATH`).

---

*Document generated as a project artifact to preserve troubleshooting context for LeHome Challenge evaluation on Isaac Sim.*

---

## 11. Restoring a complete `pretrained_model/` from Google Drive (rclone)

If the local `--policy_path` is missing `config.json` and other pretrained artifacts, copy the entire Drive `pretrained_model/` folder contents into the local `pretrained_model/` directory:

```bash
rm -rf "/data/lehome_workspace/lehome-challenge/dummy_docker_policy/pretrained_model" \
  && mkdir -p "/data/lehome_workspace/lehome-challenge/dummy_docker_policy/pretrained_model"

rclone copy \
  "gdrive:LeHome/models/lehome_challenge/DINOv2_MAP8_Registers_merged_A100/DINOv2_MAP8_Registers_merged_A100__20260429_223013__11d24394bc70/checkpoints/last/pretrained_model" \
  "/data/lehome_workspace/lehome-challenge/dummy_docker_policy/pretrained_model" \
  --progress
```
