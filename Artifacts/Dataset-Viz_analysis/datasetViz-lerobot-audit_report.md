# Implementation Plan Audit Report

A line-by-line audit of the current [implementation_plan.md](file:///home/nidhinninan/.gemini/antigravity/brain/c47ddc5d-4313-43fd-885d-a04637c50b5e/implementation_plan.md) against all evidence gathered in this conversation.

---

## 🔴 Bug 1: `--root` Path Semantics Differ Between Tools

**Severity: HIGH — Will cause "dataset not found" errors at runtime.**

The plan treats `--root` identically for `lerobot-dataset-viz` and `lerobot-edit-dataset`. They actually have **different path semantics**:

| Tool | `--root` means | `--repo-id` means | Resulting lookup path |
|---|---|---|---|
| `lerobot-dataset-viz` | **Parent directory** | Subdirectory name | `{root}/{repo-id}/` |
| `lerobot-edit-dataset` | **Direct dataset path** | Label/identifier | `{root}/` (as-is) |
| LeHome `scripts.dataset inspect` | N/A | N/A | `--dataset_root` points directly |

**Evidence** — From the [official HF docs](https://huggingface.co/docs/lerobot/en/using_dataset_tools):
> From a local folder: Add the `--root` option and set `--mode local`. For example, to search in `./my_local_data_dir/lerobot/pusht`

This means `--root=./my_local_data_dir` + `--repo-id=lerobot/pusht` → looks up at `./my_local_data_dir/lerobot/pusht/`.

**Evidence** — From the [LeHome datasets.md](file:///home/nidhinninan/Documents/Research/Hackathon/LeHome-trial/lehome_workspace/lehome-challenge/docs/datasets.md) (line 314-319):
```bash
lerobot-edit-dataset \
    --repo_id record_top_long_release_10/001 \
    --root Datasets/record/record_top_long_release_10/001 \
```
Here `--root` points **directly** at the dataset directory.

### Fix Required

For `lerobot-dataset-viz` with the VM dataset at `/data/lehome_workspace/lehome-challenge/Datasets/example/top_long_merged`:
```bash
# CORRECT
lerobot-dataset-viz \
    --repo-id top_long_merged \
    --root /data/lehome_workspace/lehome-challenge/Datasets/example \
    --mode local \
    --episode-index 0

# WRONG (current plan implies this)
lerobot-dataset-viz \
    --root /data/lehome_workspace/lehome-challenge/Datasets/example/top_long_merged \
    ...
```

For `lerobot-edit-dataset`:
```bash
# CORRECT
lerobot-edit-dataset \
    --repo_id top_long_merged \
    --root /data/lehome_workspace/lehome-challenge/Datasets/example/top_long_merged \
    --operation.type info \
    --operation.show_features true
```

---

## 🔴 Bug 2: Wrong Inspect Tool for VM Context

**Severity: HIGH — Plan will miss LeHome-specific metadata.**

The plan proposes using `lerobot-edit-dataset --operation.type info` for the `inspect` mode. But the LeHome challenge provides its **own** inspection tool that surfaces LeHome-specific metadata (like `garment_info.json`):

```bash
# LeHome-specific (on VM only)
python -m scripts.dataset inspect \
    --dataset_root Datasets/example/top_long_merged \
    --show_frames 5 \
    --show_stats
```

### Fix Required

The `inspect` mode should:
- **On VM**: Use `python -m scripts.dataset inspect` (LeHome-specific, richer output)
- **Off VM / fallback**: Use `lerobot-edit-dataset --operation.type info` (generic LeRobot)

---

## 🟡 Bug 3: `lerobot==0.4.3` May Not Support `--mode local`

**Severity: MEDIUM — Could crash at CLI parsing.**

The `--mode local` flag and `--display-compressed-images` flag are confirmed in the **0.4.2+** codebase (from GitHub issue #2898, which reports running 0.4.2 with `--mode distant`). However, the LeHome challenge pins `lerobot==0.4.3`. The flag likely exists, but we haven't verified it for this exact version.

### Fix Required

Add a version check or try/except wrapper in the script:
```bash
# Before running viz mode, verify the flag exists
lerobot-dataset-viz --help 2>&1 | grep -q "\-\-mode" || {
    echo "ERROR: --mode flag not available in this lerobot version"
    exit 1
}
```

---

## 🟡 Bug 4: Conda Not on PATH

**Severity: MEDIUM — The setup script will fail silently.**

`conda` is installed at `/home/nidhinninan/anaconda3/bin/conda` but is **not on the shell PATH**. The plan says to run `conda create -n lehome python=3.11 -y` but the user's terminal can't resolve `conda`.

### Fix Required

The setup script must either:
1. Source conda's init script first: `source ~/anaconda3/etc/profile.d/conda.sh`
2. Or use the absolute path: `~/anaconda3/bin/conda create -n lehome python=3.11 -y`

After creation, activation also requires the init script:
```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate lehome
```

---

## 🟡 Bug 5: Python Version — Local vs. VM Mismatch Risk

**Severity: MEDIUM — Could install incompatible lerobot version.**

| Environment | Python | lerobot |
|---|---|---|
| LeHome VM (pyproject.toml) | `>=3.11,<3.12` | `==0.4.3` |
| Latest LeRobot (pyproject.toml) | `>=3.12` | `0.5.1` |
| Plan's local env | `3.11` | unspecified |

If we `pip install lerobot` in a Python 3.11 env, pip will resolve to the **latest compatible version**. If lerobot `0.5.x` requires Python 3.12+, pip may either:
- Install 0.4.3 (correct, but by accident)
- Fail with a resolver error

### Fix Required

Pin the version explicitly in the setup script:
```bash
pip install lerobot==0.4.3
```

---

## 🟡 Bug 6: ffmpeg Installation Channel Not Specified

**Severity: MEDIUM — May install ffmpeg without libsvtav1 codec.**

The plan says "install ffmpeg" but doesn't specify the conda-forge channel. System ffmpeg packages often lack `libsvtav1`, which LeRobot's video encoding requires.

### Fix Required

```bash
conda install ffmpeg -c conda-forge
```

---

## 🟢 Bug 7: Missing Dataset Download Step

**Severity: LOW — Plan mentions it but doesn't give the command.**

The plan says "you will need to copy or download the relevant dataset chunks" but doesn't provide the exact command.

### Fix Required

Add to the script or docs:
```bash
# Requires huggingface-hub[cli]
pip install "huggingface-hub[cli,hf-transfer]"
hf download lehome/dataset_challenge_merged \
    --repo-type dataset \
    --local-dir Datasets/example
```

> [!NOTE]
> This downloads ~18 GB of data. For quick testing, consider downloading a single category's video chunks.

---

## 🟢 Bug 8: `replay` Mode Guard Missing

**Severity: LOW — Script will crash with confusing error if IsaacSim absent.**

The `replay` mode runs `python -m scripts.dataset_sim replay`, which imports IsaacSim. On the local machine (no IsaacSim), this will fail with a cryptic `ModuleNotFoundError`.

### Fix Required

Add a guard at the top of the `replay` mode branch:
```bash
python -c "import isaacsim" 2>/dev/null || {
    echo "ERROR: 'replay' mode requires IsaacSim (only available on the VM)"
    exit 1
}
```

---

## Summary of Required Changes

| # | Severity | Issue | Change |
|---|---|---|---|
| 1 | 🔴 HIGH | `--root` path semantics differ between tools | Split path construction logic per tool |
| 2 | 🔴 HIGH | Wrong inspect tool for VM | Use `scripts.dataset inspect` on VM, `lerobot-edit-dataset info` as fallback |
| 3 | 🟡 MED | `--mode local` flag may not exist in 0.4.3 | Add version/flag check |
| 4 | 🟡 MED | Conda not on PATH | Source `conda.sh` before any conda commands |
| 5 | 🟡 MED | Python version mismatch risk | Pin `lerobot==0.4.3` explicitly |
| 6 | 🟡 MED | ffmpeg channel not specified | Use `conda install ffmpeg -c conda-forge` |
| 7 | 🟢 LOW | Missing download command | Add `hf download` command to docs |
| 8 | 🟢 LOW | No IsaacSim guard on `replay` | Check for `isaacsim` import before running |
