# Conversation summary — DINO sweep, video backends, FFmpeg, GPU cleanup

**Date**: 2026-04-26  
**Scope**: `run_2run_dino_sweep.sh`, LeRobot training on VM, `torchcodec` / `pyav`, FFmpeg install, torchvision deprecation warnings, GPU memory cleanup, package manager choice (`apt` vs `uv pip`).

---

## 1. Original failure: DataLoader + `torchcodec` + FFmpeg

**Symptom**: Training crashed when the first batch was pulled from the dataloader. Stack trace pointed to:

- `lerobot/datasets/lerobot_dataset.py` → `_query_videos` → `decode_video_frames`
- `lerobot/datasets/video_utils.py` → `decode_video_frames_torchcodec`
- `torchcodec` failed to load native `libtorchcodec`, with root cause:

**Missing FFmpeg shared libraries** (examples from the error):

- FFmpeg 7: `libavutil.so.59` — not found  
- FFmpeg 6: `libavutil.so.58` — not found  
- FFmpeg 5: `libavutil.so.57` — not found  
- FFmpeg 4: `libavdevice.so.58` — not found  

So the VM had **no compatible FFmpeg dev/runtime libs** on the dynamic linker path that `torchcodec` expects.

**PyTorch version noted in traceback**: `2.7.0+cu126` (relevant if `torchcodec` later fails for version mismatch instead of missing libs).

---

## 2. Repo-side mitigations (local workspace)

Two changes were made under `lehome_workspace/` (mirror in VM transfer list / change log as applicable):

1. **`run_2run_dino_sweep.sh`**  
   - Added `--dataset.video_backend=pyav` so LeRobot uses the **torchvision-backed** decode path instead of `torchcodec`, avoiding the missing `libtorchcodec`/FFmpeg `.so` issue **without** requiring a working `torchcodec` stack first.

2. **`setup_lehome.sh`**  
   - Added **`ffmpeg`** to the `apt install` list so **future** VM installs get system FFmpeg libraries and `torchcodec` is more likely to work out of the box.

---

## 3. GPU “dead memory” from previous runs

**Problem**: After crashes or `Ctrl+C`, worker processes can keep `/dev/nvidia*` open; VRAM stays allocated.

**Recommended VM commands** (also captured in `To_RUN_onVM_cmds.txt`):

```bash
nvidia-smi
sudo fuser -k /dev/nvidia*
pkill -9 -f lerobot_train_with_plugins || true
pkill -9 -f lerobot_train              || true
sleep 2 && nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

`fuser -k /dev/nvidia*` is aggressive but effective for zombie dataloader workers.

---

## 4. Torchvision `UserWarning` about deprecated video I/O

**Message**:  
`torchvision/io/_video_deprecation_warning.py` — video decode/encode in torchvision deprecated from 0.22, removed in 0.24; migrate to TorchCodec.

**Where it comes from**:

- Emitted by **torchvision** when code uses **`torchvision.io` video APIs** (e.g. `VideoReader`, `torchvision.io.video`, paths that call `_raise_video_deprecation_warning()`).

**Why it appeared after the workaround**:

- LeRobot’s `decode_video_frames` **dispatches** on `video_backend`:
  - **`torchcodec`** → `decode_video_frames_torchcodec` (does **not** go through torchvision’s deprecated video stack for that path).
  - **`pyav`** / **`video_reader`** → `decode_video_frames_torchvision` → `torchvision.set_video_backend` / `torchvision.io.VideoReader` → **triggers the deprecation warning**.

So the warning is **expected** when using `--dataset.video_backend=pyav` as a fallback.

**How to remove the warning**: use a working **`torchcodec`** path again (after FFmpeg + compatible torchcodec), or filter warnings if you must stay on `pyav`.

---

## 5. Performance: `pyav` vs `torchcodec`

**Not** “fraction of a percent” in general.

- **Decode-only / dataloader-bound** runs: `pyav` (torchvision path) is often **~1.2×–3× slower** on the **video decode** portion; end-to-end can be **~10–50% slower** if the GPU is waiting on data.
- **GPU-saturated** runs: overhead may shrink to **a few percent** overall.

**Practical check**: `nvidia-smi` during training — stable high GPU util → smaller impact; low/bursty util → data pipeline (including decode) is likely significant.

---

## 6. Installing FFmpeg: `apt` vs `uv pip` vs `pip`

- **FFmpeg (shared libraries + `ffmpeg` binary)**: install with **`sudo apt install -y ffmpeg`** — **not** `pip` / `uv pip`.
- **Python packages** (e.g. `torchcodec`, extras): inside the LeRobot venv, **`uv pip`** is preferred if the project uses `uv` (e.g. `uv sync`); plain `pip` works but stay consistent.

Order of operations: **fix system FFmpeg first**; then validate `torchcodec` import; only then pin `torchcodec` versions if PyTorch compatibility errors remain.

---

## 7. Should you “deactivate” the `pyav` CLI flag?

**Yes, once `torchcodec` is verified working.**

Steps on VM:

1. Install FFmpeg (`apt`).
2. Verify libraries: `ldconfig -p | grep -E "libavutil|libavdevice|libavcodec|libavformat"`.
3. Smoke test:

   ```bash
   cd /root/data/lehome_workspace/lehome-challenge
   source .venv/bin/activate
   python -c "import torchcodec; from torchcodec.decoders import VideoDecoder; print('torchcodec OK')"
   ```

4. If OK, change training args to **`--dataset.video_backend=torchcodec`** (or remove override if default is torchcodec).

If `torchcodec` still fails after FFmpeg is present, treat it as **torchcodec ↔ PyTorch** or other runtime deps — capture the new traceback.

---

## 8. User readout: `ldconfig` after FFmpeg install

Example output showed **FFmpeg 4.x-era** sonames on x86_64, e.g.:

- `libavutil.so.56`
- `libavformat.so.58`
- `libavcodec.so.58`
- `libavdevice.so.58`

That aligns with the **FFmpeg v4** branch of `torchcodec`’s earlier error message (`libavdevice.so.58` was the missing piece for v4). So the environment moved from “no usable libs” to “FFmpeg 4 stack visible to the dynamic linker,” which is the right direction for **`torchcodec`** to load **if** the wheel matches those ABIs.

---

## 9. Files touched in this thread (reference)

| Area | File / location |
|------|------------------|
| Sweep runner | `lehome_workspace/run_2run_dino_sweep.sh` (`--dataset.video_backend=pyav` until torchcodec confirmed) |
| VM bootstrap | `lehome_workspace/setup_lehome.sh` (`ffmpeg` in apt list) |
| VM command cheatsheet | `To_RUN_onVM_cmds.txt` (GPU cleanup block) |
| Tracking (if updated in same session) | `lehome_workspace/lehome_change_log.md`, `lehome_workspace/vm_transfer_list.md` |
| This artifact | `Artifacts/Vision-Backbone/sweep/conversation_summary_2026-04-26_dino_sweep_video_backends.md` |

---

## 10. One-line takeaway

**Train unblocked** by `pyav` backend when system FFmpeg / `torchcodec` was broken; **long-term** install FFmpeg on the VM, confirm `torchcodec` imports, then switch back to **`torchcodec`** for speed and to drop the torchvision deprecation noise; use **`apt`** for FFmpeg and **`uv pip`** for Python deps inside the venv.
