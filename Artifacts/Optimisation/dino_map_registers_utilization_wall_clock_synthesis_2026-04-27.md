# DINOv2 MAP+Registers: Utilization, Wall-Clock, and Config Profile — Synthesis Artifact

**Date**: 2026-04-27  
**Scope**: Two-run DINO sweep (`DINOv2_MAP_Registers` vs `DINOv2_Baseline`), Weights & Biases system metrics, Vast.ai VM hardware, LeRobot v0.4.3 BYOP (`dino_diffusion`), grounded research (DeepWiki, Exa, RivalSearch), local project artifacts and learning notes, and **implemented** MAP training profile in repo configs/scripts.

This document is written so a future reader can **reconstruct decisions** from reasoning, not only from final parameter values.

---

## 1. Problem statement

The user ran `lehome_workspace/run_2run_dino_sweep.sh` on a VM comparing:

- **`DINOv2_Baseline`** — `facebook/dinov2-small`, `spatial_pooling: baseline`, `use_registers: false`, sweep launcher `num_workers=12` when `/dev/shm` remount succeeds.
- **`DINOv2_MAP_Registers`** — `facebook/dinov2-with-registers-small`, `spatial_pooling: map`, `use_registers: true`, `num_register_tokens: 4`, `map_num_queries: 8`, same launcher worker count in code (see §5.1 for clarification vs perceived “default workers”).

**Goal**: Maximize **hardware utilization** and **reduce wall-clock time** for the **MAP+registers** variant, without causing **OOM** or **CPU↔GPU transfer stalls**, using grounded guidance (not guesswork).

---

## 2. Observed runtime behavior (W&B + hardware context)

The user supplied **four** dashboard / spec images (also saved under the Cursor project `assets/` paths at capture time). Below is the **logical interpretation** tied to training architecture — these are symptoms, not independent “facts” until confirmed with `data_s` / `updt_s` from LeRobot logs.

### 2.1 GPU-side (from screenshot descriptions)

| Signal | Baseline (approx.) | MAP+Registers (approx.) | Interpretation |
|--------|--------------------|---------------------------|----------------|
| Wall time to completion | ~31 min | ~44 min | MAP path is **heavier per step** (larger conditioning, more activation / memory traffic). |
| VRAM allocated | ~7.5 GB (~45% of 16 GB) | ~16 GB (~98% of 16 GB) | MAP **materially increases** resident GPU memory vs CLS/mean baseline. |
| GPU utilization | Spiky, often 40–90% | Spiky, peaks to 100%, dips to 30–40% | Classic **pipeline starvation** or **memory-bound phases**: GPU not continuously fed or sometimes waiting on memory subsystem. |
| “GPU time in memory” | Variable | Very volatile, frequent 100% spikes | Consistent with **bandwidth / memory access** pressure when MAP expands token readout before U-Net conditioning. |
| Power draw | ~110–125 W (~40–42% of ~280 W cap) | ~140–165 W (~50–55% of cap) | Neither run is **power-limited**; headroom exists thermally and on the power limiter — the limiter is not “GPU can’t draw more watts,” it’s **work not fully saturating SMs** or **stalls** (data / sync / memory). |
| SM / memory clocks | Flat high | Flat high | **Not** thermal-throttle limited at reported fan speeds; bottleneck is elsewhere (pipeline or memory stalls). |

**Hardware cited by user / images**: NVIDIA **RTX 4070 Ti** class (16 GB VRAM), **~21 vCPU** (one image listed **22** cores — treat as “high-teens/low twenties” for worker tuning), **~26–32 GB** system RAM depending on listing, **PCIe 3.0 x16** (~9.4 GB/s cited), NVMe ~612 MB/s cited. **Implication**: CPU decode + dataloader + host memory bandwidth can cap GPU if video decode or transforms are heavy; **PCIe 3** caps host-to-device throughput for large batches / many images per step.

### 2.2 Host-side (from screenshot descriptions)

| Signal | Baseline | MAP+Registers | Interpretation |
|--------|----------|---------------|----------------|
| Process threads (W&B) | Spike then ~52 | ~20 flat | **Not the same as `num_workers`**. Often reflects **OpenMP / MKL threads per process**, thread pools, and framework threading. Large disparity can indicate **different effective CPU parallelism** or metric semantics — do not equate to “12 vs default workers” without verifying `ps` / launcher. |
| Disk read curve | Steady climb then end spike | Flat near zero then large end spike | Suggests **different caching / working-set / flush** behavior between runs; not definitive of “fewer workers” alone. |
| Network receive | Large spike at start (baseline) | Smaller start, jump mid-run | Likely **checkpoint / W&B / artifact** or environment differences — treat as secondary unless correlated with dataloader source (HF vs local). |

**Reasoning chain**: If VRAM is **~98%** on MAP, **increasing `batch_size`** is the **highest-risk** lever for OOM. If GPU power is **below cap** and utilization is **spiky**, **data pipeline and thread oversubscription** are prime suspects **alongside** MAP’s larger per-step memory traffic.

---

## 3. Architecture grounding (why MAP is expensive)

### 3.1 BYOP implementation (`lerobot_policy_dino`)

Policy type: **`dino_diffusion`** (`DinoDiffusionConfig` / `DinoDiffusionPolicy`).

**Resize before backbone (critical for speed and OOM)** — LeHome cameras are **480×640**; the model **must** downsample before `Dinov2Model` or patch count explodes (~1565 patches vs 256 at 224²):

```137:168:lehome_workspace/lerobot_policy_dino/src/lerobot_policy_dino/modeling_dino_diffusion.py
    def _encode_images(self, img: Tensor) -> Tensor:
        ...
        if img.shape[-2:] != (_DINO_INPUT_SIZE, _DINO_INPUT_SIZE):
            img = F.interpolate(
                img,
                size=(_DINO_INPUT_SIZE, _DINO_INPUT_SIZE),
                mode="bilinear",
                antialias=True,
            )
        ...
        # MAP path: backbone frozen (no_grad); MAP head is trainable.
        with torch.no_grad():
            outputs = self.backbone(img)
            hidden = outputs.last_hidden_state
        start_idx = 1 + (self.num_register_tokens if self.use_registers else 0)
        patch_tokens = hidden[:, start_idx:, :]
        ...
        return self.spatial_head(patch_tokens)
```

**MAP head output size** (`MAHead` flattens `K` queries × `D`):

```96:101:lehome_workspace/lerobot_policy_dino/src/lerobot_policy_dino/modeling_dino_diffusion.py
        if self.spatial_pooling == "map":
            self.spatial_head = MAPHead(
                hidden_dim=self.feature_dim,
                num_queries=self.map_num_queries,
            )
            self.head_output_dim = self.map_num_queries * self.feature_dim
```

For **DINOv2-Small**, `feature_dim = 384`. With **`map_num_queries: 8`**, **`head_output_dim = 3072` per image`**. LeHome sweep uses **three** RGB streams (`top_rgb`, `left_rgb`, `right_rgb`), so vision contributes **3072 × 3 = 9216** dims per timestep into global conditioning (plus state), vs **384 × 3 = 1152** for a single 384-d embedding per camera in the baseline CLS path.

**Justification**: Wall-clock and VRAM rise are **expected** from conditioning volume and attention over patch tokens, not merely “workers were wrong.”

### 3.2 Literature / design rationale (project artifacts)

- **`Artifacts/Vision-Backbone/DINOv2-MAP_implementation_research.md`**: MAP chosen over naive SpatialSoftmax-on-ViT (Voltron-style evidence; register-token artifact risks per Darcet et al.; flatten `K×D` to preserve distinct query readouts for U-Net).
- **`Artifacts/Vision-Backbone/dinov2_spatial_pooling_audit_thread_2026-04-25.md`**: Same trade-space — MAP vs DPT+hardened SpatialSoftmax vs full patch sequence; **registers backbone** recommended whenever spatial readout touches patch tokens.
- **`Artifacts/Vision-Backbone/sweep/10k_sweep_analysis.md`**: At 10k steps, frozen **DINOv2** ~**0.30–0.32 s/step** vs **ResNet18** ~**0.50–0.52 s/step** — frozen backbone as **compute and stability** win vs full end-to-end vision training.

These do **not** contradict MAP being slower than **DINO CLS baseline**; they explain why MAP is a **deliberate** cost for representation quality.

---

## 4. External / tooling research (how conclusions were grounded)

### 4.1 DeepWiki — `huggingface/lerobot` (TrainPipelineConfig)

Queried via DeepWiki `ask_question` (repoName `huggingface/lerobot`). **Grounded points**:

- **`num_workers`**: CLI `--num_workers` or YAML; default **4** in framework (sweep scripts override to **12** when SHM allows).
- **`batch_size`**: CLI / YAML top-level.
- **`pin_memory`**: Not a separate CLI flag in the summarized API; tied to **CUDA device** behavior in training pipeline.
- **`prefetch_factor`**: Default **2** when `num_workers > 0`; **not** described as user-exposed YAML/CLI in the same way.
- **`persistent_workers`**: Not clearly exposed.
- **`policy.use_amp`**: **`--policy.use_amp`** / YAML `policy.use_amp`; default **false** — supported path for mixed precision.
- **cuDNN**: `cudnn_deterministic` vs benchmark behavior summarized.

**Justification for recommendations**: If a knob is **not** in LeRobot’s surface config, “tune `prefetch_factor` in YAML” may be **impossible without forking** `lerobot_train` — avoiding false precision.

### 4.2 Exa (`web_search_exa`)

Themes retrieved (with URLs in search results used in-session):

- **`num_workers`**: Workers parallelize decode/prep vs main process; **too many** workers → context switch / I/O contention; **too few** → GPU waits. Rule-of-thumb varies by author (**≈ cores**, or **benchmark 2/4/8/16**); **always measure** on target hardware.
- **Shared memory / Docker**: Large `num_workers` + image batches can exhaust **`/dev/shm`** → bus errors / instability; mitigations include remounting shm (see project sweep scripts).
- **`pin_memory` + non_blocking**: Standard advice for H2D overlap when GPU-bound (LeRobot pins via device path per DeepWiki summary).

### 4.3 RivalSearchMCP

- **`social_search`**: e.g. Stack Overflow discussion on **low GPU utilization without obvious bottlenecks** — aligns with “profile; don’t assume GPU % is only metric.”
- **`news_aggregation`**: Many hits were **off-topic** (unrelated news); **not** used as evidence for PyTorch tuning.

### 4.4 Hugging Face / upstream config verification

Fetched **`TrainPipelineConfig`** and **`DatasetConfig`** from **LeRobot v0.4.3** tag on GitHub:

- `src/lerobot/configs/train.py`: top-level **`num_workers: int = 4`**, **`batch_size`**, etc.
- `src/lerobot/configs/default.py` — **`DatasetConfig`** includes **`video_backend: str`** (default from `get_safe_default_codec()`).

**Justification**: `dataset.video_backend: torchcodec` in YAML is a **valid** LeRobot 0.4.3 field name; **`torchcodec`** requires compatible **FFmpeg** system libraries and a working **`torchcodec`** install — otherwise training fails at first batch decode (see §6).

---

## 5. Local project experience (artifacts + learning files)

### 5.1 Sweep launcher behavior (`run_2run_dino_sweep.sh`)

Both named runs use the **same** `WORKERS=12` when shared-memory remount succeeds; there is **no** branch that sets “default” workers for MAP only:

```144:193:lehome_workspace/run_2run_dino_sweep.sh
    local WORKERS=12
    if [ "$SHM_REMOUNTED" = false ]; then
        WORKERS=0
        ...
    fi
    ...
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

    "$VENV_PYTHON" "$WORKSPACE_DIR/lerobot_train_with_plugins.py" \
        --config_path="$CONFIG" \
        ...
        --dataset.image_transforms.enable=false \
        --num_workers="$WORKERS" \
```

**Logical point**: If W&B showed very different “threads” or I/O between runs, the **first** explanation is **not** “MAP used default workers” from this script — investigate **OMP/MKL**, **memory pressure**, **caching**, and **metric definitions**.

### 5.2 `Artifacts/Vision-Backbone/sweep/dino_sweep_video_backends_conversation_summary_2026-04-26.md`

- **Failure mode**: `torchcodec` + missing FFmpeg `.so` → crash in `decode_video_frames` / dataloader.
- **Mitigation used historically**: `--dataset.video_backend=pyav` to unblock.
- **Performance**: `pyav` / torchvision path often **~1.2×–3×** slower on decode-heavy workloads; end-to-end **~10–50%** if GPU starved waiting on data.
- **Recommendation**: Install **FFmpeg via `apt`**, verify **`torchcodec` import**, then prefer **`torchcodec`** for speed.

**Justification for `torchcodec` in MAP profile**: Correct when verified; **revert to `pyav`** only if libraries or wheels mismatch — **throughput vs reliability** trade.

### 5.3 `Artifacts/Vision-Backbone/sweep_infra_crash_log.md`

- **`/dev/shm` too small** + image DataLoader → **SIGBUS / bus error**; remount `size=2G` + fallback `num_workers=0`.
- **Augmentation bottleneck**: `data_s` ≫ `updt_s` when transforms on — **CPU** bound.

### 5.4 `Learning.md` (user notes)

- **AMP vs batch / dataloader choke**: Example where **AMP + larger batch** made **steps slower** because **CPU could not decode/prepare** more images per step in time — **wall-clock regressed** despite “faster GPU math.”
- **`num_workers` mental model**: Parallel data processes; **do not** assume `num_workers = core count` without measuring; **vCPU-limited** VMs can be **decode-bound** regardless of AMP.
- **OOM probe pattern**: Short `lerobot-train` run with raised `--batch_size` to find ceiling (user’s workflow).

### 5.5 `LEARNING-2.md`

- **BYOP**: `get_optim_params` only **trainable** params (frozen backbone); **224 resize** to avoid CLIP pos-embed crash and DINO slowdown/OOM at native res.
- **Sweep**: **Augmentations off** for fair timing vs ResNet baseline.

### 5.6 `Artifacts/Training_Update/training-DP_augmentation_resume_strategy_150k.md`

- Augmentation for **generalization** after BC fits; **loss bump** on resume can be **healthy** — separate objective from **raw wall-clock** of a sweep.

**Synthesis**: **Augmentation improves generalization** but typically **hurts** short-run **throughput** per step when CPU-bound — do **not** enable it when optimizing **10k micro-sweep wall-clock** unless that is explicitly the experiment.

---

## 6. Recommendations (full reasoning, ordered by risk / ROI)

### 6.1 `policy.use_amp: true` (implemented for MAP YAML)

- **Mechanism**: Mixed precision reduces **activation footprint** and can increase **effective throughput** on RTX 40xx Tensor Cores.
- **OOM risk**: **Generally lowers** VRAM for same `batch_size` — **favorable** when MAP is near **98%** VRAM.
- **Pitfall (from `Learning.md`)**: If paired with **higher `batch_size`** or heavier CPU decode, **wall-clock can still regress**. **Recommended pairing**: enable AMP **first** at **unchanged `batch_size`**, measure `updt_s` / `data_s`, then consider batch changes.

### 6.2 `num_workers` (12 in scripts; 12 in MAP YAML as documentation / direct `lerobot-train` parity)

- **Mechanism**: Overlap **decode + collation** with GPU forward/backward.
- **OOM / RAM risk**: Each worker holds buffers; **too many** workers on **~26 GB RAM** + **3 cameras** + **video** can increase **RSS** and **`/dev/shm`** pressure → **OOM or bus errors** (see crash log).
- **CPU lag risk**: **Too many** workers → **contention** on disk decode and CPU caches → **longer** `data_s`.
- **Logical sweep**: Benchmark **8 / 12 / 16** (user has **~21 vCPU**); **avoid jumping to 21** without evidence.

### 6.3 Thread caps — `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` (implemented in sweep scripts, default 1)

- **Mechanism**: Many scientific libs spawn **multi-threaded BLAS** inside **each** worker process. With **`num_workers=12`**, **12 × many threads** can **oversubscribe** cores → **context switching**, **false** “low GPU util” from **CPU starvation** of the training step.
- **OOM risk**: Low.
- **Transfer lag**: Indirectly **helps** by freeing CPU time to **feed pinned batches** faster.

Implementation uses **`${VAR:-1}`** so the VM can override without editing scripts.

### 6.4 `dataset.video_backend: torchcodec` (implemented in `sweep_dino_map_registers.yaml`)

- **Mechanism**: Faster decode path **when** native stack works (per project artifact synthesis).
- **Failure mode**: Missing **`libav*`** / wrong ABI / broken **`torchcodec` wheel** → **immediate dataloader crash** (documented in Vision-Backbone sweep summary).
- **Mitigation**: One-off CLI **`--dataset.video_backend=pyav`** until FFmpeg + `torchcodec` smoke test passes (`python -c "import torchcodec; ..."`).

**Baseline YAML** was intentionally **not** changed in the implementation pass — only MAP sweep config carries `torchcodec` so the **second** job in the same shell still uses baseline YAML defaults for `video_backend` unless overridden elsewhere.

### 6.5 `batch_size` — **do not raise first** for MAP

- **Reason**: MAP already pushes **global_cond** dimension; VRAM ~**98%** in W&B for MAP run.
- **OOM risk**: **High** if increased before AMP and pipeline proof.
- **If AMP frees headroom**: Step **short** OOM probes (e.g. **10** before **12**), watch **first batch** and **10–50 steps**.

### 6.6 `map_num_queries` — architecture / speed tradeoff

- **Mechanism**: `head_output_dim = map_num_queries × 384` per camera → directly scales **U-Net input** size and **activations**.
- **Justification**: Reducing **K** (e.g. **8 → 4**) is often the **safest** “free-ish” speed lever **if** the user accepts a **different experimental model** — not equivalent for paper-perfect MAP comparison.

### 6.7 Augmentations — `dataset.image_transforms.enable: false` for sweep timing (CLI still forces false)

- **Justification**: Augmentations shift objective toward **generalization** but add **CPU** work; project logs already show **transform-bound** `data_s`. Keep **off** for **apples-to-apples speed** and MAP micro-sweeps; use **`resume_with_augmentations.sh`** style flows when **generalization** is the goal (`Artifacts/Training_Update/training-DP_augmentation_resume_strategy_150k.md`).

### 6.8 Optimizer / “different optimization”

- **Caution**: Changing AdamW / LR / scheduler changes **learning dynamics**, not just **hardware utilization**. The **`Artifacts/Training_Update/training-DP_progress_analysis_*.md`** narrative shows **LR schedule length** vs **steps** matters for whether the model **keeps learning**; this is **orthogonal** to GPU feeding unless optimizer step dominates (unusual here).

**Default stance**: **Do not** swap optimizer for “speed” until **dataloader + AMP + video backend** are settled; otherwise confounds are likely.

---

## 7. Implemented repo state after this conversation (MAP profile)

**File**: `lehome_workspace/configs/sweep_dino_map_registers.yaml`

- `policy.use_amp: **true**`
- `dataset.video_backend: **torchcodec**` (with inline comments: FFmpeg + working torchcodec; CLI fallback to `pyav`)
- `dataset.image_transforms.enable: **false**`
- Top-level `num_workers: **12**` (matches LeRobot `TrainPipelineConfig` and sweep script when SHM OK)
- Unchanged: `batch_size: 8`, `map_num_queries: 8`, registers backbone and MAP flags

**Files**: `lehome_workspace/run_2run_dino_sweep.sh`, `lehome_workspace/run_2run_dino_sweep_no-gdrive.sh`

- Export **`OMP_NUM_THREADS`**, **`MKL_NUM_THREADS`**, **`OPENBLAS_NUM_THREADS`** (default **1**, overridable from environment) immediately before `lerobot_train_with_plugins.py`.

**Tracking** (per workspace policy): `lehome_workspace/lehome_change_log.md` entry **2026-04-27 22:57:13 UTC**; `lehome_workspace/vm_transfer_list.md` **Last Updated** and file rows updated.

---

## 8. Verification checklist (VM — not run in local dev workspace per project rules)

1. **`/dev/shm`**: Confirm remount logic in sweep script or manual `df -h /dev/shm` ≥ training needs.
2. **`torchcodec`**: `python -c "import torchcodec; from torchcodec.decoders import VideoDecoder; print('ok')"` in venv; `ldconfig -p | grep libav` as needed.
3. **First run**: If decode fails, append **`--dataset.video_backend=pyav`** once to confirm training health, then return to **`torchcodec`** after fixing system libs.
4. **Metrics**: Compare **`updt_s`**, **`data_s`**, **steps/sec**, **VRAM**, not only GPU utilization %.
5. **Stray VRAM**: If runs abort, use project-documented GPU cleanup (`nvidia-smi`, `fuser`, `pkill` patterns in `To_RUN_onVM_cmds.txt` / Vision-Backbone sweep summary).

---

## 9. Source index (conversation-derived)

| Source | Role |
|--------|------|
| W&B screenshots (user) | Empirical utilization / time / VRAM patterns |
| Vast.ai hardware screenshot | CPU/RAM/PCIe/disk context |
| DeepWiki `huggingface/lerobot` | TrainPipelineConfig / AMP / dataloader defaults |
| Exa `web_search_exa` | DataLoader worker and shm community guidance |
| RivalSearchMCP `social_search` / `news_aggregation` | Thin topical signal; news mostly noise |
| GitHub raw `lerobot` v0.4.3 `train.py`, `default.py` | Confirmed `num_workers`, `dataset.video_backend` field existence |
| `lehome_workspace/lerobot_policy_dino/.../modeling_dino_diffusion.py` | MAP math, resize, register slicing |
| `Artifacts/Vision-Backbone/*`, `Artifacts/Training_Update/*`, `Learning.md`, `LEARNING-2.md`, `LEARNING-3.md` | Project-specific constraints and prior measurements |

---

## 10. One-line takeaway

**MAP+registers is supposed to be heavier than CLS baseline**; **wall-clock optimization** means **AMP + verified video decode + controlled workers + BLAS thread caps**, while treating **`batch_size` increases as high-risk** until VRAM headroom exists — **augmentations and naive worker max-out are common ways to accidentally slow training or OOM**, even when the GPU power meter shows unused watts.

## Addition Query response (WandB CPU Thread count)

When Weights & Biases (W&B) reports "21 process threads", it is counting the number of OS-level software threads spawned by your main Python training process.

Here is exactly how that translates to your Xeon E5-2696 v4 (22.0/88 CPU) hardware and active core usage:

1. Software Threads vs. Active Hardware Cores
21 process threads does not mean 21 CPU cores are actively computing at 100%. A thread is just a sequence of instructions the OS manages. Many of these 21 threads in a PyTorch/LeRobot process spend their time "sleeping"—waiting on GPU synchronization, waiting for disk I/O, or handling background tasks like W&B telemetry, rather than crunching math.

2. Why exactly 21-22 threads? (The Auto-Detection Behavior)
Your VM rental (22.0/88 CPU) means you have been allocated 22 vCPUs (hardware threads) from the host machine's total pool of 88.

By default, scientific computing libraries like PyTorch, NumPy, and OpenBLAS check the system for the number of available CPU cores and size their internal computational thread pools to match it. Because you have 22 vCPUs, the framework automatically spun up a thread pool of roughly 21-22 threads to handle CPU-bound matrix math.

3. The Danger: Thread Oversubscription
While having 21 threads in the main process is fine, it becomes a massive bottleneck when combined with PyTorch's DataLoader.

If your main process defaults to 21 math threads, and you set --num_workers=12 for your dataset loading, each of those 12 background workers might also spawn 21 threads. Suddenly, you have 250+ software threads fighting for your 22 physical vCPUs. The CPU spends all its time context-switching between threads rather than actually decoding your video frames, which starves the GPU of data.

How this was solved in your workspace
This exact scenario is why your run_2run_dino_sweep.sh script was updated to explicitly cap these thread pools right before triggering lerobot_train:

```bash
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
```
By forcing these libraries to use 1 thread per process, the CPU is freed up. Your 12 DataLoader workers (num_workers=12) can comfortably run on your 22 vCPUs (1 worker per vCPU, leaving plenty of room for the main process and OS overhead) without stepping on each other's toes.

In summary: W&B saw ~21 threads because PyTorch auto-detected your 22 vCPUs and created a matching thread pool. However, to get maximum performance out of those active cores, we actively restrict those threads to 1 so your parallel num_workers can do the heavy lifting of data loading efficiently!