## Main Diagnosis
For **DINOv2_MAP_Registers**, your bottleneck is not simply “use more GPU.” The model is close to **VRAM saturation** because MAP with `map_num_queries: 8` expands each camera from `384` dims to `8 * 384 = 3072` dims, across **3 cameras**. That creates a large U-Net conditioning vector, so raising `batch_size` is much riskier than it was for the original DINO CLS baseline.

Your current config:

```8:18:/home/nidhinninan/Documents/Research/Hackathon/LeHome-trial/lehome_workspace/configs/sweep_dino_map_registers.yaml
policy:
  type: dino_diffusion
  vision_backbone: facebook/dinov2-with-registers-small
  spatial_pooling: map
  map_num_queries: 8
  use_registers: true
  num_register_tokens: 4
  device: cuda
  push_to_hub: false
  use_amp: false
```

## Highest ROI Changes

| Priority | Change | Recommendation | OOM / Transfer Risk |
|---|---|---|---|
| 1 | `policy.use_amp` | Set `true` for MAP+registers first, **without changing batch size**. | Low-to-medium risk. Should reduce VRAM and memory traffic, but verify loss stability. |
| 2 | CPU thread caps | Export `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, maybe `OPENBLAS_NUM_THREADS=1`. | Low risk. Helps avoid 12 DataLoader workers each oversubscribing CPU threads. |
| 3 | `num_workers` | Do a short sweep: `8`, `12`, `16`. I would not jump straight to 21. | Medium risk. Too high can increase RAM/shm pressure and context switching. |
| 4 | video backend | Prefer working `torchcodec` over `pyav` once FFmpeg is fixed. | Low risk if verified. `pyav` can be 10-50% end-to-end slower when dataloader-bound. |
| 5 | `map_num_queries` | Test `4` vs current `8` for speed. | Low OOM risk, likely faster. But it changes model capacity. |
| 6 | `batch_size` | Do **not** increase first. Try only after AMP proves stable. Test `batch_size=10` before `12`. | High risk for MAP. Current screenshots suggest MAP is already near VRAM limit. |
| 7 | augmentations | Keep off for speed benchmarking. Add later for generalization, not for wall-clock speed. | High CPU-transfer risk. Your notes already saw augmentation/dataloader choking. |

## My Concrete Recommended MAP+Registers Profile

For the next VM test, I would run:

- `policy.use_amp=true`
- `batch_size=8`
- `map_num_queries=8` initially
- `num_workers=12`
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `dataset.image_transforms.enable=false`
- `torchcodec` backend if verified working

Then compare wall clock and W&B `updt_s` / `data_s`.

If stable, second test:

- keep AMP on
- keep `batch_size=8`
- test `num_workers=16`

If MAP is still too slow or VRAM-bound, third test:

- keep AMP on
- set `map_num_queries=4`
- keep `batch_size=8`

Only after those should you try `batch_size=10`. I would avoid `batch_size=12` for MAP unless AMP + lower workers show `data_s` stays low and VRAM has several GB free.

## What Not To Do

Do **not** enable augmentations while trying to maximize hardware utilization. Your own learning notes already caught the important failure mode: AMP + bigger batch can make compute faster but still double wall time if the CPU dataloader chokes on more images.

Do **not** raise workers to the full CPU count blindly. With 26.4 GB RAM, 3 camera streams, video decode, and shared-memory queues, `16` may help; `21` may just increase RAM pressure and scheduling overhead.

Do **not** unfreeze DINOv2 for speed. It will likely increase VRAM, backward compute, and optimizer state. Your frozen-backbone artifacts correctly identify freezing as both a stability and hardware-efficiency win.

## Best Speed/Quality Tradeoff

If the goal is fastest useful MAP+registers training, my preferred order is:

1. **AMP on, batch size unchanged.**
2. **Thread caps + workers sweep.**
3. **Use `torchcodec` if possible.**
4. **Try `map_num_queries=4` if MAP remains too slow.**
5. **Only then test batch size increase.**

The most dangerous lever is `batch_size`; the most underrated lever is `map_num_queries`, because it directly shrinks the MAP output and U-Net conditioning size.

## Additional Context
Here is a concise, evidence-backed read of your setup and what to tune.

## What your repo actually does

**Both sweep jobs use the same `num_workers`**, not different defaults. `run_sweep_run` sets `WORKERS=12` when shared memory remount succeeded, and passes it to every run:

```144:193:/home/nidhinninan/Documents/Research/Hackathon/LeHome-trial/lehome_workspace/run_2run_dino_sweep.sh
    local WORKERS=12
    if [ "$SHM_REMOUNTED" = false ]; then
        WORKERS=0
        echo "   (Falling back to --num_workers=0 due to shm limits)"
    fi
    ...
    "$VENV_PYTHON" "$WORKSPACE_DIR/lerobot_train_with_plugins.py" \
        --config_path="$CONFIG" \
        --steps="$STEPS" \
        --eval_freq="$EVAL_FREQ" \
        --dataset.image_transforms.enable=false \
        --num_workers="$WORKERS" \
        2>&1 | tee "$LOG_FILE"
```

So if you saw very different W&B “process threads” or I/O patterns between **DINOv2_Baseline** and **DINOv2_MAP_Registers**, that is unlikely to be “12 vs default workers” from this script; it is more plausibly **memory pressure, run order, and OS caching** (MAP uses almost all VRAM and a much wider activation path).

Your two YAMLs are aligned except for the DINO path (backbone, `spatial_pooling`, registers, `map_num_queries`) and `output_dir` / `job_name`; both use `batch_size: 8`, `policy.use_amp: false`, and transforms disabled in the CLI.

## Grounded LeRobot behavior (DeepWiki on `huggingface/lerobot`)

From the LeRobot wiki (queried via DeepWiki):

- **`num_workers`**: CLI `--num_workers` or YAML `num_workers`; **default in LeRobot is 4** (your shell script overrides to 12).
- **`batch_size`**: `--batch_size` / top-level YAML `batch_size` (you use 8).
- **`pin_memory`**: Not a separate CLI flag; it follows **`device`** (CUDA → effectively pinned paths in the training pipeline).
- **`prefetch_factor`**: Default **2** when `num_workers > 0`; **not exposed** as a user-facing YAML/CLI knob in the same way.
- **`persistent_workers`**: **Not clearly exposed** in the documented train config.
- **`policy.use_amp`**: **`--policy.use_amp`** / YAML `policy.use_amp`; default **false** — turning it on is a first-class LeRobot path for mixed precision.
- **cuDNN**: `cudnn_deterministic` toggles deterministic vs **`benchmark` mode** when false (typical throughput setup).

So: for “missing” dataloader knobs, the honest answer is **many are fixed inside LeRobot** unless you patch or upgrade to a version that exposes them.

## How your W&B panels should be read (with your code in mind)

1. **MAP+registers at ~16 GB VRAM vs baseline ~7.5 GB** matches a heavier graph: MAP head on patch tokens, register backbone, larger conditioning into the U-Net. **Longer wall time per step** is expected; you are often **memory-bandwidth / activation heavy**, which fits volatile **“GPU time in memory”** and spiky utilization.

2. **Neither run is near the ~280 W power cap** — the GPU is not power-limited; you are **not fully feeding the SMs** (data pipeline / step structure / memory stalls), consistent with community guidance that **DataLoader and CPU-side work often cap utilization** ([ML Journey on slow dataloaders](https://mljourney.com/how-to-debug-slow-pytorch-dataloaders/), [PyTorch issue on workers + memory / shm](https://github.com/pytorch/pytorch/issues/13246)).

3. **Your DINO path resizes 480×640 → 224×224 before the backbone** (good for speed vs full-resolution patches):

```136:158:/home/nidhinninan/Documents/Research/Hackathon/LeHome-trial/lehome_workspace/lerobot_policy_dino/src/lerobot_policy_dino/modeling_dino_diffusion.py
        # MOD-2 FIX: Resize to 224×224 to match DINOv2 pretraining resolution.
        ...
        if self.spatial_pooling == "baseline":
            with torch.no_grad():
                outputs = self.backbone(img)
```

So the main remaining levers are **batching / AMP / dataloader / thread oversubscription**, not “forgot to resize.”

## What to change (or not) for fastest wall-clock

### High impact, low risk to try first

| Parameter | Recommendation |
|-----------|----------------|
| **`policy.use_amp`** | Both YAMLs have `use_amp: false`. **Strong candidate to set `true`** for RTX 40xx: usually **higher throughput and lower VRAM**, which on the MAP run could **ease the ~98% VRAM plateau** and reduce memory traffic per byte of compute. Validate loss curves still look sane for your hackathon metric. |
| **`num_workers` (CLI)** | LeRobot default is 4; you use **12**. On **~21 vCPUs**, **16–18** can help **if** RAM and `/dev/shm` stay healthy. **Sweep 8 / 12 / 16** and watch **steps/sec** (not just GPU %). **Do not** assume `num_workers ≈ cores` without measuring—too many workers **hurts** on I/O- or RAM-bound setups ([Exa / multiple guides](https://mljourney.com/how-to-debug-slow-pytorch-dataloaders/)). Your script already remounts `/dev/shm` to 2G when possible; keep that, since many-worker DataLoaders **fail or slow badly** when shm is tiny ([PyTorch #13246](https://github.com/pytorch/pytorch/issues/13246)). |
| **CPU BLAS threads** | A common pitfall: **each worker** may spawn **OpenMP/MKL threads**, so **12 workers × many threads** can oversubscribe **21 cores** and **look** like “low GPU util.” Set something like **`OMP_NUM_THREADS=1`**, **`MKL_NUM_THREADS=1`** (or 2) in the VM environment for training, then retune `num_workers`. This is repeatedly recommended in performance writeups surfaced via Exa (e.g. [ML Journey](https://mljourney.com/how-to-debug-slow-pytorch-dataloaders/)). |

### `batch_size` — split strategy for the two configs

| Run | VRAM headroom (your traces) | Guidance |
|-----|-----------------------------|----------|
| **Baseline** | Large headroom (~45% of 16 GB used) | **Try increasing `batch_size`** (e.g. 12 → 16) **after** enabling AMP, until you are **just under OOM**. Larger batches often **raise GPU duty cycle** on diffusion-style training. |
| **MAP+registers** | **~98% VRAM** | **Do not raise `batch_size`** until AMP (or other savings) frees headroom. If you need larger effective batch for stability, check whether your **installed LeRobot** exposes **gradient accumulation** in `TrainPipelineConfig` / CLI; DeepWiki did not list it in the same snippet block as the dataloader fields—verify in your VM’s `lerobot` version before relying on it. |

### Things that are **fair to keep fixed** for fair comparison

- **`spatial_pooling`**, **`use_registers`**, **`vision_backbone`**, **`map_num_queries`** — changing these changes the **experiment**, not just throughput. Only shrink **`map_num_queries`** if you accept a **different model** for speed experiments.

### Low or uncertain impact from YAML alone

- **`prefetch_factor` / `persistent_workers`** — likely **not configurable** from your YAML in current LeRobot (per DeepWiki). Treat as **framework limitation** unless you vendor-patch `lerobot_train.py`.
- **`torch.compile` / `compile_model`** — DeepWiki describes it as **policy-specific** (examples given for other policies). **Do not assume** `dino_diffusion` supports the same flag without checking your policy config class on the VM.
- **`log_freq`** — lowering log/W&B frequency saves **little** compared to forward/backward; optional micro-optimization.

### Social / news tooling note

- **RivalSearch `social_search`** returned a relevant **Stack Overflow** hit on **low GPU utilization without obvious bottlenecks** ([stackoverflow.com/questions/79387312](https://stackoverflow.com/questions/79387312/low-gpu-utilization-on-pytorch-without-obvious-bottlenecks)) — aligned with “profile before chasing GPU %.”
- **`news_aggregation`** for “PyTorch training performance GPU 2025” mostly returned **unrelated** Guardian / NVIDIA RSS noise; not useful for this specific tuning.

## Hugging Face MCP

Not required for this answer: your backbones are standard **`facebook/dinov2-small`** / **`facebook/dinov2-with-registers-small`**; no HF call changes local training knobs beyond what LeRobot already loads from the Hub.

---

**Practical order on the VM:** (1) set **`OMP_NUM_THREADS=1`** (and MKL similarly), (2) enable **`policy.use_amp: true`**, (3) **sweep `num_workers`** in {8, 12, 16}, (4) **raise `batch_size` only on baseline** once stable, (5) **revisit MAP `batch_size`** only if AMP frees VRAM. Compare **wall time to fixed step count** and **steps/sec**, not raw GPU % alone.

If you want, a follow-up can be a **small patch** to `run_2run_dino_sweep.sh` to export thread env vars and optionally pass **`--num_workers` from an env var** so you do not have to edit the script for each sweep (that would trigger `lehome_change_log.md` / `vm_transfer_list.md` updates per your workspace rules).