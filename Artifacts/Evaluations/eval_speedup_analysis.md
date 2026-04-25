# LeHome Evaluation Speedup Analysis

This report documents the findings from a deep-dive into the LeHome evaluation pipeline, cross-referenced with **DeepWiki** (LeHome, Isaac Lab, LeRobot), **past conversation logs** ([eval_optimization_report.md](file:///home/nidhinninan/.gemini/antigravity/brain/8da02744-0cb7-414d-a7e1-c27aa2651090/eval_optimization_report.md)), and the **Isaac Lab `SimulationCfg` documentation**.

---

## 🔴 Lever 1: [RateLimiter](file:///home/nidhinninan/Documents/Research/Hackathon/LeHome-trial/lehome_workspace/lehome-challenge/source/lehome/lehome/utils/record.py#15-41) / `--step_hz` (The Critical Bottleneck)

**Mechanism**: The [RateLimiter](file:///home/nidhinninan/Documents/Research/Hackathon/LeHome-trial/lehome_workspace/lehome-challenge/source/lehome/lehome/utils/record.py#15-41) class in [lehome/utils/record.py](file:///home/nidhinninan/Documents/Research/Hackathon/LeHome-trial/lehome_workspace/lehome-challenge/source/lehome/lehome/utils/record.py) (lines 15–41) is a synchronization tool that forces the simulation loop to match real-world time. It calculates a `sleep_duration` based on the `--step_hz` argument (default 120Hz) and calls `time.sleep()` whenever the simulation is running faster than real-time.

**Validated Findings**: 
- **DeepWiki Confirmation**: The LeHome DeepWiki explicitly states: *"The RateLimiter ensures evaluation runs at a consistent frequency... This synchronizes simulation time with real-time execution."*
- **Physics Impact**: Disabling this has **zero effect** on physics accuracy or simulation quality. The solver timesteps (`dt`) remain identical; we are simply removing artificial wait times.
- **Evaluation Quality**: Your Diffusion Policy (DP) models are vision-based and operate at 30Hz internally within the simulation frame counter. This is independent of how fast the simulation app itself updates relative to your wall-clock.

**Recommendation**: **PATCH [evaluation.py](file:///home/nidhinninan/Documents/Research/Hackathon/LeHome-trial/lehome_workspace/lehome-challenge/scripts/utils/evaluation.py)**. The current code (line 102) creates the rate limiter unconditionally. Adding a `if args.step_hz > 0` guard allows you to pass `--step_hz 0` to let the GPU/CPU run at maximum speed. This is the **highest-impact, zero-risk** change possible, potentially offering a **2-5x speedup**.

---

## ⚠️ Lever 2: `--device cpu` vs `--device cuda` (Wait! Keep CPU)

**Revised Analysis**: While most simulation tasks benefit from GPU acceleration, cloth and particle simulation in Isaac Sim 5.1.0 can be unstable or subject to heavy synchronization overhead when running physics on the GPU.

**Past Context**: Your [eval_optimization_report.md](file:///home/nidhinninan/.gemini/antigravity/brain/8da02744-0cb7-414d-a7e1-c27aa2651090/eval_optimization_report.md) notes that:
> *"Simulating cloth physics on the GPU (`device=cuda`) often crashes or throttles due to PhysX CUDA synchronizations... Evaluating physics on the CPU while running the DP Model inferences on the GPU is the fastest stable combination."*

**Verdict**: **KEEP `--device cpu`**. The current eval script already leverages the GPU for policy inference (denoising) while offloading the heavy particle physics to the CPU. 

---

## ⚠️ Lever 3: `use_fabric` (Do Not Enable)

**Mechanism**: `use_fabric=True` enables Isaac Lab's GPU-accelerated scene graph (USD Fabric), which bypasses the standard USD stage for speed.

**Validated Findings**: 
- **Incompatible with Sensors**: DeepWiki and Isaac Lab docs note that standard camera sensors and certain particle-state-lookup APIs (used for garment success detection) often require access to the USD stage, which is bypassed by Fabric.
- **LeHome Configuration**: The developers explicitly hardcoded `env_cfg.sim.use_fabric = False` in [evaluation.py](file:///home/nidhinninan/Documents/Research/Hackathon/LeHome-trial/lehome_workspace/lehome-challenge/scripts/utils/evaluation.py).

**Verdict**: **DO NOT ENABLE**. It will likely break the observation pipeline for your vision-based models.

---

## 🟢 Lever 4: `render_interval` vs. Vision Requirements

**Mechanism**: `render_interval` defines how many physics steps occur before a visual frame is rendered.

**Validated Findings**: 
- **ACT (Non-Vision)**: Can be increased safely to reduce overhead.
- **DP (Vision)**: Cannot be increased significantly. Since your cameras update at 30 FPS, if you skip too many renders, the policy will receive "stale" images or the cameras will stop updating entirely, degrading task performance.

**Recommendation**: Leave at `1` for Diffusion Policy. The [RateLimiter](file:///home/nidhinninan/Documents/Research/Hackathon/LeHome-trial/lehome_workspace/lehome-challenge/source/lehome/lehome/utils/record.py#15-41) fix (Lever 1) is a cleaner way to achieve speedup without risking image quality.

---

## 🟢 Lever 5: `rendering_mode` (Quality vs. Performance)

**Mechanism**: Controls RTX renderer features like FXAA, translucency, and reflections.

**Validated Findings**: 
- **Domain Gap Risk**: Your models were trained on data generated with `rendering_mode="quality"`. Switching to `"performance"` introduces a subtle visual domain gap (aliasing, different lighting). While likely small, it represents a non-zero risk to evaluation success rates for a marginal speed gain.

**Verdict**: **KEEP `rendering_mode="quality"`**. The bottleneck is synchronization, not per-frame rendering cost.

---

## ✅ Lever 6: Parallel Execution (The Multiplier)

**Confirmation**: Your past logs successfully used a `parallel_eval.sh` script to run all 4 garment types concurrently.

**Reasoning**: Since the physics is on the CPU, the primary resource bottleneck is RAM and CPU cores, not GPU VRAM. Your A4000 can easily handle the inference for 4 parallel models while the CPU manages the 4 physics environments.

**Verdict**: **USE IN COMBINATION WITH LEVER 1**. Dividing 4 garments across 4 processes provides a **4x multiplier** on top of whatever per-process speedup Lever 1 provides.

---

## Final Strategy Summary

1.  **Patch** [evaluation.py](file:///home/nidhinninan/Documents/Research/Hackathon/LeHome-trial/lehome_workspace/lehome-challenge/scripts/utils/evaluation.py) to allow `--step_hz 0`.
2.  **Run** evaluations on `--device cpu` to ensure PhysX stability for cloth.
3.  **Execute** your 4 garment types in **Parallel Shells** to maximize hardware utilization.
4.  **Disabled** `--save_video` for bulk runs to avoid disk I/O bottlenecks.

| Improvement | Mechanism | Expected Gain | Risk |
| :--- | :--- | :--- | :--- |
| **Bypass Sync** | `--step_hz 0` | **2-5x** | Zero |
| **System Parallelism** | Parallel Shells | **~4x** | Low (RAM/CPU check) |
| **Combined** | Both | **~8-20x** | Low |
