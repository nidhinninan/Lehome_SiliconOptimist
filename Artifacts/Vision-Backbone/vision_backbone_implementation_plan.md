# Updated Grounded Implementation Plan: "Fixed-Steps, Time-Measured" Sweep

Based on the [Independent Audit of the Time-Injection](file:///home/nidhinninan/.gemini/antigravity/brain/2fe7fe41-fbd6-46a6-8f32-11a53d48b749/time_injection_audit.md), we are abandoning the messy approach of modifying LeRobot's internal `train.py`. The audit proves that modifying `train.py` inevitably leads to the repository fragmentation you were rightfully concerned about, while failing to cleanly execute evaluations if broken forcefully.

Instead, we will achieve Karpathy's goal of **identifying the most compute-efficient representation architecture** using an elegant, out-of-the-box solution wrapper.

## User Review Required

> [!WARNING]
> This approach contradicts your recent notes in `LEARNING-2.md`. You noted we "have to modify the actual train.py file." 
> **My proposal is that we do NOT modify it.** We can achieve mathematically identical efficiency pressure by fixing the step count and measuring the time it takes, rather than fixing the time and measuring the output.

## Proposed Changes

### [Documentation & Cleanup]

#### [MODIFY] [lehome_change_log.md](file:///home/nidhinninan/Documents/Research/Hackathon/LeHome-trial/lehome_workspace/lehome_change_log.md)
Remove the old, flawed `time_constrained_sweep.py` entry.

#### [DELETE] `lehome_workspace/time_constrained_sweep.py`
Delete the previous flawed script from the workspace.

#### [MODIFY] [LEARNING-2.md](file:///home/nidhinninan/Documents/Research/Hackathon/LeHome-trial/LEARNING-2.md)
Update the learning log to reflect the architectural pivot away from modifying `train.py`.

### [Implementation]

#### [NEW] [throughput_efficiency_sweep.py](file:///home/nidhinninan/Documents/Research/Hackathon/LeHome-trial/lehome_workspace/throughput_efficiency_sweep.py)
A much simpler, safer wrapper script.
1. It calls the standard, unmodified `lerobot-train` pipeline (via `python -m lerobot.scripts.train`).
2. It enforces parameters: `training.offline.steps=2000` and `eval.batch_size=8`, and importantly `eval_freq=2000`.
3. Because the steps are fixed and low, LeRobot will naturally reach the end, naturally evaluate, and naturally save the checkpoint exactly once.
4. The wrapper uses a Python stopwatch around the subprocess.
5. Once complete, it calculates: **`Efficiency Metric = (Evaluation Success Rate) / (Total Wall Clock Minutes)`**

By doing this, a heavy model like DINOv2 might achieve a 0.8 success rate, but if it took 10 minutes, its score is **0.08**. A lighter ResNet18 might achieve a 0.6 success rate, but if it takes 2 minutes, its score is **0.30** (Winning!). This replicates Karpathy's evolutionary pressure perfectly without corrupting the LeHome repo.

## Open Questions

- Does this "Fixed-Steps, Time-Measured" mathematical substitution satisfy your requirement for finding the most efficient backbone? 
- Are you comfortable with me deleting the old `time_constrained_sweep.py` and replacing it with `throughput_efficiency_sweep.py`?

## Verification Plan

- Ensure `throughput_efficiency_sweep.py` relies exclusively on standard Python `subprocess` and parses the standard output of `lerobot-train` without relying on any `SIGTERM` kills.
- Ensure the `lehome_change_log.md` explicitly documents this approach for VM transportability.
