# Corrected Grounded Autoresearch Integration Plan

Based on the independent audit of the previous assumption, we are pivoting away from the external subprocess wrapper. The wrapper approach was technically flawed because killing the script externally would prevent LeRobot from running its final evaluation or saving the final optimal checkpoint.

This new plan focuses on a purely grounded integration that modifies the internal training step logic of LeRobot.

## User Review Required

> [!WARNING]
> The previous `time_constrained_sweep.py` wrapper approach has been completely discarded.
> We will instead create a patched version of `lerobot_train.py` that internally checks the wall clock and gracefully triggers an end-of-training evaluation when the budget is reached.

## Proposed Changes

### [Research & Documentation]

#### [NEW] [time_injection_audit.md](file:///home/nidhinninan/.gemini/antigravity/brain/2fe7fe41-fbd6-46a6-8f32-11a53d48b749/time_injection_audit.md)
The formal independent audit detailing the technical flaws of the external wrapper approach versus the internal loop modification.

### [Implementation]

#### [DELETE] `lehome_workspace/time_constrained_sweep.py`
Remove the flawed wrapper script.

#### [NEW] [time_constrained_train.py](file:///home/nidhinninan/Documents/Research/Hackathon/LeHome-trial/lehome_workspace/time_constrained_train.py)
A custom script derived from `lerobot.scripts.train`. 
1. We will import the internal `train` logic from LeRobot.
2. We will inject a wall-clock timer directly inside the step iterator.
3. When `time.time() - start_time > TIME_BUDGET`, the script will gracefully break the loop, forcing LeRobot to execute its final evaluation and save the model checkpoint.

#### [MODIFY] [lehome_change_log.md](file:///home/nidhinninan/Documents/Research/Hackathon/LeHome-trial/lehome_workspace/lehome_change_log.md)
Update the running document to reflect the deletion of the wrapper and the addition of the new patched script, ensuring downstream LLMs have the correct code.

## Open Questions

- Since we are creating a modified copy of the `train.py` script (`time_constrained_train.py`), should I pull the raw source of `lerobot.scripts.train` from GitHub using Exa, or would you prefer I build it as a monkey-patch over the installed `lerobot` Python package? (A copied script is generally safer and closer to Karpathy's single-file editing philosophy).

## Verification Plan

### Grounded Verification
- Ensure the injected `break` mechanism cleanly drops the script into the existing evaluation function call without `os._exit()` or `kill` signals.
- Review `lehome_change_log.md` is accurately synchronized.
