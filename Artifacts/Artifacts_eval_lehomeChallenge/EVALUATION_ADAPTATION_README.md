# Host Computer Adaptation for Evaluation

This document provides the necessary host-side patches to evaluate the **Silicon Optimists** submission using the official `lehome-challenge` repository.

While our submitted Docker image (`nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell`) contains the frozen DINOv2 model and policy weights, the host computer (the machine running Isaac Sim and the evaluation scripts) requires a few critical updates to properly interface with the policy and prevent crashes.

These changes reflect the latest `VM_update` branch patches applied to our internal evaluation instance.

## Why are these changes needed?

1. **`warp-lang==1.11.1` Dependency Fix**: We added this to `pyproject.toml` to resolve a known build/runtime issue with IsaacLab dependencies on the host machine.
2. **`lerobot_eval_with_plugins.py` Wrapper**: A new evaluation entry point that explicitly registers and imports the `lerobot_policy_dino` plugin before starting the evaluation, preventing config decode errors (`Couldn't find a choice class for 'dino_diffusion'`).
3. **Hardened `lerobot_policy.py`**: We added strict validation to ensure the pretrained model directory contains `config.json` and the `.safetensors` weights, and we fixed the action-dimension inference fallback logic to support bimanual tasks accurately.
4. **Task Name Passing in `evaluation.py`**: We updated the evaluation loop to pass the `task_name` argument to the policy adapter, which is required by our hardened bimanual heuristic.

## How to Apply the Host Updates

We have provided a single bash script (`apply_host_updates.sh`) that automatically injects these updates into your fresh clone of the official `lehome-challenge` repository.

### Step-by-Step Instructions

1. **Clone the official repository** (as usual):
   ```bash
   git clone https://github.com/lehome-official/lehome-challenge.git
   cd lehome-challenge
   ```

2. **Copy the patch script** into the repository root:
   Copy the `apply_host_updates.sh` script (provided alongside this README) into your `lehome-challenge` directory.

3. **Run the patch script**:
   ```bash
   chmod +x apply_host_updates.sh
   ./apply_host_updates.sh
   ```
   *You should see output confirming that `pyproject.toml` was updated and the Python scripts were created/overwritten.*

4. **Install dependencies** (now including the `warp-lang` fix):
   ```bash
   uv sync
   ```

5. **Proceed with the main Submission README**:
   You can now follow the rest of the instructions in `README_SUBMISSION.md` (starting from Step 3: Clone and Install IsaacLab).

   **Important Note on the Evaluation Command:**
   When you reach Step 9 in the `README_SUBMISSION.md`, use the new wrapper script `lerobot_eval_with_plugins.py` instead of the default `scripts.eval`.

   For example:
   ```bash
   xvfb-run -a python lerobot_eval_with_plugins.py \
       --policy_type docker \
       --docker_url http://localhost:8081 \
       --garment_type top_long \
       --step_hz 0 \
       --device cpu \
       --headless \
       --enable_cameras
   ```

By following this adaptation guide, your host computer will be perfectly synced with the environment we used to validate the `FullDP-v2_Blackwell` Docker policy.
