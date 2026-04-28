# Instance Adaptation & Troubleshooting (Principia VM)

**Date Recorded:** 2026-04-28  
**Context:** Environment setup, storage redirection, `uv` package manager issues, and background `rclone` behavior on the Principia VM for the LeHome Challenge.

---

## 1. Critical Error: `uv sync` Fails with "No space left on device"

**Symptom:**  
While running `uv sync`, the process crashes during package extraction (e.g., `llvmlite`):
```text
× Failed to download `llvmlite==0.42.0`
├─▶ Failed to extract archive...
├─▶ I/O operation failed during extraction
╰─▶ failed to flush file `/root/.cache/uv/...`: No space left on device (os error 28)
```

**Root Cause:**
The VM's root partition (/) has run completely out of storage space. Even though the VM has around 93GB of free space on the `/data` partition, `uv` is defaulting to `/root/.cache/uv/` because the required environment variables pointing to `/data` are not active in the current terminal session.

**Resolution:** Clear the clogged space on the root drive, set the correct environment variables, and re-sync:

1. **Clear the failed cache to free up the root drive:**
   ```bash
   rm -rf /root/.cache/uv
   ```
2. **Export the correct cache directories to your current session:**
   ```bash
   export UV_CACHE_DIR="/data/uv_cache"
   export HF_HOME="/data/huggingface_cache"
   ```
   *(Note: Ensure these lines are added to `~/.bashrc` to make them permanent).*
3. **Ensure the cache directories actually exist on the `/data` drive:**
   ```bash
   mkdir -p /data/uv_cache /data/huggingface_cache
   ```
4. **Re-run the sync:**
   ```bash
   uv sync
   ```

## 2. Error: `uv: command not found`

**Symptom:**
When attempting to run `uv sync`, the terminal returns:
```text
bash: uv: command not found
```

**Root Cause:**
The `uv` executable is either not installed, or its installation directory (`/data/.local/bin`) is not in the system's PATH environment variable.

**Resolution:**

1. **Verify uv installation:**
   ```bash
   ls -l /data/.local/bin/uv
   ```
2. **If it says No such file or directory, reinstall uv using the custom path:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/data/.local/bin" sh
   ```
3. **Check `~/.bashrc` for the PATH export:**
   ```bash
   grep 'export PATH="/data/.local/bin:$PATH"' ~/.bashrc
   ```
4. **If it's missing, add it (along with the cache variables if needed):**
   ```bash
   echo 'export PATH="/data/.local/bin:$PATH"' >> ~/.bashrc
   ```
5. **Activate the changes and retry:**
   ```bash
   source ~/.bashrc
   uv sync
   ```

## 3. Rclone Checkpoint Sync Behavior During Training

**Question:**
Does the custom bash script upload checkpoints to rclone and delete them during the training process, or does it wait until the absolute end of the training?

**Answer:**
It does both, handling the intermediate steps and the final checkpoint differently to conserve disk space safely:

* **During Training (`step_*` checkpoints):**
  The script spawns a background loop that monitors the checkpoints folder. It uses `rclone move` for any `step_*` directories older than a minimum age (e.g., 15 minutes). The move command safely uploads the intermediate checkpoint to Google Drive and immediately deletes the local copy upon success, freeing up VM storage continuously.
* **At the Absolute End (Cleanup):**
  When the training exits, the `cleanup_rclone` trap triggers:
  * It uses `rclone copy` for the `last/` checkpoint. This uploads it to Google Drive but keeps the local copy so training can be easily resumed.
  * It performs one final `rclone move` for any remaining `step_*` directories.
  * It runs a local cleanup (`rm -rf`) strictly on the intermediate `step_*` or numeric folders to ensure the workspace is pristine, leaving only `last/`.

## 4. Quitting a VM Terminal Session

To safely disconnect from the VM terminal, use any of these methods:

* **Standard Command:** Type `exit` or `logout` and press Enter.
* **Keyboard Shortcut:** Press `Ctrl + D` (sends an EOF signal, identical to `exit`).
* **Frozen SSH Session:** If the terminal is completely unresponsive, press Enter, then type `~.` (tilde followed by a period) to force the local SSH client to drop the connection.

<!--
[PROMPT_SUGGESTION]How can I create an automated test to ensure my environment variables are permanently fixed?[/PROMPT_SUGGESTION]
[PROMPT_SUGGESTION]Can you help me adjust the rclone background loop timing in the script?[/PROMPT_SUGGESTION]
-->