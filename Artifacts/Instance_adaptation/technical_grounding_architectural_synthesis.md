Technical Grounding & Architectural Synthesis: LeHome Challenge
Date: 2026-04-28
Subject: Infrastructure Optimization, Vision Backbone Selection, and Performance Bottleneck Mitigation
Environment: Principia VM (96GB Root / 98GB /data partition), 22 vCPUs, RTX 4070 Ti (16GB VRAM)

1. Infrastructure & Storage Logic
1.1 Root vs. Data Partitioning
A critical logical inference was made regarding the VM's disk layout. The / (root) partition is frequently near capacity or limited in throughput, while the /dev/vdb mount at /data provides significantly more overhead (75GB+ available).

The Problem: Standard scripts often use ~/ or ${HOME}, which resolves to /home/principia (located on the full root drive).
The Interpretation: Training failures (e.g., CAS service error: IO Error: No space left on device) occurred because the Hugging Face cache and .venv defaulted to the OS drive.
The Solution: All workspace variables (WORKSPACE_DIR, HF_HOME, UV_CACHE_DIR) must be explicitly mapped to absolute paths starting with /data/ to bypass the OS drive limitations.
1.2 Pathing Syntax Errors
During the transition to absolute paths, a syntax error ($/data/...) was identified. In POSIX shells, the $ prefix is reserved for variable expansion. Using $/data caused the system to attempt creating a literal folder named $ in the root directory, resulting in Permission denied.

2. Permissions & Ownership Context
2.1 The $USER Resolution Trap
In environments where scripts are executed with sudo or within modified shell contexts, the $USER environment variable may resolve to root.

Logical Inference: If chown -R $USER:$USER is executed as root, it locks the directory to the root user. Subsequent non-privileged commands (like git clone or uv sync) then fail with Permission denied.
Hardcoding Requirement: For this specific VM, the user identity principia:principia is the stable target. The setup scripts were updated to use this hardcoded string to ensure that the /data/lehome_workspace remains accessible to the training user regardless of the sudo execution context.
3. Vision Backbone: DINOv2 with MAP + Registers
3.1 Architectural Selection (MAP vs. CLS)
The transition from DINOv2_Baseline to DINOv2_MAP_Registers was driven by a need for fine-grained spatial information.

CLS Baseline: Collapses a 16x16 grid of patches into a single global vector. While efficient, it loses the pixel-level spatial relationships required for cloth manipulation.
MAP (Multi-headed Attention Pooling): Uses $K$ learnable queries (e.g., 8) to cross-attend to the patch tokens.
Flattening Strategy: To maximize spatial knowledge for the Diffusion Policy U-Net, the $K$ queries are concatenated into a single long conditioning vector ($K \times 384$). This preserves distinct geometric concepts (e.g., gripper position vs. cloth fold) rather than averaging them into a global summary.
3.2 The Register Token Solution
A significant research finding (Darcet et al.) noted that DINOv2 generates "scratchpad" peaks in patch tokens with high-norm values.

Inference: Using naive pooling on standard DINOv2 causes the policy to "lock on" to these artifacts rather than the actual garment.
Implementation: The use of facebook/dinov2-with-registers-small allows the model to "dump" these artifacts into dedicated register tokens, which are then sliced out (hidden[:, 1 + num_registers:, :]) before the MAP head processes the spatial grid.
4. Performance & Resource Optimization
4.1 CPU vs. GPU Bottlenecks
Observations showed that even with an RTX 4070 Ti, GPU utilization was spiky (30-100%).

Interpretation: This is a classic "Pipeline Starvation" scenario. The CPU is unable to decode 3x RGB video streams (480x640) and apply transforms fast enough to saturate the GPU.
Thread Capping Logic: With 22 vCPUs, libraries like OpenBLAS and MKL attempt to spawn thread pools of 21+ threads per process. If num_workers=12 is used, the system context-switches excessively.
Optimization: Capping OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1 ensures that CPU cores are dedicated to the DataLoader workers for video decoding, significantly improving data_s (data supply time).
4.2 VRAM Management
The MAP head significantly increases the conditioning dimension ($9216$ dims per timestep for 3 cameras).

Logical Limit: MAP training pushed VRAM to ~98% (15.7/16GB).
Mitigation: policy.use_amp=true (Automatic Mixed Precision) is required to reduce the activation footprint, allowing the larger MAP architecture to fit within 16GB VRAM without reducing the batch_size.
5. Deployment Strategy: The Subvariant Problem
The evaluation logic loads garments randomly from a list of subvariants (different textures/stiffnesses) without providing a category label to the policy.

Technical Inference: A single Diffusion Policy trained on a specific category (e.g., top_short) will fail if the evaluator loads a pant_long.
Proposed Resolution: A two-stage architecture:
Visual Classifier: A lightweight ResNet or ViT looks at the first frame to identify the garment category.
Expert Policy: The classifier dynamically loads the correct DP weights for that specific category.
6. Cleanup & Recovery Procedures
To maintain the limited root partition, specific cleanup patterns were identified:

Zombie Process Removal: sudo fuser -k /dev/nvidia* to clear memory locks after failed runs.
Workspace Purge: Removing accidental OS-drive installations via sudo rm -rf /home/principia/data/....
Checkpoint Pruning: Using find "$OUTPUT/checkpoints" -maxdepth 1 -type d ... -exec rm -rf {} + to prevent the 98GB drive from filling up during long 150k-step runs.