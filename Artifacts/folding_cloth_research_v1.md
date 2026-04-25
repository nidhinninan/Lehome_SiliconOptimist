# Robotic Garment Manipulation & LeRobot Landscape (2024–2025)

The robotic cloth-folding domain has shifted from rule-based geometry to **end-to-end Imitation Learning (IL)** and **Vision-Language-Action (VLA)** foundation models. The integration of Hugging Face’s **LeRobot** with **NVIDIA Isaac Sim** is the current state-of-the-art (SOTA) for this challenge.

---

## 1. Core LeRobot Policies for Cloth Folding
For deformable objects like cloth, standard policies often struggle with "multi-modality" (the fact that there are many ways to grab a corner). The following policies are recommended:

*   **ACT (Action Chunking Transformer):** Predicts sequences ("chunks") of actions rather than single steps. This creates smoother, more continuous folds and handles the temporal dependencies of cloth manipulation well.
*   **Diffusion Policy (DP):** Superior at handling multi-modal demonstrations. If your robot needs to decide between two equally valid grasp points, DP won't "average" them (which causes failure) but will commit to one.
*   **X-VLA (X-Vision-Language-Action):** A foundation model specifically fine-tuned for dexterous manipulation. The **`lerobot/xvla-folding`** checkpoint achieved a **100% success rate** in real-world tests. It uses "soft-prompting" to adapt to different robot setups without retraining from scratch.
*   **SmolVLA:** A lightweight VLA frequently used in hackathons. It is effective for tasks where spatial reasoning (e.g., "where is the sleeve relative to the torso") is required.

---

## 2. Dataset & Environment Specifications
The **LeHome Challenge Dataset** (`lehome/dataset_challenge_merged`) provides a specialized foundation:

*   **Scale:** 1,000 total episodes (~18 GB) split equally (250 each) across `Top Long`, `Top Short`, `Pant Long`, and `Pant Short`.
*   **Complexity:** Total of ~265,000 frames. 
*   **Observation Space:** 3-stream RGB (480x640) from `top`, `left`, and `right` cameras. This high-resolution multi-view input is critical for perception but drives high VRAM usage (~1.5GB per batch).
*   **Kinematics:** 12-dimensional bimanual joint space (6 joints per arm).
*   **Normalization:** `MEAN_STD` for images and `MIN_MAX` for joint positions.

---

## 3. Strategies to Improve Success (The Subvariant Challenge)
The primary hurdle in evaluation is the **Randomized Loading** of garment subvariants.

*   **The "Subvariant" Gotcha:** Evaluation randomly loads dozens of subvariants (different meshes, textures, and stiffnesses) without providing a category label.
*   **Classifier Router (Recommended):** Train a lightweight **ResNet-18 or ViT classifier** to identify the garment category in the **first frame**. This allows the policy to "hot-swap" expert weights (e.g., loading the `top_long` Diffusion Policy) dynamically.
*   **Unified Training:** Alternatively, train a single VLA (like X-VLA or SmolVLA) on all 1,000 episodes simultaneously. VLAs' semantic understanding of "cloth" makes them more robust to visual subvariants than training ACT from scratch.
*   **Soft MimicGen (SOTA 2025):** Procedurally augment the dataset into thousands of synthetic variations in Isaac Sim to improve robustness against randomized mesh textures.
*   **Recovery Demonstrations:** Explicitly record "correction" data—starting the robot from a failed/slipped state—to handle the non-linear deformations of subvariant fabrics.

---

## 4. Tangential Projects & Resources
*   **`lerobot/xvla-folding` (Hugging Face):** The primary pre-trained model for folding.
*   **`lerobot/unitreeh1_fold_clothes`:** A massive dataset (19k rows) of humanoid cloth folding for pre-training.
*   **Dynamic Cloth Folding (`hietalajulius/dynamic-cloth-folding`):** A key GitHub reference for learning visual feedback control for cloth manipulation (IROS 2022).
*   **GarmentLab:** A unified Isaac Sim benchmark for garment physics (PBD/FEM) matching the LeHome environment.

---

## 5. Isaac Sim Simulation "Sweet Spots"
Align these **PhysX PBD** parameters to ensure Sim-to-Eval transfer:
*   **Bend Stiffness:** `1,000–2,000 N/m` (Critical for realistic garment "flow").
*   **Self-Collision Distance:** `0.005m` (To prevent the cloth from passing through itself).
*   **Friction:** Set cloth-on-cloth friction to `>0.6` to ensure folds "stick."

---

## 6. Advanced Physics Attachments & Gripper Stability
To eliminate cloth slippage (a common failure in simulation), use these advanced stability techniques:

### **A. Physics Tuning for Grasping**
*   **Collider Type:** Change gripper finger colliders from **SDF** to **Convex Hull**. SDFs often cause "tunneling" where cloth particles slip through the mesh.
*   **Friction Combine:** Set `friction_combine_mode` to **Max** for both gripper and cloth materials.
*   **Solver Stability:** Increase `max_position_iteration_count` to **64** and physics frequency to **240Hz/480Hz**.

### **B. PhysX Attachment API**
For complex folds where friction isn't enough, use the **PhysX Attachment API** to "lock" the cloth to the gripper:
*   **`PhysxPhysicsAttachment`:** Programmatically link the gripper link (Actor 1) to the cloth prim (Actor 0) during the grasp.
*   **Auto-Attachment:** Use the `PhysxAutoAttachmentAPI` to automatically find and bind the nearest cloth particles to the gripper surface.

### **C. Kinematic Nodal Control (The "Cheat" Mode)**
If physical attachments cause simulation jitter or "explosions," use **Nodal Control**:
1.  Identify cloth particle indices inside the gripper volume.
2.  Manually override their position/velocity every step to match the gripper's transform.
3.  This provides 100% stability, ensuring the cloth never slips until the gripper command is "release."
