# Robotic Garment Manipulation & LeRobot Landscape (2024–2025) - V2

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
The **LeHome Challenge Dataset** provides a specialized foundation for training these policies:

*   **Merged Dataset (`lehome/dataset_challenge_merged`):** 1,000 expert teleoperation episodes (~18 GB) split equally (250 each) across:
    *   `Top Long` (83,068 frames)
    *   `Top Short` (76,066 frames)
    *   `Pant Long` (65,909 frames)
    *   `Pant Short` (40,755 frames)
*   **Observation Space:** 3-stream RGB (480x640) from `top`, `left`, and `right` cameras. This high-resolution multi-view input is critical for depth perception but results in high VRAM usage (~1.5GB per batch).
*   **Action/State Space:** 12-dimensional bimanual joint space (6 joints per arm).
*   **Normalization:**
    *   **Images:** `MEAN_STD` (standard ImageNet scaling).
    *   **Kinematics:** `MIN_MAX` scaling for absolute joint positions.

---

## 3. Strategies to Improve Success (The Subvariant Challenge)
The primary hurdle in the LeHome evaluation is the **Randomized Loading** of garment subvariants.

*   **The "Subvariant" Gotcha:** Evaluation does not use generic categories but randomly loads dozens of subvariants (different meshes, textures, and stiffnesses) without providing a label.
*   **Classifier Router (Recommended):** Train a lightweight **ResNet-18 or ViT classifier** to identify the garment category in the **first frame** of the episode. This allows the policy to "hot-swap" expert weights for the specific garment type.
*   **Unified Training:** Alternatively, train a single VLA (like X-VLA or SmolVLA) on all 1,000 episodes simultaneously. VLAs' semantic understanding of "cloth" as a concept makes them more robust to these visual subvariants than training ACT from scratch.
*   **Soft MimicGen (SOTA 2025):** Procedurally augment the 1,000-episode dataset into thousands of synthetic variations in Isaac Sim to improve robustness against randomized mesh textures and physical parameters.
*   **Recovery Demonstrations:** Explicitly record "correction" data—start the robot from a failed/slipped state—to handle the non-linear deformations of subvariant fabrics.

---

## 4. Tangential Projects & Resources
*   **`lerobot/xvla-folding` (Hugging Face):** The primary pre-trained model for folding.
*   **`lerobot/unitreeh1_fold_clothes`:** A massive dataset (19k rows) of humanoid cloth folding for pre-training.
*   **GarmentLab:** A unified Isaac Sim benchmark for garment physics (PBD/FEM) matching the LeHome environment.

---

## 5. Isaac Sim Simulation "Sweet Spots"
Align these **PhysX PBD** parameters to ensure Sim-to-Eval transfer:
*   **Bend Stiffness:** `1,000–2,000 N/m` (Critical for realistic garment "flow").
*   **Self-Collision Distance:** `0.005m` (To prevent the cloth from passing through itself).
*   **Friction:** Set cloth-on-cloth friction to `>0.6` to ensure folds "stick."
