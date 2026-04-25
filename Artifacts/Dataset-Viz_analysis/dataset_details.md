# LeHome Challenge Dataset Specifications

This document consolidates all the dataset-related details discussed for the LeHome challenge, specifically focusing on the merged demonstration data used for training the Diffusion Policy and X-VLA models.

### Hugging Face Source
*   **Repository:** `lehome/dataset_challenge_merged`
*   **Type:** Pre-collected expert demonstrations (teleoperation)

### Local Disk Location
*   **Path:** `/data/lehome_workspace/lehome-challenge/Datasets/example/`
*   **Total Disk Usage:** ~18 GB

### Dataset Structure & Frame Counts
The merged dataset contains 1,000 total episodes split equally across 4 garment categories.

| Garment Category | Sub-folder Name | Episodes | Total Frames |
| :--- | :--- | :--- | :--- |
| Top Long | `top_long_merged` | 250 | 83,068 |
| Top Short | `top_short_merged` | 250 | 76,066 |
| Pant Long | `pant_long_merged` | 250 | 65,909 |
| Pant Short | `pant_short_merged` | 250 | 40,755 |
| **Total** | | **1,000** | **265,798** |

### Observation Space (Images)
Every single frame in the dataset captures high-resolution imagery from three distinct vantage points. This is the primary driver of VRAM usage (~900MB–1.5GB per batch sample during training).

*   **Resolution:** 480 x 640
*   **Channels:** 3 (RGB)
*   **Streams (3):** 
    *   `observation.images.top_rgb`
    *   `observation.images.left_rgb`
    *   `observation.images.right_rgb`

### Action Space & State
*   **State (`observation.state`):** 12 dimensions measuring absolute joint positions (`float32`).
    *   **Mapping:** 6 joints per arm (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper).
    *   **Normalization (Diffusion Policy):** Uses `MIN_MAX` scaling.
*   **Action ([action](file:///home/nidhinninan/Documents/Research/Hackathon/LeHome-trial/lehome_workspace/lehome-challenge/scripts/eval_policy/example_participant_policy.py#86-153)):** 12 dimensions specifying target joint positions for both arms.
    *   **Normalization (Diffusion Policy):** Uses `MIN_MAX` scaling.
    *   **Tokenization (VLAs/Foundation Models):** If training an architecture like X-VLA or SmolVLA, this continuous continuous action space must be discretized into bins (e.g., 256 tokens) to interface with the language model's vocabulary decoder.

### Data Normalization
*   **Visual Streams:** Images are processed with `MEAN_STD` normalization mapping (standard ImageNet/ResNet backbone sizing).

### Data Types & Formats
*   **Images:** Stored as `video` dtype in the LeRobot dataset schema (highly compressed MP4 chunks acting as stacked tensors).
*   **Kinematics (State/Action):** Stored as `float32`.
*   **Unit System:** Radians for joints, meters for generic spatial constraints.

### The "Subvariant" Challenge (Randomized Loading)
While there are only 4 high-level "Categories" of garments (e.g., `Top Long`), the dataset and simulation engine contain dozens of **Varieties and Subvariants** (different graphical meshes, specific physics stiffnesses, and visual textures based on real-world clothing).

**How Evaluation Mechanics use Subvariants:**
1.  During evaluation, the system does not load a generic "shirt". 
2.  It parses a `Release_test_list.txt` file (e.g., `Assets/objects/Challenge_Garment/Release/Top_Long/Top_Long.txt`).
3.  This file dictates exactly which subvariant mesh and version to load (e.g., `Top_Long_Seen_0 (Release)`).
4.  **The Gotcha:** The official Hackathon evaluation randomly loads these subvariants across all 4 top-level categories without passing a "label" to your policy. The visual variety across subvariants heavily necessitates either a unified Foundation Model (X-VLA) or a custom Classifier Router to look at the first frame and identify the garment type before passing it to the correct DP expert.
