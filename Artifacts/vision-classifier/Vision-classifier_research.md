# Research Report: Garment Classification for Robotic Folding

This report synthesizes repository-level understanding, external research, and technical best practices for training a visual classifier to route specialized Diffusion Policy experts in the LeHome Challenge.

## 1. Repository-Level Insights (huggingface/lerobot)

Based on **DeepWiki** analysis of the LeRobot codebase:
- **ProcessorStep Architecture**: LeRobot uses a modular `ProcessorStep` system (`src/lerobot/envs/processors.py`). While there is no native "ClassifierRouter," the `RewardClassifierProcessorStep` provides a template for vision-based categorical tagging (success/failure) that can be adapted for multi-class garment identification.
- **Image Preprocessing**: Correct handling of `observation.images` is critical. `XVLA` and `LIBERO` pipelines demonstrate the need for specific normalization and dimension permuting (`B, H, W, C` → `B, C, H, W`) before backbone entry.
- **Metadata Management**: Episode indexing relies on `dataset_from_index + frame_idx`. The export script MUST correctly use these global indices to avoid frame-drift across the 1000-episode dataset.

## 2. Garment Perception Research (Exa Deep Search)

Key findings from state-of-the-art robotic folding research:
- **Dataset Benchmarks**: The **Flat'n'Fold** (ICRA 2025) dataset is the most relevant baseline, featuring 44 garments across 8 categories. It highlights that "crumpled state" classification is significantly more difficult than "smoothed state" classification.
- **SpeedFolding (BiMaMa-Net)**: Successful systems often use a dedicated classification head trained on both human and self-supervised robot data.
- **OOD Challenges**: Research from **NVIDIA** and **ETH Zurich** emphasizes that classifier failure on in-production factory floors (or hackathon test sets) is usually caused by variable lighting and motion blur, which standard ImageNet models don't encounter.

## 3. Vision Backbone Comparison for Classification

| Backbone      | Pros                                                                                                                   | Cons                                                                               |
| :------------ | :--------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------- |
| **ResNet-18** | Lightweight, fast inference, easy to fine-tune on small datasets.                                                      | Prone to overfitting on specific garment textures; lower abstract feature quality. |
| **DINOv2**    | **Highly Recommended.** Exceptional "frozen" features for linear probes; extremely robust to pose/lighting variations. | Requires higher VRAM; larger checkpoint size for the router config.                |
| **CLIP**      | Strong zero-shot capabilities; understands semantic categories natively.                                               | Can be "distracted" by background text or task-irrelevant visual cues.             |

## 4. Optimized Augmentation Strategies

For top-down (overhead) robotic views, standard augmentations are insufficient. The following are critical for a robust router:
1. **Elastic Warping**: Simulates the stretching and bunching of deformable fabrics.
2. **Perspective Skewing**: Critical for generalizing across different camera mounting heights/angles on the VM.
3. **RandomResizedCrop (8% to 100%)**: Helps the model focus on local fabric patterns rather than just global shapes (which are highly variable for crumpled clothes).
4. **ColorJitter (High Saturation/Brightness)**: Mimics the harsh lighting or shadows found in robotic workcells.

## 5. Training Strategy & Router Integration

- **Phase 1: Head Warming**: Freeze the ResNet-18 backbone for 5 epochs to allow the new 4-way FC layer to align with existing features.
- **Phase 2: Progressive Fine-Tuning**: Unfreeze the backbone with a lower learning rate (`1e-4` vs `1e-2`) to adopt high-level features for specific garment categories.
- **Authoritative Label Mapping**: The `label_map.json` MUST be generated from the training script's `ImageFolder` alphabetic sort to ensure the `classifier_router_policy.py` never misroutes a "pants" image to a "top" expert.

## 6. Optimized Unified Vision Pipeline

### The Architecture Reality

You have **5 separate neural networks** that need to run:
1. **1 Classifier** → predicts garment class (0-3)
2. **1 of 4 Expert U-Nets** → generates the folding actions for that specific garment

The "Unified Pipeline" idea provides a highly efficient way to handle this without massive VRAM overhead:

```
At inference time:
──────────────────────────────────────────────────────
1. Raw Image (3 cameras) → DINOv2 backbone (frozen, ALWAYS loaded)
                                     │
                        ┌────────────┴─────────────┐
                        ▼                          ▼
               Classifier Head           Feature Vector (512d)
             (tiny 4-way FC layer)               │
                        │                         │
               Garment Class (0-3)                │
                        │                         │
                        ▼                         │
          Load the right U-Net expert ◄───────────┘
          (Top / Bottom / Sleeve / Pants)
                        │
                    Actions
──────────────────────────────────────────────────────
```

**What is actually shared:**
*   **DINOv2 backbone:** Loaded **once** in VRAM, never swapped. ~22M frozen params.
*   **U-Nets:** Only **one** of the 4 is ever in VRAM at a time. Each is ~270M learnable params.
*   **Classifier head:** Tiny linear layer, ~2K params. Negligible.

**Key Implementation Constraint:**
For this to work efficiently, **all 4 specialized DP policies must be trained with DINOv2 as their backbone**. This allows you to load the backbone once, classify in one pass, and then route the same features directly into the selected U-Net expert.

## 7. Strategy Note: Router vs. Policy

Since the 10K micro-sweep has proven DINOv2's superiority in both speed and convergence, we use DINOv2 for both the Policy and the Classifier:

- **The Policy:** Use DINOv2-Small as the frozen backbone for specialized Diffusion Policies (Pants-Expert, Top-Expert, etc.).
- **The Classifier:** Use a DINOv2-Small "Linear Probe" (just a single FC layer on top of a frozen DINOv2 backbone) as the classifier.

This creates a **Unified Vision Pipeline**: the system only needs to run the DINOv2 forward pass once to both classify the garment AND generate the expert action, preventing the "Double Backbone VRAM" issue that would occur if mixing ResNet and DINO.