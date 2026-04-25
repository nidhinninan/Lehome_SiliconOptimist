# Implement ResNet18 Garment Classifier and Routing Policy

This plan outlines the steps required to build a 4-way visual classifier using a ResNet18 backbone. The classifier will identify garment categories (e.g., `pant_short`, `top_long`) from overhead camera images and route execution to specialized Diffusion Policy experts during the LeHome evaluation.

## User Review Required

> [!WARNING]
> Please review the architecture map and specifically the **Augmentations** phase. We are defaulting to ResNet-18 for memory efficiency, allowing all four DP experts and the classifier to reside in memory (or fast-swap) without severe VRAM bottlenecks. 
> 
> Are you satisfied with the proposed directory structures under `lehome_workspace/` and `scripts/`?

## Proposed Changes

We will execute this plan in **Two Parts**, allowing you to train the classifier before writing any integration code for the LeRobot evaluator.

### 🔴 PART 1: Data Extraction & Classifier Training (Execute Now)

You do not need the 4 DP models or the Router policy to complete this stage. The goal here is purely to generate `classifier_best.pt`.

#### [NEW] `scripts/garment_classifier/sanity_check_first_frames.py`
A lightweight script that samples 15-20 random episodes from the 4 garment datasets (`pant_long`, `pant_short`, `top_long`, `top_short`) using the `LeRobotDataset` API. Extracts `observation.images.top_rgb` (or `observation.images.top`) at `frame_idx=0` and saves them to a review folder (e.g., `garment_classifier_data/sanity_check/`). This is critical for visually verifying that `frame_idx=0` reliably captures the fully laid-out garment without occlusions before bulk-exporting the entire dataset.

#### [NEW] `scripts/garment_classifier/export_garment_classifier_dataset.py`
A script that parses four local LeRobot datasets (e.g., `pant_long_merged`, `top_short_merged`), reads `frame_idx=0` (or `1`), extracts the `observation.images.top_rgb` frame, and saves it into `train/class_name/` and `val/class_name/` directories based on a defined validation split.

#### [NEW] `scripts/garment_classifier/train_garment_classifier.py`
This script trains a PyTorch `torchvision.models.resnet18` model. 
Key implementation details:
- **Augmentations**:
  - `RandomResizedCrop(224, scale=(0.7, 1.0))` (Focuses on texture/wrinkles)
  - `RandomAffine(degrees=15, translate=(0.1, 0.1))` (Mimics mild positional and rotational shifts)
  - `ColorJitter(brightness=0.25, contrast=0.25)` (Mimics variable sim lighting)
  - `GaussianBlur(kernel_size=3)` (Accounts for motion blur/camera focus differences)
- **Training Strategy**: 
  - Freeze the ResNet backbone for the first few epochs (Linear Probe).
  - Unfreeze the network and train end-to-end with a lower learning rate.
- **Output**: Generates a `.pt` checkpoint and a `label_map.json` mapping numeric indices exactly to `ImageFolder` alphabetic ordering.

***

### 🟡 PART 2: The Router Policy Integration (Deferred)

Once Part 1 is fully tested and you have your `classifier_best.pt`, we will implement the LeRobot wrapper.

#### [NEW] `lehome_workspace/lehome-challenge/scripts/eval_policy/classifier_router_policy.py`
This class interfaces with the LeHome evaluator.
- On the first frame of a new episode (`select_action` call where internal state is empty):
  - Extracts the overhead image, handling fallback naming differences (`observation.images.top_rgb` vs `observation.images.top`).
  - Passes it through the ResNet18 classifier to identify the garment class.
  - Dynamically manages LeRobot DP experts: If `lazy_load` is true, it loads the mapped policy into VRAM just for this episode and clears it on `reset()`. If eager loading, it selects from a pre-loaded dictionary of experts to skip the cold-start penalty.
- For all subsequent frames, it routes observations directly to the selected DP model.

#### [NEW] `scripts/garment_classifier/example_router_config.json`
A JSON configuration file telling the router where the classifier checkpoint is, and where the 4 DP checkpoints are. It specifically includes `"lazy_load": true` to prevent OOM errors on single-GPU hardware and `"min_confidence": 0.0` for potential fallback strategies.

#### [MODIFY] `lehome_workspace/lehome-challenge/scripts/eval_policy/__init__.py`
Update the `__init__.py` to expose the new `classifier_router` policy type to the registry.

## Verification Plan

### Remote VM Execution (User-Led)
> [!IMPORTANT]
> Because the local environment lacks `torch` and `lerobot`, all script testing and training MUST be executed on the **LeHome VM**.

1. **Part 1: Data Extraction**
   - Run `python scripts/garment_classifier/sanity_check_first_frames.py` on the VM with a few samples per class.
   - Visually inspect the generated images in the VM's file browser to ensure `frame_idx=0` is the correct target.
   - Run `python scripts/garment_classifier/export_garment_classifier_dataset.py` to build the full `ImageFolder`.

2. **Part 1: Training**
   - Execute `python scripts/garment_classifier/train_garment_classifier.py` on the VM.
   - Monitor the `val_acc` and confusion matrix in the stdout logs.
   - Verify that the `classifier_best.pt` and `label_map.json` are generated in the `--output_dir`.

3. **Part 2: Integration**
   - Test `classifier_router_policy.py` in the Isaac simulation loop using random garments.
   - Check the VM logs to confirm the correct DP expert is being instantiated on the first frame.

### Workspace Sync
- [ ] Add new scripts to `lehome_workspace/vm_transfer_list.md` to ensure they are synchronized with the VM.
- [ ] Log all script creations/modifications in `lehome_workspace/lehome_change_log.md`.
