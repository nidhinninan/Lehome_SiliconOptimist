# DINOv2 classifier training time and router compatibility requirements

Date: 2026-04-29

## Scope

This artifact preserves the technical reasoning from the follow-up discussion after the LFM vision-router research note. It answers two connected questions:

1. How long should a DINOv2-based garment classifier take to train from the LeHome dataset images?
2. Are the requirements for using DINOv2 specialists behind a router already satisfied by the existing DINO top-short / A100-style training scripts?

The key distinction throughout is:

- **Simple router**: a classifier predicts one of four garment classes, then the selected specialist DP checkpoint is loaded or called.
- **True single-backbone router**: one shared DINOv2 image encoder produces features used by both the classifier head and the selected specialist DP.

Those are not the same implementation.

## Dataset scale from `dataset_details.md`

The grounded dataset artifact reports:

| Garment category | Dataset folder | Episodes | Total frames |
| --- | --- | ---: | ---: |
| Top Long | `top_long_merged` | 250 | 83,068 |
| Top Short | `top_short_merged` | 250 | 76,066 |
| Pant Long | `pant_long_merged` | 250 | 65,909 |
| Pant Short | `pant_short_merged` | 250 | 40,755 |
| **Total** | | **1,000** | **265,798** |

Every frame has three RGB streams:

- `observation.images.top_rgb`
- `observation.images.left_rgb`
- `observation.images.right_rgb`

For the router classifier, however, the intended data unit is not every demonstration frame. The classifier is only supposed to answer the episode-level question:

```text
Which specialist DP should handle this garment?
pant_long / pant_short / top_long / top_short
```

Therefore, if the first-frame assumption holds, the natural supervised classifier dataset is:

```text
1 first frame per episode x 1,000 episodes = about 1,000 labeled images
```

If frame 0 is not always ideal, the existing design keeps `frame_idx` configurable. A slightly expanded classifier dataset could sample frames 0-3 or 0-5 per episode after visual validation:

```text
4-6 frames per episode x 1,000 episodes = about 4,000-6,000 labeled images
```

Using all 265,798 frames for this classifier is usually the wrong default. Later trajectory frames may contain robot arms, occlusions, manipulated cloth states, and action-specific artifacts. Those frames are useful for policy learning, but can weaken a first-frame route classifier by teaching it cues that will not exist at the first evaluation observation.

## Expected DINOv2 classifier training time

The recommended classifier shape is:

```text
DINOv2-Small frozen backbone
top_rgb image only
480x640 -> 224x224 resize
4-way linear head or tiny MLP head
moderate image augmentation
25-50 epochs, with early stopping by validation accuracy/confusion matrix
```

Public DINOv2 practice and local project artifacts align on this: DINOv2 works well as a frozen feature extractor / linear-probe backbone for small downstream vision datasets. The backbone has already learned broad visual structure; the LeHome classifier head only needs to learn the four route labels.

### Practical runtime estimates

Assuming the ImageFolder export already exists and the model checkpoint is cached:

| Classifier dataset | Approx. images | Likely GPU runtime |
| --- | ---: | ---: |
| Frame 0 only | ~1,000 | a few minutes |
| Frames 0-3 or 0-5 | ~4,000-6,000 | several minutes to roughly 15 minutes |
| All frames | ~265,798 | likely hours, and not recommended for first-frame routing |

These are order-of-magnitude estimates, not a benchmark from this machine. The true runtime will depend on GPU, DataLoader workers, first-time Hugging Face download/cache, whether images are pre-exported or decoded from LeRobot videos during training, and augmentation cost.

The likely bottlenecks are:

1. first-frame export / video decoding from the LeRobot dataset,
2. image augmentation and DataLoader throughput,
3. first-time DINOv2 download/cache,
4. validation/confusion-matrix passes.

The actual learning problem is small: four classes, balanced at 250 episodes each, with a frozen backbone and small trainable head.

## Should the classifier use all frames?

Default answer: **no**.

A router classifier should match the runtime observation it will use. If the runtime router classifies once at the first observation, train and validate mostly on first observations.

Good expansions:

- frame 0 by default,
- optional frame 1/2/3 if visual inspection shows frame 0 is occasionally imperfect,
- strong but plausible augmentation,
- held-out garment subvariants or held-out visual variants where possible.

Risky expansions:

- arbitrary mid-trajectory frames,
- robot-arm-heavy frames,
- frames where the garment has already been folded or grasped,
- random train/val splits that allow near-duplicate subvariant appearances in both splits.

## Specialist DP training requirements

### Simple router case

For a simple classifier-router policy, the specialist DPs do not need special shared-backbone training.

The runtime flow is:

```text
first observation -> classifier -> class key -> selected DP expert checkpoint
```

This works as long as:

- every class key maps to the correct expert checkpoint,
- each expert was trained on the matching garment family,
- each expert checkpoint can be loaded by the existing LeRobot policy loader,
- observation keys and normalization match evaluation,
- lazy loading or eager loading fits the hardware.

The classifier can be ResNet18, DINOv2, CLIP, or a VLM. The selected expert can be ResNet-DP or DINO-DP. They do not have to share a backbone for correctness. Sharing only matters for memory/latency optimization and feature reuse.

### True single-backbone case

For a true unified pipeline:

```text
one DINOv2 encoder -> classifier head + selected DP expert
```

the specialist DPs need to be trained with the same DINOv2 feature contract expected by the shared encoder.

The requirements are:

- same DINOv2 checkpoint family,
- same frozen/unfrozen choice,
- same image resize,
- same camera set,
- same normalization path,
- same feature extraction strategy,
- same pooling/readout strategy,
- same feature dimension entering the DP U-Net.

If these differ, the selected DP expert cannot safely consume the shared feature tensor. It will expect a different conditioning shape or a different representation distribution.

## Existing DINO top-short launcher status

The relevant script found in the repository is:

```text
lehome_workspace/run_train_dino_dp_top_short_150k.sh
```

There is no exact file named `run_train_dino_top_short_A100.sh` in the current workspace. The top-short script supports an A100-style profile through:

```bash
TOP_SHORT_GPU_PROFILE=a100 ./run_train_dino_dp_top_short_150k.sh
```

It uses:

```text
lehome_workspace/configs/sweep_dino_map_registers.yaml
```

and installs:

```text
lehome_workspace/lerobot_policy_dino
```

### What the top-short script/config already satisfies

For the **top_short specialist DP**, the script satisfies the important DINO specialist requirements:

```yaml
policy:
  type: dino_diffusion
  vision_backbone: facebook/dinov2-with-registers-small
  spatial_pooling: map
  map_num_queries: 8
  use_registers: true
  num_register_tokens: 4
  use_amp: true
```

The policy config includes all three RGB streams:

```yaml
observation.images.top_rgb
observation.images.left_rgb
observation.images.right_rgb
```

The implementation freezes the DINOv2 backbone:

```python
self.backbone = Dinov2Model.from_pretrained(config.vision_backbone)
for param in self.backbone.parameters():
    param.requires_grad = False
```

It resizes LeHome 480x640 images to DINOv2's 224x224 working size before encoding:

```python
img = F.interpolate(
    img,
    size=(224, 224),
    mode="bilinear",
    antialias=True,
)
```

For `spatial_pooling: map`, it uses a trainable MAP head over DINO patch tokens:

```text
patch tokens -> MAPHead(num_queries=8) -> flattened visual condition
```

So for a **DINOv2 MAP+registers top_short DP expert**, the core architecture is already aligned with the desired DINO specialist pattern.

### What it does not satisfy by itself

It does **not** satisfy the entire router/unified-backbone system alone.

Missing or incomplete pieces:

1. **Only top_short is trained by this script by default**
   - Equivalent specialist runs are still needed for:
     - `top_long_merged`
     - `pant_long_merged`
     - `pant_short_merged`
   - They should keep the same DINO config if the goal is architectural consistency.

2. **It does not train the 4-way classifier**
   - This script trains a DP expert, not the route classifier.
   - The classifier still needs its own export/train/eval flow.

3. **The current checked-in router is ResNet18-based**
   - `lehome_workspace/lehome-challenge/scripts/eval_policy/classifier_router_policy.py` loads a torchvision ResNet18 classifier checkpoint.
   - It can route to DINO DP experts, but it does not reuse their DINO backbone.

4. **It does not implement true single-backbone feature sharing**
   - Each DINO DP checkpoint owns its `DinoDiffusionModel` and DINO encoder.
   - A separate DINO classifier would normally instantiate a separate DINO encoder.
   - To share one encoder, the router/policy abstraction would need to be redesigned so the first-frame classifier and selected DP consume the same DINO feature path.

## Relationship to `run_train_dino_merged_a100.sh`

There is also:

```text
lehome_workspace/run_train_dino_merged_a100.sh
```

This script trains a DINOv2 MAP+registers policy on the merged 4-garment dataset for A100-class VMs. That is a different strategy:

- **Merged DP**: one policy learns across all four garment families.
- **Router + specialists**: classifier chooses one of four separate expert policies.

The merged A100 script is useful as a strong unified-policy baseline or fallback, but it is not the same as training four specialist DPs for a router. If using routed specialists, the top-short launcher pattern should be replicated per garment class rather than relying only on the merged run.

## Recommended path

### MVP router path

Use the existing simple router concept:

```text
DINOv2 or ResNet classifier on first top_rgb frame
-> class key
-> lazy-load matching DP expert
```

This requires:

1. Train the 4-way classifier separately.
2. Train or collect four specialist DP checkpoints.
3. Ensure `label_map.json` / class names exactly match expert checkpoint keys:
   - `pant_long`
   - `pant_short`
   - `top_long`
   - `top_short`
4. Keep lazy loading enabled unless eager loading fits VRAM.

In this path, the existing top-short DINO script is already suitable as the template for one specialist.

### Shared-backbone path

Only pursue this after the MVP works.

This requires:

1. All four specialists trained with the exact same DINOv2 MAP+registers feature contract.
2. A DINO classifier head trained on the same feature representation.
3. A router/policy wrapper that keeps one DINO encoder resident and routes features into the chosen expert head/U-Net.
4. Careful checkpoint structure so experts do not redundantly instantiate incompatible encoders.

This is cleaner architecturally but more invasive.

## Bottom line

The DINOv2 classifier should be quick to train because the correct dataset is roughly 1,000 first-frame images, not 265k policy frames. Expect minutes on a GPU once frames are exported and cached.

The existing `run_train_dino_dp_top_short_150k.sh` script already satisfies the DINOv2 MAP+registers requirements for the `top_short` specialist DP. It does not, by itself, satisfy the full router system because it trains only one expert, does not train the classifier, and does not implement true single-backbone feature sharing.

For the immediate router MVP, that is fine: train the classifier separately and use the top-short DINO launcher as the template for the other three specialist experts. Treat true shared-backbone routing as a later optimization.
