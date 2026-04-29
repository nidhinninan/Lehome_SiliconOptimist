# Preliminary research: LFM Vision VLM as a first-frame garment router

Date: 2026-04-29

## Question

Can a Liquid LFM vision-language model be fine-tuned on LeHome first-frame garment images and still classify clothes outside the exact training dataset into the same four implicit routing categories (`pant_long`, `pant_short`, `top_long`, `top_short`) for selecting specialist Diffusion Policy experts?

## Short answer

Yes, this is technically plausible, but it should be treated as an experimental router candidate rather than a guaranteed replacement for the current lightweight ResNet18/DINOv2 plans.

The strongest version of the idea is:

1. Use the existing first-frame export pipeline (`frame_idx=0` by default, configurable) to create supervised image/label examples.
2. Prompt an LFM vision-language model to choose exactly one of the four routing buckets.
3. Use structured decoding or constrained output so the model cannot emit free-form labels.
4. Establish a zero-/few-shot baseline first, then LoRA fine-tune if baseline routing is not good enough.
5. Validate on held-out garment subvariants and corruption/augmentation sets before trusting OOD routing.

## Naming correction from Exa research

Public Liquid AI material does not show a "LFM Vision 1M" model as of this research pass. The relevant public vision-language models are:

- `LFM2.5-VL-1.6B` - current recommended 1.6B VLM.
- `LFM2-VL-450M` - smallest/fastest public VLM.
- `LFM2-VL-1.6B` - older/deprecated relative to LFM2.5-VL-1.6B.
- `LFM2-VL-3B` - higher-capacity LFM2 VLM.

So "1 million" is likely a shorthand/misread of `1.6B` or confusion with the smaller `450M` model. If a separate 1M-parameter Liquid vision model exists outside the public docs, it was not found in this Exa pass.

## Evidence from Exa

Liquid's vision model docs describe LFM vision models as LFM text backbones paired with dynamic SigLIP2 image encoders, with current public sizes centered on 450M, 1.6B, and 3B-scale checkpoints.

Liquid's car-maker identification example is directly relevant because it is an image classification task built with a VLM:

- It evaluates base LFM2-VL models on an image classification task.
- It uses structured generation to force predictions into an allowed label set.
- It fine-tunes LFM2-VL checkpoints with LoRA.
- It explicitly says the approach transfers to other image classification tasks.

The most important lesson from that example is not just "fine-tune a VLM"; it is "constrain the output space." In their baseline, raw text generation can fail by emitting labels outside the allowed set. Structured generation improves robustness by forcing predictions to one valid class. For LeHome, the equivalent schema should be exactly:

```text
pant_long | pant_short | top_long | top_short
```

or a JSON object such as:

```json
{"garment_route": "top_short"}
```

with `garment_route` constrained to those four literals.

## Fit to the current LeHome router design

The current repo/artifact path assumes a compact classifier:

- `scripts/garment_classifier/sanity_check_first_frames.py`
- `scripts/garment_classifier/export_garment_classifier_dataset.py`
- `scripts/garment_classifier/train_garment_classifier.py`
- planned router policy behavior: classify once on the first observation, then route to one specialist DP expert for the episode.

An LFM VLM can use the same exported first-frame data, but it changes the inference contract:

| Current ResNet18/DINO-style router | LFM VLM router |
| --- | --- |
| Image tensor -> logits over 4 classes | Image + prompt -> constrained text/JSON label |
| Small checkpoint and low first-step latency | Larger checkpoint and higher first-step latency |
| Easy `label_map.json` alignment with ImageFolder | Must align prompt/schema labels with DP expert keys |
| Natural fit for PyTorch classifier router | Requires VLM processor/model loading and text decoding/parsing |
| Less semantic prior beyond ImageNet/self-supervised features | Broader pretrained vision-language prior may help OOD semantics |

The VLM should therefore be evaluated as a router backend, not as a drop-in replacement for `classifier_best.pt`.

## Expected OOD behavior

The reason this idea is attractive is that a VLM may already understand broad clothing concepts such as pants, shorts, long sleeves, short sleeves, tops, and garment silhouettes before seeing the LeHome sim frames. That could help with clothes outside the exact training set, especially if they differ by texture/color but remain semantically close to the four buckets.

However, fine-tuning only on LeHome sim first frames can still over-specialize the model to simulation artifacts. It may learn camera/background/color shortcuts if the dataset is small or class-balanced in a brittle way. The safe claim is:

- **Likely upside**: better semantic prior than a small classifier trained only on the exported sim crops.
- **Not guaranteed**: reliable routing for truly novel garments, unusual folds, lighting, occlusion, or camera shifts.
- **Required mitigation**: held-out subvariants, corruption tests, prompt/label ablations, confidence or margin thresholds, and confusion-matrix review.

For this project, "OOD" should be defined pragmatically as:

1. garment instances/subvariants not used in router training,
2. first frames with held-out colors/textures,
3. alternate `frame_idx` values if frame 0 is imperfect,
4. augmented lighting/blur/crop/pose variants,
5. any hackathon eval-like dataset split that differs from the four specialist training sets.

## Recommended MVP experiment

Do this before implementing a full router integration:

1. **Reuse existing export data**
   - Export first frames with the existing ImageFolder script.
   - Keep the authoritative four labels identical to the DP expert route keys.

2. **Baseline without fine-tuning**
   - Evaluate `LFM2.5-VL-1.6B` if available in the target runtime.
   - If memory is tight, also evaluate `LFM2-VL-450M`.
   - Prompt: "Classify this overhead garment image for robotic folding. Choose exactly one route: pant_long, pant_short, top_long, top_short."
   - Use structured/constrained output if the runtime supports it.

3. **LoRA fine-tune only if needed**
   - Fine-tune on image + prompt -> route label pairs.
   - Keep a held-out validation split by garment instance/subvariant where possible, not just random frames.
   - Watch for overfitting: near-perfect train accuracy with class-specific OOD failures is a bad router.

4. **Compare against the simple router**
   - Compare ResNet18, DINOv2 linear probe, base VLM, and LoRA VLM on the same splits.
   - Choose the smallest/fastest model that meets routing reliability.

5. **Only then integrate**
   - Add a router backend abstraction if the VLM wins on held-out/OOD routing enough to justify added latency and memory.
   - Preserve the "classify once on first frame, route for the episode" control flow.

## Sources checked

- Liquid AI docs, "Vision Models": `https://docs.liquid.ai/lfm/models/vision-models`
- Liquid AI docs, "Fine tuning LFM2-VL to identify car makers from images": `https://docs.liquid.ai/examples/customize-models/car-maker-identification`
- Liquid AI docs, "LFM2-VL-450M": `https://docs.liquid.ai/lfm/models/lfm2-vl-450m`
- Liquid AI blog, "LFM2-VL: Efficient Vision-Language Models": `https://liquid.ai/blog/lfm2-vl-efficient-vision-language-models`
- Hugging Face docs, "LFM2-VL": `https://huggingface.co/docs/transformers/model_doc/lfm2_vl`
- Hugging Face model card, `LiquidAI/LFM2.5-VL-1.6B`: `https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B`

## Go/no-go gates

Use a VLM router only if it clears these practical gates:

- Accuracy is higher than the ResNet18/DINO baseline on held-out garment subvariants.
- Confusions are safe enough for policy routing; e.g. `top_short` vs `top_long` errors may still be damaging, but `top_*` vs `pant_*` errors are especially severe.
- First-step latency is acceptable for evaluation.
- VRAM does not interfere with lazy loading the selected specialist DP.
- Outputs are constrained to the four valid route keys.
- The implementation has a deterministic fallback for low-confidence or malformed predictions.

## Bottom line

Fine-tuning an LFM VLM on first frames is feasible and may improve semantic generalization outside the exact training dataset. The best preliminary recommendation is to run it as an MVP benchmark against the existing ResNet18/DINO router candidates, using constrained four-label output and held-out/OOD evaluation. If it wins materially, integrate it as an alternate router backend; if not, keep the lightweight classifier path for lower memory, lower latency, and simpler deployment.

