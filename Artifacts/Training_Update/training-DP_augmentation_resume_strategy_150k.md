# Augmentation-Enabled Resume Strategy — 93K to 150K Steps

## 1. Status at Step 93K
Current training has reached a **loss of 0.022**, meaning it is consistently reproducing robot demonstrations with very high accuracy. However, with only 250 episodes, there is a probability of "memorization" (overfitting to exact pixels).

## 2. The Final Push: 93K → 150K
To maximize the success rate in the LeHome evaluation, we are transitioning from pure **Behavioral Cloning (BC)** to **Augmented BC**.

### Resume Strategy
Instead of finishing the original training run (which uses no augmentations), we will **resume** from the 80K or 93K checkpoint while enabling the full suit of 6 image transforms:

**Command (Draccus Overrides):**
```bash
./resume_with_augmentations.sh outputs/train/dp_top_short/checkpoints/last
```

### Expected Benefits
*   **Visual Invariance:** The model will no longer rely on exact pixel intensities.
*   **Robustness:** Brightness, jitter, and rotation/affine transforms force the model to look for the "shape" of the sleeve and hand rather than static coordinates.
*   **Successful Evaluation:** This strategy is the best way to leverage the remaining 60K training steps to gain generalization without the cost of a full "from-scratch" run.

## 3. Augmentation Defaults (Verified)
| Transform | Range | Weight |
|---|---|---|
| Brightness | [0.8, 1.2] | 1.0 |
| Contrast | [0.8, 1.2] | 1.0 |
| Saturation | [0.5, 1.5] | 1.0 |
| Hue | [-0.05, 0.05] | 1.0 |
| Sharpness | [0.5, 1.5] | 1.0 |
| Affine | deg: ±5°, trans: 5% | 1.0 |

> [!TIP]
> The loss will likely jump from **0.022 → ~0.030** immediately upon resume. **This is good.** It means the model is being challenged by the new "noisy" environment and is actually learning to be robust. By 150K, it should settle back down to ~0.018–0.020 with much higher general reliability.
