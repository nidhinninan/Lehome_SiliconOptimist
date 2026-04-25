# DP Training Strategy & Evaluation Guide: LeHome Challenge

This guide outlines how to handle the official evaluation protocol, which uses **randomized garment loading** without category labels.

---

## 🔍 The Evaluation Reality
The hackathon's hidden test set will load garments from all 4 categories randomly. Your policy will **not** be told which garment is in the scene. 

> [!WARNING]
> If you train 4 separate DP models (one per garment) and simply provide one to the evaluator, it will fail on 75% of the episodes.

---

## ⚡ Attack Vector 1: The Unified Policy (Single DP)
Train one Diffusion Policy model on the **complete merged dataset** (all 4 garments).

- **Pros:** Simplest submission (one checkpoint). No need for custom code.
- **Cons:** Extremely hard to converge. The model might "blur" different folding trajectories together, leading to sloppy execution.
- **Best for:** When you have a lot of compute time and want to avoid complex "wrapper" code.

---

## 🚀 Attack Vector 2: The Multi-Policy Router (Custom Policy)
Maintain your 4 highly-accurate DP models and wrap them in a **Decision Layer**.

1. **The Classifier:** Train a tiny image classifier (ResNet18 or MobileNet) to predict the garment type from the `top_rgb` camera.
2. **The Logic:**
   - On the first frame of an episode, run the classifier.
   - If it predicts "pants," load the `dp_pant_long` weights.
   - If it predicts "shirt," load the `dp_top_long` weights.
- **Pros:** Maximizes accuracy for each garment type. Uses the models you already have.
- **Cons:** Requires implementing a [CustomPolicy](file:///home/nidhinninan/Documents/Research/Hackathon/LeHome-trial/lehome_workspace/lehome-challenge/scripts/eval_policy/example_participant_policy.py#14-153) class based on the hackathon's [example_participant_policy.py](file:///home/nidhinninan/Documents/Research/Hackathon/LeHome-trial/lehome_workspace/lehome-challenge/scripts/eval_policy/example_participant_policy.py).
- **Best for:** High-accuracy performance if you have already trained per-garment experts.

---

## 🤖 Attack Vector 3: The VLA/Foundation Model
Bypass category-switching entirely by using a Vision-Language-Action model like **X-VLA** or **SmolVLA**.

- **Pros:** Naturally handles visual variety. Generalizes better to unseen garments.
- **Cons:** Massive VRAM requirements (requires A100 or L40S). Slower inference.
- **Best for:** If you have access to high-end hardware and want a more "future-proof" AI agent.

---

## 🎯 Recommended Action Plan

1. **Step 1:** Train your expert per-garment DP models to get a baseline score.
2. **Step 2:** Decide on your "Router" strategy. If the experts score high, it is worth the effort to wrap them in a classifier.
3. **Step 3:** If the experts still struggle, pivot to an **X-VLA** training run on an A100/L40S to leverage pre-trained visual common sense.
