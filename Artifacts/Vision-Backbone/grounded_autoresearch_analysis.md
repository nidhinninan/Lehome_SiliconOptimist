# Grounded Autoresearch & LeRobot Analysis

This report synthesizes the "Deep Repo level programatical understanding" of Karpathy's `autoresearch` and Hugging Face's `lerobot` to establish a grounded framework for vision backbone sweeps.

## 1. Karpathy's Autoresearch: The "5-Minute Invariant"

Based on `karpathy/autoresearch` repository analysis via DeepWiki and Exa:

### Core Design Principles
- **Wall-Clock Priority**: Training is strictly limited to 300 seconds (enforced in `train.py`). This forces the agent to optimize for **throughput** and **efficiency**, not just parameter count.
- **Bits Per Byte (BPB)**: A vocabulary-independent metric (`evaluate_bpb` in `prepare.py`) used to compare models with different architectures or tokenizers fairly.
- **Single-File Mutability**: The agent only edits `train.py`. The infrastructure (`prepare.py`) is immutable to prevent "cheating."

### Grounded Code: The Timer Implementation
```python
# From karpathy/autoresearch/train.py
t_start_training = time.time()
# ... inside training loop ...
if step > 10 and (time.time() - t_start_training) >= TIME_BUDGET:
    break
```

---

## 2. LeRobot Integration: The Training Loop

Analysis of `huggingface/lerobot` suggests that while it primarily uses step-based updates, the script structure is modular enough for time-injection.

### Grounded Code: LeRobot Offline Training
The `lerobot-train` script orchestrates updates. For the Vision Backbone sweep, the "Success Rate" in simulation is the equivalent of the `val_bpb` metric—it represents the final "learned" quality of the representation.

---

## 3. The Synthesis: "Success Rate per GPU Minute"

To maximize "one-shotedness" and grounding, we will implement a metric called **"SR-per-Minute."**

| Factor | Calculation | Rationale |
| :--- | :--- | :--- |
| **Output** | Max Success Rate (Eval) | Measures task competency. |
| **Effort** | Total Wall Time (Train + Eval) | Measures real-world compute cost. |
| **Metric** | `SR / total_minutes` | Favors backbones that learn fast and run efficiently. |

### Adaptation of Autoresearch Adaptations
Looking at forks like `AutoResearchClaw` and `tonitangpotato/autoresearch-engram`:
- We can implement a simple **Persistent Memory** (JSON) to track which backbones have been tested across different learning rates or batch sizes.
- However, per your preference for **"strict backbone comparison,"** our initial wrapper will focus on identifying the most efficient vision representation (e.g., DINOv2 vs. ResNet-18) within a fixed time window.

---

## 4. Grounded Plan for `time_constrained_sweep.py`

Identified specific `lerobot` CLI arguments to use in the wrapper:
- `policy.vision_backbone=...`
- `dataset.repo_id=...`
- `training.offline.steps=999999` (set high, to be killed by the timer).

> [!NOTE]
> This analysis was generated using repo-level context from DeepWiki and Exa, ensuring all script locations and parameter assumptions are grounded in the actual source code.
