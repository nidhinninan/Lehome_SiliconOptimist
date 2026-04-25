# GPU Profitability Report: LeHome Challenge Baselines

This report evaluates the **cost-effectiveness** (Work Done per Credit) of various GPU architectures when training **Diffusion Policy (DP)**, **ACT**, and **X-VLA** models.

## GPU Tier Comparison (Brev/Principia Economics)

| GPU Model | Arch | VRAM | Your Price/hr | Throughput / $ (DP/ACT) | Throughput / $ (X-VLA) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RTX A4000** | Ampere | 16GB | **$0.33** | 🏆 **Maximum** | ❌ **Incompatible (OOM)** |
| **NVIDIA L4** | Ada | 24GB | ~$0.50 | 🥈 **High** | 🥇 **Maximum (Value)** |
| **NVIDIA L40S** | Ada | 48GB | ~$1.30 | 🥉 **Medium** | 🥈 **High (Perf)** |
| **NVIDIA A100** | Ampere | 80GB | $1.87 | 📉 **Lowest** | 🥉 **Medium** |
| **NVIDIA T4** | Turing | 16GB | ~$0.25 | ❌ **Very Poor** | ❌ **Incompatible** |

---

## Technical Context: "The Bottleneck Shift"

### 1. The vCPU/DataLoader Bottleneck (DP & ACT)
For Diffusion Policy and ACT, the GPU is rarely the limit. The bottleneck is the **CPU decoding and augmenting images**.
*   **The A100 Trap:** A single A100 instance shares its vCPUs. If you run 4 trials, they choke each other out.
*   **The A4000 Edge:** At **$0.33/hr**, the A4000 is the "Value King." Running 4 separate A4000 instances gives you **4x the dedicated vCPU bandwidth** for only **$1.32/hr** total—cheaper and faster than one A100.

### 2. The VRAM Wall (X-VLA)
X-VLA (0.9B parameters) requires significant memory for **Optimizer States** (Adam). 
*   **16GB (A4000/T4):** Will consistently Out-of-Memory (OOM) during training.
*   **24GB (L4):** The **best value** for X-VLA. It fits the model securely with `bfloat16` and small batch sizes (~1-4).
*   **48GB (L40S):** The **best performance** for X-VLA. It allows for larger batch sizes (8-16) and faster convergence than the L4, while being significantly cheaper than the A100 ($1.30 vs $1.87).

---

## Final Strategy Recommendations

### ⚡ Setup A: The "Parallel Researcher" (Diffusion Policy / ACT)
*   **Hardware:** **4x RTX A4000 (16GB)** instances.
*   **Goal:** Maximum hyperparameter search throughput.
*   **Benefit:** Lowest cost-per-trial. Isolated failure points (one crash doesn't kill all 4).

### 🚀 Setup B: The "Foundation Tuner" (X-VLA Custom Policy)
*   **Hardware:** **1x NVIDIA L40S (48GB)** (First Choice) or **1x A100 (80GB)** (Backup).
*   **Goal:** Fine-tuning the 0.9B parameter model on the merged 4-garment dataset.
*   **Benefit:** L40S has newest generation Tensor Cores and enough VRAM to handle the large transformer blocks without OOM.

---

## 🚨 Independent Audit v2: AMP-Corrected Epoch Analysis

> [!IMPORTANT]
> The original estimates did **NOT** account for Mixed Precision (`use_amp=True`) or `gradient_checkpointing`. This audit corrects those assumptions using the verified frame counts from the `lehome/dataset_challenge_merged` repository.

### VRAM Budget: X-VLA (0.9B params)

| Memory Component | FP32 (No AMP) | BF16 + Grad Ckpt |
| :--- | :--- | :--- |
| Weights | 3.6 GB | **1.8 GB** |
| Gradients | 3.6 GB | **1.8 GB** |
| Adam States (always FP32) | 10.8 GB | **10.8 GB** |
| **Fixed overhead** | **18.0 GB** | **14.4 GB** |
| Activations per sample | ~4–6 GB | **~1.2 GB** (checkpointed) |

### The Math (X-VLA)
*   **Total Frames**: Inspection of `lehome/dataset_challenge_merged` confirms exactly **265,798 frames** (1,000 episodes total). 
*   **Epoch Calculation:** `Steps per Epoch = 265,798 / Batch Size`.

If you aim for **1 full epoch** of X-VLA fine-tuning:
*   **A100 (BS=32):** `266k / 32 = ~8,312 steps`.
*   **L40S (BS=20):** `266k / 20 = ~13,300 steps`.
*   **L4 (BS=6):** `266k / 6 = ~44,333 steps`.

### AMP-Corrected Epoch & Cost Analysis (Merged Dataset: ~266k frames)

| GPU | AMP Batch Size | Steps for 1 Epoch | Steps/hr | Est. Time | Hourly Rate | **True Total Cost** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A100** | 32 | 8,312 | ~3,500 | **~2.4 hrs** | $1.87 | 🏆 **$4.49** |
| **L40S** | 20 | 13,300 | ~2,500 | **~5.3 hrs** | $1.86 | 🥈 **$9.86** |
| **L4** | 6 | 44,333 | ~600 | ~73.9 hrs | $0.80 | ❌ **$59.12** |

---

### DP VRAM Budget (271M params, 3x Cameras at 480x640)

> [!WARNING]
> **VRAM CORRECTION:** Real-world testing proved activation memory per sample is much higher than theoretical estimates (~900MB per sample even with AMP) due to the triple 480x640 high-res image inputs pushing through the ResNet backbone.

| Config | Fixed Overhead | Activation/sample | A4000 Max BS | L4 Max BS | L40S Max BS |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **No AMP (FP32)** | ~1.0 GB | ~1.5 GB | **8** | ~14 | ~30 |
| **AMP (BF16)** | ~1.0 GB | ~0.9 GB | **12–14** | ~24 | ~50 |

### Epoch Math (Merged Dataset: ~266k frames, Target: 3 epochs)

| GPU | AMP | Batch Size | Steps/Epoch | Steps (3 Epochs) | Steps/hr | Time | $/hr | **Total Cost** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A4000** | ✅ | **12** | 22,150 | 66,450 | ~3,100 | **~21.4 hrs** | $0.33 | 🏆 **$7.06** |
| **A4000** | ❌ | 8 | 33,225 | 99,675 | ~6,800 | ~14.7 hrs | $0.33 | $4.85 |
| **L4** | ✅ | 24 | 11,075 | 33,225 | ~2,500 | ~13.2 hrs | $0.80 | $10.56 |
| **L40S** | ✅ | 48 | 5,537 | 16,612 | ~2,000 | ~8.3 hrs | $1.86 | $15.43 |

> [!NOTE]
> Weirdly, running the A4000 at BS=8 **without AMP** was measured to be faster per-step than BS=12 with AMP (6800 steps/hr vs 3100 steps/hr limit). This indicates the CPU dataloader gets completely choked decoding the extra 4 samples per batch, wiping out any GPU compute gains from mixed precision.

### DP Recommendation:
- **Single trial:** 1x A4000 (No AMP, BS=8). Cost: **~$4.85** for 3 full epochs on merged data. It remains the value king due to the $0.33/hr price.
- **Hyperparameter sweep (4 configs):** 4x A4000 in parallel. Cost: **~$19.40 total**, done in ~14.7 hours.
