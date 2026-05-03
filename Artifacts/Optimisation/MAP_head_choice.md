For the **complete merged dataset** and a **longer 150k–200k run**, I would choose **`map_num_queries: 8`** as the main long-term setting, not 4.

The balance is: **4 is the safer compression setting**, but for your case it may be *too* tight because the policy must absorb **4 garment families**, many subvariants, **3 camera views**, and bimanual cloth geometry into one shared diffusion policy. At 150k–200k steps, the model has more time to learn specialization among the MAP slots. With only 4 queries, you risk under-conditioning the U-Net: the model may get a smoother/lower-risk representation, but lose fine distinctions like sleeve vs torso, left/right cloth boundary, gripper contact region, and category-specific geometry.

I would not call 8 “high” here. For this architecture, **8 is the balanced setting**. It gives each image 8 learned spatial summaries, so across 3 cameras the policy gets enough visual slots without jumping into the more risky range like **12 or 16**, where I’d be more worried about overfitting, memory pressure, and unused capacity.

So my recommendation:

```yaml
policy:
  map_num_queries: 8
```

Use **4** only if one of these happens:

- `K=8` is too slow or too close to OOM on the VM.
- Training loss improves but eval/sim behavior is clearly worse, suggesting overfit or brittle visual conditioning.
- You want a deliberately regularized baseline to compare against 8.

But if you can afford it, I’d train the serious merged-dataset run at **8 queries**, then treat **4** as the ablation. The theory and the dataset shape both point to **8 as the better long-term bet**, while **4 is the conservative fallback**. I would avoid going above 8 until you have actual behavior metrics proving that 8 is capacity-limited.

## `map_num_queries`: higher vs lower

In your setup, **`map_num_queries`** is **`K`** in the MAP head: **`K` learnable queries** each cross-attend the DINO patch tokens; the head output is **`K × 384`** per image (then ×3 cameras), and that vector is what conditions the diffusion U-Net.

```96:101:lehome_workspace/lerobot_policy_dino/src/lerobot_policy_dino/modeling_dino_diffusion.py
        if self.spatial_pooling == "map":
            self.spatial_head = MAPHead(
                hidden_dim=self.feature_dim,
                num_queries=self.map_num_queries,
            )
            self.head_output_dim = self.map_num_queries * self.feature_dim
```

So:

| | **Lower `map_num_queries`** (e.g. 4) | **Higher `map_num_queries`** (e.g. 8, 12, 16) |
|---|--------------------------------------|-----------------------------------------------|
| **What the model can represent** | Fewer **independent** pooled “views” of the scene. Each query can still attend anywhere, but you have **fewer slots** to specialize (e.g. gripper, fold, boundary, background). | More slots → in principle **richer** conditioning: more ways to keep **distinct** spatial/semantic summaries before they are flattened into `global_cond`. |
| **U‑Net / optimization** | **Smaller** `global_cond` → often **easier** optimization, less risk of overfitting tiny quirks of the high‑dim cond vector. | **Larger** cond → **more capacity** for the policy to use, but also **harder** to learn without enough data; can overfit or underuse extra dims. |
| **Compute & memory** | **Less** MAP attention work; **much smaller** conditioning → **faster steps**, **lower VRAM**. | **More** MAP work + **much larger** U‑Net `global_cond_dim` → **slower**, **higher VRAM** (you already saw MAP near full 16 GB with `K=8`). |

**Capability change in one sentence:** you are not changing DINO’s patch grid; you are changing **how many compressed “summary tokens”** (queries) are allowed before the U‑Net. **More queries = more expressive bottleneck between ViT patches and the U‑Net; fewer queries = stronger compression**, possibly losing fine distinctions the extra queries would have captured.

## Practical guidance for you

- If the goal is **speed / VRAM** and the policy still does well in sim: try **lower** (e.g. **4** vs **8**) and compare **task metrics**, not only loss.  
- If the goal is **max representational headroom** for hard cloth geometry and you have headroom on VRAM and data: **higher** can help **in principle**, but past a point returns diminish and cost grows roughly **linearly in K** for that MAP block and **linearly in K** for conditioning size into the U‑Net.

**Voltron-style intuition** (from your artifacts): multiple queries exist so different queries can **specialize** on different parts of the scene; **too few** merges those roles; **too many** adds capacity you may not need and pays full price in **cond dim + compute**.

There is no universally “correct” K: **8 is a reasonable middle**; **4** is the usual next knob when **wall-clock or OOM** bites; **>8** only if you measure a clear **behavior** win and can afford the cost.