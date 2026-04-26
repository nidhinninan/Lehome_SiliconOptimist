**Implementation Plan: DINOv2 Spatial Pooling Adaptation (MAP Head + Registers Variant)**

This plan is directly grounded in the audit synthesis from `@Artifacts/Vision-Backbone/dinov2_spatial_pooling_audit_thread_2026-04-25.md` (the primary source). It adopts the **recommended Option 1 (MAP head)** as the primary path because:

- Voltron (RSS 2023) provides the strongest controlled evidence for frozen-ViT visuomotor control: Multi-headed Attention Pooling (MAP) nearly doubled success rates vs. naive CLS/mean pooling.
- It avoids the concrete risks of raw SpatialSoftmax on frozen ViT patch tokens (register artifacts per Darcet et al. ICLR'24/arXiv:2309.16588, late-layer global mixing, extreme bottleneck on frozen encoder, coarse 16×16 grid for cloth wrinkles).
- It stays low-parameter (~0.6–1.5M trainable params for a 1-block cross-attention head) and compatible with the existing frozen-backbone BYOP pattern in `lehome_workspace/lerobot_policy_dino/`.
- Fallback to a "hardened" Option 2 (DPT-style multi-layer reassemble + conv + SpatialSoftmax on `dinov2-with-registers-*`, higher-res input like 252²→18×18 grid) is noted if keypoint inductive bias is strictly required.

The plan is **executable by a lower-code-generation model** (or human): every step includes exact file paths, line references (from current `modeling_dino_diffusion.py`/`configuration_dino_diffusion.py`), concrete code snippets to insert/replace, parameter choices, test commands, and VM handoff instructions. It respects LeRobot v0.4.3 BYOP mechanics (from DeepWiki MCP on `huggingface/lerobot` — cited below), the existing `DinoDiffusionModel`/`DinoDiffusionPolicy` subclassing pattern, `lerobot-train` integration, and LeHome workspace tracking rules.

All changes stay inside the existing `lerobot_policy_dino` BYOP package (Path A from `lehome-lerobot-custom-policy/SKILL.md`). No new packages or eval-only `BasePolicy` adapters unless requested. Training/eval verification **must occur on the remote VM** (per `.cursor/rules/lehome-environment-split.mdc` — never run torch/lerobot locally).

### Grounding Sources Used (per requested skills/MCPs)
- **@.cursor/skills/get-code-context-exa/SKILL.md & @mcp:exa (semantic/code context)**: Prioritized for LeRobot/Voltron/Octo snippets on MAP/TokenLearner, SpatialSoftmax impls, ViT patch reshaping, and DINOv2 registers slicing. Queries followed the skill's patterns (language/framework/version/exact identifiers like "DiffusionPolicy subclassing v0.4.3", "MAP head Voltron robotics", "Dinov2Model last_hidden_state registers slice").
- **@mcp:deepwiki (huggingface/lerobot)**: Used `ask_question` and `read_wiki_structure` (schemas checked via descriptors first). Key insights: `@PreTrainedConfig.register_subclass`, factory discovery (`_get_policy_cls_from_policy_name` via config `__module__` swap `configuration_*` → `modeling_*`), `validate_features`, `get_optim_params` to exclude frozen backbone, `DiffusionConditionalUnet1d` global_cond expectations, and BYOP package structure (`lerobot_policy_*`). See DeepWiki outputs for exact registry/processor patterns. (Repo mapping from `DeepWiki_servers.md` and `.cursor/rules/deepwiki-mcp.mdc`.)
- **@user-BrightData / RivalSearchMCP signals** (via audit): Confirmed no direct "SpatialSoftmax + frozen DINOv2 + DiffusionPolicy" precedent; strong preference for registers variant + learned pooling (MAP/TokenLearner) over naive softmax; Bluesky hits on DINOv3 fine-tuning gains and dense patch features for manipulation.
- **Local context**: Full read of `modeling_dino_diffusion.py` (current CLS/mean-pool at L106-109, `_encode_images` L89-109, `DinoDiffusionModel` L43+, `single_step_dim` calc L59-62), `configuration_dino_diffusion.py`, Grep results across Artifacts/ + lehome_workspace/, `lehome-lerobot-custom-policy/SKILL.md`, prior sweep logs, and `AGENTS.md`/rules.
- **Brainstorming/Skills alignment** (`brainstorming/SKILL.md`, `using-superpowers`, `lehome-lerobot-custom-policy`): Explored context first (files, git status, artifacts), proposed approaches with trade-offs (MAP vs. hardened SpatialSoftmax vs. full patch-sequence), focused on isolation (new `SpatialHead` nn.Module), YAGNI (minimal changes to existing BYOP), and reconstructability for VM transfer. (No implementation until plan approval; this document serves as the executable spec.)

**Trade-off Summary** (per brainstorming skill):
1. **MAP Head (recommended, primary)**: Best evidence, implicit geometry via attention, robust to registers. ~1M params. Trade-off: Slightly higher compute than pure SpatialSoftmax.
2. **Hardened SpatialSoftmax (Option 2 fallback)**: Matches original proposal but with DPT multi-layer, registers slice (`[:, 1+4:,:]`), conv upscale to 32×32, higher input res. Trade-off: Risk of artifact locking if not careful.
3. **Full patch sequence (Option 3)**: No pooling, cross-attend in U-Net. Trade-off: Higher VRAM/sequence length (256 tokens × N_cams).

**Success Criteria** (measurable on VM):
- Training runs without OOM/crashes (10k-step micro-sweep or full).
- `get_optim_params` correctly excludes backbone (frozen).
- Global conditioning dim updates correctly; policy converges faster/better than current CLS baseline on cloth-folding metric (per prior sweeps).
- Registers slicing verified (no high-norm artifact peaks in feature maps).
- Logs added to `lehome_workspace/lehome_change_log.md` + `vm_transfer_list.md` (last step).

### Detailed Step-by-Step Implementation

**Phase 0: Prep (Local)**
1. Update default in `configuration_dino_diffusion.py`:
   ```python
   # Add after existing fields (around L21)
   use_registers: bool = True
   num_register_tokens: int = 4
   spatial_head_type: Literal["map", "spatial_softmax", "none"] = "map"  # from typing import Literal
   map_num_queries: int = 8
   spatial_softmax_num_keypoints: int = 32  # for fallback; produces ~64-dim output
   ```
   - Override `__post_init__` to validate new fields (e.g., if `spatial_head_type == "map"` require `map_num_queries > 0`).
   - Update `vision_backbone` default to `"facebook/dinov2-with-registers-small"` (or add logic to map "dinov2-small" → registers variant).
   - Bump `feature_dim` logic or add `head_output_dim` property used in `single_step_dim` calc.

2. Add to `DinoDiffusionConfig.validate_features()` (extend existing).

**Phase 1: Core Changes to modeling_dino_diffusion.py (Main File)**
- **Add new `SpatialHead` module** (new class after imports, before `DinoDiffusionModel` — ~50-80 lines). Cite Voltron/Octo patterns from Exa/DeepWiki.
  - For **MAP** (primary):
    ```python
    class MAPHead(nn.Module):
        """Multi-headed Attention Pooling (Voltron-style) for frozen ViT patches.
        Learns K query vectors that cross-attend to patch tokens.
        Output: (B, K * hidden) or projected fixed dim.
        """
        def __init__(self, hidden_dim: int = 384, num_queries: int = 8, num_heads: int = 8, dropout: float = 0.1):
            super().__init__()
            self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim))  # Learnable queries
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            self.norm = nn.LayerNorm(hidden_dim)
            self.proj = nn.Linear(hidden_dim, hidden_dim)  # Optional projection

        def forward(self, patch_tokens: Tensor) -> Tensor:  # patch_tokens: (B, num_patches, D)
            B = patch_tokens.shape[0]
            queries = self.queries.unsqueeze(0).expand(B, -1, -1)
            attn_output, _ = self.cross_attn(queries, patch_tokens, patch_tokens)
            pooled = self.norm(attn_output + queries)  # Residual
            return self.proj(pooled.mean(dim=1))  # or flatten/concat queries → (B, pooled_dim)
    ```
  - For SpatialSoftmax fallback: Use `SpatialSoftmax` from `lerobot.policies.diffusion` or robomimic-style (reshape `(B, D, H, W)`, `Conv2d(D, K*2, 1)`, softmax over spatial, expected coords). Add DPT-like multi-layer tap if time (sample layers 3/6/9/12 via `outputs.hidden_states` if available in `Dinov2Model`).

- **Update `DinoDiffusionModel.__init__`** (after backbone L51-56):
  ```python
  self.use_registers = config.use_registers
  self.num_register_tokens = config.num_register_tokens
  self.spatial_head_type = config.spatial_head_type
  if self.spatial_head_type == "map":
      self.spatial_head = MAPHead(
          hidden_dim=self.feature_dim,
          num_queries=config.map_num_queries
      )
      self.head_output_dim = config.map_num_queries * self.feature_dim  # or projected dim
  elif self.spatial_head_type == "spatial_softmax":
      # ... similar, output_dim = config.spatial_softmax_num_keypoints * 2
      pass
  else:
      self.head_output_dim = self.feature_dim
  # Update single_step_dim to use self.head_output_dim instead of raw feature_dim
  single_step_dim = config.robot_state_feature.shape[0] + self.head_output_dim * num_images
  ```

- **Refactor `_encode_images`** (replace L89-109 entirely — critical for registers + reshape):
  ```python
  def _encode_images(self, img: Tensor) -> Tensor:
      """Updated per audit: registers slicing, patch reshape + MAP/Spatial head.
      Drops CLS + registers, applies spatial readout. Returns (B*N, head_output_dim).
      """
      if img.shape[-2:] != (_DINO_INPUT_SIZE, _DINO_INPUT_SIZE):
          img = F.interpolate(img, size=(_DINO_INPUT_SIZE, _DINO_INPUT_SIZE), mode="bilinear", antialias=True)
      with torch.no_grad():
          outputs = self.backbone(img)
          hidden = outputs.last_hidden_state  # (B*N, 1 + registers + patches, D)
          if self.use_registers and self.num_register_tokens > 0:
              hidden = hidden[:, 1 + self.num_register_tokens :, :]  # Drop CLS + registers
          else:
              hidden = hidden[:, 1:, :]  # Drop CLS only
          if self.spatial_head_type != "none" and hasattr(self, 'spatial_head'):
              return self.spatial_head(hidden)  # (B*N, head_output_dim)
          return hidden.mean(dim=1)  # fallback
  ```
  - Add comment citing audit (lines 114-116 in sub-agent output) and DINOv2 HF docs for reshape (`unflatten` alternative if keeping 2D map).

- **Update `_prepare_global_conditioning`** and `single_step_dim` calc to use `head_output_dim` (L59-62, L123-124). Ensure `view(B, N, -1)` works with new dim.
- **Update `get_optim_params`** in `DinoDiffusionPolicy` (already filters `requires_grad`; ensure head params are trainable).
- Add `reset()` if stateful (deque for MAP if needed).

**Phase 2: Config, Training, and Integration**
- Update YAML configs in `lehome_workspace/` (e.g., `sweep_dino.yaml`, `train_dp.yaml` equivalents): set `policy.type: dino_diffusion`, new fields like `spatial_head_type: map`, `vision_backbone: "facebook/dinov2-with-registers-small"`.
- Add `processor_dino_diffusion.py` if new preprocessing needed for higher-res or feature mapping (per skill; currently minimal).
- Bump input size optionally (`_DINO_INPUT_SIZE = 252`) for better grid (18×18); update PE interpolation note (audit commit e1277af).
- Update `lerobot_train_with_plugins.py` / `setup_lehome.sh` / `run_10k_sweep.sh` if defaults change.
- Ensure `get_optim_params` and normalization exclude backbone (already mostly there).

**Phase 3: Testing & Verification (VM-only)**
- Local: `python -m pytest` or simple import test (no torch run).
- Transfer: Add updated files to `lehome_workspace/vm_transfer_list.md` (with `(Updated: $(date))`).
- On VM: Run modified `run_10k_sweep.sh` or `lerobot-train` with new config. Compare loss/ success vs. baseline CLS run. Check feature maps for artifact-free spatial peaks. Use `parallel_eval.sh` for LeHome eval.
- Log results to Artifacts/ (e.g., new sweep readout). Monitor VRAM (MAP is cheap).

**Phase 4: Logging (LAST step — per lehome-workspace-tracking.mdc)**
- Append detailed entry to `lehome_workspace/lehome_change_log.md` (timestamp, paths, diff snippets of key changes like `_encode_images` and MAPHead, reconstruction notes).
- Update `vm_transfer_list.md` (global Last Updated + per-file tags).
- Commit with descriptive message (use git via Shell if needed, but only after user approval).

**Risks & Rollback**
- Registers slicing off-by-one → scrambled grid (test with small batch, inspect shapes).
- Dim mismatch in U-Net global_cond → crash (update `single_step_dim` first).
- VRAM spike on multi-cam → start with 1 camera, lower `map_num_queries`.
- If MAP underperforms, fallback to hardened SpatialSoftmax (add DPT reassemble from HF `DPT` or Depth-Anything patterns via Exa).

**Next Steps (per brainstorming skill)**
This plan is the "design doc" equivalent. Review it. If approved, I will:
1. Write it to `Artifacts/Vision-Backbone/dinov2_spatial_pooling_implementation_plan.md` (with full code templates).
2. Implement via targeted edits (Read → StrReplace/Write, then ReadLints).
3. Log changes + transfer list.
4. Instruct VM run + report back (losses, eval metrics).

Does this plan look correct? Any changes to chosen head (MAP vs. SpatialSoftmax), parameters, scope (e.g., multi-layer DPT, higher res, full patch cross-attn in UNet), or success metrics? Should we prioritize a specific LeHome garment/task? (One question at a time per skill.)

Once approved, I'll execute (or invoke writing-plans equivalent if a dedicated skill file is located). All citations from DeepWiki/Exa/audit will be preserved in code comments and the plan doc.

## Flattening queries to preserve strong spatial knowledge

Based on the context of robotic manipulation and Diffusion Policies (DP) using frozen ViT backbones, **flattening/concatenating the queries into one long vector is the better approach** for preserving strong spatial knowledge.

Here is the reasoning:

1. **Preserving Spatial Distinctness:** The entire point of using Multi-headed Attention Pooling (MAP) over a simple CLS token or global average pool is to allow the network to extract multiple *distinct* spatial features (e.g., one query might learn to attend to the gripper, another to a cloth fold, another to an object boundary). If you mean-pool those $K$ queries back down to a single 384-dimensional vector at the end, you are forcing all those distinct geometric concepts to collapse back into a single global summary. 
2. **Diffusion Policy's Global Conditioning:** The standard Diffusion Policy U-Net is designed to take a relatively large, flat `global_cond` vector. It does not have a strict bottleneck requirement on the vision side. Flattening $K=8$ queries of dimension 384 results in a 3072-dimensional vector per camera. This is well within the normal operating range for DP (which often concatenates features from multiple cameras and proprioception anyway).
3. **Robotics Precedents (Voltron & Octo):** In architectures like Octo (which uses TokenLearner/MAP variants) and Voltron, the resulting $K$ tokens are typically preserved as a sequence (if feeding into a transformer) or flattened (if feeding into an MLP/U-Net bottleneck). They are *not* mean-pooled together, because doing so destroys the multi-object/multi-part resolution the head just worked to extract.

**Conclusion:**
Choose **flattening all queries** (`output_dim = map_num_queries * hidden_size`). It retains the high-capacity, spatially-resolved features that the U-Net needs to precisely localize the robot and the deformable objects (cloth) in the scene.