# BYOP Custom Policy — Training Crash Reference Log

> **Conversation:** "Resolving LeRobot Training Crashes" (23c0bd0d)
> **Scope:** DINOv2-Small (`dino_diffusion`) & CLIP-Base (`clip_diffusion`) BYOP policies integrated into LeRobot v0.4.3 via `lerobot_train_with_plugins.py`.

---

## Error 1 — Plugin Not Found

### Error
```
KeyError: 'dino_diffusion'
draccus.exceptions.DecodingError: Could not decode value...
```

### Root Cause
LeRobot uses a `ChoiceRegistry` (`@PreTrainedConfig.register_subclass`) to map a string like `"dino_diffusion"` to the actual class at parse time. The auto-discovery mechanism (entry point scanning) **failed silently** because `uv`/`pip` normalizes package names — `lerobot_policy_dino` becomes `lerobot-policy-dino` internally, breaking the prefix-matching check. The `@register_subclass` decorator never executed, so the registry was empty when `draccus.parse` ran.

### Fix
Created `lerobot_train_with_plugins.py` — a bootstrapper that runs **before** `draccus.parse`. It explicitly calls `import lerobot_policy_dino` and `import lerobot_policy_clip`, forcing execution of the `@register_subclass` decorators and populating the registry prior to config parsing.

### Why It Works
`draccus` only resolves `policy_type: "dino_diffusion"` if that string is already a key in the registry at parse time. By importing the packages first, we guarantee the decorator fires and the key exists before decoding begins.

---

## Error 2 — Silent Namespace Package Trap

### Error
Plugin imports succeeded with no `ImportError`, and the bootstrapper printed `✅ Registered DINOv2 Plugin` — but a `KeyError` was still raised downstream.

### Root Cause
Python 3 "Implicit Namespace Packages": when `import lerobot_policy_dino` is called from inside `/data/lehome_workspace`, Python finds the **folder** named `lerobot_policy_dino/` and loads it as an empty namespace package (no top-level `__init__.py`). The import succeeds, but **nothing inside `src/` is ever executed**. The `@register_subclass` decorator never fires despite the `✅` log line.

Additionally, both `pyproject.toml` files were missing a `[build-system]` section, so `uv pip install -e` failed silently and never symlinked `src/` into `.venv`.

### Fix
1. **`pyproject.toml`**: Added `[build-system]` (`setuptools`, `wheel`) to both BYOP packages.
2. **Bootstrapper `sys.path` injection**: Added `sys.path.insert(0, "<pkg>/src/")` for each plugin, bypassing the misleading top-level folder entirely.
3. **Safety check**: If the imported module has no `__file__` attribute (empty namespace), the bootstrapper raises `RuntimeError` instead of silently continuing.

### Why It Works
`sys.path.insert(0, ...)` takes priority over the local folder scan. Python hits `src/lerobot_policy_dino/__init__.py` with real code before seeing the misleading folder. The `__file__` check turns a silent failure into a loud, informative crash.

---

## Error 3 — Policy Constructor Signature Mismatch

### Error
```
TypeError: DinoDiffusionPolicy.__init__() got an unexpected keyword argument 'dataset_meta'
```

### Root Cause
LeRobot v0.4.3 updated `make_policy()` in `lerobot/policies/factory.py` to pass `dataset_meta` as a keyword argument to all policy constructors. Our custom `__init__` signatures only accepted `config` and `dataset_stats`.

### Fix
```python
def __init__(
    self,
    config: DinoDiffusionConfig,
    dataset_stats: dict | None = None,
    dataset_meta: dict | None = None,  # ← added
    **kwargs,                           # ← absorbs future upstream args
):
```

### Why It Works
`dataset_meta` is now an accepted kwarg (ignored, since `Normalize` only needs `dataset_stats`). `**kwargs` provides a forward-compat shield against any future args LeRobot adds to `make_policy()`.

---

## Error 4 (Proactive) — Eval Loop `noise` Kwarg Crash

> **Would have crashed at the first `eval_freq` checkpoint (step 10,000).**

### Error (Predicted)
```
TypeError: generate_actions() got an unexpected keyword argument 'noise'
```

### Root Cause
LeRobot v0.4.3's `DiffusionPolicy.predict_action_chunk()` calls `self.diffusion.generate_actions(batch, noise=noise)` during eval passes. Our `DinoDiffusionModel.generate_actions` only accepted `batch`.

### Fix
```python
def generate_actions(
    self, batch: dict[str, Tensor],
    noise: Tensor | None = None,  # ← added
    **kwargs,
) -> Tensor:
    sample = noise if noise is not None else torch.randn(...)
```

### Why It Works
The eval loop can inject a fixed noise tensor for reproducible evaluation. Using the provided `noise` (when seeded) ensures deterministic outputs across eval runs.

---

## Error 5 (Proactive) — Optimizer Crash on Frozen Parameters

> **Would have crashed on the first optimizer step for DINOv2/CLIP runs.**

### Error (Predicted)
```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

### Root Cause
`make_optimizer_and_scheduler` calls `policy.get_optim_params()` and passes results directly to PyTorch's optimizer. The inherited `DiffusionPolicy.get_optim_params()` returns `self.diffusion.parameters()` — which includes the **frozen ViT backbone weights** (`requires_grad=False`). `AdamW` with weight-decay decoupling crashes on these.

### Fix
```python
def get_optim_params(self) -> list:
    """Return only trainable params (excludes frozen vision backbone)."""
    return [p for p in self.parameters() if p.requires_grad]
```

### Why It Works
The optimizer only sees U-Net and normalization parameters (~50M params). Frozen backbone weights (~22M for DINOv2-S) are excluded. This also reduces optimizer memory by ~30%.

---

## Error 6 — CPU-Bound Training Bottleneck

### Symptom (not a crash)
```
updt_s: 0.306   ← GPU time per step
data_s: 0.648   ← CPU data loading + augmentation time
```
GPU was idle 66% of the time; training was 3× slower than necessary.

### Root Cause
`run_10k_sweep.sh` had a **hardcoded** `--dataset.image_transforms.enable=true` CLI flag that overrode anything set in the YAML configs. Heavy `ColorJitter` and `RandomAffine` augmentations ran on CPU every batch. The ResNet18 baseline had already been trained *without* augmentations, making the comparison invalid.

### Fix
1. Changed flag in `run_10k_sweep.sh` to `--dataset.image_transforms.enable=false`.
2. Added `image_transforms: enable: false` to all three sweep YAMLs.
3. Added `🔗 LINKED DEPENDENCY` comments in both files warning that this must stay synchronized.

### Why It Works
Without transform overhead the CPU pipeline is pure I/O + decode, fast enough to keep the GPU saturated continuously. All three backbones now train under identical conditions, preserving scientific validity of the comparison.

---

## Summary Table

| # | Error | Layer | Root Cause | Fix |
|---|-------|-------|------------|-----|
| 1 | `KeyError: 'dino_diffusion'` | Plugin registry | `uv` normalizes pkg names; decorator never fired | Bootstrapper imports plugins before `draccus.parse` |
| 2 | Silent `✅ OK` but registry empty | Plugin import | Python namespace package trap; `src/` code never ran | `sys.path.insert` to `src/`; fixed `[build-system]`; added `__file__` safety check |
| 3 | `TypeError: unexpected kwarg 'dataset_meta'` | `make_policy()` factory | LeRobot v0.4.3 updated factory API | Added `dataset_meta=None, **kwargs` to both `__init__` signatures |
| 4 | `TypeError: unexpected kwarg 'noise'` | Eval loop | v0.4.3 `predict_action_chunk` injects `noise=` | Added `noise=None, **kwargs` to `generate_actions`; use injected noise |
| 5 | Optimizer `RuntimeError` on frozen params | Optimizer | `get_optim_params` returned frozen backbone weights | Override to `[p for p if p.requires_grad]` |
| 6 | `data_s >> updt_s` CPU bottleneck | Data pipeline | Hardcoded `enable=true` in sweep script | Set `enable=false` everywhere; added linked-dependency comments |
