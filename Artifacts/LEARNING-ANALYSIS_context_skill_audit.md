# Audit: DeepWiki-Only vs DeepWiki + Exa Context Skill

## Methodology
I am comparing two versions of the implementation plan produced during this conversation:
- **V1 (DeepWiki-only)**: The plan created after the first turn, which used `mcp_deepwiki_ask_question` (which **failed** — repo not indexed), `search_web` (generic), and local file inspection.
- **V2 (DeepWiki + Exa)**: The plan after augmenting with `mcp_exa_web_search_exa` (the code-context skill) and a successful DeepWiki query against `huggingface/lerobot`.

I evaluate across 5 dimensions relevant to "one-shotness" — i.e., whether a developer could take this plan, implement it without additional research, and have it work on the first try.

---

## Scoring Rubric

| Dimension | Description | Scale |
|---|---|---|
| **API Accuracy** | Are the class names, import paths, and constructor signatures correct? | 1–5 |
| **Completeness** | Does it cover all the moving parts (features dict, create, add_frame, transforms, gotchas)? | 1–5 |
| **Copy-Paste Readiness** | Can the code snippets be dropped into a real project and run? | 1–5 |
| **Gotcha Coverage** | Does it warn about silent failure modes (geometric sync, temporal consistency, naming conventions)? | 1–5 |
| **Provenance / Traceability** | Are claims backed by specific source files, PRs, or class names that can be verified? | 1–5 |

---

## V1: DeepWiki-Only (+ generic web search)

### What happened
- DeepWiki call **failed** (`lerobot/lerobot` was not indexed; should have been `huggingface/lerobot`).
- Fell back to `search_web`, which returned high-level summaries with no code.
- Local file inspection of `dataset_record.py` provided the LeHome-specific feature dict pattern.

### Audit

| Dimension | Score | Notes |
|---|---|---|
| **API Accuracy** | 3/5 | Feature dict structure was correct (copied from local `dataset_record.py`). But augmentation section was vague — said "set `image_transforms.enable` to `True` in the training configuration" without specifying *which class* or *which constructor parameter*. No import paths for transforms. |
| **Completeness** | 3/5 | Covered features, `LeRobotDataset.create`, `add_frame`, and `save_episode`. Missing: the `ImageTransforms`/`ImageTransformsConfig` initialization, `RandomSubsetApply`, and geometric consistency concerns. |
| **Copy-Paste Readiness** | 2/5 | Feature definition and recording loop snippets were usable. But the augmentation section had **no runnable code** — just a prose description of "enable it in config". A developer would need to go research the actual API. |
| **Gotcha Coverage** | 2/5 | Mentioned "ensure geometric transforms are consistent across cameras" as a generic warning. No mention of proprioception negation, `SharpnessJitter`, or `RandomSubsetApply`. |
| **Provenance** | 2/5 | No specific file paths, PRs, or class names cited for the augmentation pipeline. The local `dataset_record.py` was cited implicitly but not linked. |
| **Total** | **12/25** | |

---

## V2: DeepWiki + Exa Context Skill

### What happened
- `mcp_exa_web_search_exa` returned **8 high-signal results** including:
  - PR #234 (the original augmentation PR by `@marinabar`)
  - `src/lerobot/datasets/factory.py` (the `make_dataset` function showing `ImageTransforms` instantiation)
  - `src/lerobot/datasets/lerobot_dataset.py` (the `__init__` signature showing `image_transforms: Callable | None`)
  - `twarner/lerobot-augmented` dataset card (showing geometric flip + proprioception negation strategy)
  - Official LeRobotDataset v3.0 docs with `ImageTransformsConfig` examples
- DeepWiki query against `huggingface/lerobot` **succeeded** and confirmed `RandomSubsetApply` usage.

### Audit

| Dimension | Score | Notes |
|---|---|---|
| **API Accuracy** | 5/5 | Correct import paths (`from lerobot.datasets.transforms import ImageTransforms, ImageTransformsConfig, RandomSubsetApply`). Correct constructor signature (`image_transforms=transforms`). Correct config dataclass fields (`enable`, `max_num_transforms`, `random_order`). |
| **Completeness** | 5/5 | Covers all 6 stages: feature definition → dataset creation → recording loop → training-time augmentation → advanced custom transforms → geometric consistency. |
| **Copy-Paste Readiness** | 4/5 | All code snippets have correct imports and are structurally complete. Docked 1 point because `ImageTransformsConfig.tfs` dict (the per-transform config) is not shown in the main example — it's only referenced in the `RandomSubsetApply` section. |
| **Gotcha Coverage** | 4/5 | Explicitly warns about: (1) `observation.images.` prefix requirement, (2) geometric flip → proprioception negation, (3) offline augmentation strategy. Missing: no explicit warning about temporal consistency with `delta_timestamps` (same augmentation must apply across the temporal window). |
| **Provenance** | 5/5 | Cites PR #234, `twarner/lerobot-augmented` dataset, specific source files (`factory.py`, `lerobot_dataset.py`), and the `torchvision.transforms.v2` dependency. |
| **Total** | **23/25** | |

---

## Summary

| Metric | V1 (DeepWiki-only) | V2 (DeepWiki + Exa) | Delta |
|---|---|---|---|
| API Accuracy | 3 | 5 | **+2** |
| Completeness | 3 | 5 | **+2** |
| Copy-Paste Readiness | 2 | 4 | **+2** |
| Gotcha Coverage | 2 | 4 | **+2** |
| Provenance | 2 | 5 | **+3** |
| **Total** | **12/25** | **23/25** | **+11 (+92%)** |

## Key Findings

> [!IMPORTANT]
> The Exa code-context skill was the **primary driver** of improvement, not DeepWiki. The initial DeepWiki call failed entirely (wrong repo path), and even the successful retry only confirmed what Exa had already provided. Exa's value was in surfacing **actual source code** (`factory.py`, `lerobot_dataset.py`) and **real-world usage patterns** (PR #234, `twarner/lerobot-augmented`).

### What Exa added that DeepWiki couldn't:
1. **PR #234 context**: The original augmentation implementation PR, including the `RandomSubsetApply` and `SharpnessJitter` classes.
2. **Source file snippets**: Actual constructor signatures from `lerobot_dataset.py` and the `make_dataset` factory function.
3. **Community patterns**: The `twarner/lerobot-augmented` dataset showed a real-world geometric augmentation strategy (flip + negate proprioception).

### What DeepWiki added that Exa couldn't:
1. **Structured API overview**: Clean summary of `ImageTransformsConfig` fields and `RandomSubsetApply` parameters.
2. **Wiki navigation**: Pointers to related documentation pages for deeper exploration.

### One-Shotness Verdict
- **V1**: A developer would need **2–3 additional research cycles** to find the correct transform classes, import paths, and gotchas.
- **V2**: A developer could implement the full pipeline with **0–1 additional lookups** (the missing `delta_timestamps` temporal consistency warning being the only gap).

> [!TIP]
> **Recommendation for the skill**: The `get-code-context-exa` skill should be triggered **before** or **in parallel with** DeepWiki for technical implementation questions. Exa surfaces source-level detail that DeepWiki's summarization layer often abstracts away. The optimal workflow is: **Exa for code artifacts → DeepWiki for architectural context → Local codebase for project-specific patterns**.
