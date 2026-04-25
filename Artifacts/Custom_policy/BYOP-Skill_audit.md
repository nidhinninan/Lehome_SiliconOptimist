# BYOP Skill audit: `lehome-lerobot-custom-policy`

Audit of the `lehome-lerobot-custom-policy` skill against the standards in [`.cursor/skills/skill-creator/SKILL.md`](../../.cursor/skills/skill-creator/SKILL.md).

---

## 1. Progressive disclosure and structure

**Requirement:** Keep `SKILL.md` under 500 lines; use hierarchy for large reference material.

**Status:** Pass (strong).

The main `SKILL.md` is concise (~69 lines) and focuses on the high-level workflow. Version-specific verification is deferred to `reference.md`, which limits context bloat while keeping a traceable source of truth for LeRobot v0.4.3.

---

## 2. Triggering and description

**Requirement:** The description should state what the skill does and when to use it; descriptions can be “pushy” to reduce under-triggering.

**Status:** Pass.

The frontmatter description names concrete triggers (e.g. LeHome Challenge, custom policy, `lerobot-train`, `policy_type`, evaluation scripts) and states what the skill guides (BYOP training vs LeHome eval adapters).

---

## 3. Groundedness and granularity

**Requirement:** Concrete guidance and examples where helpful.

**Status:** Pass, with one improvement idea.

The skill correctly separates two paths: LeRobot BYOP (training) vs LeHome eval (`BasePolicy` / `PolicyRegistry`). It anchors claims to repo paths under `lehome_workspace/lehome-challenge/` and to official docs / tagged upstream.

**Suggestion:** For maximum “no-guess” scripting, add a small **Templates** section (skill-creator’s template pattern): minimal skeletons for BYOP (`configuration_*` / registration) and for eval (`BasePolicy` + `PolicyRegistry` + `__init__.py` import), copied or summarized from real files so signatures stay literal.

---

## 4. Domain organization

**Requirement:** If multiple frameworks or variants exist, organize by variant.

**Status:** Pass.

Path A (training / BYOP) and Path B (eval-only custom) are clearly separated, matching how the hackathon repo actually works.

---

## Questions answered

### Is pointing to other files a reasonable approach?

**Yes.** Skill-creator’s progressive disclosure model expects the main skill to stay short and to **point** to deeper material (docs in-repo, `reference.md`, upstream URLs) rather than duplicating entire files. That saves tokens and keeps the skill maintainable.

### Will the agent find the right code snippets?

It will **if pointers are specific enough.** The current skill points to concrete files (`docs/training.md`, `evaluation.py`, `lerobot_policy.py`, etc.), which is appropriate. For the most error-prone spots (decorators, factory naming, eval registration), optional **inline templates** reduce the chance the model improvises APIs without opening files.

---

## Recommendation

Add a **Templates** subsection to `SKILL.md` (or a sibling `examples.md` linked one level deep) with minimal, file-grounded skeletons for:

- BYOP: `@PreTrainedConfig.register_subclass(...)`, `*Config` / `*Policy` / `processor_*` naming aligned with LeRobot v0.4.3 factory behavior (see `reference.md`).
- Eval: subclass `BasePolicy`, `@PolicyRegistry.register(...)`, import in `scripts/eval_policy/__init__.py`.

This would raise granularity to match skill-creator’s “examples / template” guidance without replacing the rest of the progressive-disclosure design.
