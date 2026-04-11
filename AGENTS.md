# LeHome trial — project context for agents

## Cursor rules

Policies live in [`.cursor/rules/`](.cursor/rules/). Each file is an `.mdc` with YAML frontmatter (`alwaysApply` and/or `globs`). Prefer editing those files rather than duplicating long policy text here.

## Skills

Workflow skills live under [`.cursor/skills/`](.cursor/skills/). A parallel tree exists at [`.agent/skills/`](.agent/skills/) for another IDE; when you change a skill, update both copies so they stay aligned.

## Authoritative references

- [`DeepWiki_servers.md`](DeepWiki_servers.md) — catalog of DeepWiki `Owner/Repo` identifiers.
- [`lehome_workspace/lehome_change_log.md`](lehome_workspace/lehome_change_log.md) — log every create/modify under `lehome_workspace/` (see the workspace-tracking project rule).
- [`.cursor/plans/`](.cursor/plans/) — active implementation plans generated in Cursor.
- [`Artifacts/`](Artifacts/) — research notes, training readouts, and analysis.
