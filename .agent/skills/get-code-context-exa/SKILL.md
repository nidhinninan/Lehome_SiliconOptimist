---
name: get-code-context-exa
description: Code context using Exa. Finds real snippets and docs from GitHub, StackOverflow, and technical docs. Use when searching for code examples, API syntax, library documentation, or debugging help.
context: fork
---

# Code Context Role (Exa Base)

## Overview
This skill implements a specialized "Code Context" role that leverages the `get_code_context_exa` tool from your base Exa MCP server. By using this skill, the assistant applies optimized query patterns and token management to retrieve high-signal technical documentation.

## Tool Utilization
- **Primary Tool**: `get_code_context_exa` (available via the base `exa` MCP server).
- **Restriction**: When this skill is active, prioritize using this tool over generic web searches for technical queries.

## Token Isolation (Critical)
Never run Exa in main context. Always spawn Task agents to ensure the main conversation stays clean:
- Agent calls `get_code_context_exa`
- Agent extracts the minimum viable snippet(s) + constraints
- Agent deduplicates near-identical results (mirrors, forks, repeated StackOverflow answers) before presenting
- Agent returns copyable snippets + brief explanation
- Main context stays clean regardless of search volume

## When to Use
Use this tool for ANY programming-related request:
- API usage and syntax
- SDK/library examples
- config and setup patterns
- framework "how to" questions
- debugging when you need authoritative snippets

## Inputs (Supported)
`get_code_context_exa` supports:
- `query` (string, required)
- `tokensNum` (number, optional; default ~5000; typical range 1000–50000)

## Query Writing Patterns (High Signal)
To reduce noise, the assistant must structure queries as follows:
- **Include Language**: e.g., "Go generics" instead of "generics".
- **Include Context**: Include **framework + version**. e.g., "Next.js 14 routing" or "Python 3.12 syntax".
- **Exact Identifiers**: Use exact function names, error messages, or config keys if available.

## Dynamic Tuning for `tokensNum`
- **Focused snippet**: 1000–3000 tokens.
- **Balanced research**: 5000 tokens (default).
- **Complex integration**: 10000–20000 tokens.
- Only go larger when necessary to avoid dumping excessive context.

## Output Structure
Return information in this order:
1) **Minimal Working Snippets**: Copy-paste friendly code.
2) **Technical Constraints**: Versions, gotchas, or dependencies.
3) **Sources**: URLs for further reading.

---
*Note: This skill utilizes the standard `exa` server configured in your project settings.*
