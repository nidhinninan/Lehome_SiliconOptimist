# DeepWiki MCP Usage Rules

Use the `deepwiki` MCP tools to gather deep understanding, code snippets, and implementation details for external repositories.

## Known Repositories (Current Catalog)
Use these rules whenever the following repositories or related libraries are mentioned:
- **LeRobot** (huggingface/lerobot)
- **IsaacLab** (isaac-sim/IsaacLab)
- **IsaacSim** (isaac-sim/IsaacSim)

## Source of Truth for Mappings
**CRITICAL**: While the list above provides common triggers, always consult [DeepWiki_servers.md](file:///home/nidhinninan/Documents/Research/Hackathon/LeHome-trial/DeepWiki_servers.md) for the authoritative list and actual repository identifiers. If you encounter an external robotics/simulation repository not listed above, check this file to see if it has been added.

### How to Parse Mapping File
The mapping file follows the format `Friendly Name : URL`.
1. Read the line for the project you are investigating.
2. Extract the `Owner/Repo` identifier from the end of the URL (e.g., `https://deepwiki.com/huggingface/lerobot` -> `huggingface/lerobot`).
3. Use the extracted string as the `repoName` for all `deepwiki` tools.

## Guidelines for Use

### 1. Proactive Research & Implementation
Before writing or modifying code that interacts with these libraries:
- **Gather Code Snippets**: Search for reference implementations of specific functions or classes.
- **Verify Parameters**: Use `ask_question` to determine exact parameter names, types, and constraints (e.g., "What are the required parameters for the DiffusionPolicy constructor?").
- **Understand Constraints**: Identify hardware requirements, environment variables, or configuration dependencies.

### 2. Hallucination Prevention
To ensure 100% accuracy and avoid "guessing" API signatures:
- **Cite Sources**: Explicitly mention if information or snippets were retrieved via DeepWiki.
- **Cross-Reference**: If you are unsure about a function's behavior, use `read_wiki_contents` rather than making assumptions.
- **Reference Snippets**: Use actual code snippets from the wiki as the primary template for your work.

### 3. Tool Selection
- **Overview**: `read_wiki_structure` to browse topics.
- **Quick Reference**: `ask_question` for parameters, logic, or examples.
- **Deep Dive**: `read_wiki_contents` for full context on a module.
