# LeHome Workspace Tracking Rule

All edits to files within the `lehome_workspace/` directory MUST be mirrored in the central change log to ensure work is transportable and grounded for downstream LLM use.

## Required Actions

1. **Mirrored Logging**: Every time a file in `lehome_workspace/` is created or modified, you MUST record the change in [lehome_change_log.md](file:///home/nidhinninan/Documents/Research/Hackathon/LeHome-trial/lehome_workspace/lehome_change_log.md).
2. **Log Format**:
    - **Date/Time**: Timestamp of the change.
    - **File Path**: The absolute or relative path within the workspace.
    - **Description**: A short summary of why the change was made.
    - **Code Snippet/Diff**: A code block containing the new file content (for new files) or a clear diff (for modified files).
3. **One-Shotedness**: The log must be detailed enough that a fresh LLM instance could reconstruct the modified state of the repository solely from this document.
4. **Tool Use**: Perform this logging as the LAST step of any task that modifies the workspace.
