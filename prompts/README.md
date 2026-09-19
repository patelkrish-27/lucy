# Lucy prompts

These files contain the model instructions used by Lucy's runtime. Keep prompt text here so it can be reviewed and tuned without editing Rust logic.

- `triage.md` — main-model request triage and task decomposition.
- `controller.md` — main-model closed-loop verification, continuation and recovery.
- `tool_selector.md` — main-model one-subtask-to-one-tool selector.

They are embedded into the `lucy-runtime` binary at compile time with Rust `include_str!`.
