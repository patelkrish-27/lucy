# jcode → Lucy migration map

Lucy is taking the useful runtime primitives from jcode while keeping its own product architecture.

1. `jcode-agent-runtime` → Lucy interrupt/cancellation semantics in `lucy-core`.
2. `jcode-tool-core` → Lucy's `Tool`, `ToolContext`, tool schema contracts, and execution loop.
3. `jcode-command-risk` → Lucy's deterministic shell-risk classifier and reflection/deny gate.
4. `jcode-plan` / `jcode-task-types` → next stage: general task graph and subtask execution.
5. `jcode-session-types` / `jcode-storage` → next stage: persistent sessions and memory.
6. `jcode-protocol` / `jcode-transport` → next stage: real MCP transport and remote tools.
7. `jcode-tui-*` → selectively port UI primitives while retaining Lucy's UX.
8. provider crates → Lucy provider adapters, so the agent remains model-agnostic.

## Phase 1 complete

The first runtime pass now has:

- structured model turns with tool calls
- bounded agent/tool loop
- cancellation checks before and during shell execution
- shell timeout and bounded stdout/stderr capture
- deterministic destructive-command assessment
- absolute denial for protected system/credential paths
- a reflection path for ambiguous destructive commands
- GitHub Actions validation on every push/PR

## Upstream reference

The safety design is informed by `1jehuang/jcode` and is licensed under its MIT terms. See `THIRD_PARTY.md`.
