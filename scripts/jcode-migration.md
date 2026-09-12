# jcode -> Lucy migration map

Upstream project: https://github.com/1jehuang/jcode

Recommended migration order:

1. `jcode-agent-runtime` -> harden `lucy-core::InterruptSignal` and `lucy-agent` turn lifecycle.
2. `jcode-tool-core` / `jcode-tool-types` -> expand `lucy-tools` with permissions, output limits, stdin, and tool contexts.
3. `jcode-protocol` / `jcode-transport` -> implement `lucy-mcp` protocol + transport.
4. `jcode-plan` / `jcode-task-types` -> add Lucy's general task graph and subtask execution.
5. `jcode-session-types` / `jcode-storage` -> add persistent sessions and memory.
6. `jcode-tui-*` -> selectively port rendering primitives; retain Lucy's own UX.
7. provider crates -> create a Lucy provider adapter layer rather than binding the agent to one vendor.
