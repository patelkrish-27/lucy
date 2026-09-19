# Lucy

**Lucy — Your AI Computer Buddy**

Lucy is a Rust-native, terminal-first AI computer agent. The architecture is modular so the UI, agent loop, tools, MCP integration, memory, and model providers can evolve independently.

## Workspace

- `lucy-core` — shared domain types, tool interface, model interface, cancellation/interrupt primitives
- `lucy-agent` — general model planning and tool execution
- `lucy-tools` — native tool registry and built-in tools
- `lucy-mcp` — MCP configuration and persistent stdio transport
- `lucy-hyprfast` — HyprFast capability catalog, routing, and strategy selection
- `lucy-runtime` — session/runtime composition and hierarchical computer-task planning
- `lucy-tui` — Ratatui terminal interface
- `lucy` — executable CLI

## Agent architecture

Lucy uses one reasoning model and a deterministic Rust execution runtime:

```text
User request
    ↓
Context builder
    ├── session history
    ├── long-term memory
    ├── current capability route
    └── live execution evidence
    ↓
Main model
    ├── chat response, or
    └── structured task plan
            ↓
Dependency scheduler
            ↓
Main model selects exactly one tool
            ↓
Rust validation + approval policy
            ↓
MCP / HyprFast / local tool execution
            ↓
Result + observation
            ↓
Verification
    ├── complete
    ├── continue
    └── main-model recovery/replan
```

The main model owns strategy, decomposition, tool selection, verification, and recovery. There is no secondary or cheap model in the execution path. The runtime supplies exact tool schemas for the current subtask, validates the selected tool and arguments, enforces approvals and execution budgets, and records live evidence.

For example, `draw a human in the existing Excalidraw tab` can become observation first, followed by drawing operations. The model is explicitly instructed not to invent UI state, IDs, coordinates, tab numbers, or arguments that are not supported by the supplied context.

## jcode foundation

Lucy is **not** a blind copy of jcode. jcode's architecture provides useful patterns for an agent runtime, central tool abstractions, TUI layers, provider boundaries, task/session types, and cancellation. Lucy re-implements the core boundaries under Lucy names so product-specific coding-agent assumptions don't leak into Lucy.

## Development

```bash
cargo fmt --all -- --check
cargo check --workspace
cargo test --workspace
cargo run -p lucy
```

## License

MIT. See `THIRD_PARTY.md` for attribution and migration notes.
