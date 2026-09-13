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

## HyprFast task architecture

Lucy does not blindly map a detected intent directly to a computer tool. For computer-operation tasks it uses a hierarchical pipeline:

```text
User request
    ↓
Cheap task decomposer
    ↓
Ordered concrete subtasks
    ↓
HyprFast category router
    ↓
Cheap command planner + exact tool schemas + current results
    ↓
One validated HyprFast command
    ↓
HyprFast execution
    ↓
Result becomes context for the next subtask
```

For example, `draw a human in the existing Excalidraw tab` can become observation first, then concrete drawing operations. The cheap planner receives only the relevant HyprFast tools and their exact JSON schemas for each subtask, rather than the entire HyprFast catalog. It is explicitly instructed not to invent UI state, IDs, coordinates, tab numbers, or arguments that are not supported by the supplied context.

Set `LUCY_PLANNER_MODEL` to choose the cheap planning model. It defaults to `gpt-4o-mini`. The main agent model remains controlled by `OPENAI_MODEL`.

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
