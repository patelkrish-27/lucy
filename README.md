# Lucy

**Lucy — Your AI Computer Buddy**

Lucy is a Rust-native, terminal-first AI computer agent. The initial architecture is intentionally modular so the UI, agent loop, tools, MCP integration, memory, and model providers can evolve independently.

## Workspace

- `lucy-core` — shared domain types, tool interface, model interface, cancellation/interrupt primitives
- `lucy-agent` — orchestration boundary between model planning and tool execution
- `lucy-tools` — native tool registry and built-in tools
- `lucy-mcp` — MCP configuration and transport boundary
- `lucy-tui` — Ratatui terminal interface
- `lucy` — executable CLI

## jcode foundation

Lucy is **not** a blind copy of jcode. jcode's current architecture provides useful patterns for Lucy: an agent runtime, central tool abstractions, TUI layers, provider boundaries, task/session types, and cancellation. This repository re-implements the core boundaries under Lucy names so product-specific coding-agent assumptions don't leak into Lucy.

The next migration phase can selectively bring mature jcode components into `vendor/jcode-*` or replace Lucy modules with adapted implementations where doing so is technically justified.

## Development

```bash
cargo check --workspace
cargo run -p lucy
```

## License

MIT. See `THIRD_PARTY.md` for attribution and migration notes.

## Runtime status

Phase 1 now includes a structured model-turn API, a bounded tool execution loop, cancellation-aware shell execution, output limits, timeouts, and deterministic shell-risk gating. The remaining provider and MCP work is intentionally separated from the agent core.
