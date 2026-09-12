# Lucy

Lucy is a Rust-native AI computer buddy. The project is designed as a general-purpose agent foundation: it can plan work, execute tools, integrate MCP servers, manage sessions, and present an interactive terminal UI.

## Workspace

- `lucy-core` — shared agent, tool, provider, and interrupt abstractions
- `lucy-agent` — orchestration and task execution
- `lucy-tools` — native tool registry and shell execution
- `lucy-mcp` — MCP-facing abstraction/config boundary
- `lucy-tui` — Ratatui + Crossterm terminal interface
- `lucy-cli` — `lucy` executable

## Why this structure?

The foundation is intentionally separated so mature pieces from the open-source `jcode` project can be migrated selectively without inheriting coding-agent product assumptions wholesale.

## Status

Early foundation. The next phase is the selective migration of mature jcode components for tool execution, cancellation, planning, sessions/storage, transport, and provider integrations.

## Development

Requires a recent stable Rust toolchain.

```bash
cargo run -p lucy-cli
```

## License

See `THIRD_PARTY.md` for attribution and source-lineage notes.
