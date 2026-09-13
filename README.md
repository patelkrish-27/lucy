# Lucy

**Lucy — Your AI Computer Buddy**

Lucy is a Rust-native, terminal-first AI agent designed to operate a computer through explicit tool boundaries. The current runtime connects keyboard input, live voice transcription, model planning, native tools, MCP tools, cancellation, and persistent sessions.

## Workspace

- `lucy-core` — shared domain types, events, tool/provider interfaces and interrupts
- `lucy-agent` — bounded model/tool orchestration with concurrent independent calls
- `lucy-tools` — shell, filesystem, search and git tools
- `lucy-mcp` — MCP server discovery and callable tool proxies
- `lucy-runtime` — application composition, MCP loading and session persistence
- `lucy-tui` — Ratatui interface and keyboard/voice event loop
- `lucy-stt` — live microphone capture and Groq Whisper transcription
- `lucy` — executable CLI

## Quick start

```bash
export OPENAI_API_KEY=...
export GROQ_API_KEY=...       # optional, enables voice
cargo run -p lucy
```

For an OpenAI-compatible local endpoint:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:1234/v1
export OPENAI_MODEL=your-model
```

## MCP

Copy `config/mcp.toml.example` to `~/.config/lucy/mcp.toml` and add your servers. Lucy will discover their tools and make them available to the agent.

## Controls

- `Enter` — run the typed task
- `Super+C`, `Ctrl+Space`, `F2`, `F9`, `Alt+V`, or `Ctrl+M` — record one voice task
- `Ctrl+C` — interrupt the current task
- `Esc` — quit

## State

Session history is persisted to `~/.local/state/lucy/session.json` by default. Override with `LUCY_SESSION_FILE`.

## Safety

Lucy can execute shell commands, so do not treat it as a sandbox. A small catastrophic-command block is enabled by default. Set `LUCY_ALLOW_DANGEROUS=1` only when you intentionally want unrestricted shell execution. The next hardening step is interactive per-tool approval and sandboxed execution.

## Development

```bash
cargo check --workspace
cargo test --workspace
```

MIT. See `THIRD_PARTY.md` for attribution and migration notes.
