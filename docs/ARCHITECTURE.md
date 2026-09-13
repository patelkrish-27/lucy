# Lucy architecture

```text
keyboard / voice
       │
       ▼
  lucy-tui / CLI
       │
       ▼
  lucy-runtime ───────────── persistent session
       │
       ▼
   lucy-agent
       │
       ├── model provider
       │
       ├── native tools
       │     ├── shell
       │     ├── read/write files
       │     ├── directory listing
       │     ├── recursive search
       │     └── git
       │
       └── MCP tools
             │
             └── external services
```

Every side effect is executed behind the `Tool` boundary. The TUI only renders events and submits user intent.

## Runtime flow

1. `lucy-runtime` creates the provider, native registry, MCP tools and persistent session.
2. Keyboard or voice input calls `LucyRuntime::submit`.
3. `lucy-agent` runs a bounded tool loop and emits structured `AgentEvent`s.
4. Tool calls from one model turn execute concurrently; results return in model order.
5. History events are persisted under `~/.local/state/lucy/session.json` by default.
6. Ctrl+C fires the shared interrupt signal; active shell commands are terminated.

## MCP

MCP servers are configured in `~/.config/lucy/mcp.toml` (or `LUCY_MCP_CONFIG`). Lucy discovers `tools/list` and exposes each server tool as a normal Lucy tool. Calls use the standard JSON-RPC `tools/call` method.

## Safety

The shell tool has a small built-in block list for catastrophic system-wide commands. `LUCY_ALLOW_DANGEROUS=1` can disable that guard when the operator deliberately wants unrestricted execution. A production release should replace this with interactive per-tool approvals and a richer policy engine.
