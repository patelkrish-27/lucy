# Lucy + ADK-Rust capability union

This document records the capability-by-capability comparison between Lucy `main` and Zavora AI's `adk-rust` `main` at integration time (ADK-Rust `2.2.0`).

## Lucy already owns

- Native Rust TUI and CLI UX.
- Voice/STT push-to-talk path.
- The latency-sensitive main agent loop.
- Main-model-first planning and recovery.
- Cheap-model-only HyprFast command compilation.
- HyprFast capability discovery and routing.
- Dependency-aware execution waves and conservative concurrency.
- State-changing action verification.
- Approval gates and deny/replan policy.
- Action/replan/decision budgets and repeated-action loop prevention.
- Persistent turn/session history and legacy migration.
- Local tools and MCP server registration.
- Provider-facing tool-call-preserving history.
- UTF-8-safe output/context truncation and fast-path safeguards.

These pieces remain Lucy-owned because they are product-specific and sit directly on the hot execution path.

## ADK-Rust capabilities added through `lucy-adk`

| Capability | Union decision | Integration |
|---|---|---|
| Persistent semantic memory | Missing from Lucy | SQLite memory service, async and failure-isolated |
| Pluggable session-service abstraction/backends | Partially overlapping | Available through ADK re-exports for future backends |
| Sequential/parallel/loop agent primitives | Missing as generic reusable agents | Available through ADK |
| Graph workflows | Missing | Available through ADK |
| Artifacts | Missing | Available through ADK |
| Guardrails | Missing | Available through ADK |
| Skills + progressive disclosure | Missing | Available through ADK |
| Plugins | Missing | Available through ADK |
| Code execution tools | Missing | Available through ADK feature set |
| Sandboxed execution | Missing | Available through ADK feature set |
| Browser/computer-use framework primitives | Lucy has HyprFast-specific routing | ADK facilities are available without replacing HyprFast |
| Realtime agent primitives | Lucy has STT push-to-talk | ADK realtime layer is available for future bidirectional sessions |
| Audio pipelines | Lucy has STT | ADK audio features are available behind opt-in features |
| Evaluation | Missing | ADK eval crates available |
| OpenTelemetry telemetry | Lucy has tracing logs | ADK telemetry is available; no duplicate hot-path subscriber is forced |
| A2A server/protocol | Missing | Available through ADK |
| Authentication helpers | Missing | Available through ADK |
| Expanded model provider ecosystem | Lucy currently centers on OpenAI-compatible providers | ADK providers available as an extension surface |
| MCP sampling/transports | Lucy already has MCP | ADK MCP capabilities available where richer transport/sampling is needed |

## Deep-merge rules

1. Lucy's existing main-model-first execution loop remains authoritative.
2. `HyprFast` keeps the only cheap-model exception: subtask-to-exact-command compilation.
3. ADK-Rust is consumed natively as Rust crates; no Python runtime or subprocess bridge is introduced.
4. Persistent memory is asynchronous and best-effort so startup and command latency are not gated by ADK storage.
5. Existing Lucy sessions/history remain the source of truth for conversation turns; ADK memory is an additional long-term semantic memory plane.
6. ADK's generic orchestration primitives are available to new workflows without rewriting the existing HyprFast scheduler.
7. Large ADK feature families stay feature-gated instead of forcing every optional backend into the default Lucy binary.

## Current integration

`crates/lucy-adk` provides a small facade over ADK-Rust and enables the useful standard/extension features. `LucyRuntime` owns one optional `LucyAdk` service and asynchronously records completed assistant interactions to SQLite memory. Memory initialization failure only disables the extension; it never blocks Lucy startup.

The workspace therefore has a single Rust process architecture:

```text
Lucy TUI / CLI
      |
      v
Lucy Runtime (existing hot path)
      |
      +--> Main model / HyprFast / MCP / local tools
      |
      +--> ADK-Rust extension plane
             +--> persistent memory
             +--> workflows / graphs
             +--> skills / plugins
             +--> artifacts / code / sandbox
             +--> eval / telemetry / protocols
```
