# Third-party attribution

Lucy is independently written, but its safety/runtime design is informed by the public `1jehuang/jcode` project.

Upstream project: https://github.com/1jehuang/jcode
License: MIT
Upstream copyright: Copyright (c) 2025 Jeremy Huang
Current upstream LICENSE blob: `e921bccf34499d0c806a2131c2faa95d375c1c96`

Lucy specifically adapts concepts from jcode's `jcode-agent-runtime`, `jcode-tool-core`, and `jcode-command-risk`. The shell risk policy in `crates/lucy-tools/src/risk.rs` is a Lucy implementation inspired by those interfaces and safety rules rather than a verbatim vendored copy.

The upstream MIT notice requires preservation of the copyright and permission notice when substantial upstream source is copied. The complete notice is retained here because Lucy's design work materially draws from those upstream modules.

## ADK-Rust

Lucy also integrates the `zavora-ai/adk-rust` project as an optional native Rust agent capability layer through `crates/lucy-adk`.

Upstream project: https://github.com/zavora-ai/adk-rust
Version integrated: `2.2.0`
License: Apache License 2.0
Upstream copyright: Copyright (c) 2026 Zavora AI

Lucy does not vendor or replace its latency-sensitive execution engine with ADK-Rust. The integration consumes ADK-Rust as a library for complementary capabilities such as persistent memory, workflow/graph primitives, artifacts, skills, plugins, code/sandbox facilities, evaluation, telemetry, authentication, and expanded agent protocols.
