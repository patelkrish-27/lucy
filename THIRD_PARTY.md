# Third-party attribution

Lucy is independently written, but its safety/runtime design is informed by the public `1jehuang/jcode` project.

Upstream project: https://github.com/1jehuang/jcode
License: MIT
Upstream copyright: Copyright (c) 2025 Jeremy Huang
Current upstream LICENSE blob: `e921bccf34499d0c806a2131c2faa95d375c1c96`

Lucy specifically adapts concepts from jcode's `jcode-agent-runtime`, `jcode-tool-core`, and `jcode-command-risk`. The shell risk policy in `crates/lucy-tools/src/risk.rs` is a Lucy implementation inspired by those interfaces and safety rules rather than a verbatim vendored copy.

The upstream MIT notice requires preservation of the copyright and permission notice when substantial upstream source is copied. The complete notice is retained here because Lucy's design work materially draws from those upstream modules.
