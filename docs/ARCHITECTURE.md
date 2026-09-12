# Lucy architecture

```text
TUI / CLI
   │
   ▼
lucy-agent ──────────────── model provider(s)
   │
   ├── planner / turns
   ├── permissions
   ├── cancellation
   └── tool calls
         │
         ├── native tools (lucy-tools)
         └── MCP tools (lucy-mcp)
                  │
                  ▼
              external systems
```

The UI must not perform side effects itself. Every side effect goes through a tool or provider boundary so later desktop/browser/computer-control surfaces can reuse the same agent engine.
