# Lucy Ultra-Fast Computer Control

Lucy should optimize for **wall-clock task completion**, not token count alone. A literal 100x speedup is not guaranteed for every task because browser/network latency and model inference are external bottlenecks, but the architecture can remove most avoidable overhead.

## Highest-impact techniques

### 1. One strategic main-model call

Use the main model to create the structured plan once. Do not ask it to re-decide after every successful action. Re-enter the main model only for failure, unexpected state, recovery, or final evidence-based completion.

### 2. Main-model tool selection

The main model selects the exact tool and arguments for each ready subtask using only the allowed schemas and current execution evidence. Keep the selection prompt bounded to the current subtask rather than the entire capability catalog.

### 3. Macro/batch tools

If HyprFast exposes a safe batch/macro capability, prefer one batch command for several deterministic operations. This is the largest practical multiplier for workflows such as `open -> navigate -> type -> press`.

### 4. Event-driven verification

Do not repeatedly poll screenshots or full browser state after every action. Maintain a live state cache from events where available and verify only the postcondition that matters. Hyprland exposes a live event socket for this purpose. See the Hyprland IPC documentation.

### 5. Native compositor fast paths

For desktop/window/workspace operations, prefer direct Hyprland IPC or a HyprFast command backed by it over pixel-level automation. Hyprland documents synchronous IPC and recommends batching control calls when multiple operations are needed.

### 6. Parallel read-only observations

Independent observations can run concurrently. Never parallelize conflicting browser/UI mutations. Use dependency waves and resource/conflict keys to make this deterministic.

### 7. Keep exact schemas ahead of catalog metadata

The command compiler needs executable schemas and current state. A huge capability index must never crowd those out of the model context.

### 8. Avoid screenshots when structured state exists

Prefer browser accessibility/DOM snapshots, element hints, URLs, titles, and compositor state. Use vision only when structured observations cannot answer the question.

### 9. Reuse MCP sessions

Keep one warm HyprFast MCP process per configured server. Avoid repeatedly spawning the MCP process or rediscovering tools during a task.

### 10. Cache deterministic decisions

Cache capability routing and safe, state-independent compiler decisions. Never blindly cache state-dependent UI coordinates, element IDs, tabs, or windows.

## Target execution loop

```text
USER
  -> MAIN MODEL: plan once
  -> dependency scheduler
  -> main-model tool selection
  -> HyprFast macro/direct command
  -> event/state update
  -> next independent command
  -> verify postcondition
  -> MAIN MODEL only if recovery is required
```

## Latency budget

For a local desktop task, the target should be approximately:

- planner: one model round trip
- command compilation: one tiny fast round trip per semantic subtask, or zero for deterministic compiled macros
- execution: one warm MCP call, preferably one batch/macro call
- verification: event/state lookup instead of screenshot polling
- recovery: only on actual failure

The critical anti-pattern is:

```text
main model -> tool -> main model -> tool -> screenshot -> main model -> tool
```

The desired pattern is:

```text
main model -> [tool selection -> batch tool -> state event] x N -> verify
```

## Benchmark requirements

Lucy should measure:

- end-to-end latency
- main-model calls/task
- cheap-model calls/task
- MCP calls/task
- screenshot/vision calls/task
- tool execution latency
- verification latency
- tokens in/out
- task success rate
- recovery rate

Report p50/p95/p99 latency and compare the same deterministic task suite against OpenCode. Do not claim 100x without benchmark evidence.
