# Lucy

**Lucy — Your AI Computer Buddy**

Lucy is a Rust-native, terminal-first AI computer agent. The architecture is modular so the UI, agent loop, tools, MCP integration, memory, and model providers can evolve independently.

## Workspace

- `lucy-core` — shared domain types, tool interface, model interface, cancellation/interrupt primitives
- `lucy-agent` — general model planning and tool execution
- `lucy-tools` — native tool registry and built-in tools
- `lucy-mcp` — MCP configuration and persistent stdio transport
- `lucy-hyprfast` — HyprFast capability catalog, routing, and strategy selection
- `lucy-knowledge` — durable knowledge base: plain Markdown topics, provenance, deterministic FTS recall
- `lucy-runtime` — session/runtime composition and hierarchical computer-task planning
- `lucy-mascot` — the mascot: approved artwork, embedded and resized, drawn as half blocks
- `lucy-tui` — Ratatui terminal interface
- `lucy` — executable CLI

## Knowledge base (lucy-knowledge)

Lucy keeps a durable, human-readable knowledge base under
`~/.config/lucy/knowledge` (override: `$LUCY_KNOWLEDGE_DIR`). It is built as a
library Lucy reads in three escalating levels, matching how the production
systems (Grok Build, OpenClaw, MUSE) manage memory:

1. **Index** — a generated topic list (`render_digest`, ~1.2k char budget)
   injected into every turn's prompt. Titles, not content.
2. **Recall** — deterministic FTS5 search (BM25) injected per request
   (~6k char budget). Only *promoted* owner/agent claims are eligible.
3. **Open** — `kb_search` / `kb_get` tools the model calls mid-task to pull a
   full topic on demand. Read-only, no approval prompt.

Safety properties are structural, not heuristic:

- **Provenance columns** (`owner` / `agent` / `untrusted` / `system`) are
  written by classification code and never parsed from text. `untrusted`
  (page text, tool results, files) is searchable on demand but **cannot be
  promoted or auto-injected**, so a web page cannot plant persistent
  instructions.
- **Promotion gate** — captured claims start *unpromoted*; they become
  injectable only after a recall-weighted count and only for `owner`/`agent`
  origin.
- **Tombstone on forget** — `/forget <slug>` hides instead of deleting, and the
  row stays for the audit.
- **Secret filter** — credential-shaped text is refused at the door.

Run `/kb-dream` in the TUI to consolidate: promote recalled claims and reindex
everything from the Markdown files (the files, not the index, are the source of
truth). With `knowledge.capture_enabled = true`, every finished turn also runs a
background extractor that stores durable owner facts as new claims.

## Model providers

Any OpenAI-compatible endpoint can be connected from `/settings` → **1. Connect
providers**. The first row picks a vendor preset, which fills the base URL and
pre-fills the API key from that vendor's environment variable:

| Preset | Base URL | Key env var |
| --- | --- | --- |
| OpenRouter | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| OpenAI | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| Groq | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` |
| Ollama | `http://127.0.0.1:11434/v1` | — (loopback, no key) |
| OpenChat (local) | `http://127.0.0.1:11435/v1` | — (loopback, no key) |
| Custom | type it | — |

`[Test]` probes `GET {base}/models` and `[Save]` stores the provider with every
model id it found, so all 458 OpenRouter models land in the L1/L2/L3 dropdowns
as `openrouter/<vendor>/<model>`.

The legacy text endpoint (`[models] text_base_url`) gets the same treatment
automatically: at every startup Lucy probes its `GET {base}/models` and, when
its model list changed, registers/updates a matching provider entry and
persists it (`lucy_agent::sync_text_endpoint`). Model ids the endpoint marks
`deprecated` are kept for selection but never picked as a fresh default — when
the compiled-in default (`DEFAULT_TEXT_MODEL`) no longer resolves to a live
id, it is rebound to the endpoint's first live model. Renaming models on the
server therefore needs no code change.

OpenRouter gets extra pieces of handling:

- `HTTP-Referer` / `X-Title` attribution headers on every request to
  `openrouter.ai` (`lucy_agent::apply_endpoint_headers`), so Lucy is
  identifiable on openrouter.ai rankings.
- Model ids must be `vendor/model`. A configured name that cannot exist there
  (a leftover `gemini-web` from the local proxy) is corrected to
  `OPENROUTER_DEFAULT_MODEL` instead of 404ing on every turn
  (`lucy_config::model_for_endpoint`).
- Every request states an output ceiling
  (`lucy_agent::MAX_OUTPUT_TOKENS`). Omitting `max_tokens` is not "unbounded"
  there: the gateway sizes the request against the model's *full* context
  window, so a 65k-context model becomes a ~65k-token reservation and a
  free-tier key with no credits is rejected with `402 Payment Required: You
  requested up to 65535 tokens, but can only afford 54814`. It reads like a
  billing problem and is really a missing field.

`complete_json` / `complete_text` resolve a `provider_id/model` key through the
config (`OpenAIProvider::target_for`) instead of using the legacy
`text_base_url` directly, so a stored `openrouter/…` key reaches OpenRouter
rather than the local OpenChat proxy on `127.0.0.1:11435`, which would reject it
as an unknown model name. A bare model name still uses the legacy endpoint,
which is what a keyless local server depends on.

## Agent architecture

Lucy implements the whiteboard command pipeline (see
`crates/lucy-runtime/src/command.rs` — the single place that documents the
order):

```text
Lucy command (own STT → text; TTS beep acknowledges)
    │
    ▼
Identify: does the command need only a response,
          or actions to reach a goal state?          (decider-serve, 1 forward pass)
    │
    ├── requires only response ──► decide reasoning level 1 / 2 / 3
    │                              └── answer with L1 (Flash-lite) /
    │                                  L2 (Flash) / L3 (Pro) model
    │
    └── needs action ──► hyprfast SKILL + user request ──► L3 (Pro) LLM
                         └── valid list of hyprfast commands
                               └── Lucy executes them one by one in order
                                   └── task completed
```

### The act path (what actually runs)

`harness.agent_loop_enabled` (on by default) picks the two-speed agent loop over
the blind command plan. `LucyRuntime::execute_goal_outcome` is the single
decision point, so the CLI and the TUI can never disagree about which path a
goal took. The loop is four phases with one slow (LLM) call in the middle:

```text
PHASE 0  SITE       0 slow   go to the site the goal names, if we are not on it
PHASE 1  PLAN       1 slow   goal -> 2..6 objectives, each with a success_probe
PHASE 2  ACT        0 slow   per objective: perceive -> decide -> act -> verify
PHASE 3  REPLAN     1 slow   only on genuine deviation, bounded by harness budgets
PHASE 4  FINAL      0 slow   fast verify of the whole goal, then an honest summary
```

Two design points carry most of the weight:

- **`success_probe` is ground truth.** `success_check` is answered by a small
  vision model on a half-resolution screenshot, which is a weak oracle for
  anything the eye reads as "it works". So the planner also writes one
  JavaScript expression per objective, evaluated with `browser_evaluate` after
  every action — `Array.from(document.querySelectorAll('video')).some(v =>
  !v.paused)` ends an objective on the first attempt instead of after a budget
  of identical clicks. Probes are sanitised (`agent_loop::sanitize_probe`): one
  side-effect-free expression, no statement separators, no mutating or
  exfiltrating verbs, arrow-IIFEs unwrapped to their single expression.
- **Navigation is a fact, not a decision.** A site named in the goal
  (`fast_perception::site_url_for_goal`) is resolved before the planner is
  called. Left to the planner, "play despacito on youtube" came back as "type
  the URL into the address bar" — which the fast lane cannot do, because it
  acts on page elements, so it typed `youtube.com` into YouTube's own search
  box.
- **Instructions name their target, they never count to it.** The fast lane
  resolves `suggested_action` with a small local model that only sees the
  element names on screen. "the first video result" is answered with whatever
  that model reads first in DOM order — on a YouTube search page, the "All"
  filter chip — and the click silently does nothing. The plan prompt therefore
  requires the target's visible name ("click the video result titled 'Luis Fonsi
  - Despacito ft. Daddy Yankee'") and forbids ordinals outright.

`execute_goal` (the blind plan) is still reachable, and is the fallback when
`agent_loop_enabled` is false. A planner that returns nothing usable — an empty
`commands` array, an unparseable reply, a reasoning model that returns no
content — falls through to the agent loop rather than ending the run.

### Reporting

A run is reported complete on positive evidence only, and there are three
independent guards — each added because a real run passed the other two while
claiming a video was playing when it was not:

- **A failure stays on the books.** A replan replaces the *remaining*
  objectives, so an objective that never verified used to disappear from the
  `done`/`total` ratio the moment the revised plan was shorter. A 2-objective
  plan whose first objective failed, replanned into two easy sub-steps that
  both passed, reported `3/3 verified` and `✔ Goal completed`. `AgentRunStats`
  now keeps `objectives_failed`, and completion requires it to be empty.
- **The goal's own probe outranks the per-objective tally.** The last
  objective's `success_probe` is consulted for the whole-goal check, because it
  reads the page rather than a screenshot. A 2B vision model will agree that
  "the video is playing" while looking at a page of search results; a probe
  cannot.
- **The report and the ✔/⚠ symbol are one decision.** `AgentRunStats::complete`
  is set once, in PHASE 4, and the CLI and TUI read it. The symbol used to be
  recomputed from the ratio separately, which is how a run printed `✔` beside a
  report that said it was only partial.

When the evidence is short, the run reports `⚠ Partial` and names what is
still outstanding — including the objectives a replan replaced, which are no
longer in the plan. The blind plan's stats line says so explicitly (`N blind
plan step(s), no verification of the end state`) instead of printing an empty
two-speed split.

In code that is:

1. **Voice in** — `lucy-stt` (`GroqStt`, own STT) transcribes; `lucy-stt::feedback`
   plays the ack/done/error beeps and offers the `$LUCY_TTS_COMMAND` TTS hook.
2. **Step 1, identify** — `classify_turn` asks `decider-serve` once (branch +
   reasoning level), degrading to heuristics when the classifier is down.
3. **Step 2A, respond** — `answer_turn` replies with the tier's model
   (`L1 → flash-lite`, `L2 → flash`, `L3 → pro`; an unbound tier degrades to
   the next *cheaper* tier, and only a fully unbound stack falls to the
   compiled-in default text model).
4. **Step 2B, act** — `plan_commands` sends the hyprfast SKILL
   (`skills/lucy/SKILL.md`, capped at 12k chars) + the request + the tool
   catalog to the Level 3 model for an ordered command list, then
   `execute_plan_sequentially` runs the commands **one by one in order**
   (fail-fast, approval-gated, status streamed), falling back to the subtask
   automation loop when the planner returns nothing usable.

There is no separate "main model". Lucy has exactly four selectable models —
Level 1, Level 2, Level 3 and the voice model — and **Level 3 is the anchor**:
it plans every action, and `Manual` chat mode pins every turn to it. An older
`config.toml` carrying `chat.main_model` is migrated on load: that value seeds
Level 3, and the legacy key is then dropped.

## Permissions and automode

Lucy gates tool calls behind an approval prompt. Three modes, set with `/auto`
(or the **5. Automode** row in `/settings`, persisted in `config.toml` as
`approvals.mode`):

| `/auto`  | `approvals.mode` | Behaviour                                    |
| -------- | ---------------- | -------------------------------------------- |
| `on`     | `never`          | Automode: never asks. Run `/auto off` to stop. |
| `off`    | `write`          | Asks before risky/destructive tools (default) |
| `always` | `always`         | Asks before every tool, including read-only   |

Answering a prompt with `a` (always allow) writes the tool to
`approvals.always_allow`, so the choice survives a restart instead of
re-prompting every session. A permanently allowed tool is never prompted for
again, in any mode.

Automode has no prompts, so its safety net is the kill switch: **`/stop`,
`Esc`, or `Ctrl+C`** stops the running task from any state, including while a
prompt is open. `Ctrl+C` works during a prompt where `Esc` would only deny that
one call.

## MCP servers

HyprFast is built in and discovered automatically. Additional servers come from
`~/.config/lucy/mcp.toml` (see `config/mcp.toml.example`); the file is optional.
A server that fails to start is reported at startup and by `/doctor` rather than
disappearing silently.

Lucy's MCP client speaks **stdio** only. Servers that are HTTP-only need a
transport bridge — for example Slack's official server at
`https://mcp.slack.com/mcp` is Streamable-HTTP and requires a registered Slack
app, so the deprecated `@modelcontextprotocol/server-slack` is no longer the
right entry.

## jcode foundation

Lucy is **not** a blind copy of jcode. jcode's architecture provides useful patterns for an agent runtime, central tool abstractions, TUI layers, provider boundaries, task/session types, and cancellation. Lucy re-implements the core boundaries under Lucy names so product-specific coding-agent assumptions don't leak into Lucy.

## The mascot

`lucy-mascot` draws Lucy herself. She is the approved pose library
(`crates/lucy-mascot/assets/poses/`), sliced out of the reference sheet and
embedded with `include_bytes!`, resampled per frame with a Lanczos filter —
never a procedural painter and never ASCII art, so her figure is always the
artwork that was approved. Where the terminal speaks the Kitty, Sixel, or
iTerm2 graphics protocol the art is transmitted as pixels (via
`ratatui-image`), so the detail in her face is preserved; everywhere else the
same poses are resized and packed into upper-half-block cells. Transparent
pixels reset the cell, which is why she stands on your terminal theme instead
of a card colour that could never match it.

She carries seven moods, and which one she wears is derived from the flags the
TUI already tracks (`App::mascot_mood`): an approval prompt outranks everything,
then listening, then busy — split into talking / working / thinking by whether
tokens or a tool are in flight — then a short `Happy` hold after a turn ends,
otherwise `Idle`. Nothing inspects a request or a phase string, so a new
activity phase upstream needs no new vocabulary here.

Two consumers share the one renderer: the TUI, which packs two pixels into every
cell with the upper-half block `▀`, and `lucy mascot [dir] [--scale N]`, which
writes the same frames as PNGs plus an HTML sheet so the art can be judged as an
image rather than squinted at through a character grid.

The only motion is a gentle bob, a pure function of the session clock and phased
per mood, so she never crawls and an export reproduces exactly what the TUI
drew. `/mascot [auto|large|small|off]` sizes her; `LUCY_MASCOT_COLOR=truecolor|256`
overrides colour detection for terminals whose environment lies.

## Development

```bash
cargo fmt --all -- --check
cargo check --workspace
cargo test --workspace
cargo run -p lucy
```

## License

MIT. See `THIRD_PARTY.md` for attribution and migration notes.
