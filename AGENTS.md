# AGENTS.md — working rules for this repository

## The one rule: no task-specific hardcoding

**Lucy must work for any task the user names. Never encode a workflow, a
destination, or a heuristic for one specific task or one specific site.**

"Play despacito song on yt" is the task this repository was debugged against.
It is a *regression fixture*, not a design centre. Nothing in `src/` may exist
because of it. If a change only makes that sentence work, it is wrong even when
it passes its test.

Concretely, the models are the router. Code may:

- discover capabilities (read the live tool catalog, ask the MCP server what
  tools exist);
- enforce contracts and safety (JSON shapes, probe sandboxing, approval gates);
- handle cheap deterministic preconditions (is CDP up? is the user asking about
  a URL?).

Code may **not**:

- contain a list of sites, app names, or task keywords that decides *what* to do;
- encode a site's DOM, URL shape, or quirks as a special case;
- carry one task's measured forensic details inside a prompt that every other
  task has to read;
- branch on a specific goal string.

If a capability seems to need a rule, the rule belongs in a **prompt the model
reads**, not in a branch the model cannot see.

### Why

Two reasons, both learned the hard way:

1. **A static branch is a closed world.** A list of sites has to be extended by
   a code change and a release for every site the user ever asks about.
2. **The model already knows more than the list does.** The Level 3 planner is
   told the live tool catalog and the goal, and can name a destination URL it
   has never seen in this codebase. A `match` statement cannot.

The failure mode is quiet and looks like success: the model narrates a plan in
prose, the user sees a confident reply, and nothing ran. A hardcoded shortcut
that misses is *worse* than no shortcut, because it hides the miss.

### Testing against a single task

Tests need concrete strings, so `despacito` and friends will still appear in
`#[cfg(test)]` bodies — that is fine and expected. Two rules:

- **Never in the test's name.** `parses_despacito_subtasks` implies a despacito
  feature. Name it for the shape under test: `parses_a_numbered_subtask_list`.
- **Assert the general rule.** A test should hold for any input of that shape.
  Prefer a table over a single anecdote, and keep one anecdote per incident so
  the regression stays traceable.

---

## Offender inventory

Each entry is a `match`/`contains`/constant that decides behaviour from task
words.

### Resolved

| Where | Was | Now |
| --- | --- | --- |
| `fast_perception::BROWSER_START_URL` | every run started on `youtube.com` | `about:blank`; the planner names the destination |
| `fast_perception::KNOWN_SITES`, `site_url_for_goal` | ~40-entry site→URL table | URL-only: a destination is a URL the user typed, nothing else |
| `fast_perception::goal_needs_browser` | ~35 hardcoded keywords | structural signals in code + `harness.browser_goal_keywords` in config |
| `automation::extract_target_url` | `flight`→Travel, `youtube`→YouTube, **else Google** | deleted; `goal_target_url` returns a typed URL or nothing |
| `browser_policy::verify_done` | host check against a *guessed* URL, so a non-Google task could never verify | host check skipped when the goal named no destination |
| `agent_loop::AGENT_PLAN_INSTRUCTIONS` | ~30 lines of one site's DOM forensics | the general rules those examples taught |
| `agent_loop::AGENT_REPLAN_INSTRUCTIONS` | same, plus a hardcoded probe recipe | same |
| `agent_loop::extract_type_text` | verb list (`type `, `fill `, `search for `) | quoting convention, which needs no vocabulary |
| `hyprfast` strategy label | goal words (`youtube`, `song`, `play`) | derived from the resolved tool set |
| `turn::is_open_only_goal` | verb list (`play`, `click`, `search`, …) | decided by arity — a bare destination is short |

### Remaining, and why

| Where | What | Status |
| --- | --- | --- |
| `hyprfast::score` / `desktop_score` | keyword lists (`youtube`, `song`, `window`, …) pick which tools are offered | **dead path.** `route`, `route_domain` and `context_for` have no production caller; the planner gets its tools from `planner_tool_set` over the live catalog instead. Delete with `Route`/`context_for`, or wire up deliberately — do not extend the lists. |
| `hyprfast::is_fast_path` | goal-word list | same dead path. |
| `heuristic_turn_classification` | verb list | load-bearing *by design* as the last-resort path when the classifier and the routing LLM are both down. It is the safety net, and it is now reachable again whenever the routing LLM fails — see the `VerifyOutcome::Unavailable` arm in `classify_turn`. Narrowing it means re-deciding that fallback, not just deleting words. |
| `config.harness.browser_goal_keywords` | topic words | **intentional.** This is the one topic list, and it is user-editable, so a new noun needs no release. Add to it rather than to code. |

Prompts that name a task only as an *illustration of a general rule* are fine
and stay. The prohibition is on knowledge that only one task benefits from.

---

## Adding a capability

Put it where the model can see it, not where a branch has to know about it:

- **A new tool** → it appears in the MCP/live catalog; the planner brief is
  built from that catalog at runtime (`router::build_tool_brief`). No code.
- **A new site or workflow** → nothing. The planner writes the URL.
- **A new way to phrase a goal** → nothing. Routing is intent-based
  (`turn::classify_turn` → verifier reads the request against the catalog).
- **New probe/verify knowledge** → the planner authors the probe per task.

If a change needs a new `if goal.contains("…")`, it is the wrong shape. Say so
and route it through the prompt instead.
