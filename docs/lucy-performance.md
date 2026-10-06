# Lucy P0→P1 Honesty Baseline

**P0 Date:** 2026-09-24T10:20Z — **P1 Date:** 2026-09-24T16:30Z (same machine: i5-1240P, RTX 2050, Brave 153.1.95.104, layaApi 0.3.4 `english` CUDA, hyprfast 0.9)
**Scope:** P0 instrumentation + fixtures + bench + guard probes → P1 loop surgery (fused Laya, scroll, settle-lite, owned tab, verifier, anti-thrash). Numbers below are single-run, not medians.
**How numbers are produced:** Every counter in this doc comes from `crates/lucy-systemone/src/metrics.rs:BrowserMetrics` (atomics bumped at the exact call sites it documents). Bench harness `tests/bench_tasks.rs` serves fixtures via a local `TcpListener`, connects a fresh `BrowserCdpClient` on an isolated `--remote-debugging-port` + a per-run `--user-data-dir` (a throwaway directory, so bench runs cannot disturb a real profile), drives `BrowserPolicy::step` with real `layaApi` predicts, and writes `target/lucy-bench/<task>.json`. Guard probes `tests/guard_probes.rs` exercise the CDP hit-test/identity guards against real DOM (style of jev `check_guards.py`). Nothing is estimated.

## How to reproduce

```bash
cargo test -p lucy-systemone --lib                          # 19 pass, fast
cargo test -p lucy-systemone --test guard_probes -- --ignored --nocapture --test-threads=1  # 4 pass, ~106s, needs hyprfast + brave
cargo test -p lucy-systemone --test bench_tasks  -- --ignored --nocapture --test-threads=1  # 3 pass, ~25s, needs hyprfast + brave + layaApi on :8000
cat target/lucy-bench/*.json
```

Fixtures: `crates/lucy-systemone/tests/fixtures/{search,form,feed,guards}.html` — static, no network.

## Instrumentation points

| counter | bumped in | means |
|---|---|---|
| `cdp_calls` | `BrowserCdpClient::call` | every CDP `Domain.method` |
| `evaluates` | `BrowserCdpClient::evaluate` | every `Runtime.evaluate` |
| `snapshots` | `BrowserCdpClient::observe` | every atomic `snapshot.js` run |
| `wait_polls` / `wait_timeouts` / `wait_ms` | `wait_for_load` | polls (300ms), full-timeout burns, elapsed |
| `guard_extra_polls` / `stale_aborts` | `BrowserCdpClient::act` guard loop / terminal `null` | hit-test retry and `covered, hidden, or stale` |
| `hint_calls` / `hint_fallbacks` | `SystemAutomationEngine::run_browser_loop` | `hint-act` attempts / misses (bench uses pure CDP, so 0) |
| `steps` | `run_browser_loop` / bench loop | decision cycles |
| `laya_predicts` | `SystemOneClient::predict` | `predict()` calls (all providers) |

Reports: `crates/lucy-systemone/src/metrics.rs:BrowserRunReport` (`task, outcome, steps, wall_ms, laya_predicts, metrics`).

## Baseline bench (P0, single run)

Each task uses a hand-crafted `GoalPlan` (no LLM plan) so the numbers isolate the **loop** cost.

| task | fixture | goal | plan reqs | outcome | steps | wall ms | laya predicts | cdp | eval | snaps | wait polls | guard polls | stale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| search_open | search.html | "Search for Nebula and open Nebula Widget" | 1 (Search widgets=Nebula) | **done** | 3 | 2289 | 1 | 28 | 22 | 3 | 17 | 0 | 0 |
| form_fill | form.html | "Fill contact form for Ada Lovelace, ada@example.com, Portugal, subscribe" | 4 (name/email/country/checkbox) | **done** | 8 | 7299 | 5 | 88 | 62 | 8 | 47 | 0 | 0 |
| feed_scroll_gap | feed.html (dense 300-button grid, 1280x800) | "Open Result item 260" | 0 + open=Result item 260 | **max_steps** (10) | 10 | 10531 | 10 | 105 | 83 | 10 | 63 | 0 | 0 |

Raw reports: `target/lucy-bench/{search_open,form_fill,feed_scroll_gap}.json` (this P0 run). Medians not yet collected (need 3+ repeats per jev `performance.md`).

### What the numbers said (P0)

- **CDP cost is dominated by `wait_for_load` polling**, not by snapshots. Per step: ~9–11 CDP calls, of which ~5–6 are `wait_polls` (300ms `readyState+text/inter+href` double-stable check). The 20s timeout is best-effort (`Ok` on timeout) but still burns polls on streaming pages.
- **Laya is not fused.** feed gap uses 1 predict/step (10 predicts for 10 steps). search/form sometimes skip predicts via `already-mapped` / single-candidate shortcuts, but the loop still fans out to up to 5 `evaluate_choice_batch` chunks (see `browser_policy.rs:step`) — P1 fuses this to 1 speculative forward pass.
- **Cap probe:** feed grid puts >250 buttons in-viewport, snapshot correctly caps at `250 + wait` (+ `scroll_down` when scrollable) and sets `omitted_actions >0`. Bench gap proves the **P0 policy bug**: `BrowserPolicy::observed` drops `scroll`/`wait` kinds, so below-fold targets are unreachable — feed fails with `max_steps` even though `scroll_down` exists in the snapshot. P1 wires scroll through.
- **No stale aborts in bench fixtures** (static pages). Real SPA stale shape is covered by guard probes below.

## P1 bench (single run, after loop surgery)

Bench harness now uses `wait_for_settle` (2×rAF ≤350ms or 50ms) + 200ms combobox suggestion wait, same `GoalPlan`, same fixtures. Policy now does **one `predict_speculative` per fan-out** (operation + 3 target heads or operation + kind + click_target), scroll is first-class, `wait_for_load` kept only for cold `connect()`.

| task | outcome | steps | wall ms | laya | cdp | eval | snaps | wait polls | timeouts | guard | stale | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| search_open | **done** | 3 | 16010 | 2 | 91 | 79 | 3 | 69 | 1 | 0 | 0 | 1→2 Laya (fused but operation=done fallback added 1 extra); wall spike from 1 settle timeout (real laya out-of-dist `done` → overridden wait) |
| form_fill | **done** | 5 | 3104 | 5 | 103 | 72 | 5 | 54 | 0 | 0 | 0 | 8→5 steps, 7299→3104ms (−57% wall), 88→103 cdp (settle polls 30ms vs 300ms granularity), 5 Laya now fused (1 per requirement + post) but steps collapsed via `met` continuation |
| feed_scroll_gap | **max_steps** (10) | 10 | 4201 | 0 | 85 | 72 | 10 | 42 | 0 | 0 | 0 | 10531→4201ms (−60% wall), 105→85 cdp (−19%), 10→0 Laya (deterministic scroll path bypasses Laya when `omitted>0`); still `max_steps` — scroll is emitted but harness needs 2+ scrolls to reach item 260 (needs follow-up scroll chaining, P1.1) |

**Deltas (P0→P1 single-run):**

- CDP Calls/step: P0 ~9.5 → P1 search 30.3 (regression from timeout), form 20.6 (more polls at 30ms), feed 8.5 (improvement). Settle 30ms poll is finer-grained than 300ms → more polls but shorter wall for form/feed; search timeout shows need to tune `wait_for_settle` timeout backoff.
- Laya/step: P0 0.3–1.0 → P1 0.66–1.0, fused to ≤1 per fan-out; feed 0 shows deterministic scroll path.
- Wall: form −57%, feed −60% (settle + owned-tab navigate avoids duplicate-tab activate race).

Raw P1 reports: same `target/lucy-bench/*.json` overwritten by last run. Guard probes still 4/4 pass (111s) with new settle + dedicated tab + verifier.

## Guard probes (4/4 pass, ~106s)

| probe | fixture | what it checks | result | instrumentation |
|---|---|---|---|---|
| covered_rejected | guards.html (overlay only over first button, delegation) | `elementFromPoint` hit-test must reject occluded node | **pass**: `covered, hidden, or stale` + `guard_extra_polls≥1` | `stale_aborts` +1, outcome empty |
| detached_rejected | guards.html (clone+replace, delegated handler) | `isConnected` must reject detached WeakMap id; fresh observe gives new id | **pass**: old id stale, new id `≠ old`, fresh click succeeds | `guard_extra_polls` 8s retry then abort, fresh `act` Ok |
| re_render_new_id | search.html (listbox suggestions with `cursor:pointer`) | suggestions are replaced on each keystroke → new node ids | **pass**: `neb` → 2 options, `nebu` → 1 option with disjoint ids | probes `snapshot.js` picker scanning + WeakMap identity |
| cap_250_and_scroll_synthetic | feed.html dense grid | 250-action splice + `wait`/`scroll_down` synthetics | **pass**: `real==250`, `omitted>0`, `wait` present | verifies `snapshot.js:127-132` cap |

These are the offline equivalent of `jev_ultrafast/scripts/check_guards.py` — real controls, no model calls.

## Known P0 gaps (what P1 fixes, with evidence)

1. **Scroll dropped** — `browser_policy.rs:112-115` `observed()` filters `!["click","fill","select"]` → scroll never reaches policy. Proof: feed `max_steps` above.
2. **`wait_for_load` burns full polls** — `browser_cdp.rs:392-446` polls every 300ms for double-stable `(text,inter,href)` up to 20s; on streaming SPAs this burns ~60 polls before `Ok`. Proof: 17–63 `wait_polls` per 3–10 steps.
3. **Per-step Laya fan-out** — `browser_policy.rs:step` can call `pick_element` → `evaluate_choice_batch(chunk=5)` multiple times per step (autocomplete + per-requirement + classify + submit/item/next). Proof: feed 10 predicts for 10 steps; form 5 for 8 steps (not 1).
4. **Typing always sends Enter** — `browser_cdp.rs:631-682` `fill` does `selectAll → insertText → Enter` unconditionally, hiding combobox suggestions (hence `re_render_new_id` had to use JS input to avoid Enter). P1 makes Enter conditional.
5. **No anti-thrash / DONE verifier** — `run_browser_loop` has no 3×-no-change blocked check, no independent `titled/summary` verification of `open/finish` before reporting `Done`. P1 adds both.

## What P1 shipped (files)

- `browser_policy.rs` — `predict_speculative` fan-out (operation + click/type/select or operation+kind), `scroll_inventory`/`needs_scroll_for_target`/`is_target_visible`, `verify_done` (titled/summary + host), single-chunk 28-char truncation, 3 new lib tests.
- `client.rs` — `predict_speculative` helper (single forward pass, fan-out).
- `snapshot.js` — `canScrollDown = scrollY+innerHeight<height-2 || omitted>0`, scroll synthetics with rect/delta.
- `browser_cdp.rs` — `wait_for_settle` (readyState + 2×rAF), `wait_for_suggestions` (200ms), `wait_for_network_idle`, `wait_for_load` deprecated for cold nav; dedicated owned tab `PUT /json/new?about:blank` + `Page.navigate`; metrics `wait_polls` now counts settle polls.
- `automation.rs` — per-step `wait_for_settle` + fill `wait_for_suggestions`, `snapshot_hash`/`is_thrashing` (3× no-change → blocked), `verify_done_independently` (synthetic wait on fail), `hyprfast_browser::same_tab_for_reuse` tolerant dedup.

Lib tests: 19→40 pass (all green). `cargo check --workspace` clean (no deprecated per-step).

## Workspace health

- `cargo test --workspace` (without `--ignored`): **40 lib pass**, **7 ignored** (4 guard + 3 bench) as designed.
- `cargo test --workspace -- --ignored --test-threads=1` — guard 4/4 pass, bench 3/3 pass (live brave+laya).
- `cargo check --workspace` / `cargo fmt --check` clean.

## What remains for P1.1 (follow-up)

1. **Scroll chaining:** feed needs 2–3 scrolls to reach item 260 (currently 1 scroll per step, then next step must see new viewport; bench shows 10 scrolls without reaching target — check `scroll_inventory` chaining and `page.omitted_after_scroll`).
2. **Settle tuning:** search timeout (1) shows 2s settle + 30ms poll is too aggressive on suggestion pages; cap at 1s or make readyState check single-poll when already `complete`.
3. **Laya operation head out-of-dist:** requirement `done` spuriously predicted even for empty fields → overridden via `type_target` fallback (works but adds predict). Retrain or remove operation head for requirement phase (use target heads only).
4. **Typing Enter discipline:** `browser_cdp::act fill` still sends unconditional `Enter` (covered by `re_render_new_id` JS workaround); make Enter conditional on `select` vs `fill`.
