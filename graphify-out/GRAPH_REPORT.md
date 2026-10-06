# Graph Report - lucy  (2026-09-26)

## Corpus Check
- 62 files · ~70,405 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1330 nodes · 3223 edges · 58 communities (51 shown, 7 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 68 edges (avg confidence: 0.81)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Session State & Store (lucy-adk)
- Laya Daemon Bridge & Predict
- HyprFast Tool Catalog & Routing
- SystemOne Automation Loop & Bench
- Configuration & Doctor
- Speech-to-Text (Groq STT)
- Browser Policy & Speculative Steps
- LLM Provider & Memory Extraction
- ADK Memory Curation & Search
- Browser CDP Client
- MCP Config & Stdio Transport
- Skill Discovery & Temp Skills
- CLI Entrypoint & Commands
- TUI Views & Rendering
- TUI App State Model
- Runtime Router LLM Calls
- HyprFast Browser Action Bridge
- Core Event & Turn Types
- Harness v2 Architecture Design
- Capability Discovery & Tool Schemas
- TUI Run Loop & Voice Entry
- Tool Execution Dispatch
- Subtask Planning & Skill Briefs
- TUI Slash Commands
- Guard Probes & Settle Timing
- ADK Execution & Event Mapping
- Tool Registry & Definitions
- Static Fixtures & Bench Tasks
- Guard Probe Test Harness
- Core Config & Serialization
- Tool Surface, Skills & Destructive Gate
- Runtime Automation & Text-Gen
- Approval Gate
- Glob Tool & Path Matching
- Cancellation / Interrupt Signal
- Voice Recording
- LLM Call Budget & Prompt Design
- Workspace Crate Dependency Graph
- Agent Pipeline & Subtask Planner
- Tool Provenance & Shell Risk Policy
- CI Workflow & Dev Commands
- Modular Workspace Architecture
- Tool Registry Defaults
- Chat Message Types
- v1 vs v2 Harness Trace Analysis
- Feed Fixture Stress & 250-Cap
- Shell Tool
- ListDir Tool
- MCP Stub Tool
- ReadFile Tool
- SearchFiles Tool
- EditFile Tool
- Git Tool
- WriteFile Tool
- Branch Cleanup Workflow
- WhatsApp Web Protocol

## God Nodes (most connected - your core abstractions)
1. `LucyConfig` - 51 edges
2. `SystemOneClient` - 51 edges
3. `LucyRuntime` - 44 edges
4. `App` - 43 edges
5. `BrowserCdpClient` - 32 edges
6. `LucySessionService` - 29 edges
7. `SessionData` - 28 edges
8. `OpenAIProvider` - 27 edges
9. `HyprFastCatalog` - 27 edges
10. `SystemAutomationEngine` - 25 edges

## Surprising Connections (you probably didn't know these)
- `Fixed goal_statement + success_condition Contract` --semantically_similar_to--> `Immutable Session Goal Object`  [INFERRED] [semantically similar]
  prompts/router.md → docs/lucy-harness-architecture.md
- `Per-Subtask success_condition` --semantically_similar_to--> `Immutable Session Goal Object`  [INFERRED] [semantically similar]
  prompts/subtasks.md → docs/lucy-harness-architecture.md
- `Search Fixture: SPA with Stale Node IDs` --semantically_similar_to--> `Ambiguity Policy (Kept, Tightened)`  [INFERRED] [semantically similar]
  crates/lucy-systemone/tests/fixtures/search.html → docs/lucy-harness-architecture.md
- `Lucy Router Prompt` --semantically_similar_to--> `Router / Intent Classifier (Call 1)`  [INFERRED] [semantically similar]
  prompts/router.md → docs/lucy-harness-architecture.md
- `ADK-Rust Optional Capability Layer` --conceptually_related_to--> `Lucy - Your AI Computer Buddy`  [INFERRED]
  THIRD_PARTY.md → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **v2 Harness LLM Call Pipeline** — docs_lucy_harness_architecture_router_intent_classifier, docs_lucy_harness_architecture_planner_executor, docs_lucy_harness_architecture_tool_runtime, docs_lucy_harness_architecture_deterministic_recovery_table, docs_lucy_harness_architecture_recovery_call, docs_lucy_harness_architecture_verifier [EXTRACTED 1.00]
- **lucy-systemone Static Fixture Suite** — crates_lucy_systemone_tests_fixtures_search_spa_stale_id, crates_lucy_systemone_tests_fixtures_form_accessible_contact_form, crates_lucy_systemone_tests_fixtures_feed_dense_action_grid, crates_lucy_systemone_tests_fixtures_guards_hit_test_guards [EXTRACTED 1.00]
- **Deterministic-First Escalation Ladders** — docs_lucy_harness_architecture_deterministic_preflight, docs_lucy_harness_architecture_deterministic_recovery_table, docs_lucy_harness_architecture_destructive_action_gate, skills_lucy_skill_stagehand_first_priority [INFERRED 0.75]

## Communities (58 total, 7 thin omitted)

### Community 0 - "Session State & Store (lucy-adk)"
Cohesion: 0.06
Nodes (44): adk_session_roundtrip_owns_history_and_state(), event_for_turn(), event_to_turn(), local_user_id(), LucySessionService, Arc, AsRef, Debug (+36 more)

### Community 1 - "Laya Daemon Bridge & Predict"
Cohesion: 0.05
Nodes (48): DaemonProcess, LayaDaemonBridge, Arc, BufReader, Child, ChildStdin, ChildStdout, Into (+40 more)

### Community 2 - "HyprFast Tool Catalog & Routing"
Cohesion: 0.07
Nodes (54): BTreeMap, CachedToolDefs, Capability, categorizes_tools(), classify(), command_exists(), computer_use_config(), computer_use_enabled() (+46 more)

### Community 3 - "SystemOne Automation Loop & Bench"
Cohesion: 0.06
Nodes (42): ActionSpace, Arc, F, HashMap, Into, Mutex, Option, P (+34 more)

### Community 4 - "Configuration & Doctor"
Cohesion: 0.08
Nodes (33): AppearanceConfig, ApprovalConfig, BrowserConfig, command_exists(), config_exists(), config_path(), doctor(), GeneralConfig (+25 more)

### Community 5 - "Speech-to-Text (Groq STT)"
Cohesion: 0.08
Nodes (41): AudioCapture, AudioError, AudioFrame, CaptureConfig, adk_capture_config(), adk_vad_preserves_lucy_threshold(), AdkGroqStt, collected_frame() (+33 more)

### Community 6 - "Browser Policy & Speculative Steps"
Cohesion: 0.13
Nodes (37): BrowserAction, BrowserPageSnapshot, BrowserPolicy, describe(), fold(), GoalPlan, GoalRequirement, is_field() (+29 more)

### Community 7 - "LLM Provider & Memory Extraction"
Cohesion: 0.08
Nodes (25): extracts_prose_embedded_json(), largest_balanced_object(), OpenAIProvider, Arc, Client, Mutex, Option, Result (+17 more)

### Community 8 - "ADK Memory Curation & Search"
Cohesion: 0.09
Nodes (29): Content, contains_any(), content_text(), curate_interaction(), durable_preferences_and_facts_are_saved(), explicit_instructions_are_saved(), is_tombstone(), is_transient() (+21 more)

### Community 9 - "Browser CDP Client"
Cohesion: 0.10
Nodes (24): BrowserCdpClient, collect_cdp_diagnostics(), command_exists(), Arc, AtomicU64, Duration, HashMap, Mutex (+16 more)

### Community 10 - "MCP Config & Stdio Transport"
Cohesion: 0.13
Nodes (27): computer_use_config(), def(), load_config(), McpServerConfig, McpToolProxy, read_response(), register_server(), register_server_with_defs() (+19 more)

### Community 11 - "Skill Discovery & Temp Skills"
Cohesion: 0.13
Nodes (30): brief_composes_sections_in_order(), brief_without_skills_omits_section(), build_tool_brief(), discover_skills(), discovers_skills_from_disk(), first_dir_wins_on_name_collision(), literal_json_braces_survive_render(), missing_dirs_are_ignored() (+22 more)

### Community 12 - "CLI Entrypoint & Commands"
Cohesion: 0.15
Nodes (28): act_command(), ask_command(), config_command(), decide_command(), FileMakeWriter, find_session(), init_logging(), load_stt() (+20 more)

### Community 13 - "TUI Views & Rendering"
Cohesion: 0.15
Nodes (25): Rect, cursor_row_col(), build_chat_lines(), collapses_blank_lines(), draw(), draw_active(), draw_approval_popup(), draw_help_popup() (+17 more)

### Community 14 - "TUI App State Model"
Cohesion: 0.15
Nodes (12): activity_phase_tracks_busy_until_idle(), App, ApprovalDialog, model_picker_opens_on_current_and_cycles(), model_suggestions_dedupe_and_prefer_live_label(), MsgKind, progress_feed_caps_and_dedups(), Instant (+4 more)

### Community 15 - "Runtime Router LLM Calls"
Cohesion: 0.15
Nodes (9): adk_event_for_turn(), LucyRuntime, Arc, Event, MemoryEntry, Result, Self, format_history() (+1 more)

### Community 16 - "HyprFast Browser Action Bridge"
Cohesion: 0.16
Nodes (15): act_fast(), ensure_browser_runtime(), ground(), Hint, hint_act(), hint_batch(), hint_snapshot(), HintRect (+7 more)

### Community 17 - "Core Event & Turn Types"
Cohesion: 0.17
Nodes (20): AgentEvent, AssistantTurn, ExecutionMode, InterruptMessage, InterruptSource, LucyError, Option, String (+12 more)

### Community 18 - "Harness v2 Architecture Design"
Cohesion: 0.17
Nodes (21): Batch-Tool Preference by Schema Removal, Strict JSON Schema Data Contracts, Five Load-Bearing Design Principles, Deterministic Preflight (0 LLM calls), Deterministic Recovery Table, execution_trace Object, Failure: Plan Can Silently Lose the Original Goal, Implementation Checklist for the Coding Agent (+13 more)

### Community 19 - "Capability Discovery & Tool Schemas"
Cohesion: 0.18
Nodes (12): discover_capabilities(), discover_extra_servers(), format_laya_verdict(), friendly_main_error(), is_auth_error(), McpToolFull, qualified_mcp_name(), Mutex (+4 more)

### Community 20 - "TUI Run Loop & Voice Entry"
Cohesion: 0.18
Nodes (17): cleanup(), Arc, CrosstermBackend, Option, Result, Stdout, Terminal, run() (+9 more)

### Community 21 - "Tool Execution Dispatch"
Cohesion: 0.29
Nodes (6): Path, PathBuf, ToolContext, Option, Result, Value

### Community 22 - "Subtask Planning & Skill Briefs"
Cohesion: 0.15
Nodes (14): format_subtasks(), front_matter_field(), render_subtasks_prompt(), Option, Self, String, skill_brief(), skill_brief_lines() (+6 more)

### Community 23 - "TUI Slash Commands"
Cohesion: 0.24
Nodes (16): apply_model(), command_hint(), format_tool_finish(), format_tool_start(), handle_slash(), handle_slash_offline(), refresh_sessions(), resolve_session_arg() (+8 more)

### Community 24 - "Guard Probes & Settle Timing"
Cohesion: 0.14
Nodes (17): Delegated Click Handler for Cloned Nodes, Guards Fixture: Occlusion and Node-Replacement Probes, BrowserCdpClient, Static Fixture Set (search/form/feed/guards), P0 Gap: Scroll Dropped in observed(), P0 Gap: Typing Always Sends Enter, P0 Gap: wait_for_load Burns Full Polls, Guard Probes (tests/guard_probes.rs) (+9 more)

### Community 25 - "ADK Execution & Event Mapping"
Cohesion: 0.24
Nodes (12): creates_stable_adk_identity_per_execution(), history_to_event(), LucyExecution, maps_tool_results_to_function_response(), maps_user_history_to_adk_event(), Event, Into, Option (+4 more)

### Community 26 - "Tool Registry & Definitions"
Cohesion: 0.23
Nodes (10): Send, Tool, glob_walk(), Arc, HashMap, HashSet, Path, String (+2 more)

### Community 27 - "Static Fixtures & Bench Tasks"
Cohesion: 0.14
Nodes (16): Form Fixture: Accessible Contact Form, Form Outcome Live Region, Search Outcome Live Region, Search Fixture: SPA with Stale Node IDs, Suggestion Re-Render on Every Keystroke, Bench Harness (tests/bench_tasks.rs), BrowserPolicy::step, evaluate_choice_batch (chunked Laya calls) (+8 more)

### Community 28 - "Guard Probe Test Harness"
Cohesion: 0.30
Nodes (15): cap_250_and_scroll_synthetic(), cdp_port_for(), covered_rejected(), detached_rejected(), fixtures_dir(), re_render_new_id(), F, JoinHandle (+7 more)

### Community 29 - "Core Config & Serialization"
Cohesion: 0.19
Nodes (7): Formatter, Result, Self, D, Error, Ok, S

### Community 30 - "Tool Surface, Skills & Destructive Gate"
Cohesion: 0.13
Nodes (15): Ambiguity Policy (Kept, Tightened), Destructive-Action Runtime Gate, Open X Never Means Launch a Second Window, Combined Router + Planner-Executor Prompt, browserbase/stagehand 4.0.2, CDP Browser Automation Layer, hyprfast Tool Surface (47 tools), Lucy Base Skill (single consulted skill) (+7 more)

### Community 31 - "Runtime Automation & Text-Gen"
Cohesion: 0.16
Nodes (10): discover_hyprfast_catalog(), Box, Option, Pin, Send, Sync, UnboundedSender, Fn (+2 more)

### Community 32 - "Approval Gate"
Cohesion: 0.19
Nodes (9): ApprovalDecision, ApprovalGate, Arc, HashMap, HashSet, Mutex, RwLock, UnboundedSender (+1 more)

### Community 33 - "Glob Tool & Path Matching"
Cohesion: 0.17
Nodes (7): allowed_command(), edit_preview(), glob_path_match(), glob_segment_match(), GlobTool, read_limited(), R

### Community 34 - "Cancellation / Interrupt Signal"
Cohesion: 0.20
Nodes (6): AtomicBool, InterruptSignal, AtomicU64, Debug, Default, Notify

### Community 35 - "Voice Recording"
Cohesion: 0.21
Nodes (10): is_voice_hotkey(), Recording, Arc, Instant, Option, Result, String, UnboundedSender (+2 more)

### Community 36 - "LLM Call Budget & Prompt Design"
Cohesion: 0.20
Nodes (12): Target LLM Call Budget, Call-Count Instrumentation Metric, Failure: No Native Tool Calling, Provider-Native tools Parameter and tool_calls Array, Planner-Executor (Call 2), Principle: One Call Plans, Many Tools Execute, Merging Router + Planner for Simple Tasks, chat/act Classification Step (+4 more)

### Community 37 - "Workspace Crate Dependency Graph"
Cohesion: 0.48
Nodes (12): lucy, lucy-adk, lucy-agent, lucy-config, lucy-core, lucy-hyprfast, lucy-mcp, lucy-runtime (+4 more)

### Community 38 - "Agent Pipeline & Subtask Planner"
Cohesion: 0.22
Nodes (11): Grounding in the Request's Own Words, Per-Subtask success_condition, Sequential and Cumulative Subtask Assumption, Lucy Subtask Planner Prompt, Agent Architecture Pipeline, Context Builder, Dependency Scheduler, Lucy - Your AI Computer Buddy (+3 more)

### Community 39 - "Tool Provenance & Shell Risk Policy"
Cohesion: 0.24
Nodes (11): lucy-tools (native tool registry and built-in tools), jcode Foundation, MIT License and Third-Party Attribution, Local Tools (shell/read_file/write_file/edit_file/list_dir/search_files/git/glob), Shell Dangerous-Command Guard (LUCY_ALLOW_DANGEROUS), zavora-ai/adk-rust 2.2.0 (Apache-2.0), ADK-Rust Optional Capability Layer, 1jehuang/jcode (MIT, Copyright 2025 Jeremy Huang) (+3 more)

### Community 40 - "CI Workflow & Dev Commands"
Cohesion: 0.20
Nodes (10): CI Workflow, CI check Job, libasound2-dev System Dependency, Workspace Cargo Check + Test Gate, BrowserMetrics (metrics.rs), BrowserRunReport, P0 Bench Baseline, P1 Loop Surgery (+2 more)

### Community 41 - "Modular Workspace Architecture"
Cohesion: 0.27
Nodes (10): lucy-agent (model planning and tool execution), lucy (executable CLI), lucy-core (shared domain types, tool/model interfaces, cancellation), lucy-hyprfast (HyprFast catalog, routing, strategy selection), lucy-mcp (MCP config and persistent stdio transport), lucy-runtime (session/runtime composition, hierarchical task planning), lucy-tui (Ratatui terminal interface), Modular Workspace Architecture (+2 more)

### Community 42 - "Tool Registry Defaults"
Cohesion: 0.43
Nodes (4): default_registry(), filtered_defs_hide_unrouted_mcp_tools(), Self, T

### Community 43 - "Chat Message Types"
Cohesion: 0.57
Nodes (3): ChatMsg, Self, String

### Community 44 - "v1 vs v2 Harness Trace Analysis"
Cohesion: 0.32
Nodes (8): Failure: Environment Bootstrapping Routed Through Reasoning Model, Failure: One LLM Call Per Action Plus Per Verification, Harness Architecture v2: Minimal-Call, Zero-Ambiguity Design, Observed 13-Call Trace, Principle: Verify the Outcome, Not the Step, v1 TRIAGE / Tool Selector / Closed-Loop Controller Pipeline (planner.rs), v2 Harness (crates/lucy-runtime/src/harness.rs), First Establishes Start, Last Verifies End State

### Community 45 - "Feed Fixture Stress & 250-Cap"
Cohesion: 0.40
Nodes (6): 250-Action Cap and Below-Fold Scroll Stress, Feed Fixture: Dense 300-Button Grid, Feed Outcome Live Region, Guards Outcome Live Region, Probe: cap_250_and_scroll_synthetic, snapshot.js 250-Action Cap

### Community 46 - "Shell Tool"
Cohesion: 0.33
Nodes (3): Default, Duration, ShellTool

### Community 54 - "Branch Cleanup Workflow"
Cohesion: 0.67
Nodes (3): Delete Merged PR Branches Workflow, delete-merged-branch Job, Git refs DELETE API Call

### Community 55 - "WhatsApp Web Protocol"
Cohesion: 0.67
Nodes (3): WhatsApp Web DOM Selectors, WhatsApp Fullscreen Window Rule, WhatsApp Web Keyboard Protocol

## Ambiguous Edges - Review These
- `Principle: One Call Plans, Many Tools Execute` → `Tool-Naming Split (names tools, not steps or arguments)`  [AMBIGUOUS]
  prompts/router.md · relation: conceptually_related_to

## Knowledge Gaps
- **24 isolated node(s):** `CI Workflow`, `libasound2-dev System Dependency`, `Delete Merged PR Branches Workflow`, `Git refs DELETE API Call`, `lucy (executable CLI)` (+19 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Principle: One Call Plans, Many Tools Execute` and `Tool-Naming Split (names tools, not steps or arguments)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `LucyRuntime` connect `Runtime Router LLM Calls` to `Session State & Store (lucy-adk)`, `Laya Daemon Bridge & Predict`, `Cancellation / Interrupt Signal`, `SystemOne Automation Loop & Bench`, `Configuration & Doctor`, `LLM Provider & Memory Extraction`, `ADK Memory Curation & Search`, `Capability Discovery & Tool Schemas`, `Subtask Planning & Skill Briefs`, `Runtime Automation & Text-Gen`?**
  _High betweenness centrality (0.239) - this node is a cross-community bridge._
- **Why does `LucyConfig` connect `Configuration & Doctor` to `Laya Daemon Bridge & Predict`, `Speech-to-Text (Groq STT)`, `LLM Provider & Memory Extraction`, `TUI App State Model`, `Runtime Router LLM Calls`, `Capability Discovery & Tool Schemas`?**
  _High betweenness centrality (0.195) - this node is a cross-community bridge._
- **Why does `SystemOneClient` connect `Laya Daemon Bridge & Predict` to `SystemOne Automation Loop & Bench`, `Configuration & Doctor`, `Browser Policy & Speculative Steps`, `Runtime Router LLM Calls`?**
  _High betweenness centrality (0.110) - this node is a cross-community bridge._
- **What connects `CI Workflow`, `libasound2-dev System Dependency`, `Delete Merged PR Branches Workflow` to the rest of the system?**
  _24 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Session State & Store (lucy-adk)` be split into smaller, more focused modules?**
  _Cohesion score 0.055757575757575756 - nodes in this community are weakly interconnected._
- **Should `Laya Daemon Bridge & Predict` be split into smaller, more focused modules?**
  _Cohesion score 0.054706163401815576 - nodes in this community are weakly interconnected._