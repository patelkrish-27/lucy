# Lucy vs. Competitive Landscape — Feature Analysis & Implementation Plan

**Date:** 2026-10-06  
**Repos analyzed:** Hermes Agent, OpenClaw, Perry, OpenDots  
**Lucy version:** 0.1.0 (Rust workspace, 15 crates)

---

## Executive Summary

Lucy's core differentiator is its **two-speed agentic loop** (slow LLM planning + fast Decider-2B execution) with a knowledge base, MCP tool integration, and a Rust-native TUI. However, all four competitors have features Lucy lacks. This report identifies the gaps and plans the implementation.

---

## 1. Hermes Agent (nousresearch/hermes-agent) — 251k stars

### What Hermes has that Lucy doesn't

| Feature | Hermes | Lucy | Gap |
|---------|--------|------|-----|
| **Self-improving skills** | Creates skills from experience, improves them during use, nudges itself to persist knowledge | Static `SKILL.md` files, no self-improvement | **HIGH** — Skills are the primary way an agent learns; Lucy's are frozen |
| **FTS5 session search** | Searches past conversations with LLM summarization for cross-session recall | Sessions exist but no cross-session search | **HIGH** — Lucy forgets everything between sessions |
| **Cron/scheduled automations** | Built-in cron scheduler with delivery to any platform | None | **MEDIUM** — No way to run tasks on a schedule |
| **Subagent delegation** | Spawns isolated subagents for parallel workstreams | Single-agent only | **MEDIUM** — Complex tasks could be parallelized |
| **User modeling** | Honcho dialectic user modeling — builds deepening model of who you are | Basic memory extraction (ADK) | **MEDIUM** — Lucy's memory is flat facts, not a user model |
| **Trajectory compression** | Batch trajectory generation, compression for training | None | **LOW** — Research feature, not user-facing |
| **Plugin system** | Plugin catalog, ACP adapter, agentskills.io standard | MCP only | **MEDIUM** — MCP is good but no native plugin SDK |
| **Multi-platform messaging** | Telegram, Discord, Slack, WhatsApp, Signal, Email | Mobile gateway only | **HIGH** — Lucy can't reach users on messaging platforms |
| **7 terminal backends** | Local, Docker, SSH, Singularity, Modal, Daytona, Vercel | Local only | **LOW** — Lucy is desktop-focused by design |
| **Context files** | Project context that shapes every conversation | Knowledge base (different concept) | **LOW** — Lucy's knowledge base covers this differently |

### Key takeaway for Lucy
The **self-improving skills** and **cross-session search** are the two highest-value features. They make Lucy feel like she's learning and remembering, not starting fresh every session.

---

## 2. OpenClaw (openclaw/openclaw) — 391k stars

### What OpenClaw has that Lucy doesn't

| Feature | OpenClaw | Lucy | Gap |
|---------|----------|------|-----|
| **Gateway architecture** | Local control plane for sessions, tools, events, channels | Mobile gateway (WebSocket/REST) | **MEDIUM** — Lucy's gateway is simpler but functional |
| **Multi-channel** | WhatsApp, Telegram, Slack, Discord, Google Chat, Signal, iMessage, 20+ | Mobile app only | **HIGH** — Users want to reach Lucy where they already are |
| **Plugin SDK** | Native plugin system with ClawHub marketplace | MCP only | **MEDIUM** — MCP is universal but plugins are tighter |
| **Team deployment** | Shared team deployment with config-only difference | Single-user only | **LOW** — Lucy is personal by design |
| **Security: DM pairing** | Unknown senders paired by default, approval flow | Approval gate exists | **LOW** — Lucy has approvals, not pairing |
| **Sandboxing** | Container isolation for tools | None | **MEDIUM** — Lucy runs tools on host |
| **Companion apps** | Native apps for macOS, iOS, Android, Windows, Linux | Mobile gateway (web-based) | **MEDIUM** — No native app experience |
| **Telemetry controls** | Anonymous feature statistics, opt-in | None | **LOW** — Privacy-focused already |

### Key takeaway for Lucy
**Multi-channel support** (Telegram, Discord, Slack) is the highest-value feature. The **plugin SDK** would also make Lucy more extensible without requiring MCP servers.

---

## 3. Perry (TheM1N9/perry) — 31 stars

### What Perry has that Lucy doesn't

| Feature | Perry | Lucy | Gap |
|---------|-------|------|-----|
| **Subscription-based models** | Uses ChatGPT/Claude/Grok subscriptions — no API bill | API keys required | **MEDIUM** — Different business model, not directly applicable |
| **Desktop companion/pet** | Hotkey-activated desktop companion with chats, to-dos, approvals | Mascot (visual only) | **HIGH** — Lucy's mascot is decorative, not functional |
| **Screen context awareness** | "What's this error?" — hotkey shows companion the current window | Desktop tools exist but no "ask about this window" | **HIGH** — Natural UX for desktop assistant |
| **To-do list with nudges** | Keeps to-do list, nudges until done | None | **MEDIUM** — Task management is a natural assistant feature |
| **Web page watching** | Watches pages, notifies on change | None | **MEDIUM** — Proactive monitoring |
| **Auto-update** | Self-updates at night | None | **LOW** — Deployment concern |
| **Single-file data** | All data in one file in ~/.perry | Multiple stores (ADK, knowledge, sessions) | **LOW** — Lucy's architecture is more sophisticated |
| **Composio integration** | Gmail, Google Calendar, Notion via Composio | MCP servers can do this | **LOW** — MCP covers this |

### Key takeaway for Lucy
**Desktop companion with screen context** is the highest-value feature. The **to-do list with nudges** and **web page watching** are also strong additions that make Lucy feel more proactive.

---

## 4. OpenDots (CopilotKit/OpenDots) — 3.6k stars

### What OpenDots has that Lucy doesn't

| Feature | OpenDots | Lucy | Gap |
|---------|----------|------|-----|
| **Spaces/documents** | Home for working documents, searchable library, visual editor | None | **HIGH** — Lucy has no document workspace |
| **Specialist agents** | Multiple named agents with roles, instructions, tools | Single agent | **HIGH** — Different specialists for different tasks |
| **Per-agent computers** | Per-agent browser profiles, files, shell, permissions | Shared browser | **MEDIUM** — Isolation between agents |
| **Review before saving** | Human-in-the-loop approval cards before saving | Approval gate exists | **LOW** — Lucy has approvals, not review cards |
| **Background work** | Scheduled server-side turns in original conversation | None | **HIGH** — Lucy can't run background tasks |
| **AG-UI protocol** | Standard protocol for agent-interface communication | Custom events | **MEDIUM** — Standard protocol enables ecosystem |
| **Slack integration** | Managed Slack connection with allowlists | None | **MEDIUM** — Part of multi-channel |
| **Automatic Learning** | Per-Dot Learning containers, skill delivery | Knowledge capture exists | **MEDIUM** — More structured learning |
| **Pages with visual editor** | Slash commands, autosave, revision checks | None | **MEDIUM** — Document editing |
| **Voice calls** | WebRTC speech with separate compute agent | STT exists | **MEDIUM** — Full voice calls |

### Key takeaway for Lucy
**Specialist agents** and **background work** are the highest-value features. **Spaces/documents** would also make Lucy more useful as a daily driver.

---

## Prioritized Implementation Plan

### Phase 1: High-Value, Low-Risk (Week 1-2)

| # | Feature | Source | Crates Affected | Complexity |
|---|---------|--------|-----------------|------------|
| 1 | **Self-improving skills** | Hermes | `lucy-runtime`, new `lucy-skills` | Medium |
| 2 | **Cross-session search** | Hermes | `lucy-runtime`, `lucy-adk` | Medium |
| 3 | **Scheduled automations** | Hermes/OpenDots | new `lucy-scheduler` | Medium |
| 4 | **To-do list with nudges** | Perry | new `lucy-tasks` | Low |

### Phase 2: High-Value, Medium-Risk (Week 3-4)

| # | Feature | Source | Crates Affected | Complexity |
|---|---------|--------|-----------------|------------|
| 5 | **Desktop companion with screen context** | Perry | `lucy-tui`, `lucy-hyprfast` | Medium |
| 6 | **Multi-channel gateway (Telegram, Discord)** | OpenClaw | `lucy-gateway` | High |
| 7 | **Web page watching** | Perry | new `lucy-watch` | Low |
| 8 | **Background work execution** | OpenDots | `lucy-runtime` | Medium |

### Phase 3: High-Value, Higher-Risk (Week 5-6)

| # | Feature | Source | Crates Affected | Complexity |
|---|---------|--------|-----------------|------------|
| 9 | **Specialist agents** | OpenDots | `lucy-runtime`, `lucy-tui` | High |
| 10 | **Spaces/documents** | OpenDots | new `lucy-spaces` | High |
| 11 | **Plugin SDK** | OpenClaw | new `lucy-plugins` | High |
| 12 | **Subagent delegation** | Hermes | `lucy-runtime` | High |

---

## 8 Parallel Agent Tasks

Each agent will implement one feature area. They are designed to be independent with minimal cross-agent dependencies.

### Agent 1: Self-Improving Skills System
**Source:** Hermes Agent  
**Goal:** Enable Lucy to create new skills from successful task patterns and improve existing skills based on outcomes.  
**Deliverables:**
- New `lucy-skills` crate with skill creation/improvement logic
- Skill template system (SKILL.md with front-matter)
- Integration with `lucy-runtime` to record skill usage outcomes
- Auto-skill creation when a task pattern repeats 3+ times
- Skill improvement pass that updates SKILL.md based on success/failure rates

### Agent 2: Cross-Session Search & User Modeling
**Source:** Hermes Agent  
**Goal:** Enable Lucy to search past conversations and build a user model across sessions.  
**Deliverables:**
- FTS5 index over ADK session events
- `search_sessions(query, limit)` API on `LucyRuntime`
- LLM summarization of search results for cross-session recall
- User model extraction (preferences, patterns, working style)
- Integration with `answer_turn` to inject relevant past context

### Agent 3: Scheduled Automations & Background Work
**Source:** Hermes Agent + OpenDots  
**Goal:** Enable Lucy to run tasks on a schedule and execute background work.  
**Deliverables:**
- New `lucy-scheduler` crate with cron-like scheduling
- Background task execution (fire-and-forget with status tracking)
- Integration with `lucy-runtime` for task execution
- TUI views for scheduled tasks and background work status
- Config for schedule storage and delivery preferences

### Agent 4: Multi-Channel Gateway (Telegram + Discord)
**Source:** OpenClaw  
**Goal:** Extend the Lucy gateway to support Telegram and Discord channels.  
**Deliverables:**
- Telegram bot integration (long-polling or webhook)
- Discord bot integration (gateway connection)
- Channel abstraction in `lucy-gateway` (unified message type)
- Per-channel session management
- Config for channel tokens and allowed users

### Agent 5: Desktop Companion with Screen Context
**Source:** Perry  
**Goal:** Add a functional desktop companion that can answer questions about the current screen.  
**Deliverables:**
- Desktop companion window (always-on-top, hotkey-activated)
- Screen capture + context injection into conversation
- "What's this?" feature — screenshot → LLM analysis
- Integration with `lucy-hyprfast` for screen state
- TUI view for companion mode

### Agent 6: Task Management & Web Page Watching
**Source:** Perry  
**Goal:** Add to-do list management with nudges and web page monitoring.  
**Deliverables:**
- New `lucy-tasks` crate with task CRUD
- Task nudge system (remind until done)
- Web page watcher (poll for changes, notify on diff)
- Integration with `lucy-runtime` for task execution
- TUI views for tasks and watched pages

### Agent 7: Specialist Agents
**Source:** OpenDots  
**Goal:** Enable multiple named specialist agents with different roles and tools.  
**Deliverables:**
- Specialist agent configuration (name, role, instructions, tools)
- Per-agent session and tool registry
- Agent routing (which specialist handles which task)
- TUI views for specialist selection and management
- Integration with `lucy-runtime` for multi-agent execution

### Agent 8: Spaces & Documents
**Source:** OpenDots  
**Goal:** Add a document workspace where Lucy can create, edit, and organize pages.  
**Deliverables:**
- New `lucy-spaces` crate with page CRUD
- Markdown editor with slash commands
- Page library with search
- Save conversation as page
- Integration with `lucy-runtime` for page context
- TUI views for spaces and page editing

---

## Architecture Principles (from AGENTS.md)

All implementations must follow Lucy's core rule: **no task-specific hardcoding**.

- **No site lists, app names, or task keywords** that decide what to do
- **No DOM/URL shapes** encoded as special cases
- **Models are the router** — code enforces contracts and safety
- **Capabilities are discovered** from the live tool catalog
- **Prompts the model reads** are where task knowledge belongs

Each agent must read `AGENTS.md` before starting work.

---

## Success Criteria

1. All 8 features compile and pass `cargo test`
2. No regressions in existing tests
3. Each feature has integration tests
4. TUI remains responsive with new features
5. New features are configurable (can be disabled)
6. Documentation updated for each feature
