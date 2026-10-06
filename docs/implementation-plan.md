# Lucy Feature Implementation Plan

**Date:** 2026-10-06  
**Goal:** Implement 8 competitive features from Hermes Agent, OpenClaw, Perry, and OpenDots

---

## TODO Checklist

### Phase 1: Foundation (Agents 1-4)
- [ ] Agent 1: Self-Improving Skills System
- [ ] Agent 2: Cross-Session Search & User Modeling
- [ ] Agent 3: Scheduled Automations & Background Work
- [ ] Agent 4: Multi-Channel Gateway (Telegram + Discord)

### Phase 2: User Experience (Agents 5-6)
- [ ] Agent 5: Desktop Companion with Screen Context
- [ ] Agent 6: Task Management & Web Page Watching

### Phase 3: Advanced (Agents 7-8)
- [ ] Agent 7: Specialist Agents
- [ ] Agent 8: Spaces & Documents

---

## Detailed Implementation Plan

### Agent 1: Self-Improving Skills System

**Source:** Hermes Agent  
**Crates:** New `lucy-skills`, modify `lucy-runtime`

**Architecture:**
```
lucy-skills/
  src/
    lib.rs          — SkillStore, SkillInfo, SkillTemplate
    creator.rs      — Pattern detection → skill creation
    improver.rs     — Outcome tracking → skill improvement
    templates.rs    — Skill template definitions
  tests/
    creator_tests.rs
    improver_tests.rs
```

**Key APIs:**
```rust
pub struct SkillStore {
    // Records skill usage outcomes
    pub async fn record_outcome(&self, skill: &str, success: bool, context: &str);
    // Creates a new skill from a repeated pattern
    pub async fn maybe_create_skill(&self, pattern: &TaskPattern) -> Option<SkillInfo>;
    // Improves existing skills based on outcomes
    pub async fn improve_skills(&self) -> Vec<SkillImprovement>;
}
```

**Integration points:**
- `lucy-runtime/src/router.rs` — hook into skill discovery
- `lucy-runtime/src/agent_loop.rs` — record outcomes after each run
- `lucy-tui/src/commands.rs` — `/skills` command to view/manage skills

**Rules from AGENTS.md:**
- No task-specific hardcoding in skill creation
- Skills are discovered from disk, not hardcoded
- The model decides which skill to use, not code

---

### Agent 2: Cross-Session Search & User Modeling

**Source:** Hermes Agent  
**Crates:** Modify `lucy-runtime`, `lucy-adk`

**Architecture:**
```
lucy-runtime/src/
  search.rs       — FTS5 session search
  user_model.rs   — User model extraction and storage
```

**Key APIs:**
```rust
pub struct SessionSearch {
    pub async fn search(&self, query: &str, limit: usize) -> Vec<SessionSummary>;
    pub async fn summarize_results(&self, results: &[SessionSummary]) -> String;
}

pub struct UserModel {
    pub async fn extract(&self, events: &[Event]) -> UserProfile;
    pub async fn update(&self, event: &Event);
    pub fn preferences(&self) -> &UserPreferences;
}
```

**Integration points:**
- `lucy-runtime/src/lib.rs` — `search_sessions()` method
- `lucy-runtime/src/turn.rs` — inject past context into prompts
- `lucy-adk/src/session.rs` — FTS5 index over events

---

### Agent 3: Scheduled Automations & Background Work

**Source:** Hermes Agent + OpenDots  
**Crates:** New `lucy-scheduler`

**Architecture:**
```
lucy-scheduler/
  src/
    lib.rs          — Scheduler, ScheduledTask, TaskStore
    cron.rs         — Cron expression parser and matcher
    executor.rs     — Background task execution
    delivery.rs     — Delivery to channels (TUI, gateway)
  tests/
    cron_tests.rs
    scheduler_tests.rs
```

**Key APIs:**
```rust
pub struct Scheduler {
    pub async fn schedule(&self, task: ScheduledTask) -> TaskId;
    pub async fn cancel(&self, id: TaskId);
    pub async fn list(&self) -> Vec<ScheduledTask>;
    pub async fn run_pending(&self);
}

pub struct ScheduledTask {
    pub schedule: CronExpression,
    pub goal: String,
    pub delivery: DeliveryConfig,
    pub enabled: bool,
}
```

**Integration points:**
- `lucy-runtime/src/lib.rs` — scheduler handle
- `lucy-tui/src/commands.rs` — `/schedule` command
- `lucy-config/src/lib.rs` — scheduler config

---

### Agent 4: Multi-Channel Gateway (Telegram + Discord)

**Source:** OpenClaw  
**Crates:** Modify `lucy-gateway`

**Architecture:**
```
lucy-gateway/src/
  channels/
    mod.rs          — Channel trait
    telegram.rs     — Telegram bot (long-polling)
    discord.rs      — Discord bot (gateway)
  session.rs        — Per-channel session routing
```

**Key APIs:**
```rust
pub trait Channel {
    async fn start(&self, gateway: Arc<GatewayState>) -> Result<()>;
    async fn send_message(&self, chat_id: &str, text: &str) -> Result<()>;
    async fn send_approval(&self, chat_id: &str, approval: &ApprovalRequest) -> Result<()>;
}

pub struct TelegramChannel { /* ... */ }
pub struct DiscordChannel { /* ... */ }
```

**Integration points:**
- `lucy-gateway/src/lib.rs` — channel registry
- `lucy-gateway/src/server.rs` — channel message routing
- `lucy-config/src/lib.rs` — channel config (tokens, allowed users)

---

### Agent 5: Desktop Companion with Screen Context

**Source:** Perry  
**Crates:** Modify `lucy-tui`, `lucy-hyprfast`

**Architecture:**
```
lucy-tui/src/
  companion.rs     — Desktop companion window
  screen_ctx.rs    — Screen capture and context injection
```

**Key APIs:**
```rust
pub struct DesktopCompanion {
    pub async fn show(&self);
    pub async fn hide(&self);
    pub async fn capture_screen(&self) -> ScreenCapture;
    pub async fn ask_about_screen(&self, question: &str) -> String;
}

pub struct ScreenCapture {
    pub image: Vec<u8>,
    pub window_title: String,
    pub screen_state: ScreenState,
}
```

**Integration points:**
- `lucy-tui/src/views.rs` — companion overlay
- `lucy-hyprfast/src/lib.rs` — screen capture tools
- `lucy-runtime/src/fast_perception.rs` — screen state

---

### Agent 6: Task Management & Web Page Watching

**Source:** Perry  
**Crates:** New `lucy-tasks`

**Architecture:**
```
lucy-tasks/
  src/
    lib.rs          — Task, TaskStore, TaskNudge
    watch.rs        — WebPageWatcher, PageDiff
    nudge.rs        — Nudge scheduler
  tests/
    task_tests.rs
    watch_tests.rs
```

**Key APIs:**
```rust
pub struct TaskStore {
    pub async fn add(&self, task: Task) -> TaskId;
    pub async fn complete(&self, id: TaskId);
    pub async fn list(&self) -> Vec<Task>;
    pub async fn nudge(&self) -> Vec<Task>; // overdue tasks
}

pub struct WebPageWatcher {
    pub async fn watch(&self, url: &str, interval: Duration);
    pub async fn check(&self) -> Vec<PageChange>;
}
```

**Integration points:**
- `lucy-runtime/src/lib.rs` — task store handle
- `lucy-tui/src/commands.rs` — `/tasks` command
- `lucy-config/src/lib.rs` — task config

---

### Agent 7: Specialist Agents

**Source:** OpenDots  
**Crates:** Modify `lucy-runtime`, `lucy-tui`

**Architecture:**
```
lucy-runtime/src/
  specialists.rs   — SpecialistAgent, SpecialistRegistry
  routing.rs      — Task-to-specialist routing
```

**Key APIs:**
```rust
pub struct SpecialistAgent {
    pub name: String,
    pub role: String,
    pub instructions: String,
    pub tools: Vec<String>,
    pub session: SessionData,
}

pub struct SpecialistRegistry {
    pub async fn register(&self, agent: SpecialistAgent);
    pub async fn route(&self, task: &str) -> Option<&SpecialistAgent>;
    pub async fn execute(&self, agent: &str, task: &str) -> Result<String>;
}
```

**Integration points:**
- `lucy-runtime/src/lib.rs` — specialist registry
- `lucy-tui/src/views.rs` — specialist selection UI
- `lucy-config/src/lib.rs` — specialist config

---

### Agent 8: Spaces & Documents

**Source:** OpenDots  
**Crates:** New `lucy-spaces`

**Architecture:**
```
lucy-spaces/
  src/
    lib.rs          — Space, Page, PageStore
    editor.rs       — Markdown editor with slash commands
    search.rs       — Page library search
  tests/
    space_tests.rs
    editor_tests.rs
```

**Key APIs:**
```rust
pub struct PageStore {
    pub async fn create_page(&self, space: &str, title: &str) -> PageId;
    pub async fn save_page(&self, id: PageId, content: &str);
    pub async fn get_page(&self, id: PageId) -> Option<Page>;
    pub async fn search(&self, query: &str) -> Vec<Page>;
    pub async fn save_conversation(&self, space: &str, messages: &[TurnMessage]) -> PageId;
}
```

**Integration points:**
- `lucy-runtime/src/lib.rs` — page store handle
- `lucy-tui/src/views.rs` — spaces and page editing UI
- `lucy-config/src/lib.rs` — spaces config

---

## Cross-Cutting Concerns

### Configuration
All new features must be configurable via `LucyConfig`:
```rust
pub struct SkillsConfig { pub enabled: bool, pub auto_create: bool, pub auto_improve: bool }
pub struct SearchConfig { pub enabled: bool, pub max_results: usize }
pub struct SchedulerConfig { pub enabled: bool, pub max_tasks: usize }
pub struct ChannelsConfig { pub telegram: Option<TelegramConfig>, pub discord: Option<DiscordConfig> }
pub struct CompanionConfig { pub enabled: bool, pub hotkey: String }
pub struct TasksConfig { pub enabled: bool, pub nudge_interval_secs: u64 }
pub struct SpecialistsConfig { pub enabled: bool, pub agents: Vec<SpecialistConfig> }
pub struct SpacesConfig { pub enabled: bool, pub default_space: String }
```

### Testing Strategy
- Unit tests for all new modules
- Integration tests with `LucyRuntime`
- TUI tests with `TestBackend`
- No regressions in existing tests

### Documentation
- Each feature gets a doc comment on its public API
- TUI commands documented in help text
- Config options documented in config template
