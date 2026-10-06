//! Lucy runtime: session + persistent-memory composition + the turn pipeline.
//!
//! Every user command flows through one classification forward pass
//! (`classify_turn` → `decider-serve`) that yields the branch *and* the
//! reasoning tier, then the branch's own call: `answer_turn` for chat, or
//! `execute_goal_outcome` — the one ReAct entry — for act. This crate owns:
//! - ADK session persistence (`sessions`)
//! - ADK long-term memory via [`LucyAdk`] (see `memory` for the LLM-backed
//!   extractor, which is kept but not auto-invoked)
//! - the knowledge base via `lucy_knowledge` (see `knowledge` for the
//!   three-level read path and the capture pass)
//! - Local user/assistant-message persistence + the turn pipeline
pub mod agent_loop;
pub mod command;
pub mod fast_perception;
pub mod knowledge;
mod memory;
pub mod observation;
pub mod react;
mod router;
pub mod search;
mod sessions;
mod turn;
pub mod user_model;
use anyhow::{Result, anyhow};
pub use command::{
    ACTION_SKILL_BUDGET, CommandKind, CommandSource, LucyCommand, pipeline_summary,
    tier_model_family,
};
use lucy_adk::{LucyAdk, LucySessionService};
use lucy_agent::{ModelProvider, OpenAIProvider, sync_text_endpoint};
use lucy_config::LucyConfig;
use lucy_core::{
    ApprovalGate, AssistantTurn, InterruptSignal, SessionData, TokenUsage, TurnMessage,
};
use lucy_hyprfast::HyprFastCatalog;
use lucy_knowledge::{KbGetTool, KbSearchTool, KnowledgeStore};
use lucy_mcp::{McpServerConfig, McpToolDefinition, StdioMcpClient};
pub use lucy_systemone::{
    DeciderClient, DeciderHealth, DeciderProbe, DeciderQuestion, DeciderRequest, PredictOutcome,
    TurnBranch, TurnClassification, heuristic_turn_classification,
};
use lucy_systemone::{SystemAutomationEngine, SystemOneClient};
use router::{
    SkillInfo, build_tool_brief, discover_skills, parse_subtasks, render_subtasks_prompt,
    seed_lucy_skill, skill_brief_lines, skill_dirs, tool_catalog_brief,
};
pub use react::{
    DECISION_MAX_OUTPUT_TOKENS, Decision, DecisionCall, LoopbackModel, PageCheck, REACT_SYSTEM,
    ReactBudget, ReactDeps, ReactModel, ReactOutcome, ReactStats, ReactStop,
    render_react_system_prompt, run_react, run_react_goal, run_react_outcome,
};
pub use router::{Subtask, format_subtasks};
use sessions::now;
pub use sessions::trim_history;
pub use search::{SessionEvent, SessionSearch, SessionSummary};
pub use user_model::{UserModel, UserProfile};
use std::{collections::HashMap, path::PathBuf, sync::Arc};
use tokio::sync::Mutex;
pub use turn::{
    PLAN_SYSTEM, ROUTING_CONFIDENCE_MIN, PlannedCommand, RouteSource, RoutingError, TurnRoute,
    VERIFY_INSTRUCTIONS, VERIFY_SYSTEM, VerifyOutcome, describe_model, format_plan, level_model_map,
    parse_plan, parse_verify_answer, planner_prompt, render_action_plan_prompt, render_plan_prompt,
    render_verify_prompt, summarize_output, text_model_keys, unknown_plan_tools,
};

/// Output ceiling for a chat reply.
///
/// The system prompt asks for a concise answer, and the ceiling is what makes
/// that request binding: a generation-rate-limited model streams at ~30 tok/s,
/// so an unbounded "concise" reply is minutes of the user watching a cursor.
/// 1024 tokens is several paragraphs, well past what this prompt asks for.
pub const ANSWER_MAX_OUTPUT_TOKENS: u32 = 1_024;

/// One MCP tool as the follow-up planning call will see it: the qualified
/// executable name plus its full input schema.
///
/// `server` and `tool` are what make the qualified `name` executable again.
/// The name says *how the model spells the call*; only the pair says *which
/// child process answers it and under what name that process expects*.
/// `lucy_hyprfast` merges the computer-use server's definitions into the one
/// catalog behind a `computer_use_` prefix, so a name alone cannot tell the
/// two servers apart — a call routed by name alone reaches hyprfast with a
/// tool that only computer-use has.
#[derive(Debug, Clone)]
pub struct McpToolFull {
    /// The MCP server that owns this tool: `hyprfast`, `computer_use`, or a
    /// name from `mcp.toml`.
    pub server: String,
    /// The name the owning server advertises, unqualified. This is what a
    /// `tools/call` must carry.
    pub tool: String,
    /// The qualified registry name (`mcp_<server>_<tool>`) the planner, the
    /// brief, and the executor all resolve against.
    pub name: String,
    pub description: String,
    pub schema: serde_json::Value,
}

/// Runtime session cache. Persistence and history authority live exclusively in ADK SessionService.
pub struct LucyRuntime {
    provider: Arc<dyn ModelProvider>,
    system_one: Arc<SystemOneClient>,
    automation_engine: Arc<SystemAutomationEngine>,
    session: Arc<Mutex<SessionData>>,
    session_service: Arc<LucySessionService>,
    interrupt: InterruptSignal,
    /// `RwLock` so `/auto` can flip the mode in place; `config()` hands out a
    /// snapshot for read-only callers.
    config: std::sync::RwLock<LucyConfig>,
    adk: Arc<LucyAdk>,
    /// Lucy's durable knowledge base. Plain Markdown under
    /// [`LucyConfig::knowledge_dir`] with a rebuildable SQLite index beside it;
    /// see the `knowledge` module for how a turn reads it.
    knowledge: Arc<KnowledgeStore>,
    /// Tencent-style layered memory/assets hub sharing the same knowledge root.
    memory_hub: Arc<lucy_knowledge::MemoryHub>,
    tool_brief: String,
    skills: Vec<SkillInfo>,
    mcp_tools: Vec<McpToolFull>,
    /// Launcher config for every MCP server that contributed a tool, keyed by
    /// the server name on each [`McpToolFull`]. The tool registry is rebuilt
    /// per call, and a proxy needs its command/args/env to spawn; without this
    /// an extra server's tools are advertised but not reachable.
    mcp_server_configs: HashMap<String, lucy_mcp::McpServerConfig>,
    /// One human-readable line per MCP server that failed to start, plus any
    /// `mcp.toml` read error. Surfaced at startup and by `/doctor` so a broken
    /// server is visible instead of silently missing.
    pub mcp_problems: Vec<String>,
    /// Live `lucy-hyprfast` capability catalog (also cached on disk). Used to
    /// validate planned commands and to tell destructive steps from routine ones.
    hyprfast_catalog: Option<HyprFastCatalog>,
    /// Shared approval gate: the TUI resolves prompts raised while a
    /// sequential command plan runs.
    approval: ApprovalGate,
    /// Cached `decider-serve` client. Built once so the HTTP connection pool
    /// and the discovered `/predict` path are reused across turns. `None` only
    /// when the client cannot be constructed, in which case routing falls back
    /// to heuristics.
    decider: Option<Arc<DeciderClient>>,
    /// FTS5 index over every persisted event, for cross-session search.
    session_search: Arc<SessionSearch>,
    /// Durable user preferences/facts extracted from the event stream.
    user_model: Arc<UserModel>,
}
/// Environment for the `hyprfast` MCP child.
///
/// hyprfast is the only launcher — it spawns the CDP browser from
/// `browser_open` and respawns it from the crash-recovery daemon — so this env
/// is the whole path by which the `[browser]` settings reach the browser
/// process. Anything left out here is a setting that changes nothing.
///
/// The port matters as much as the flags: the fast lane reaches the browser
/// through `browser_open` on this MCP child, so if `HYPRFAST_CDP_PORT` is not
/// passed, `browser_open` launches on hyprfast's own default while the direct
/// CDP path attaches to lucy's configured port — two different browsers, and a
/// `browser_open` that appears to succeed while lucy waits on a port nobody
/// opened. Both variables are always set, since "unset" would mean the child
/// silently disagreed with its parent.
fn hyprfast_child_env(config: &LucyConfig) -> std::collections::HashMap<String, String> {
    let mut env = std::collections::HashMap::new();
    if let Some(flags) = config.browser.launch_args_value() {
        env.insert(lucy_config::HYPRFAST_BROWSER_ARGS_ENV.to_string(), flags);
    }
    if let Some(dir) = config.browser.user_data_dir_value() {
        env.insert(lucy_config::HYPRFAST_USER_DATA_DIR_ENV.to_string(), dir);
    }
    env.insert(
        lucy_config::HYPRFAST_CDP_HOST_ENV.to_string(),
        "127.0.0.1".to_string(),
    );
    env.insert(
        lucy_config::HYPRFAST_CDP_PORT_ENV.to_string(),
        config.browser.cdp_port.to_string(),
    );
    env
}

impl LucyRuntime {
    pub async fn new() -> Result<Self> {
        let mut config = LucyConfig::load()?;
        // The legacy `[models] text_base_url` endpoint names no provider entry
        // until it is probed, so its models would never reach the dropdowns
        // and the last-resort default would be chosen blind. Probe it once
        // here — a dead endpoint just logs and leaves the old list in place.
        if let Err(e) = sync_text_endpoint(&mut config).await {
            tracing::debug!("endpoint model sync skipped: {e:#}");
        }
        let provider = Arc::new(OpenAIProvider::from_config(&config)?);
        let system_one = Arc::new(SystemOneClient::from_lucy_config(&config));
        let automation_engine = Arc::new(SystemAutomationEngine::with_browser_config(
            system_one.clone(),
            config.hyprfast.command.clone(),
            &config.browser,
        ));
        let hf_cfg = McpServerConfig {
            name: "hyprfast".into(),
            command: config.hyprfast.command.clone(),
            args: config.hyprfast.args.clone(),
            env: hyprfast_child_env(&config),
        };
        let state_dir = config
            .sessions
            .dir
            .clone()
            .or_else(|| std::env::var("LUCY_SESSIONS_DIR").ok().map(PathBuf::from))
            .unwrap_or_else(|| {
                PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
                    .join(".local/state/lucy/sessions")
            });
        let legacy_path = config
            .sessions
            .file
            .clone()
            .or_else(|| std::env::var("LUCY_SESSION_FILE").ok().map(PathBuf::from))
            .unwrap_or_else(|| {
                PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
                    .join(".local/state/lucy/session.json")
            });
        // Session store, ADK memory, and capability discovery (skills on
        // disk + MCP tool handshakes) are independent I/O: run them
        // concurrently. System-1 needs no warmup — the user starts
        // decider-serve manually and each predict() fails fast when it is
        // missing — so startup never blocks on the decision backend.
        let (
            session_open_res,
            adk_opened,
            (
                tool_brief,
                skills,
                mcp_tools,
                catalog,
                mcp_problems,
                server_configs,
            ),
        ) = tokio::join!(
            LucySessionService::open(&state_dir),
            LucyAdk::open(&state_dir),
            discover_capabilities(hf_cfg),
        );
        // Cross-session search and the user model are cheap SQLite stores
        // beside the session db; a dead path degrades to an in-memory index
        // rather than blocking startup.
        let search_path = state_dir.join("session-search.db");
        let session_search = Arc::new(
            match SessionSearch::new(&search_path).await {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(error=%e, "session search db unavailable; using in-memory index");
                    SessionSearch::new(":memory:").await?
                }
            },
        );
        let user_model_path = state_dir.join("user-model.db");
        let user_model = Arc::new(
            match UserModel::new(&user_model_path).await {
                Ok(m) => m,
                Err(e) => {
                    tracing::warn!(error=%e, "user model db unavailable; using in-memory store");
                    UserModel::new(":memory:").await?
                }
            },
        );
        // Knowledge is a plain directory of Markdown files, so opening it is
        // cheap and cannot meaningfully fail — but it is off the startup
        // critical path anyway, since nothing waits on it to answer a turn.
        let knowledge_opened = KnowledgeStore::open(config.knowledge_dir()).await;
        let memory_hub_opened = lucy_knowledge::MemoryHub::open(config.knowledge_dir()).await;
        let approvals_mode = config.approval_mode().to_owned();
        let approvals_allow = config.always_allow().to_vec();
        // Built from a clone so the config can still be moved into the runtime.
        let runtime_config = config.clone();
        let session_service = Arc::new(session_open_res?);
        let _ = session_service.import_legacy_file(&legacy_path).await;
        let session = session_service
            .ensure_current(config.sessions.resume)
            .await?;
        let adk = Arc::new(adk_opened);
        let knowledge = Arc::new(knowledge_opened);
        let memory_hub = Arc::new(memory_hub_opened);
        // Register discovered skills as reusable Memory Hub assets. Skills remain
        // file-backed and executable through the existing skill subsystem; the hub
        // only makes them discoverable alongside Wiki/CodeGraph/Chat Memory.
        for skill in &skills {
            let _ = memory_hub.add_asset(
                lucy_knowledge::AssetKind::Skill,
                &skill.name,
                &skill.description,
                "skills",
                "local",
                "private",
            ).await;
        }
        Ok(Self {
            provider,
            system_one,
            automation_engine,
            session: Arc::new(Mutex::new(session)),
            session_service,
            interrupt: InterruptSignal::new(),
            config: std::sync::RwLock::new(config),
            adk,
            knowledge,
            memory_hub,
            tool_brief,
            skills,
            mcp_server_configs: server_configs
                .into_iter()
                .map(|c| (c.name.clone(), c))
                .collect(),
            mcp_tools,
            mcp_problems,
            hyprfast_catalog: catalog,
            approval: ApprovalGate::with_mode_and_allow(
                tokio::sync::mpsc::unbounded_channel::<lucy_core::AgentEvent>().0,
                approvals_mode,
                approvals_allow,
            ),
            decider: match turn::decider(&runtime_config) {
                Ok(client) => Some(Arc::new(client)),
                Err(e) => {
                    tracing::warn!("{e:#} — classification requests will fall back to heuristics");
                    None
                }
            },
            session_search,
            user_model,
        })
    }
    pub fn interrupt(&self) {
        self.interrupt.fire()
    }
    pub fn config(&self) -> LucyConfig {
        self.config
            .read()
            .map(|c| c.clone())
            .unwrap_or_else(|_| LucyConfig::default())
    }

    /// Run `f` against the live config under a write lock.
    pub fn with_config_mut<R>(&self, f: impl FnOnce(&mut LucyConfig) -> R) -> R {
        let mut guard = self
            .config
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        f(&mut guard)
    }
    pub fn usage(&self) -> TokenUsage {
        self.provider.usage()
    }
    pub fn adk_memory_enabled(&self) -> bool {
        self.adk.memory_enabled()
    }

    /// The knowledge store, always available. A disabled or unavailable store
    /// answers every query with nothing rather than failing the turn.
    pub fn knowledge_store(&self) -> Arc<KnowledgeStore> {
        self.knowledge.clone()
    }

    /// Shared layered memory and portable knowledge assets.
    pub fn memory_hub(&self) -> Arc<lucy_knowledge::MemoryHub> {
        self.memory_hub.clone()
    }

    pub fn knowledge_enabled(&self) -> bool {
        self.config().knowledge_enabled()
    }

    /// The live knowledge config, with its budgets clamped to what the store
    /// will actually render.
    pub fn knowledge_config(&self) -> lucy_config::KnowledgeConfig {
        let cfg = self.config().knowledge.clone();
        let (digest, recall) = knowledge::effective_budgets(&cfg);
        lucy_config::KnowledgeConfig {
            digest_budget_chars: digest,
            recall_budget_chars: recall,
            ..cfg
        }
    }

    /// The knowledge read tools, for the tool registry. Read-only and
    /// approval-free: neither can change the machine.
    pub fn register_knowledge_tools(&self, registry: &mut lucy_tools::ToolRegistry) {
        registry.register(KbSearchTool::new(self.knowledge.clone()));
        registry.register(KbGetTool::new(self.knowledge.clone()));
        registry.register(lucy_knowledge::tools::MemorySearchTool::new(self.memory_hub.clone()));
        registry.register(lucy_knowledge::tools::MemoryRecallTool::new(self.memory_hub.clone(), self.knowledge.clone()));
        registry.register(lucy_knowledge::tools::MemoryStatusTool::new(self.memory_hub.clone()));
        registry.register(lucy_knowledge::tools::MemoryAssetsTool::new(self.memory_hub.clone()));
        registry.register(lucy_knowledge::tools::MemorySlimTool::new(self.memory_hub.clone()));
        registry.register(lucy_knowledge::tools::CodeGraphTool::new(self.memory_hub.clone()));
        registry.register(lucy_knowledge::tools::MemoryBindTool::new(self.memory_hub.clone()));
    }

    /// The model the capture pass runs on.
    ///
    /// The capture call is background work on a turn that has already been
    /// answered, so it uses the cheap tier rather than the planner's model. A
    /// memory extractor that needs a frontier model to notice "I prefer Rust"
    /// is not earning its cost.
    pub fn capture_model(&self) -> lucy_agent::ModelTarget {
        let key = self
            .config()
            .resolve_level_model(lucy_config::ReasoningLevel::L2);
        self.target_for(&key)
            .or_else(|_| self.target_for(&self.config().default_text_model()))
            .unwrap_or_else(|_| {
                // Last resort: the provider's own default. `ModelTarget` needs
                // an endpoint, and the provider does not expose one, so a
                // loopback default is used — the same shape the agent loop
                // falls back to when its own target resolution fails.
                lucy_agent::ModelTarget::new(
                    "http://127.0.0.1:11435/v1",
                    None,
                    self.provider.model(),
                )
            })
    }

    fn capture_provider(&self) -> Arc<dyn ModelProvider> {
        self.provider.clone()
    }
    pub fn system_one(&self) -> &Arc<SystemOneClient> {
        &self.system_one
    }
    /// The slow lane as a trait object, so the agent loop can be driven by a
    /// stub in tests without changing the runtime's concrete provider.
    pub fn provider_dyn(&self) -> &dyn ModelProvider {
        self.provider.as_ref()
    }
    /// The shared interrupt, cloned into anything that must observe a cancel.
    pub fn interrupt_signal(&self) -> InterruptSignal {
        self.interrupt.clone()
    }
    /// The shared approval gate, cloned so a run can point it at its own
    /// event channel without disturbing the runtime's copy.
    pub fn approval_gate_handle(&self) -> ApprovalGate {
        self.approval.clone()
    }
    pub fn automation_engine(&self) -> &Arc<SystemAutomationEngine> {
        &self.automation_engine
    }
    pub async fn search_memory(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<lucy_adk::adk_memory::MemoryEntry>> {
        self.adk.search_memory(query, limit).await
    }
    /// Cross-session FTS5 search over the persisted event log.
    pub async fn search_sessions(&self, query: &str, limit: usize) -> Vec<SessionSummary> {
        self.session_search.search(query, limit).await
    }
    /// The durable user model (preferences/facts extracted from events).
    pub fn user_model(&self) -> Arc<UserModel> {
        self.user_model.clone()
    }
    /// The session search index itself, for indexing or rebuilds.
    pub fn session_search(&self) -> Arc<SessionSearch> {
        self.session_search.clone()
    }
    pub async fn remember_fact(&self, fact: &str) -> Result<()> {
        self.adk.remember_fact(fact).await
    }
    /// Persist a user message locally (no LLM call).
    /// The turn is appended to the session cache and the ADK event log,
    /// with history trimmed to `max_history`.
    pub async fn save_user_message(&self, prompt: String) -> Result<()> {
        self.interrupt.reset();
        let mut s = self.session.lock().await;
        if s.title.trim().is_empty() || s.title == "untitled" {
            s.title = SessionData::autotitle_from(&prompt);
            let _ = self
                .session_service
                .update_title(&s.session_id, s.title.clone())
                .await;
        }
        let message = TurnMessage::User(prompt);
        let event = adk_event_for_turn(&message);
        let indexable = message.clone();
        s.history.push(message);
        trim_history(&mut s.history, self.config().sessions.max_history);
        s.updated_at = now();
        let owner_id = s.session_id.clone();
        drop(s);
        let _ = self.session_service.save_event(&owner_id, event).await;
        // Cross-session search + user modeling observe every persisted turn,
        // never a branch on which task it was.
        let session_event = SessionEvent::from_turn(&owner_id.0.to_string(), &indexable);
        if let Err(e) = self.session_search.index_event(&session_event).await {
            tracing::warn!(error=%e, "session search indexing failed");
        }
        if let Err(e) = self.user_model.update(&session_event).await {
            tracing::warn!(error=%e, "user model update failed");
        }
        // L0 is raw conversation memory: durable, searchable on demand, and
        // deliberately not injected wholesale into every prompt.
        let _ = self.memory_hub.remember_conversation("user", &indexable_text(&indexable), &owner_id.0.to_string()).await;
        Ok(())
    }
    /// Persist an assistant reply locally (no LLM call).
    pub async fn save_assistant_text(&self, text: String) -> Result<()> {
        let message = TurnMessage::Assistant(AssistantTurn {
            text: Some(text),
            tool_calls: Vec::new(),
        });
        let event = adk_event_for_turn(&message);
        let indexable = message.clone();
        let mut s = self.session.lock().await;
        s.history.push(message);
        trim_history(&mut s.history, self.config().sessions.max_history);
        s.updated_at = now();
        let owner_id = s.session_id.clone();
        drop(s);
        let _ = self.session_service.save_event(&owner_id, event).await;
        let session_event = SessionEvent::from_turn(&owner_id.0.to_string(), &indexable);
        if let Err(e) = self.session_search.index_event(&session_event).await {
            tracing::warn!(error=%e, "session search indexing failed");
        }
        let _ = self.memory_hub.remember_conversation("assistant", &indexable_text(&indexable), &owner_id.0.to_string()).await;
        Ok(())
    }
    /// **Step 1 + Step 2.** Route a turn through the classification model and
    /// return the concrete routing decision (branch, tier, target model).
    /// `Err` when routing has no verdict (decider down / undecided): the
    /// caller shows the error and runs no tools.
    pub async fn route_turn(
        &self,
        prompt: &str,
    ) -> std::result::Result<TurnRoute, RoutingError> {
        turn::route_turn(self, prompt, None).await
    }

    /// **Step 1.** The single classification forward pass on its own, for
    /// callers that want to display the verdict before routing.
    pub async fn classify_turn(
        &self,
        prompt: &str,
    ) -> std::result::Result<TurnClassification, RoutingError> {
        turn::classify_turn(self, prompt).await
    }

    /// **Step 1b (dynamic, no keywords).** Ask the main LLM whether the
    /// request needs any tool from the live catalog. `None` means the model
    /// was unavailable or undecided.
    pub async fn verify_needs_actions(&self, prompt: &str) -> Option<bool> {
        turn::verify_needs_actions(self, prompt).await
    }

    /// The cached `decider-serve` client, if one was built at startup.
    pub fn decider(&self) -> Option<&DeciderClient> {
        self.decider.as_deref()
    }

    /// True when the classification model is configured and switched on.
    pub fn classifier_enabled(&self) -> bool {
        turn::classifier_enabled(self)
    }

    /// A fresh `decider-serve` client for ad-hoc `/health` and `/predict`
    /// calls. [`Self::decider`] returns the cached one used for turn routing.
    pub fn decider_client(&self) -> Result<DeciderClient> {
        turn::decider(&self.config())
    }

    /// The Level 3 model's ordered `hyprfast` command plan (empty when the
    /// planner produced nothing usable).
    pub async fn plan_commands(&self, goal: &str) -> Vec<PlannedCommand> {
        turn::plan_commands(self, goal).await
    }

    /// Run a plan's commands one by one in order, streaming status per step.
    pub async fn execute_plan_sequentially(
        &self,
        goal: &str,
        plan: &[PlannedCommand],
        event_tx: Option<tokio::sync::mpsc::UnboundedSender<lucy_core::AgentEvent>>,
    ) -> Result<String> {
        turn::execute_plan_sequentially(self, goal, plan, event_tx).await
    }

    /// The tool registry used to validate and run planned commands.
    pub async fn tool_registry(&self) -> Result<lucy_tools::ToolRegistry> {
        turn::tool_registry(self).await
    }

    /// The whole tool universe as one OpenAI `tools` array: local tools, the
    /// knowledge read tools when knowledge is on, every hyprfast capability,
    /// and every configured MCP server's tools.
    ///
    /// Built from the same sources the registry is, so what the model is
    /// offered and what a tool call can dispatch are the same set — see
    /// [`turn::openai_tools`] for the shape and the filtering rules.
    pub fn openai_tools(&self) -> Vec<serde_json::Value> {
        turn::openai_tools(
            &turn::local_tool_definitions(self),
            &self.mcp_tools,
            &self.destructive_tool_names(),
        )
    }

    /// The reasoning level the Level 3 planner should use.
    pub fn planner_model_key(&self) -> String {
        turn::planner_model_key(self)
    }

    /// The exact prompt [`Self::plan_commands`] sends: instructions, operating
    /// rules, the capped skill body, and the hint-first tool brief. Inspection
    /// only — no LLM call.
    pub async fn planner_prompt(&self, goal: &str) -> String {
        turn::planner_prompt(self, goal).await
    }

    /// Live hyprfast capability catalog, when the handshake succeeded.
    pub fn hyprfast_catalog(&self) -> Option<&HyprFastCatalog> {
        self.hyprfast_catalog.as_ref()
    }

    /// [`Self::route_turn`], reusing a verdict the caller already displayed.
    pub async fn route_turn_with(
        &self,
        prompt: &str,
        precomputed: Option<TurnClassification>,
    ) -> std::result::Result<TurnRoute, RoutingError> {
        turn::route_turn(self, prompt, precomputed).await
    }

    /// **Step 2A.** The command needs only a text response: call the model
    /// bound to the routed tier and return the reply.
    pub async fn answer_turn(&self, prompt: &str, route: &TurnRoute) -> Result<String> {
        self.answer_turn_with_knowledge(prompt, route, None).await
    }

    /// [`Self::answer_turn`] with the knowledge section attached.
    ///
    /// Chat is where a cheap model's context matters most: the reply is capped at
    /// [`ANSWER_MAX_OUTPUT_TOKENS`] and the model is a flash-tier, so the index
    /// is capped at a table of contents and recall is capped at a handful of
    /// claims. A chat turn gets knowledge, not the library.
    pub async fn answer_turn_with_knowledge(
        &self,
        prompt: &str,
        route: &TurnRoute,
        preferred_topic: Option<&str>,
    ) -> Result<String> {
        if self.interrupt.is_set() {
            return Err(lucy_core::LucyError::Cancelled.into());
        }
        let target = self.target_for(&route.model_key)?;
        let knowledge = knowledge::prompt_section(self, prompt, preferred_topic).await;
        // No tools are bound to this call, so the prompt must not imply any.
        //
        // The old prompt listed the whole hyprfast tool set and then said, of
        // a task like 'play despacito song on yt', to "state the hyprfast
        // commands you will execute". A model with no tools cannot execute
        // anything, so obeying that instruction produced exactly the reply the
        // user saw: "I'll open YouTube, search for Despacito, click the top
        // result, and verify that it's playing" — a plan narrated as a result,
        // with nothing done. The tool list was the tell: it invited the model to
        // talk about acting instead of noticing it could not.
        let system = "You are Lucy, a concise assistant on the user's own Linux desktop. \
You are answering in chat. You have NO tools in this reply: nothing you write here can open an \
app, drive a browser, or change anything on this machine. \
If the user asked for something to be DONE on the machine rather than explained, say plainly in \
one line that it needs to be carried out and cannot be done from here — never present a plan as \
if it were under way, and never claim to have opened, clicked, played, run, or searched for \
anything. Answer questions, explanations and small talk directly. Reply concisely.";
        let reply = self
            .provider
            .complete_text_on(
                &target,
                "answer_turn",
                system,
                &format!("{prompt}{knowledge}"),
                self.interrupt.clone(),
                Some(ANSWER_MAX_OUTPUT_TOKENS),
            )
            .await
            .map_err(|e| anyhow!("{}", friendly_main_error(&e.to_string())))?;
        if reply.trim().is_empty() {
            return Err(anyhow!(
                "model '{}' returned an empty reply — check the model and API key in /settings",
                target.model
            ));
        }
        Ok(reply)
    }

    /// **The single act entry.** Every surface — `lucy act`, `lucy agent`, the
    /// TUI's `/agent`, the TUI's plain prompt, and the gateway — reaches a run
    /// through here, so no two surfaces can disagree about how a goal was
    /// executed. See [`react`] for the perceive → decide → execute → observe
    /// contract.
    ///
    /// `route` no longer chooses a path, because there is only one. It is kept
    /// in the signature because every caller already holds a [`TurnRoute`] and
    /// the model label is worth naming in the run's status line.
    pub async fn execute_goal_outcome(
        &self,
        goal: &str,
        route: Option<&TurnRoute>,
        event_tx: Option<tokio::sync::mpsc::UnboundedSender<lucy_core::AgentEvent>>,
    ) -> Result<react::ReactOutcome> {
        // The one place a run is started, so the CLI, the TUI and the gateway
        // cannot drift into running the same goal two different ways.
        if let (Some(r), Some(tx)) = (route, &event_tx) {
            let _ = tx.send(lucy_core::AgentEvent::Status {
                message: format!("ReAct · {} chooses each step", r.model_label),
            });
        }
        react::run_react_outcome(self, goal, event_tx).await
    }

    /// Alias for [`Self::execute_goal_outcome`]. `lucy agent` and the TUI's
    /// `/agent` are named after this, and a caller holding either name should
    /// not have to care that there is one loop rather than two.
    pub async fn execute_goal_agent_outcome(
        &self,
        goal: &str,
        event_tx: Option<tokio::sync::mpsc::UnboundedSender<lucy_core::AgentEvent>>,
    ) -> Result<react::ReactOutcome> {
        self.execute_goal_outcome(goal, None, event_tx).await
    }

    /// [`Self::execute_goal_outcome`] reduced to its answer, for a caller that
    /// only wants the string.
    pub async fn execute_goal_agent(
        &self,
        goal: &str,
        event_tx: Option<tokio::sync::mpsc::UnboundedSender<lucy_core::AgentEvent>>,
    ) -> Result<String> {
        react::run_react_goal(self, goal, event_tx).await
    }

    /// **Step 2B, superseded.** The blind command list: ask the Level 3 model
    /// for an ordered `hyprfast` plan, then run it one step at a time with no
    /// re-observation between steps.
    ///
    /// No entry point reaches this any more — [`Self::execute_goal_outcome`] is
    /// the single act entry and it runs the ReAct loop, which shows the model
    /// what each tool actually returned and so cannot execute a stale plan.
    /// Kept because [`Self::plan_commands`] and
    /// [`Self::execute_plan_sequentially`] are `pub` and `lucy plan-smoke`
    /// exercises the planner directly.
    pub async fn execute_goal(
        &self,
        goal: &str,
        route: Option<&TurnRoute>,
        event_tx: Option<tokio::sync::mpsc::UnboundedSender<lucy_core::AgentEvent>>,
    ) -> Result<String> {
        let plan = self.plan_commands(goal).await;
        if plan.is_empty() {
            if let Some(tx) = &event_tx {
                let _ = tx.send(lucy_core::AgentEvent::Status {
                    message: format!(
                        "planner returned no commands for '{goal}' — running the agent loop instead"
                    ),
                });
            }
            return self.execute_goal_agent(goal, event_tx).await;
        }

        if let Some(tx) = &event_tx {
            let _ = tx.send(lucy_core::AgentEvent::Status {
                message: match route {
                    Some(r) => format!("Plan: {} step(s) from {}", plan.len(), r.model_label),
                    // `lucy act` skips routing, so there is no model to name.
                    None => format!("Plan: {} step(s)", plan.len()),
                },
            });
        }
        self.execute_plan_sequentially(goal, &plan, event_tx).await
    }

    /// Resolve a stored `provider_id/model` key into a callable target,
    /// falling back to the legacy text-model endpoint.
    pub fn target_for(&self, model_key: &str) -> Result<lucy_agent::ModelTarget> {
        lucy_agent::ModelTarget::from_config(&self.config(), model_key)
            .map_err(|e| anyhow!("{e:#}"))
    }

    /// `auto_compact` is on: summarize the older turns once the live history
    /// outgrows the trigger. Returns a note for the chat log, or `None` when
    /// nothing was compacted (disabled, too short, or the summarizer failed —
    /// in which case plain trimming still applies).
    pub async fn auto_compact_if_needed(&self) -> Result<Option<String>> {
        if !self.config().auto_compact() {
            return Ok(None);
        }
        let trigger = self.config().sessions.max_history;
        let keep = lucy_adk::DEFAULT_KEEP_TURNS;
        let snapshot = self.session.lock().await.clone();
        if !lucy_adk::should_compact(&snapshot.history, trigger) {
            return Ok(None);
        }
        let provider = self.provider.clone();
        let interrupt = self.interrupt.clone();
        let summary = lucy_adk::summarize_with(&snapshot.history, keep, |prompt| {
            let p = provider.clone();
            let intr = interrupt.clone();
            async move {
                p.complete_text(
                    &p.model(),
                    "auto_compact",
                    lucy_adk::COMPACT_SYSTEM,
                    &prompt,
                    intr,
                )
                .await
                .map_err(|e| anyhow!("{e:#}"))
            }
        })
        .await;
        let mut s = self.session.lock().await;
        match summary {
            Some(text) => {
                let result = lucy_adk::apply_compaction(&mut s, &text, keep);
                Ok(Some(format!(
                    "auto-compacted {} -> {} turns",
                    result.before, result.after
                )))
            }
            None => {
                // Summarizer unavailable: fall back to a hard trim so the live
                // context still respects the configured ceiling.
                let before = s.history.len();
                if before > trigger {
                    trim_history(&mut s.history, trigger);
                    return Ok(Some(format!(
                        "auto-compact summarizer unavailable — trimmed {before} -> {} turns",
                        s.history.len()
                    )));
                }
                Ok(None)
            }
        }
    }

    /// The approval gate shared with the TUI, so a prompt raised during
    /// sequential command execution can be answered from the UI.
    pub fn approval_gate(&self) -> &ApprovalGate {
        &self.approval
    }

    /// Switch approval mode at runtime and persist it, so automode survives a
    /// restart instead of silently reverting to `write`.
    pub fn set_approval_mode(&self, mode: &str) -> anyhow::Result<String> {
        anyhow::ensure!(
            lucy_config::APPROVAL_MODES.contains(&mode),
            "unknown approval mode {mode:?} (expected never|write|always)"
        );
        let mut cfg = LucyConfig::load()?;
        cfg.set_approval_mode(mode);
        cfg.save()?;
        // Also update the in-memory copy so `/status` reflects it immediately.
        self.with_config_mut(|live| live.set_approval_mode(mode));
        self.approval.set_mode(mode);
        Ok(mode.to_owned())
    }

    /// Tool names the live hyprfast catalog marks `destructive`. Empty when no
    /// catalog is available, in which case gating falls back to the registry's
    /// own verdicts.
    pub fn destructive_tool_names(&self) -> std::collections::HashSet<String> {
        self.hyprfast_catalog
            .as_ref()
            .map(|c| {
                c.tools
                    .values()
                    .filter(|t| t.destructive)
                    .map(|t| t.name.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Persist the gate's always-allow set. Called after the user answers a
    /// prompt with "always allow", which previously only mutated memory and so
    /// was forgotten on the next launch.
    pub fn persist_always_allowed(&self) -> anyhow::Result<()> {
        let tools = self.approval.always_allowed_tools();
        let mut cfg = LucyConfig::load()?;
        cfg.set_always_allow(tools);
        cfg.save()
    }

    /// [`Self::persist_always_allowed`], but only when the gate learned
    /// something config does not already record — so a run can call it on every
    /// exit without rewriting the config file each time.
    ///
    /// `configured` is the always-allow list as it was when the run started.
    /// `Ok(false)` means there was nothing to save.
    pub fn persist_newly_allowed(&self, configured: &[String]) -> anyhow::Result<bool> {
        if !self.approval.has_unsaved_always_allow(configured) {
            return Ok(false);
        }
        self.persist_always_allowed()?;
        Ok(true)
    }

    /// Ask the main LLM to split a goal into an ordered subtask list
    /// (`prompts/subtasks.md`, JSON mode). Never fails: on any error or
    /// unusable output the goal itself becomes the single subtask.
    pub async fn decompose_goal(&self, goal: String) -> Vec<Subtask> {
        if self.interrupt.is_set() {
            return vec![Subtask::fallback(&goal)];
        }
        let rendered = render_subtasks_prompt(&goal);
        let model = self.provider.model();
        let system = "You are Lucy, an autonomous computer-use assistant. Follow the subtask instructions exactly and return only the specified JSON.";
        match self
            .provider
            .complete_json(
                &model,
                "decompose_goal",
                system,
                &rendered,
                self.interrupt.clone(),
            )
            .await
        {
            Ok(value) => {
                let subs = parse_subtasks(&value);
                if subs.is_empty() {
                    vec![Subtask::fallback(&goal)]
                } else {
                    subs
                }
            }
            Err(_) => vec![Subtask::fallback(&goal)],
        }
    }

    /// History without the just-saved user tail, so the router does not see
    /// the request twice.
    async fn history_without_tail(&self, prompt: &str) -> Vec<TurnMessage> {
        let history_snapshot = self.session.lock().await.history.clone();
        let mut hist = history_snapshot;
        if matches!(hist.last(),Some(TurnMessage::User(t)) if t==prompt) {
            hist.pop();
        }
        hist
    }

    /// Execute a decomposed plan top to bottom: each subtask runs the fast
    /// action loop with a sliced step budget. Fail-fast: later subtasks
    /// assume earlier ones, so the first failure stops the plan with the
    /// failing subtask named.
    pub async fn run_automation_with_plan(
        &self,
        goal: &str,
        subtasks: &[Subtask],
        event_tx: Option<tokio::sync::mpsc::UnboundedSender<lucy_core::AgentEvent>>,
    ) -> Result<String> {
        let total = self.config().planner.max_subtasks.min(25);
        let per_subtask = (total / subtasks.len().max(1)).max(3);
        let mut completed = 0;
        for sub in subtasks {
            if self.interrupt.is_set() {
                return Err(lucy_core::LucyError::Cancelled.into());
            }
            if let Some(tx) = &event_tx {
                let _ = tx.send(lucy_core::AgentEvent::Status {
                    message: format!(
                        "Subtask {}/{}: {}",
                        sub.index,
                        subtasks.len(),
                        sub.description
                    ),
                });
            }
            let text_gen = self.make_text_gen();
            match self
                .automation_engine
                .run_loop(&sub.description, per_subtask, event_tx.clone(), text_gen)
                .await
            {
                Ok(_) => completed += 1,
                Err(e) => {
                    return Err(anyhow!(
                        "Subtask {}/{} failed ({}): {e:#}",
                        sub.index,
                        subtasks.len(),
                        sub.description
                    ));
                }
            }
        }
        Ok(format!(
            "Goal completed: {goal} ({completed}/{} subtasks)",
            subtasks.len()
        ))
    }

    /// Field-text generator closure for the automation loop (main LLM).
    fn make_text_gen(
        &self,
    ) -> impl Fn(
        &str,
        &str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send>>
    + Send
    + Sync
    + 'static {
        let provider = self.provider.clone();
        let interrupt = self.interrupt.clone();
        let model = provider.model();
        move |goal: &str, field: &str| {
            let p = provider.clone();
            let intr = interrupt.clone();
            let m = model.clone();
            let g = goal.to_string();
            let f = field.to_string();
            Box::pin(async move {
                let system = "You are a concise field text generator for computer automation. Return ONLY the exact text string to type into the requested field, nothing else.";
                let prompt = format!("Task Goal: {g}\nTarget Field: {f}\nText to enter:");
                p.complete_text(&m, "automation_text_gen", system, &prompt, intr)
                    .await
                    .map_err(|e| anyhow!("{}", friendly_main_error(&e.to_string())))
            })
                as std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send>>
        }
    }
    /// The exact `{tool_catalog_brief}` string the Router sees.
    pub fn tool_brief(&self) -> &str {
        &self.tool_brief
    }
    /// Skills discovered at startup (`skills/*/SKILL.md`). Drop a new skill
    /// dir in and it appears here after restart — no code change.
    pub fn skills(&self) -> &[SkillInfo] {
        &self.skills
    }
    /// Full text of one skill file for the follow-up planning call.
    pub fn skill_body(&self, name: &str) -> Option<String> {
        self.skills
            .iter()
            .find(|s| s.name == name)
            .and_then(|s| s.body().ok())
    }
    /// Every MCP tool discovered at startup with its full input schema, keyed
    /// by the qualified executable name. The follow-up planning call receives
    /// exactly these schemas for the tools it selected.
    pub fn mcp_tools(&self) -> &[McpToolFull] {
        &self.mcp_tools
    }

    /// The launcher config for the MCP server that owns `server`, as
    /// discovery found it.
    ///
    /// This is what makes a tool from an extra `mcp.toml` server executable:
    /// the name alone cannot say which binary answers, and routing every MCP
    /// tool at the hyprfast child is how a discovered server ends up
    /// advertised but unrunnable.
    pub fn mcp_server_config(&self, server: &str) -> Option<&lucy_mcp::McpServerConfig> {
        self.mcp_server_configs.get(server)
    }
    pub async fn history(&self) -> Vec<TurnMessage> {
        self.session.lock().await.history.clone()
    }
}

/// Discover everything the Router brief covers, in one concurrent pass:
/// the vendored lucy skill (seeded first) plus user skills from disk,
/// local tools (always present), HyprFast MCP tools (live handshake, cached
/// catalog on failure), and extra configured MCP servers (concurrent probe
/// with timeout). Returns the brief plus the full-fidelity records the
/// follow-up planning call needs.
type Discovery = (
    String,
    Vec<SkillInfo>,
    Vec<McpToolFull>,
    Option<HyprFastCatalog>,
    Vec<String>,
    Vec<lucy_mcp::McpServerConfig>,
);

async fn discover_capabilities(hf_cfg: McpServerConfig) -> Discovery {
    if let Err(e) = seed_lucy_skill() {
        tracing::warn!(error=%e,"lucy skill seeding skipped");
    }
    let skills = discover_skills(&skill_dirs());
    let local: Vec<String> = tool_catalog_brief().lines().map(str::to_owned).collect();
    let (catalog, (extra, mcp_problems)) = tokio::join!(
        discover_hyprfast_catalog(hf_cfg.clone()),
        discover_extra_servers()
    );
    // Removed tools are stripped at discovery — not just hidden from the
    // planner — so no registry, prompt, or plan can name them even when the
    // installed hyprfast binary still advertises them. Single source of
    // truth: `lucy_hyprfast::is_removed_tool`.
    let mut live_catalog = catalog.clone();
    if let Some(c) = live_catalog.as_mut() {
        c.tools
            .retain(|_, cap| !lucy_hyprfast::is_removed_tool(&cap.name));
    }
    let mut sections: Vec<(String, Vec<(String, String)>)> = Vec::new();
    let mut mcp_tools: Vec<McpToolFull> = Vec::new();
    // The launcher config for every server that actually contributed tools,
    // so the registry can be rebuilt with the same command/args/env that
    // discovery used. A tool whose server config is missing cannot be routed,
    // so a tool that made it into `mcp_tools` must have its config here too.
    let mut server_configs: Vec<lucy_mcp::McpServerConfig> = Vec::new();
    if let Some(catalog) = live_catalog.as_ref() {
        let mut caps: Vec<&lucy_hyprfast::ToolCapability> = catalog.tools.values().collect();
        caps.sort_by(|a, b| a.name.cmp(&b.name));
        let mut tools = Vec::new();
        // The catalog is a union of two servers (see `hyprfast_server_for`),
        // and each tool has to be routed back to the one that owns it, so the
        // configs are collected per-server rather than assumed to be hyprfast.
        let mut catalog_servers: Vec<&'static str> = Vec::new();
        for c in caps {
            let name = lucy_hyprfast::full_name(&c.name);
            tools.push((name.clone(), c.description.clone()));
            let server = hyprfast_server_for(&c.name);
            if !catalog_servers.contains(&server) {
                catalog_servers.push(server);
            }
            mcp_tools.push(McpToolFull {
                server: server.to_owned(),
                tool: hyprfast_tool_for(&c.name).to_owned(),
                name,
                description: c.description.clone(),
                schema: c.input_schema.clone(),
            });
        }
        if !tools.is_empty() {
            sections.push((
                "HyprFast MCP tools (desktop/window/browser/hints/tasks)".into(),
                tools,
            ));
            for server in catalog_servers {
                server_configs.push(match server {
                    "computer_use" => lucy_hyprfast::computer_use_config(),
                    _ => hf_cfg.clone(),
                });
            }
        }
    }
    for (server, cfg, defs) in extra {
        let mut defs_sorted = defs;
        defs_sorted.sort_by(|a, b| a.name.cmp(&b.name));
        let mut tools = Vec::new();
        for d in defs_sorted {
            if lucy_hyprfast::is_removed_tool(&d.name) {
                continue;
            }
            let name = qualified_mcp_name(&server, &d.name);
            let desc = d.description.clone().unwrap_or_else(|| "MCP tool".into());
            tools.push((name.clone(), desc.clone()));
            mcp_tools.push(McpToolFull {
                server: server.clone(),
                tool: d.name.clone(),
                name,
                description: desc,
                schema: d.input_schema.clone(),
            });
        }
        if !tools.is_empty() {
            sections.push((format!("MCP server '{server}'"), tools));
            server_configs.push(cfg);
        }
    }
    mcp_tools.sort_by(|a, b| a.name.cmp(&b.name));
    let brief = build_tool_brief(local, &sections, skill_brief_lines(&skills));
    (
        brief,
        skills,
        mcp_tools,
        live_catalog,
        mcp_problems,
        server_configs,
    )
}

/// Which MCP server a catalog tool name belongs to.
///
/// `lucy_hyprfast` merges the computer-use server's definitions into the one
/// catalog behind a `computer_use_` prefix, so the catalog is a union of two
/// servers and the prefix is the only thing that says which one answers a call.
/// This is the same rule `lucy_hyprfast::full_name` encodes when it picks a
/// prefix; keeping the two in step is why both live here rather than in the
/// discovery loop.
fn hyprfast_server_for(tool_name: &str) -> &'static str {
    if tool_name.starts_with("computer_use_") {
        "computer_use"
    } else {
        "hyprfast"
    }
}

/// The name the owning server expects in a `tools/call`, i.e. the catalog name
/// without the `computer_use_` marker the merge added.
fn hyprfast_tool_for(tool_name: &str) -> &str {
    tool_name
        .strip_prefix("computer_use_")
        .unwrap_or(tool_name)
}

async fn discover_hyprfast_catalog(hf_cfg: McpServerConfig) -> Option<HyprFastCatalog> {
    match HyprFastCatalog::discover_with_definitions(hf_cfg).await {
        Ok((catalog, _, _)) => {
            let _ = catalog.save().await;
            Some(catalog)
        }
        Err(e) => {
            tracing::warn!(error=%e,"HyprFast MCP unavailable; falling back to cached catalog");
            HyprFastCatalog::load().await.ok()
        }
    }
}

/// Probe every non-builtin MCP server from `mcp.toml`.
///
/// Returns the servers that came up AND a diagnostic per server that did not.
/// The previous version returned only the successes and discarded every error
/// with `unwrap_or_default()` and a `match … { _ => None }`, so a broken or
/// misconfigured server (a deprecated Slack package, a missing token) simply
/// vanished — which is why "the Slack plugin" looked broken with no clue why.
async fn discover_extra_servers() -> (
    Vec<(String, lucy_mcp::McpServerConfig, Vec<McpToolDefinition>)>,
    Vec<String>,
) {
    match lucy_mcp::load_config() {
        Ok(servers) => probe_extra_servers(servers).await,
        Err(e) => {
            let msg = format!("mcp.toml could not be read: {e:#}");
            tracing::warn!(error=%msg, "MCP config unreadable");
            (Vec::new(), vec![msg])
        }
    }
}

/// Probe the configured servers, dropping the built-ins (already discovered)
/// and reporting every failure.
///
/// The launcher config travels back with the definitions so the tool registry
/// can be rebuilt later against the same command, args, and env. Dropping it
/// here is what made every extra server's tools callable only by name: a
/// registry built from names alone has no way to know which child process
/// answers.
async fn probe_extra_servers(
    servers: Vec<lucy_mcp::McpServerConfig>,
) -> (
    Vec<(String, lucy_mcp::McpServerConfig, Vec<McpToolDefinition>)>,
    Vec<String>,
) {
    let servers: Vec<_> = servers
        .into_iter()
        .filter(|s| {
            !s.name.eq_ignore_ascii_case("hyprfast") && !s.name.eq_ignore_ascii_case("computer_use")
        })
        .collect();
    if servers.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut probing = tokio::task::JoinSet::new();
    for server in servers {
        probing.spawn(async move {
            let name = server.name.clone();
            let cmd = server.command.clone();
            let res = tokio::time::timeout(
                std::time::Duration::from_secs(10),
                StdioMcpClient::new(server.clone()).list_tools(),
            )
            .await;
            let diagnostic = match res {
                Ok(Ok(defs)) if defs.is_empty() => Some(format!(
                    "MCP server '{name}': connected but exposed no tools"
                )),
                Ok(Ok(defs)) => return Some((name, server, defs, None)),
                Ok(Err(e)) => Some(format!(
                    "MCP server '{name}' ({cmd}) failed: {}",
                    root_cause(&e)
                )),
                Err(_) => Some(format!("MCP server '{name}' ({cmd}) timed out after 10s")),
            };
            Some((name, server, Vec::new(), diagnostic))
        });
    }
    let mut out = Vec::new();
    let mut problems = Vec::new();
    while let Some(done) = probing.join_next().await {
        let Ok(Some((name, config, defs, diagnostic))) = done else {
            continue;
        };
        if let Some(d) = diagnostic {
            tracing::warn!("{d}");
            problems.push(d);
        } else {
            out.push((name, config, defs));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    (out, problems)
}

/// The innermost message, which is the actionable part of an `anyhow` chain
/// (`failed to spawn MCP server: No such file or directory (os error 2)`).
fn root_cause(e: &anyhow::Error) -> String {
    e.chain()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join(": ")
}

fn qualified_mcp_name(server: &str, tool: &str) -> String {
    let clean = |v: &str| {
        v.chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect::<String>()
    };
    format!("mcp_{}_{}", clean(server), clean(tool))
}

/// True when the provider error looks like an auth failure (bad/expired key).
pub fn is_auth_error(msg: &str) -> bool {
    let m = msg.to_ascii_lowercase();
    m.contains("401")
        || m.contains("403")
        || m.contains("unauthorized")
        || m.contains("forbidden")
        || m.contains("invalid api key")
        || m.contains("invalid_api_key")
        || m.contains("incorrect api key")
        || m.contains("bearer") && m.contains("invalid")
        || m.contains("api key")
            && (m.contains("invalid")
                || m.contains("expired")
                || m.contains("missing")
                || m.contains("not set"))
}

/// Wrap a raw LLM error with an actionable API-key hint when it looks like an
/// auth failure. Never swallows the original message.
pub fn friendly_main_error(raw: &str) -> String {
    if is_auth_error(raw) {
        format!(
            "API key is not working ({raw}) — open Settings (Ctrl+,) and connect a provider, or export OPENCHAT_API_KEY / LUCY_MAIN_API_KEY (run: lucy config doctor)"
        )
    } else if raw.contains("API key is not set") {
        raw.to_string()
    } else {
        format!("LLM request failed: {raw}")
    }
}

fn indexable_text(message: &TurnMessage) -> String {
    match message {
        TurnMessage::User(text) => text.clone(),
        TurnMessage::Assistant(turn) => turn.text.clone().unwrap_or_default(),
        TurnMessage::Tool(tool) => tool.output.to_string(),
    }
}

fn adk_event_for_turn(message: &TurnMessage) -> adk_core::Event {
    let mut e = adk_core::Event::new(format!("lucy-{}", uuid::Uuid::new_v4()));
    match message {
        TurnMessage::User(t) => {
            e.author = "user".into();
            e.set_content(adk_core::Content::new("user").with_text(t.clone()));
        }
        TurnMessage::Assistant(t) => {
            e.author = "lucy".into();
            e.set_content(
                adk_core::Content::new("assistant").with_text(t.text.clone().unwrap_or_default()),
            );
        }
        TurnMessage::Tool(t) => {
            e.author = "lucy".into();
            e.set_content(adk_core::Content::new("tool").with_text(t.output.to_string()));
        }
    }
    e
}

#[cfg(test)]
mod browser_launch_env_tests {
    use super::hyprfast_child_env;
    use lucy_config::{HYPRFAST_BROWSER_ARGS_ENV, LucyConfig};

    #[test]
    fn browser_launch_flags_reach_the_hyprfast_child() {
        // Rule under test: one `[browser] launch_args` list governs both the
        // launch lucy performs and the launch hyprfast performs.
        let mut cfg = LucyConfig::default();
        cfg.browser.launch_args = vec!["--disable-gpu".into()];
        let env = hyprfast_child_env(&cfg);
        assert_eq!(
            env.get(HYPRFAST_BROWSER_ARGS_ENV).map(String::as_str),
            Some("--disable-gpu")
        );
    }

    #[test]
    fn an_explicitly_empty_flag_list_means_no_variable() {
        // An empty env is the point: hyprfast must fall back to the browser's
        // own defaults rather than read an empty variable as a flag. Clearing
        // the list is the only way to ask for that now that the default is
        // populated.
        let mut cfg = LucyConfig::default();
        cfg.browser.launch_args = vec![];
        assert!(hyprfast_child_env(&cfg).is_empty());
    }

    #[test]
    fn a_fresh_config_reaches_the_hyprfast_child() {
        // The rule that matters in practice: the GPU-off default has to
        // arrive at the launcher that actually spawns the browser, or it is
        // just a config value that reads like it did something.
        let env = hyprfast_child_env(&LucyConfig::default());
        assert_eq!(
            env.get(HYPRFAST_BROWSER_ARGS_ENV).map(String::as_str),
            Some("--disable-gpu")
        );
    }
}

#[cfg(test)]
mod mcp_discovery_tests {
    use super::*;

    /// Which server answers a catalog tool, and what it is called there.
    ///
    /// `lucy_hyprfast` merges the computer-use server's definitions into the
    /// one catalog behind a `computer_use_` prefix, so these two functions are
    /// the only things that can tell a `computer_use_*` entry apart from a
    /// hyprfast one. Getting them wrong does not fail a build — it routes a
    /// call to a child process that never had the tool, which surfaces as a
    /// model call that quietly does nothing.
    #[test]
    fn a_catalog_name_routes_to_the_server_that_owns_it() {
        // Table over the general rule, not one anecdote: any hyprfast tool
        // stays with hyprfast, any `computer_use_`-prefixed one belongs to
        // computer-use, and the call name is what that server advertises.
        for (catalog_name, server, tool) in [
            ("browser_navigate", "hyprfast", "browser_navigate"),
            ("hint_act", "hyprfast", "hint_act"),
            ("hypr", "hyprfast", "hypr"),
            (
                "computer_use_left_click",
                "computer_use",
                "left_click",
            ),
            (
                "computer_use_screenshot",
                "computer_use",
                "screenshot",
            ),
        ] {
            assert_eq!(hyprfast_server_for(catalog_name), server, "{catalog_name}");
            assert_eq!(hyprfast_tool_for(catalog_name), tool, "{catalog_name}");
        }
    }

    #[test]
    fn the_routed_server_matches_the_qualified_name_the_planner_sees() {
        // The two must agree or the model is handed a name the executor
        // resolves to a different child: `full_name` picks the prefix, and
        // `hyprfast_server_for` picks the config. Same input, same answer.
        for catalog_name in [
            "browser_navigate",
            "computer_use_left_click",
            "desktop",
        ] {
            let qualified = lucy_hyprfast::full_name(catalog_name);
            assert_eq!(
                qualified,
                format!(
                    "mcp_{}_{}",
                    hyprfast_server_for(catalog_name),
                    hyprfast_tool_for(catalog_name)
                ),
                "{catalog_name}"
            );
        }
    }

    /// Point `lucy_mcp::load_config` at a per-test throwaway file.
    ///
    /// The env var is process-wide and these tests run concurrently, so each
    /// one needs its own path — sharing one file made the first test's contents
    /// leak into the others.
    fn mcp_config(name: &str, text: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("lucy-mcp-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join(format!("{name}.toml"));
        std::fs::write(&path, text).expect("write mcp.toml");
        // SAFETY: each test sets this before its own single read, and the
        // tests that would race are given distinct files.
        unsafe { std::env::set_var("LUCY_MCP_CONFIG", &path) };
        path
    }

    fn server(name: &str, command: &str) -> lucy_mcp::McpServerConfig {
        lucy_mcp::McpServerConfig {
            name: name.into(),
            command: command.into(),
            args: vec![],
            env: Default::default(),
        }
    }

    /// The regression that matters: a server that cannot start must be
    /// REPORTED. It used to be dropped by `match … { _ => None }`, which is
    /// why a broken Slack entry looked simply "not installed".
    #[tokio::test]
    async fn a_server_that_fails_to_start_is_reported_not_swallowed() {
        let (ok, problems) =
            probe_extra_servers(vec![server("slack", "definitely-not-a-real-binary-xyz")]).await;
        assert!(ok.is_empty(), "a dead command must not report tools");
        assert_eq!(problems.len(), 1, "exactly one diagnostic: {problems:?}");
        let p = &problems[0];
        assert!(p.contains("slack"), "{p}");
        assert!(
            p.contains("failed") || p.contains("timed out"),
            "must name the failure mode: {p}"
        );
    }

    #[tokio::test]
    async fn an_unreadable_mcp_toml_is_reported() {
        mcp_config("corrupt", "this is not = valid toml [[[");
        let (ok, problems) = discover_extra_servers().await;
        assert!(ok.is_empty());
        assert!(!problems.is_empty(), "a parse error must be surfaced");
        assert!(
            problems[0].contains("mcp.toml"),
            "must name the file: {:?}",
            problems[0]
        );
    }

    #[tokio::test]
    async fn builtin_servers_are_not_double_probed() {
        let (ok, problems) = probe_extra_servers(vec![
            server("hyprfast", "definitely-not-a-real-binary-xyz"),
            server("computer_use", "definitely-not-a-real-binary-xyz"),
        ])
        .await;
        assert!(ok.is_empty());
        assert!(
            problems.is_empty(),
            "hyprfast/computer_use are built in and must be skipped, not reported: {problems:?}"
        );
    }
}
