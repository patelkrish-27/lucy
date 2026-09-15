//! Lucy runtime: builds the agent, tools and session store, then runs every
//! user command through the hierarchical loop in [`planner`].
//!
//! Module layout:
//! - `planner` — main-model triage, cheap-model command compiler, closed-loop
//!   execution with verification and recovery.
//! - `sessions` — opencode-style multi-session CRUD, compaction, trimming.
//! - `execution` — dependency-safe execution waves (scheduler foundation).

mod execution;
mod planner;
mod sessions;

pub use execution::{ExecutionWave, build_waves, has_dependency_cycle, parallel_candidate};
pub use planner::SubTask;
pub use sessions::trim_history;

use anyhow::{Result, anyhow};
use lucy_agent::{Agent, OpenAIProvider};
use lucy_config::LucyConfig;
use lucy_core::{
    AgentEvent, ApprovalDecision, ApprovalGate, AssistantTurn, ExecutionMode, InterruptSignal,
    SessionData, SessionId, SessionStore, TokenUsage, ToolContext, ToolResult, TurnMessage,
};
use lucy_hyprfast::HyprFastCatalog;
use lucy_mcp::{load_config, register_server};
use lucy_tools::{ToolRegistry, default_registry};
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::PathBuf,
    sync::Arc,
};
use tokio::sync::{Mutex, mpsc};

use planner::{
    DecisionKind, action_requires_verification, build_context, decide_next, plan_command,
    triage_request, truncate_json, verification_subtask,
};
use sessions::now;

const MAX_CONTEXT_CHARS: usize = 16_000;
const MAX_MAIN_DECISIONS: usize = 64;

pub struct LucyRuntime {
    agent: Arc<Agent<OpenAIProvider>>,
    registry: Arc<ToolRegistry>,
    provider: Arc<OpenAIProvider>,
    session: Arc<Mutex<SessionData>>,
    store: SessionStore,
    #[allow(dead_code)]
    legacy_path: PathBuf,
    interrupt: InterruptSignal,
    working_dir: PathBuf,
    hyprfast: Option<HyprFastCatalog>,
    config: LucyConfig,
    approvals: ApprovalGate,
}

impl LucyRuntime {
    pub async fn new() -> Result<Self> {
        let config = LucyConfig::load()?;
        let provider = Arc::new(OpenAIProvider::from_config(&config)?);
        let mut registry = default_registry();
        // HyprFast's LLM-backed tools (stagehand_act/observe/extract, ground)
        // need a Gemini key. Users set it in Lucy (`cheap_api_key`), but the
        // hyprfast subprocess only sees env vars / ~/.config/hyprfast/stagehand.env
        // — which goes stale (exactly what broke the YT session: stagehand.env
        // held a revoked key while Lucy's cheap key was valid). Forward Lucy's
        // key so the subprocess always uses a working one.
        let mut hf_env = std::collections::HashMap::new();
        if let Some(cheap_key) = config.cheap_api_key() {
            if !cheap_key.trim().is_empty() {
                for var in [
                    "GEMINI_API_KEY",
                    "GOOGLE_API_KEY",
                    "GOOGLE_GENERATIVE_AI_API_KEY",
                ] {
                    hf_env.insert(var.to_string(), cheap_key.clone());
                }
            }
        }
        let hf_cfg = lucy_mcp::McpServerConfig {
            name: "hyprfast".into(),
            command: config.hyprfast.command.clone(),
            args: config.hyprfast.args.clone(),
            env: hf_env,
        };
        let hyprfast = match HyprFastCatalog::discover(hf_cfg.clone()).await {
            Ok(c) => {
                tracing::info!(tools = c.len(), "HyprFast MCP connected");
                let _ = c.save().await;
                Some(c)
            }
            Err(e) => {
                tracing::warn!(error=%e,"HyprFast MCP unavailable; continuing without desktop capabilities");
                None
            }
        };
        for server in load_config()? {
            if server.name.eq_ignore_ascii_case("hyprfast") {
                continue;
            }
            if let Err(e) = register_server(&mut registry, server.clone()).await {
                tracing::warn!(server=%server.name,error=%e,"MCP server unavailable")
            }
        }
        if hyprfast.is_some() {
            if let Err(e) = register_server(&mut registry, hf_cfg).await {
                tracing::warn!(error=%e,"failed to register HyprFast MCP tools")
            }
        }
        let legacy_path = config
            .sessions
            .file
            .clone()
            .or_else(|| std::env::var("LUCY_SESSION_FILE").ok().map(PathBuf::from))
            .unwrap_or_else(|| {
                PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
                    .join(".local/state/lucy/session.json")
            });
        let store_dir = config
            .sessions
            .dir
            .clone()
            .or_else(|| std::env::var("LUCY_SESSIONS_DIR").ok().map(PathBuf::from))
            .unwrap_or_else(SessionStore::default_dir);
        let store = SessionStore::new(store_dir);
        // Opencode parity: migrate legacy single file once, then work purely in the store.
        let _ = store.migrate_legacy_file(&legacy_path).await;
        let session = if config.sessions.resume {
            match store.list().await {
                Ok(list) if !list.is_empty() => {
                    // Most recent session wins (list is newest-first).
                    match store.load(&list[0].id).await {
                        Ok(s) => s,
                        Err(_) => store.create(None).await?,
                    }
                }
                _ => store.create(None).await?,
            }
        } else {
            store.create(None).await?
        };
        let registry = Arc::new(registry);
        let agent = Arc::new(Agent::new(provider.clone(), registry.clone()));
        let (approval_tx, _approval_rx) = mpsc::unbounded_channel();
        let approvals = ApprovalGate::new(approval_tx);
        if let Ok(mut m) = approvals.mode.write() {
            *m = config.approvals.mode.clone();
        }
        Ok(Self {
            agent,
            registry,
            provider,
            session: Arc::new(Mutex::new(session)),
            store,
            legacy_path,
            interrupt: InterruptSignal::new(),
            working_dir: std::env::current_dir()?,
            hyprfast,
            config,
            approvals,
        })
    }
    pub fn interrupt(&self) {
        self.interrupt.fire()
    }
    pub fn hyprfast_catalog(&self) -> Option<&HyprFastCatalog> {
        self.hyprfast.as_ref()
    }
    pub fn route_hyprfast(&self, prompt: &str) -> Option<lucy_hyprfast::Route> {
        self.hyprfast.as_ref().map(|c| c.route(prompt))
    }
    pub fn config(&self) -> &LucyConfig {
        &self.config
    }
    pub fn approval_state(&self) -> ApprovalGate {
        self.approvals.clone()
    }
    pub fn resolve_approval(&self, call_id: &str, d: ApprovalDecision) -> bool {
        self.approvals.resolve(call_id, d)
    }
    /// Current approval mode (`never` = auto, `write` = default, `always` = paranoid).
    pub fn approval_mode(&self) -> String {
        self.approvals.approval_mode()
    }
    pub fn is_auto(&self) -> bool {
        self.approvals.is_auto()
    }
    /// Set approval mode and persist it to the config file so it survives
    /// restarts. Accepts `never|auto|on` (auto-approve), `write` (default),
    /// `always` (ask for everything). Returns the normalized mode.
    pub fn set_approval_mode(&self, mode: &str) -> anyhow::Result<String> {
        let normalized = self.approvals.set_mode(mode);
        if !["never", "write", "always"].contains(&normalized.as_str()) {
            return Err(anyhow!(
                "approvals.mode must be never|write|always (auto|default|strict also accepted)"
            ));
        }
        let mut cfg = LucyConfig::load()?;
        cfg.approvals.mode = normalized.clone();
        cfg.save()?;
        Ok(normalized)
    }
    /// `true` → auto-approve everything (`never`, no permission popups);
    /// `false` → back to default `write` prompting. Persists to config file.
    pub fn set_auto_approve(&self, auto: bool) -> anyhow::Result<String> {
        self.set_approval_mode(if auto { "never" } else { "write" })
    }
    pub fn usage(&self) -> TokenUsage {
        self.provider.usage()
    }
    pub fn set_model(&self, model: &str) -> anyhow::Result<String> {
        let name = model.trim().to_owned();
        if name.is_empty() {
            return Err(anyhow!("model name must not be empty"));
        }
        let mut cfg = LucyConfig::load()?;
        cfg.models.main = name.clone();
        cfg.save()?;
        self.provider.set_model(name.clone());
        Ok(name)
    }
    pub async fn submit(&self, prompt: String) -> Result<mpsc::UnboundedReceiver<AgentEvent>> {
        self.interrupt.reset();
        if self.session.lock().await.history.len() > self.config.sessions.max_history {
            let _ = self.compact().await;
        }
        // Auto-title untitled sessions from the first prompt (opencode behaviour).
        let session_snapshot = {
            let mut s = self.session.lock().await;
            if s.title.trim().is_empty() || s.title == "untitled" {
                s.title = SessionData::autotitle_from(&prompt);
            }
            s.touch();
            let _ = self.store.save(&s).await;
            s.clone()
        };
        let route = self.route_hyprfast(&prompt);
        if let Some(r) = &route {
            tracing::debug!(strategy=%r.strategy,candidates=r.candidates.len(),fast_path=r.fast_path,"HyprFast route selected");
        }
        // Every command flows through the same hierarchical loop: the main model
        // triages (chat vs act) with system prompt + history + new request, plans
        // subtasks, the cheap model compiles each subtask to one command, and the
        // main model executes, verifies and recovers. The general agent is only a
        // fallback when no desktop catalog (or the planner itself) is unavailable.
        let source = if self.hyprfast.is_some() {
            match self
                .plan_and_execute(
                    prompt.clone(),
                    session_snapshot.history.clone(),
                    route.clone(),
                )
                .await
            {
                Ok(rx) => rx,
                Err(e) => {
                    tracing::warn!(error=%e,"hierarchical planner failed; falling back to general agent");
                    self.agent
                        .execute_with_history_filtered(
                            prompt,
                            session_snapshot.history,
                            Some(self.working_dir.clone()),
                            self.interrupt.clone(),
                            route
                                .map(|r| r.candidates.into_iter().collect())
                                .unwrap_or_default(),
                        )
                        .await?
                }
            }
        } else {
            self.agent
                .execute_with_history_filtered(
                    prompt,
                    session_snapshot.history,
                    Some(self.working_dir.clone()),
                    self.interrupt.clone(),
                    route
                        .map(|r| r.candidates.into_iter().collect())
                        .unwrap_or_default(),
                )
                .await?
        };
        let (tx, rx) = mpsc::unbounded_channel();
        let session_store = self.session.clone();
        let store = self.store.clone();
        let max_history = self.config.sessions.max_history;
        let owner_id = session_snapshot.session_id.clone();
        tokio::spawn(async move {
            let mut source = source;
            while let Some(event) = source.recv().await {
                if let AgentEvent::History { message } = &event {
                    // If the user switched sessions mid-run, persist to the owner session file
                    // instead of corrupting the newly active conversation.
                    let current_id = session_store.lock().await.session_id.clone();
                    if current_id == owner_id {
                        let mut session = session_store.lock().await;
                        session.history.push(message.clone());
                        trim_history(&mut session.history, max_history);
                        session.updated_at = now();
                        let _ = store.save(&session).await;
                    } else {
                        if let Ok(mut owner) = store.load(&owner_id).await {
                            owner.history.push(message.clone());
                            trim_history(&mut owner.history, max_history);
                            owner.updated_at = now();
                            let _ = store.save(&owner).await;
                        }
                    }
                }
                let _ = tx.send(event);
            }
        });
        Ok(rx)
    }
    async fn plan_and_execute(
        &self,
        prompt: String,
        history: Vec<TurnMessage>,
        route: Option<lucy_hyprfast::Route>,
    ) -> Result<mpsc::UnboundedReceiver<AgentEvent>> {
        let (tx, rx) = mpsc::unbounded_channel();
        let provider = self.provider.clone();
        let registry = self.registry.clone();
        let catalog = self
            .hyprfast
            .clone()
            .ok_or_else(|| anyhow!("HyprFast catalog unavailable"))?;
        let interrupt = self.interrupt.clone();
        let working_dir = self.working_dir.clone();
        let planner_cfg = self.config.planner.clone();
        let verify_actions = self.config.hyprfast.verify_actions;
        let main_model = self.provider.model();
        let command_model = self.config.models.hyprfast_command.clone();
        let approvals = self.approvals.clone();
        tokio::spawn(async move {
            let run = async {
                let _ = tx.send(AgentEvent::History {
                    message: TurnMessage::User(prompt.clone()),
                });
                let _ = tx.send(AgentEvent::Status {
                    message: "Understanding your request…".into(),
                });
                let _ = tx.send(AgentEvent::Progress {
                    message: "Understanding your request…".into(),
                });
                // Step 1 — main model triage: answer directly (chat) or break down (act).
                let triage = triage_request(
                    &provider,
                    &main_model,
                    &prompt,
                    &catalog,
                    &route,
                    &history,
                    &interrupt,
                )
                .await?;
                if triage.mode == "chat" {
                    let text = triage
                        .reply
                        .filter(|r| !r.trim().is_empty())
                        .unwrap_or_else(|| "Done.".to_string());
                    let assistant = TurnMessage::Assistant(AssistantTurn {
                        text: Some(text.clone()),
                        tool_calls: Vec::new(),
                    });
                    let _ = tx.send(AgentEvent::History { message: assistant });
                    let _ = tx.send(AgentEvent::TextDelta { text });
                    Ok::<(), anyhow::Error>(())
                } else if triage.mode != "act" {
                    return Err(anyhow!(
                        "main model returned unknown triage mode: {}",
                        triage.mode
                    ));
                } else {
                    if triage.subtasks.is_empty()
                        || triage.subtasks.len() > planner_cfg.max_subtasks
                    {
                        return Err(anyhow!("main model returned an invalid subtask count"));
                    }
                    let _ = tx.send(AgentEvent::Progress {
                        message: format!(
                            "Plan ready: {} step{}…",
                            triage.subtasks.len(),
                            if triage.subtasks.len() == 1 { "" } else { "s" }
                        ),
                    });
                    let mut queue: VecDeque<SubTask> = triage.subtasks.into_iter().collect();
                    let mut completed: HashMap<String, Value> = HashMap::new();
                    let mut notes = Vec::new();
                    let mut decisions = 0usize;
                    let mut done_count = 0usize;
                    while let Some(subtask) = queue.pop_front() {
                        if interrupt.is_set() {
                            return Err(lucy_core::LucyError::Cancelled.into());
                        }
                        if decisions >= MAX_MAIN_DECISIONS {
                            return Err(anyhow!(
                                "main model exceeded computer-operation decision budget"
                            ));
                        }
                        let step_no = done_count + 1;
                        let _ = tx.send(AgentEvent::Status {
                            message: format!("Step {step_no}: {}", subtask.goal),
                        });
                        let _ = tx.send(AgentEvent::Progress {
                            message: format!("Step {step_no}: {}", subtask.goal),
                        });
                        // Step 2 — route the subtask to a small allowed tool set, then let the
                        // cheap model compile it to exactly one command. files|shell subtasks
                        // use built-in local tools; everything else uses the desktop catalog.
                        let cat = subtask.category.to_ascii_lowercase();
                        let (allowed, context) = if cat == "files"
                            || cat == "shell"
                            || cat == "system"
                        {
                            let names = registry.local_tool_names();
                            if names.is_empty() {
                                return Err(anyhow!(
                                    "no local tools available for subtask: {}",
                                    subtask.goal
                                ));
                            }
                            let mut ctx = String::from(
                                "Local machine tools (read/write files, run shell commands, git). Prefer read-only observation tools when only inspecting; never invent file contents or command output.\n",
                            );
                            for id in &subtask.depends_on {
                                if let Some(v) = completed.get(id) {
                                    ctx.push_str(&format!("\nDependency {} result: {}", id, v));
                                }
                            }
                            if ctx.len() > MAX_CONTEXT_CHARS {
                                ctx.truncate(MAX_CONTEXT_CHARS);
                                ctx.push_str("\n[context truncated]");
                            }
                            (names, ctx)
                        } else {
                            let subroute = catalog.route_domain(&subtask.category, &subtask.goal);
                            let names: HashSet<String> =
                                subroute.candidates.iter().cloned().collect();
                            if names.is_empty() {
                                return Err(anyhow!(
                                    "no HyprFast tools matched subtask: {}",
                                    subtask.goal
                                ));
                            }
                            let ctx =
                                build_context(&catalog, &subroute, &completed, &subtask.depends_on);
                            (names, ctx)
                        };
                        let schemas = registry.definitions_for_names(&allowed);
                        let command = plan_command(
                            &provider,
                            &command_model,
                            &prompt,
                            &subtask,
                            &schemas,
                            &context,
                            &interrupt,
                        )
                        .await?;
                        if !allowed.contains(&command.tool) {
                            return Err(anyhow!(
                                "command model selected tool outside routed capability set: {}",
                                command.tool
                            ));
                        }
                        if !command.arguments.is_object() {
                            return Err(anyhow!("command model arguments must be a JSON object"));
                        }
                        // Structured log so `LUCY_LOG_FILE=... lucy` captures the
                        // exact planned command per subtask (session history only
                        // keeps tool outputs, not inputs — which made the YT
                        // session undebuggable).
                        {
                            let args = command.arguments.to_string();
                            let preview: String = args.chars().take(400).collect();
                            tracing::info!(
                                subtask = %subtask.goal,
                                tool = %command.tool,
                                args = %preview,
                                "executing planned command"
                            );
                        }
                        let call_id = format!("plan-{}", decisions + 1);
                        let gate = approvals.with_events(tx.clone());
                        if gate.needs_approval(
                            &command.tool,
                            registry.requires_approval(&command.tool),
                        ) {
                            let _ = tx.send(AgentEvent::Progress {
                                message: format!("Needs your approval: {}…", command.tool),
                            });
                            match gate.ask(&call_id, &command.tool, &command.arguments).await {
                                ApprovalDecision::AllowOnce | ApprovalDecision::AllowAlways => {}
                                ApprovalDecision::Deny => {
                                    let denied =
                                        serde_json::json!({"denied by user":command.tool.clone()});
                                    let _ = tx.send(AgentEvent::ToolFinished {
                                        id: call_id.clone(),
                                        name: command.tool.clone(),
                                        output: denied.clone(),
                                        is_error: true,
                                    });
                                    if !planner_cfg.replan_on_failure {
                                        return Err(anyhow!("subtask denied: {}", subtask.goal));
                                    }
                                    let _ = tx.send(AgentEvent::Progress {
                                        message: format!(
                                            "You denied {} — replanning…",
                                            command.tool
                                        ),
                                    });
                                    completed
                                        .insert(format!("{}_failure", subtask.id), denied.clone());
                                    let decision = decide_next(
                                        &provider,
                                        &main_model,
                                        &prompt,
                                        &queue,
                                        &completed,
                                        &subtask,
                                        &denied,
                                        &catalog,
                                        &interrupt,
                                    )
                                    .await?;
                                    decisions += 1;
                                    if let Some(s) = decision.subtask {
                                        queue.push_front(s)
                                    }
                                    continue;
                                }
                            }
                        }
                        let _ = tx.send(AgentEvent::ToolStarted {
                            id: call_id.clone(),
                            name: command.tool.clone(),
                            input: command.arguments.clone(),
                        });
                        let ctx = ToolContext {
                            session_id: SessionId::default(),
                            tool_call_id: call_id.clone(),
                            working_dir: Some(working_dir.clone()),
                            execution_mode: ExecutionMode::Agent,
                            events: tx.clone(),
                            interrupt: interrupt.clone(),
                        };
                        let result = registry
                            .execute(&command.tool, command.arguments.clone(), ctx)
                            .await;
                        if let Err(e) = result.as_ref() {
                            tracing::warn!(
                                subtask = %subtask.goal,
                                tool = %command.tool,
                                error = %e,
                                "planned command failed"
                            );
                        }
                        let output = match result {
                            Ok(v) => {
                                let _ = tx.send(AgentEvent::ToolFinished {
                                    id: call_id.clone(),
                                    name: command.tool.clone(),
                                    output: v.clone(),
                                    is_error: false,
                                });
                                v
                            }
                            Err(e) => {
                                let err = serde_json::json!({"error":e.to_string()});
                                let _ = tx.send(AgentEvent::ToolFinished {
                                    id: call_id.clone(),
                                    name: command.tool.clone(),
                                    output: err.clone(),
                                    is_error: true,
                                });
                                if !planner_cfg.replan_on_failure {
                                    return Err(anyhow!("subtask failed: {}: {}", subtask.goal, e));
                                }
                                let _ = tx.send(AgentEvent::Progress {
                                    message: format!("Step failed ({e}) — replanning…"),
                                });
                                completed.insert(format!("{}_failure", subtask.id), err.clone());
                                let decision = decide_next(
                                    &provider,
                                    &main_model,
                                    &prompt,
                                    &queue,
                                    &completed,
                                    &subtask,
                                    &err,
                                    &catalog,
                                    &interrupt,
                                )
                                .await?;
                                decisions += 1;
                                if let Some(s) = decision.subtask {
                                    queue.push_front(s)
                                }
                                continue;
                            }
                        };
                        let state = truncate_json(output.clone());
                        completed.insert(subtask.id.clone(), state.clone());
                        notes.push(subtask.goal.clone());
                        done_count += 1;
                        let _ = tx.send(AgentEvent::History {
                            message: TurnMessage::Tool(ToolResult {
                                call_id,
                                name: command.tool.clone(),
                                output: state.clone(),
                                is_error: false,
                            }),
                        });
                        // Local read-only tools need no observation loop; everything else that
                        // changes state is verified by the main model before reporting success.
                        let must_verify = verify_actions
                            && planner_cfg.verify_state
                            && registry.requires_approval(&command.tool)
                            && action_requires_verification(
                                &catalog,
                                &command.tool,
                                &subtask.category,
                                command.verify.is_some(),
                            )
                            && !subtask.id.starts_with("verify-");
                        if must_verify {
                            let _ = tx.send(AgentEvent::Status {
                                message: "Verifying the result…".into(),
                            });
                            let _ = tx.send(AgentEvent::Progress {
                                message: "Verifying the result…".into(),
                            });
                        }
                        decisions += 1;
                        // Step 3 — main model manages: confirm the work, continue, or recover.
                        let decision = decide_next(
                            &provider,
                            &main_model,
                            &prompt,
                            &queue,
                            &completed,
                            &subtask,
                            &state,
                            &catalog,
                            &interrupt,
                        )
                        .await?;
                        if must_verify {
                            if let Some(next) = decision.subtask {
                                queue.push_front(next)
                            }
                            queue.push_front(verification_subtask(
                                &catalog,
                                &command.tool,
                                &subtask.goal,
                                &subtask.category,
                                decisions,
                            ));
                            continue;
                        }
                        match decision.decision {
                            DecisionKind::Complete => {
                                let _ = tx.send(AgentEvent::Status {
                                    message: format!("Done: {}", decision.reason),
                                });
                                let _ = tx.send(AgentEvent::Progress {
                                    message: format!("Done: {}", decision.reason),
                                });
                                break;
                            }
                            DecisionKind::Continue => {
                                if let Some(next) = decision.subtask {
                                    let _ = tx.send(AgentEvent::Progress {
                                        message: format!("Confirmed — next: {}", next.goal),
                                    });
                                    queue.push_front(next)
                                }
                            }
                            DecisionKind::Replan => {
                                if let Some(next) = decision.subtask {
                                    let _ = tx.send(AgentEvent::Progress {
                                        message: format!("Adjusting plan: {}", next.goal),
                                    });
                                    queue.push_front(next)
                                } else if queue.is_empty() {
                                    return Err(anyhow!(
                                        "main model requested replanning but supplied no recovery subtask"
                                    ));
                                }
                            }
                        }
                    }
                    // Finale — main model confirms the work in one short plain-English summary.
                    let capped_results =
                        truncate_json(serde_json::to_value(&completed).unwrap_or(Value::Null))
                            .to_string();
                    let summary_user = format!(
                        "Task:\n{}\n\nSteps performed:\n{}\n\nCompleted step results/state:\n{}\n\nSummarize what was actually done and the outcome, for the user. Only describe what the results evidence. For media playback, only claim the media is playing if the results show playing evidence (pause button, playing state, advancing currentTime) — a watch/search page load alone is NOT playback.",
                        prompt,
                        notes.join("\n"),
                        capped_results
                    );
                    let text=match provider.complete_json(&main_model,"You are Lucy, a warm and friendly AI assistant. Summarize completed computer work in English. Be concise: two to four short sentences, plus a '- ' bullet list only if several distinct things were done. Format for a plain-text terminal: short paragraphs separated by blank lines, one list item per line starting with '- ', no **bold** markers and no backticks. Never claim actions that are not evidenced by the results. Reply ONLY JSON: {\"summary\":\"...\"}.",&summary_user,interrupt.clone()).await{
      Ok(v)=>v.get("summary").and_then(|s|s.as_str()).map(str::to_owned).filter(|s|!s.trim().is_empty()).unwrap_or_else(||format!("Completed {} step(s).",notes.len())),
      Err(_)=>format!("Completed {} step(s).",notes.len()),
    };
                    let _ = tx.send(AgentEvent::Progress {
                        message: text.clone(),
                    });
                    let assistant = TurnMessage::Assistant(AssistantTurn {
                        text: Some(text.clone()),
                        tool_calls: Vec::new(),
                    });
                    let _ = tx.send(AgentEvent::History { message: assistant });
                    let _ = tx.send(AgentEvent::TextDelta { text });
                    Ok::<(), anyhow::Error>(())
                }
            };
            if let Err(e) = run.await {
                let _ = tx.send(AgentEvent::Error {
                    message: e.to_string(),
                });
            }
            let _ = tx.send(AgentEvent::Done);
        });
        Ok(rx)
    }
    pub async fn history(&self) -> Vec<TurnMessage> {
        self.session.lock().await.history.clone()
    }
}
