//! Lucy runtime: combines the latency-sensitive Lucy execution engine with
//! optional ADK-Rust extension services.
mod execution;
mod harness;
mod memory;
mod planner;
mod prompts;
mod sessions;
pub use execution::{build_waves, has_dependency_cycle, parallel_candidate, ExecutionWave};
pub use harness::{
    browser_candidates, capability_summary_line, cdp_port_responds, deterministic_recovery,
    has_debug_flag, name_looks_like_launch, name_looks_like_observation, parse_router_envelope,
    parse_verifier, preflight_browser, preflight_for_domains, validate_recovery_calls,
    validate_tool_plan, verifier_user_message, CallBudget, ExecutionTrace, GoalObject,
    PreflightState, RecoveryFix, StepResult, VerifierDecision,
};
pub use planner::SubTask;
pub use sessions::trim_history;
use std::{collections::{HashMap, HashSet}, path::PathBuf, sync::Arc};
use anyhow::{anyhow, Result};
use harness::{recovery_user_message, validate_tool_plan as validate_plan_calls};
use lucy_adk::{LucyAdk, LucySessionService};
use lucy_agent::{Agent, OpenAIProvider};
use lucy_config::LucyConfig;
use lucy_core::{AgentEvent, ApprovalDecision, ApprovalGate, AssistantTurn, ExecutionMode, InterruptSignal, SessionData, SessionId, TokenUsage, ToolCall, ToolContext, ToolResult, TurnMessage};
use lucy_hyprfast::HyprFastCatalog;
use lucy_mcp::{load_config, StdioMcpClient};
use lucy_tools::{default_registry, ToolRegistry};
use serde_json::Value;
use tokio::sync::{mpsc, Mutex};
use sessions::now;
const MAX_CONTEXT_CHARS: usize = 16_000;
const MAX_ACTIONS: usize = 64;

/// Runtime session cache. Persistence and history authority live exclusively in ADK SessionService.
pub struct LucyRuntime { agent:Arc<Agent<OpenAIProvider>>, registry:Arc<ToolRegistry>, provider:Arc<OpenAIProvider>, session:Arc<Mutex<SessionData>>, session_service:Arc<LucySessionService>, interrupt:InterruptSignal, working_dir:PathBuf, hyprfast:Option<HyprFastCatalog>, config:LucyConfig, approvals:ApprovalGate, adk:Arc<LucyAdk> }
impl LucyRuntime {
    pub async fn new()->Result<Self>{
        let config=LucyConfig::load()?;let provider=Arc::new(OpenAIProvider::from_config(&config)?);let mut registry=default_registry();
        let hf_cfg=lucy_mcp::McpServerConfig{name:"hyprfast".into(),command:config.hyprfast.command.clone(),args:config.hyprfast.args.clone(),env:Default::default()};
        let state_dir=config.sessions.dir.clone().or_else(||std::env::var("LUCY_SESSIONS_DIR").ok().map(PathBuf::from)).unwrap_or_else(||PathBuf::from(std::env::var("HOME").unwrap_or_else(|_|".".into())).join(".local/state/lucy/sessions"));
        let legacy_path=config.sessions.file.clone().or_else(||std::env::var("LUCY_SESSION_FILE").ok().map(PathBuf::from)).unwrap_or_else(||PathBuf::from(std::env::var("HOME").unwrap_or_else(|_|".".into())).join(".local/state/lucy/session.json"));
        // Run the independent slow I/O concurrently instead of sequentially:
        // MCP discovery (spawns server processes), session-store open, and
        // ADK memory open. Previously these ran one after another, and MCP
        // discovery alone paid the ~1.4s `npx` handshake twice.
        let (discover_res, session_open_res, adk_opened) = tokio::join!(
            HyprFastCatalog::discover_with_definitions(hf_cfg.clone()),
            LucySessionService::open(&state_dir),
            LucyAdk::open(&state_dir),
        );
        let hyprfast=match discover_res{Ok((c,hf_defs,cu_defs))=>{tracing::info!(tools=c.len(),"HyprFast MCP connected");let _=c.save().await;
            // Register execution proxies from the ALREADY-FETCHED definitions.
            // Proxy clients connect lazily on the first real tool call, so
            // this spawns zero processes. (The old code called
            // register_server() here, which re-ran a full MCP handshake per
            // server — including the slow `npx` one — a second time.)
            let hf_n=lucy_mcp::register_server_with_defs(&mut registry,hf_cfg,hf_defs);
            let cu_n=if cu_defs.is_empty(){0}else{lucy_mcp::register_server_with_defs(&mut registry,lucy_mcp::computer_use_config(),cu_defs)};
            tracing::info!(hyprfast_tools=hf_n,computer_use_tools=cu_n,"MCP tools registered");
            Some(c)},Err(e)=>{tracing::warn!(error=%e,"HyprFast MCP unavailable; continuing without desktop capabilities");None}};
        // User-configured extra servers still need a live tools/list (their
        // schemas are not cached). hyprfast/computer_use are already covered
        // above, so skip them — and probe the rest concurrently instead of
        // one blocking handshake at a time.
        let extra_servers=load_config()?.into_iter().filter(|s|!s.name.eq_ignore_ascii_case("hyprfast")&&!s.name.eq_ignore_ascii_case("computer_use")).collect::<Vec<_>>();
        let mut probing=tokio::task::JoinSet::new();
        for server in extra_servers{probing.spawn(async move{let name=server.name.clone();let defs=StdioMcpClient::new(server.clone()).list_tools().await;match defs{Ok(d)=>Ok::<_,anyhow::Error>((server,d)),Err(e)=>Err(anyhow!("{name}: {e}") )}});}
        while let Some(done)=probing.join_next().await{match done.map_err(|e|anyhow!("MCP probe task failed: {e}"))?{Ok((server,defs))=>{let _=lucy_mcp::register_server_with_defs(&mut registry,server,defs);},Err(e)=>tracing::warn!(error=%e,"MCP server unavailable")}}
        let session_service=Arc::new(session_open_res?);let _=session_service.import_legacy_file(&legacy_path).await;
        let session=session_service.ensure_current(config.sessions.resume).await?;
        let registry=Arc::new(registry);let agent=Arc::new(Agent::new(provider.clone(),registry.clone()));let(approval_tx,_approval_rx)=mpsc::unbounded_channel();let approvals=ApprovalGate::new(approval_tx);if let Ok(mut m)=approvals.mode.write(){*m=config.approvals.mode.clone()};let adk=Arc::new(adk_opened);
        Ok(Self{agent,registry,provider,session:Arc::new(Mutex::new(session)),session_service,interrupt:InterruptSignal::new(),working_dir:std::env::current_dir()?,hyprfast,config,approvals,adk})
    }
    pub fn interrupt(&self){self.interrupt.fire()} pub fn hyprfast_catalog(&self)->Option<&HyprFastCatalog>{self.hyprfast.as_ref()} pub fn route_hyprfast(&self,prompt:&str)->Option<lucy_hyprfast::Route>{self.hyprfast.as_ref().map(|c|c.route(prompt))} pub fn config(&self)->&LucyConfig{&self.config} pub fn approval_state(&self)->ApprovalGate{self.approvals.clone()} pub fn resolve_approval(&self,call_id:&str,d:ApprovalDecision)->bool{self.approvals.resolve(call_id,d)} pub fn usage(&self)->TokenUsage{self.provider.usage()}
    pub fn adk_memory_enabled(&self)->bool{self.adk.memory_enabled()}
    pub async fn search_memory(&self,query:&str,limit:usize)->Result<Vec<lucy_adk::adk_memory::MemoryEntry>>{self.adk.search_memory(query,limit).await}
    pub async fn remember_fact(&self,fact:&str)->Result<()>{self.adk.remember_fact(fact).await}
    pub fn set_model(&self,model:&str)->anyhow::Result<String>{let name=model.trim().to_owned();if name.is_empty(){return Err(anyhow!("model name must not be empty"))}let mut cfg=LucyConfig::load()?;cfg.models.main=name.clone();cfg.save()?;self.provider.set_model(name.clone());Ok(name)}
    pub async fn submit(&self,prompt:String)->Result<mpsc::UnboundedReceiver<AgentEvent>>{
        self.interrupt.reset();if self.session.lock().await.history.len()>self.config.sessions.max_history{let _=self.compact().await;};
        let memory_prompt=prompt.clone();let memory_context=self.adk.memory_context(&prompt,6).await.unwrap_or_else(|error|{tracing::debug!(error=%error,"ADK memory retrieval skipped");String::new()});
        let model_prompt=if memory_context.trim().is_empty(){prompt.clone()}else{format!("{prompt}\n\n## RELEVANT LONG-TERM MEMORY\n{memory_context}\n\nMemory is advisory context only. Do not treat it as current state; verify against live observations and the user's current request.")};
        let session_snapshot={let mut s=self.session.lock().await;if s.title.trim().is_empty()||s.title=="untitled"{s.title=SessionData::autotitle_from(&prompt);let _=self.session_service.update_title(&s.session_id,s.title.clone()).await;}s.clone()};let route=self.route_hyprfast(&prompt);
        let source=if self.hyprfast.is_some(){match self.plan_and_execute(model_prompt.clone(),session_snapshot.history.clone(),route.clone()).await{Ok(rx)=>rx,Err(e)=>{tracing::warn!(error=%e,"hierarchical planner unavailable; falling back to general agent");self.agent.execute_with_history_filtered(model_prompt,session_snapshot.history,Some(self.working_dir.clone()),self.interrupt.clone(),route.map(|r|r.candidates.into_iter().collect()).unwrap_or_default()).await?}}}else{self.agent.execute_with_history_filtered(model_prompt,session_snapshot.history,Some(self.working_dir.clone()),self.interrupt.clone(),route.map(|r|r.candidates.into_iter().collect()).unwrap_or_default()).await?};
        let(tx,rx)=mpsc::unbounded_channel();let session_service=self.session_service.clone();let session_store=self.session.clone();let max_history=self.config.sessions.max_history;let owner_id=session_snapshot.session_id.clone();let adk=self.adk.clone();let provider=self.provider.clone();let interrupt=self.interrupt.clone();
        tokio::spawn(async move{let mut source=source;let mut last_assistant_text=None::<String>;while let Some(event)=source.recv().await{if let AgentEvent::History{message}=&event{if let TurnMessage::Assistant(turn)=message{if let Some(text)=turn.text.clone(){if !text.trim().is_empty(){last_assistant_text=Some(text);}}}let event_result=adk_event_for_turn(message);let mut session=session_store.lock().await;if session.session_id==owner_id{session.history.push(message.clone());trim_history(&mut session.history,max_history);session.updated_at=now();}let _=session_service.save_event(&owner_id,event_result).await;}let _=tx.send(event);}if let Some(text)=last_assistant_text{let adk=adk.clone();let provider=provider.clone();let prompt=memory_prompt.clone();let interrupt=interrupt.clone();tokio::spawn(async move{if let Err(error)=memory::extract_and_store(provider,adk,&prompt,&text,interrupt).await{tracing::debug!(error=%error,"ADK memory extraction skipped");}});}});Ok(rx)
    }
    /// Harness v2 (`docs/lucy-harness-architecture.md`): one Router+Planner
    /// call with native `tool_calls`, deterministic preflight + recovery in
    /// code (0 LLM calls), and one terminal Verifier call. Budget: 2 calls
    /// healthy, 3–4 with one genuine deviation.
    async fn plan_and_execute(&self,prompt:String,history:Vec<TurnMessage>,route:Option<lucy_hyprfast::Route>)->Result<mpsc::UnboundedReceiver<AgentEvent>>{
        let(tx,rx)=mpsc::unbounded_channel();let provider=self.provider.clone();let registry=self.registry.clone();let catalog=self.hyprfast.clone().ok_or_else(||anyhow!("HyprFast catalog unavailable"))?;let interrupt=self.interrupt.clone();let working_dir=self.working_dir.clone();let browser_cfg=self.config.browser.clone();let harness_cfg=self.config.harness.clone();let main_model=self.provider.model();let approvals=self.approvals.clone();let fallback_agent=self.agent.clone();let fallback_working_dir=self.working_dir.clone();let fallback_interrupt=self.interrupt.clone();
        tokio::spawn(async move{let run=async{
            let _=tx.send(AgentEvent::Status{message:"Understanding your request…".into()});
            let mut budget=CallBudget::new(harness_cfg.max_llm_calls_single_task.max(2));
            // 0. Route (cheap, no LLM) + domain inference for preflight/schema scope.
            let route_owned=route.clone().unwrap_or_else(||catalog.route(&prompt));
            let domains=domains_for(&catalog,&route_owned,&prompt);
            let resolved_binary=lucy_config::LucyConfig{ browser: browser_cfg.clone(), ..lucy_config::LucyConfig::default() }.resolve_browser_binary();
            // 1. Deterministic preflight (§4.2): 0 LLM calls, idempotent.
            let pf=preflight_for_domains(&domains,&browser_cfg,&resolved_binary).await;
            if let PreflightState::Failed(reason)=pf{
                let _=tx.send(AgentEvent::History{message:TurnMessage::User(prompt.clone())});
                return Err(anyhow!("environment preflight failed: {reason}"));
            }
            let browser_ready=cdp_port_responds(browser_cfg.cdp_port);
            // 2. Offered schemas (§11.4.1 + §6): batch covers → singles hidden;
            // launch hidden when a session already exists. Local tools are few
            // and always offered so file/shell intents stay expressible.
            let filtered=catalog.planner_tool_set(&route_owned.candidates,browser_ready);
            let mut allowed:HashSet<String>=filtered.into_iter().collect();
            for n in registry.local_tool_names(){allowed.insert(n);}
            if allowed.is_empty(){return Err(anyhow!("no tools available for this request"));}
            let schemas=registry.definitions_for_names(&allowed);
            let planner_user=planner_user_text(&prompt,&catalog,&route_owned,browser_ready,&resolved_binary,&browser_cfg);
            // 3. Router+Planner (Call 1): native tool calling, full sequence.
            budget.record("router+planner");
            let mut turn=provider.plan_with_tools(prompts::ROUTER_PLANNER,&history,&planner_user,&schemas,interrupt.clone()).await?;
            let mut envelope=parse_router_envelope(turn.text.as_deref(),&prompt);
            // Chat mode: no tool calls → immediate reply (1 call total).
            if turn.tool_calls.is_empty(){
                let _=tx.send(AgentEvent::History{message:TurnMessage::User(prompt.clone())});
                let text=envelope.reply.filter(|r|!r.trim().is_empty()).or(turn.text.clone()).unwrap_or_else(||"Done.".to_string());
                let text=planner::strip_action_claims(&text);
                let assistant=TurnMessage::Assistant(AssistantTurn{text:Some(text.clone()),tool_calls:Vec::new()});
                let _=tx.send(AgentEvent::History{message:assistant});
                let _=tx.send(AgentEvent::TextDelta{text});
                return Ok::<(),anyhow::Error>(());
            }
            if envelope.mode!="act"{
                // Model called tools but labeled the envelope chat: trust the
                // tools (act), deriving the goal from the prompt.
                envelope=parse_router_envelope(None,&prompt);
            }
            // Immutable goal object (§7): created once, never rewritten.
            let goal=GoalObject::new(
                envelope.goal_statement.unwrap_or_else(||prompt.clone()),
                envelope.success_condition.unwrap_or_else(||format!("The requested outcome is observably true: {prompt}")),
                domains.clone(),
            )?;
            // Strict validation (§7): reject-and-retry once, then escalate.
            if let Err(e)=validate_plan_with_registry(&turn.tool_calls,Some(&catalog),browser_ready,&registry){
                tracing::warn!(error=%e,"planner output invalid; retrying once");
                budget.record("router+planner-retry");
                let retry_user=format!("{planner_user}\n\nYour previous output was invalid: {e}. Return ONLY a valid full tool_calls sequence ending in a read-only observation step.");
                turn=provider.plan_with_tools(prompts::ROUTER_PLANNER,&history,&retry_user,&schemas,interrupt.clone()).await?;
                validate_plan_with_registry(&turn.tool_calls,Some(&catalog),browser_ready,&registry)?;
            }
            let _=tx.send(AgentEvent::History{message:TurnMessage::User(prompt.clone())});
            let _=tx.send(AgentEvent::Progress{message:format!("Plan ready: {} tool call{}…",turn.tool_calls.len(),if turn.tool_calls.len()==1{""}else{"s"})});
            // 4. Tool runtime (§4.4): execute the FULL sequence, 0 LLM calls.
            let mut actions=0usize;
            let (mut trace,mut failed)=run_tool_sequence(&turn.tool_calls,&registry,Some(&catalog),&working_dir,&interrupt,&approvals.with_events(tx.clone()),&tx,&browser_cfg,&resolved_binary,"plan",&mut actions,harness_cfg.max_step_retries).await?;
            // 5. Genuine deviation only → scoped recovery (§4.6), bounded.
            let mut recoveries=0usize;
            while failed.is_some() && recoveries<harness_cfg.max_recoveries{
                if interrupt.is_set(){return Err(lucy_core::LucyError::Cancelled.into());}
                let (failed_tool,failed_input,failed_err)=failed.clone().expect("checked");
                recoveries+=1;
                budget.record(format!("recovery-{recoveries}"));
                let _=tx.send(AgentEvent::Progress{message:format!("Step '{failed_tool}' needs a different approach — recovering…")});
                let completed_summary=trace.completed_summary();
                let rec_system=prompts::RECOVERY
                    .replace("{goal_statement}",&goal.goal_statement)
                    .replace("{success_condition}",&goal.success_condition)
                    .replace("{completed_steps_summary}",if completed_summary.is_empty(){"(none — the first step failed)"}else{&completed_summary})
                    .replace("{failed_step}",&format!("{failed_tool} {failed_input}"))
                    .replace("{error_detail}",&failed_err.to_string());
                let rec_user=recovery_user_message(&goal,&completed_summary,&format!("{failed_tool} {failed_input}"),&failed_err);
                let rec_turn=provider.plan_with_tools(&rec_system,&history,&rec_user,&schemas,interrupt.clone()).await?;
                if rec_turn.tool_calls.is_empty(){
                    return Err(anyhow!("recovery reported no viable path: {}",rec_turn.text.unwrap_or_default()));
                }
                // Recovery invariant: must lead back to the success condition,
                // never a bare environment fix-up (§4.6). Retry once.
                if let Err(e)=validate_recovery_calls(&rec_turn.tool_calls,Some(&catalog),cdp_port_responds(browser_cfg.cdp_port),&goal){
                    tracing::warn!(error=%e,"recovery output invalid; retrying once");
                    let retry_user=format!("{rec_user}\n\nYour previous recovery was invalid: {e}. Return the SMALLEST replacement sequence that reaches the original success condition, ending in observation.");
                    let retry=provider.plan_with_tools(&rec_system,&history,&retry_user,&schemas,interrupt.clone()).await?;
                    budget.record(format!("recovery-{recoveries}-retry"));
                    validate_recovery_calls(&retry.tool_calls,Some(&catalog),cdp_port_responds(browser_cfg.cdp_port),&goal)?;
                    let (t2,f2)=run_tool_sequence(&retry.tool_calls,&registry,Some(&catalog),&working_dir,&interrupt,&approvals.with_events(tx.clone()),&tx,&browser_cfg,&resolved_binary,&format!("recovery-{recoveries}"),&mut actions,harness_cfg.max_step_retries).await?;
                    trace=t2;failed=f2;
                } else {
                    let (t2,f2)=run_tool_sequence(&rec_turn.tool_calls,&registry,Some(&catalog),&working_dir,&interrupt,&approvals.with_events(tx.clone()),&tx,&browser_cfg,&resolved_binary,&format!("recovery-{recoveries}"),&mut actions,harness_cfg.max_step_retries).await?;
                    trace=t2;failed=f2;
                }
            }
            if let Some((failed_tool,_failed_input,failed_err))=failed{
                return Err(anyhow!("could not complete '{}': recovery budget exhausted (last error in '{}': {})",goal.goal_statement,failed_tool,failed_err));
            }
            // 6. Verifier (final call): outcome, not step (§4.7). Complete or
            // recover — there is no third "continue with empty plan" branch.
            budget.record("verifier");
            let final_obs=trace.final_observation();
            let ver_system=prompts::VERIFIER
                .replace("{success_condition}",&goal.success_condition)
                .replace("{final_observation}",&final_obs.to_string());
            let ver_value=provider.complete_json(&main_model,&ver_system,&verifier_user_message(&goal.success_condition,&final_obs),interrupt.clone()).await?;
            let mut decision=parse_verifier(&ver_value)?;
            // One verifier-driven recovery lap at most: a "recover" verdict
            // with budget left gets a scoped fix + re-verify, not an open loop.
            if !decision.complete && recoveries<harness_cfg.max_recoveries{
                if interrupt.is_set(){return Err(lucy_core::LucyError::Cancelled.into());}
                budget.record("verifier-recovery");
                let unmet=decision.unmet_reason.clone().unwrap_or_else(||"goal not observably met".into());
                let _=tx.send(AgentEvent::Progress{message:format!("Verifying… {unmet} — fixing…")});
                let rec_system=prompts::RECOVERY
                    .replace("{goal_statement}",&goal.goal_statement)
                    .replace("{success_condition}",&goal.success_condition)
                    .replace("{completed_steps_summary}",&trace.completed_summary())
                    .replace("{failed_step}",&format!("verifier: {unmet}"))
                    .replace("{error_detail}",&final_obs.to_string());
                let rec_user=recovery_user_message(&goal,&trace.completed_summary(),&format!("verifier: {unmet}"),&final_obs);
                let rec_turn=provider.plan_with_tools(&rec_system,&history,&rec_user,&schemas,interrupt.clone()).await?;
                validate_recovery_calls(&rec_turn.tool_calls,Some(&catalog),cdp_port_responds(browser_cfg.cdp_port),&goal)?;
                let (t2,f2)=run_tool_sequence(&rec_turn.tool_calls,&registry,Some(&catalog),&working_dir,&interrupt,&approvals.with_events(tx.clone()),&tx,&browser_cfg,&resolved_binary,"verify-recovery",&mut actions,harness_cfg.max_step_retries).await?;
                if let Some((t,_,e))=f2{return Err(anyhow!("verifier-driven recovery failed in '{t}': {e}"));}
                trace=t2;
                budget.record("verifier-2");
                let final_obs2=trace.final_observation();
                let ver_system2=prompts::VERIFIER
                    .replace("{success_condition}",&goal.success_condition)
                    .replace("{final_observation}",&final_obs2.to_string());
                let ver_value2=provider.complete_json(&main_model,&ver_system2,&verifier_user_message(&goal.success_condition,&final_obs2),interrupt.clone()).await?;
                decision=parse_verifier(&ver_value2)?;
            }
            // §10.8: hold the implementation accountable to the §3 budget.
            if budget.over_budget() && domains.len()<=2{
                tracing::warn!(calls=budget.calls,limit=budget.limit,labels=?budget.labels,"single-task LLM call budget exceeded; this is a bug, not caution");
            } else {
                tracing::info!(calls=budget.calls,labels=?budget.labels,"harness v2 call budget");
            }
            if !decision.complete{
                let unmet=decision.unmet_reason.unwrap_or_else(||"goal not observably met".into());
                return Err(anyhow!("goal not met ({}): {}",goal.success_condition,unmet));
            }
            let _=tx.send(AgentEvent::Status{message:decision.evidence.clone()});
            let text=format!("Done: {}",decision.evidence);
            let assistant=TurnMessage::Assistant(AssistantTurn{text:Some(text.clone()),tool_calls:Vec::new()});
            let _=tx.send(AgentEvent::History{message:assistant});
            let _=tx.send(AgentEvent::TextDelta{text});
            Ok::<(),anyhow::Error>(())
        };
        // V2 failure before any History event → fall back to the general
        // agent (which emits its own User event), mirroring the old triage
        // fallback. Failures after History stay as Error events so the
        // session never stores a prompt twice or loses the goal.
        if let Err(e)=run.await{
            // Heuristic: if we never got past planning, delegate instead of
            // surfacing a raw planning error.
            let msg=e.to_string();
            let pre_history=!msg.contains("preflight failed")
                && (msg.contains("planner")||msg.contains("tool_calls")||msg.contains("Recovery")||msg.contains("recovery reported"));
            if pre_history{
                tracing::warn!(error=%e,"harness v2 planning failed; falling back to general agent");
                let _=tx.send(AgentEvent::Status{message:"Planning didn't work out — trying a different approach…".into()});
                match fallback_agent.execute_with_history_filtered(prompt.clone(),history.clone(),Some(fallback_working_dir.clone()),fallback_interrupt.clone(),HashSet::new()).await{
                    Ok(mut fb)=>{while let Some(event)=fb.recv().await{let _=tx.send(event);}}
                    Err(fb_err)=>{let _=tx.send(AgentEvent::Error{message:format!("{e}; fallback also failed: {fb_err}")});}
                }
            } else {
                let _=tx.send(AgentEvent::Error{message:msg});
            }
        }
        let _=tx.send(AgentEvent::Done);});Ok(rx)
    }
    pub async fn history(&self)->Vec<TurnMessage>{self.session.lock().await.history.clone()}
}

/// Domain inference for preflight scope + goal object (§4.1: compact domain
/// names, never full schemas). Derived from route candidates; file/shell
/// intents add `files` so local tools stay expressible.
fn domains_for(catalog:&HyprFastCatalog,route:&lucy_hyprfast::Route,prompt:&str)->Vec<String>{
    let mut domains:Vec<String>=Vec::new();
    for cand in &route.candidates{
        if let Some(cap)=catalog.capability_for_mcp_name(cand){
            let d=format!("{:?}",cap.domain).to_ascii_lowercase();
            if !domains.contains(&d){domains.push(d);}
        }
    }
    let lower=prompt.to_ascii_lowercase();
    if ["file","read ","write ","create ","delete ","directory","folder","shell","command","run ","git ","search files"].iter().any(|t|lower.contains(t)) && !domains.contains(&"files".to_string()){
        domains.push("files".into());
    }
    if domains.is_empty(){domains.push("general".into());}
    domains
}

/// Planner user message: goal + compact capability context + ambient
/// preflight state. Zero-cost context only — never a fresh observation call.
fn planner_user_text(prompt:&str,catalog:&HyprFastCatalog,route:&lucy_hyprfast::Route,browser_ready:bool,binary:&str,browser_cfg:&lucy_config::BrowserConfig)->String{
    let mut text=format!("## CURRENT USER REQUEST\n{prompt}\n");
    text.push_str(&format!("\n## CAPABILITY SUMMARY (domains + counts only)\n{}\n",capability_summary_line(catalog)));
    text.push_str(&format!("\n## CURRENT CAPABILITY ROUTE\n{}\n",catalog.context_for(route)));
    text.push_str(&format!("\n## ENVIRONMENT PREFLIGHT (already checked by the runtime; do not re-check)\nbrowser_ready={browser_ready} browser_binary={binary} cdp_port={}\n",browser_cfg.cdp_port));
    text.push_str("\n## PLANNING RULE\nPlan from the desired outcome. Current observations and tool evidence outrank assumptions and stale history. Return the FULL ordered tool_calls sequence ending in a read-only observation step.\n");
    if text.len()>MAX_CONTEXT_CHARS{text.truncate(MAX_CONTEXT_CHARS);text.push_str("\n[context truncated]");}
    text
}

/// Strict plan validation (§7 + §11.4): contract shape plus existence in the
/// registry — an invented tool name fails here, never at runtime.
fn validate_plan_with_registry(calls:&[ToolCall],catalog:Option<&HyprFastCatalog>,browser_ready:bool,registry:&ToolRegistry)->Result<()>{
    validate_plan_calls(calls,catalog,browser_ready)?;
    for c in calls{
        if registry.get(&c.name).is_none(){
            return Err(anyhow!("planner selected unknown tool outside the offered schema: {}",c.name));
        }
    }
    Ok(())
}

/// §4.4 Tool runtime (code, 0 LLM calls): execute the full `tool_calls`
/// array in order, applying the deterministic recovery table (§4.5) on
/// deviations. Returns the trace plus the first unrecovered failure as
/// (tool, input, error) for the scoped recovery call — or `None` when every
/// step succeeded.
///
/// Approval denials and cancellation are hard errors (no recovery): a deny
/// is the user's decision, not a deviation to route around.
#[allow(clippy::too_many_arguments)]
async fn run_tool_sequence(
    calls:&[ToolCall],
    registry:&ToolRegistry,
    catalog:Option<&HyprFastCatalog>,
    working_dir:&std::path::Path,
    interrupt:&InterruptSignal,
    approvals:&ApprovalGate,
    tx:&mpsc::UnboundedSender<AgentEvent>,
    browser_cfg:&lucy_config::BrowserConfig,
    resolved_binary:&str,
    call_id_prefix:&str,
    actions:&mut usize,
    max_retries:usize,
)->Result<(ExecutionTrace,Option<(String,Value,Value)>)>{
    use std::time::{Duration, Instant};
    let started=Instant::now();
    let mut steps:Vec<StepResult>=Vec::new();
    let mut repeats:HashMap<String,usize>=HashMap::new();
    for call in calls{
        if interrupt.is_set(){return Err(lucy_core::LucyError::Cancelled.into());}
        if *actions>=MAX_ACTIONS{return Err(anyhow!("computer-operation action budget exhausted; stopping to prevent a loop"));}
        let fingerprint=format!("{}|{}",call.name,call.input);
        let seen_count=repeats.entry(fingerprint).or_insert(0);
        *seen_count+=1;
        if *seen_count>3{return Err(anyhow!("repeated identical action detected; stopping to prevent an execution loop"));}
        *actions+=1;
        let call_id=format!("{call_id_prefix}-{actions}");
        // §6 destructive gate: catalog `destructive` metadata forces approval
        // regardless of the prompt path — the runtime check, not the model.
        let destructive=catalog.and_then(|c|c.capability_for_mcp_name(&call.name)).map(|c|c.destructive).unwrap_or(false);
        if approvals.needs_approval(&call.name,registry.requires_approval(&call.name)||destructive){
            let _=tx.send(AgentEvent::Progress{message:format!("Needs your approval: {}…",call.name)});
            match approvals.ask(&call_id,&call.name,&call.input).await{
                ApprovalDecision::AllowOnce|ApprovalDecision::AllowAlways=>{},
                ApprovalDecision::Deny=>{
                    let denied=serde_json::json!({"denied by user":call.name.clone()});
                    let _=tx.send(AgentEvent::ToolFinished{id:call_id.clone(),name:call.name.clone(),output:denied.clone(),is_error:true});
                    return Err(anyhow!("tool '{}' denied by user; stopping",call.name));
                }
            }
        }
        let _=tx.send(AgentEvent::ToolStarted{id:call_id.clone(),name:call.name.clone(),input:call.input.clone()});
        let exec=|input:Value,call_id:String|{
            let working_dir=working_dir.to_path_buf();
            let interrupt=interrupt.clone();
            let tx=tx.clone();
            async move{
                let ctx=ToolContext{session_id:SessionId::default(),tool_call_id:call_id,working_dir:Some(working_dir),execution_mode:ExecutionMode::Agent,events:tx,interrupt:interrupt.clone()};
                registry.execute(&call.name,input,ctx).await
            }
        };
        match exec(call.input.clone(),call_id.clone()).await{
            Ok(output)=>{
                let _=tx.send(AgentEvent::ToolFinished{id:call_id.clone(),name:call.name.clone(),output:output.clone(),is_error:false});
                let state=planner::truncate_json(output.clone());
                let _=tx.send(AgentEvent::History{message:TurnMessage::Tool(ToolResult{call_id:call_id.clone(),name:call.name.clone(),output:state,is_error:false})});
                steps.push(StepResult{tool:call.name.clone(),status:"ok".into(),result:output});
            }
            Err(e)=>{
                let err_str=e.to_string();
                let _=tx.send(AgentEvent::ToolFinished{id:call_id.clone(),name:call.name.clone(),output:serde_json::json!({"error":err_str}),is_error:true});
                // §4.5 deterministic recovery first — the LLM sees this only
                // if the single deterministic retry also fails.
                let page_loading={
                    let l=err_str.to_ascii_lowercase();
                    l.contains("loading")||l.contains("readystate")||l.contains("not yet rendered")
                };
                let fix=if max_retries==0{RecoveryFix::Escalate}else{deterministic_recovery(&err_str,page_loading)};
                let retry_ok:Option<Value>=match fix{
                    RecoveryFix::Escalate=>None,
                    RecoveryFix::RetrySame=>{
                        let _=tx.send(AgentEvent::Progress{message:format!("Step '{}' timed out — retrying once…",call.name)});
                        match exec(call.input.clone(),format!("{call_id}-retry")).await{
                            Ok(v)=>Some(v),
                            Err(e2)=>{
                                let _=tx.send(AgentEvent::ToolFinished{id:format!("{call_id}-retry"),name:call.name.clone(),output:serde_json::json!({"error":e2.to_string()}),is_error:true});
                                None
                            }
                        }
                    }
                    RecoveryFix::WaitThenRetry=>{
                        let _=tx.send(AgentEvent::Progress{message:format!("Step '{}' hit a loading page — waiting, then retrying once…",call.name)});
                        tokio::select!{
                            _=tokio::time::sleep(Duration::from_secs(2))=>{},
                            _=interrupt.notified()=>{return Err(lucy_core::LucyError::Cancelled.into());}
                        }
                        match exec(call.input.clone(),format!("{call_id}-retry")).await{
                            Ok(v)=>Some(v),
                            Err(e2)=>{
                                let _=tx.send(AgentEvent::ToolFinished{id:format!("{call_id}-retry"),name:call.name.clone(),output:serde_json::json!({"error":e2.to_string()}),is_error:true});
                                None
                            }
                        }
                    }
                    RecoveryFix::RelaunchBrowserThenRetry=>{
                        let _=tx.send(AgentEvent::Progress{message:"Browser connection lost — re-running preflight…".into()});
                        match preflight_browser(browser_cfg,resolved_binary).await{
                            PreflightState::Ready=>match exec(call.input.clone(),format!("{call_id}-retry")).await{
                                Ok(v)=>Some(v),
                                Err(e2)=>{
                                    let _=tx.send(AgentEvent::ToolFinished{id:format!("{call_id}-retry"),name:call.name.clone(),output:serde_json::json!({"error":e2.to_string()}),is_error:true});
                                    None
                                }
                            },
                            PreflightState::Failed(reason)=>{
                                let _=tx.send(AgentEvent::Progress{message:format!("Browser preflight failed: {reason}")});
                                None
                            }
                        }
                    }
                };
                match retry_ok{
                    Some(output)=>{
                        let _=tx.send(AgentEvent::ToolFinished{id:format!("{call_id}-retry"),name:call.name.clone(),output:output.clone(),is_error:false});
                        let state=planner::truncate_json(output.clone());
                        let _=tx.send(AgentEvent::History{message:TurnMessage::Tool(ToolResult{call_id:call_id.clone(),name:call.name.clone(),output:state,is_error:false})});
                        steps.push(StepResult{tool:call.name.clone(),status:"ok".into(),result:output});
                    }
                    None=>{
                        // Timeouts retry with backoff already spent; surface
                        // the original error for the scoped recovery call.
                        let err_value=serde_json::json!({"error":err_str,"tool":call.name,"fix_attempted":format!("{fix:?}")});
                        steps.push(StepResult{tool:call.name.clone(),status:"error".into(),result:err_value.clone()});
                        let trace=ExecutionTrace::new(steps,started.elapsed().as_millis() as u64);
                        return Ok((trace,Some((call.name.clone(),call.input.clone(),err_value))));
                    }
                }
            }
        }
    }
    let trace=ExecutionTrace::new(steps,started.elapsed().as_millis() as u64);
    Ok((trace,None))
}

fn adk_event_for_turn(message:&TurnMessage)->adk_core::Event{let mut e=adk_core::Event::new(format!("lucy-{}",uuid::Uuid::new_v4()));match message{TurnMessage::User(t)=>{e.author="user".into();e.set_content(adk_core::Content::new("user").with_text(t.clone()));}TurnMessage::Assistant(t)=>{e.author="lucy".into();e.set_content(adk_core::Content::new("assistant").with_text(t.text.clone().unwrap_or_default()));}TurnMessage::Tool(t)=>{e.author="lucy".into();e.set_content(adk_core::Content::new("tool").with_text(t.output.to_string()));}}e}
