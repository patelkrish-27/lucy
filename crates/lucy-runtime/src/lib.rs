//! Lucy runtime: builds the agent, tools and session store, then runs every
//! user command through the hierarchical loop in [`planner`].
//!
//! Module layout:
//! - `planner` — main-model triage, cheap-model command compiler, closed-loop
//!   execution with verification and recovery.
//! - `prompts` — embedded, reviewable model behavior contracts.
//! - `sessions` — opencode-style multi-session CRUD, compaction, trimming.
//! - `execution` — dependency-safe execution waves (scheduler foundation).

mod execution;
mod planner;
mod prompts;
mod sessions;

pub use execution::{build_waves, has_dependency_cycle, parallel_candidate, ExecutionWave};
pub use planner::SubTask;
pub use sessions::trim_history;

use std::{collections::{HashMap, HashSet, VecDeque},path::PathBuf,sync::Arc};
use anyhow::{anyhow, Result};
use lucy_agent::{Agent, OpenAIProvider};
use lucy_config::LucyConfig;
use lucy_core::{AgentEvent, ApprovalDecision, ApprovalGate, InterruptSignal, SessionData, SessionId, SessionStore, TokenUsage, ToolContext, TurnMessage, AssistantTurn, ToolResult, ExecutionMode};
use lucy_hyprfast::HyprFastCatalog;
use lucy_mcp::{load_config, register_server};
use lucy_tools::{default_registry, ToolRegistry};
use serde_json::Value;
use tokio::sync::{mpsc, Mutex};

use planner::{action_requires_verification, build_context, decide_next, plan_command, triage_request, truncate_json, validate_plan, verification_subtask, DecisionKind};
use sessions::now;

const MAX_CONTEXT_CHARS: usize = 16_000;
const MAX_MAIN_DECISIONS: usize = 64;

pub struct LucyRuntime { agent: Arc<Agent<OpenAIProvider>>, registry: Arc<ToolRegistry>, provider: Arc<OpenAIProvider>, session: Arc<Mutex<SessionData>>, store: SessionStore, #[allow(dead_code)] legacy_path: PathBuf, interrupt: InterruptSignal, working_dir: PathBuf, hyprfast: Option<HyprFastCatalog>, config: LucyConfig, approvals: ApprovalGate }

impl LucyRuntime {
 pub async fn new() -> Result<Self> {
  let config=LucyConfig::load()?;
  let provider=Arc::new(OpenAIProvider::from_config(&config)?);
  let mut registry=default_registry();
  let hf_cfg=lucy_mcp::McpServerConfig{name:"hyprfast".into(),command:config.hyprfast.command.clone(),args:config.hyprfast.args.clone(),env:Default::default()};
  let hyprfast=match HyprFastCatalog::discover(hf_cfg.clone()).await{Ok(c)=>{tracing::info!(tools=c.len(),"HyprFast MCP connected");let _=c.save().await;Some(c)},Err(e)=>{tracing::warn!(error=%e,"HyprFast MCP unavailable; continuing without desktop capabilities");None}};
  for server in load_config()? { if server.name.eq_ignore_ascii_case("hyprfast"){continue} if let Err(e)=register_server(&mut registry,server.clone()).await{tracing::warn!(server=%server.name,error=%e,"MCP server unavailable")} }
  if hyprfast.is_some(){if let Err(e)=register_server(&mut registry,hf_cfg).await{tracing::warn!(error=%e,"failed to register HyprFast MCP tools")}}
  let legacy_path=config.sessions.file.clone().or_else(||std::env::var("LUCY_SESSION_FILE").ok().map(PathBuf::from)).unwrap_or_else(||PathBuf::from(std::env::var("HOME").unwrap_or_else(|_|".".into())).join(".local/state/lucy/session.json"));
  let store_dir=config.sessions.dir.clone().or_else(||std::env::var("LUCY_SESSIONS_DIR").ok().map(PathBuf::from)).unwrap_or_else(SessionStore::default_dir);
  let store=SessionStore::new(store_dir);
  let _ = store.migrate_legacy_file(&legacy_path).await;
  let session=if config.sessions.resume{match store.list().await{Ok(list) if !list.is_empty()=>{match store.load(&list[0].id).await{Ok(s)=>s,Err(_)=>store.create(None).await?}},_=>store.create(None).await?,}}else{store.create(None).await?};
  let registry=Arc::new(registry);let agent=Arc::new(Agent::new(provider.clone(),registry.clone()));
  let(approval_tx,_approval_rx)=mpsc::unbounded_channel();let approvals=ApprovalGate::new(approval_tx);if let Ok(mut m)=approvals.mode.write(){*m=config.approvals.mode.clone();}
  Ok(Self{agent,registry,provider,session:Arc::new(Mutex::new(session)),store,legacy_path,interrupt:InterruptSignal::new(),working_dir:std::env::current_dir()?,hyprfast,config,approvals})
 }
 pub fn interrupt(&self){self.interrupt.fire()}
 pub fn hyprfast_catalog(&self)->Option<&HyprFastCatalog>{self.hyprfast.as_ref()}
 pub fn route_hyprfast(&self,prompt:&str)->Option<lucy_hyprfast::Route>{self.hyprfast.as_ref().map(|c|c.route(prompt))}
 pub fn config(&self)->&LucyConfig{&self.config}
 pub fn approval_state(&self)->ApprovalGate{self.approvals.clone()}
 pub fn resolve_approval(&self,call_id:&str,d:ApprovalDecision)->bool{self.approvals.resolve(call_id,d)}
 pub fn usage(&self)->TokenUsage{self.provider.usage()}
 pub fn set_model(&self,model:&str)->anyhow::Result<String>{let name=model.trim().to_owned();if name.is_empty(){return Err(anyhow!("model name must not be empty"));}let mut cfg=LucyConfig::load()?;cfg.models.main=name.clone();cfg.save()?;self.provider.set_model(name.clone());Ok(name)}
 pub async fn submit(&self,prompt:String)->Result<mpsc::UnboundedReceiver<AgentEvent>> {
  self.interrupt.reset();
  if self.session.lock().await.history.len() > self.config.sessions.max_history { let _ = self.compact().await; }
  let session_snapshot={let mut s=self.session.lock().await;if s.title.trim().is_empty()||s.title=="untitled"{s.title=SessionData::autotitle_from(&prompt);}s.touch();let _=self.store.save(&s).await;s.clone()};
  let route=self.route_hyprfast(&prompt);if let Some(r)=&route{tracing::debug!(strategy=%r.strategy,candidates=r.candidates.len(),fast_path=r.fast_path,"HyprFast route selected");}
  let source=if self.hyprfast.is_some(){match self.plan_and_execute(prompt.clone(),session_snapshot.history.clone(),route.clone()).await{Ok(rx)=>rx,Err(e)=>{tracing::warn!(error=%e,"hierarchical planner failed; falling back to general agent");self.agent.execute_with_history_filtered(prompt,session_snapshot.history,Some(self.working_dir.clone()),self.interrupt.clone(),route.map(|r|r.candidates.into_iter().collect()).unwrap_or_default()).await?}}}else{self.agent.execute_with_history_filtered(prompt,session_snapshot.history,Some(self.working_dir.clone()),self.interrupt.clone(),route.map(|r|r.candidates.into_iter().collect()).unwrap_or_default()).await?};
  let(tx,rx)=mpsc::unbounded_channel();let session_store=self.session.clone();let store=self.store.clone();let max_history=self.config.sessions.max_history;let owner_id=session_snapshot.session_id.clone();tokio::spawn(async move{let mut source=source;while let Some(event)=source.recv().await{if let AgentEvent::History{message}=&event{let current_id=session_store.lock().await.session_id.clone();if current_id==owner_id{let mut session=session_store.lock().await;session.history.push(message.clone());trim_history(&mut session.history,max_history);session.updated_at=now();let _=store.save(&session).await;}else{if let Ok(mut owner)=store.load(&owner_id).await{owner.history.push(message.clone());trim_history(&mut owner.history,max_history);owner.updated_at=now();let _=store.save(&owner).await;}}}let _=tx.send(event);}});Ok(rx)
 }
 async fn plan_and_execute(&self,prompt:String,history:Vec<TurnMessage>,route:Option<lucy_hyprfast::Route>)->Result<mpsc::UnboundedReceiver<AgentEvent>>{
  let(tx,rx)=mpsc::unbounded_channel();let provider=self.provider.clone();let registry=self.registry.clone();let catalog=self.hyprfast.clone().ok_or_else(||anyhow!("HyprFast catalog unavailable"))?;let interrupt=self.interrupt.clone();let working_dir=self.working_dir.clone();let planner_cfg=self.config.planner.clone();let verify_actions=self.config.hyprfast.verify_actions;let main_model=self.provider.model();let command_model=self.config.models.hyprfast_command.clone();let approvals=self.approvals.clone();
  tokio::spawn(async move{let run=async{
    let _=tx.send(AgentEvent::History{message:TurnMessage::User(prompt.clone())});
    let _=tx.send(AgentEvent::Status{message:"Understanding your request…".into()});
    let _=tx.send(AgentEvent::Progress{message:"Understanding your request…".into()});
    let triage=triage_request(&provider,&main_model,&prompt,&catalog,&route,&history,&interrupt).await?;
    if triage.mode=="chat"{let text=triage.reply.filter(|r|!r.trim().is_empty()).unwrap_or_else(||"Done.".to_string());let assistant=TurnMessage::Assistant(AssistantTurn{text:Some(text.clone()),tool_calls:Vec::new()});let _=tx.send(AgentEvent::History{message:assistant});let _=tx.send(AgentEvent::TextDelta{text});Ok::<(),anyhow::Error>(())}
    else if triage.mode!="act"{return Err(anyhow!("main model returned unknown triage mode: {}",triage.mode));}
    else {
      if triage.subtasks.is_empty()||triage.subtasks.len()>planner_cfg.max_subtasks{return Err(anyhow!("main model returned an invalid subtask count"))}
      validate_plan(&triage.subtasks)?;
      let _=tx.send(AgentEvent::Progress{message:format!("Plan ready: {} step{}…",triage.subtasks.len(),if triage.subtasks.len()==1{""}else{"s"})});
      let mut queue:VecDeque<SubTask>=triage.subtasks.into_iter().collect();let mut completed:HashMap<String,Value>=HashMap::new();let mut notes=Vec::new();let mut decisions=0usize;let mut done_count=0usize;let mut deferred=0usize;
      while let Some(subtask)=queue.pop_front(){
        if interrupt.is_set(){return Err(lucy_core::LucyError::Cancelled.into())}
        if decisions>=MAX_MAIN_DECISIONS{return Err(anyhow!("main model exceeded computer-operation decision budget"))}
        let deps_ready=subtask.depends_on.iter().all(|id|completed.contains_key(id));
        if !deps_ready{queue.push_back(subtask);deferred+=1;if deferred>=queue.len().max(1){return Err(anyhow!("execution plan is blocked by unresolved dependencies"));}continue;}deferred=0;
        let step_no=done_count+1;let _=tx.send(AgentEvent::Status{message:format!("Step {step_no}: {}",subtask.goal)});let _=tx.send(AgentEvent::Progress{message:format!("Step {step_no}: {}",subtask.goal)});
        let cat=subtask.category.to_ascii_lowercase();
        let(allowed,context)=if cat=="files"||cat=="shell"||cat=="system"{let names=registry.local_tool_names();if names.is_empty(){return Err(anyhow!("no local tools available for subtask: {}",subtask.goal))}let mut ctx=String::from("Local machine tools. Prefer read-only observation tools when only inspecting; never invent file contents or command output.\n");for id in &subtask.depends_on{if let Some(v)=completed.get(id){ctx.push_str(&format!("\nDependency {} result: {}",id,v));}}if ctx.len()>MAX_CONTEXT_CHARS{ctx.truncate(MAX_CONTEXT_CHARS);ctx.push_str("\n[context truncated]");}(names,ctx)}else{let subroute=catalog.route_domain(&subtask.category,&subtask.goal);let names:HashSet<String>=subroute.candidates.iter().cloned().collect();if names.is_empty(){return Err(anyhow!("no HyprFast tools matched subtask: {}",subtask.goal))}let ctx=build_context(&catalog,&subroute,&completed,&subtask.depends_on);(names,ctx)};
        let schemas=registry.definitions_for_names(&allowed);let command=plan_command(&provider,&command_model,&prompt,&subtask,&schemas,&context,&interrupt).await?;if !allowed.contains(&command.tool){return Err(anyhow!("command model selected tool outside routed capability set: {}",command.tool))}if !command.arguments.is_object(){return Err(anyhow!("command model arguments must be a JSON object"))}
        let call_id=format!("plan-{}",decisions+1);let gate=approvals.with_events(tx.clone());
        if gate.needs_approval(&command.tool,registry.requires_approval(&command.tool)){let _=tx.send(AgentEvent::Progress{message:format!("Needs your approval: {}…",command.tool)});match gate.ask(&call_id,&command.tool,&command.arguments).await{ApprovalDecision::AllowOnce|ApprovalDecision::AllowAlways=>{},ApprovalDecision::Deny=>{let denied=serde_json::json!({"denied by user":command.tool.clone()});let _=tx.send(AgentEvent::ToolFinished{id:call_id.clone(),name:command.tool.clone(),output:denied.clone(),is_error:true});if !planner_cfg.replan_on_failure{return Err(anyhow!("subtask denied: {}",subtask.goal))}let _=tx.send(AgentEvent::Progress{message:format!("You denied {} — replanning…",command.tool)});completed.insert(format!("{}_failure",subtask.id),denied.clone());let decision=decide_next(&provider,&main_model,&prompt,&queue,&completed,&subtask,&denied,&catalog,&interrupt).await?;decisions+=1;if let Some(s)=decision.subtask{validate_plan(std::slice::from_ref(&s))?;queue.push_front(s)}continue;}}}
        let _=tx.send(AgentEvent::ToolStarted{id:call_id.clone(),name:command.tool.clone(),input:command.arguments.clone()});let ctx=ToolContext{session_id:SessionId::default(),tool_call_id:call_id.clone(),working_dir:Some(working_dir.clone()),execution_mode:ExecutionMode::Agent,events:tx.clone(),interrupt:interrupt.clone()};
        let result=registry.execute(&command.tool,command.arguments.clone(),ctx).await;let output=match result{Ok(v)=>{let _=tx.send(AgentEvent::ToolFinished{id:call_id.clone(),name:command.tool.clone(),output:v.clone(),is_error:false});v},Err(e)=>{let err=serde_json::json!({"error":e.to_string()});let _=tx.send(AgentEvent::ToolFinished{id:call_id.clone(),name:command.tool.clone(),output:err.clone(),is_error:true});if !planner_cfg.replan_on_failure{return Err(anyhow!("subtask failed: {}: {}",subtask.goal,e))}let _=tx.send(AgentEvent::Progress{message:format!("Step failed ({e}) — replanning…")});completed.insert(format!("{}_failure",subtask.id),err.clone());let decision=decide_next(&provider,&main_model,&prompt,&queue,&completed,&subtask,&err,&catalog,&interrupt).await?;decisions+=1;if let Some(s)=decision.subtask{validate_plan(std::slice::from_ref(&s))?;queue.push_front(s)}continue;}};
        let state=truncate_json(output.clone());completed.insert(subtask.id.clone(),state.clone());notes.push(subtask.goal.clone());done_count+=1;let _=tx.send(AgentEvent::History{message:TurnMessage::Tool(ToolResult{call_id,name:command.tool.clone(),output:state.clone(),is_error:false})});
        let must_verify=verify_actions&&planner_cfg.verify_state&&action_requires_verification(&catalog,&command.tool,&subtask.category,command.verify.is_some())&&!subtask.id.starts_with("verify-");
        if must_verify{let _=tx.send(AgentEvent::Status{message:"Verifying the result…".into()});let _=tx.send(AgentEvent::Progress{message:"Verifying the result…".into()});queue.push_front(verification_subtask(&catalog,&command.tool,&subtask.goal,&subtask.category,decisions+1));continue;}
        // Successful planned steps do not need a main-model round trip. The initial
        // plan already owns strategy; the controller is reserved for failures,
        // denials, and verification results where new evidence can change the plan.
        if subtask.id.starts_with("verify-"){decisions+=1;let decision=decide_next(&provider,&main_model,&prompt,&queue,&completed,&subtask,&state,&catalog,&interrupt).await?;if let Some(s)=decision.subtask{validate_plan(std::slice::from_ref(&s))?;queue.push_front(s)}else if matches!(decision.decision,DecisionKind::Replan)&&queue.is_empty(){return Err(anyhow!("main model requested replanning but supplied no recovery subtask"));}else if matches!(decision.decision,DecisionKind::Complete){let _=tx.send(AgentEvent::Status{message:format!("Done: {}",decision.reason)});let _=tx.send(AgentEvent::Progress{message:format!("Done: {}",decision.reason)});break;}}
      }
      let capped_results=truncate_json(serde_json::to_value(&completed).unwrap_or(Value::Null)).to_string();let summary_user=format!("Task:\n{}\n\nSteps performed:\n{}\n\nCompleted step results/state:\n{}\n\nSummarize what was actually done and the outcome, for the user. Only describe what the results evidence.",prompt,notes.join("\n"),capped_results);let text=match provider.complete_json(&main_model,"You are Lucy, a warm and friendly AI assistant. Summarize completed computer work in English. Be concise: two to four short sentences, plus a '- ' bullet list only if several distinct things were done. Format for a plain-text terminal: short paragraphs separated by blank lines, one list item per line starting with '- ', no **bold** markers and no backticks. Never claim actions that are not evidenced by the results. Reply ONLY JSON: {\"summary\":\"...\"}.",&summary_user,interrupt.clone()).await{Ok(v)=>v.get("summary").and_then(|s|s.as_str()).map(str::to_owned).filter(|s|!s.trim().is_empty()).unwrap_or_else(||format!("Completed {} step(s).",notes.len())),Err(_)=>format!("Completed {} step(s).",notes.len()),};let _=tx.send(AgentEvent::Progress{message:text.clone()});let assistant=TurnMessage::Assistant(AssistantTurn{text:Some(text.clone()),tool_calls:Vec::new()});let _=tx.send(AgentEvent::History{message:assistant});let _=tx.send(AgentEvent::TextDelta{text});Ok::<(),anyhow::Error>(())
    }
  };if let Err(e)=run.await{let _=tx.send(AgentEvent::Error{message:e.to_string()});}let _=tx.send(AgentEvent::Done);});Ok(rx)
 }
 pub async fn history(&self)->Vec<TurnMessage>{self.session.lock().await.history.clone()}
}
