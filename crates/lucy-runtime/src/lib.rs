use std::{collections::{HashMap, HashSet, VecDeque}, path::PathBuf, sync::Arc};
use anyhow::{anyhow, Result};
use lucy_agent::{Agent, OpenAIProvider};
use lucy_config::LucyConfig;
use lucy_core::{AgentEvent, ApprovalDecision, ApprovalGate, InterruptSignal, SessionData, SessionId, SessionMeta, SessionStore, TokenUsage, ToolContext, TurnMessage, AssistantTurn, ToolResult, ExecutionMode};
use lucy_hyprfast::HyprFastCatalog;
use lucy_mcp::{load_config, register_server};
use lucy_tools::{default_registry, ToolRegistry};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{mpsc, Mutex};

const MAX_CONTEXT_CHARS: usize = 16_000;
const MAX_MAIN_DECISIONS: usize = 64;

pub struct LucyRuntime { agent: Arc<Agent<OpenAIProvider>>, registry: Arc<ToolRegistry>, provider: Arc<OpenAIProvider>, session: Arc<Mutex<SessionData>>, store: SessionStore, #[allow(dead_code)] legacy_path: PathBuf, interrupt: InterruptSignal, working_dir: PathBuf, hyprfast: Option<HyprFastCatalog>, config: LucyConfig, approvals: ApprovalGate }
#[derive(Debug, Clone, Deserialize, Serialize)] struct SubTask { id: String, goal: String, #[serde(default)] category: String, #[serde(default)] depends_on: Vec<String> }
#[derive(Debug, Clone, Deserialize)] struct PlannedCommand { tool: String, arguments: Value, #[serde(default)] verify: Option<String> }
#[derive(Debug, Clone, Deserialize)] #[serde(rename_all = "snake_case")] enum DecisionKind { Continue, Replan, Complete }
#[derive(Debug, Clone, Deserialize)] struct MainDecision { decision: DecisionKind, #[serde(default)] subtask: Option<SubTask>, #[serde(default)] reason: String }

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
  // Opencode parity: migrate legacy single file once, then work purely in the store.
  let _ = store.migrate_legacy_file(&legacy_path).await;
  let session=if config.sessions.resume{
    match store.list().await{
      Ok(list) if !list.is_empty()=>{
        // Most recent session wins (list is newest-first).
        match store.load(&list[0].id).await{Ok(s)=>s,Err(_)=>store.create(None).await?}
      },
      _=>store.create(None).await?,
    }
  }else{
    store.create(None).await?
  };
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
  pub async fn clear_session(&self)->anyhow::Result<()>{let mut s=self.session.lock().await;s.session_id=SessionId::default();s.history.clear();s.updated_at=now();self.store.save(&s).await?;Ok(())}
  pub async fn compact(&self)->anyhow::Result<String>{
   let max=self.config.sessions.max_history;
   let(before,recent,dropped)={let s=self.session.lock().await;if s.history.len()<=max{return Ok("nothing to compact".to_string());}let total=s.history.len();let split=total.saturating_sub(20);(total,s.history[split..].to_vec(),s.history[..split].to_vec())};
   let full=format!("{:?}",dropped);let transcript: String=full.chars().take(12000).collect();
   let model=self.provider.model();
   let summary=self.provider.complete_json(&model,"Summarize this agent conversation prefix in English into 10 dense bullets: decisions, file changes, tool results, open tasks. Plain text only, no markdown.",&transcript,InterruptSignal::new()).await?;
   let summary_text=match summary{Value::String(s)=>s,other=>other.to_string()};
   let mut s=self.session.lock().await;
   let mut history=Vec::with_capacity(recent.len()+1);
   history.push(TurnMessage::User(format!("Conversation summary so far:\n{summary_text}")));
   history.extend(recent);
   s.history=history;s.updated_at=now();self.store.save(&s).await?;
   Ok(format!("compacted {before} -> {} messages",s.history.len()))
  }
 pub fn sessions_dir(&self)->&PathBuf{&self.store.dir}
 // ---- opencode-style session API ----
 pub async fn current_meta(&self)->SessionMeta{SessionMeta::from(&*self.session.lock().await)}
 pub async fn current_id(&self)->SessionId{self.session.lock().await.session_id.clone()}
 pub async fn list_sessions(&self)->Result<Vec<SessionMeta>>{self.store.list().await}
 pub async fn new_session(&self,title:Option<String>)->Result<SessionMeta>{
   self.interrupt.reset();
   let s=self.store.create(title).await?;
   let meta=SessionMeta::from(&s);
   *self.session.lock().await=s;
   Ok(meta)
 }
 pub async fn switch_session(&self,id:&SessionId)->Result<SessionMeta>{
   self.interrupt.reset();
   let s=self.store.load(id).await?;
   let meta=SessionMeta::from(&s);
   *self.session.lock().await=s;
   Ok(meta)
 }
 pub async fn rename_current(&self,title:String)->Result<SessionMeta>{
   let mut s=self.session.lock().await;
   let t=title.trim().to_owned();
   if !t.is_empty(){s.title=t;}
   s.updated_at=now();
   self.store.save(&s).await?;
   Ok(SessionMeta::from(&*s))
 }
 pub async fn delete_session(&self,id:&SessionId)->Result<bool>{
   let current=self.session.lock().await.session_id.clone();
   let was_current=current==*id;
   self.store.delete(id).await?;
   if was_current{
     let remaining=self.store.list().await.unwrap_or_default();
     let next=if let Some(m)=remaining.first(){self.store.load(&m.id).await?}else{self.store.create(None).await?};
     *self.session.lock().await=next;
   }
   Ok(was_current)
 }
 pub async fn fork_current(&self)->Result<SessionMeta>{
   let src=self.session.lock().await.clone();
   let forked=self.store.fork(&src,"(fork)").await?;
   let meta=SessionMeta::from(&forked);
   *self.session.lock().await=forked;
   Ok(meta)
 }
 pub async fn clear_current(&self)->Result<SessionMeta>{
   let mut s=self.session.lock().await;
   s.history.clear();
   s.updated_at=now();
   self.store.save(&s).await?;
   Ok(SessionMeta::from(&*s))
 }
 /// Compact history to the last `keep` messages (opencode `/compact` analogue).
 /// Returns (before, after).
 pub async fn compact_current(&self,keep:usize)->Result<(usize,usize)>{
   let mut s=self.session.lock().await;
   let before=s.history.len();
   let keep=keep.max(1);
   if before>keep{
     s.history.drain(0..before-keep);
   }
   s.updated_at=now();
   self.store.save(&s).await?;
   Ok((before,s.history.len()))
 }
 pub async fn export_current(&self,path:PathBuf)->Result<()>{
   let s=self.session.lock().await;
   s.save_to_file(&path).await
 }
  pub async fn submit(&self,prompt:String)->Result<mpsc::UnboundedReceiver<AgentEvent>> {
   self.interrupt.reset();
   if self.session.lock().await.history.len() > self.config.sessions.max_history { let _ = self.compact().await; }
  // Auto-title untitled sessions from the first prompt (opencode behaviour).
  let session_snapshot={let mut s=self.session.lock().await;if s.title.trim().is_empty()||s.title=="untitled"{s.title=SessionData::autotitle_from(&prompt);}s.touch();let _=self.store.save(&s).await;s.clone()};
  let route=self.route_hyprfast(&prompt);if let Some(r)=&route{tracing::debug!(strategy=%r.strategy,candidates=r.candidates.len(),fast_path=r.fast_path,"HyprFast route selected");}
  // Every command flows through the same hierarchical loop: the main model
  // triages (chat vs act) with system prompt + history + new request, plans
  // subtasks, the cheap model compiles each subtask to one command, and the
  // main model executes, verifies and recovers. The general agent is only a
  // fallback when no desktop catalog (or the planner itself) is unavailable.
  let source=if self.hyprfast.is_some(){match self.plan_and_execute(prompt.clone(),session_snapshot.history.clone(),route.clone()).await{Ok(rx)=>rx,Err(e)=>{tracing::warn!(error=%e,"hierarchical planner failed; falling back to general agent");self.agent.execute_with_history_filtered(prompt,session_snapshot.history,Some(self.working_dir.clone()),self.interrupt.clone(),route.map(|r|r.candidates.into_iter().collect()).unwrap_or_default()).await?}}}else{self.agent.execute_with_history_filtered(prompt,session_snapshot.history,Some(self.working_dir.clone()),self.interrupt.clone(),route.map(|r|r.candidates.into_iter().collect()).unwrap_or_default()).await?};
  let(tx,rx)=mpsc::unbounded_channel();let session_store=self.session.clone();let store=self.store.clone();let max_history=self.config.sessions.max_history;let owner_id=session_snapshot.session_id.clone();tokio::spawn(async move{let mut source=source;while let Some(event)=source.recv().await{if let AgentEvent::History{message}=&event{
    // If the user switched sessions mid-run, persist to the owner session file
    // instead of corrupting the newly active conversation.
    let current_id=session_store.lock().await.session_id.clone();
    if current_id==owner_id{
      let mut session=session_store.lock().await;session.history.push(message.clone());trim_history(&mut session.history,max_history);session.updated_at=now();let _=store.save(&session).await;
    } else {
      if let Ok(mut owner)=store.load(&owner_id).await{owner.history.push(message.clone());trim_history(&mut owner.history,max_history);owner.updated_at=now();let _=store.save(&owner).await;}
    }
  }let _=tx.send(event);}});Ok(rx)
 }
 async fn plan_and_execute(&self,prompt:String,history:Vec<TurnMessage>,route:Option<lucy_hyprfast::Route>)->Result<mpsc::UnboundedReceiver<AgentEvent>>{
   let(tx,rx)=mpsc::unbounded_channel();let provider=self.provider.clone();let registry=self.registry.clone();let catalog=self.hyprfast.clone().ok_or_else(||anyhow!("HyprFast catalog unavailable"))?;let interrupt=self.interrupt.clone();let working_dir=self.working_dir.clone();let planner_cfg=self.config.planner.clone();let verify_actions=self.config.hyprfast.verify_actions;let main_model=self.provider.model();let command_model=self.config.models.hyprfast_command.clone();let approvals=self.approvals.clone();
  tokio::spawn(async move{let run=async{
    let _=tx.send(AgentEvent::History{message:TurnMessage::User(prompt.clone())});
    let _=tx.send(AgentEvent::Status{message:"Understanding your request…".into()});
    let _=tx.send(AgentEvent::Progress{message:"Understanding your request…".into()});
    // Step 1 — main model triage: answer directly (chat) or break down (act).
    let triage=triage_request(&provider,&main_model,&prompt,&catalog,&route,&history,&interrupt).await?;
    if triage.mode=="chat"{
      let text=triage.reply.filter(|r|!r.trim().is_empty()).unwrap_or_else(||"Done.".to_string());
      let assistant=TurnMessage::Assistant(AssistantTurn{text:Some(text.clone()),tool_calls:Vec::new()});
      let _=tx.send(AgentEvent::History{message:assistant});let _=tx.send(AgentEvent::TextDelta{text});Ok::<(),anyhow::Error>(())
    } else if triage.mode!="act"{
      return Err(anyhow!("main model returned unknown triage mode: {}",triage.mode));
    } else {
    if triage.subtasks.is_empty()||triage.subtasks.len()>planner_cfg.max_subtasks{return Err(anyhow!("main model returned an invalid subtask count"))}
    let _=tx.send(AgentEvent::Progress{message:format!("Plan ready: {} step{}…",triage.subtasks.len(),if triage.subtasks.len()==1{""}else{"s"})});
    let mut queue:VecDeque<SubTask>=triage.subtasks.into_iter().collect();
    let mut completed:HashMap<String,Value>=HashMap::new();
    let mut notes=Vec::new();
    let mut decisions=0usize;
    let mut done_count=0usize;
    while let Some(subtask)=queue.pop_front(){
      if interrupt.is_set(){return Err(lucy_core::LucyError::Cancelled.into())}
      if decisions>=MAX_MAIN_DECISIONS{return Err(anyhow!("main model exceeded computer-operation decision budget"))}
      let step_no=done_count+1;
      let _=tx.send(AgentEvent::Status{message:format!("Step {step_no}: {}",subtask.goal)});
      let _=tx.send(AgentEvent::Progress{message:format!("Step {step_no}: {}",subtask.goal)});
      // Step 2 — route the subtask to a small allowed tool set, then let the
      // cheap model compile it to exactly one command. files|shell subtasks
      // use built-in local tools; everything else uses the desktop catalog.
      let cat=subtask.category.to_ascii_lowercase();
      let (allowed,context)=if cat=="files"||cat=="shell"||cat=="system"{
        let names=registry.local_tool_names();
        if names.is_empty(){return Err(anyhow!("no local tools available for subtask: {}",subtask.goal))}
        let mut ctx=String::from("Local machine tools (read/write files, run shell commands, git). Prefer read-only observation tools when only inspecting; never invent file contents or command output.\n");
        for id in &subtask.depends_on{if let Some(v)=completed.get(id){ctx.push_str(&format!("\nDependency {} result: {}",id,v));}}
        if ctx.len()>MAX_CONTEXT_CHARS{ctx.truncate(MAX_CONTEXT_CHARS);ctx.push_str("\n[context truncated]");}
        (names,ctx)
      } else {
        let subroute=catalog.route_domain(&subtask.category,&subtask.goal);
        let names:HashSet<String>=subroute.candidates.iter().cloned().collect();
        if names.is_empty(){return Err(anyhow!("no HyprFast tools matched subtask: {}",subtask.goal))}
        let ctx=build_context(&catalog,&subroute,&completed,&subtask.depends_on);
        (names,ctx)
      };
      let schemas=registry.definitions_for_names(&allowed);
      let command=plan_command(&provider,&command_model,&prompt,&subtask,&schemas,&context,&interrupt).await?;
      if !allowed.contains(&command.tool){return Err(anyhow!("command model selected tool outside routed capability set: {}",command.tool))}
      if !command.arguments.is_object(){return Err(anyhow!("command model arguments must be a JSON object"))}
       let call_id=format!("plan-{}",decisions+1);
       let gate=approvals.with_events(tx.clone());
       if gate.needs_approval(&command.tool,registry.requires_approval(&command.tool)){
        let _=tx.send(AgentEvent::Progress{message:format!("Needs your approval: {}…",command.tool)});
        match gate.ask(&call_id,&command.tool,&command.arguments).await{
         ApprovalDecision::AllowOnce|ApprovalDecision::AllowAlways=>{},
         ApprovalDecision::Deny=>{
          let denied=serde_json::json!({"denied by user":command.tool.clone()});
          let _=tx.send(AgentEvent::ToolFinished{id:call_id.clone(),name:command.tool.clone(),output:denied.clone(),is_error:true});
          if !planner_cfg.replan_on_failure{return Err(anyhow!("subtask denied: {}",subtask.goal))}
          let _=tx.send(AgentEvent::Progress{message:format!("You denied {} — replanning…",command.tool)});
          completed.insert(format!("{}_failure",subtask.id),denied.clone());
          let decision=decide_next(&provider,&main_model,&prompt,&queue,&completed,&subtask,&denied,&catalog,&interrupt).await?;
          decisions+=1;
          if let Some(s)=decision.subtask{queue.push_front(s)}
          continue;
         }
        }
       }
       let _=tx.send(AgentEvent::ToolStarted{id:call_id.clone(),name:command.tool.clone(),input:command.arguments.clone()});
      let ctx=ToolContext{session_id:SessionId::default(),tool_call_id:call_id.clone(),working_dir:Some(working_dir.clone()),execution_mode:ExecutionMode::Agent,events:tx.clone(),interrupt:interrupt.clone()};
      let result=registry.execute(&command.tool,command.arguments.clone(),ctx).await;
      let output=match result{
        Ok(v)=>{let _=tx.send(AgentEvent::ToolFinished{id:call_id.clone(),name:command.tool.clone(),output:v.clone(),is_error:false});v},
        Err(e)=>{let err=serde_json::json!({"error":e.to_string()});let _=tx.send(AgentEvent::ToolFinished{id:call_id.clone(),name:command.tool.clone(),output:err.clone(),is_error:true});
          if !planner_cfg.replan_on_failure{return Err(anyhow!("subtask failed: {}: {}",subtask.goal,e))}
          let _=tx.send(AgentEvent::Progress{message:format!("Step failed ({e}) — replanning…")});
          completed.insert(format!("{}_failure",subtask.id),err.clone());
          let decision=decide_next(&provider,&main_model,&prompt,&queue,&completed,&subtask,&err,&catalog,&interrupt).await?;
          decisions+=1;
          if let Some(s)=decision.subtask{queue.push_front(s)}
          continue;
        }
      };
      let state=truncate_json(output.clone());
      completed.insert(subtask.id.clone(),state.clone());
      notes.push(subtask.goal.clone());
      done_count+=1;
      let _=tx.send(AgentEvent::History{message:TurnMessage::Tool(ToolResult{call_id,name:command.tool.clone(),output:state.clone(),is_error:false})});
      // Local read-only tools need no observation loop; everything else that
      // changes state is verified by the main model before reporting success.
      let must_verify=verify_actions&&planner_cfg.verify_state&&registry.requires_approval(&command.tool)&&action_requires_verification(&catalog,&command.tool,&subtask.category,command.verify.is_some())&&!subtask.id.starts_with("verify-");
      if must_verify{let _=tx.send(AgentEvent::Status{message:"Verifying the result…".into()});let _=tx.send(AgentEvent::Progress{message:"Verifying the result…".into()});}
      decisions+=1;
      // Step 3 — main model manages: confirm the work, continue, or recover.
      let decision=decide_next(&provider,&main_model,&prompt,&queue,&completed,&subtask,&state,&catalog,&interrupt).await?;
      if must_verify{
        if let Some(next)=decision.subtask{queue.push_front(next)}
        queue.push_front(verification_subtask(&catalog,&command.tool,&subtask.goal,&subtask.category,decisions));
        continue;
      }
      match decision.decision{
        DecisionKind::Complete=>{let _=tx.send(AgentEvent::Status{message:format!("Done: {}",decision.reason)});let _=tx.send(AgentEvent::Progress{message:format!("Done: {}",decision.reason)});break},
        DecisionKind::Continue=>{if let Some(next)=decision.subtask{let _=tx.send(AgentEvent::Progress{message:format!("Confirmed — next: {}",next.goal)});queue.push_front(next)}},
        DecisionKind::Replan=>{if let Some(next)=decision.subtask{let _=tx.send(AgentEvent::Progress{message:format!("Adjusting plan: {}",next.goal)});queue.push_front(next)}else if queue.is_empty(){return Err(anyhow!("main model requested replanning but supplied no recovery subtask"))}}
      }
    }
    // Finale — main model confirms the work in one short plain-English summary.
    let capped_results=truncate_json(serde_json::to_value(&completed).unwrap_or(Value::Null)).to_string();
    let summary_user=format!("Task:\n{}\n\nSteps performed:\n{}\n\nCompleted step results/state:\n{}\n\nSummarize what was actually done and the outcome, for the user. Only describe what the results evidence.",prompt,notes.join("\n"),capped_results);
    let text=match provider.complete_json(&main_model,"You are Lucy, a warm and friendly AI assistant. Summarize completed computer work in English. Be concise: two to four short sentences, plus a '- ' bullet list only if several distinct things were done. Format for a plain-text terminal: short paragraphs separated by blank lines, one list item per line starting with '- ', no **bold** markers and no backticks. Never claim actions that are not evidenced by the results. Reply ONLY JSON: {\"summary\":\"...\"}.",&summary_user,interrupt.clone()).await{
      Ok(v)=>v.get("summary").and_then(|s|s.as_str()).map(str::to_owned).filter(|s|!s.trim().is_empty()).unwrap_or_else(||format!("Completed {} step(s).",notes.len())),
      Err(_)=>format!("Completed {} step(s).",notes.len()),
    };
    let _=tx.send(AgentEvent::Progress{message:text.clone()});
    let assistant=TurnMessage::Assistant(AssistantTurn{text:Some(text.clone()),tool_calls:Vec::new()});
    let _=tx.send(AgentEvent::History{message:assistant});let _=tx.send(AgentEvent::TextDelta{text});Ok::<(),anyhow::Error>(())
    }
  };if let Err(e)=run.await{let _=tx.send(AgentEvent::Error{message:e.to_string()});}let _=tx.send(AgentEvent::Done);});Ok(rx)
 }
 pub async fn history(&self)->Vec<TurnMessage>{self.session.lock().await.history.clone()}
}

async fn parse_with_retry<T>(provider:&OpenAIProvider,model:&str,system:&str,user:&str,interrupt:&InterruptSignal)->Result<T> where T:for<'de> Deserialize<'de>{
 let v=provider.complete_json(model,system,user,interrupt.clone()).await?;
 match serde_json::from_value::<T>(v){
  Ok(t)=>Ok(t),
  Err(e)=>{let retry_user=format!("{user}\n\nYour previous output was invalid JSON: {e}. Return ONLY valid JSON matching the requested schema.");let v2=provider.complete_json(model,system,&retry_user,interrupt.clone()).await?;Ok(serde_json::from_value::<T>(v2)?)}
 }
}

#[derive(Debug, Clone, Deserialize)] struct Triage { #[serde(default)] mode:String, #[serde(default)] reply:Option<String>, #[serde(default)] subtasks:Vec<SubTask> }

async fn triage_request(provider:&OpenAIProvider,model:&str,prompt:&str,catalog:&HyprFastCatalog,route:&Option<lucy_hyprfast::Route>,history:&[TurnMessage],interrupt:&InterruptSignal)->Result<Triage>{
 let route_context=route.as_ref().map(|r|catalog.context_for(r)).unwrap_or_default();
 let recent_history=history.iter().rev().take(8).map(|m|format!("{:?}",m)).collect::<Vec<_>>().join("\n");
 let system=r#"You are Lucy's primary model. You own understanding, strategy, state, dependencies, verification and recovery for EVERY user request. Decide first: if the request needs NO tool use (greeting, question you can answer directly, acknowledgement), return {"mode":"chat","reply":"short warm friendly plain-English reply, two to four short sentences, no markdown"}. Otherwise return {"mode":"act","subtasks":[...]} breaking the request into ordered subtasks, each small enough for exactly ONE command. Write every goal in English as a short imperative phrase: it is shown to the user as live progress text. Categories: browser|desktop|vision|excalidraw|clipboard|tasks|stagehand|hints for on-screen computer work, files|shell for local files, commands and programs. When current UI state matters, START with an observation subtask rather than assuming an app, window, tab, canvas, element, coordinate or focus. Use dependencies to pass observation results to later actions. Do not choose tool names or arguments. Return ONLY JSON."#;
  let user=format!("New request:\n{}\n\nRecent session context:\n{}\n\nInitial tool context:\n{}",prompt,recent_history,route_context);
 let t:Triage=parse_with_retry(provider,model,system,&user,interrupt).await?;
 validate_triage(t)
}

fn validate_triage(t:Triage)->Result<Triage>{
 let mode=t.mode.to_ascii_lowercase();
 if mode!="chat"&&mode!="act"{return Err(anyhow!("main model returned unknown triage mode: {}",t.mode))}
 if mode=="act"&&t.subtasks.is_empty(){return Err(anyhow!("main model chose act but supplied no subtasks"))}
 Ok(Triage{mode,reply:t.reply,subtasks:t.subtasks})
}

async fn decide_next(provider:&OpenAIProvider,model:&str,prompt:&str,remaining:&VecDeque<SubTask>,completed:&HashMap<String,Value>,last_subtask:&SubTask,last_result:&Value,catalog:&HyprFastCatalog,interrupt:&InterruptSignal)->Result<MainDecision>{
 let remaining_json=serde_json::to_string(remaining)?;let completed_json=serde_json::to_string(completed)?;let system=r#"You are Lucy's primary closed-loop controller. Decide what should happen AFTER the last command. You are the only model allowed to reason about overall strategy. Inspect the actual result/state; never assume success merely because a tool returned. If the goal is complete, return complete. If the remaining plan is still valid, return continue with no subtask. If state differs, information is missing, or an action failed, return replan with ONE concrete observation/recovery/action subtask. A replan subtask must be executable by one command (categories: browser|desktop|vision|excalidraw|clipboard|tasks|stagehand|hints for on-screen work, files|shell for local files and programs). Prefer observing before acting when state is uncertain. Do not choose tool names or arguments. Write the reason in English as one short sentence, since it is shown to the user. Return ONLY JSON: {\"decision\":\"continue|replan|complete\",\"subtask\":null or {\"id\":\"replan-1\",\"goal\":\"...\",\"category\":\"...\",\"depends_on\":[]},\"reason\":\"short reason\"}."#;
  let user=format!("Task:\n{}\n\nLast subtask:\n{}\nLast result/state:\n{}\n\nRemaining planned subtasks:\n{}\n\nAll completed results/state:\n{}\n\nHyprFast capability summary:\n{}",prompt,serde_json::to_string(last_subtask)?,last_result,remaining_json,completed_json,serde_json::to_string(&catalog.summary())?);
  parse_with_retry(provider,model,system,&user,interrupt).await
}

async fn plan_command(provider:&OpenAIProvider,model:&str,prompt:&str,subtask:&SubTask,schemas:&[Value],context:&str,interrupt:&InterruptSignal)->Result<PlannedCommand>{let system=r#"You are Lucy's fast command compiler. This is your ONLY job. You receive ONE already-planned subtask from Lucy's primary model, a small set of allowed tools, their exact JSON schemas, and execution context. Select exactly ONE allowed tool and produce exact arguments conforming to its schema. Do not redesign the task, decompose it, invent state, or make strategic decisions. Do not invent fields, tool names, tabs, windows, coordinates, IDs, or other state. If the provided context is insufficient, select an allowed observation tool instead. For verification subtasks, choose a read-only observation tool and do not modify anything. Return ONLY JSON: {\"tool\":\"exact allowed tool name\",\"arguments\":{},\"verify\":\"optional short verification\"}."#;let tools=serde_json::to_string(schemas)?;let user=format!("Original task (context only):\n{}\n\nSubtask category: {}\nSubtask: {}\nDependencies: {:?}\n\nCurrent HyprFast context:\n{}\n\nAllowed tools and schemas:\n{}",prompt,subtask.category,subtask.goal,subtask.depends_on,context,tools);parse_with_retry(provider,model,system,&user,interrupt).await}
fn build_context(catalog:&HyprFastCatalog,route:&lucy_hyprfast::Route,completed:&HashMap<String,Value>,depends_on:&[String])->String{let mut out=catalog.context_for(route);for id in depends_on{if let Some(v)=completed.get(id){out.push_str(&format!("\nDependency {} result: {}",id,v));}}if out.len()>MAX_CONTEXT_CHARS{out.truncate(MAX_CONTEXT_CHARS);out.push_str("\n[context truncated]");}out}
fn truncate_json(v:Value)->Value{let s=v.to_string();if s.len()<=MAX_CONTEXT_CHARS{return v}let end=s.char_indices().nth(MAX_CONTEXT_CHARS).map(|(i,_)|i).unwrap_or(s.len());serde_json::json!({"truncated":true,"preview":&s[..end]})}
fn now()->u64{std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()}

fn action_requires_verification(catalog:&HyprFastCatalog,tool:&str,category:&str,explicit_verify:bool)->bool{
 if explicit_verify{return true;}
 let Some(capability)=catalog.capability_for_mcp_name(tool) else{return true};
 if capability.read_only{return false;}
 let _ = category;
 true
}

fn verification_category(catalog:&HyprFastCatalog,tool:&str,fallback:&str)->String{
 catalog.capability_for_mcp_name(tool).map(|capability|match capability.domain{
  lucy_hyprfast::Domain::Browser|lucy_hyprfast::Domain::Stagehand|lucy_hyprfast::Domain::Hints=>"browser".to_owned(),
  lucy_hyprfast::Domain::Excalidraw=>"excalidraw".to_owned(),
  lucy_hyprfast::Domain::Vision=>"vision".to_owned(),
  lucy_hyprfast::Domain::Desktop|lucy_hyprfast::Domain::Tasks|lucy_hyprfast::Domain::Clipboard|lucy_hyprfast::Domain::System|lucy_hyprfast::Domain::Unknown=>"desktop".to_owned(),
 }).unwrap_or_else(||fallback.to_owned())
}

fn verification_subtask(catalog:&HyprFastCatalog,tool:&str,original_goal:&str,fallback_category:&str,id:usize)->SubTask{
 SubTask{id:format!("verify-{id}"),goal:format!("Observe and verify the current state after: {original_goal}. Confirm whether the intended change actually happened; do not make another change."),category:verification_category(catalog,tool,fallback_category),depends_on:Vec::new()}
}

pub fn trim_history(history: &mut Vec<TurnMessage>, max_history: usize) {
    if history.len() <= max_history {
        return;
    }
    let drop_n = history.len() - max_history;
    if history.first().map_or(false, |m| matches!(m, TurnMessage::User(_))) && history.len() > 1 {
        let drain_count = drop_n.min(history.len() - 1);
        history.drain(1..1 + drain_count);
    } else {
        history.drain(0..drop_n);
    }
}

#[cfg(test)]
mod verification_tests{
 use super::*;
 use lucy_mcp::McpToolDefinition;
 fn tool(name:&str,description:&str)->McpToolDefinition{McpToolDefinition{name:name.into(),description:Some(description.into()),input_schema:serde_json::json!({"type":"object"})}}
 #[test]
 fn state_changing_tool_requires_verification(){let catalog=HyprFastCatalog::from_tools(vec![tool("browser_click","click browser element"),tool("screenshot","capture desktop")]);assert!(action_requires_verification(&catalog,"mcp_hyprfast_browser_click","browser",false));assert!(!action_requires_verification(&catalog,"mcp_hyprfast_screenshot","vision",false));assert!(action_requires_verification(&catalog,"unknown_tool","browser",false));}
 #[test]
 fn verification_uses_browser_domain_for_browser_actions(){let catalog=HyprFastCatalog::from_tools(vec![tool("browser_click","click browser element")]);let subtask=verification_subtask(&catalog,"mcp_hyprfast_browser_click","click the button","desktop",7);assert_eq!(subtask.id,"verify-7");assert_eq!(subtask.category,"browser");assert!(subtask.goal.contains("do not make another change"));}
 #[test]
 fn triage_accepts_chat_and_act_modes(){
  let chat:Triage=serde_json::from_value(serde_json::json!({"mode":"chat","reply":"Hello! How can I help?"})).expect("chat triage parses");
  let chat=validate_triage(chat).expect("chat validates");
  assert_eq!(chat.mode,"chat");
  assert!(chat.subtasks.is_empty());
  let act:Triage=serde_json::from_value(serde_json::json!({"mode":"act","subtasks":[{"id":"1","goal":"Open the browser","category":"browser"}]})).expect("act triage parses");
  let act=validate_triage(act).expect("act validates");
  assert_eq!(act.mode,"act");
  assert_eq!(act.subtasks.len(),1);
  assert_eq!(act.subtasks[0].category,"browser");
 }
 #[test]
 fn triage_rejects_unknown_mode_and_empty_act(){
  let unknown:Triage=serde_json::from_value(serde_json::json!({"mode":"dance"})).expect("parses");
  assert!(validate_triage(unknown).is_err());
  let empty:Triage=serde_json::from_value(serde_json::json!({"mode":"act","subtasks":[]})).expect("parses");
  assert!(validate_triage(empty).is_err());
 }
 #[test]
 fn files_shell_subtasks_use_local_tools(){
  use lucy_tools::default_registry;
  let registry=default_registry();
  let local=registry.local_tool_names();
  assert!(local.contains("shell"));
  assert!(local.contains("read_file"));
  assert!(!local.iter().any(|n|n.starts_with("mcp_")));
  let schemas=registry.definitions_for_names(&local);
  assert_eq!(schemas.len(),local.len());
  assert!(!schemas.is_empty());
 }
 #[test]
 fn trim_history_preserves_initial_user_prompt(){
     let mut hist = vec![
         TurnMessage::User("initial task".into()),
         TurnMessage::User("intermediate 1".into()),
         TurnMessage::User("intermediate 2".into()),
         TurnMessage::User("intermediate 3".into()),
         TurnMessage::User("latest turn".into()),
     ];
     trim_history(&mut hist, 3);
     assert_eq!(hist.len(), 3);
     assert!(matches!(&hist[0], TurnMessage::User(t) if t == "initial task"));
     assert!(matches!(&hist[1], TurnMessage::User(t) if t == "intermediate 3"));
     assert!(matches!(&hist[2], TurnMessage::User(t) if t == "latest turn"));
 }
}
