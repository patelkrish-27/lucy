use std::{collections::{HashMap, HashSet, VecDeque}, path::PathBuf, sync::Arc};
use anyhow::{anyhow, Result};
use lucy_agent::{Agent, OpenAIProvider};
use lucy_config::LucyConfig;
use lucy_core::{AgentEvent, InterruptSignal, SessionData, SessionId, ToolContext, TurnMessage, AssistantTurn, ToolResult, ExecutionMode};
use lucy_hyprfast::HyprFastCatalog;
use lucy_mcp::{load_config, register_server};
use lucy_tools::{default_registry, ToolRegistry};
use serde::Deserialize;
use serde_json::Value;
use tokio::sync::{mpsc, Mutex};

const MAX_CONTEXT_CHARS: usize = 16_000;
const MAX_MAIN_DECISIONS: usize = 64;

pub struct LucyRuntime { agent: Arc<Agent<OpenAIProvider>>, registry: Arc<ToolRegistry>, provider: Arc<OpenAIProvider>, session: Arc<Mutex<SessionData>>, session_path: PathBuf, interrupt: InterruptSignal, working_dir: PathBuf, hyprfast: Option<HyprFastCatalog>, config: LucyConfig }
#[derive(Debug, Clone, Deserialize)] struct Decomposition { #[serde(default)] subtasks: Vec<SubTask> }
#[derive(Debug, Clone, Deserialize)] struct SubTask { id: String, goal: String, #[serde(default)] category: String, #[serde(default)] depends_on: Vec<String> }
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
  let path=config.sessions.file.clone().or_else(||std::env::var("LUCY_SESSION_FILE").ok().map(PathBuf::from)).unwrap_or_else(||PathBuf::from(std::env::var("HOME").unwrap_or_else(|_|".".into())).join(".local/state/lucy/session.json"));
  let session=if config.sessions.resume{match SessionData::load_from_file(&path).await{Ok(s)=>s,Err(e)=>{if path.exists(){return Err(e)}SessionData::new(SessionId::default())}}}else{SessionData::new(SessionId::default())};
  let registry=Arc::new(registry);let agent=Arc::new(Agent::new(provider.clone(),registry.clone()));
  Ok(Self{agent,registry,provider,session:Arc::new(Mutex::new(session)),session_path:path,interrupt:InterruptSignal::new(),working_dir:std::env::current_dir()?,hyprfast,config})
 }
 pub fn interrupt(&self){self.interrupt.fire()}
 pub fn hyprfast_catalog(&self)->Option<&HyprFastCatalog>{self.hyprfast.as_ref()}
 pub fn route_hyprfast(&self,prompt:&str)->Option<lucy_hyprfast::Route>{self.hyprfast.as_ref().map(|c|c.route(prompt))}
 pub fn config(&self)->&LucyConfig{&self.config}
 pub async fn submit(&self,prompt:String)->Result<mpsc::UnboundedReceiver<AgentEvent>> {
  self.interrupt.reset();let session=self.session.lock().await.clone();let route=self.route_hyprfast(&prompt);if let Some(r)=&route{tracing::debug!(strategy=%r.strategy,candidates=r.candidates.len(),fast_path=r.fast_path,"HyprFast route selected");}
  let use_hierarchical=self.hyprfast.is_some()&&looks_like_computer_task(&prompt);
  let source=if use_hierarchical{match self.plan_and_execute(prompt.clone(),session.history.clone(),route.clone()).await{Ok(rx)=>rx,Err(e)=>{tracing::warn!(error=%e,"hierarchical HyprFast planner failed; falling back to general agent");self.agent.execute_with_history_filtered(prompt,session.history,Some(self.working_dir.clone()),self.interrupt.clone(),route.map(|r|r.candidates.into_iter().collect()).unwrap_or_default()).await?}}}else{self.agent.execute_with_history_filtered(prompt,session.history,Some(self.working_dir.clone()),self.interrupt.clone(),route.map(|r|r.candidates.into_iter().collect()).unwrap_or_default()).await?};
  let(tx,rx)=mpsc::unbounded_channel();let session_store=self.session.clone();let path=self.session_path.clone();let max_history=self.config.sessions.max_history;tokio::spawn(async move{let mut source=source;while let Some(event)=source.recv().await{if let AgentEvent::History{message}=&event{let mut session=session_store.lock().await;session.history.push(message.clone());if session.history.len()>max_history{let drop_n=session.history.len()-max_history;session.history.drain(0..drop_n);}session.updated_at=now();let _=session.save_to_file(&path).await;}let _=tx.send(event);}});Ok(rx)
 }
 async fn plan_and_execute(&self,prompt:String,history:Vec<TurnMessage>,route:Option<lucy_hyprfast::Route>)->Result<mpsc::UnboundedReceiver<AgentEvent>>{
  let(tx,rx)=mpsc::unbounded_channel();let provider=self.provider.clone();let registry=self.registry.clone();let catalog=self.hyprfast.clone().ok_or_else(||anyhow!("HyprFast catalog unavailable"))?;let interrupt=self.interrupt.clone();let working_dir=self.working_dir.clone();let planner_cfg=self.config.planner.clone();let main_model=self.config.models.main.clone();let command_model=self.config.models.hyprfast_command.clone();
  tokio::spawn(async move{let run=async{
    let _=tx.send(AgentEvent::History{message:TurnMessage::User(prompt.clone())});
    let _=tx.send(AgentEvent::Status{message:format!("Understanding with {}…",main_model)});
    let decomposition=decompose(&provider,&main_model,&prompt,&catalog,&route,&history,&interrupt).await?;
    if decomposition.subtasks.is_empty()||decomposition.subtasks.len()>planner_cfg.max_subtasks{return Err(anyhow!("main model returned an invalid subtask count"))}
    let mut queue:VecDeque<SubTask>=decomposition.subtasks.into_iter().collect();
    let mut completed:HashMap<String,Value>=HashMap::new();
    let mut notes=Vec::new();
    let mut decisions=0usize;
    while let Some(subtask)=queue.pop_front(){
      if interrupt.is_set(){return Err(lucy_core::LucyError::Cancelled.into())}
      if decisions>=MAX_MAIN_DECISIONS{return Err(anyhow!("main model exceeded computer-operation decision budget"))}
      let _=tx.send(AgentEvent::Status{message:format!("Preparing: {}",subtask.goal)});
      let subroute=catalog.route_domain(&subtask.category,&subtask.goal);
      let allowed:HashSet<String>=subroute.candidates.iter().cloned().collect();
      if allowed.is_empty(){return Err(anyhow!("no HyprFast tools matched subtask: {}",subtask.goal))}
      let schemas=registry.definitions_for_names(&allowed);
      let context=build_context(&catalog,&subroute,&completed,&subtask.depends_on);
      let command=plan_command(&provider,&command_model,&prompt,&subtask,&schemas,&context,&interrupt).await?;
      if !allowed.contains(&command.tool){return Err(anyhow!("command model selected tool outside routed capability set: {}",command.tool))}
      if !command.arguments.is_object(){return Err(anyhow!("command model arguments must be a JSON object"))}
      let call_id=format!("plan-{}",decisions+1);
      let _=tx.send(AgentEvent::ToolStarted{id:call_id.clone(),name:command.tool.clone()});
      let ctx=ToolContext{session_id:SessionId::default(),tool_call_id:call_id.clone(),working_dir:Some(working_dir.clone()),execution_mode:ExecutionMode::Agent,events:tx.clone(),interrupt:interrupt.clone()};
      let result=registry.execute(&command.tool,command.arguments.clone(),ctx).await;
      let output=match result{
        Ok(v)=>{let _=tx.send(AgentEvent::ToolFinished{id:call_id.clone(),output:v.clone()});v},
        Err(e)=>{let err=serde_json::json!({"error":e.to_string()});let _=tx.send(AgentEvent::ToolFinished{id:call_id.clone(),output:err.clone()});
          if !planner_cfg.replan_on_failure{return Err(anyhow!("subtask failed: {}: {}",subtask.goal,e))}
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
      let _=tx.send(AgentEvent::History{message:TurnMessage::Tool(ToolResult{call_id,name:command.tool.clone(),output:state.clone(),is_error:false})});
      if planner_cfg.verify_state && should_verify(&subtask,&command){
        let _=tx.send(AgentEvent::Status{message:"Checking resulting UI state…".into()});
      }
      decisions+=1;
      let decision=decide_next(&provider,&main_model,&prompt,&queue,&completed,&subtask,&state,&catalog,&interrupt).await?;
      match decision.decision{
        DecisionKind::Complete=>{let _=tx.send(AgentEvent::Status{message:format!("Completed: {}",decision.reason)});break},
        DecisionKind::Continue=>{
          if let Some(next)=decision.subtask{queue.push_front(next)}
        },
        DecisionKind::Replan=>{
          if let Some(next)=decision.subtask{queue.push_front(next)}else if queue.is_empty(){return Err(anyhow!("main model requested replanning but supplied no recovery subtask"))}
        }
      }
    }
    let text=format!("Completed {} computer-operation step(s).",notes.len());
    let assistant=TurnMessage::Assistant(AssistantTurn{text:Some(text.clone()),tool_calls:Vec::new()});
    let _=tx.send(AgentEvent::History{message:assistant});let _=tx.send(AgentEvent::TextDelta{text});Ok::<(),anyhow::Error>(())
  };if let Err(e)=run.await{let _=tx.send(AgentEvent::Error{message:e.to_string()});}let _=tx.send(AgentEvent::Done);});Ok(rx)
 }
 pub async fn history(&self)->Vec<TurnMessage>{self.session.lock().await.history.clone()}
}

async fn decompose(provider:&OpenAIProvider,model:&str,prompt:&str,catalog:&HyprFastCatalog,route:&Option<lucy_hyprfast::Route>,history:&[TurnMessage],interrupt:&InterruptSignal)->Result<Decomposition>{
 let route_context=route.as_ref().map(|r|catalog.context_for(r)).unwrap_or_default();
 let recent_history=history.iter().rev().take(8).map(|m|format!("{:?}",m)).collect::<Vec<_>>().join("\n");
 let system=r#"You are Lucy's primary computer-operation planner. You own understanding, strategy, state, dependencies, verification and recovery. Break the user's request into ordered subtasks, where each subtask is small enough for exactly one HyprFast command. When current UI state matters, START with an observation subtask rather than assuming an app, window, tab, canvas, element, coordinate or focus. Use dependencies to pass observation results to later actions. Do not choose tool names or arguments. Return ONLY JSON: {\"subtasks\":[{\"id\":\"1\",\"goal\":\"precise action or observation\",\"category\":\"browser|desktop|vision|excalidraw|clipboard|tasks|stagehand|hints\",\"depends_on\":[]}]}"#;
 let user=format!("Original task:\n{}\n\nRecent session context:\n{}\n\nInitial HyprFast context:\n{}",prompt,recent_history,route_context);
 Ok(serde_json::from_value(provider.complete_json(model,system,&user,interrupt.clone()).await?)?)
}

async fn decide_next(provider:&OpenAIProvider,model:&str,prompt:&str,remaining:&VecDeque<SubTask>,completed:&HashMap<String,Value>,last_subtask:&SubTask,last_result:&Value,catalog:&HyprFastCatalog,interrupt:&InterruptSignal)->Result<MainDecision>{
 let remaining_json=serde_json::to_string(remaining)?;let completed_json=serde_json::to_string(completed)?;let system=r#"You are Lucy's primary closed-loop computer-operation controller. Decide what should happen AFTER the last HyprFast command. You are the only model allowed to reason about overall strategy. Inspect the actual result/state; never assume success merely because a tool returned. If the goal is complete, return complete. If the remaining plan is still valid, return continue with no subtask. If state differs, information is missing, or an action failed, return replan with ONE concrete observation/recovery/action subtask. A replan subtask must be executable by one HyprFast command. Prefer observing before acting when state is uncertain. Do not choose HyprFast tool names or arguments. Return ONLY JSON: {\"decision\":\"continue|replan|complete\",\"subtask\":null or {\"id\":\"replan-1\",\"goal\":\"...\",\"category\":\"...\",\"depends_on\":[]},\"reason\":\"short reason\"}."#;
 let user=format!("Task:\n{}\n\nLast subtask:\n{}\nLast result/state:\n{}\n\nRemaining planned subtasks:\n{}\n\nAll completed results/state:\n{}\n\nHyprFast capability summary:\n{}",prompt,serde_json::to_string(last_subtask)?,last_result,remaining_json,completed_json,serde_json::to_string(&catalog.summary())?);
 Ok(serde_json::from_value(provider.complete_json(model,system,&user,interrupt.clone()).await?)?)
}

async fn plan_command(provider:&OpenAIProvider,model:&str,prompt:&str,subtask:&SubTask,schemas:&[Value],context:&str,interrupt:&InterruptSignal)->Result<PlannedCommand>{let system=r#"You are Lucy's fast HyprFast command compiler. This is your ONLY job. You receive ONE already-planned subtask from Lucy's primary model, a small set of allowed HyprFast tools, their exact JSON schemas, and execution context. Select exactly ONE allowed tool and produce exact arguments conforming to its schema. Do not redesign the task, decompose it, invent state, or make strategic decisions. Do not invent fields, tool names, tabs, windows, coordinates, IDs, or other state. If the provided context is insufficient, select an allowed observation tool instead. Return ONLY JSON: {\"tool\":\"exact allowed tool name\",\"arguments\":{},\"verify\":\"optional short verification\"}."#;let tools=serde_json::to_string(schemas)?;let user=format!("Original task (context only):\n{}\n\nSubtask category: {}\nSubtask: {}\nDependencies: {:?}\n\nCurrent HyprFast context:\n{}\n\nAllowed tools and schemas:\n{}",prompt,subtask.category,subtask.goal,subtask.depends_on,context,tools);Ok(serde_json::from_value(provider.complete_json(model,system,&user,interrupt.clone()).await?)?)}
fn build_context(catalog:&HyprFastCatalog,route:&lucy_hyprfast::Route,completed:&HashMap<String,Value>,depends_on:&[String])->String{let mut out=catalog.context_for(route);for id in depends_on{if let Some(v)=completed.get(id){out.push_str(&format!("\nDependency {} result: {}",id,v));}}if out.len()>MAX_CONTEXT_CHARS{out.truncate(MAX_CONTEXT_CHARS);out.push_str("\n[context truncated]");}out}
fn truncate_json(v:Value)->Value{let s=v.to_string();if s.len()<=MAX_CONTEXT_CHARS{return v}let end=s.char_indices().nth(MAX_CONTEXT_CHARS).map(|(i,_)|i).unwrap_or(s.len());serde_json::json!({"truncated":true,"preview":&s[..end]})}
fn should_verify(subtask:&SubTask,command:&PlannedCommand)->bool{command.verify.is_some()||matches!(subtask.category.as_str(),"vision"|"browser"|"excalidraw")}
fn looks_like_computer_task(text:&str)->bool{let t=text.to_ascii_lowercase();["browser","brave","chrome","firefox","tab","window","workspace","desktop","screen","screenshot","click","type","keyboard","mouse","excalidraw","draw","diagram","clipboard","copy","paste","open app","launch app"].iter().any(|x|t.contains(x))}
fn now()->u64{std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()}
