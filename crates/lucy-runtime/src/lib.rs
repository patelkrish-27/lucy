use std::{collections::{HashMap, HashSet}, path::PathBuf, sync::Arc};
use anyhow::{anyhow, Result};
use lucy_agent::{Agent, OpenAIProvider};
use lucy_config::LucyConfig;
use lucy_core::{AgentEvent, InterruptSignal, SessionData, SessionId, ToolContext, TurnMessage, AssistantTurn, ToolResult, ExecutionMode};
use lucy_hyprfast::{default_config, HyprFastCatalog};
use lucy_mcp::{load_config, register_server};
use lucy_tools::{default_registry, ToolRegistry};
use serde::Deserialize;
use serde_json::Value;
use tokio::sync::{mpsc, Mutex};

const DEFAULT_MAX_SUBTASKS: usize = 16;
const MAX_CONTEXT_CHARS: usize = 16_000;

pub struct LucyRuntime {
 agent: Arc<Agent<OpenAIProvider>>, registry: Arc<ToolRegistry>, provider: Arc<OpenAIProvider>,
 session: Arc<Mutex<SessionData>>, session_path: PathBuf, interrupt: InterruptSignal,
 working_dir: PathBuf, hyprfast: Option<HyprFastCatalog>, config: LucyConfig,
}
#[derive(Debug, Clone, Deserialize)] struct Decomposition { #[serde(default)] subtasks: Vec<SubTask> }
#[derive(Debug, Clone, Deserialize)] struct SubTask { id: String, goal: String, #[serde(default)] category: String, #[serde(default)] depends_on: Vec<String> }
#[derive(Debug, Clone, Deserialize)] struct PlannedCommand { tool: String, arguments: Value, #[serde(default)] verify: Option<String> }

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
  let use_planner=self.hyprfast.is_some()&&looks_like_computer_task(&prompt);
  let source=if use_planner{match self.plan_and_execute(prompt.clone(),session.history.clone(),route.clone()).await{Ok(rx)=>rx,Err(e)=>{tracing::warn!(error=%e,"hierarchical HyprFast planner failed; falling back to general agent");self.agent.execute_with_history_filtered(prompt,session.history,Some(self.working_dir.clone()),self.interrupt.clone(),route.map(|r|r.candidates.into_iter().collect()).unwrap_or_default()).await?}}}else{self.agent.execute_with_history_filtered(prompt,session.history,Some(self.working_dir.clone()),self.interrupt.clone(),route.map(|r|r.candidates.into_iter().collect()).unwrap_or_default()).await?};
  let(tx,rx)=mpsc::unbounded_channel();let session_store=self.session.clone();let path=self.session_path.clone();let max_history=self.config.sessions.max_history;tokio::spawn(async move{let mut source=source;while let Some(event)=source.recv().await{if let AgentEvent::History{message}=&event{let mut session=session_store.lock().await;session.history.push(message.clone());if session.history.len()>max_history{let drop_n=session.history.len()-max_history;session.history.drain(0..drop_n);}session.updated_at=now();let _=session.save_to_file(&path).await;}let _=tx.send(event);}});Ok(rx)
 }
 async fn plan_and_execute(&self,prompt:String,_history:Vec<TurnMessage>,route:Option<lucy_hyprfast::Route>)->Result<mpsc::UnboundedReceiver<AgentEvent>>{
  let(tx,rx)=mpsc::unbounded_channel();let provider=self.provider.clone();let registry=self.registry.clone();let catalog=self.hyprfast.clone().ok_or_else(||anyhow!("HyprFast catalog unavailable"))?;let interrupt=self.interrupt.clone();let working_dir=self.working_dir.clone();let max_subtasks=self.config.planner.max_subtasks.min(DEFAULT_MAX_SUBTASKS.max(self.config.planner.max_subtasks));let planner_model=self.config.models.planner.clone();
  tokio::spawn(async move{let run=async{let user=TurnMessage::User(prompt.clone());let _=tx.send(AgentEvent::History{message:user});let _=tx.send(AgentEvent::Status{message:"Decomposing task…".into()});let decomposition=decompose(&provider,&planner_model,&prompt,&catalog,&route,&interrupt).await?;if decomposition.subtasks.is_empty()||decomposition.subtasks.len()>max_subtasks{return Err(anyhow!("planner returned an invalid subtask count"))}
    let mut completed:HashMap<String,Value>=HashMap::new();let mut notes=Vec::new();
    for(index,subtask)in decomposition.subtasks.iter().enumerate(){if interrupt.is_set(){return Err(lucy_core::LucyError::Cancelled.into())}let _=tx.send(AgentEvent::Status{message:format!("Subtask {}/{}: {}",index+1,decomposition.subtasks.len(),subtask.goal)});let subroute=catalog.route_domain(&subtask.category,&subtask.goal);let allowed:HashSet<String>=subroute.candidates.iter().cloned().collect();if allowed.is_empty(){return Err(anyhow!("no HyprFast tools matched subtask: {}",subtask.goal))}let schemas=registry.definitions_for_names(&allowed);let context=build_context(&catalog,&subroute,&completed,&subtask.depends_on);let command=plan_command(&provider,&planner_model,&prompt,subtask,&schemas,&context,&interrupt).await?;if !allowed.contains(&command.tool){return Err(anyhow!("planner selected tool outside routed capability set: {}",command.tool))}if !command.arguments.is_object(){return Err(anyhow!("planner arguments must be a JSON object"))}
      let call_id=format!("plan-{}",index+1);let _=tx.send(AgentEvent::ToolStarted{id:call_id.clone(),name:command.tool.clone()});let ctx=ToolContext{session_id:SessionId::default(),tool_call_id:call_id.clone(),working_dir:Some(working_dir.clone()),execution_mode:ExecutionMode::Agent,events:tx.clone(),interrupt:interrupt.clone()};let result=registry.execute(&command.tool,command.arguments.clone(),ctx).await;let output=match result{Ok(v)=>v,Err(e)=>{let _=tx.send(AgentEvent::ToolFinished{id:call_id.clone(),output:serde_json::json!({"error":e.to_string()})});return Err(anyhow!("subtask failed: {}: {}",subtask.goal,e))}};let _=tx.send(AgentEvent::ToolFinished{id:call_id.clone(),output:output.clone()});completed.insert(subtask.id.clone(),truncate_json(output));if let Some(v)=command.verify{notes.push(format!("{} — {}",subtask.goal,v));}else{notes.push(subtask.goal.clone())};let tool_msg=TurnMessage::Tool(ToolResult{call_id,name:command.tool,output:completed.get(&subtask.id).cloned().unwrap_or(Value::Null),is_error:false});let _=tx.send(AgentEvent::History{message:tool_msg});
    }
    let text=format!("Completed {} computer-operation step(s).",notes.len());let assistant=TurnMessage::Assistant(AssistantTurn{text:Some(text.clone()),tool_calls:Vec::new()});let _=tx.send(AgentEvent::History{message:assistant});let _=tx.send(AgentEvent::TextDelta{text});Ok::<(),anyhow::Error>(())};if let Err(e)=run.await{let _=tx.send(AgentEvent::Error{message:e.to_string()});}let _=tx.send(AgentEvent::Done);});Ok(rx)
 }
 pub async fn history(&self)->Vec<TurnMessage>{self.session.lock().await.history.clone()}
}
async fn decompose(provider:&OpenAIProvider,model:&str,prompt:&str,catalog:&HyprFastCatalog,route:&Option<lucy_hyprfast::Route>,interrupt:&InterruptSignal)->Result<Decomposition>{let route_context=route.as_ref().map(|r|catalog.context_for(r)).unwrap_or_default();let system=r#"You are Lucy's task decomposer. Break a computer-operation request into ordered, concrete, executable subtasks. Every subtask must be small enough that ONE HyprFast MCP tool call can accomplish it. Include prerequisite observation steps when current UI state matters. Never assume the requested app/window/tab is active. Use the exact HyprFast capability category needed for each step. Return ONLY JSON: {\"subtasks\":[{\"id\":\"1\",\"goal\":\"precise action or observation\",\"category\":\"browser|desktop|vision|excalidraw|clipboard|tasks|stagehand|hints\",\"depends_on\":[]}]}. Do not invent tool names or arguments yet."#;let user=format!("Original task:\n{}\n\nInitial routed HyprFast context:\n{}",prompt,route_context);Ok(serde_json::from_value(provider.complete_json(model,system,&user,interrupt.clone()).await?)?) }
async fn plan_command(provider:&OpenAIProvider,model:&str,prompt:&str,subtask:&SubTask,schemas:&[Value],context:&str,interrupt:&InterruptSignal)->Result<PlannedCommand>{let system=r#"You are Lucy's HyprFast command planner. You receive ONE subtask, a tiny set of allowed HyprFast tools, their exact JSON schemas, and current execution context. Select exactly ONE allowed tool and produce exact arguments that conform to its schema. Do not invent fields, tool names, tabs, windows, coordinates, IDs, or other state. If the subtask needs state that is not present, choose an observation tool from the allowed set instead. Return ONLY JSON: {\"tool\":\"exact allowed tool name\",\"arguments\":{},\"verify\":\"optional short verification\"}. The tool name and argument object must be directly executable by Lucy."#;let tools=serde_json::to_string(schemas)?;let user=format!("Original task:\n{}\n\nSubtask category: {}\nSubtask: {}\nDependencies: {:?}\n\nCurrent HyprFast context:\n{}\n\nAllowed tools and schemas:\n{}",prompt,subtask.category,subtask.goal,subtask.depends_on,context,tools);Ok(serde_json::from_value(provider.complete_json(model,system,&user,interrupt.clone()).await?)?) }
fn build_context(catalog:&HyprFastCatalog,route:&lucy_hyprfast::Route,completed:&HashMap<String,Value>,depends_on:&[String])->String{let mut out=catalog.context_for(route);for id in depends_on{if let Some(v)=completed.get(id){out.push_str(&format!("\nDependency {} result: {}",id,v));}}if out.len()>MAX_CONTEXT_CHARS{out.truncate(MAX_CONTEXT_CHARS);out.push_str("\n[context truncated]");}out}
fn truncate_json(v:Value)->Value{let s=v.to_string();if s.len()<=MAX_CONTEXT_CHARS{return v}let end=s.char_indices().nth(MAX_CONTEXT_CHARS).map(|(i,_)|i).unwrap_or(s.len());serde_json::json!({"truncated":true,"preview":&s[..end]})}
fn looks_like_computer_task(text:&str)->bool{let t=text.to_ascii_lowercase();["browser","brave","chrome","firefox","tab","window","workspace","desktop","screen","screenshot","click","type","keyboard","mouse","excalidraw","draw","diagram","clipboard","copy","paste","open app","launch app"].iter().any(|x|t.contains(x))}
fn now()->u64{std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()}
