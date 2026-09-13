use std::{path::PathBuf,sync::Arc};
use anyhow::Result;
use lucy_agent::{Agent,OpenAIProvider};
use lucy_core::{AgentEvent,InterruptSignal,SessionData,SessionId,TurnMessage};
use lucy_hyprfast::{default_config,HyprFastCatalog};
use lucy_mcp::{load_config,register_server};
use lucy_tools::default_registry;
use tokio::sync::{mpsc,Mutex};

pub struct LucyRuntime{agent:Arc<Agent<OpenAIProvider>>,session:Arc<Mutex<SessionData>>,session_path:PathBuf,interrupt:InterruptSignal,working_dir:PathBuf,hyprfast:Option<HyprFastCatalog>}
impl LucyRuntime{
 pub async fn new()->Result<Self>{let provider=Arc::new(OpenAIProvider::from_env()?);let mut registry=default_registry();let hyprfast=match HyprFastCatalog::discover_default().await{Ok(c)=>{tracing::info!(tools=c.len(),"HyprFast MCP connected");let _=c.save().await;Some(c)},Err(e)=>{tracing::warn!(error=%e,"HyprFast MCP unavailable; continuing without desktop capabilities");None}};for server in load_config()?{if server.name.eq_ignore_ascii_case("hyprfast"){continue}if let Err(e)=register_server(&mut registry,server.clone()).await{tracing::warn!(server=%server.name,error=%e,"MCP server unavailable")}}if hyprfast.is_some(){if let Err(e)=register_server(&mut registry,default_config()).await{tracing::warn!(error=%e,"failed to register HyprFast MCP tools")}}let path=std::env::var("LUCY_SESSION_FILE").map(PathBuf::from).unwrap_or_else(|_|PathBuf::from(std::env::var("HOME").unwrap_or_else(|_|".".into())).join(".local/state/lucy/session.json"));let session=match SessionData::load_from_file(&path).await{Ok(s)=>s,Err(e)=>{if path.exists(){return Err(e)}SessionData::new(SessionId::default())}};let agent=Arc::new(Agent::new(provider,Arc::new(registry)));Ok(Self{agent,session:Arc::new(Mutex::new(session)),session_path:path,interrupt:InterruptSignal::new(),working_dir:std::env::current_dir()?,hyprfast})}
 pub fn interrupt(&self){self.interrupt.fire()}
 pub fn hyprfast_catalog(&self)->Option<&HyprFastCatalog>{self.hyprfast.as_ref()}
 pub async fn submit(&self,prompt:String)->Result<mpsc::UnboundedReceiver<AgentEvent>>{self.interrupt.reset();let session=self.session.lock().await.clone();let mut source=self.agent.execute_with_history(prompt,session.history,Some(self.working_dir.clone()),self.interrupt.clone()).await?;let(tx,rx)=mpsc::unbounded_channel();let session_store=self.session.clone();let path=self.session_path.clone();tokio::spawn(async move{while let Some(event)=source.recv().await{if let AgentEvent::History{message}=&event{let mut session=session_store.lock().await;session.history.push(message.clone());session.updated_at=now();let _=session.save_to_file(&path).await;}let _=tx.send(event);}});Ok(rx)}
 pub async fn history(&self)->Vec<TurnMessage>{self.session.lock().await.history.clone()}
}
fn now()->u64{std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()}
