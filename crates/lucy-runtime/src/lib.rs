use std::{path::PathBuf,sync::Arc};
use anyhow::Result;
use lucy_agent::{Agent,OpenAIProvider};
use lucy_core::{AgentEvent,InterruptSignal,SessionData,TurnMessage};
use lucy_mcp::{load_config,register_server};
use lucy_tools::default_registry;
use tokio::sync::{mpsc,Mutex};

pub struct LucyRuntime{agent:Arc<Agent<OpenAIProvider>>,session:Arc<Mutex<SessionData>>,session_path:PathBuf,interrupt:InterruptSignal,working_dir:PathBuf}
impl LucyRuntime{
 pub async fn new()->Result<Self>{
  let provider=Arc::new(OpenAIProvider::from_env()?);let mut registry=default_registry();
  for server in load_config()?{if let Err(e)=register_server(&mut registry,server.clone()).await{tracing::warn!(server=%server.name,error=%e,"MCP server unavailable");}}
  let path=std::env::var("LUCY_SESSION_FILE").map(PathBuf::from).unwrap_or_else(|_|PathBuf::from(std::env::var("HOME").unwrap_or_else(|_|".".into())).join(".local/state/lucy/session.json"));
  let session=SessionData::load_from_file(&path).await.unwrap_or_else(|_|SessionData::new(Default::default()));let agent=Arc::new(Agent::new(provider,Arc::new(registry)));Ok(Self{agent,session:Arc::new(Mutex::new(session)),session_path:path,interrupt:InterruptSignal::new(),working_dir:std::env::current_dir()?})
 }
 pub fn interrupt(&self){self.interrupt.fire()}
 pub async fn submit(&self,prompt:String)->Result<mpsc::UnboundedReceiver<AgentEvent>>{
  self.interrupt.reset();let history=self.session.lock().await.history.clone();let mut source=self.agent.execute_with_history(prompt,history,Some(self.working_dir.clone()),self.interrupt.clone()).await?;let(tx,rx)=mpsc::unbounded_channel();let session=self.session.clone();let path=self.session_path.clone();tokio::spawn(async move{while let Some(event)=source.recv().await{if let AgentEvent::History{message}= &event{let mut s=session.lock().await;s.history.push(message.clone());s.updated_at=now();let _=s.save_to_file(&path).await;}let _=tx.send(event);}});Ok(rx)
 }
 pub async fn history(&self)->Vec<TurnMessage>{self.session.lock().await.history.clone()}
}
fn now()->u64{std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()}
