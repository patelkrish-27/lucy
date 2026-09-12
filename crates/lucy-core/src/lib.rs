use std::{path::{Path, PathBuf}, sync::{Arc, atomic::{AtomicBool, AtomicU64, Ordering}}};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tokio::sync::{mpsc, Notify};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionMode { Agent, Direct }
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterruptSource { User, System, BackgroundTask }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterruptMessage { pub content: String, pub urgent: bool, pub source: InterruptSource }
pub type InterruptQueue = Arc<std::sync::Mutex<Vec<InterruptMessage>>>;

#[derive(Clone)]
pub struct InterruptSignal { flag: Arc<AtomicBool>, epoch: Arc<AtomicU64>, notify: Arc<Notify> }
impl InterruptSignal {
    pub fn new() -> Self { Self { flag: Arc::new(AtomicBool::new(false)), epoch: Arc::new(AtomicU64::new(0)), notify: Arc::new(Notify::new()) } }
    pub fn fire(&self) { self.epoch.fetch_add(1, Ordering::SeqCst); self.flag.store(true, Ordering::SeqCst); self.notify.notify_waiters(); }
    pub fn is_set(&self) -> bool { self.flag.load(Ordering::SeqCst) }
    pub fn reset(&self) { self.flag.store(false, Ordering::SeqCst); }
    pub fn epoch(&self) -> u64 { self.epoch.load(Ordering::SeqCst) }
    pub fn reset_if_epoch(&self, epoch: u64) -> bool {
        if self.epoch() != epoch { return false; }
        self.flag.store(false, Ordering::SeqCst);
        if self.epoch() != epoch { self.flag.store(true, Ordering::SeqCst); self.notify.notify_waiters(); return false; }
        true
    }
    pub async fn notified(&self) {
        let notified = self.notify.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        if self.is_set() { return; }
        notified.await;
    }
}
impl Default for InterruptSignal { fn default() -> Self { Self::new() } }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionId(pub Uuid);
impl Default for SessionId { fn default() -> Self { Self(Uuid::new_v4()) } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall { pub id: String, pub name: String, pub input: Value }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult { pub call_id: String, pub name: String, pub output: Value, pub is_error: bool }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TurnMessage {
    User(String),
    Assistant(String),
    Tool(ToolResult),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTurn {
    pub text: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default)]
    pub stop: bool,
}

#[derive(Debug, Clone)]
pub struct ModelRequest {
    pub session_id: SessionId,
    pub prompt: String,
    pub history: Vec<TurnMessage>,
    pub tools: Vec<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentEvent { TextDelta { text: String }, Thinking { text: String }, ToolStarted { id: String, name: String }, ToolFinished { id: String, output: Value }, Status { message: String }, Error { message: String }, Done }

#[derive(Debug, Clone)]
pub struct ToolContext { pub session_id: SessionId, pub tool_call_id: String, pub working_dir: Option<PathBuf>, pub execution_mode: ExecutionMode, pub events: mpsc::UnboundedSender<AgentEvent>, pub interrupt: InterruptSignal }
impl ToolContext { pub fn resolve_path(&self, path: &Path) -> PathBuf { if path.is_absolute() { path.to_path_buf() } else if let Some(base) = &self.working_dir { base.join(path) } else { path.to_path_buf() } } }

#[derive(Debug, Error)]
pub enum LucyError { #[error("cancelled")] Cancelled, #[error("tool not found: {0}")] ToolNotFound(String), #[error("invalid input: {0}")] InvalidInput(String), #[error("provider error: {0}")] Provider(String) }

#[async_trait::async_trait]
pub trait Tool: Send + Sync { fn name(&self) -> &str; fn description(&self) -> &str; fn parameters_schema(&self) -> Value; async fn execute(&self, input: Value, ctx: ToolContext) -> anyhow::Result<Value>; }

#[async_trait::async_trait]
pub trait ModelProvider: Send + Sync {
    async fn run_turn(&self, request: ModelRequest, events: mpsc::UnboundedSender<AgentEvent>, interrupt: InterruptSignal) -> anyhow::Result<ModelTurn>;
}
