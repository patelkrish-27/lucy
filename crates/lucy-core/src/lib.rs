mod session;
pub use session::{SessionData, SessionMeta, SessionStore, now_secs};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};
use thiserror::Error;
use tokio::sync::{Notify, mpsc};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionMode {
    Agent,
    Direct,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterruptSource {
    User,
    System,
    BackgroundTask,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterruptMessage {
    pub content: String,
    pub urgent: bool,
    pub source: InterruptSource,
}
pub type InterruptQueue = Arc<std::sync::Mutex<Vec<InterruptMessage>>>;
#[derive(Clone)]
pub struct InterruptSignal {
    flag: Arc<AtomicBool>,
    epoch: Arc<AtomicU64>,
    notify: Arc<Notify>,
}
impl std::fmt::Debug for InterruptSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InterruptSignal")
            .field("flag", &self.flag.load(Ordering::SeqCst))
            .field("epoch", &self.epoch.load(Ordering::SeqCst))
            .finish()
    }
}
impl InterruptSignal {
    pub fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
            epoch: Arc::new(AtomicU64::new(0)),
            notify: Arc::new(Notify::new()),
        }
    }
    pub fn fire(&self) {
        self.epoch.fetch_add(1, Ordering::SeqCst);
        self.flag.store(true, Ordering::SeqCst);
        self.notify.notify_waiters()
    }
    pub fn is_set(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
    pub fn reset(&self) {
        self.flag.store(false, Ordering::SeqCst)
    }
    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::SeqCst)
    }
    pub fn reset_if_epoch(&self, epoch: u64) -> bool {
        if self.epoch() != epoch {
            return false;
        }
        self.flag.store(false, Ordering::SeqCst);
        if self.epoch() != epoch {
            self.flag.store(true, Ordering::SeqCst);
            self.notify.notify_waiters();
            return false;
        }
        true
    }
    pub async fn notified(&self) {
        let notified = self.notify.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        if self.is_set() {
            return;
        }
        notified.await
    }
}
impl Default for InterruptSignal {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionId(pub Uuid);
impl Default for SessionId {
    fn default() -> Self {
        Self(Uuid::new_v4())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: Value,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolResult {
    pub call_id: String,
    pub name: String,
    pub output: Value,
    pub is_error: bool,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssistantTurn {
    pub text: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}
impl Serialize for AssistantTurn {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(Serialize)]
        struct Full<'a> {
            #[serde(default, skip_serializing_if = "Option::is_none")]
            text: &'a Option<String>,
            #[serde(default, skip_serializing_if = "<[_]>::is_empty")]
            tool_calls: &'a [ToolCall],
        }
        Full {
            text: &self.text,
            tool_calls: &self.tool_calls,
        }
        .serialize(s)
    }
}
impl<'de> Deserialize<'de> for AssistantTurn {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Compat {
            LegacyText(String),
            Structured {
                #[serde(default)]
                text: Option<String>,
                #[serde(default)]
                tool_calls: Vec<ToolCall>,
            },
        }
        match Compat::deserialize(d)? {
            Compat::LegacyText(text) => Ok(Self {
                text: Some(text),
                tool_calls: Vec::new(),
            }),
            Compat::Structured { text, tool_calls } => Ok(Self { text, tool_calls }),
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TurnMessage {
    User(String),
    Assistant(AssistantTurn),
    Tool(ToolResult),
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
}
impl TokenUsage {
    pub fn add(&mut self, o: &Self) {
        self.prompt_tokens += o.prompt_tokens;
        self.completion_tokens += o.completion_tokens;
        self.total_tokens += o.total_tokens;
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTurn {
    pub text: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default)]
    pub stop: bool,
    #[serde(default)]
    pub usage: Option<TokenUsage>,
}
#[derive(Debug, Clone)]
pub struct ModelRequest {
    pub session_id: SessionId,
    pub prompt: String,
    pub history: Vec<TurnMessage>,
    pub tools: Vec<Value>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentEvent {
    TextDelta {
        text: String,
    },
    Thinking {
        text: String,
    },
    ToolStarted {
        id: String,
        name: String,
        input: Value,
    },
    ToolFinished {
        id: String,
        name: String,
        output: Value,
        is_error: bool,
    },
    ApprovalRequest {
        id: String,
        name: String,
        input: Value,
    },
    History {
        message: TurnMessage,
    },
    Status {
        message: String,
    },
    Progress {
        message: String,
    },
    Error {
        message: String,
    },
    Done,
}
pub fn truncate_tool_output(val: &Value, max_lines: usize) -> Value {
    match val {
        Value::String(s) => Value::String(truncate_str(s, max_lines)),
        Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (k, v) in map {
                if let Value::String(s) = v {
                    out.insert(k.clone(), Value::String(truncate_str(s, max_lines)));
                } else {
                    out.insert(k.clone(), v.clone());
                }
            }
            Value::Object(out)
        }
        other => other.clone(),
    }
}
pub fn truncate_str(s: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    if lines.len() <= max_lines || max_lines < 10 {
        return s.to_string();
    }
    let head_count = (max_lines * 60) / 100;
    let tail_count = max_lines.saturating_sub(head_count);
    let dropped = lines.len().saturating_sub(head_count + tail_count);
    let mut out = Vec::with_capacity(max_lines + 2);
    out.extend(lines[..head_count].iter().cloned());
    let note = format!("... [truncated {dropped} lines] ...");
    out.push(&note);
    out.extend(lines[lines.len() - tail_count..].iter().cloned());
    out.join("\n")
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApprovalDecision {
    AllowOnce,
    AllowAlways,
    Deny,
}
#[derive(Clone)]
pub struct ApprovalGate {
    pub events: mpsc::UnboundedSender<AgentEvent>,
    pub pending:
        Arc<std::sync::Mutex<HashMap<String, tokio::sync::oneshot::Sender<ApprovalDecision>>>>,
    pub mode: Arc<std::sync::RwLock<String>>,
    pub always_allow: Arc<std::sync::Mutex<HashSet<String>>>,
}
impl std::fmt::Debug for ApprovalGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApprovalGate")
            .field(
                "mode",
                &self.mode.read().map(|g| g.clone()).unwrap_or_default(),
            )
            .finish()
    }
}
impl ApprovalGate {
    pub fn new(events: mpsc::UnboundedSender<AgentEvent>) -> Self {
        Self {
            events,
            pending: Arc::new(std::sync::Mutex::new(HashMap::new())),
            mode: Arc::new(std::sync::RwLock::new("write".to_string())),
            always_allow: Arc::new(std::sync::Mutex::new(HashSet::new())),
        }
    }
    pub fn with_events(&self, events: mpsc::UnboundedSender<AgentEvent>) -> ApprovalGate {
        ApprovalGate {
            events,
            pending: self.pending.clone(),
            mode: self.mode.clone(),
            always_allow: self.always_allow.clone(),
        }
    }
    /// Current approval mode: `never` (auto-approve everything), `write`
    /// (ask for state-changing tools — default), `always` (ask for every tool).
    pub fn approval_mode(&self) -> String {
        self.mode
            .read()
            .map(|g| g.clone())
            .unwrap_or_else(|_| "write".to_string())
    }
    /// Set approval mode. Accepts `never|auto|off` (auto), `write` (default),
    /// `always|paranoid`. `auto/on` map to `never` for discoverability.
    pub fn set_mode(&self, mode: &str) -> String {
        let normalized = match mode.trim().to_ascii_lowercase().as_str() {
            "never" | "auto" | "off" | "on" | "yes" | "y" => "never".to_string(),
            "write" | "default" | "ask" => "write".to_string(),
            "always" | "paranoid" | "strict" => "always".to_string(),
            other => other.to_string(),
        };
        if let Ok(mut m) = self.mode.write() {
            *m = normalized.clone();
        }
        normalized
    }
    /// Convenience: `true` → auto-approve everything (`never`),
    /// `false` → back to default `write` prompting.
    pub fn set_auto(&self, auto: bool) -> String {
        self.set_mode(if auto { "never" } else { "write" })
    }
    pub fn is_auto(&self) -> bool {
        self.approval_mode() == "never"
    }
    /// Forget per-tool `always allow` (the `a` key) grants.
    pub fn clear_allowances(&self) {
        if let Ok(mut g) = self.always_allow.lock() {
            g.clear();
        }
    }
    pub fn needs_approval(&self, tool_name: &str, requires: bool) -> bool {
        let mode = self
            .mode
            .read()
            .map(|g| g.clone())
            .unwrap_or_else(|_| "write".to_string());
        if mode == "never" {
            return false;
        }
        if mode == "always" {
            return true;
        }
        if !requires {
            return false;
        }
        match self.always_allow.lock() {
            Ok(g) => !g.contains(tool_name),
            Err(_) => true,
        }
    }
    pub async fn ask(&self, call_id: &str, name: &str, input: &Value) -> ApprovalDecision {
        let rx = {
            let (tx, rx) = tokio::sync::oneshot::channel::<ApprovalDecision>();
            if self
                .pending
                .lock()
                .map(|mut p| p.insert(call_id.to_string(), tx))
                .is_err()
            {
                return ApprovalDecision::Deny;
            }
            rx
        };
        let event = AgentEvent::ApprovalRequest {
            id: call_id.to_string(),
            name: name.to_string(),
            input: input.clone(),
        };
        let _ = self.events.send(event);
        match tokio::time::timeout(std::time::Duration::from_secs(300), rx).await {
            Ok(Ok(d)) => {
                if d == ApprovalDecision::AllowAlways {
                    if let Ok(mut g) = self.always_allow.lock() {
                        g.insert(name.to_string());
                    }
                }
                d
            }
            _ => {
                if let Ok(mut p) = self.pending.lock() {
                    p.remove(call_id);
                }
                ApprovalDecision::Deny
            }
        }
    }
    pub fn resolve(&self, call_id: &str, d: ApprovalDecision) -> bool {
        let sender = if let Ok(mut p) = self.pending.lock() {
            p.remove(call_id)
        } else {
            return false;
        };
        match sender {
            Some(s) => {
                let _ = s.send(d);
                true
            }
            None => false,
        }
    }
}
#[derive(Debug, Clone)]
pub struct ToolContext {
    pub session_id: SessionId,
    pub tool_call_id: String,
    pub working_dir: Option<PathBuf>,
    pub execution_mode: ExecutionMode,
    pub events: mpsc::UnboundedSender<AgentEvent>,
    pub interrupt: InterruptSignal,
}
impl ToolContext {
    pub fn resolve_path(&self, path: &Path) -> PathBuf {
        if path.is_absolute() {
            path.to_path_buf()
        } else if let Some(base) = &self.working_dir {
            base.join(path)
        } else {
            path.to_path_buf()
        }
    }
}
#[derive(Debug, Error)]
pub enum LucyError {
    #[error("cancelled")]
    Cancelled,
    #[error("tool not found: {0}")]
    ToolNotFound(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("provider error: {0}")]
    Provider(String),
}
#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> Value;
    fn requires_approval(&self) -> bool {
        true
    }
    async fn execute(&self, input: Value, ctx: ToolContext) -> anyhow::Result<Value>;
}
#[async_trait::async_trait]
pub trait ModelProvider: Send + Sync {
    async fn run_turn(
        &self,
        request: ModelRequest,
        events: mpsc::UnboundedSender<AgentEvent>,
        interrupt: InterruptSignal,
    ) -> anyhow::Result<ModelTurn>;
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn assistant_turn_deserializes_legacy_string() {
        let message: TurnMessage = serde_json::from_str(r#"{"Assistant":"hello"}"#)
            .expect("legacy message should deserialize");
        match message {
            TurnMessage::Assistant(turn) => {
                assert_eq!(turn.text.as_deref(), Some("hello"));
                assert!(turn.tool_calls.is_empty())
            }
            _ => panic!("expected assistant message"),
        }
    }
    #[test]
    fn test_truncate_str_keeps_head_and_tail() {
        let lines: Vec<String> = (0..100).map(|i| format!("line {i}")).collect();
        let text = lines.join("\n");
        let truncated = truncate_str(&text, 20);
        assert!(truncated.contains("line 0"));
        assert!(truncated.contains("line 99"));
        assert!(truncated.contains("... [truncated 80 lines] ..."));
    }
}
