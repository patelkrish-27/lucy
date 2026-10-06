pub mod error_text;
mod model_log;
mod session;
pub use error_text::{Explained, explain, friendly, payload_message};
pub use model_log::{
    ERROR_CHARS, EXCERPT_CHARS, ModelCallKind, ModelCallRecord, log as log_model_call,
    model_log_path, prompts_enabled, truncate_chars,
};
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
    /// Where approval prompts are published. Swappable so a long-lived gate can
    /// be re-pointed at whichever event channel the current run owns (`None`
    /// swallows prompts, which is what an offline caller wants).
    pub events: Arc<std::sync::Mutex<Option<mpsc::UnboundedSender<AgentEvent>>>>,
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
/// How long an unanswered prompt blocks a run before it is refused.
///
/// A prompt nobody answered is not an approval, so the timeout denies rather
/// than allows — a UI that died mid-run must not wave the next destructive step
/// through.
pub const APPROVAL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

impl ApprovalGate {
    pub fn new(events: mpsc::UnboundedSender<AgentEvent>) -> Self {
        Self {
            events: Arc::new(std::sync::Mutex::new(Some(events))),
            pending: Arc::new(std::sync::Mutex::new(HashMap::new())),
            mode: Arc::new(std::sync::RwLock::new("write".to_string())),
            always_allow: Arc::new(std::sync::Mutex::new(HashSet::new())),
        }
    }
    /// Re-point approval prompts at a different event channel.
    pub fn set_events(&self, events: mpsc::UnboundedSender<AgentEvent>) {
        if let Ok(mut slot) = self.events.lock() {
            *slot = Some(events);
        }
    }
    /// Stop publishing approval prompts (e.g. the run that owned the channel
    /// has finished).
    pub fn clear_events(&self) {
        if let Ok(mut slot) = self.events.lock() {
            *slot = None;
        }
    }
    fn publish(&self, event: AgentEvent) {
        if let Ok(slot) = self.events.lock() {
            if let Some(tx) = slot.as_ref() {
                let _ = tx.send(event);
            }
        }
    }
    pub fn with_events(&self, events: mpsc::UnboundedSender<AgentEvent>) -> ApprovalGate {
        ApprovalGate {
            events: Arc::new(std::sync::Mutex::new(Some(events))),
            pending: self.pending.clone(),
            mode: self.mode.clone(),
            always_allow: self.always_allow.clone(),
        }
    }
    /// A gate that is not attached to a long-lived agent loop: same shared
    /// state, but seeded with an explicit `approvals.mode`. Used by the
    /// runtime's sequential command executor.
    pub fn with_mode(events: mpsc::UnboundedSender<AgentEvent>, mode: impl Into<String>) -> Self {
        Self {
            events: Arc::new(std::sync::Mutex::new(Some(events))),
            pending: Arc::new(std::sync::Mutex::new(HashMap::new())),
            mode: Arc::new(std::sync::RwLock::new(mode.into())),
            always_allow: Arc::new(std::sync::Mutex::new(HashSet::new())),
        }
    }

    /// As [`Self::with_mode`], but seeded with the tools the user has already
    /// allowed permanently, so an "always allow" from a previous session is
    /// honoured instead of re-prompting on the first call after startup.
    pub fn with_mode_and_allow(
        events: mpsc::UnboundedSender<AgentEvent>,
        mode: impl Into<String>,
        always_allow: impl IntoIterator<Item = String>,
    ) -> Self {
        let gate = Self::with_mode(events, mode);
        gate.seed_always_allowed(always_allow);
        gate
    }
    pub fn needs_approval(&self, tool_name: &str, requires: bool) -> bool {
        let mode = self.current_mode();
        // Automode: never stop. The kill switch is the safety net, not a prompt.
        if mode == "never" {
            return false;
        }
        // A tool the user explicitly allowed never prompts again, in ANY mode.
        // Previously `always` short-circuited above this check, so hitting
        // "always allow" had no lasting effect in that mode.
        if self.is_always_allowed(tool_name) {
            return false;
        }
        if mode == "always" {
            return true;
        }
        requires
    }

    /// The active mode, falling back to `write` for a poisoned lock or an
    /// unparseable value so a hand-edited typo cannot silently disable prompts.
    pub fn current_mode(&self) -> String {
        self.mode
            .read()
            .map(|g| g.trim().to_owned())
            .ok()
            .filter(|m| matches!(m.as_str(), "never" | "write" | "always"))
            .unwrap_or_else(|| "write".to_string())
    }

    /// Switch mode at runtime (the `/auto` command) and notify waiters.
    pub fn set_mode(&self, mode: impl Into<String>) {
        if let Ok(mut m) = self.mode.write() {
            *m = mode.into();
        }
    }

    pub fn is_always_allowed(&self, tool_name: &str) -> bool {
        self.always_allow
            .lock()
            .map(|g| g.contains(tool_name))
            .unwrap_or(false)
    }

    /// Snapshot the always-allow set for persisting to config.
    pub fn always_allowed_tools(&self) -> Vec<String> {
        match self.always_allow.lock() {
            Ok(g) => g.iter().cloned().collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Whether a tool the user answered with "always" is remembered *only* in
    /// this process — the case where the run that recorded it did not persist.
    /// Cheap, so it can gate a write on every prompt resolution.
    pub fn has_unsaved_always_allow(&self, configured: &[String]) -> bool {
        !self.newly_allowed(configured).is_empty()
    }

    /// Seed the always-allow set from persisted config at startup.
    pub fn seed_always_allowed(&self, tools: impl IntoIterator<Item = String>) {
        if let Ok(mut g) = self.always_allow.lock() {
            g.extend(tools);
        }
    }
    pub async fn ask(&self, call_id: &str, name: &str, input: &Value) -> ApprovalDecision {
        let Some(rx) = self.open_prompt(call_id, name, input) else {
            return ApprovalDecision::Deny;
        };
        match tokio::time::timeout(APPROVAL_TIMEOUT, rx).await {
            Ok(Ok(d)) => {
                self.record_always_allow(name, d);
                d
            }
            _ => {
                self.discard_prompt(call_id);
                ApprovalDecision::Deny
            }
        }
    }

    /// [`Self::ask`], but a kill switch that fires *while the prompt is open*
    /// abandons the wait instead of sitting out the whole timeout.
    ///
    /// [`Self::wait_cancelled`] cannot do this on its own: it returns as soon as
    /// `call_id` is absent from `pending`, and a prompt is only pending from
    /// inside `ask`. Calling it before `ask` therefore observes an empty map and
    /// returns immediately — the interrupt window it was added for is exactly the
    /// one it never covered. So the two are raced here, with the pending entry
    /// installed first.
    ///
    /// `None` means stopped: the caller must abandon the call. The entry is
    /// removed on both paths, so a prompt answered after the fact resolves
    /// nothing instead of leaking a sender.
    pub async fn ask_cancellable(
        &self,
        interrupt: &InterruptSignal,
        call_id: &str,
        name: &str,
        input: &Value,
    ) -> Option<ApprovalDecision> {
        let rx = self.open_prompt(call_id, name, input)?;
        let decision = tokio::select! {
            answered = tokio::time::timeout(APPROVAL_TIMEOUT, rx) => match answered {
                Ok(Ok(d)) => d,
                // Timeout, or the answering task is gone. Both are a refusal:
                // a prompt nobody answered must not become a silent allow.
                _ => ApprovalDecision::Deny,
            },
            // `notified` returns at once when the flag is already set, so a
            // kill switch that landed before the prompt opened is caught too.
            _ = interrupt.notified() => {
                self.discard_prompt(call_id);
                return None;
            }
        };
        self.record_always_allow(name, decision);
        Some(decision)
    }

    /// Register a prompt and publish it, returning the channel that resolves it.
    /// `None` when the pending map is poisoned, which is a refusal to prompt.
    fn open_prompt(
        &self,
        call_id: &str,
        name: &str,
        input: &Value,
    ) -> Option<tokio::sync::oneshot::Receiver<ApprovalDecision>> {
        let (tx, rx) = tokio::sync::oneshot::channel::<ApprovalDecision>();
        if self
            .pending
            .lock()
            .map(|mut p| p.insert(call_id.to_string(), tx))
            .is_err()
        {
            return None;
        }
        self.publish(AgentEvent::ApprovalRequest {
            id: call_id.to_string(),
            name: name.to_string(),
            input: input.clone(),
        });
        Some(rx)
    }

    fn discard_prompt(&self, call_id: &str) {
        if let Ok(mut p) = self.pending.lock() {
            p.remove(call_id);
        }
    }

    /// An "always allow" is remembered for the rest of the process, in every
    /// mode — persisting it to config is the caller's job.
    fn record_always_allow(&self, name: &str, decision: ApprovalDecision) {
        if decision == ApprovalDecision::AllowAlways
            && let Ok(mut g) = self.always_allow.lock()
        {
            g.insert(name.to_owned());
        }
    }

    /// A prompt the user answered with "always" that config did not already
    /// know about. Sorted, so a caller can persist it deterministically.
    pub fn newly_allowed(&self, configured: &[String]) -> Vec<String> {
        let Ok(g) = self.always_allow.lock() else {
            return Vec::new();
        };
        let mut out: Vec<String> = g
            .iter()
            .filter(|tool| !configured.iter().any(|c| c == *tool))
            .cloned()
            .collect();
        out.sort();
        out
    }
    /// Wait until this prompt is answered or the run is stopped. `true` means
    /// stopped — the caller must abandon the call.
    ///
    /// Only usable for a prompt that is *already* open; it returns immediately
    /// when `call_id` is not in [`Self::pending`]. Calling it before the prompt
    /// is published therefore observes an empty map and returns at once, which
    /// means it cannot cover the window it looks like it covers. Callers that
    /// want a stop to interrupt an ask want [`Self::ask_cancellable`].
    pub async fn wait_cancelled(&self, interrupt: &InterruptSignal, call_id: &str) -> bool {
        loop {
            if interrupt.is_set() {
                return true;
            }
            if !self
                .pending
                .lock()
                .map(|p| p.contains_key(call_id))
                .unwrap_or(false)
            {
                return false;
            }
            interrupt.notified().await;
        }
    }

    /// Ids of every prompt still awaiting an answer. The kill switch drains
    /// these so a cancelled run cannot hang on a 300s `ask()` timeout.
    pub fn pending_ids(&self) -> Vec<String> {
        match self.pending.lock() {
            Ok(p) => p.keys().cloned().collect(),
            Err(_) => Vec::new(),
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

    /// Resolve a filesystem path inside the current workspace boundary.
    /// File tools use this stricter variant so an agent cannot escape via
    /// absolute paths, `..`, or a symlink that points outside the workspace.
    /// For writes, the nearest existing parent is canonicalized first.
    pub fn resolve_path_checked(&self, path: &Path) -> anyhow::Result<PathBuf> {
        let root = self.working_dir.clone().unwrap_or(std::env::current_dir()?);
        let root = std::fs::canonicalize(&root).map_err(|e| {
            anyhow::anyhow!("workspace root is unavailable: {}: {e}", root.display())
        })?;
        let candidate = if path.is_absolute() {
            path.to_path_buf()
        } else {
            root.join(path)
        };
        let canonical = match std::fs::canonicalize(&candidate) {
            Ok(existing) => existing,
            Err(_) => {
                let mut ancestor = candidate.clone();
                let mut missing = Vec::new();
                while !ancestor.exists() {
                    if let Some(name) = ancestor.file_name() {
                        missing.push(name.to_os_string());
                    }
                    if !ancestor.pop() {
                        return Err(anyhow::anyhow!(
                            "path cannot be resolved safely: {}", candidate.display()
                        ));
                    }
                }
                let mut resolved = std::fs::canonicalize(&ancestor)?;
                for name in missing.iter().rev() {
                    resolved.push(name);
                }
                resolved
            }
        };
        if !canonical.starts_with(&root) {
            return Err(anyhow::anyhow!(
                "path escapes Lucy's workspace: {}", path.display()
            ));
        }
        Ok(canonical)
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
    fn gate(mode: &str) -> ApprovalGate {
        ApprovalGate::with_mode(mpsc::unbounded_channel().0, mode)
    }

    #[test]
    fn automode_never_prompts() {
        let g = gate("never");
        assert!(!g.needs_approval("write_file", true));
        assert!(!g.needs_approval("anything", true));
    }

    #[test]
    fn write_mode_prompts_only_for_flagged_tools() {
        let g = gate("write");
        assert!(!g.needs_approval("read_file", false));
        assert!(g.needs_approval("write_file", true));
    }

    #[test]
    fn always_allow_wins_in_every_mode() {
        // Regression: `always` used to short-circuit before the always_allow
        // check, so "always allow" had no lasting effect in that mode.
        for mode in ["write", "always"] {
            let g = gate(mode);
            assert!(g.needs_approval("write_file", true), "{mode}");
            g.always_allow.lock().unwrap().insert("write_file".into());
            assert!(!g.needs_approval("write_file", true), "{mode}");
        }
    }

    #[test]
    fn always_mode_prompts_for_read_only_tools_too() {
        let g = gate("always");
        assert!(g.needs_approval("read_file", false));
    }

    #[test]
    fn a_bogus_mode_falls_back_to_write_rather_than_silently_never() {
        let g = gate("yolo");
        assert_eq!(g.current_mode(), "write");
        assert!(g.needs_approval("write_file", true));
    }

    #[tokio::test]
    async fn seeded_always_allow_survives_gate_construction() {
        let g = ApprovalGate::with_mode_and_allow(
            mpsc::unbounded_channel().0,
            "write",
            vec!["edit_file".to_string()],
        );
        assert!(!g.needs_approval("edit_file", true));
        assert!(g.needs_approval("write_file", true));
    }

    #[tokio::test]
    async fn wait_cancelled_reports_a_stopped_run() {
        let g = gate("write");
        let interrupt = InterruptSignal::new();
        // No pending prompt: returns immediately, run not stopped.
        assert!(!g.wait_cancelled(&interrupt, "call-1").await);
    }

    /// A stop that lands *while the prompt is open* has to end the wait. This is
    /// the case `wait_cancelled` alone cannot cover: it returns as soon as
    /// `call_id` is not pending, and a prompt is pending only from inside
    /// `ask` — so checking it before asking observes an empty map and returns
    /// immediately, leaving the run to sit out the full timeout.
    #[tokio::test]
    async fn a_stop_while_a_prompt_is_open_abandons_it() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let g = ApprovalGate::with_mode(tx, "write");
        let interrupt = InterruptSignal::new();
        let stopper = interrupt.clone();
        // The gate is cloned into the asker so this test can still inspect
        // `pending` afterwards — the handle is shared, the handle is not moved.
        let asker_gate = g.clone();
        let asker = tokio::spawn(async move {
            asker_gate
                .ask_cancellable(&interrupt, "call-1", "rm_rf", &Value::Null)
                .await
        });
        // Wait for the prompt to actually be published before stopping, so this
        // asserts the race rather than the already-set-flag shortcut.
        let AgentEvent::ApprovalRequest { id, .. } = rx.recv().await.expect("a prompt")
        else {
            panic!("expected an approval request");
        };
        stopper.fire();
        assert_eq!(asker.await.expect("joins"), None, "stopped, not answered");
        assert!(g.pending_ids().is_empty(), "nothing left pending");
        // The prompt was real: it reached a channel with a live receiver.
        assert_eq!(id, "call-1");
    }

    /// A prompt answered before any stop comes back as itself, so racing the
    /// interrupt costs an answered prompt nothing.
    #[tokio::test]
    async fn an_answered_prompt_is_returned_despite_the_race() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let g = ApprovalGate::with_mode(tx, "write");
        let interrupt = InterruptSignal::new();
        let asker = {
            let g = g.clone();
            tokio::spawn(async move {
                g.ask_cancellable(&interrupt, "call-1", "write_file", &Value::Null)
                    .await
            })
        };
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert!(g.resolve("call-1", ApprovalDecision::AllowOnce));
        assert_eq!(
            asker.await.expect("joins"),
            Some(ApprovalDecision::AllowOnce)
        );
    }

    /// A stop that already fired is caught even though the prompt opened after
    /// it, because `notified` returns at once when the flag is set.
    #[tokio::test]
    async fn a_stop_that_already_fired_is_caught_before_the_prompt() {
        let g = gate("write");
        let interrupt = InterruptSignal::new();
        interrupt.fire();
        assert_eq!(
            g.ask_cancellable(&interrupt, "call-1", "rm_rf", &Value::Null)
                .await,
            None
        );
        assert!(g.pending_ids().is_empty());
    }

    /// An unanswered prompt is a refusal, not a silent allow: the caller gets
    /// `Some(Deny)` rather than a decision it can act on.
    #[tokio::test]
    async fn an_unanswered_prompt_denies_rather_than_allows() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let g = ApprovalGate::with_mode(tx, "always");
        let interrupt = InterruptSignal::new();
        // The prompt is published and then abandoned, so its `oneshot` sender is
        // dropped without an answer: the caller must see a denial, not a
        // decision it may act on.
        let abandoner = g.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            abandoner.discard_prompt("call-1");
        });
        assert_eq!(
            g.ask_cancellable(&interrupt, "call-1", "rm_rf", &Value::Null)
                .await,
            Some(ApprovalDecision::Deny)
        );
    }

    /// What the run has to write down, and nothing more: a tool config already
    /// lists is not unsaved work.
    #[tokio::test]
    async fn only_always_allows_config_does_not_know_are_unsaved() {
        let g = gate("write");
        assert!(!g.has_unsaved_always_allow(&[]));
        g.always_allow.lock().unwrap().insert("edit_file".into());
        assert_eq!(g.newly_allowed(&[]), vec!["edit_file".to_owned()]);
        assert!(g.has_unsaved_always_allow(&[]));
        assert!(!g.has_unsaved_always_allow(&["edit_file".to_owned()]));
        // Sorted, so the persisted list is stable across writes.
        g.always_allow.lock().unwrap().insert("add_file".into());
        assert_eq!(
            g.newly_allowed(&[]),
            vec!["add_file".to_owned(), "edit_file".to_owned()]
        );
    }

    /// "Always allow" is remembered for the process in every prompting mode,
    /// which is what makes the run's persist step worth doing.
    #[tokio::test]
    async fn an_always_answer_is_recorded_by_the_gate() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let g = ApprovalGate::with_mode(tx, "always");
        let asker = {
            let g = g.clone();
            tokio::spawn(async move { g.ask("call-1", "write_file", &Value::Null).await })
        };
        let AgentEvent::ApprovalRequest { id, .. } = rx.recv().await.expect("a prompt")
        else {
            panic!("expected an approval request");
        };
        assert!(g.resolve(&id, ApprovalDecision::AllowAlways));
        assert_eq!(
            asker.await.expect("joins"),
            ApprovalDecision::AllowAlways
        );
        assert!(g.is_always_allowed("write_file"));
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
