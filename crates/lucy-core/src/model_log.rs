//! Unified log of every model call Lucy makes.
//!
//! Three kinds of calls flow through this module:
//! - **voice** — Groq STT transcriptions (`lucy-stt`)
//! - **classification** — `decider-serve` `/predict` forward passes
//!   (`lucy-systemone`)
//! - **llm** — OpenAI-compatible `/chat/completions` calls
//!   (`lucy-agent`)
//!
//! Each call appends one JSON object per line to
//! `~/.local/state/lucy/model-calls.jsonl` (overridable with `LUCY_MODEL_LOG`).
//! Logging is best-effort and never fails the caller: I/O errors are dropped
//! after one `tracing::warn!`. API keys are never recorded — the record carries
//! only the endpoint host path, model name, latencies, token counts, and
//! truncated excerpts.
//!
//! Prompt/response excerpts (first [`EXCERPT_CHARS`] chars) are included by
//! default for debugging; set `LUCY_MODEL_LOG_PROMPTS=0` to omit them.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Max chars kept for prompt/response excerpts and detail strings.
pub const EXCERPT_CHARS: usize = 500;
/// Max chars kept for error messages.
pub const ERROR_CHARS: usize = 300;

/// Which backend served the call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelCallKind {
    /// OpenAI-compatible chat completion (`lucy-agent`).
    Llm,
    /// `decider-serve` classification forward pass (`lucy-systemone`).
    Classification,
    /// Speech-to-text transcription (`lucy-stt`).
    Voice,
    /// A client-side tool execution (fast lane / Stagehand), not an LLM call.
    Tool,
}

impl ModelCallKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Llm => "llm",
            Self::Classification => "classification",
            Self::Voice => "voice",
            Self::Tool => "tool",
        }
    }
}

/// One logged model call. Serialized as a single JSONL line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCallRecord {
    /// RFC 3339 UTC timestamp, e.g. `2026-09-27T12:34:56.789Z`.
    pub ts: String,
    pub kind: ModelCallKind,
    /// Leaf operation: `complete_text`, `complete_json`, `predict`,
    /// `classify_turn`, `transcribe_bytes`, `transcribe_live`, …
    pub operation: String,
    /// Higher-level reason: `answer_turn`, `plan_commands`, `route_verify`,
    /// `decompose_goal`, `transcribe_file`, … `None` when unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub purpose: Option<String>,
    /// Endpoint that served the call (no credentials, ever).
    #[serde(default)]
    pub endpoint: String,
    /// Model name as sent on the wire.
    #[serde(default)]
    pub model: String,
    pub latency_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    #[serde(default)]
    pub success: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Short machine-readable outcome: classifier verdict
    /// (`branch=no level=3 conf=0.91`), transcript length, …
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_excerpt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_excerpt: Option<String>,
}

impl ModelCallRecord {
    pub fn new(
        kind: ModelCallKind,
        operation: impl Into<String>,
        endpoint: impl Into<String>,
        model: impl Into<String>,
        latency_ms: u64,
    ) -> Self {
        Self {
            ts: Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
            kind,
            operation: operation.into(),
            purpose: None,
            endpoint: endpoint.into(),
            model: model.into(),
            latency_ms,
            prompt_tokens: None,
            completion_tokens: None,
            total_tokens: None,
            success: true,
            error: None,
            detail: None,
            prompt_excerpt: None,
            response_excerpt: None,
        }
    }

    pub fn with_purpose(mut self, purpose: impl Into<String>) -> Self {
        let p = purpose.into();
        self.purpose = if p.trim().is_empty() { None } else { Some(p) };
        self
    }

    pub fn with_usage(mut self, prompt: u64, completion: u64, total: u64) -> Self {
        self.prompt_tokens = Some(prompt);
        self.completion_tokens = Some(completion);
        self.total_tokens = Some(total);
        self
    }

    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(truncate_chars(&detail.into(), EXCERPT_CHARS));
        self
    }

    /// Mark the call failed with a truncated error message.
    pub fn failed(mut self, error: impl Into<String>) -> Self {
        self.success = false;
        let e = truncate_chars(&error.into(), ERROR_CHARS);
        self.error = if e.trim().is_empty() {
            Some("unknown error".to_owned())
        } else {
            Some(e)
        };
        self
    }

    /// Attach truncated prompt/response excerpts unless
    /// `LUCY_MODEL_LOG_PROMPTS=0`.
    pub fn with_excerpts(mut self, prompt: &str, response: &str) -> Self {
        if prompts_enabled() {
            if !prompt.trim().is_empty() {
                self.prompt_excerpt = Some(truncate_chars(prompt, EXCERPT_CHARS));
            }
            if !response.trim().is_empty() {
                self.response_excerpt = Some(truncate_chars(response, EXCERPT_CHARS));
            }
        }
        self
    }
}

/// Append one record to the model-call log. Best-effort: never panics, never
/// returns an error — a logging failure must not break a turn.
pub fn log(record: &ModelCallRecord) {
    tracing::info!(
        kind = record.kind.as_str(),
        operation = %record.operation,
        purpose = record.purpose.as_deref().unwrap_or("-"),
        model = %record.model,
        latency_ms = record.latency_ms,
        success = record.success,
        "model call"
    );
    if let Err(e) = append_record(record) {
        tracing::warn!(
            "model-call log unavailable ({}): {e:#}",
            model_log_path().display()
        );
    }
}

fn append_record(record: &ModelCallRecord) -> anyhow::Result<()> {
    let path = model_log_path();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut line = serde_json::to_string(record)?;
    line.push('\n');
    use std::io::Write;
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?
        .write_all(line.as_bytes())?;
    Ok(())
}

/// Where the JSONL log lives. `LUCY_MODEL_LOG` wins; otherwise
/// `~/.local/state/lucy/model-calls.jsonl` (next to the session store).
pub fn model_log_path() -> PathBuf {
    if let Ok(p) = std::env::var("LUCY_MODEL_LOG")
        && !p.trim().is_empty()
    {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".local/state/lucy/model-calls.jsonl")
}

/// False when `LUCY_MODEL_LOG_PROMPTS` is `0`/`false`/`no` (case-insensitive).
pub fn prompts_enabled() -> bool {
    match std::env::var("LUCY_MODEL_LOG_PROMPTS") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        ),
        Err(_) => true,
    }
}

/// Keep the first `max` chars, appending `…` when truncated.
pub fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_owned();
    }
    let mut out: String = s.chars().take(max).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn temp_log(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("lucy-model-log-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn record_serializes_with_kind_and_outcome() {
        let r = ModelCallRecord::new(
            ModelCallKind::Llm,
            "complete_json",
            "https://x.test/v1",
            "flash",
            42,
        )
        .with_purpose("route_verify")
        .with_usage(10, 5, 15)
        .with_detail("branch=no level=3")
        .with_excerpts("hello?", r#"{"needs_actions":true}"#);
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["kind"], "llm");
        assert_eq!(v["operation"], "complete_json");
        assert_eq!(v["purpose"], "route_verify");
        assert_eq!(v["total_tokens"], 15);
        assert_eq!(v["success"], true);
        assert!(v.get("error").is_none());
        let s = serde_json::to_string(&r).unwrap();
        assert!(!s.contains("Bearer"), "no credentials may appear: {s}");
    }

    #[test]
    fn failed_marks_success_false_and_truncates() {
        let r = ModelCallRecord::new(
            ModelCallKind::Voice,
            "transcribe_bytes",
            "groq",
            "whisper",
            7,
        )
        .failed("x".repeat(5000));
        assert!(!r.success);
        let e = r.error.unwrap();
        assert!(e.chars().count() <= ERROR_CHARS + 1);
        assert!(e.ends_with('…'));
    }

    #[test]
    fn excerpts_are_capped_and_opt_out() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::remove_var("LUCY_MODEL_LOG_PROMPTS") };
        let r = ModelCallRecord::new(ModelCallKind::Llm, "complete_text", "e", "m", 1)
            .with_excerpts(&"y".repeat(5000), "ok");
        assert_eq!(r.prompt_excerpt.unwrap().chars().count(), EXCERPT_CHARS + 1);
        unsafe { std::env::set_var("LUCY_MODEL_LOG_PROMPTS", "0") };
        let r2 = ModelCallRecord::new(ModelCallKind::Llm, "complete_text", "e", "m", 1)
            .with_excerpts("hello", "world");
        assert!(r2.prompt_excerpt.is_none());
        assert!(r2.response_excerpt.is_none());
        unsafe { std::env::remove_var("LUCY_MODEL_LOG_PROMPTS") };
    }

    #[test]
    fn appends_one_json_line_per_call() {
        let _guard = ENV_LOCK.lock().unwrap();
        let path = temp_log("jsonl");
        unsafe { std::env::set_var("LUCY_MODEL_LOG", &path) };
        assert_eq!(model_log_path(), path);
        let r = ModelCallRecord::new(
            ModelCallKind::Classification,
            "classify_turn",
            "http://localhost:8001",
            "decider-2b",
            12,
        )
        .with_detail("branch=yes level=1");
        log(&r);
        log(&r);
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.lines().count(), 2);
        for line in text.lines() {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            assert_eq!(v["kind"], "classification");
        }
        let _ = std::fs::remove_file(&path);
        unsafe { std::env::remove_var("LUCY_MODEL_LOG") };
    }

    #[test]
    fn default_path_lives_next_to_the_session_store() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::remove_var("LUCY_MODEL_LOG") };
        let p = model_log_path();
        assert!(p.ends_with("model-calls.jsonl"), "{}", p.display());
        assert!(p.to_string_lossy().contains(".local/state/lucy"));
    }
}
