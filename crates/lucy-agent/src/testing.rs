//! Canned model responses for tests, behind the `testing` feature.
//!
//! The two-speed agent loop's central claim is that a healthy task spends
//! 1–2 slow (LLM) calls no matter how many fast calls it makes. Asserting that
//! needs a slow lane that is both free and countable, which is what this is:
//! [`StubProvider`] answers from a queue of prepared values and records every
//! call it served. An exhausted queue is an error, not a repeat, so a test
//! cannot accidentally pass by having the stub keep helping.

use crate::provider::{ModelProvider, ModelTarget};
use anyhow::{Result, anyhow};
use lucy_core::{AssistantTurn, InterruptSignal, TokenUsage};
use serde_json::Value;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// A [`ModelProvider`] that replays prepared answers and counts its calls.
#[derive(Debug, Default)]
pub struct StubProvider {
    json_queue: Mutex<Vec<Value>>,
    text_queue: Mutex<Vec<String>>,
    /// Prepared [`AssistantTurn`]s, i.e. what a tool-calling model returned: a
    /// reply, a set of calls, or both.
    tools_queue: Mutex<Vec<AssistantTurn>>,
    model: Mutex<String>,
    usage: Mutex<TokenUsage>,
    json_calls: AtomicUsize,
    text_calls: AtomicUsize,
    tool_calls_served: AtomicUsize,
    /// `(purpose, user_prompt)` per call, in order — lets a test assert which
    /// pipeline stage spent the call, not just how many did.
    seen: Mutex<Vec<(String, String)>>,
    /// Tool names offered per tool-calling call, in call order — lets a test
    /// assert *which* catalog the loop put in front of the model.
    seen_tools: Mutex<Vec<Vec<String>>>,
}

impl StubProvider {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            model: Mutex::new("stub-model".to_owned()),
            ..Self::default()
        })
    }

    /// Queue one JSON answer. Answers are served in the order queued.
    pub fn push_json(self: &Arc<Self>, value: Value) -> Arc<Self> {
        self.json_queue
            .lock()
            .expect("stub json queue poisoned")
            .push(value);
        self.clone()
    }

    /// Queue one text answer.
    pub fn push_text(self: &Arc<Self>, text: impl Into<String>) -> Arc<Self> {
        self.text_queue
            .lock()
            .expect("stub text queue poisoned")
            .push(text.into());
        self.clone()
    }

    /// Queue one tool-calling turn: what the model replied with, tool calls
    /// included.
    pub fn push_turn(self: &Arc<Self>, turn: AssistantTurn) -> Arc<Self> {
        self.tools_queue
            .lock()
            .expect("stub tools queue poisoned")
            .push(turn);
        self.clone()
    }

    /// Queue one tool-calling turn that is only calls.
    pub fn push_calls(self: &Arc<Self>, calls: Vec<lucy_core::ToolCall>) -> Arc<Self> {
        self.push_turn(AssistantTurn {
            text: None,
            tool_calls: calls,
        })
    }

    /// Total slow-lane calls served (JSON + text + tool-calling).
    pub fn calls(&self) -> usize {
        self.json_calls.load(Ordering::SeqCst)
            + self.text_calls.load(Ordering::SeqCst)
            + self.tool_calls_served.load(Ordering::SeqCst)
    }

    pub fn json_calls(&self) -> usize {
        self.json_calls.load(Ordering::SeqCst)
    }

    pub fn text_calls(&self) -> usize {
        self.text_calls.load(Ordering::SeqCst)
    }

    /// Tool-calling turns served.
    pub fn tools_calls(&self) -> usize {
        self.tool_calls_served.load(Ordering::SeqCst)
    }

    /// Pipeline stages served so far, in call order.
    pub fn purposes(&self) -> Vec<String> {
        self.seen
            .lock()
            .expect("stub call log poisoned")
            .iter()
            .map(|(purpose, _)| purpose.clone())
            .collect()
    }

    /// `(purpose, user_prompt)` per call, in call order.
    pub fn seen(&self) -> Vec<(String, String)> {
        self.seen.lock().expect("stub call log poisoned").clone()
    }

    /// Tool names offered per tool-calling call, in call order.
    pub fn tools_seen(&self) -> Vec<Vec<String>> {
        self.seen_tools.lock().expect("stub tools log poisoned").clone()
    }

    fn record(&self, purpose: &str, user: &str) {
        self.seen
            .lock()
            .expect("stub call log poisoned")
            .push((purpose.to_owned(), user.to_owned()));
    }

    fn record_tools(&self, tools: &[Value]) {
        self.seen_tools
            .lock()
            .expect("stub tools log poisoned")
            .push(
                tools
                    .iter()
                    .filter_map(|t| t.get("name").and_then(Value::as_str).map(str::to_owned))
                    .collect(),
            );
    }
}

#[async_trait::async_trait]
impl ModelProvider for StubProvider {
    fn model(&self) -> String {
        self.model
            .lock()
            .map(|g| g.clone())
            .unwrap_or_else(|_| "stub-model".to_owned())
    }
    fn set_model(&self, model: String) {
        if let Ok(mut g) = self.model.lock() {
            *g = model;
        }
    }
    fn usage(&self) -> TokenUsage {
        self.usage.lock().map(|g| g.clone()).unwrap_or_default()
    }
    async fn complete_json(
        &self,
        _model: &str,
        purpose: &str,
        _system: &str,
        user: &str,
        interrupt: InterruptSignal,
    ) -> Result<Value> {
        if interrupt.is_set() {
            return Err(lucy_core::LucyError::Cancelled.into());
        }
        self.json_calls.fetch_add(1, Ordering::SeqCst);
        self.record(purpose, user);
        let mut q = self
            .json_queue
            .lock()
            .map_err(|_| anyhow!("stub json queue poisoned"))?;
        if q.is_empty() {
            return Err(anyhow!(
                "StubProvider: no JSON answer queued for '{purpose}'"
            ));
        }
        Ok(q.remove(0))
    }
    async fn complete_text(
        &self,
        _model: &str,
        purpose: &str,
        _system: &str,
        user: &str,
        interrupt: InterruptSignal,
    ) -> Result<String> {
        if interrupt.is_set() {
            return Err(lucy_core::LucyError::Cancelled.into());
        }
        self.text_calls.fetch_add(1, Ordering::SeqCst);
        self.record(purpose, user);
        let mut q = self
            .text_queue
            .lock()
            .map_err(|_| anyhow!("stub text queue poisoned"))?;
        if q.is_empty() {
            return Err(anyhow!(
                "StubProvider: no text answer queued for '{purpose}'"
            ));
        }
        Ok(q.remove(0))
    }
    async fn complete_json_on(
        &self,
        _target: &ModelTarget,
        purpose: &str,
        _system: &str,
        user: &str,
        interrupt: InterruptSignal,
        _max_tokens: Option<u32>,
    ) -> Result<Value> {
        self.complete_json("", purpose, "", user, interrupt).await
    }
    async fn complete_text_on(
        &self,
        _target: &ModelTarget,
        purpose: &str,
        _system: &str,
        user: &str,
        interrupt: InterruptSignal,
        _max_tokens: Option<u32>,
    ) -> Result<String> {
        self.complete_text("", purpose, "", user, interrupt).await
    }
    async fn complete_with_tools_on(
        &self,
        _target: &ModelTarget,
        purpose: &str,
        _system: &str,
        user: &str,
        tools: &[Value],
        interrupt: InterruptSignal,
        _max_tokens: Option<u32>,
    ) -> Result<AssistantTurn> {
        if interrupt.is_set() {
            return Err(lucy_core::LucyError::Cancelled.into());
        }
        self.tool_calls_served.fetch_add(1, Ordering::SeqCst);
        self.record(purpose, user);
        self.record_tools(tools);
        let mut q = self
            .tools_queue
            .lock()
            .map_err(|_| anyhow!("stub tools queue poisoned"))?;
        if q.is_empty() {
            return Err(anyhow!(
                "StubProvider: no tool-calling turn queued for '{purpose}'"
            ));
        }
        Ok(q.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn serves_queued_answers_and_counts_calls() {
        let stub = StubProvider::new()
            .push_json(json!({"a": 1}))
            .push_json(json!({"b": 2}))
            .push_text("done");
        let intr = InterruptSignal::new();
        let first =
            ModelProvider::complete_json(stub.as_ref(), "m", "plan", "s", "u", intr.clone())
                .await
                .unwrap();
        assert_eq!(first, json!({"a": 1}));
        ModelProvider::complete_text(stub.as_ref(), "m", "summary", "s", "u", intr.clone())
            .await
            .unwrap();
        ModelProvider::complete_json_on(
            stub.as_ref(),
            &ModelTarget::new("http://x.test", None, "m"),
            "replan",
            "s",
            "u",
            intr,
            None,
        )
        .await
        .unwrap();
        assert_eq!(stub.calls(), 3);
        assert_eq!(stub.json_calls(), 2);
        assert_eq!(stub.text_calls(), 1);
        assert_eq!(stub.purposes(), vec!["plan", "summary", "replan"]);
    }

    #[tokio::test]
    async fn an_exhausted_queue_errors_instead_of_repeating() {
        let stub = StubProvider::new();
        let intr = InterruptSignal::new();
        let err = ModelProvider::complete_json(stub.as_ref(), "m", "plan", "s", "u", intr.clone())
            .await
            .expect_err("no answer queued");
        assert!(err.to_string().contains("plan"), "{err:#}");
        assert_eq!(stub.calls(), 1);
    }

    #[tokio::test]
    async fn an_interrupt_cancels_before_serving() {
        let stub = StubProvider::new().push_json(json!({"a": 1}));
        let intr = InterruptSignal::new();
        intr.fire();
        let err = ModelProvider::complete_json(stub.as_ref(), "m", "plan", "s", "u", intr)
            .await
            .expect_err("cancelled");
        assert!(err.to_string().contains("cancelled"), "{err:#}");
        assert_eq!(stub.json_calls(), 0);
    }

    fn call(id: &str, name: &str, input: Value) -> lucy_core::ToolCall {
        lucy_core::ToolCall {
            id: id.into(),
            name: name.into(),
            input,
        }
    }

    /// A tool-calling turn is served from its own queue, and it counts as a
    /// slow-lane call: a loop that reached for tools instead of prose spends the
    /// same budget it is measured on.
    #[tokio::test]
    async fn tool_calling_turns_are_served_and_counted() {
        let stub = StubProvider::new()
            .push_calls(vec![call("c1", "read_file", json!({"path": "a.txt"}))])
            .push_turn(AssistantTurn {
                text: Some("done".into()),
                tool_calls: vec![],
            });
        let intr = InterruptSignal::new();
        let catalog = [json!({"name": "read_file", "description": "read", "input_schema": {}})];
        let first = ModelProvider::complete_with_tools_on(
            stub.as_ref(),
            &ModelTarget::new("http://x.test", None, "m"),
            "act",
            "sys",
            "read a.txt",
            &catalog,
            intr.clone(),
            Some(256),
        )
        .await
        .expect("queued turn");
        assert_eq!(first.tool_calls[0].name, "read_file");
        assert_eq!(first.tool_calls[0].input, json!({"path": "a.txt"}));
        assert_eq!(first.text, None);

        let second = ModelProvider::complete_with_tools_on(
            stub.as_ref(),
            &ModelTarget::new("http://x.test", None, "m"),
            "act",
            "sys",
            "anything else?",
            &[],
            intr,
            None,
        )
        .await
        .expect("queued prose turn");
        assert_eq!(second.text.as_deref(), Some("done"));

        assert_eq!(stub.tools_calls(), 2);
        assert_eq!(stub.calls(), 2);
        assert_eq!(stub.purposes(), vec!["act", "act"]);
        // The catalog is recorded per call, so a test can assert what was
        // offered — including that a later turn offered nothing.
        assert_eq!(
            stub.tools_seen(),
            vec![vec!["read_file".to_owned()], Vec::<String>::new()]
        );
    }

    /// An exhausted queue is still an error rather than an empty turn, for the
    /// same reason as the JSON lane: a test must not pass by the stub quietly
    /// reporting "no tools needed".
    #[tokio::test]
    async fn an_exhausted_tools_queue_errors_instead_of_returning_an_empty_turn() {
        let stub = StubProvider::new();
        let err = ModelProvider::complete_with_tools_on(
            stub.as_ref(),
            &ModelTarget::new("http://x.test", None, "m"),
            "act",
            "s",
            "u",
            &[],
            InterruptSignal::new(),
            None,
        )
        .await
        .expect_err("no turn queued");
        assert!(err.to_string().contains("act"), "{err:#}");
        assert_eq!(stub.calls(), 1);
    }

    #[tokio::test]
    async fn an_interrupt_cancels_a_tool_calling_turn_before_serving() {
        let stub = StubProvider::new().push_calls(vec![call("c1", "read_file", json!({}))]);
        let intr = InterruptSignal::new();
        intr.fire();
        let err = ModelProvider::complete_with_tools_on(
            stub.as_ref(),
            &ModelTarget::new("http://x.test", None, "m"),
            "act",
            "s",
            "u",
            &[],
            intr,
            None,
        )
        .await
        .expect_err("cancelled");
        assert!(err.to_string().contains("cancelled"), "{err:#}");
        assert_eq!(stub.tools_calls(), 0);
    }
}
