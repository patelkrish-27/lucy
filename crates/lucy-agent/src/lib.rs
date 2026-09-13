use std::sync::Arc;
use anyhow::Result;
use lucy_core::*;
use lucy_tools::ToolRegistry;
use tokio::sync::mpsc;

pub mod provider;
pub use provider::OpenAIProvider;

const MAX_TOOL_TURNS: usize = 64;

pub struct Agent<P: ModelProvider> {
    provider: Arc<P>,
    tools: Arc<ToolRegistry>,
}

impl<P: ModelProvider + 'static> Agent<P> {
    pub fn new(provider: Arc<P>, tools: Arc<ToolRegistry>) -> Self { Self { provider, tools } }

    pub async fn execute(&self, prompt: String, working_dir: Option<std::path::PathBuf>, interrupt: InterruptSignal) -> Result<mpsc::UnboundedReceiver<AgentEvent>> {
        let (tx, rx) = mpsc::unbounded_channel();
        let provider = self.provider.clone();
        let tools = self.tools.clone();
        tokio::spawn(async move {
            if let Err(err) = run_loop(provider, tools, prompt, working_dir, interrupt, tx.clone()).await {
                let _ = tx.send(AgentEvent::Error { message: err.to_string() });
            }
            let _ = tx.send(AgentEvent::Done);
        });
        Ok(rx)
    }
}

async fn run_loop<P: ModelProvider>(
    provider: Arc<P>,
    tools: Arc<ToolRegistry>,
    prompt: String,
    working_dir: Option<std::path::PathBuf>,
    interrupt: InterruptSignal,
    tx: mpsc::UnboundedSender<AgentEvent>,
) -> Result<()> {
    let session_id = SessionId::default();
    let mut history = vec![TurnMessage::User(prompt.clone())];
    let mut turns = 0usize;

    loop {
        if interrupt.is_set() { return Err(LucyError::Cancelled.into()); }
        if turns >= MAX_TOOL_TURNS { return Err(anyhow::anyhow!("tool-turn limit reached ({MAX_TOOL_TURNS})")); }
        turns += 1;
        let _ = tx.send(AgentEvent::Status { message: format!("Planning turn {turns}…") });

        let request = ModelRequest {
            session_id: session_id.clone(),
            prompt: prompt.clone(),
            history: history.clone(),
            tools: tools.definitions(),
        };
        let turn = provider.run_turn(request, tx.clone(), interrupt.clone()).await?;

        if let Some(text) = turn.text.clone() {
            let _ = tx.send(AgentEvent::TextDelta { text: text.clone() });
        }
        history.push(TurnMessage::Assistant(AssistantTurn {
            text: turn.text.clone(),
            tool_calls: turn.tool_calls.clone(),
        }));

        if turn.tool_calls.is_empty() || turn.stop {
            break;
        }

        let tool_calls = turn.tool_calls;
        for call in &tool_calls {
            if interrupt.is_set() { return Err(LucyError::Cancelled.into()); }
            let _ = tx.send(AgentEvent::ToolStarted { id: call.id.clone(), name: call.name.clone() });
        }

        // Independent tool calls from one model turn can run concurrently. Keep
        // results in model order so the next request remains deterministic.
        let results = futures::future::join_all(tool_calls.into_iter().map(|call| {
            let tools = tools.clone();
            let interrupt = interrupt.clone();
            let tx = tx.clone();
            let session_id = session_id.clone();
            let working_dir = working_dir.clone();
            async move {
                let ctx = ToolContext {
                    session_id,
                    tool_call_id: call.id.clone(),
                    working_dir,
                    execution_mode: ExecutionMode::Agent,
                    events: tx.clone(),
                    interrupt: interrupt.clone(),
                };
                let result = tools.execute(&call.name, call.input.clone(), ctx).await;
                (call, result)
            }
        })).await;

        for (call, result) in results {
            let (output, is_error) = match result {
                Ok(output) => (output, false),
                Err(err) => (serde_json::json!({"error": err.to_string()}), true),
            };
            let _ = tx.send(AgentEvent::ToolFinished { id: call.id.clone(), output: output.clone() });
            history.push(TurnMessage::Tool(ToolResult { call_id: call.id, name: call.name, output, is_error }));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;
    use serde_json::json;
    use std::collections::VecDeque;
    use std::sync::Mutex;
    use std::time::Duration;

    struct RecordingProvider {
        turns: Mutex<VecDeque<ModelTurn>>,
        requests: Mutex<Vec<ModelRequest>>,
    }

    impl RecordingProvider {
        fn new(turns: Vec<ModelTurn>) -> Self {
            Self {
                turns: Mutex::new(turns.into()),
                requests: Mutex::new(Vec::new()),
            }
        }

        fn requests(&self) -> Vec<ModelRequest> {
            self.requests.lock().expect("requests lock poisoned").clone()
        }
    }

    #[async_trait::async_trait]
    impl ModelProvider for RecordingProvider {
        async fn run_turn(
            &self,
            request: ModelRequest,
            _events: mpsc::UnboundedSender<AgentEvent>,
            _interrupt: InterruptSignal,
        ) -> anyhow::Result<ModelTurn> {
            self.requests.lock().expect("requests lock poisoned").push(request);
            self.turns
                .lock()
                .expect("turns lock poisoned")
                .pop_front()
                .ok_or_else(|| anyhow!("no turns left"))
        }
    }

    struct DelayEchoTool;
    #[async_trait::async_trait]
    impl Tool for DelayEchoTool {
        fn name(&self) -> &str {
            "delay_echo"
        }
        fn description(&self) -> &str {
            "delayed echo"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            json!({
                "type":"object",
                "properties":{
                    "delay_ms":{"type":"integer"},
                    "result":{"type":"string"}
                },
                "required":["delay_ms","result"]
            })
        }
        async fn execute(&self, input: serde_json::Value, _ctx: ToolContext) -> anyhow::Result<serde_json::Value> {
            let delay_ms = input
                .get("delay_ms")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| anyhow!("delay_ms is required"))?;
            let result = input
                .get("result")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| anyhow!("result is required"))?;
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            Ok(json!(result))
        }
    }

    #[tokio::test]
    async fn preserves_assistant_tool_calls_and_tool_results_in_history() {
        let provider = Arc::new(RecordingProvider::new(vec![
            ModelTurn {
                text: Some("Thinking".into()),
                tool_calls: vec![ToolCall {
                    id: "tool-1".into(),
                    name: "delay_echo".into(),
                    input: json!({"delay_ms": 1, "result": "ok"}),
                }],
                stop: false,
            },
            ModelTurn {
                text: Some("Done".into()),
                tool_calls: vec![],
                stop: true,
            },
        ]));
        let mut registry = ToolRegistry::new();
        registry.register(DelayEchoTool);
        let tools = Arc::new(registry);
        let (tx, _rx) = mpsc::unbounded_channel();

        run_loop(
            provider.clone(),
            tools,
            "hello".into(),
            None,
            InterruptSignal::new(),
            tx,
        )
        .await
        .expect("agent loop should succeed");

        let requests = provider.requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].history.len(), 1);

        let second_history = &requests[1].history;
        assert!(matches!(second_history[0], TurnMessage::User(_)));
        match &second_history[1] {
            TurnMessage::Assistant(turn) => {
                assert_eq!(turn.text.as_deref(), Some("Thinking"));
                assert_eq!(turn.tool_calls.len(), 1);
                assert_eq!(turn.tool_calls[0].id, "tool-1");
            }
            _ => panic!("expected assistant turn with tool calls"),
        }
        match &second_history[2] {
            TurnMessage::Tool(result) => {
                assert_eq!(result.call_id, "tool-1");
                assert_eq!(result.output, json!("ok"));
            }
            _ => panic!("expected tool result"),
        }
    }

    #[tokio::test]
    async fn preserves_tool_result_order_for_concurrent_tool_calls() {
        let provider = Arc::new(RecordingProvider::new(vec![
            ModelTurn {
                text: None,
                tool_calls: vec![
                    ToolCall {
                        id: "first".into(),
                        name: "delay_echo".into(),
                        input: json!({"delay_ms": 50, "result": "first"}),
                    },
                    ToolCall {
                        id: "second".into(),
                        name: "delay_echo".into(),
                        input: json!({"delay_ms": 1, "result": "second"}),
                    },
                ],
                stop: false,
            },
            ModelTurn {
                text: Some("done".into()),
                tool_calls: vec![],
                stop: true,
            },
        ]));
        let mut registry = ToolRegistry::new();
        registry.register(DelayEchoTool);
        let tools = Arc::new(registry);
        let (tx, _rx) = mpsc::unbounded_channel();

        run_loop(
            provider.clone(),
            tools,
            "order".into(),
            None,
            InterruptSignal::new(),
            tx,
        )
        .await
        .expect("agent loop should succeed");

        let requests = provider.requests();
        assert_eq!(requests.len(), 2);
        let second_history = &requests[1].history;
        let tool_results: Vec<&ToolResult> = second_history
            .iter()
            .filter_map(|m| match m {
                TurnMessage::Tool(result) => Some(result),
                _ => None,
            })
            .collect();
        assert_eq!(tool_results.len(), 2);
        assert_eq!(tool_results[0].call_id, "first");
        assert_eq!(tool_results[1].call_id, "second");
    }

    #[tokio::test]
    async fn returns_cancelled_when_interrupt_is_set() {
        let provider = Arc::new(RecordingProvider::new(vec![]));
        let tools = Arc::new(ToolRegistry::new());
        let (tx, _rx) = mpsc::unbounded_channel();
        let interrupt = InterruptSignal::new();
        interrupt.fire();

        let err = run_loop(provider, tools, "cancel".into(), None, interrupt, tx)
            .await
            .expect_err("run loop should cancel");
        assert!(err.to_string().contains("cancelled"));
    }

    #[tokio::test]
    async fn enforces_tool_turn_limit() {
        let repeated_turn = ModelTurn {
            text: None,
            tool_calls: vec![ToolCall {
                id: "loop".into(),
                name: "delay_echo".into(),
                input: json!({"delay_ms": 0, "result": "loop"}),
            }],
            stop: false,
        };
        let provider = Arc::new(RecordingProvider::new(vec![repeated_turn; MAX_TOOL_TURNS + 1]));
        let mut registry = ToolRegistry::new();
        registry.register(DelayEchoTool);
        let tools = Arc::new(registry);
        let (tx, _rx) = mpsc::unbounded_channel();

        let err = run_loop(
            provider,
            tools,
            "loop".into(),
            None,
            InterruptSignal::new(),
            tx,
        )
        .await
        .expect_err("run loop should stop at turn limit");
        assert!(err.to_string().contains("tool-turn limit reached"));
    }
}
