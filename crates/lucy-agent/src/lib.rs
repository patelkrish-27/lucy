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
            history.push(TurnMessage::Assistant(text));
        }

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
