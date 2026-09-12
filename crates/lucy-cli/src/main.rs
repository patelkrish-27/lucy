use std::sync::Arc;
use lucy_core::*;
use lucy_agent::Agent;
use lucy_tools::{ShellTool, ToolRegistry};
use tokio::sync::mpsc;

struct PlaceholderProvider;

#[async_trait::async_trait]
impl ModelProvider for PlaceholderProvider {
    async fn run_turn(&self, request: ModelRequest, _events: mpsc::UnboundedSender<AgentEvent>, interrupt: InterruptSignal) -> anyhow::Result<ModelTurn> {
        if interrupt.is_set() { return Err(LucyError::Cancelled.into()); }
        Ok(ModelTurn {
            text: Some(format!("Lucy foundation received: {}", request.prompt)),
            tool_calls: Vec::new(),
            stop: true,
        })
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let mut registry = ToolRegistry::new();
    registry.register(ShellTool::default());
    let agent = Agent::new(Arc::new(PlaceholderProvider), Arc::new(registry));
    let mut rx = agent.execute("hello Lucy".into(), std::env::current_dir().ok(), InterruptSignal::new()).await?;
    while let Some(event) = rx.recv().await { println!("{event:?}"); }
    Ok(())
}
