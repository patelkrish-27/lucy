use std::sync::Arc;
use anyhow::Result;
use lucy_core::{AgentEvent, InterruptSignal, ModelProvider};
use lucy_tools::{ShellTool, ToolRegistry};
use async_trait::async_trait;

struct PlaceholderProvider;
#[async_trait]
impl ModelProvider for PlaceholderProvider {
    async fn run(&self, prompt: &str, events: tokio::sync::mpsc::UnboundedSender<AgentEvent>, interrupt: InterruptSignal) -> Result<()> {
        if interrupt.is_set() { return Ok(()); }
        let _ = events.send(AgentEvent::TextDelta { text: format!("Lucy foundation received: {prompt}") });
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    let mut registry = ToolRegistry::new();
    registry.register(ShellTool);
    let _agent = lucy_agent::Agent::new(Arc::new(PlaceholderProvider), Arc::new(registry));
    lucy_tui::run(None)
}
