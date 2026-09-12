use std::sync::Arc;
use anyhow::Result;
use lucy_core::*;
use lucy_tools::ToolRegistry;
use tokio::sync::mpsc;

pub struct Agent<P: ModelProvider> { provider: Arc<P>, tools: Arc<ToolRegistry> }
impl<P: ModelProvider> Agent<P> {
    pub fn new(provider: Arc<P>, tools: Arc<ToolRegistry>) -> Self { Self { provider, tools } }
    pub async fn execute(&self, prompt: String, working_dir: Option<std::path::PathBuf>, interrupt: InterruptSignal) -> Result<mpsc::UnboundedReceiver<AgentEvent>> {
        let (tx, rx) = mpsc::unbounded_channel();
        let provider = self.provider.clone();
        let tools = self.tools.clone();
        tokio::spawn(async move {
            let _ = tx.send(AgentEvent::Status { message: "Planning…".into() });
            if let Err(err) = provider.run(&prompt, tx.clone(), interrupt.clone()).await {
                let _ = tx.send(AgentEvent::Error { message: err.to_string() });
                let _ = tx.send(AgentEvent::Done);
                return;
            }
            let _ = tools;
            let _ = working_dir;
            let _ = tx.send(AgentEvent::Done);
        });
        Ok(rx)
    }
}
