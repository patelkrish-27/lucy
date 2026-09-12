use std::sync::Arc;
use lucy_core::*;
use lucy_agent::Agent;
use lucy_tools::{ShellTool, ToolRegistry};
use lucy_stt::GroqStt;
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

    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("transcribe") => {
            let path = args.next().ok_or_else(|| anyhow::anyhow!("usage: lucy transcribe <audio-file>"))?;
            let stt = GroqStt::from_env()?;
            let text = stt.transcribe_file(&path).await?;
            println!("{text}");
            return Ok(());
        }
        Some("--help") | Some("-h") => {
            println!("Lucy\n\nUsage:\n  lucy transcribe <audio-file>  Transcribe audio with Groq Whisper\n  lucy                       Run the Lucy foundation\n\nEnvironment:\n  GROQ_API_KEY              Required for Groq STT\n  LUCY_STT_MODEL            Optional, defaults to whisper-large-v3-turbo\n  LUCY_STT_LANGUAGE         Optional language code, e.g. en\n  LUCY_STT_PROMPT           Optional transcription context");
            return Ok(());
        }
        _ => {}
    }

    let mut registry = ToolRegistry::new();
    registry.register(ShellTool::default());
    let agent = Agent::new(Arc::new(PlaceholderProvider), Arc::new(registry));
    let mut rx = agent.execute("hello Lucy".into(), std::env::current_dir().ok(), InterruptSignal::new()).await?;
    while let Some(event) = rx.recv().await { println!("{event:?}"); }
    Ok(())
}
