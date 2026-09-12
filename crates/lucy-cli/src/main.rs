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
        Ok(ModelTurn { text: Some(format!("Lucy foundation received: {}", request.prompt)), tool_calls: Vec::new(), stop: true })
    }
}

async fn run_agent(agent: &Agent<PlaceholderProvider>, prompt: String) -> anyhow::Result<()> {
    let mut rx = agent.execute(prompt, std::env::current_dir().ok(), InterruptSignal::new()).await?;
    while let Some(event) = rx.recv().await { println!("{event:?}"); }
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("transcribe") => {
            let path = args.next().ok_or_else(|| anyhow::anyhow!("usage: lucy transcribe <audio-file>"))?;
            let stt = GroqStt::from_env()?;
            println!("{}", stt.transcribe_file(&path).await?);
            return Ok(());
        }
        Some("voice") => {
            let stt = Arc::new(GroqStt::from_env()?);
            let mut registry = ToolRegistry::new();
            registry.register(ShellTool::default());
            let agent = Arc::new(Agent::new(Arc::new(PlaceholderProvider), Arc::new(registry)));
            println!("Lucy voice mode — speak naturally. Ctrl+C to stop.");
            println!("Microphone listens automatically; pause after an utterance to send it to Lucy.\n");
            loop {
                match stt.listen_once().await {
                    Ok(text) => {
                        println!("You: {text}");
                        if let Err(err) = run_agent(&agent, text).await { eprintln!("Lucy error: {err}"); }
                        println!();
                    }
                    Err(err) => {
                        eprintln!("Voice input error: {err}");
                        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                    }
                }
            }
        }
        Some("--help") | Some("-h") => {
            println!("Lucy\n\nUsage:\n  lucy voice                    Live microphone → Groq Whisper → Lucy\n  lucy transcribe <audio-file>  Transcribe an existing audio file\n  lucy                          Run the Lucy foundation\n\nEnvironment:\n  GROQ_API_KEY       Required for Groq STT\n  LUCY_STT_MODEL     Optional; defaults to whisper-large-v3-turbo\n  LUCY_STT_LANGUAGE  Optional language code, e.g. en\n  LUCY_STT_PROMPT    Optional transcription context");
            return Ok(());
        }
        _ => {}
    }
    let mut registry = ToolRegistry::new();
    registry.register(ShellTool::default());
    let agent = Agent::new(Arc::new(PlaceholderProvider), Arc::new(registry));
    run_agent(&agent, "hello Lucy".into()).await
}
