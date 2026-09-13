use std::sync::Arc;
use lucy_runtime::LucyRuntime;
use lucy_stt::GroqStt;

#[tokio::main]
async fn main()->anyhow::Result<()>{
 tracing_subscriber::fmt::init();
 let mut args=std::env::args().skip(1);
 match args.next().as_deref(){
  Some("transcribe")=>{let path=args.next().ok_or_else(||anyhow::anyhow!("usage: lucy transcribe <audio-file>"))?;let stt=GroqStt::from_env()?;println!("{}",stt.transcribe_file(path).await?);}
  Some("--help")|Some("-h")=>println!("Lucy — AI computer buddy\n\nUsage:\n  lucy                         Start Lucy\n  lucy transcribe <file>      Transcribe an audio file\n\nKeys:\n  Enter                       Run task\n  Super+C / Ctrl+Space / F2   Speak a task\n  Ctrl+C                      Stop current task\n  Esc                         Quit\n\nEnvironment:\n  OPENAI_API_KEY              Required for agent mode\n  OPENAI_MODEL                Optional model (default: gpt-4o)\n  OPENAI_BASE_URL             Optional OpenAI-compatible endpoint\n  GROQ_API_KEY                Required for voice\n  LUCY_MCP_CONFIG             Optional MCP TOML path\n  LUCY_ALLOW_DANGEROUS=1      Disable built-in shell danger block (use deliberately)\n  LUCY_SESSION_FILE           Optional session JSON path"),
  _=>{let runtime=LucyRuntime::new().await.ok().map(Arc::new);let stt=GroqStt::from_env().ok().map(Arc::new);lucy_tui::run_voice(stt,runtime).await?;}
 }
 Ok(())
}
