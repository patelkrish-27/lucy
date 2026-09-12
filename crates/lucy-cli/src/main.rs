use std::sync::Arc;

use lucy_stt::GroqStt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("transcribe") => {
            let path = args
                .next()
                .ok_or_else(|| anyhow::anyhow!("usage: lucy transcribe <audio-file>"))?;
            let stt = GroqStt::from_env()?;
            println!("{}", stt.transcribe_file(&path).await?);
        }
        Some("voice") => {
            let stt = GroqStt::from_env().ok().map(Arc::new);
            lucy_tui::run_voice(stt).await?;
        }
        Some("--help") | Some("-h") => {
            println!(
                "Lucy\n\nUsage:\n  lucy                         Open the Lucy TUI (keyboard + voice)\n  lucy voice                   Open the voice-enabled TUI\n  lucy transcribe <audio-file> Transcribe an existing audio file\n\nTUI controls:\n  Type + Enter                 Send keyboard prompt\n  Super+C / Ctrl+Space / F2 / F9 / Alt+V / Ctrl+M  Listen for one spoken command\n  Esc                          Quit\n\nVoice input:\n  Microphone -> speech detection -> WAV chunk -> Groq Whisper -> command shown in TUI\n  (Tip: If Super+C is captured by Hyprland, use Ctrl+Space or F2)\n\nEnvironment:\n  GROQ_API_KEY                 Required for voice transcription\n  LUCY_STT_MODEL               Optional; defaults to whisper-large-v3-turbo\n  LUCY_STT_LANGUAGE            Optional language code, e.g. en\n  LUCY_STT_PROMPT              Optional transcription context"
            );
        }
        _ => {
            let stt = GroqStt::from_env().ok().map(Arc::new);
            lucy_tui::run_voice(stt).await?;
        }
    }

    Ok(())
}
