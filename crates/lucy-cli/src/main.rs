use std::sync::Arc;

use lucy_config::LucyConfig;
use lucy_stt::GroqStt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("config") => config_command(args.collect())?,
        Some("transcribe") => {
            let path = args.next().ok_or_else(|| anyhow::anyhow!("usage: lucy transcribe <audio-file>"))?;
            let stt = GroqStt::from_env()?;
            println!("{}", stt.transcribe_file(&path).await?);
        }
        Some("voice") => {
            let stt = GroqStt::from_env().ok().map(Arc::new);
            lucy_tui::run_voice(stt).await?;
        }
        Some("--help") | Some("-h") => print_help(),
        _ => {
            let stt = GroqStt::from_env().ok().map(Arc::new);
            lucy_tui::run_voice(stt).await?;
        }
    }
    Ok(())
}

fn config_command(args: Vec<String>) -> anyhow::Result<()> {
    let mut cfg = LucyConfig::load()?;
    match args.first().map(String::as_str) {
        None | Some("show") => println!("{}", toml::to_string_pretty(&cfg)?),
        Some("path") => println!("{}", LucyConfig::path()?.display()),
        Some("get") => {
            let key = args.get(1).ok_or_else(|| anyhow::anyhow!("usage: lucy config get <key>"))?;
            println!("{}", cfg.get(key)?);
        }
        Some("set") => {
            let key = args.get(1).ok_or_else(|| anyhow::anyhow!("usage: lucy config set <key> <value>"))?;
            let value = args.get(2).ok_or_else(|| anyhow::anyhow!("usage: lucy config set <key> <value>"))?;
            cfg.set(key, value)?;
            cfg.save()?;
            println!("Updated {key}");
        }
        Some("reset") => {
            cfg.reset();
            cfg.save()?;
            println!("Lucy configuration reset to defaults.");
        }
        Some("init") => {
            cfg.init_if_missing()?;
            println!("{}", LucyConfig::path()?.display());
        }
        Some("doctor") => {
            println!("Lucy configuration\n");
            for (name, ok, detail) in lucy_config::doctor() {
                println!("{} {:<12} {}", if ok { "✓" } else { "✗" }, name, detail);
            }
        }
        Some(other) => anyhow::bail!("unknown config command: {other}"),
    }
    Ok(())
}

fn print_help() {
    println!("Lucy\n\nUsage:\n  lucy                         Open the Lucy TUI\n  lucy voice                   Open the voice-enabled TUI\n  lucy transcribe <audio-file> Transcribe an audio file\n  lucy config                  Show configuration\n  lucy config show             Show configuration\n  lucy config get <key>        Read a setting\n  lucy config set <key> <value> Change a setting\n  lucy config path              Show config file path\n  lucy config init              Create config file if missing\n  lucy config reset             Reset settings to defaults\n  lucy config doctor            Check Lucy configuration\n");
}
