use std::sync::Arc;

use lucy_config::LucyConfig;
use lucy_stt::GroqStt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_logging();

    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("config") => config_command(args.collect())?,
        Some("session") | Some("sessions") => session_command(args.collect()).await?,
        Some("transcribe") => {
            let path = args
                .next()
                .ok_or_else(|| anyhow::anyhow!("usage: lucy transcribe <audio-file>"))?;
            let stt = load_stt()?;
            println!("{}", stt.transcribe_file(&path).await?);
        }
        Some("voice") => {
            let stt = load_stt().ok().map(Arc::new);
            lucy_tui::run_voice(stt).await?;
        }
        Some("--help") | Some("-h") => print_help(),
        _ => {
            let stt = load_stt().ok().map(Arc::new);
            lucy_tui::run_voice(stt).await?;
        }
    }
    Ok(())
}

#[derive(Clone)]
struct FileMakeWriter {
    file: Arc<std::sync::Mutex<std::fs::File>>,
}
impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for FileMakeWriter {
    type Writer = TeeWriter;
    fn make_writer(&'a self) -> Self::Writer {
        TeeWriter {
            stderr: std::io::stderr(),
            file: self.file.clone(),
        }
    }
}
struct TeeWriter {
    stderr: std::io::Stderr,
    file: Arc<std::sync::Mutex<std::fs::File>>,
}
impl std::io::Write for TeeWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if let Ok(mut f) = self.file.lock() {
            let _ = std::io::Write::write_all(&mut *f, buf);
            let _ = std::io::Write::flush(&mut *f);
        }
        std::io::Write::write(&mut self.stderr, buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        if let Ok(mut f) = self.file.lock() {
            let _ = std::io::Write::flush(&mut *f);
        }
        std::io::Write::flush(&mut self.stderr)
    }
}

fn init_logging() {
    let level = std::env::var("LUCY_LOG_LEVEL")
        .ok()
        .filter(|s| !s.trim().is_empty());
    let log_file = std::env::var("LUCY_LOG_FILE")
        .ok()
        .filter(|s| !s.trim().is_empty());
    let writer: Option<FileMakeWriter> = log_file.and_then(|p| {
        let path = std::path::PathBuf::from(&p);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    eprintln!(
                        "warning: could not create log dir {}: {e}",
                        parent.display()
                    );
                    return None;
                }
            }
        }
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            Ok(f) => Some(FileMakeWriter {
                file: Arc::new(std::sync::Mutex::new(f)),
            }),
            Err(e) => {
                eprintln!("warning: could not open log file {}: {e}", path.display());
                None
            }
        }
    });
    let filter: Option<tracing_subscriber::EnvFilter> = level
        .map(|l| match tracing_subscriber::EnvFilter::try_new(&l) {
            Ok(f) => Some(f),
            Err(e) => {
                eprintln!("warning: invalid LUCY_LOG_LEVEL {l:?}: {e}; using default filter");
                None
            }
        })
        .flatten();
    match (writer, filter) {
        (Some(w), Some(f)) => {
            tracing_subscriber::fmt()
                .with_env_filter(f)
                .with_writer(w)
                .init();
        }
        (Some(w), None) => {
            tracing_subscriber::fmt().with_writer(w).init();
        }
        (None, Some(f)) => {
            tracing_subscriber::fmt()
                .with_env_filter(f)
                .with_writer(std::io::stderr)
                .init();
        }
        (None, None) => {
            tracing_subscriber::fmt::init();
        }
    }
}

fn load_stt() -> anyhow::Result<GroqStt> {
    // Prefer config file (LucyConfig covers file + env), fall back to env-only.
    if let Ok(cfg) = LucyConfig::load() {
        if let Ok(stt) = GroqStt::from_config(&cfg) {
            return Ok(stt);
        }
    }
    GroqStt::from_env()
}

fn config_command(args: Vec<String>) -> anyhow::Result<()> {
    let mut cfg = LucyConfig::load()?;
    match args.first().map(String::as_str) {
        None | Some("show") => println!("{}", toml::to_string_pretty(&cfg)?),
        Some("path") => println!("{}", LucyConfig::path()?.display()),
        Some("get") => {
            let key = args
                .get(1)
                .ok_or_else(|| anyhow::anyhow!("usage: lucy config get <key>"))?;
            println!("{}", cfg.get(key)?);
        }
        Some("set") => {
            let key = args
                .get(1)
                .ok_or_else(|| anyhow::anyhow!("usage: lucy config set <key> <value>"))?;
            let value = args
                .get(2)
                .ok_or_else(|| anyhow::anyhow!("usage: lucy config set <key> <value>"))?;
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
    println!(
        "Lucy\n\nUsage:\n  lucy                         Open the Lucy TUI\n  lucy voice                   Open the voice-enabled TUI\n  lucy transcribe <audio-file> Transcribe an audio file\n  lucy session list            List sessions (opencode-style)\n  lucy session new [title]     Create a new session\n  lucy session show [id]       Show session details\n  lucy session rename <id> <title>  Rename a session\n  lucy session delete <id>     Delete a session\n  lucy session clear <id>      Clear a session's history\n  lucy session export <id> <file.json>  Export a session\n  lucy config                  Show configuration\n  lucy config show             Show configuration\n  lucy config get <key>        Read a setting\n  lucy config set <key> <value> Change a setting\n  lucy config path              Show config file path\n  lucy config init              Create config file if missing\n  lucy config reset             Reset settings to defaults\n  lucy config doctor            Check Lucy configuration\n"
    );
}

fn sessions_store() -> anyhow::Result<lucy_core::SessionStore> {
    let cfg = LucyConfig::load().unwrap_or_default();
    let dir = cfg
        .sessions
        .dir
        .clone()
        .or_else(|| {
            std::env::var("LUCY_SESSIONS_DIR")
                .ok()
                .map(std::path::PathBuf::from)
        })
        .unwrap_or_else(lucy_core::SessionStore::default_dir);
    Ok(lucy_core::SessionStore::new(dir))
}

fn find_session(list: &[lucy_core::SessionMeta], arg: &str) -> Option<lucy_core::SessionMeta> {
    if let Ok(n) = arg.parse::<usize>() {
        if n >= 1 && n <= list.len() {
            return Some(list[n - 1].clone());
        }
    }
    let low = arg.to_ascii_lowercase();
    list.iter()
        .find(|m| {
            m.id.0.to_string().to_ascii_lowercase().starts_with(&low)
                || m.title.to_ascii_lowercase().contains(&low)
        })
        .cloned()
}

async fn session_command(args: Vec<String>) -> anyhow::Result<()> {
    let store = sessions_store()?;
    match args.first().map(String::as_str) {
        None | Some("list") | Some("ls") => {
            let list = store.list().await?;
            if list.is_empty() {
                println!("No sessions yet. Run `lucy` and chat, or `lucy session new`.");
            } else {
                for (i, m) in list.iter().enumerate() {
                    println!(
                        "{:>2}. {}  [{} msgs]  {}  {}",
                        i + 1,
                        m.title,
                        m.message_count,
                        &m.id.0.to_string()[..8.min(m.id.0.to_string().len())],
                        m.preview.chars().take(60).collect::<String>()
                    );
                }
            }
        }
        Some("new") => {
            let title = args.get(1).cloned();
            let s = store.create(title).await?;
            println!("{} {}", s.session_id.0, s.title);
        }
        Some("show") => {
            let list = store.list().await?;
            let meta = args
                .get(1)
                .and_then(|a| find_session(&list, a))
                .or_else(|| list.first().cloned())
                .ok_or_else(|| anyhow::anyhow!("no sessions found"))?;
            let s = store.load(&meta.id).await?;
            println!("{} ({})", s.title, s.session_id.0);
            println!("messages: {}  updated: {}", s.history.len(), s.updated_at);
            for m in &s.history {
                println!("{:?}", m);
            }
        }
        Some("rename") => {
            let id_arg = args
                .get(1)
                .ok_or_else(|| anyhow::anyhow!("usage: lucy session rename <id|number> <title>"))?;
            let title = args
                .get(2..)
                .map(|s| s.join(" "))
                .filter(|t| !t.trim().is_empty())
                .ok_or_else(|| anyhow::anyhow!("usage: lucy session rename <id|number> <title>"))?;
            let list = store.list().await?;
            let meta = find_session(&list, id_arg)
                .ok_or_else(|| anyhow::anyhow!("no session matches '{id_arg}'"))?;
            let mut s = store.load(&meta.id).await?;
            s.title = title.trim().to_owned();
            s.updated_at = lucy_core::now_secs();
            store.save(&s).await?;
            println!("Renamed to: {}", s.title);
        }
        Some("delete") | Some("rm") => {
            let id_arg = args
                .get(1)
                .ok_or_else(|| anyhow::anyhow!("usage: lucy session delete <id|number>"))?;
            let list = store.list().await?;
            let meta = find_session(&list, id_arg)
                .ok_or_else(|| anyhow::anyhow!("no session matches '{id_arg}'"))?;
            store.delete(&meta.id).await?;
            println!("Deleted: {}", meta.title);
        }
        Some("clear") => {
            let id_arg = args
                .get(1)
                .ok_or_else(|| anyhow::anyhow!("usage: lucy session clear <id|number>"))?;
            let list = store.list().await?;
            let meta = find_session(&list, id_arg)
                .ok_or_else(|| anyhow::anyhow!("no session matches '{id_arg}'"))?;
            let mut s = store.load(&meta.id).await?;
            s.history.clear();
            s.updated_at = lucy_core::now_secs();
            store.save(&s).await?;
            println!("Cleared: {}", s.title);
        }
        Some("export") => {
            let id_arg = args.get(1).ok_or_else(|| {
                anyhow::anyhow!("usage: lucy session export <id|number> <file.json>")
            })?;
            let file = args.get(2).ok_or_else(|| {
                anyhow::anyhow!("usage: lucy session export <id|number> <file.json>")
            })?;
            let list = store.list().await?;
            let meta = find_session(&list, id_arg)
                .ok_or_else(|| anyhow::anyhow!("no session matches '{id_arg}'"))?;
            let s = store.load(&meta.id).await?;
            s.save_to_file(file).await?;
            println!("Exported {} to {file}", s.title);
        }
        Some(other) => anyhow::bail!(
            "unknown session command: {other}\nusage: lucy session [list|new|show|rename|delete|clear|export]"
        ),
    }
    Ok(())
}
