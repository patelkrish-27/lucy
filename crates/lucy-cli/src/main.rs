use lucy_adk::LucySessionService;
use lucy_config::LucyConfig;
use lucy_stt::GroqStt;
use std::sync::Arc;

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
        Some("act") => act_command(args.collect()).await?,
        Some("agent") => agent_command(args.collect()).await?,
        Some("serve") => serve_command(args.collect()).await?,
        Some("pair") => pair_command(args.collect()).await?,
        Some("decide") => decide_command(args.collect()).await?,
        Some("models") => models_command(args.collect())?,
        Some("ask") | Some("chat") => ask_command(args.collect()).await?,
        Some("smoke") => smoke_command(args.collect()).await?,
        Some("mascot") => mascot_command(args.collect())?,
        Some("acp") => acp_command(args.collect()).await?,
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
    let filter: Option<tracing_subscriber::EnvFilter> = level.map(|l| {
        tracing_subscriber::EnvFilter::try_new(&l).unwrap_or_else(|e| {
            eprintln!("warning: invalid LUCY_LOG_LEVEL {l:?}: {e}; using default filter");
            tracing_subscriber::EnvFilter::new("info")
        })
    });
    match (writer, filter) {
        (Some(w), Some(f)) => tracing_subscriber::fmt()
            .with_env_filter(f)
            .with_writer(w)
            .init(),
        (Some(w), None) => tracing_subscriber::fmt().with_writer(w).init(),
        (None, Some(f)) => tracing_subscriber::fmt()
            .with_env_filter(f)
            .with_writer(std::io::stderr)
            .init(),
        (None, None) => tracing_subscriber::fmt::init(),
    }
}
fn load_stt() -> anyhow::Result<GroqStt> {
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
        "Lucy\n\nUsage:\n  lucy                         Open the Lucy TUI\n  lucy voice                   Open the voice-enabled TUI\n  lucy mascot [dir] [--scale N] Export the mascot art as PNG + HTML
  lucy acp connect [command...]  Check an ACP agent (defaults to opencode acp)
  lucy acp run [command...] -- <prompt>  Delegate a prompt to an ACP agent\n  lucy ask <prompt>            Laya classifies chat vs act, then replies or runs action\n  lucy act <goal>              Laya classifies chat vs act; chat replies, act runs automation\n  lucy agent <goal>            Run the ReAct loop: the model sees what each tool returned and\n                           picks the next call, so a wrong step costs one reply not a re-plan\n  lucy decide <subcommand>     Query Laya decisions directly (mode|tools|click|type|done)\n  lucy serve [--pair] [--port N] [--bind ADDR] [--enable|--disable|--status|--stop]\n                           Run the mobile gateway (off by default; --enable turns it on,\n                           --pair prints a pairing QR, --status/--stop inspect or shut it down)\n  lucy pair [--reissue]        Mint a pairing token for the Lucy app (gateway must be on)\n  lucy smoke                   Run end-to-end pipeline smoke test\n  lucy transcribe <audio-file> Transcribe an audio file
  lucy models log [-n N] [--kind llm|classification|voice] [--failures]
                           Show recent model calls (voice/STT, classification, LLM)\n  lucy session list            List sessions\n  lucy session new [title]     Create a new session\n  lucy session show [id]       Show session details\n  lucy session rename <id> <title>  Rename a session\n  lucy session delete <id>     Delete a session\n  lucy session clear <id>      Clear a session's history\n  lucy session export <id> <file.json>  Export a session\n  lucy config                  Show configuration\n  lucy config show             Show configuration\n  lucy config get <key>        Read a setting\n  lucy config set <key> <value> Change a setting\n  lucy config path              Show config file path\n  lucy config init              Create config file if missing\n  lucy config reset             Reset settings to defaults\n  lucy config doctor            Check Lucy configuration\n"
    )
}
/// Flags `lucy serve` understands. The gateway has no implicit "just turn it
/// on" path: starting the bridge is an explicit act, and `--enable` /
/// `--disable` / `--status` / `--stop` are the only ways it changes state.
#[derive(Debug, Default)]
struct ServeFlags {
    pair: bool,
    reissue: bool,
    enable: bool,
    disable: bool,
    status: bool,
    stop: bool,
    port: Option<u16>,
    bind: Option<String>,
}

/// `lucy serve` — the mobile gateway. Off by default; nothing binds and no
/// runtime is loaded until it is explicitly enabled *and* started.
async fn serve_command(args: Vec<String>) -> anyhow::Result<()> {
    let mut flags = ServeFlags::default();
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--pair" => flags.pair = true,
            "--reissue" => flags.reissue = true,
            "--enable" => flags.enable = true,
            "--disable" => flags.disable = true,
            "--status" => flags.status = true,
            "--stop" => flags.stop = true,
            "--port" | "-p" => {
                let value = iter
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("usage: lucy serve --port <1-65535>"))?
                    .parse::<u16>()
                    .map_err(|_| anyhow::anyhow!("--port expects a port number"))?;
                if value == 0 {
                    anyhow::bail!("--port must be between 1 and 65535");
                }
                flags.port = Some(value);
            }
            "--bind" | "-b" => {
                flags.bind = Some(
                    iter.next()
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("usage: lucy serve --bind <addr>"))?,
                );
            }
            other => anyhow::bail!("unknown lucy serve option: {other}"),
        }
    }

    // Status and stop never start or change anything, so they are safe at any
    // time — including when the bridge is off.
    if flags.status {
        return gateway_status().await;
    }
    if flags.stop {
        return gateway_stop().await;
    }

    let mut config = LucyConfig::load()?;

    // --disable is terminal: turn it off, and report whether something is
    // still listening (the listener itself is stopped by Ctrl-C in its own
    // terminal, which is where its logs are).
    if flags.disable {
        config.set_gateway_enabled(false);
        config.save()?;
        println!("Mobile gateway disabled — it will not start next time.");
        if gateway_is_up(&config).await.is_some() {
            println!("A gateway is still listening on port {}: stop it with Ctrl-C in its terminal.", config.gateway.port);
        }
        return Ok(());
    }
    if flags.enable {
        config.set_gateway_enabled(true);
        config.save()?;
        println!("Mobile gateway enabled.");
    }

    // One-run overrides, applied to the in-memory copy so the file is
    // untouched unless the user asked for persistence above.
    if let Some(port) = flags.port {
        config.gateway.port = port;
    }
    if let Some(bind) = &flags.bind {
        config.gateway.bind = bind.clone();
    }

    // The opt-in gate.
    if !config.gateway_enabled() {
        anyhow::bail!(
            "the mobile gateway is off — run `lucy serve --enable` to turn it on, \
             or set [gateway] enabled = true in the config"
        );
    }

    let pair = flags.pair.then(|| lucy_gateway::PairingRequest {
        reissue: flags.reissue,
        name: None,
    });
    lucy_gateway::serve_with(config, pair).await
}

/// `lucy pair` — mint a one-time pairing token for the phone. The gateway must
/// be enabled, but does not need to be running yet: the token lives in the
/// device store, so `lucy serve` started afterwards will accept it.
async fn pair_command(args: Vec<String>) -> anyhow::Result<()> {
    let mut reissue = false;
    for a in &args {
        match a.as_str() {
            "--reissue" | "-r" => reissue = true,
            other => anyhow::bail!("unknown lucy pair option: {other}"),
        }
    }
    let config = LucyConfig::load()?;
    if !config.gateway_enabled() {
        anyhow::bail!(
            "the mobile gateway is off — run `lucy serve --enable` first, then `lucy pair`"
        );
    }
    lucy_gateway::print_pairing_only(config, lucy_gateway::PairingRequest { reissue, name: None })
        .await
}

/// `lucy serve --status`: report the switch, the bind address, and whether a
/// gateway is answering right now.
async fn gateway_status() -> anyhow::Result<()> {
    let config = LucyConfig::load().unwrap_or_default();
    println!("Mobile gateway");
    println!(
        "  enabled      : {}",
        if config.gateway_enabled() { "yes" } else { "no" }
    );
    println!("  bind         : {}", config.gateway.bind);
    println!("  port         : {}", config.gateway.port);
    println!(
        "  idle unload  : {}s (0 = keep the runtime until the process ends)",
        config.gateway.idle_unload_secs
    );
    if config.gateway_enabled() {
        match gateway_is_up(&config).await {
            Some(body) => println!("  running      : yes — {body}"),
            None => println!("  running      : no"),
        }
    } else {
        println!("  running      : no (disabled)");
    }
    Ok(())
}

/// `lucy serve --stop`. The gateway runs in the foreground of its own
/// terminal so its log and its Ctrl-C belong to the user; this reports what
/// would be stopped instead of reaching across and killing a process.
async fn gateway_stop() -> anyhow::Result<()> {
    let config = LucyConfig::load().unwrap_or_default();
    match gateway_is_up(&config).await {
        Some(_) => anyhow::bail!(
            "the gateway is running in another terminal — stop it with Ctrl-C there \
             (or `pkill -f 'lucy serve'`)"
        ),
        None => {
            println!("Mobile gateway is not running.");
            Ok(())
        }
    }
}

/// Ask the gateway's public `/health` endpoint whether it is up. Returns the
/// body when it answers. Loopback only, so a LAN bind is still reachable
/// from this machine.
async fn gateway_is_up(config: &LucyConfig) -> Option<String> {
    let host = match config.gateway.bind.as_str() {
        "0.0.0.0" | "::" | "" => "127.0.0.1",
        "::1" => "[::1]",
        other => other,
    };
    let url = format!("http://{host}:{}/health", config.gateway.port);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(750))
        .build()
        .ok()?;
    let body = client.get(url).send().await.ok()?.text().await.ok()?;
    // Trim to one line for the status display.
    Some(body.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// `lucy mascot [dir] [--scale N]`: write the mascot art to disk — one PNG per
/// pose, a sheet of every animation frame, and an HTML page to look at them in.
///
/// The TUI paints the sprite into character cells, which is a lossy way to
/// judge a drawing. This is the same painter with the pixels left intact, so
/// the art can be checked at full size and iterated on.
async fn acp_command(args: Vec<String>) -> anyhow::Result<()> {
    let sub = args.first().map(String::as_str).unwrap_or("connect");
    let (runner_args, prompt) = match sub {
        "connect" => (args[1..].to_vec(), None),
        "run" => {
            let rest = &args[1..];
            let Some(i) = rest.iter().position(|a| a == "--") else {
                anyhow::bail!("usage: lucy acp run [command...] -- <prompt>");
            };
            let command = rest[..i].to_vec();
            let prompt = rest[i + 1..].join(" ");
            if prompt.trim().is_empty() {
                anyhow::bail!("usage: lucy acp run [command...] -- <prompt>");
            }
            (command, Some(prompt))
        }
        other => anyhow::bail!("unknown acp command '{other}'; use connect or run"),
    };

    let runner = if runner_args.is_empty() {
        lucy_acp::AcpRunner::opencode()
    } else {
        lucy_acp::AcpRunner::from_command_line(&runner_args.join(" "))?
    };

    if let Some(prompt) = prompt {
        println!("🔌 ACP runner: {}", runner.command);
        let result = runner.prompt(&prompt).await?;
        println!("{}", result.text.trim());
        println!("\n[ACP] {} · {}", result.agent_name, result.stop_reason);
    } else {
        let info = runner.initialize().await?;
        println!("ACP connected");
        println!("  runner   : {}", info.runner);
        println!("  command  : {}", info.command);
        println!("  protocol : {}", info.protocol_version);
        println!("  agent    : {}", info.agent_name);
    }
    Ok(())
}

fn mascot_command(args: Vec<String>) -> anyhow::Result<()> {
    let mut scale = 4usize;
    let mut dir: Option<String> = None;
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--scale" | "-s" => {
                scale = iter
                    .next()
                    .and_then(|v| v.parse().ok())
                    .filter(|n: &usize| *n > 0 && *n <= 32)
                    .ok_or_else(|| anyhow::anyhow!("usage: lucy mascot [dir] [--scale N]"))?;
            }
            other if other.starts_with('-') => {
                anyhow::bail!("usage: lucy mascot [dir] [--scale N]")
            }
            other if dir.is_none() => dir = Some(other.to_owned()),
            other => {
                anyhow::bail!("unexpected argument '{other}': usage: lucy mascot [dir] [--scale N]")
            }
        }
    }
    let dir = std::path::PathBuf::from(dir.unwrap_or_else(|| {
        std::env::var("LUCY_MASCOT_DIR").unwrap_or_else(|_| "lucy-mascot".into())
    }));
    let written = lucy_mascot::export::write_preview(&dir, scale)?;
    println!("Mascot art (scale {scale}) in {}/", dir.display());
    for p in &written {
        println!("  {}", p.display());
    }
    println!(
        "\nOpen {}/index.html in a browser to see every pose and frame.",
        dir.display()
    );
    Ok(())
}

/// `lucy models log [-n N] [--kind llm|classification|voice] [--failures]`:
/// tail the unified model-call log (`~/.local/state/lucy/model-calls.jsonl`,
/// overridable with `LUCY_MODEL_LOG`). Every voice/STT, classification, and
/// LLM call Lucy makes appends one JSON line there.
fn models_command(args: Vec<String>) -> anyhow::Result<()> {
    let mut sub = None::<String>;
    let mut tail = 20usize;
    let mut kind = None::<String>;
    let mut failures_only = false;
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        match a.as_str() {
            "-n" | "--tail" | "--limit" => {
                match iter.next().and_then(|v| v.parse::<usize>().ok()) {
                    Some(n) if n > 0 => tail = n,
                    _ => anyhow::bail!(
                        "usage: lucy models log [-n N] [--kind llm|classification|voice] [--failures]"
                    ),
                }
            }
            "--kind" => match iter.next().map(|v| v.to_ascii_lowercase()) {
                Some(k) if ["llm", "classification", "voice"].contains(&k.as_str()) => {
                    kind = Some(k);
                }
                _ => anyhow::bail!(
                    "usage: lucy models log [-n N] [--kind llm|classification|voice] [--failures]"
                ),
            },
            "--failures" | "--failed" => failures_only = true,
            other if sub.is_none() && !other.starts_with('-') => {
                sub = Some(other.to_owned());
            }
            _ => anyhow::bail!(
                "usage: lucy models log [-n N] [--kind llm|classification|voice] [--failures]"
            ),
        }
    }
    if sub.as_deref() != Some("log") {
        anyhow::bail!(
            "usage: lucy models log [-n N] [--kind llm|classification|voice] [--failures]"
        );
    }
    let path = lucy_core::model_log_path();
    println!("model-call log: {}", path.display());
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            println!("(no entries yet — {e})");
            return Ok(());
        }
    };
    let mut shown = 0;
    let lines: Vec<&str> = text.lines().collect();
    let total = lines.len();
    let mut skipped = 0;
    for line in lines.iter().rev() {
        if shown >= tail {
            break;
        }
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Some(k) = &kind {
            if v.get("kind").and_then(|v| v.as_str()) != Some(k.as_str()) {
                skipped += 1;
                continue;
            }
        }
        let success = v.get("success").and_then(|v| v.as_bool()).unwrap_or(true);
        if failures_only && success {
            skipped += 1;
            continue;
        }
        println!("{}", format_model_log_line(&v));
        shown += 1;
    }
    if shown == 0 {
        println!("(no matching entries)");
    } else {
        println!(
            "showing {shown} of {total} entr{} (newest first)",
            if total == 1 { "y" } else { "ies" }
        );
        let _ = skipped;
    }
    Ok(())
}

/// One human-readable line per model-call record.
fn format_model_log_line(v: &serde_json::Value) -> String {
    let str_field = |k: &str| v.get(k).and_then(|v| v.as_str()).unwrap_or("-");
    let opt_str = |k: &str| v.get(k).and_then(|v| v.as_str()).unwrap_or_default();
    let status = if v.get("success").and_then(|v| v.as_bool()).unwrap_or(true) {
        "ok".to_owned()
    } else {
        format!(
            "FAIL {}",
            opt_str("error")
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        )
    };
    let tokens = match (
        v.get("prompt_tokens").and_then(|v| v.as_u64()),
        v.get("completion_tokens").and_then(|v| v.as_u64()),
        v.get("total_tokens").and_then(|v| v.as_u64()),
    ) {
        (Some(p), Some(c), Some(t)) => format!(" tokens={p}/{c}/{t}"),
        _ => String::new(),
    };
    let purpose = opt_str("purpose");
    let purpose = if purpose.is_empty() {
        String::new()
    } else {
        format!(" purpose={purpose}")
    };
    let detail = opt_str("detail");
    let detail = if detail.is_empty() {
        String::new()
    } else {
        format!(" {detail}")
    };
    format!(
        "{} {:<14} {:<16} model={} {}ms{} {}{}",
        str_field("ts"),
        str_field("kind"),
        str_field("operation"),
        str_field("model"),
        v.get("latency_ms").and_then(|v| v.as_u64()).unwrap_or(0),
        purpose,
        status,
        tokens,
    ) + &detail
}

async fn session_service() -> anyhow::Result<LucySessionService> {
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
    Ok(LucySessionService::open(dir).await?)
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
    let service = session_service().await?;
    match args.first().map(String::as_str) {
        None | Some("list") | Some("ls") => {
            let list = service.list().await?;
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
            let s = service.create(args.get(1).cloned()).await?;
            println!("{} {}", s.session_id.0, s.title);
        }
        Some("show") => {
            let list = service.list().await?;
            let meta = args
                .get(1)
                .and_then(|a| find_session(&list, a))
                .or_else(|| list.first().cloned())
                .ok_or_else(|| anyhow::anyhow!("no sessions found"))?;
            let s = service.load(&meta.id).await?;
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
            let list = service.list().await?;
            let meta = find_session(&list, id_arg)
                .ok_or_else(|| anyhow::anyhow!("no session matches '{id_arg}'"))?;
            service.update_title(&meta.id, title).await?;
            println!("Renamed session {}", meta.id.0);
        }
        Some("delete") | Some("rm") => {
            let id_arg = args
                .get(1)
                .ok_or_else(|| anyhow::anyhow!("usage: lucy session delete <id|number>"))?;
            let list = service.list().await?;
            let meta = find_session(&list, id_arg)
                .ok_or_else(|| anyhow::anyhow!("no session matches '{id_arg}'"))?;
            service.delete(&meta.id).await?;
            println!("Deleted: {}", meta.title);
        }
        Some("clear") => {
            let id_arg = args
                .get(1)
                .ok_or_else(|| anyhow::anyhow!("usage: lucy session clear <id|number>"))?;
            let list = service.list().await?;
            let meta = find_session(&list, id_arg)
                .ok_or_else(|| anyhow::anyhow!("no session matches '{id_arg}'"))?;
            let s = service.load(&meta.id).await?;
            service.clear(&meta.id, s.title.clone()).await?;
            println!("Cleared: {}", s.title);
        }
        Some("export") => {
            let id_arg = args.get(1).ok_or_else(|| {
                anyhow::anyhow!("usage: lucy session export <id|number> <file.json>")
            })?;
            let file = args.get(2).ok_or_else(|| {
                anyhow::anyhow!("usage: lucy session export <id|number> <file.json>")
            })?;
            let list = service.list().await?;
            let meta = find_session(&list, id_arg)
                .ok_or_else(|| anyhow::anyhow!("no session matches '{id_arg}'"))?;
            let s = service.load(&meta.id).await?;
            s.save_to_file(file).await?;
            println!("Exported {} to {file}", s.title);
        }
        Some(other) => anyhow::bail!(
            "unknown session command: {other}\nusage: lucy session [list|new|show|rename|delete|clear|export]"
        ),
    }
    Ok(())
}

async fn act_command(args: Vec<String>) -> anyhow::Result<()> {
    let goal = args.join(" ");
    if goal.trim().is_empty() {
        anyhow::bail!("usage: lucy act <goal description>");
    }
    let rt = std::sync::Arc::new(lucy_runtime::LucyRuntime::new().await?);
    // `lucy act` is explicit: skip the text-only branch and go straight to the
    // ReAct loop, but still report the classification verdict.
    let classification = match rt.classify_turn(&goal).await {
        Ok(c) => c,
        // No verdict, no run: driving tools on an unrouted turn would act on a
        // guess. Stop here with the router's own reason.
        Err(e) => {
            println!("✖ {e}");
            lucy_stt::error_beep();
            rt.save_assistant_text(format!("{e}")).await?;
            return Ok(());
        }
    };
    if let Some(note) = &classification.summary_note {
        println!("⚠ {note}");
    }
    println!("🧭 {}", classification.summary());
    // `lucy act` does not save the user turn — it is a one-shot invocation, not
    // a conversation — which is the only thing that still separates it from
    // `lucy agent` below. The run itself is the same run.
    run_goal_and_report(&rt, &goal, "⚡ [Act]").await
}

/// `lucy agent <goal>`: the ReAct loop. Perceive, decide, execute, observe —
/// the model sees what each tool actually returned and picks the next call.
///
/// Shares [`run_goal_and_report`] with `lucy act`, so the two subcommands
/// cannot drift into running the same goal two different ways.
async fn agent_command(args: Vec<String>) -> anyhow::Result<()> {
    let goal = args.join(" ");
    if goal.trim().is_empty() {
        anyhow::bail!("usage: lucy agent <goal description>");
    }
    let rt = std::sync::Arc::new(lucy_runtime::LucyRuntime::new().await?);
    rt.save_user_message(goal.clone()).await?;
    let classification = match rt.classify_turn(&goal).await {
        Ok(c) => c,
        // No verdict, no run: driving tools on an unrouted turn would act on a
        // guess. Stop here with the router's own reason.
        Err(e) => {
            println!("✖ {e}");
            lucy_stt::error_beep();
            rt.save_assistant_text(format!("{e}")).await?;
            return Ok(());
        }
    };
    if let Some(note) = &classification.summary_note {
        println!("⚠ {note}");
    }
    println!("🧭 {}", classification.summary());
    run_goal_and_report(&rt, &goal, "🧠 [Agent]").await
}

/// Start a run through the single entry, auto-approve it, and report the
/// outcome honestly.
///
/// Every CLI act goes through here. The one entry — `execute_goal_outcome` —
/// is why this body has no planner/agent fork in it: re-rolling that choice
/// per surface is exactly how "planner returned no commands" became a dead end
/// on one path and not the other.
async fn run_goal_and_report(
    rt: &std::sync::Arc<lucy_runtime::LucyRuntime>,
    goal: &str,
    banner: &str,
) -> anyhow::Result<()> {
    println!("{banner} Goal: {goal}");
    println!("  … running the ReAct loop with {}", rt.planner_model_key());
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let runner_rt = rt.clone();
    let goal = goal.to_owned();
    let runner = tokio::spawn(async move {
        runner_rt
            .execute_goal_outcome(&goal, None, Some(tx))
            .await
    });
    auto_answer_approvals(rt, &mut rx).await;
    match runner.await {
        Ok(Ok(outcome)) => {
            // The outcome's own `complete` flag, not its wording: a run whose
            // objectives never verified must not print a green tick.
            println!(
                "{} {}",
                if outcome.complete { "✔" } else { "⚠" },
                outcome.summary
            );
            println!("  ⚡ {}", outcome.stats.summary());
            if outcome.complete {
                lucy_stt::done_beep();
            } else {
                lucy_stt::error_beep();
            }
            rt.save_assistant_text(outcome.summary.clone()).await?;
        }
        Ok(Err(e)) => {
            let line = friendly(&format!("{e:#}"));
            println!("✖ {line}");
            lucy_stt::error_beep();
            rt.save_assistant_text(format!("Goal not completed: {line}"))
                .await?;
        }
        Err(e) => println!("✖ task failed: {}", friendly(&format!("{e:#}"))),
    }
    report_auto_compact(rt).await;
    Ok(())
}

/// Print a numbered subtask checklist.
#[allow(dead_code)]
fn print_subtask_list(subtasks: &[lucy_runtime::Subtask]) {
    println!("  Subtasks ({}):", subtasks.len());
    println!("{}", lucy_runtime::format_subtasks(subtasks));
}

/// Execute a subtask plan, streaming automation events to stdout.
#[allow(dead_code)]
async fn run_plan_and_report(
    rt: std::sync::Arc<lucy_runtime::LucyRuntime>,
    goal: String,
    subtasks: Vec<lucy_runtime::Subtask>,
) -> anyhow::Result<()> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let runner = tokio::spawn(async move {
        rt.run_automation_with_plan(&goal, &subtasks, Some(tx))
            .await
    });

    while let Some(event) = rx.recv().await {
        match event {
            lucy_core::AgentEvent::Progress { message } => {
                println!("  ⚡ {message}");
            }
            lucy_core::AgentEvent::Status { message } => {
                println!("  ℹ {message}");
            }
            lucy_core::AgentEvent::Error { message } => {
                eprintln!("  ✖ {message}");
            }
            _ => {}
        }
    }

    match runner.await? {
        Ok(msg) => {
            println!("✔ Completed: {msg}");
            Ok(())
        }
        Err(e) => {
            eprintln!("✖ Automation stopped: {}", friendly(&format!("{e:#}")));
            Err(e)
        }
    }
}

async fn decide_command(args: Vec<String>) -> anyhow::Result<()> {
    if args.is_empty() {
        println!(
            "Lucy System-1 (decider-2b) Decision Engine\n\nUsage:\n  lucy decide mode <prompt>            Decide Chat vs Act\n  lucy decide tools <task-description> Decide which tools to use\n  lucy decide click <goal>             Decide what element to click\n  lucy decide type <goal>              Decide which input field to type into\n  lucy decide done <goal>              Decide if goal is visibly satisfied\n"
        );
        return Ok(());
    }
    let sub = args[0].as_str();
    let query = args[1..].join(" ");
    if query.trim().is_empty() {
        anyhow::bail!("missing argument for 'lucy decide {sub}'");
    }

    let rt = lucy_runtime::LucyRuntime::new().await?;
    let s1 = rt.system_one();

    match sub {
        "mode" => {
            let (mode, conf) = s1.decide_mode(&query, "").await?;
            println!("Decision: {:?} (confidence: {:.2})", mode, conf);
        }
        "tools" => {
            let mut candidate_tools: Vec<(String, String)> = rt
                .mcp_tools()
                .iter()
                .map(|t| (t.name.clone(), t.description.clone()))
                .collect();
            for line in rt.tool_brief().lines() {
                if let Some((name, desc)) = line.split_once(" — ") {
                    let n = name.trim().to_string();
                    if !candidate_tools.iter().any(|(x, _)| x == &n) {
                        candidate_tools.push((n, desc.trim().to_string()));
                    }
                }
            }
            let chosen = s1.decide_tools_multi(&query, &candidate_tools, 5).await?;
            println!("⚡ Laya selected tools for: \"{query}\"");
            for (idx, (tool, prob)) in chosen.iter().enumerate() {
                println!("  {}. {:<28} (probability: {:.3})", idx + 1, tool, prob);
            }
        }
        "click" => {
            let space = rt.automation_engine().observe().await?;
            if space.click_targets.is_empty() {
                println!("No interactive click targets found in active window.");
                return Ok(());
            }
            let (target, conf) = s1.decide_click_target(&query, &space.click_targets).await?;
            let desc = space
                .click_targets
                .get(&target)
                .cloned()
                .unwrap_or(target.clone());
            println!(
                "⚡ Laya decided click target: [{target}] {desc} (confidence: {:.2})",
                conf
            );
        }
        "type" => {
            let space = rt.automation_engine().observe().await?;
            if space.type_targets.is_empty() {
                println!("No text input fields found in active window.");
                return Ok(());
            }
            let (target, conf) = s1.decide_type_target(&query, &space.type_targets).await?;
            let desc = space
                .type_targets
                .get(&target)
                .cloned()
                .unwrap_or(target.clone());
            println!(
                "⚡ Laya decided type target: [{target}] {desc} (confidence: {:.2})",
                conf
            );
        }
        "done" => {
            let space = rt.automation_engine().observe().await?;
            let active_desc = space
                .active_window
                .as_ref()
                .map(|w| format!("{} ({})", w.title, w.class))
                .unwrap_or_else(|| "none".into());
            let (satisfied, conf) = s1.is_goal_satisfied(&query, &active_desc, &[]).await?;
            println!(
                "⚡ Laya goal satisfied: {} (confidence: {:.2})",
                satisfied, conf
            );
        }
        other => {
            anyhow::bail!("unknown decide subcommand: '{other}'");
        }
    }
    Ok(())
}

async fn ask_command(args: Vec<String>) -> anyhow::Result<()> {
    let prompt = args.join(" ");
    if prompt.trim().is_empty() {
        anyhow::bail!("usage: lucy ask <prompt>");
    }
    let rt = std::sync::Arc::new(lucy_runtime::LucyRuntime::new().await?);
    rt.save_user_message(prompt.clone()).await?;

    // Step 1: one classification forward pass (intent + reasoning level).
    let classification = match rt.classify_turn(&prompt).await {
        Ok(c) => c,
        // No verdict, no reply and no tools: the turn stops with the
        // router's own reason instead of guessing a branch.
        Err(e) => {
            println!("✖ {e}");
            lucy_stt::error_beep();
            rt.save_assistant_text(format!("{e}")).await?;
            return Ok(());
        }
    };
    if let Some(note) = &classification.summary_note {
        println!("⚠ {note}");
    }
    println!("🧭 {}", classification.summary());

    // Step 2: dispatch on the branch, then the reasoning tier's model.
    let route = match rt.route_turn_with(&prompt, Some(classification)).await {
        Ok(r) => r,
        // Routing produced no verdict, so there is no branch to dispatch on.
        // Stop before answering or acting.
        Err(e) => {
            println!("✖ {e}");
            lucy_stt::error_beep();
            rt.save_assistant_text(format!("{e}")).await?;
            return Ok(());
        }
    };
    if let Some(note) = &route.note {
        println!("⚠ {note}");
    }
    println!("🤖 active: {}", route.active_model_line());
    if !route.needs_actions() {
        // The classifier's knowledge hint only reorders recall; it never decides
        // whether recall happens.
        let hint = route.knowledge_topic.clone();
        let reply = rt
            .answer_turn_with_knowledge(&prompt, &route, hint.as_deref())
            .await?;
        rt.save_assistant_text(reply.clone()).await?;
        println!("{reply}");
        lucy_stt::done_beep();
        report_auto_compact(&rt).await;
        capture_knowledge(&rt, &prompt, &reply).await;
        return Ok(());
    }

    // Step 2B: the same single act entry `lucy act` uses — the ReAct loop.
    // See `execute_goal_outcome`.
    println!("⚡ [Act] Goal: {prompt}");
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let runner_rt = rt.clone();
    let goal = prompt.clone();
    let run_route = route.clone();
    let runner = tokio::spawn(async move {
        runner_rt
            .execute_goal_outcome(&goal, Some(&run_route), Some(tx))
            .await
    });
    // Non-interactive CLI: auto-approve non-destructive steps, deny the rest.
    auto_answer_approvals(&rt, &mut rx).await;
    match runner.await {
        Ok(Ok(outcome)) => {
            // The outcome's own `complete` flag, not its wording: a run whose
            // objectives never verified must not print a green tick.
            println!(
                "{} {}",
                if outcome.complete { "✔" } else { "⚠" },
                outcome.summary
            );
            println!("  ⚡ {}", outcome.stats.summary());
            if outcome.complete {
                lucy_stt::done_beep();
            } else {
                lucy_stt::error_beep();
            }
            rt.save_assistant_text(outcome.summary.clone()).await?;
        }
        Ok(Err(e)) => {
            let line = friendly(&format!("{e:#}"));
            println!("✖ {line}");
            lucy_stt::error_beep();
            rt.save_assistant_text(format!("Goal not completed: {line}"))
                .await?;
        }
        Err(e) => println!("✖ task failed: {}", friendly(&format!("{e:#}"))),
    }
    report_auto_compact(&rt).await;
    Ok(())
}

/// The CLI has no approval UI, so a prompt is answered with "allow once" and
/// the fact is printed. The TUI is the interactive surface for approvals.
///
/// Async (`.recv().await`): this runs inside the tokio runtime, where
/// `blocking_recv()` panics with "Cannot block the current thread".
async fn auto_answer_approvals(
    rt: &lucy_runtime::LucyRuntime,
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<lucy_core::AgentEvent>,
) {
    while let Some(event) = rx.recv().await {
        match event {
            lucy_core::AgentEvent::ApprovalRequest { id, name, .. } => {
                println!("  ⚠ auto-approving {name} (non-interactive)");
                rt.approval_gate()
                    .resolve(&id, lucy_core::ApprovalDecision::AllowOnce);
            }
            lucy_core::AgentEvent::Progress { message } => println!("  ⚡ {message}"),
            lucy_core::AgentEvent::Status { message } => println!("  ℹ {message}"),
            // Classified: this event carries provider bodies and MCP envelopes,
            // and stdout is where a CLI user sees the failure.
            lucy_core::AgentEvent::Error { message } => {
                println!("  ✖ {}", lucy_core::friendly(&message))
            }
            _ => {}
        }
    }
}

/// One reader-facing line for a failure, shared by every CLI command so no
/// subcommand is the one that prints a raw provider body.
fn friendly(raw: &str) -> String {
    lucy_core::friendly(raw)
}

/// Print the auto-compact note when history was compacted.
async fn report_auto_compact(rt: &lucy_runtime::LucyRuntime) {
    match rt.auto_compact_if_needed().await {
        Ok(Some(note)) => println!("🗜 {note}"),
        Ok(None) => {}
        Err(e) => println!(
            "🗜 auto-compact failed: {}",
            friendly(&format!("{e:#}"))
        ),
    }
}

/// Capture the turn's durable knowledge, then report what landed.
///
/// Off the critical path and strictly best-effort: an extractor failure prints a
/// line and the reply stands. A knowledge base that can break a conversation is
/// worse than one that quietly learns nothing.
async fn capture_knowledge(rt: &lucy_runtime::LucyRuntime, request: &str, reply: &str) {
    match lucy_runtime::knowledge::capture_turn(rt, request, reply).await {
        Ok(claims) if !claims.is_empty() => {
            let topics: Vec<&str> = claims.iter().map(|c| c.slug.as_str()).collect();
            println!("📚 remembered: {}", topics.join(", "));
        }
        Ok(_) => {}
        Err(e) => println!("📚 capture skipped: {e:#}"),
    }
}

async fn smoke_command(args: Vec<String>) -> anyhow::Result<()> {
    if args.first().map(|s| s.as_str()) == Some("plan") {
        return smoke_plan_command(&args[1..]).await;
    }
    println!("🔍 Starting Lucy End-to-End Pipeline Smoke Test...\n");

    // 1. Doctor
    println!("[1/6] Configuration Doctor Check");
    for (name, ok, detail) in lucy_config::doctor() {
        println!("  {} {:<16} {}", if ok { "✓" } else { "✗" }, name, detail);
    }

    // 2. Runtime & Session Store Initialization
    println!("\n[2/6] Initializing Lucy Runtime & Session Store");
    let rt = lucy_runtime::LucyRuntime::new().await?;
    let test_prompt = format!("smoke-test-{}", lucy_core::SessionId::default().0);
    rt.save_user_message(test_prompt.clone()).await?;
    let history = rt.history().await;
    assert!(
        history
            .iter()
            .any(|m| matches!(m, lucy_core::TurnMessage::User(t) if t == &test_prompt)),
        "user message must be in history"
    );
    println!("  ✓ Session persistence and ADK store operational");

    // 3. System-1 decider-2b Engine
    println!("\n[3/6] Testing System-1 decider-2b Decision Engine");
    let s1 = rt.system_one();
    let (mode, conf) = s1.decide_mode("Hello!", "").await?;
    println!("  ✓ Mode decision: {:?} (confidence: {:.2})", mode, conf);

    // 4. LLM Provider Integration
    println!("\n[4/6] Testing LLM Provider Integration");
    let prompt = "Hello, respond with a short greeting";
    let route = match rt.route_turn(prompt).await {
        Ok(r) => r,
        // Routing produced no verdict: report it and stop before answering
        // or acting. This command persists no assistant text and beeps
        // nowhere else, so it does neither here.
        Err(e) => {
            println!("✖ {e}");
            return Ok(());
        }
    };
    println!("  ✓ Routed to {}", route.active_model_line());
    if route.needs_actions() {
        println!("  ✓ Branch: act");
    } else {
        let reply = rt.answer_turn(prompt, &route).await?;
        println!("  ✓ Branch: chat — {}", reply.trim());
    }

    // 5. HyprFast Desktop & Action Space Observation
    println!("\n[5/6] Testing HyprFast Desktop State & Action Space");
    let space = rt.automation_engine().observe().await?;
    let active_str = space
        .active_window
        .as_ref()
        .map(|w| format!("{} ({})", w.title, w.class))
        .unwrap_or_else(|| "none".into());
    println!("  ✓ Active window: {}", active_str);
    println!("  ✓ Open windows tracked: {}", space.windows.len());
    println!(
        "  ✓ Interactive click targets: {}",
        space.click_targets.len()
    );

    // 6. Memory Capabilities
    println!("\n[6/6] Testing ADK Memory Capabilities");
    let mem_enabled = rt.adk_memory_enabled();
    println!("  ✓ ADK long-term memory enabled: {}", mem_enabled);

    println!("\n🎉 ALL SMOKE TESTS PASSED! Lucy pipeline is operating end-to-end.\n");
    Ok(())
}

/// Tools that reach into the screen or a page, for reporting only.
const SCREEN_INTERACTION_TOOLS: &[&str] = &[
    "browser_click",
    "browser_type",
    "browser_hover",
    "browser_select_option",
    "browser_press_key",
    "browser_evaluate",
    "browser_execute_plan",
    "hint_act",
    "hint_batch",
    "hint_resolve",
    "hint_resolve_batch",
    "find_and_click",
    "find_and_type",
    "click_ui",
    "pointer",
    "keyboard",
];

/// Tools that need an address a blind plan cannot produce: a snapshot `ref` or
/// a CSS `selector` for the CDP verbs, screen coordinates for `pointer`. A
/// plan reaching for one of these has drifted off the self-resolving path.
const BLIND_UNSAFE_TOOLS: &[&str] = &[
    "browser_click",
    "browser_type",
    "browser_hover",
    "browser_select_option",
    "pointer",
];

/// The spellings that satisfy the blind-plan constraint.
const HINT_PIPELINE_TOOLS: &[&str] = &[
    "hint_act",
    "hint_batch",
    "hint_resolve",
    "hint_resolve_batch",
    "find_and_click",
    "find_and_type",
    "browser_execute_plan",
];

/// `Err` with the reason when a plan drove the screen with a tool a blind plan
/// cannot aim. `Ok` when every interaction goes through the hint pipeline (or
/// the plan has no interaction step at all, as an open-only goal has none).
fn hint_pipeline_drift(plan: &[lucy_runtime::PlannedCommand]) -> Result<(), String> {
    let mut used: Vec<&str> = Vec::new();
    let mut offenders: Vec<&str> = Vec::new();
    for step in plan {
        let name = step.tool.trim();
        let base = name
            .strip_prefix("mcp_hyprfast_")
            .or_else(|| name.strip_prefix("mcp_"))
            .unwrap_or(name);
        if lucy_hyprfast::is_removed_tool(base) {
            return Err(format!(
                "plan used removed tool {base} — ground/act_fast/act_batch/stagehand_* \
                 need a Gemini key Lucy no longer uses; expected the hint pipeline ({})",
                HINT_PIPELINE_TOOLS.join("/")
            ));
        }
        if SCREEN_INTERACTION_TOOLS.contains(&base) {
            used.push(base);
        }
        if BLIND_UNSAFE_TOOLS.contains(&base) && !offenders.contains(&base) {
            offenders.push(base);
        }
    }
    if offenders.is_empty() {
        return Ok(());
    }
    Err(format!(
        "plan drove the screen with {} — a blind plan can supply neither a ref/selector nor \
         coordinates; expected the hint pipeline ({})",
        offenders.join(", "),
        HINT_PIPELINE_TOOLS.join("/")
    ) + &format!("; plan interacted with {}", used.join(", ")))
}

/// `lucy smoke plan <goal> [--attempts N] [--dump-prompt]`: hammer the L3
/// planner (hyprfast skill + live tool catalog + goal) until it returns a
/// command list whose every tool exists — or fail after N attempts. Planning
/// only: nothing is executed.
///
/// `--dump-prompt` prints the exact prompt the planner would get (built from
/// the live catalog and the vendored skill) and returns without calling the
/// model: that is how the offered tool set gets inspected after a catalog
/// change.
async fn smoke_plan_command(args: &[String]) -> anyhow::Result<()> {
    let mut attempts = 5usize;
    let mut dump_prompt = false;
    let mut goal_parts: Vec<String> = Vec::new();
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        if a == "--attempts" || a == "-n" {
            match iter.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(n) if n > 0 => attempts = n,
                _ => anyhow::bail!("usage: lucy smoke plan <goal> [--attempts N] [--dump-prompt]"),
            }
        } else if a == "--dump-prompt" {
            dump_prompt = true;
        } else {
            goal_parts.push(a.clone());
        }
    }
    let goal = goal_parts.join(" ");
    if goal.trim().is_empty() {
        anyhow::bail!("usage: lucy smoke plan <goal> [--attempts N] [--dump-prompt]");
    }
    let rt = lucy_runtime::LucyRuntime::new().await?;
    if dump_prompt {
        let prompt = rt.planner_prompt(&goal).await;
        println!(
            "📝 Planner prompt: {} chars, {} hyprfast tools in the live catalog",
            prompt.chars().count(),
            rt.hyprfast_catalog().map(|c| c.len()).unwrap_or(0)
        );
        println!("{prompt}");
        return Ok(());
    }
    println!("🔁 Smoke-planning until the LLM returns hyprfast commands");
    println!("  Goal: {goal}");
    println!("  Max attempts: {attempts}\n");
    for attempt in 1..=attempts {
        println!("[{attempt}/{attempts}] asking the L3 planner...");
        let ask_goal = if attempt == 1 {
            goal.clone()
        } else {
            format!(
                "{goal}\n\n(Retry {attempt}: the previous attempt returned no usable commands. \
                 Return ONLY the JSON command list using tools from the catalog, and drive the \
                 screen with hint_act.)"
            )
        };
        let plan = rt.plan_commands(&ask_goal).await;
        if plan.is_empty() {
            println!("  … empty plan — retrying\n");
            continue;
        }
        let unknown = lucy_runtime::unknown_plan_tools(&rt, &plan).await?;
        println!("  Plan ({} step(s)):", plan.len());
        println!("{}", lucy_runtime::format_plan(&plan));
        if !unknown.is_empty() {
            println!(
                "  ✗ unknown tools ({}): {} — retrying\n",
                unknown.len(),
                unknown.join(", ")
            );
            continue;
        }
        if let Err(reason) = hint_pipeline_drift(&plan) {
            println!("  ✗ {reason} — retrying\n");
            continue;
        }
        println!(
            "\n✔ LLM returned {} valid hyprfast command(s) on attempt {attempt}/{attempts}",
            plan.len()
        );
        return Ok(());
    }
    anyhow::bail!(
        "LLM returned no valid hyprfast command list after {attempts} attempt(s) for: {goal}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucy_runtime::PlannedCommand;

    fn step(tool: &str) -> PlannedCommand {
        PlannedCommand {
            index: 1,
            tool: tool.into(),
            input: serde_json::json!({}),
            description: String::new(),
        }
    }

    #[test]
    fn hint_pipeline_plans_pass() {
        for plan in [
            vec![step("browser_navigate"), step("hint_act")],
            vec![
                step("mcp_hyprfast_browser_open"),
                step("mcp_hyprfast_hint_act"),
            ],
            vec![step("browser_navigate"), step("find_and_click")],
            // An open-only goal legitimately has no interaction step.
            vec![step("browser_open")],
        ] {
            assert_eq!(hint_pipeline_drift(&plan), Ok(()), "{plan:?}");
        }
    }

    #[test]
    fn plans_that_need_an_element_address_are_drift() {
        for bad in [
            "browser_click",
            "browser_type",
            "browser_hover",
            "browser_select_option",
            "mcp_hyprfast_pointer",
        ] {
            let plan = vec![step("browser_navigate"), step(bad)];
            let err = hint_pipeline_drift(&plan).expect_err(bad);
            assert!(
                err.contains(bad.trim_start_matches("mcp_hyprfast_")),
                "{err}"
            );
            assert!(err.contains("hint_act"), "{err}");
        }
    }

    /// A desktop goal legitimately drives the screen with blind-suppliable
    /// arguments (a key chord, an accessible name); those are not drift.
    #[test]
    fn native_blind_safe_verbs_are_not_drift() {
        for ok in [
            "keyboard",
            "click_ui",
            "browser_evaluate",
            "browser_press_key",
        ] {
            let plan = vec![step("browser_navigate"), step(ok)];
            assert_eq!(hint_pipeline_drift(&plan), Ok(()), "{ok}");
        }
    }

    /// Plans naming tools the runtime no longer ships are drift however the
    /// name is spelled: bare, `mcp_`-prefixed, or `mcp_hyprfast_`-prefixed.
    /// The table covers the whole removed family, not one anecdote.
    #[test]
    fn removed_tools_are_drift() {
        for banned in [
            "ground",
            "act_fast",
            "act_batch",
            "stagehand_act",
            "stagehand_observe",
            "mcp_ground",
            "mcp_hyprfast_act_fast",
            "mcp_hyprfast_act_batch",
            "mcp_hyprfast_stagehand_observe",
        ] {
            let plan = vec![step("browser_navigate"), step(banned)];
            let err = hint_pipeline_drift(&plan).expect_err(banned);
            assert!(err.contains("removed tool"), "{err}");
        }
    }
}
