mod settings;

use std::collections::VecDeque;
use std::io::{self, Stdout};
use std::sync::Arc;
use std::time::{Duration, Instant};
use crossterm::{event::{self,Event,KeyCode,KeyEvent,KeyEventKind,KeyModifiers},execute,terminal::{disable_raw_mode,enable_raw_mode,EnterAlternateScreen,LeaveAlternateScreen}};
use lucy_config::LucyConfig;
use lucy_core::{AgentEvent, ApprovalDecision, SessionMeta, TokenUsage, TurnMessage};
use lucy_runtime::LucyRuntime;
use lucy_stt::{GroqStt, HoldCapture};
use ratatui::{backend::CrosstermBackend,layout::{Alignment,Constraint,Direction,Layout,Rect},style::{Color,Modifier,Style},text::{Line,Span,Text},widgets::{Block,Borders,Clear,List,ListItem,Paragraph,Wrap},Terminal};
use tokio::sync::mpsc;

const MASCOT:[&str;9]=["      ╭────────────────────╮      ","      │                    │      ","      │    ███      ███    │      ","      │    ███      ███    │      ","      │                    │      ","      │       ╭────╮       │      ","      │      ╰──────╯      │      ","      │                    │      ","      ╰────────────────────╯      "];
const MAX_MSGS: usize = 800;
/// Persistent work/progress notes ("Plan ready: 3 steps", "Step 1: …") —
/// kept in the transcript so the user sees what happened, not just the
/// transient spinner row.
const MAX_PROGRESS: usize = 8;

// ---- chat model (human/agent separation + streaming) ----
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsgKind { User, Lucy, System, Tool }

#[derive(Debug, Clone)]
pub struct ChatMsg { kind: MsgKind, text: String }

impl ChatMsg {
    fn user(t: String) -> Self { Self { kind: MsgKind::User, text: t } }
    fn lucy(t: String) -> Self { Self { kind: MsgKind::Lucy, text: t } }
    fn system(t: String) -> Self { Self { kind: MsgKind::System, text: t } }
    fn tool(t: String) -> Self { Self { kind: MsgKind::Tool, text: t } }
}

#[derive(Debug, Clone)]
pub struct ApprovalDialog {
    pub id: String,
    pub name: String,
    pub input: String,
}

pub struct App {
    pub status: String,
    pub listening: bool,
    pub input: String,
    pub settings: bool,
    pub settings_selected: usize,
    pub config: LucyConfig,
    pub history: Vec<String>,
    pub hist_idx: Option<usize>,
    pub hist_draft: String,
    pub cursor: usize,
    /// Offset from bottom in *visual* (wrapped) rows. 0 = pinned to newest.
    pub scroll: usize,
    pub model_label: String,
    pub usage: TokenUsage,
    pub approval: Option<ApprovalDialog>,
    pub messages: Vec<ChatMsg>,
    pub streaming: String,
    pub tool_active: Option<String>,
    /// Live background-activity feedback (thinking / working / writing …).
    /// Rendered as an animated spinner row so the user always knows work
    /// is in flight — even before the first token arrives.
    pub busy: bool,
    pub phase: String,
    pub phase_detail: String,
    pub phase_since: Option<Instant>,
    pub progress: VecDeque<String>,
    pub session_title: String,
    pub session_id_short: String,
    pub session_count: usize,
    pub show_sessions: bool,
    pub sessions: Vec<SessionMeta>,
    pub sess_selected: usize,
    pub show_help: bool,
}

impl App {
    fn new(config: LucyConfig) -> Self {
        Self {
            status: "Ready".into(),
            listening: false,
            input: String::new(),
            settings: false,
            settings_selected: 0,
            config,
            history: Vec::new(),
            hist_idx: None,
            hist_draft: String::new(),
            cursor: 0,
            scroll: 0,
            model_label: String::new(),
            usage: TokenUsage::default(),
            approval: None,
            messages: Vec::new(),
            streaming: String::new(),
            tool_active: None,
            busy: false,
            phase: String::new(),
            phase_detail: String::new(),
            phase_since: None,
            progress: VecDeque::new(),
            session_title: "untitled".into(),
            session_id_short: String::new(),
            session_count: 0,
            show_sessions: false,
            sessions: Vec::new(),
            sess_selected: 0,
            show_help: false,
        }
    }

    fn is_pinned(&self) -> bool { self.scroll == 0 }

    fn pin(&mut self) { self.scroll = 0; }

    /// Keep the viewport stable when new rows arrive while scrolled up.
    fn grow(&mut self, added_visual_rows: usize) {
        if !self.is_pinned() {
            self.scroll = self.scroll.saturating_add(added_visual_rows);
        }
    }

    fn push_msg(&mut self, msg: ChatMsg) {
        // Rough visual-row estimate for scroll stability (refined at draw time).
        let rows = msg.text.split('\n').count().max(1).saturating_add(1);
        self.grow(rows);
        self.messages.push(msg);
        if self.messages.len() > MAX_MSGS {
            let excess = self.messages.len() - MAX_MSGS;
            self.messages.drain(0..excess);
        }
    }

    fn append_stream(&mut self, delta: &str) {
        if self.streaming.is_empty() && self.is_pinned() {
            // stay pinned
        } else if !self.is_pinned() {
            self.grow(delta.split('\n').count().max(1));
        }
        self.streaming.push_str(delta);
    }

    fn flush_stream(&mut self) {
        let text = self.streaming.trim().to_owned();
        self.streaming.clear();
        self.tool_active = None;
        if !text.is_empty() {
            self.push_msg(ChatMsg::lucy(text));
        }
    }

    /// Mark background work as in-flight with a human phase label
    /// ("Thinking", "Writing", "Using bash", …) plus optional detail.
    /// Called the moment the user hits Enter so there is never a dead gap.
    fn set_phase(&mut self, phase: &str, detail: &str) {
        self.busy = true;
        self.phase = phase.to_owned();
        if !detail.trim().is_empty() {
            self.phase_detail = detail.trim().to_owned();
        }
        if self.phase_since.is_none() {
            self.phase_since = Some(Instant::now());
        }
        self.pin();
        self.status = if self.phase_detail.is_empty() {
            format!("{phase}…")
        } else {
            format!("{phase}… — {}", truncate_one_line(&self.phase_detail, 80))
        };
    }

    fn mark_idle(&mut self) {
        self.busy = false;
        self.phase.clear();
        self.phase_detail.clear();
        self.phase_since = None;
    }

    /// Persistent progress note from the hierarchical loop ("Plan ready: 3
    /// steps", "Step 2: open the browser", "Done: …"). Stays in the
    /// transcript; the newest note also drives the spinner + header status.
    fn push_progress(&mut self, msg: &str) {
        let m = msg.trim().to_owned();
        if m.is_empty() {
            return;
        }
        if self.progress.back().is_some_and(|l| l == &m) {
            return;
        }
        self.grow(1);
        self.progress.push_back(m.clone());
        while self.progress.len() > MAX_PROGRESS {
            self.progress.pop_front();
        }
        self.busy = true;
        if self.phase.trim().is_empty() {
            self.phase = "Working".to_owned();
        }
        self.phase_detail = m.clone();
        if self.phase_since.is_none() {
            self.phase_since = Some(Instant::now());
        }
        self.pin();
        self.status = format!("{}… — {}", self.phase, truncate_one_line(&m, 80));
    }

    fn busy_elapsed_ms(&self) -> u128 {
        self.phase_since.map(|t| t.elapsed().as_millis()).unwrap_or(0)
    }

    fn load_turns(&mut self, turns: &[TurnMessage]) {
        self.messages.clear();
        self.streaming.clear();
        self.tool_active = None;
        self.progress.clear();
        self.mark_idle();
        for m in turns {
            match m {
                TurnMessage::User(t) => self.messages.push(ChatMsg::user(t.clone())),
                TurnMessage::Assistant(turn) => {
                    if let Some(t) = turn.text.as_deref() {
                        if !t.trim().is_empty() {
                            self.messages.push(ChatMsg::lucy(t.to_owned()));
                        }
                    }
                    for c in &turn.tool_calls {
                        self.messages.push(ChatMsg::tool(format!("{} {}", c.name, truncate_one_line(&c.input.to_string(), 120))));
                    }
                }
                TurnMessage::Tool(res) => {
                    let preview = truncate_one_line(&res.output.to_string(), 160);
                    self.messages.push(ChatMsg::tool(format!("{} → {}", res.name, preview)));
                }
            }
        }
        // Cap on load as well.
        if self.messages.len() > MAX_MSGS {
            let excess = self.messages.len() - MAX_MSGS;
            self.messages.drain(0..excess);
        }
        self.pin();
    }
}

fn truncate_one_line(s: &str, max: usize) -> String {
    let one: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if one.chars().count() > max {
        let mut t: String = one.chars().take(max).collect();
        t.push('…');
        t
    } else {
        one
    }
}

fn char_byte_idx(s:&str,char_idx:usize)->usize{
    if char_idx==0{ return 0; }
    s.char_indices().nth(char_idx).map(|(b,_)|b).unwrap_or(s.len())
}
fn truncate_model_label(s:&str)->String{
    if s.chars().count()>16{ s.chars().take(16).collect() } else { s.to_owned() }
}

fn format_tokens(n:u64)->String{
    if n < 1000 {
        format!("{n} tok")
    } else if n < 1_000_000 {
        let v = n as f64 / 1000.0;
        if v >= 100.0 { format!("{:.0}k", v) } else { format!("{:.1}k", v) }
    } else {
        let v = n as f64 / 1_000_000.0;
        format!("{:.1}M", v)
    }
}

fn line_home_cursor(input: &str, cursor: usize) -> usize {
    let chars: Vec<char> = input.chars().collect();
    let cur = cursor.min(chars.len());
    let mut i = cur;
    while i > 0 && chars[i - 1] != '\n' {
        i -= 1;
    }
    i
}

fn line_end_cursor(input: &str, cursor: usize) -> usize {
    let chars: Vec<char> = input.chars().collect();
    let cur = cursor.min(chars.len());
    let mut i = cur;
    while i < chars.len() && chars[i] != '\n' {
        i += 1;
    }
    i
}

fn cursor_row_col(input: &str, cursor: usize) -> (usize, usize) {
    let count = input.chars().count();
    let cur = cursor.min(count);
    let prefix: String = input.chars().take(cur).collect();
    let row = prefix.chars().filter(|&c| c == '\n').count();
    let col = prefix.rfind('\n').map(|idx| prefix[idx + 1..].chars().count()).unwrap_or(cur);
    // NB: rfind on the char-collected prefix is byte-based but '\n' is 1 byte,
    // so slicing at idx+1 is always a char boundary.
    (row, col)
}

// ---------- slash commands (opencode-style) ----------
const COMMANDS: &[(&str, &str)] = &[
    ("/new", "start a new session — /new [title]"),
    ("/sessions", "list sessions (Ctrl+O)"),
    ("/switch", "switch session — /switch <number|id>"),
    ("/rename", "rename current session — /rename <title>"),
    ("/delete", "delete a session — /delete [<number|id>]"),
    ("/fork", "fork current session (keeps history)"),
    ("/clear", "clear current session history"),
    ("/compact", "trim history — /compact [keep_n=40]"),
    ("/history", "show input history"),
    ("/status", "show session + runtime status"),
    ("/model", "show current model"),
    ("/export", "export session — /export <file.json>"),
    ("/help", "show help (F1)"),
    ("/settings", "open settings (Ctrl+,)"),
    ("/quit", "quit lucy"),
];

fn format_tool_start(name: &str, input: &serde_json::Value) -> String {
    match name {
        "shell" | "bash" => {
            if let Some(cmd) = input.get("command").and_then(|v| v.as_str()) {
                let cmd_short = if cmd.chars().count() > 60 {
                    let mut s: String = cmd.chars().take(57).collect();
                    s.push_str("...");
                    s
                } else {
                    cmd.to_string()
                };
                format!("bash: {cmd_short}")
            } else {
                format!("{name} …")
            }
        }
        "read_file" | "write_file" | "edit_file" => {
            let action = match name {
                "read_file" => "read",
                "write_file" => "write",
                "edit_file" => "edit",
                _ => name,
            };
            if let Some(p) = input.get("path").and_then(|v| v.as_str()) {
                format!("{action}: {p}")
            } else {
                format!("{name} …")
            }
        }
        "glob" | "search_files" => {
            let pat = input.get("pattern").or_else(|| input.get("query")).and_then(|v| v.as_str()).unwrap_or("");
            format!("{name}: \"{pat}\"")
        }
        "git" => {
            if let Some(args) = input.get("args").and_then(|v| v.as_array()) {
                let s: Vec<&str> = args.iter().filter_map(|v| v.as_str()).collect();
                format!("git {}", s.join(" "))
            } else {
                "git …".to_string()
            }
        }
        _ => {
            if name.starts_with("mcp_") || name.starts_with("hyprfast_") {
                let short = name.trim_start_matches("mcp_").trim_start_matches("hyprfast_");
                format!("{short} …")
            } else {
                format!("{name} …")
            }
        }
    }
}

fn format_tool_finish(name: &str, output: &serde_json::Value, is_error: bool) -> String {
    if is_error {
        let err_msg = output.get("error").and_then(|v| v.as_str()).unwrap_or("failed");
        let short = if err_msg.chars().count() > 70 {
            let mut s: String = err_msg.chars().take(67).collect();
            s.push_str("...");
            s
        } else {
            err_msg.to_string()
        };
        format!("✖ {name}: {short}")
    } else {
        match name {
            "shell" | "bash" => {
                let code = output.get("exit_code").and_then(|v| v.as_i64()).unwrap_or(0);
                let stdout_lines = output.get("stdout").and_then(|v| v.as_str()).map(|s| s.lines().count()).unwrap_or(0);
                format!("✔ bash: exit {code} ({stdout_lines} lines)")
            }
            "read_file" => {
                let lines = output.get("content").and_then(|v| v.as_str()).map(|s| s.lines().count()).unwrap_or(0);
                format!("✔ read: {lines} lines")
            }
            "write_file" | "edit_file" => {
                format!("✔ {name}: done")
            }
            "glob" | "search_files" => {
                let count = output.get("files").or_else(|| output.get("matches")).and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
                format!("✔ {name}: {count} found")
            }
            _ => {
                format!("✔ {name}: done")
            }
        }
    }
}

fn command_hint(input: &str) -> Option<String> {
    let q = input.trim();
    if !q.starts_with('/') || q.contains(' ') {
        return None;
    }
    let mut hits: Vec<&&str> = COMMANDS.iter().map(|(c, _)| c).filter(|c| c.starts_with(q)).collect();
    if hits.is_empty() {
        // fuzzy: contains
        hits = COMMANDS.iter().map(|(c, _)| c).filter(|c| c.contains(&q[1..])).collect();
    }
    if hits.is_empty() {
        None
    } else {
        Some(hits.into_iter().take(5).map(|s| s.to_string()).collect::<Vec<_>>().join("   "))
    }
}

/// Handle a `/command`. Returns true when the input was a command (no agent submit).
async fn handle_slash(rt: &LucyRuntime, app: &mut App, raw: &str) -> bool {
    let text = raw.trim();
    if !text.starts_with('/') {
        return false;
    }
    let mut parts = text.splitn(2, char::is_whitespace);
    let cmd = parts.next().unwrap_or("").to_ascii_lowercase();
    let arg = parts.next().unwrap_or("").trim().to_owned();
    match cmd.as_str() {
        "/new" => {
            match rt.new_session(if arg.is_empty() { None } else { Some(arg) }).await {
                Ok(meta) => {
                    app.session_title = meta.title.clone();
                    app.session_id_short = short_id(&meta.id.0.to_string());
                    app.session_count = meta.message_count;
                    app.load_turns(&rt.history().await);
                    app.push_msg(ChatMsg::system(format!("New session: {} ({})", app.session_title, app.session_id_short)));
                    app.pin();
                    app.status = "Ready — new session".into();
                }
                Err(e) => app.push_msg(ChatMsg::system(format!("Failed to create session: {e}"))),
            }
            true
        }
        "/sessions" => {
            refresh_sessions(rt, app).await;
            app.show_sessions = true;
            app.status = "Sessions — ↑/↓ select · Enter switch · d delete · n new · Esc close".into();
            true
        }
        "/switch" => {
            refresh_sessions(rt, app).await;
            if arg.is_empty() {
                app.show_sessions = true;
                app.status = "Pick a session — ↑/↓ + Enter".into();
                return true;
            }
            match resolve_session_arg(&app.sessions, &arg) {
                Some(meta) => {
                    match rt.switch_session(&meta.id).await {
                        Ok(switched) => {
                            app.session_title = switched.title.clone();
                            app.session_id_short = short_id(&switched.id.0.to_string());
                            app.session_count = switched.message_count;
                            app.load_turns(&rt.history().await);
                            app.push_msg(ChatMsg::system(format!("Switched to: {}", app.session_title)));
                            app.status = "Ready".into();
                        }
                        Err(e) => app.push_msg(ChatMsg::system(format!("Switch failed: {e}"))),
                    }
                }
                None => app.push_msg(ChatMsg::system(format!("No session matches '{arg}'. Use /sessions to list."))),
            }
            true
        }
        "/rename" => {
            if arg.is_empty() {
                app.push_msg(ChatMsg::system("Usage: /rename <title>".into()));
            } else {
                match rt.rename_current(arg.clone()).await {
                    Ok(meta) => {
                        app.session_title = meta.title.clone();
                        app.push_msg(ChatMsg::system(format!("Renamed to: {}", meta.title)));
                    }
                    Err(e) => app.push_msg(ChatMsg::system(format!("Rename failed: {e}"))),
                }
            }
            true
        }
        "/delete" => {
            refresh_sessions(rt, app).await;
            let target = if arg.is_empty() {
                app.sessions.first().cloned().filter(|m| {
                    // default: current session when no arg
                    format!("{}", m.id.0).starts_with(&app.session_id_short) || m.title == app.session_title
                }).or_else(|| app.sessions.first().cloned())
            } else {
                resolve_session_arg(&app.sessions, &arg)
            };
            match target {
                Some(meta) => match rt.delete_session(&meta.id).await {
                    Ok(_) => {
                        let cur = rt.current_meta().await;
                        app.session_title = cur.title.clone();
                        app.session_id_short = short_id(&cur.id.0.to_string());
                        app.session_count = cur.message_count;
                        app.load_turns(&rt.history().await);
                        app.push_msg(ChatMsg::system(format!("Deleted session: {}", meta.title)));
                    }
                    Err(e) => app.push_msg(ChatMsg::system(format!("Delete failed: {e}"))),
                },
                None => app.push_msg(ChatMsg::system("No matching session to delete.".into())),
            }
            true
        }
        "/fork" => {
            match rt.fork_current().await {
                Ok(meta) => {
                    app.session_title = meta.title.clone();
                    app.session_id_short = short_id(&meta.id.0.to_string());
                    app.session_count = meta.message_count;
                    app.load_turns(&rt.history().await);
                    app.push_msg(ChatMsg::system(format!("Forked session: {}", meta.title)));
                }
                Err(e) => app.push_msg(ChatMsg::system(format!("Fork failed: {e}"))),
            }
            true
        }
        "/clear" => {
            match rt.clear_session().await {
                Ok(_) => {
                    app.messages.clear();
                    app.streaming.clear();
                    app.tool_active = None;
                    app.progress.clear();
                    app.mark_idle();
                    app.scroll = 0;
                    app.pin();
                    app.status = "Session cleared".into();
                }
                Err(e) => app.status = format!("Clear failed: {e}"),
            }
            true
        }
        "/compact" => {
            match rt.compact().await {
                Ok(msg) => app.status = msg,
                Err(e) => app.status = format!("Compact failed: {e}"),
            }
            // Refresh visible history after compaction.
            app.load_turns(&rt.history().await);
            app.pin();
            true
        }
        "/history" => {
            if app.history.is_empty() {
                app.push_msg(ChatMsg::system("No input history yet.".into()));
            } else {
                let mut out = String::from("Input history:\n");
                for (i, h) in app.history.iter().rev().take(10).enumerate() {
                    out.push_str(&format!("  {}. {}\n", i + 1, truncate_one_line(h, 100)));
                }
                app.push_msg(ChatMsg::system(out));
            }
            true
        }
        "/status" => {
            let meta = rt.current_meta().await;
            let usage_note = "tokens tracked per model call";
            app.push_msg(ChatMsg::system(format!(
                "Session: {} ({})\nMessages: {} · Model: {} · Dir: {}\n{usage_note}",
                meta.title,
                short_id(&meta.id.0.to_string()),
                meta.message_count,
                app.model_label,
                rt.sessions_dir().display(),
            )));
            true
        }
        "/model" => {
            if arg.is_empty() {
                let cur = if app.model_label.is_empty() { "(unknown)".to_owned() } else { app.model_label.clone() };
                app.push_msg(ChatMsg::lucy(format!("Lucy › Current model: {cur}")));
            } else {
                match rt.set_model(&arg) {
                    Ok(name) => {
                        app.model_label = name.clone();
                        app.status = format!("Model: {name}");
                    }
                    Err(e) => app.status = format!("Model error: {e}"),
                }
            }
            true
        }
        "/usage" => {
            let u = rt.usage();
            app.push_msg(ChatMsg::lucy(format!(
                "Lucy › Usage — prompt: {} tokens, completion: {} tokens, total: {} tokens",
                u.prompt_tokens, u.completion_tokens, u.total_tokens
            )));
            true
        }
        "/doctor" => {
            let checks = lucy_config::doctor();
            let mut out = String::from("Lucy › Doctor\n\n");
            for (name, ok, detail) in checks {
                let mark = if ok { "ok" } else { "!!" };
                out.push_str(&format!("- {mark} {name}: {detail}\n"));
            }
            app.push_msg(ChatMsg::lucy(out.trim_end().to_owned()));
            true
        }
        "/export" => {
            if arg.is_empty() {
                app.push_msg(ChatMsg::system("Usage: /export <file.json>".into()));
            } else {
                match rt.export_current(std::path::PathBuf::from(&arg)).await {
                    Ok(()) => app.push_msg(ChatMsg::system(format!("Exported session to {arg}"))),
                    Err(e) => app.push_msg(ChatMsg::system(format!("Export failed: {e}"))),
                }
            }
            true
        }
        "/help" | "/?" => {
            app.push_msg(ChatMsg::lucy(
                "Lucy › Available commands\n\n- /help — show this help\n- /clear — clear session history\n- /model [name] — show or set model\n- /compact — trim history into a summary\n- /usage — show token usage\n- /doctor — run config checks\n\nKeys\n\n- F2 hold-to-talk voice\n- PgUp/PgDn scroll\n- Esc cancel/quit".to_owned(),
            ));
            true
        }
        "/settings" | "/config" => {
            app.settings = true;
            true
        }
        "/quit" | "/exit" | "/q" => {
            app.status = "quit-requested".into();
            true
        }
        _ => {
            app.status = format!("Unknown command {cmd} — try /help");
            true
        }
    }
}

fn short_id(full: &str) -> String {
    full.chars().take(8).collect()
}

/// Slash handling when no runtime is available (degraded mode).
fn handle_slash_offline(app: &mut App, raw: &str) {
    let text = raw.trim();
    let mut parts = text.splitn(2, char::is_whitespace);
    let cmd = parts.next().unwrap_or("").to_ascii_lowercase();
    let arg = parts.next().unwrap_or("").trim().to_owned();
    match cmd.as_str() {
        "/help" | "/?" => {
            app.push_msg(ChatMsg::lucy(
                "Lucy › Available commands\n\n- /help — show this help\n- /clear — clear session history\n- /model [name] — show or set model\n- /compact — trim history into a summary\n- /usage — show token usage\n- /doctor — run config checks\n\nKeys\n\n- F2 hold-to-talk voice\n- PgUp/PgDn scroll\n- Esc cancel/quit".to_owned(),
            ));
        }
        "/clear" => {
            app.messages.clear();
            app.streaming.clear();
            app.tool_active = None;
            app.progress.clear();
            app.scroll = 0;
            app.pin();
            app.status = "Session cleared".into();
        }
        "/model" => {
            if arg.is_empty() {
                let cur = if app.model_label.is_empty() { "(unknown)".to_owned() } else { app.model_label.clone() };
                app.push_msg(ChatMsg::lucy(format!("Lucy › Current model: {cur}")));
            } else {
                app.status = "Setup required before changing model.".into();
            }
        }
        "/compact" => {
            app.status = "Setup required before compact works.".into();
        }
        "/usage" => {
            app.push_msg(ChatMsg::lucy("Lucy › Usage — prompt: 0 tokens, completion: 0 tokens, total: 0 tokens".to_owned()));
        }
        "/doctor" => {
            let checks = lucy_config::doctor();
            let mut out = String::from("Lucy › Doctor\n\n");
            for (name, ok, detail) in checks {
                let mark = if ok { "ok" } else { "!!" };
                out.push_str(&format!("- {mark} {name}: {detail}\n"));
            }
            app.push_msg(ChatMsg::lucy(out.trim_end().to_owned()));
        }
        "/quit" | "/exit" | "/q" => {
            app.status = "quit-requested".into();
        }
        _ => {
            app.status = format!("Unknown command {cmd} — try /help");
        }
    }
}

async fn refresh_sessions(rt: &LucyRuntime, app: &mut App) {
    match rt.list_sessions().await {
        Ok(list) => {
            // Preselect current session.
            let mut sel = 0;
            for (i, m) in list.iter().enumerate() {
                if short_id(&m.id.0.to_string()) == app.session_id_short {
                    sel = i;
                    break;
                }
            }
            app.sessions = list;
            app.sess_selected = sel.min(app.sessions.len().saturating_sub(1));
        }
        Err(e) => {
            app.push_msg(ChatMsg::system(format!("Failed to list sessions: {e}")));
        }
    }
}

fn resolve_session_arg(sessions: &[SessionMeta], arg: &str) -> Option<SessionMeta> {
    let a = arg.trim();
    // 1-based number
    if let Ok(n) = a.parse::<usize>() {
        if n >= 1 && n <= sessions.len() {
            return Some(sessions[n - 1].clone());
        }
    }
    // id prefix or title match
    let low = a.to_ascii_lowercase();
    sessions
        .iter()
        .find(|m| {
            m.id.0.to_string().to_ascii_lowercase().starts_with(&low)
                || m.title.to_ascii_lowercase().contains(&low)
        })
        .cloned()
}

fn draw(terminal:&mut Terminal<CrosstermBackend<Stdout>>,app:&App)->anyhow::Result<()>{
    terminal.draw(|frame|{
        let area=frame.area();
        if app.settings{
            if app.messages.is_empty() && app.streaming.is_empty(){ draw_welcome(frame,area,app); } else { draw_active(frame,area,app); }
            settings::draw(frame,area,&app.config,app.settings_selected);
        } else if app.messages.is_empty() && app.streaming.is_empty(){
            draw_welcome(frame,area,app)
        } else {
            draw_active(frame,area,app)
        }
        if app.show_sessions{
            draw_sessions_popup(frame,area,app);
        }
        if app.show_help{
            draw_help_popup(frame,area);
        }
        if app.approval.is_some(){
            draw_approval_popup(frame,area,app);
        }
    })?;
    Ok(())
}

fn draw_approval_popup(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App) {
    let Some(dlg) = app.approval.as_ref() else { return };
    let w = (area.width * 70 / 100).max(20).min(area.width.saturating_sub(2));
    let h = (area.height * 50 / 100).max(10).min(area.height.saturating_sub(2));
    let x = area.x + area.width.saturating_sub(w) / 2;
    let y = area.y + area.height.saturating_sub(h) / 2;
    let popup = Rect { x, y, width: w, height: h };
    frame.render_widget(Clear, popup);
    let block = Block::default().title(" Approval needed ").borders(Borders::ALL).border_style(Style::default().fg(Color::Yellow));
    frame.render_widget(block.clone(), popup);
    let inner = block.inner(popup);
    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(vec![
        Span::styled("Tool: ", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(dlg.name.clone()),
    ]));
    lines.push(Line::from(""));
    for l in dlg.input.lines() {
        lines.push(Line::from(Span::raw(l.to_owned())));
    }
    lines.push(Line::from(""));
    lines.push(Line::from("[y] allow once   [a] always allow   [n] deny"));
    frame.render_widget(Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }), inner);
}

fn draw_welcome(frame:&mut ratatui::Frame<'_>,area:Rect,app:&App){
    let v=Layout::default().direction(Direction::Vertical).constraints([Constraint::Min(3),Constraint::Length(11),Constraint::Length(3),Constraint::Length(3),Constraint::Length(3)]).split(area);
    frame.render_widget(Paragraph::new(Line::from(vec![Span::styled("LUCY",Style::default().add_modifier(Modifier::BOLD)),Span::raw("  ·  your computer companion")])).alignment(Alignment::Center),v[0]);
    let mascot=Text::from(MASCOT.iter().map(|x|Line::from(Span::raw(*x))).collect::<Vec<_>>());
    frame.render_widget(Paragraph::new(mascot).alignment(Alignment::Center),v[1]);
    let status=if app.listening{"◉  Listening… release to send"}else{"What can I do for you?"};
    frame.render_widget(Paragraph::new(status).alignment(Alignment::Center),v[2]);
    render_input(frame,v[3],app,"  Tell Lucy what to do — / for commands  ");
    let ptt=app.config.voice.push_to_talk.to_ascii_uppercase();
    frame.render_widget(Paragraph::new(format!("[Enter] send  [/] commands  [Hold {}] voice  [Ctrl+O] sessions  [F1] help  [Esc] quit",ptt)).alignment(Alignment::Center).style(Style::default().add_modifier(Modifier::DIM)),v[4]);
}

fn build_chat_lines(app: &App) -> Vec<Line<'static>> {
    let mut lines: Vec<Line>=Vec::new();
    for m in &app.messages{
        match m.kind {
            MsgKind::User => {
                lines.push(Line::from(vec![
                    Span::styled("You   ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                    Span::styled(m.text.clone(), Style::default().fg(Color::White)),
                ]));
            }
            MsgKind::Lucy => {
                for (i, bl) in format_reply_lines(&m.text).iter().enumerate(){
                    lines.push(render_body_line(bl,i==0));
                }
            }
            MsgKind::System => {
                for bl in format_reply_lines(&m.text) {
                    lines.push(Line::from(vec![
                        Span::styled("· ", Style::default().fg(Color::Yellow).add_modifier(Modifier::DIM)),
                        Span::styled(bl, Style::default().fg(Color::Yellow).add_modifier(Modifier::DIM)),
                    ]));
                }
            }
            MsgKind::Tool => {
                let is_err = m.text.starts_with('✖');
                let is_done = m.text.starts_with('✔');
                let style = if is_err {
                    Style::default().fg(Color::Red)
                } else if is_done {
                    Style::default().fg(Color::DarkGray)
                } else {
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::DIM)
                };
                let icon = if is_err || is_done { "" } else { "⚙ " };
                lines.push(Line::from(vec![
                    Span::styled(icon, style),
                    Span::styled(m.text.clone(), style),
                ]));
            }
        }
        lines.push(Line::from(""));
    }
    // Live streaming buffer.
    if !app.streaming.trim().is_empty() {
        for (i, bl) in format_reply_lines(app.streaming.trim()).iter().enumerate(){
            lines.push(render_body_line(bl, i==0 && true));
        }
        lines.push(Line::from(Span::styled("▍", Style::default().fg(Color::Green).add_modifier(Modifier::DIM))));
        lines.push(Line::from(""));
    } else if let Some(tool) = &app.tool_active {
        lines.push(Line::from(vec![
            Span::styled("⚙ ", Style::default().fg(Color::Magenta)),
            Span::styled(format!("Using {tool}…"), Style::default().fg(Color::Gray).add_modifier(Modifier::DIM)),
        ]));
        lines.push(Line::from(""));
    }
    // Persistent progress feed: what the planner is doing / did.
    for p in &app.progress {
        lines.push(Line::from(vec![
            Span::styled("› ", Style::default().fg(Color::Green).add_modifier(Modifier::DIM)),
            Span::styled(p.clone(), Style::default().fg(Color::Gray).add_modifier(Modifier::DIM)),
        ]));
    }
    if !app.progress.is_empty() {
        lines.push(Line::from(""));
    }
    // Background-activity feedback: animated spinner + phase + elapsed + detail.
    // This is the row that tells the user "something is happening, wait" —
    // it appears the instant they hit Enter, long before the first token.
    if app.busy {
        const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let ms = app.busy_elapsed_ms();
        let frame = FRAMES[((ms / 100) % FRAMES.len() as u128) as usize];
        let label = if app.phase.trim().is_empty() { "Working".to_owned() } else { app.phase.clone() };
        let mut spans = vec![
            Span::styled(format!("{frame} {label}…"), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ];
        let secs = ms / 1000;
        if secs > 0 {
            spans.push(Span::styled(format!(" ({secs}s)"), Style::default().fg(Color::Gray).add_modifier(Modifier::DIM)));
        }
        // The newest feed line already shows the detail — don't repeat it.
        let dup = app.progress.back().is_some_and(|l| l == &app.phase_detail);
        if !dup && !app.phase_detail.trim().is_empty() {
            spans.push(Span::styled(
                format!(" — {}", truncate_one_line(&app.phase_detail, 110)),
                Style::default().fg(Color::Gray).add_modifier(Modifier::DIM),
            ));
        }
        lines.push(Line::from(spans));
        lines.push(Line::from(""));
    }
    if lines.is_empty() {
        lines.push(Line::from(Span::styled("No messages yet — say hi or type /help.", Style::default().add_modifier(Modifier::DIM))));
    }
    lines
}

/// Visual (wrapped) row count — the fix for "new chats not visible".
/// Ratatui's scroll offset operates on wrapped rows, but the old code counted
/// logical lines only, so after enough wrapping the computed bottom fell short
/// and the viewport got stuck above the newest messages.
fn visual_rows(lines: &[Line], inner_width: usize) -> usize {
    let w = inner_width.max(1) as u32;
    lines.iter().map(|l| {
        let lw = l.width() as u32;
        if lw == 0 { 1 } else { ((lw + w - 1) / w).max(1) as usize }
    }).sum()
}

fn draw_active(frame:&mut ratatui::Frame<'_>,area:Rect,app:&App){
    let v=Layout::default().direction(Direction::Vertical).constraints([Constraint::Length(2),Constraint::Min(3),Constraint::Length(3),Constraint::Length(2)]).split(area);
    let h=Layout::default().direction(Direction::Horizontal).constraints([Constraint::Min(10),Constraint::Length(34)]).split(v[0]);
    let sess = if app.session_title.trim().is_empty() { "untitled".to_owned() } else { app.session_title.clone() };
    let left = format!("● LUCY  ·  {sess}");
    frame.render_widget(Paragraph::new(Line::from(vec![
        Span::styled("● LUCY",Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(format!("  ·  {sess}")),
    ])).style(Style::default()),h[0]);
    let _ = left;
    let right_text=if app.listening{ "◉ Listening…".to_owned() } else {
        let m=truncate_model_label(&app.model_label);
        let tokens=format_tokens(app.usage.total_tokens);
        if m.is_empty(){ format!("{tokens} · {}", app.status) } else { format!("{m} · {tokens} · {}", app.status) }
    };
    frame.render_widget(Paragraph::new(right_text).alignment(Alignment::Right),h[1]);

    let lines = build_chat_lines(app);
    let inner_h = v[1].height.saturating_sub(2) as usize;
    let inner_w = v[1].width.saturating_sub(2) as usize;
    let inner = inner_h.max(1);
    let total = visual_rows(&lines, inner_w.max(1));
    let max_scroll = total.saturating_sub(inner);
    let clamped = app.scroll.min(max_scroll);
    // Paragraph scroll = rows from top; offset-from-bottom = max - clamped.
    let scroll_top = max_scroll.saturating_sub(clamped) as u16;
    let title=if clamped>0{ format!(" Activity · {} · ▲ scrolled (End to latest) ", sess) } else { format!(" Activity · {} ", sess) };
    frame.render_widget(Paragraph::new(Text::from(lines)).block(Block::default().title(title).borders(Borders::ALL)).wrap(Wrap{trim:false}).scroll((scroll_top,0)),v[1]);

    // Slash-command autocomplete hint above the input.
    if app.input.starts_with('/') {
        if let Some(hint) = command_hint(&app.input) {
            let hint_area = Rect { x: v[2].x, y: v[2].y.saturating_sub(1), width: v[2].width, height: 1 };
            frame.render_widget(Paragraph::new(hint).style(Style::default().fg(Color::Cyan).add_modifier(Modifier::DIM)), hint_area);
        }
    }
    let input_title = if app.input.starts_with('/') { "  Command  " } else { "  Ask Lucy — / for commands  " };
    render_input(frame,v[2],app,input_title);
    frame.render_widget(Paragraph::new("[Enter] send   [/] commands   [Ctrl+N] new   [Ctrl+O] sessions   [F1] help   [Esc] quit").alignment(Alignment::Center).style(Style::default().add_modifier(Modifier::DIM)),v[3]);
}

fn draw_sessions_popup(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App) {
    let popup = Rect { x: area.width/8, y: area.height/8, width: area.width*6/8, height: area.height*6/8 };
    frame.render_widget(Clear, popup);
    let block = Block::default().title(" Sessions — ↑/↓ select · Enter switch · d delete · n new · Esc close ").borders(Borders::ALL).border_style(Style::default().fg(Color::Cyan));
    frame.render_widget(block.clone(), popup);
    let inner = block.inner(popup);
    if app.sessions.is_empty() {
        frame.render_widget(Paragraph::new("No sessions yet — press n for a new one.").alignment(Alignment::Center), inner);
        return;
    }
    let items: Vec<ListItem> = app.sessions.iter().enumerate().map(|(i, m)| {
        let cur = short_id(&m.id.0.to_string()) == app.session_id_short;
        let style = if i == app.sess_selected {
            Style::default().bg(Color::Rgb(45,45,65)).fg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };
        let marker = if cur { "● " } else { "  " };
        let line = format!("{marker}{:>2}. {}  [{} msgs]  {}", i + 1, truncate_one_line(&m.title, 40), m.message_count, short_id(&m.id.0.to_string()));
        ListItem::new(Line::from(vec![Span::styled(line, style)])).style(style)
    }).collect();
    frame.render_widget(List::new(items), inner);
}

fn draw_help_popup(frame: &mut ratatui::Frame<'_>, area: Rect) {
    let popup = Rect { x: area.width/8, y: area.height/8, width: area.width*6/8, height: area.height*6/8 };
    frame.render_widget(Clear, popup);
    let block = Block::default().title(" Lucy help — Esc/F1 to close ").borders(Borders::ALL).border_style(Style::default().fg(Color::Green));
    frame.render_widget(block.clone(), popup);
    let inner = block.inner(popup);
    let mut lines = vec![
        Line::from(Span::styled("Chat", Style::default().add_modifier(Modifier::BOLD))),
        Line::from("  Enter send · Shift+chars type · Up/Down input history · PgUp/PgDn scroll · End latest"),
        Line::from(""),
        Line::from(Span::styled("Sessions (opencode-style)", Style::default().add_modifier(Modifier::BOLD))),
    ];
    for (c, d) in COMMANDS {
        lines.push(Line::from(vec![
            Span::styled(format!("  {c:<10}"), Style::default().fg(Color::Cyan)),
            Span::raw(*d),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled("Keys", Style::default().add_modifier(Modifier::BOLD))));
    lines.push(Line::from("  Ctrl+N new session · Ctrl+O sessions · Ctrl+, settings · F1 help · Esc cancel/quit"));
    frame.render_widget(Paragraph::new(Text::from(lines)).wrap(Wrap{trim:false}), inner);
}
/// Break a raw assistant reply into readable display lines:
/// - run-on `* **Item:** ...` bullets each get their own line
/// - `**` bold markers and backticks are stripped (no markdown in terminal)
/// - at most one blank line in a row, no leading/trailing blanks
fn format_reply_lines(text:&str)->Vec<String>{
    let mut s=text.replace("\r\n","\n");
    s=s.replace("* **","\n• ");
    s=s.replace("**","");
    s=s.replace('`',"");
    // A bullet left mid-line (e.g. "…done. • Next…") starts its own line.
    let mut lines:Vec<String>=Vec::new();
    for raw in s.split('\n'){
        let mut rest=raw.trim_end();
        // Split off any mid-line " • " continuations.
        loop{
            if let Some(idx)=rest.find(" • "){
                let (head,tail)=rest.split_at(idx+1); // tail starts with "• "
                let head=head.trim_end_matches([' ','•']).trim_end();
                if !head.is_empty(){ lines.push(head.to_owned()); }
                rest=tail.trim_start();
            } else { break; }
        }
        // Strip a leftover leading "* "/"- " bullet into "• ".
        let t=rest.trim_start();
        if let Some(b)=t.strip_prefix("* ").or_else(||t.strip_prefix("- ")){
            lines.push(format!("• {b}"));
        } else {
            lines.push(rest.trim_end().to_owned());
        }
    }
    // Collapse 2+ blank lines into one, trim ends.
    let mut out:Vec<String>=Vec::with_capacity(lines.len());
    let mut blank=false;
    for l in lines{
        if l.trim().is_empty(){
            if !blank{ out.push(String::new()); }
            blank=true;
        } else { out.push(l); blank=false; }
    }
    while out.first().is_some_and(|l|l.is_empty()){ out.remove(0); }
    while out.last().is_some_and(|l|l.is_empty()){ out.pop(); }
    if out.is_empty(){ out.push(String::new()); }
    out
}
fn render_body_line(line:&str,first:bool)->Line<'static>{
    let prefix=if first{"Lucy  "}else{"      "};
    let pre=Span::styled(prefix.to_owned(),Style::default().fg(Color::Green).add_modifier(Modifier::BOLD));
    if line.trim().is_empty(){ return Line::from(""); }
    if let Some(rest)=line.strip_prefix("• "){
        // Bold the lead ("Title:") so items scan easily.
        if let Some(idx)=rest.find(": "){
            let (title,body)=rest.split_at(idx+1);
            return Line::from(vec![pre,Span::raw("• "),Span::styled(title.to_owned(),Style::default().add_modifier(Modifier::BOLD)),Span::raw(body.to_owned())]);
        }
        return Line::from(vec![pre,Span::raw("• "),Span::raw(rest.to_owned())]);
    }
    Line::from(vec![pre,Span::raw(line.to_owned())])
}
#[cfg(test)]
mod format_tests{
    use super::*;
    #[test]
    fn splits_run_on_bullets_and_strips_markers(){
        let raw="As Lucy, my superpowers center on operation:* **Direct Control:** Execute shell. * **Files:** Read with `ripgrep`.";
        let lines=format_reply_lines(raw);
        assert!(lines.iter().any(|l|l.starts_with("• Direct Control:")), "got {lines:?}");
        assert!(lines.iter().any(|l|l.starts_with("• Files:")), "got {lines:?}");
        assert!(!lines.iter().any(|l|l.contains("**")||l.contains('`')), "got {lines:?}");
    }
    #[test]
    fn collapses_blank_lines(){
        let lines=format_reply_lines("a\n\n\nb\n");
        assert_eq!(lines,vec!["a".to_owned(),"".to_owned(),"b".to_owned()]);
    }
    #[test]
    fn matches_voice_hotkey_case_insensitively_and_fallbacks(){
        let f2 = KeyEvent::new(KeyCode::F(2), KeyModifiers::NONE);
        let f3 = KeyEvent::new(KeyCode::F(3), KeyModifiers::NONE);
        let esc = KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);
        assert!(is_voice_hotkey(&f2, "f2"));
        assert!(is_voice_hotkey(&f2, "F2"));
        assert!(is_voice_hotkey(&f3, "f3"));
        assert!(!is_voice_hotkey(&f3, "f2"));
        assert!(!is_voice_hotkey(&esc, "f2"));
    }
    #[test]
    fn visual_rows_counts_wrapping(){
        let lines = vec![Line::from("a".repeat(100))];
        // width 10 -> 10 visual rows
        assert_eq!(visual_rows(&lines, 10), 10);
        assert_eq!(visual_rows(&[Line::from("")], 10), 1);
    }
    #[test]
    fn streaming_appends_not_spams(){
        let mut app = App::new(LucyConfig::default());
        app.append_stream("hello ");
        app.append_stream("world");
        assert_eq!(app.streaming, "hello world");
        assert!(app.messages.is_empty());
        app.flush_stream();
        assert_eq!(app.messages.len(), 1);
        assert!(app.streaming.is_empty());
    }
    #[test]
    fn progress_feed_caps_and_dedups(){
        let mut app = App::new(LucyConfig::default());
        app.push_progress("Step 1: open the browser");
        app.push_progress("Step 1: open the browser"); // consecutive dup ignored
        assert_eq!(app.progress.len(), 1);
        assert!(app.busy);
        assert_eq!(app.phase_detail, "Step 1: open the browser");
        for i in 0..20 {
            app.push_progress(&format!("note {i}"));
        }
        assert_eq!(app.progress.len(), MAX_PROGRESS);
        assert!(app.progress.back().unwrap().starts_with("note "));
    }
    #[test]
    fn activity_phase_tracks_busy_until_idle(){
        let mut app = App::new(LucyConfig::default());
        assert!(!app.busy);
        app.set_phase("Thinking", "sending to model");
        assert!(app.busy);
        assert_eq!(app.phase, "Thinking");
        assert!(app.phase_since.is_some());
        app.set_phase("Writing", "");
        assert_eq!(app.phase, "Writing");
        // Empty detail must not wipe the previous detail.
        assert_eq!(app.phase_detail, "sending to model");
        app.mark_idle();
        assert!(!app.busy);
        assert!(app.phase_since.is_none());
    }
    #[test]
    fn resolves_session_arg(){
        use lucy_core::{SessionData, SessionId};
        let mk = |title: &str| {
            let s = SessionData::new(SessionId::default()).with_title(title);
            SessionMeta::from(&s)
        };
        let sessions = vec![mk("alpha"), mk("beta")];
        assert!(resolve_session_arg(&sessions, "1").is_some());
        assert!(resolve_session_arg(&sessions, "alp").is_some());
        assert!(resolve_session_arg(&sessions, "zzz").is_none());
    }
}
fn render_input(frame:&mut ratatui::Frame<'_>,area:Rect,app:&App,title:&str){
    let hint = if app.input.is_empty() && !title.contains("Command") {
        app.input.clone()
    } else { app.input.clone() };
    let _ = hint;
    frame.render_widget(Paragraph::new(format!("> {}",app.input)).block(Block::default().title(title).borders(Borders::ALL)).wrap(Wrap{trim:true}),area);
    let (row, col) = cursor_row_col(&app.input, app.cursor);
    let row0 = area.y.saturating_add(1);
    let max_y = area.bottom().saturating_sub(1);
    let y = row0.saturating_add(row as u16).min(max_y);
    let base_x = if row == 0 { area.x.saturating_add(3) } else { area.x.saturating_add(1) };
    let x = base_x.saturating_add(col as u16).min(area.right().saturating_sub(1));
    frame.set_cursor_position((x, y));
}
fn cleanup(t:&mut Terminal<CrosstermBackend<Stdout>>)->anyhow::Result<()>{
    disable_raw_mode()?;
    execute!(t.backend_mut(),LeaveAlternateScreen)?;
    t.show_cursor()?;
    Ok(())
}

fn is_voice_hotkey(key:&KeyEvent,ptt_config:&str)->bool{
    let ptt=ptt_config.trim().to_ascii_lowercase();
    if ptt.starts_with('f'){
        if let Ok(num)=ptt[1..].parse::<u8>(){
            return key.code==KeyCode::F(num);
        }
    }
    key.code==KeyCode::F(2)
}

// A recording started by a toggle press of the PTT key.
struct Recording {
    cap: HoldCapture,
    started_at: Instant,
}

// Stop recording and dispatch transcription. Safe to call with hold == None.
fn stop_recording(
    hold: &mut Option<Recording>,
    stt: Option<&Arc<GroqStt>>,
    app: &mut App,
    voice_tx: &mpsc::UnboundedSender<Result<String, String>>,
) {
    if let Some(rec) = hold.take() {
        app.listening = false;
        if let Some(stt) = stt {
            let stt = Arc::clone(stt);
            let tx = voice_tx.clone();
            app.status = "Transcribing…".into();
            tokio::spawn(async move {
                let res = stt.finish_hold(rec.cap).await.map_err(|e| e.to_string());
                let _ = tx.send(res);
            });
        } else {
            app.status = "Voice disabled — set GROQ_API_KEY".into();
        }
    }
}

async fn submit_text(rt: Option<&Arc<LucyRuntime>>, app: &mut App, agent_rx: &mut Option<mpsc::UnboundedReceiver<AgentEvent>>, text: String) {
    // Slash commands are handled locally — they never hit the agent.
    if text.starts_with('/') {
        if let Some(rt) = rt {
            let was_quit = text.trim().to_ascii_lowercase() == "/quit"
                || text.trim().to_ascii_lowercase() == "/exit"
                || text.trim().to_ascii_lowercase() == "/q";
            handle_slash(rt, app, &text).await;
            if was_quit {
                // marker read by the main loop to exit cleanly
            }
        } else {
            handle_slash_offline(app, &text);
        }
        app.pin();
        return;
    }
    app.push_msg(ChatMsg::user(text.clone()));
    app.pin();
    if let Some(rt)=rt{
        // Interrupt any running turn before starting a new one (opencode-style).
        if agent_rx.is_some(){
            rt.interrupt();
            *agent_rx=None;
            app.flush_stream();
        }
        // Instant feedback — the spinner row renders on the very next frame,
        // before any LLM event arrives.
        app.set_phase("Thinking", "sending to model");
        match rt.submit(text).await{
            Ok(rx)=>*agent_rx=Some(rx),
            Err(e)=>{ app.mark_idle(); app.status=format!("Error: {e}"); },
        }
    } else {
        app.status="Setup required: LLM API Key missing — open Settings (Ctrl+,) set keys or run: lucy config doctor".into();
    }
}

pub async fn run_voice(stt:Option<Arc<GroqStt>>)->anyhow::Result<()>{
    let config=LucyConfig::load()?;
    // Try to create runtime but allow degraded mode if API key is missing (any brand)
    let (runtime_opt, runtime_error): (Option<Arc<LucyRuntime>>, Option<String>) = match LucyRuntime::new().await {
        Ok(r) => (Some(Arc::new(r)), None),
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("API key") || msg.contains("OPENAI_API_KEY") || msg.contains("ANTHROPIC_API_KEY") {
                (None, Some(msg))
            } else {
                (None, Some(format!("{msg} (run: lucy config doctor)")))
            }
        }
    };
    enable_raw_mode()?;
    let mut stdout=io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend=CrosstermBackend::new(stdout);
    let mut terminal=Terminal::new(backend)?;
    let mut app=App::new(config);
    app.model_label=runtime_opt.as_ref().map(|r| r.config().models.main.clone()).unwrap_or(app.config.models.main.clone());
    // Hydrate current session into the new chat model.
    if let Some(rt) = runtime_opt.as_ref() {
        let meta = rt.current_meta().await;
        app.session_title = meta.title.clone();
        app.session_id_short = short_id(&meta.id.0.to_string());
        app.session_count = meta.message_count;
        app.load_turns(&rt.history().await);
        if app.messages.is_empty() {
            app.status = "Ready — type /help for commands".into();
        }
    }
    if let Some(err) = runtime_error.clone() {
        app.status = format!("Setup required: {err} — press Ctrl+, for settings or run: lucy config doctor");
    } else if stt.is_none() && app.messages.is_empty(){
        app.status="Ready — voice optional (set GROQ_API_KEY) · /help for commands".into();
    }
    let (voice_tx,mut voice_rx)=mpsc::unbounded_channel::<Result<String,String>>();
    // Toggle recording: None = idle, Some = actively recording.
    let mut recording:Option<Recording>=None;
    // Safety cap: auto-send after 60 seconds even if user forgets to press F2 again.
    const MAX_RECORD:Duration=Duration::from_secs(60);
    let mut agent_rx:Option<mpsc::UnboundedReceiver<AgentEvent>>=None;
    let result=loop{
        draw(&mut terminal,&app)?;
        let mut agent_done=false;
        if let Some(rx)=agent_rx.as_mut(){
            loop{
                match rx.try_recv(){
                    Ok(event)=>{ match event{
                        AgentEvent::Status{message}=>{
                            // Map engine statuses to human phases.
                            let low = message.to_ascii_lowercase();
                            let phase = if low.contains("planning") || low.contains("understanding") || low.contains("preparing") {
                                "Thinking"
                            } else if low.contains("observing") || low.contains("verify") {
                                "Verifying"
                            } else if low.contains("completed") || low.contains("wrapping") {
                                "Wrapping up"
                            } else {
                                "Working"
                            };
                            app.set_phase(phase, &message);
                        },
                        // FIX: append deltas into one streaming bubble instead of one
                        // message per delta (the old code spammed `commands` and, combined
                        // with logical-line scroll math, pushed newest chats out of view).
                        AgentEvent::TextDelta{text}=>{
                            app.append_stream(&text);
                            app.tool_active=None;
                            app.set_phase("Writing", "");
                        },
                        AgentEvent::ToolStarted{name, input, ..}=>{
                            app.flush_stream();
                            let label = format_tool_start(&name, &input);
                            app.tool_active = Some(label.clone());
                            app.push_msg(ChatMsg::tool(format!("{label} …")));
                            app.set_phase(&label, "");
                        },
                        AgentEvent::ToolFinished{name, output, is_error, ..}=>{
                            app.tool_active = None;
                            let summary = format_tool_finish(&name, &output, is_error);
                            app.push_msg(ChatMsg::tool(summary));
                            app.set_phase("Working", "");
                        },
                        AgentEvent::History{..}=>{},
                        AgentEvent::ApprovalRequest{id,name,input}=>{
                            let pretty = format!("{input:#}");
                            let truncated = if pretty.chars().count() > 800 {
                                let mut t: String = pretty.chars().take(800).collect();
                                t.push('…');
                                t
                            } else { pretty };
                            app.approval = Some(ApprovalDialog { id, name: name.clone(), input: truncated });
                            app.listening = false;
                            app.status = format!("Approval needed: {name} (y/a/n)");
                        },
                        AgentEvent::Thinking{text}=>app.set_phase("Thinking", &text),
                        AgentEvent::Progress{message}=>app.push_progress(&message),
                        AgentEvent::Error{message}=>{
                            app.flush_stream();
                            app.mark_idle();
                            app.push_msg(ChatMsg::system(format!("Error: {message}")));
                            app.status=format!("Error: {message}");
                            app.pin();
                        },
                        AgentEvent::Done=>{ agent_done=true; break; },
                    } },
                    Err(mpsc::error::TryRecvError::Empty)=>break,
                    Err(mpsc::error::TryRecvError::Disconnected)=>{ agent_done=true; break; },
                }
            }
        }
        if agent_done{
            agent_rx=None;
            app.flush_stream();
            app.mark_idle();
            app.pin();
            // Refresh session meta (title/count may have changed).
            if let Some(rt) = runtime_opt.as_ref() {
                let meta = rt.current_meta().await;
                app.session_title = meta.title.clone();
                app.session_id_short = short_id(&meta.id.0.to_string());
                app.session_count = meta.message_count;
            }
            // Only overwrite transient statuses, keep setup errors visible.
            if !app.status.starts_with("Setup required") {
                app.status="Ready".into();
            }
        }
        if let Some(rt)=runtime_opt.as_ref(){ app.usage = rt.usage(); }
        while let Ok(result)=voice_rx.try_recv(){
            app.listening=false;
            match result{
                Ok(text) if !text.trim().is_empty()=>{
                    let text=text.trim().to_owned();
                    submit_text(runtime_opt.as_ref(), &mut app, &mut agent_rx, text).await;
                    if app.status=="quit-requested"{ break; }
                },
                Ok(_)=>app.status="I didn't catch anything — try again".into(),
                Err(e) if e.contains("too short")||e.contains("no speech")=>app.status=format!("Speak after pressing {}, then press it again to send", "F2").into(),
                Err(e)=>app.status=format!("Voice error: {e}"),
            }
        }
        if app.status=="quit-requested"{ break Ok(()); }
        // Safety cap: auto-send if user forgets to press F2 a second time.
        if let Some(rec) = &recording {
            if rec.started_at.elapsed() >= MAX_RECORD {
                stop_recording(&mut recording, stt.as_ref(), &mut app, &voice_tx);
            }
        }
        if event::poll(Duration::from_millis(100))?{
            if let Event::Key(key)=event::read()?{
                // Only act on key-press events (ignore repeats and releases entirely).
                if key.kind != KeyEventKind::Press { continue; }

                // Approval modal takes precedence over everything: y/a/n/Esc
                // resolve, all other keys are swallowed.
                if let Some(dlg) = app.approval.clone() {
                    let decision = match key.code {
                        KeyCode::Char('y') | KeyCode::Char('Y') if !key.modifiers.intersects(KeyModifiers::CONTROL|KeyModifiers::ALT|KeyModifiers::SUPER) => Some(ApprovalDecision::AllowOnce),
                        KeyCode::Char('a') | KeyCode::Char('A') if !key.modifiers.intersects(KeyModifiers::CONTROL|KeyModifiers::ALT|KeyModifiers::SUPER) => Some(ApprovalDecision::AllowAlways),
                        KeyCode::Char('n') | KeyCode::Char('N') if !key.modifiers.intersects(KeyModifiers::CONTROL|KeyModifiers::ALT|KeyModifiers::SUPER) => Some(ApprovalDecision::Deny),
                        KeyCode::Esc => Some(ApprovalDecision::Deny),
                        _ => None,
                    };
                    if let Some(d) = decision {
                        if let Some(rt) = runtime_opt.as_ref() {
                            rt.resolve_approval(&dlg.id, d);
                        }
                        app.approval = None;
                        app.set_phase("Working", "approved — resuming");
                    }
                    continue;
                }

                // Help overlay takes precedence.
                if app.show_help {
                    match key.code {
                        KeyCode::Esc | KeyCode::F(1) => { app.show_help = false; },
                        _ => { app.show_help = false; },
                    }
                    continue;
                }

                // Sessions overlay.
                if app.show_sessions {
                    match key.code{
                        KeyCode::Esc=>{ app.show_sessions=false; app.status="Ready".into(); },
                        KeyCode::Up=>{ if !app.sessions.is_empty(){ app.sess_selected=app.sess_selected.saturating_sub(1); } },
                        KeyCode::Down=>{ if !app.sessions.is_empty(){ app.sess_selected=(app.sess_selected+1)%app.sessions.len(); } },
                        KeyCode::Enter=>{
                            if let Some(meta)=app.sessions.get(app.sess_selected).cloned(){
                                if let Some(rt)=runtime_opt.as_ref(){
                                    match rt.switch_session(&meta.id).await{
                                        Ok(switched)=>{
                                            app.session_title=switched.title.clone();
                                            app.session_id_short=short_id(&switched.id.0.to_string());
                                            app.session_count=switched.message_count;
                                            app.load_turns(&rt.history().await);
                                            app.push_msg(ChatMsg::system(format!("Switched to: {}", switched.title)));
                                            app.pin();
                                        },
                                        Err(e)=>app.push_msg(ChatMsg::system(format!("Switch failed: {e}"))),
                                    }
                                }
                            }
                            app.show_sessions=false;
                            app.status="Ready".into();
                        },
                        KeyCode::Char('d') if !key.modifiers.intersects(KeyModifiers::CONTROL|KeyModifiers::ALT|KeyModifiers::SUPER)=>{
                            if let Some(meta)=app.sessions.get(app.sess_selected).cloned(){
                                if let Some(rt)=runtime_opt.as_ref(){
                                    let _=rt.delete_session(&meta.id).await;
                                    refresh_sessions(rt,&mut app).await;
                                }
                            }
                        },
                        KeyCode::Char('n') if !key.modifiers.intersects(KeyModifiers::CONTROL|KeyModifiers::ALT|KeyModifiers::SUPER)=>{
                            if let Some(rt)=runtime_opt.as_ref(){
                                if let Ok(meta)=rt.new_session(None).await{
                                    app.session_title=meta.title.clone();
                                    app.session_id_short=short_id(&meta.id.0.to_string());
                                    app.session_count=meta.message_count;
                                    app.load_turns(&[]);
                                    app.show_sessions=false;
                                    app.status="Ready — new session".into();
                                }
                            }
                        },
                        _=>{},
                    }
                    continue;
                }

                if app.settings{
                    match key.code{
                        KeyCode::Esc=>{  app.settings=false; },
                        KeyCode::Up=>{ app.settings_selected=app.settings_selected.saturating_sub(1); },
                        KeyCode::Down=>{ app.settings_selected=(app.settings_selected+1)%settings::ITEMS.len(); },
                        KeyCode::Left=>settings::change(&mut app.config,app.settings_selected,-1),
                        KeyCode::Right=>settings::change(&mut app.config,app.settings_selected,1),
                        KeyCode::Enter=>{ settings::change(&mut app.config,app.settings_selected,1); },
                        KeyCode::Backspace=>{
                            let sel=app.settings_selected;
                            if matches!(sel,0|1|2|3|4|5|12|13|14){
                                let cur=settings::raw_value(&app.config, sel);
                                let mut v=cur; v.pop();
                                settings::edit_text(&mut app.config, sel, &v);
                            } else {
                                settings::change(&mut app.config,app.settings_selected,-1);
                            }
                        },
                        KeyCode::Char(c) if !key.modifiers.intersects(KeyModifiers::CONTROL|KeyModifiers::ALT|KeyModifiers::SUPER)=>{
                            let sel=app.settings_selected;
                            if matches!(sel,0|1|2|3|4|5|12|13|14){
                                let cur=settings::raw_value(&app.config, sel);
                                let next=format!("{cur}{c}");
                                settings::edit_text(&mut app.config, sel, &next);
                            }
                        },
                        KeyCode::Char('s') if key.modifiers.contains(KeyModifiers::CONTROL)=>{
                            match app.config.save(){
                                Ok(())=>app.status="Settings saved — restart to apply".into(),
                                Err(e)=>app.status=format!("Settings error: {e}"),
                            }
                        },
                        _=>{},
                    }
                    continue;
                }
                if key.code==KeyCode::Char(',')&&key.modifiers.contains(KeyModifiers::CONTROL){
                    app.settings=true;
                    continue;
                }
                // Ctrl+N: new session · Ctrl+O: sessions · F1: help
                if key.code==KeyCode::Char('n')&&key.modifiers.contains(KeyModifiers::CONTROL){
                    if let Some(rt)=runtime_opt.as_ref(){
                        match rt.new_session(None).await{
                            Ok(meta)=>{
                                app.session_title=meta.title.clone();
                                app.session_id_short=short_id(&meta.id.0.to_string());
                                app.session_count=meta.message_count;
                                app.load_turns(&[]);
                                app.push_msg(ChatMsg::system(format!("New session: {} ({})", meta.title, app.session_id_short)));
                                app.pin();
                                app.status="Ready — new session".into();
                            },
                            Err(e)=>app.status=format!("New session failed: {e}"),
                        }
                    }
                    continue;
                }
                if key.code==KeyCode::Char('o')&&key.modifiers.contains(KeyModifiers::CONTROL){
                    if runtime_opt.is_some(){
                        refresh_sessions(runtime_opt.as_ref().unwrap(),&mut app).await;
                        app.show_sessions=true;
                        app.status="Sessions — ↑/↓ select · Enter switch · d delete · n new · Esc close".into();
                    }
                    continue;
                }
                if key.code==KeyCode::F(1){
                    app.show_help=!app.show_help;
                    continue;
                }

                let is_ptt = is_voice_hotkey(&key, &app.config.voice.push_to_talk);

                if is_ptt {
                    let ptt_label = app.config.voice.push_to_talk.to_ascii_uppercase();
                    if recording.is_some() {
                        // Second press → stop and transcribe.
                        stop_recording(&mut recording, stt.as_ref(), &mut app, &voice_tx);
                    } else if let Some(stt) = stt.as_ref() {
                        // First press → start recording.
                        match stt.start_hold() {
                            Ok(cap) => {
                                recording = Some(Recording { cap, started_at: Instant::now() });
                                app.listening = true;
                                app.status = format!("🎙 Listening… press {ptt_label} again to send");
                            }
                            Err(e) => {
                                app.status = format!("Voice error: {e}");
                            }
                        }
                    } else {
                        app.status = "Voice disabled — set GROQ_API_KEY".into();
                    }
                    continue;
                }

                // Multiline: Alt+Enter or Ctrl+J inserts a newline at cursor (no submit).
                if key.code == KeyCode::Enter && (key.modifiers.contains(KeyModifiers::ALT) || key.modifiers.contains(KeyModifiers::CONTROL)) {
                    let bi = char_byte_idx(&app.input, app.cursor.min(app.input.chars().count()));
                    app.input.insert(bi, '\n');
                    app.cursor += 1;
                    app.hist_idx = None;
                    continue;
                }
                if let KeyCode::Char(c) = key.code {
                    if (c == 'j' || c == 'J') && key.modifiers.contains(KeyModifiers::CONTROL) && !key.modifiers.intersects(KeyModifiers::ALT|KeyModifiers::SUPER) {
                        let bi = char_byte_idx(&app.input, app.cursor.min(app.input.chars().count()));
                        app.input.insert(bi, '\n');
                        app.cursor += 1;
                        app.hist_idx = None;
                        continue;
                    }
                }

                match key.code{
                    KeyCode::Esc=>{
                        if let Some(rec)=recording.take(){
                            drop(rec.cap);
                            app.listening=false;
                            app.status="Cancelled".into();
                        } else if app.show_help {
                            app.show_help=false;
                        } else if agent_rx.is_some(){
                            if let Some(rt)=runtime_opt.as_ref(){ rt.interrupt(); }
                            app.set_phase("Cancelling", "");
                        } else {
                            break Ok(());
                        }
                    },
                    KeyCode::Up=>{
                        if !app.history.is_empty(){
                            if app.hist_idx.is_none(){
                                app.hist_draft=app.input.clone();
                                let idx=app.history.len()-1;
                                app.hist_idx=Some(idx);
                                app.input=app.history[idx].clone();
                                app.cursor=app.input.chars().count();
                            } else if let Some(idx)=app.hist_idx{
                                if idx>0{
                                    let nidx=idx-1;
                                    app.hist_idx=Some(nidx);
                                    app.input=app.history[nidx].clone();
                                    app.cursor=app.input.chars().count();
                                }
                            }
                        }
                    },
                    KeyCode::Down=>{
                        if let Some(idx)=app.hist_idx{
                            if idx+1<app.history.len(){
                                let nidx=idx+1;
                                app.hist_idx=Some(nidx);
                                app.input=app.history[nidx].clone();
                                app.cursor=app.input.chars().count();
                            } else {
                                app.hist_idx=None;
                                app.input=app.hist_draft.clone();
                                app.cursor=app.input.chars().count();
                            }
                        }
                    },
                    KeyCode::PageUp=>{ app.scroll=app.scroll.saturating_add(10); },
                    KeyCode::PageDown=>{ app.scroll=app.scroll.saturating_sub(10); },
                    KeyCode::Home if key.modifiers.contains(KeyModifiers::CONTROL)=>{ app.scroll=usize::MAX; },
                    KeyCode::End if key.modifiers.contains(KeyModifiers::CONTROL)=>{ app.pin(); },
                    KeyCode::Left=>{ app.cursor=app.cursor.saturating_sub(1); },
                    KeyCode::Right=>{ let n=app.input.chars().count(); app.cursor=(app.cursor+1).min(n); },
                    KeyCode::Home=>{ app.cursor=line_home_cursor(&app.input, app.cursor); },
                    KeyCode::End=>{ app.cursor=line_end_cursor(&app.input, app.cursor); },
                    KeyCode::Char(c) if key.modifiers.contains(KeyModifiers::CONTROL)&&c.to_ascii_lowercase()=='a'=>{ app.cursor=0; },
                    KeyCode::Char(c) if key.modifiers.contains(KeyModifiers::CONTROL)&&c.to_ascii_lowercase()=='e'=>{ app.cursor=app.input.chars().count(); },
                    KeyCode::Char(c) if !key.modifiers.intersects(KeyModifiers::CONTROL|KeyModifiers::ALT|KeyModifiers::SUPER)=>{
                        let bi=char_byte_idx(&app.input,app.cursor.min(app.input.chars().count()));
                        app.input.insert(bi,c);
                        app.cursor+=1;
                        app.hist_idx=None;
                    },
                    KeyCode::Backspace=>{
                        if app.cursor>0{
                            let count=app.input.chars().count();
                            let cur=app.cursor.min(count);
                            let bi=char_byte_idx(&app.input,cur-1);
                            let ch_len=app.input[bi..].chars().next().map(|c|c.len_utf8()).unwrap_or(0);
                            if ch_len>0{ app.input.drain(bi..bi+ch_len); }
                            app.cursor=cur-1;
                        }
                        app.hist_idx=None;
                    },
                    KeyCode::Enter if !key.modifiers.intersects(KeyModifiers::CONTROL|KeyModifiers::ALT|KeyModifiers::SUPER)&&!app.input.trim().is_empty()=>{
                        let text=app.input.trim().to_owned();
                        app.input.clear();
                        app.cursor=0;
                        // Slash and normal lines alike go to input history.
                        if app.history.last().is_none_or(|l|l!=&text){
                            app.history.push(text.clone());
                            if app.history.len()>200{
                                let excess=app.history.len()-200;
                                app.history.drain(0..excess);
                            }
                        }
                        app.hist_idx=None;
                        app.hist_draft.clear();
                        submit_text(runtime_opt.as_ref(), &mut app, &mut agent_rx, text).await;
                        if app.status=="quit-requested"{ break Ok(()); }
                    },
                    _=>{},
                }
            }
        }
    };
    let cleanup_result=cleanup(&mut terminal);
    result.and(cleanup_result)
}
pub async fn run()->anyhow::Result<()>{
    let cfg=LucyConfig::load()?;
    let stt=GroqStt::from_config(&cfg).ok().map(Arc::new);
    run_voice(stt).await
}
