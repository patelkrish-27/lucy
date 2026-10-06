//! Slash commands (opencode-style): dispatch, session management, submit.
//!
//! `handle_slash` never touches the agent — commands run locally against
//! `LucyRuntime`. Plain text goes to the hierarchical loop via `submit_text`.

use std::sync::Arc;

use lucy_core::{ApprovalDecision, SessionMeta};
use lucy_runtime::LucyRuntime;

use super::model::{App, ChatMsg, MascotSize};
use super::util::{short_id, truncate_one_line};

// ---------- slash commands (opencode-style) ----------
pub(crate) const COMMANDS: &[(&str, &str)] = &[
    ("/new", "start a new session — /new [title]"),
    ("/sessions", "list sessions (Ctrl+O)"),
    ("/switch", "switch session — /switch <number|id>"),
    ("/rename", "rename current session — /rename <title>"),
    ("/delete", "delete a session — /delete [<number|id>]"),
    ("/fork", "fork current session (keeps history)"),
    ("/clear", "clear current session history"),
    ("/compact", "trim history — /compact [keep_n=40]"),
    ("/agent", "run the ReAct loop — /agent <goal>"),
    (
        "/auto",
        "automode: never ask — /auto [on|off|always|status]",
    ),
    ("/stop", "kill switch: stop the running task (Ctrl+C)"),
    (
        "/serve",
        "mobile bridge: on | off | status — /serve [on|off|status]",
    ),
    ("/history", "show input history"),
    ("/mascot", "mascot size — /mascot [auto|large|small|off]"),
    (
        "/companion",
        "desktop companion overlay — /companion [show|hide|toggle|<note>]",
    ),
    ("/status", "show session + runtime status"),
    ("/export", "export session — /export <file.json>"),
    ("/usage", "show token usage"),
    ("/doctor", "run config checks"),
    ("/memory", "memory hub — /memory [status|search <query>|slim|assets]"),
    ("/wiki-ingest", "index Markdown into memory — /wiki-ingest <file>"),
    ("/codegraph", "index Rust symbols — /codegraph <file>"),
    ("/help", "show help (F2)"),
    ("/settings", "open settings (Ctrl+,)"),
    ("/quit", "quit lucy"),
];

/// The `/help` body. One constant, because it used to be duplicated between the
/// online and offline handlers — which is how `/model` outlived the command:
/// the two copies drifted and neither was checked against [`COMMANDS`].
///
/// Models and providers are named here by pointing at `/settings`, the one
/// place that changes them.
const HELP_TEXT: &str = "Lucy › Available commands\n\n- /help — show this help\n- /clear — clear session history\n- /settings — model, provider, permissions (Ctrl+,)\n- /compact — trim history into a summary\n- /agent <goal> — run the ReAct loop\n- /usage — show token usage\n- /doctor — run config checks\n- /memory — layered memory status/search/slim/assets\n- /wiki-ingest <file> — index Markdown knowledge\n- /codegraph <file> — index Rust symbols\n\nKeys\n\n- F2 hold-to-talk voice\n- PgUp/PgDn scroll\n- Esc cancel/quit";

/// Commands matching the current input, for the `/` suggestion popup.
///
/// A bare `/` lists everything. Otherwise we try a prefix match first and fall
/// back to a substring match on the typed query, so `/se` finds `/settings`
/// and `/sessions` while `/ession` still finds `/sessions`.
pub(crate) fn command_matches(input: &str) -> Vec<(&'static str, &'static str)> {
    let q = input.trim();
    if !q.starts_with('/') || q.contains(char::is_whitespace) {
        return Vec::new();
    }
    let stem = &q[1..];
    let mut hits: Vec<(&'static str, &'static str)> = COMMANDS
        .iter()
        .copied()
        .filter(|(c, _)| c.starts_with(q))
        .collect();
    if hits.is_empty() && !stem.is_empty() {
        hits = COMMANDS
            .iter()
            .copied()
            .filter(|(c, _)| c[1..].contains(stem))
            .collect();
    }
    hits
}

/// Human-readable one-line label for a tool call start (`bash: …`, `read: …`).
#[allow(dead_code)]
pub(crate) fn format_tool_start(name: &str, input: &serde_json::Value) -> String {
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
            let pat = input
                .get("pattern")
                .or_else(|| input.get("query"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
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
                let short = name
                    .trim_start_matches("mcp_")
                    .trim_start_matches("hyprfast_");
                format!("{short} …")
            } else {
                format!("{name} …")
            }
        }
    }
}

/// Human-readable one-line summary of a finished tool call (`✔` / `✖`).
#[allow(dead_code)]
pub(crate) fn format_tool_finish(name: &str, output: &serde_json::Value, is_error: bool) -> String {
    if is_error {
        let err_msg = output
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("failed");
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
                let code = output
                    .get("exit_code")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                let stdout_lines = output
                    .get("stdout")
                    .and_then(|v| v.as_str())
                    .map(|s| s.lines().count())
                    .unwrap_or(0);
                format!("✔ bash: exit {code} ({stdout_lines} lines)")
            }
            "read_file" => {
                let lines = output
                    .get("content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.lines().count())
                    .unwrap_or(0);
                format!("✔ read: {lines} lines")
            }
            "write_file" | "edit_file" => {
                format!("✔ {name}: done")
            }
            "glob" | "search_files" => {
                let count = output
                    .get("files")
                    .or_else(|| output.get("matches"))
                    .and_then(|v| v.as_array())
                    .map(|a| a.len())
                    .unwrap_or(0);
                format!("✔ {name}: {count} found")
            }
            _ => {
                format!("✔ {name}: done")
            }
        }
    }
}

/// Handle a `/command`. Returns true when the input was a command (no agent submit).
pub(crate) async fn handle_slash(
    rt: &Arc<LucyRuntime>,
    app: &mut App,
    raw: &str,
    task_tx: &tokio::sync::mpsc::UnboundedSender<TaskEvent>,
) -> bool {
    let text = raw.trim();
    if !text.starts_with('/') {
        return false;
    }
    let mut parts = text.splitn(2, char::is_whitespace);
    let cmd = parts.next().unwrap_or("").to_ascii_lowercase();
    let arg = parts.next().unwrap_or("").trim().to_owned();
    match cmd.as_str() {
        "/new" => {
            match rt
                .new_session(if arg.is_empty() { None } else { Some(arg) })
                .await
            {
                Ok(meta) => {
                    app.session_title = meta.title.clone();
                    app.session_id_short = short_id(&meta.id.0.to_string());
                    app.session_count = meta.message_count;
                    app.load_turns(&rt.history().await);
                    app.push_msg(ChatMsg::system(format!(
                        "New session: {} ({})",
                        app.session_title, app.session_id_short
                    )));
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
            app.status =
                "Sessions — ↑/↓ select · Enter switch · d delete · n new · Esc close".into();
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
                Some(meta) => match rt.switch_session(&meta.id).await {
                    Ok(switched) => {
                        app.session_title = switched.title.clone();
                        app.session_id_short = short_id(&switched.id.0.to_string());
                        app.session_count = switched.message_count;
                        app.load_turns(&rt.history().await);
                        app.push_msg(ChatMsg::system(format!(
                            "Switched to: {}",
                            app.session_title
                        )));
                        app.status = "Ready".into();
                    }
                    Err(e) => app.push_msg(ChatMsg::system(format!("Switch failed: {e}"))),
                },
                None => app.push_msg(ChatMsg::system(format!(
                    "No session matches '{arg}'. Use /sessions to list."
                ))),
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
                app.sessions
                    .first()
                    .cloned()
                    .filter(|m| {
                        // default: current session when no arg
                        format!("{}", m.id.0).starts_with(&app.session_id_short)
                            || m.title == app.session_title
                    })
                    .or_else(|| app.sessions.first().cloned())
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
        "/agent" => {
            if arg.is_empty() {
                app.push_msg(ChatMsg::system("Usage: /agent <goal>".into()));
            } else {
                let rt = rt.clone();
                let task_tx = task_tx.clone();
                let goal = arg.clone();
                app.push_msg(ChatMsg::user(goal.clone()));
                app.pin();
                app.set_phase("Automating", "starting the ReAct loop");
                tokio::spawn(async move {
                    if let Err(e) = rt.save_user_message(goal.clone()).await {
                        let _ = task_tx.send(TaskEvent::SetStatus(format!("Error: {e}")));
                        let _ = task_tx.send(TaskEvent::Finished);
                        return;
                    }
                    let meta = rt.current_meta().await;
                    let _ = task_tx.send(TaskEvent::SetSessionTitle(meta.title));
                    let _ = task_tx.send(TaskEvent::SetSessionCount(meta.message_count));
                    run_goal(&rt, &task_tx, &goal, None).await;
                });
            }
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
        "/mascot" => {
            // A bare `/mascot` cycles; an argument picks one. The sizes are
            // enumerated rather than described, so a new one needs no new text.
            let sizes: Vec<&str> = MascotSize::ALL.iter().map(|s| s.label()).collect();
            let arg = arg.trim();
            let next = if arg.is_empty() {
                app.mascot_size.next()
            } else {
                match MascotSize::parse(arg) {
                    Some(s) => s,
                    None => {
                        app.push_msg(ChatMsg::system(format!(
                            "Unknown size '{arg}'. Try one of: {}.",
                            sizes.join(", ")
                        )));
                        return true;
                    }
                }
            };
            app.mascot_size = next;
            let note = match next {
                MascotSize::Off => "Mascot hidden. /mascot brings her back.".to_owned(),
                other => format!(
                    "Mascot: {}. She is {} right now.",
                    other.label(),
                    app.mascot_mood().label().to_lowercase()
                ),
            };
            app.push_msg(ChatMsg::system(note));
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
            // MCP servers that failed to come up. Previously these were dropped
            // silently, so a broken server looked like "not installed".
            let problems = rt.mcp_problems.clone();
            if !problems.is_empty() {
                out.push_str("\nMCP servers\n\n");
                for p in &problems {
                    out.push_str(&format!("- !! {p}\n"));
                }
            } else {
                out.push_str("\n- ok MCP servers: all configured servers started\n");
            }
            out.push_str(&format!(
                "\n- ok Permissions: {} ({} tool(s) always allowed)",
                rt.config().approval_mode_label(),
                rt.approval_gate().always_allowed_tools().len()
            ));
            app.push_msg(ChatMsg::lucy(out.trim_end().to_owned()));
            true
        }
        "/memory" => {
            let parts = arg.trim().splitn(2, ' ');
            match parts.next().unwrap_or("status") {
                "" | "status" => {
                    let s = rt.memory_hub().stats().await;
                    app.push_msg(ChatMsg::lucy(format!(
                        "Memory Hub — L0 conversation: {} · L1 atoms: {} · L2 scenarios: {} · L3 persona: {} · assets: {}",
                        s.conversation, s.atom, s.scenario, s.persona, s.assets
                    )));
                }
                "search" => {
                    let query = parts.next().unwrap_or("").trim();
                    if query.is_empty() {
                        app.push_msg(ChatMsg::system("Usage: /memory search <query>".into()));
                    } else {
                        let hits = rt.memory_hub().search(query, 8).await;
                        if hits.is_empty() {
                            app.push_msg(ChatMsg::system("Memory: no matches".into()));
                        } else {
                            let mut out = String::from("Lucy › Memory recall\n\n");
                            for hit in hits {
                                out.push_str(&format!("- [{}] {} — {}\n", format!("{:?}", hit.layer).to_lowercase(), hit.title, hit.content));
                            }
                            app.push_msg(ChatMsg::lucy(out.trim_end().to_owned()));
                        }
                    }
                }
                "slim" => {
                    match rt.memory_hub().slim().await {
                        Ok((before, after, removed)) => app.push_msg(ChatMsg::lucy(format!(
                            "Memory slim complete — removed {removed} duplicate atom(s); {} → {} total memory records",
                            before.conversation + before.atom + before.scenario + before.persona,
                            after.conversation + after.atom + after.scenario + after.persona
                        ))),
                        Err(e) => app.push_msg(ChatMsg::system(format!("Memory slim failed: {e}"))),
                    }
                }
                "assets" => {
                    let assets = rt.memory_hub().assets(None).await;
                    if assets.is_empty() {
                        app.push_msg(ChatMsg::system("Memory Hub: no assets registered".into()));
                    } else {
                        let mut out = String::from("Lucy › Memory assets\n\n");
                        for asset in assets {
                            out.push_str(&format!("- [{:?}] {} v{} ({})\n", asset.kind, asset.name, asset.version, asset.visibility));
                        }
                        app.push_msg(ChatMsg::lucy(out.trim_end().to_owned()));
                    }
                }
                other => app.push_msg(ChatMsg::system(format!(
                    "Unknown memory action '{other}' — use /memory [status|search <query>|slim|assets]"
                ))),
            }
            true
        }
        "/wiki-ingest" => {
            if arg.trim().is_empty() {
                app.push_msg(ChatMsg::system("Usage: /wiki-ingest <file.md>".into()));
            } else {
                match rt.memory_hub().ingest_wiki(std::path::PathBuf::from(arg.trim())).await {
                    Ok(n) => app.push_msg(ChatMsg::lucy(format!("Wiki indexed — {n} section(s) added to Memory Hub"))),
                    Err(e) => app.push_msg(ChatMsg::system(format!("Wiki ingest failed: {e}"))),
                }
            }
            true
        }
        "/codegraph" => {
            if arg.trim().is_empty() {
                app.push_msg(ChatMsg::system("Usage: /codegraph <file.rs>".into()));
            } else {
                match rt.memory_hub().index_rust_file(std::path::PathBuf::from(arg.trim())).await {
                    Ok(n) => app.push_msg(ChatMsg::lucy(format!("CodeGraph indexed — {n} Rust symbol(s)"))),
                    Err(e) => app.push_msg(ChatMsg::system(format!("CodeGraph indexing failed: {e}"))),
                }
            }
            true
        }
        "/knowledge" => {
            // The browseable surface of the knowledge base: the generated index
            // plus where the files live, because a store a human cannot edit is
            // a store nobody will ever correct.
            app.push_msg(ChatMsg::lucy(
                lucy_runtime::knowledge::digest_text(rt).await,
            ));
            true
        }
        "/forget" => {
            if arg.is_empty() {
                app.push_msg(ChatMsg::system(
                    "Usage: /forget <topic-slug> — see /knowledge for the slugs".into(),
                ));
            } else {
                // Tombstone, never delete: the row stays for the audit trail and
                // a rebuilt index from the files cannot resurrect it.
                match lucy_runtime::knowledge::forget(rt, &arg).await {
                    Ok(0) => app.push_msg(ChatMsg::system(format!(
                        "No live knowledge under '{arg}' — nothing forgotten"
                    ))),
                    Ok(n) => app.push_msg(ChatMsg::system(format!(
                        "Forgot {n} claim(s) under '{arg}' — it will not be injected again"
                    ))),
                    Err(e) => app.push_msg(ChatMsg::system(format!("Forget failed: {e}"))),
                }
            }
            true
        }
        "/kb-dream" => {
            // The consolidation pass, run on demand before it is ever automatic.
            // Promotion is recall-weighted and closed to untrusted origins, so
            // this cannot promote anything a page wrote.
            let report = lucy_runtime::knowledge::promote_recalled(rt).await;
            let rebuilt = lucy_runtime::knowledge::reindex(rt).await;
            let (promoted, considered, refused) = match &report {
                Ok(r) => (r.promoted.len(), r.considered, r.refused_untrusted),
                Err(e) => {
                    app.push_msg(ChatMsg::system(format!(
                        "Consolidation could not run: {}",
                        lucy_core::friendly(&format!("{e:#}"))
                    )));
                    return true;
                }
            };
            app.push_msg(ChatMsg::lucy(format!(
                "Knowledge consolidated — promoted {promoted} of {considered} candidate(s), \
                 refused {refused} untrusted, reindexed {} claim(s) from disk",
                rebuilt.unwrap_or(0)
            )));
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
            app.push_msg(ChatMsg::lucy(HELP_TEXT.to_owned()));
            true
        }
        "/auto" => {
            handle_auto(rt, app, &arg).await;
            true
        }
        "/stop" => {
            stop_run(rt, app);
            true
        }
        "/serve" => {
            handle_serve(app, &arg);
            true
        }
        "/settings" | "/config" => {
            app.settings = true;
            app.settings_state = super::settings::SettingsState::open(&app.config);
            true
        }
        "/companion" => {
            handle_companion(app, &arg);
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

/// `/auto [on|off|status]` — turn automode on or off, or report the current
/// setting. The approval dialog has advertised this command in its footer for
/// as long as it has existed, with no handler behind it: typing `/auto on`
/// fell through to the catch-all and printed "Unknown command".
async fn handle_auto(rt: &Arc<LucyRuntime>, app: &mut App, arg: &str) {
    let verb = arg.trim().to_ascii_lowercase();
    let current = rt.approval_gate().current_mode();
    let always = rt.approval_gate().always_allowed_tools().len();

    let set = match verb.as_str() {
        "" | "status" | "show" => {
            let mode_line = format!(
                "automode {} — Lucy {}",
                if current == "never" { "on" } else { "off" },
                match current.as_str() {
                    "never" => "will not ask for permission",
                    "always" => "asks before every tool",
                    _ => "asks only before risky tools",
                }
            );
            app.push_msg(ChatMsg::lucy(format!(
                "Lucy › {mode_line}\n\n/auto on — run without asking\n/auto off — ask before risky tools\n/auto always — ask before every tool\n{always} tool(s) permanently allowed"
            )));
            app.pin();
            return;
        }
        "on" | "never" => "never",
        "off" | "write" => "write",
        "always" => "always",
        other => {
            app.push_msg(ChatMsg::system(format!(
                "Unknown /auto option '{other}' — use on, off, always, or status"
            )));
            return;
        }
    };

    match rt.set_approval_mode(set) {
        Ok(mode) => {
            let msg = match mode.as_str() {
                "never" => "automode on — Lucy will run without asking for permission.\nStop it any time with /stop, Esc, or Ctrl+C.".to_owned(),
                "always" => "automode off — Lucy will ask before every tool.".to_owned(),
                _ => "automode off — Lucy will ask before risky tools.".to_owned(),
            };
            app.push_msg(ChatMsg::lucy(msg));
            app.pin();
            app.status = format!("Approval mode: {mode}");
        }
        Err(e) => {
            app.push_msg(ChatMsg::system(format!(
                "Could not change approval mode: {e}"
            )));
        }
    }
}

/// `/serve [on|off|status]` — the mobile bridge switch.
///
/// Turning it on here only flips the persistent switch and tells the user what
/// to run next; the bridge itself is a separate foreground process (`lucy
/// serve`) so its port, its log, and its Ctrl-C belong to the user rather than
/// to a background child of the TUI. That is also why an idle bridge costs
/// nothing: it is not running at all until someone starts it.
fn handle_serve(app: &mut App, arg: &str) {
    let verb = arg.trim().to_ascii_lowercase();
    let mut cfg = app.config.clone();
    match verb.as_str() {
        "on" => {
            cfg.set_gateway_enabled(true);
            match cfg.save() {
                Ok(()) => app.push_msg(ChatMsg::lucy(
                    "Lucy › Mobile bridge on.\n\nStart it in a terminal with:  lucy serve --pair\n\n\
                     It listens on {}:{} and stays off again the moment that process ends.".into(),
                )),
                Err(e) => {
                    app.push_msg(ChatMsg::system(format!("Could not save the gateway switch: {e}")))
                }
            }
        }
        "off" => {
            cfg.set_gateway_enabled(false);
            match cfg.save() {
                Ok(()) => app.push_msg(ChatMsg::lucy(
                    "Lucy › Mobile bridge off. Stop any running `lucy serve` with Ctrl-C.".into(),
                )),
                Err(e) => app.push_msg(ChatMsg::system(format!(
                    "Could not save the gateway switch: {e}"
                ))),
            }
        }
        "" | "status" | "show" => {
            let line = format!(
                "Mobile bridge: {} · {}:{} · idle unload {}s (0 = keep the runtime alive)",
                if cfg.gateway_enabled() { "on" } else { "off" },
                cfg.gateway.bind,
                cfg.gateway.port,
                cfg.gateway.idle_unload_secs
            );
            let note = if cfg.gateway_enabled() {
                "\n\nRun `lucy serve --pair` to let a phone pair."
            } else {
                "\n\nRun `/serve on` to allow it."
            };
            app.push_msg(ChatMsg::lucy(format!("Lucy › {line}{note}")));
        }
        other => {
            app.status = format!("Unknown /serve option '{other}' — use on, off, or status");
        }
    }
}

/// `/companion [show|hide|toggle|<note>]` — the desktop companion overlay.
/// A bare `/companion` toggles; anything else is shown as the panel's note.
fn handle_companion(app: &mut App, arg: &str) {
    match arg.trim().to_ascii_lowercase().as_str() {
        "" | "toggle" => {
            app.companion.toggle();
            app.status = if app.companion.is_visible() {
                "Companion shown".into()
            } else {
                "Companion hidden".into()
            };
        }
        "show" | "on" => {
            app.companion.show();
            app.status = "Companion shown".into();
        }
        "hide" | "off" => {
            app.companion.hide();
            app.status = "Companion hidden".into();
        }
        _ => {
            app.companion.handle_message(arg.trim());
            app.status = "Companion shown".into();
        }
    }
}

/// `/stop` — the kill switch. Fires the interrupt, drops any pending approval
/// dialog, and denies everything still queued so a blocked `ask()` returns
/// instead of waiting out its 300s timeout.
pub(crate) fn stop_run(rt: &Arc<LucyRuntime>, app: &mut App) {
    rt.interrupt();
    if let Some(dlg) = app.approval.take() {
        rt.approval_gate().resolve(&dlg.id, ApprovalDecision::Deny);
    }
    // Anything else still waiting on a prompt is denied too: the loop is being
    // stopped, so a later prompt must not block shutdown.
    for id in rt.approval_gate().pending_ids() {
        rt.approval_gate().resolve(&id, ApprovalDecision::Deny);
    }
    app.set_phase("Stopped", "cancelled by /stop");
    app.status = "Stopped — interrupt sent".into();
}

/// Capture the turn's durable knowledge in the background.
///
/// Detached on purpose. Capture costs one model call, and the turn is already
/// answered and on screen — the alternative is making the user watch a cursor
/// for a memory write. Nothing here can fail a turn: every error is dropped,
/// because a knowledge base that breaks conversation is worse than one that
/// quietly learns nothing.
pub fn capture_in_background(rt: Arc<LucyRuntime>, request: String, reply: String) {
    tokio::spawn(async move {
        match lucy_runtime::knowledge::capture_turn(&rt, &request, &reply).await {
            Ok(claims) if !claims.is_empty() => {
                tracing::info!(count = claims.len(), "captured durable knowledge");
            }
            Ok(_) => {}
            Err(e) => tracing::warn!(error = %format!("{e:#}"), "knowledge capture skipped"),
        }
    });
}

/// Slash handling when no runtime is available (degraded mode).
fn handle_slash_offline(app: &mut App, raw: &str) {
    // No command here takes an argument, so only the verb is split off. The
    // one command that used to (`/model <name>`) needs a runtime to resolve a
    // provider, which is exactly what degraded mode lacks.
    let cmd = raw
        .trim()
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    match cmd.as_str() {
        "/help" | "/?" => {
            app.push_msg(ChatMsg::lucy(HELP_TEXT.to_owned()));
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
        "/compact" => {
            app.status = "Setup required before compact works.".into();
        }
        "/agent" => {
            app.status = "Setup required before /agent works.".into();
        }
        "/usage" => {
            app.push_msg(ChatMsg::lucy(
                "Lucy › Usage — prompt: 0 tokens, completion: 0 tokens, total: 0 tokens".to_owned(),
            ));
        }
        "/knowledge" | "/forget" | "/kb-dream" => {
            app.push_msg(ChatMsg::system(format!(
                "Setup required before {cmd} works."
            )));
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
        "/auto" => {
            app.push_msg(ChatMsg::system("Setup required before /auto works.".into()));
        }
        "/stop" => {
            app.set_phase("Stopped", "no active run");
            app.status = "Nothing to stop".into();
        }
        "/settings" | "/config" => {
            app.settings = true;
            app.settings_state = super::settings::SettingsState::open(&app.config);
        }
        "/companion" => {
            handle_companion(app, raw.splitn(2, char::is_whitespace).nth(1).unwrap_or(""));
        }
        _ => {
            app.status = format!("Unknown command {cmd} — try /help");
        }
    }
}

pub(crate) async fn refresh_sessions(rt: &LucyRuntime, app: &mut App) {
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

/// Events the spawned turn task pushes back to the main loop.
#[derive(Debug)]
pub(crate) enum TaskEvent {
    PushMsg(ChatMsg),
    SetPhase {
        phase: String,
        detail: String,
    },
    SetStatus(String),
    SetSessionTitle(String),
    SetSessionCount(usize),
    /// A planned command needs approval before it runs.
    ApprovalRequest {
        id: String,
        name: String,
        input: String,
    },
    /// The `model · tier` line shown in the chat header.
    SetActiveModel(String),
    Finished,
}

/// Translate a runtime `AgentEvent` into UI updates. Returns false for events
/// the chat log has no use for.
fn pump_event(
    rt: &Arc<LucyRuntime>,
    event: lucy_core::AgentEvent,
    task_tx: &tokio::sync::mpsc::UnboundedSender<TaskEvent>,
) -> bool {
    match event {
        lucy_core::AgentEvent::Progress { message } => {
            let _ = task_tx.send(TaskEvent::PushMsg(ChatMsg::tool(message.clone())));
            let _ = task_tx.send(TaskEvent::SetPhase {
                phase: "Automating".into(),
                detail: message,
            });
        }
        lucy_core::AgentEvent::Status { message } => {
            let _ = task_tx.send(TaskEvent::PushMsg(ChatMsg::system(message.clone())));
            let _ = task_tx.send(TaskEvent::SetPhase {
                phase: "Automating".into(),
                detail: message,
            });
        }
        lucy_core::AgentEvent::Error { message } => {
            // Rendered through the shared classifier: this event carries provider
            // bodies and MCP envelopes verbatim, and the chat log is the one place
            // a reader is guaranteed to be looking.
            let _ = task_tx.send(TaskEvent::PushMsg(ChatMsg::system(format!(
                "✖ {}",
                lucy_core::friendly(&message)
            ))));
            let _ = task_tx.send(TaskEvent::SetPhase {
                phase: "Automating".into(),
                detail: lucy_core::friendly(&message),
            });
        }
        lucy_core::AgentEvent::ApprovalRequest { id, name, input } => {
            let _ = task_tx.send(TaskEvent::ApprovalRequest {
                id,
                name,
                input: truncate_one_line(&input.to_string(), 300),
            });
        }
        lucy_core::AgentEvent::Done => return false,
        _ => return false,
    }
    let _ = rt;
    true
}

/// Run a goal to completion: spawn the runner, stream its events into the chat
/// log, save the outcome, then compact once the history outgrows the ceiling.
///
/// One entry, [`execute_goal_outcome`], for every act — `/agent` and a plain
/// prompt that needs actions reach the same ReAct loop, so the two cannot
/// drift. `route` only supplies the model label for the run's status line.
async fn run_goal(
    rt: &Arc<LucyRuntime>,
    task_tx: &tokio::sync::mpsc::UnboundedSender<TaskEvent>,
    goal: &str,
    route: Option<&lucy_runtime::TurnRoute>,
) {
    let _ = task_tx.send(TaskEvent::SetPhase {
        phase: "Automating".into(),
        detail: "running the ReAct loop".into(),
    });
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<lucy_core::AgentEvent>();
    let rt_clone = rt.clone();
    let goal = goal.to_owned();
    let route = route.cloned();
    let run = tokio::spawn(async move {
        rt_clone
            .execute_goal_outcome(&goal, route.as_ref(), Some(tx))
            .await
    });

    while let Some(event) = rx.recv().await {
        pump_event(rt, event, task_tx);
    }
    let outcome = run.await;
    let final_msg = match outcome {
        // A run that verified every objective is a completion; anything else
        // gets a warning mark, because the tick is what the user reads.
        Ok(Ok(outcome)) if outcome.complete => format!("✔ {}", outcome.summary),
        Ok(Ok(outcome)) => format!("⚠ {}", outcome.summary),
        Ok(Err(e)) => format!("✖ Goal not completed: {}", lucy_core::friendly(&format!("{e:#}"))),
        Err(e) => format!("✖ Task failed: {}", lucy_core::friendly(&format!("{e:#}"))),
    };
    let _ = rt.save_assistant_text(final_msg.clone()).await;
    let _ = task_tx.send(TaskEvent::PushMsg(ChatMsg::lucy(final_msg)));
    let count = rt.current_meta().await.message_count;
    let _ = task_tx.send(TaskEvent::SetSessionCount(count));
    let _ = task_tx.send(TaskEvent::SetStatus("Ready".into()));
    // Close the turn here, while the reply is still the last thing on screen.
    // Compaction below is background housekeeping: folding it into the turn
    // would both inflate the time the user is asking about and push the reply
    // out from under its own timing row.
    let _ = task_tx.send(TaskEvent::Finished);

    // auto_compact: summarize the history once it outgrows the ceiling.
    match rt.auto_compact_if_needed().await {
        Ok(Some(note)) => {
            let _ = task_tx.send(TaskEvent::PushMsg(ChatMsg::system(note)));
        }
        Ok(None) => {}
        Err(e) => {
            let _ = task_tx.send(TaskEvent::PushMsg(ChatMsg::system(format!(
                "auto-compact failed: {}",
                lucy_core::friendly(&format!("{e:#}"))
            ))));
        }
    }
}

/// One user turn, end to end: classify → branch → reply or sequential actions.
pub(crate) async fn submit_prompt(
    rt: Option<&Arc<LucyRuntime>>,
    app: &mut App,
    task_tx: &tokio::sync::mpsc::UnboundedSender<TaskEvent>,
    text: String,
) {
    // Slash commands are handled locally.
    if text.starts_with('/') {
        if let Some(rt) = rt {
            let was_quit = text.trim().to_ascii_lowercase() == "/quit"
                || text.trim().to_ascii_lowercase() == "/exit"
                || text.trim().to_ascii_lowercase() == "/q";
            handle_slash(rt, app, &text, task_tx).await;
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
    let Some(rt) = rt.cloned() else {
        app.status =
            "Setup required: connect a provider in /settings, then press [Test] and [Save]".into();
        return;
    };
    let rt = rt.clone();
    let task_tx = task_tx.clone();
    // Synchronous instant feedback: spinner + welcome→active switch paint
    // on the very next frame (the caller also forces an immediate draw).
    app.set_phase("Thinking", "saving your message");
    tokio::spawn(async move {
        if let Err(e) = rt.save_user_message(text.clone()).await {
            let _ = task_tx.send(TaskEvent::SetStatus(format!("Error: {e}")));
            let _ = task_tx.send(TaskEvent::Finished);
            return;
        }
        let meta = rt.current_meta().await;
        let _ = task_tx.send(TaskEvent::SetSessionTitle(meta.title));
        let _ = task_tx.send(TaskEvent::SetSessionCount(meta.message_count));

        // Step 1 — one forward pass against the classification model.
        let _ = task_tx.send(TaskEvent::SetPhase {
            phase: "Thinking".into(),
            detail: "classifying intent + reasoning level".into(),
        });
        let classification = match rt.classify_turn(&text).await {
            Ok(c) => c,
            // No routing verdict: the decider is unavailable/undecided and no
            // tools may run. Surface the error in both status bar and chat
            // log, persist it, and stop the turn here.
            Err(e) => {
                let msg = format!("✖ {e}");
                let _ = task_tx.send(TaskEvent::SetStatus(msg.clone()));
                let _ = task_tx.send(TaskEvent::PushMsg(ChatMsg::system(msg.clone())));
                let _ = rt.save_assistant_text(msg).await;
                let _ = task_tx.send(TaskEvent::Finished);
                return;
            }
        };
        // What the classifier thinks this turn will need from the library. Purely
        // advisory: it reorders knowledge recall and cannot decide whether recall
        // happens, because a 2B classifier choosing what to retrieve unaided is
        // exactly the failure the small-model retrieval ablations describe.
        let knowledge_hint = classification.knowledge_topic.clone();
        if let Some(note) = classification.summary_note.clone() {
            // Routing second-opinion note (e.g. the LLM verify overrode an
            // under-confident classifier): say so instead of hiding the
            // override.
            let _ = task_tx.send(TaskEvent::PushMsg(ChatMsg::system(format!("⚠ {note}"))));
        }
        let _ = task_tx.send(TaskEvent::PushMsg(ChatMsg::system(
            classification.summary(),
        )));

        // Step 2 dispatch. Err means no routing verdict (decider
        // unavailable/undecided): surface it in both status bar and chat log,
        // persist it, and stop the turn — no tools run on an unrouted turn.
        let route = match rt.route_turn_with(&text, Some(classification)).await {
            Ok(r) => r,
            Err(e) => {
                let msg = format!("✖ {e}");
                let _ = task_tx.send(TaskEvent::SetStatus(msg.clone()));
                let _ = task_tx.send(TaskEvent::PushMsg(ChatMsg::system(msg.clone())));
                let _ = rt.save_assistant_text(msg).await;
                let _ = task_tx.send(TaskEvent::Finished);
                return;
            }
        };
        let _ = task_tx.send(TaskEvent::SetActiveModel(route.active_model_line()));
        if let Some(note) = route.note.clone() {
            let _ = task_tx.send(TaskEvent::PushMsg(ChatMsg::system(format!("⚠ {note}"))));
        }

        if !route.needs_actions() {
            // Step 2A — text-only: answer with the routed tier's model.
            let _ = task_tx.send(TaskEvent::SetPhase {
                phase: "Writing".into(),
                detail: route.active_model_line(),
            });
            match rt
                .answer_turn_with_knowledge(&text, &route, knowledge_hint.as_deref())
                .await
            {
                Ok(reply) => {
                    let _ = rt.save_assistant_text(reply.clone()).await;
                    let _ = task_tx.send(TaskEvent::PushMsg(ChatMsg::lucy(reply.clone())));
                    // Capture runs *after* the reply is on screen and in its own
                    // task, so the user never waits for a memory-extraction call.
                    capture_in_background(rt.clone(), text.clone(), reply.clone());
                    let count = rt.current_meta().await.message_count;
                    let _ = task_tx.send(TaskEvent::SetSessionCount(count));
                    let _ = task_tx.send(TaskEvent::SetStatus("Ready".into()));
                }
                Err(e) => {
                    let friendly = friendly_error(&e.to_string());
                    let _ = rt.save_assistant_text(friendly.clone()).await;
                    let _ = task_tx.send(TaskEvent::PushMsg(ChatMsg::lucy(friendly)));
                    let count = rt.current_meta().await.message_count;
                    let _ = task_tx.send(TaskEvent::SetSessionCount(count));
                    let _ = task_tx.send(TaskEvent::SetStatus("Error".into()));
                }
            }
            let _ = task_tx.send(TaskEvent::Finished);
            return;
        }

        // Step 2B: the goal needs actions. One ReAct loop, the same one
        // `/agent` runs.
        run_goal(&rt, &task_tx, &text, Some(&route)).await;
    });
}

/// One reader-facing line for a failed turn.
///
/// `lucy_core::friendly` is the single classifier (JSON bodies, HTTP status
/// classes, transport failures → a sentence plus a next step), so this only adds
/// the voice prefix. Before it existed each surface grew its own keyword list —
/// this one knew about 401s and nothing else, so a 429 or a DNS failure reached
/// the chat as the provider's raw body.
fn friendly_error(raw: &str) -> String {
    format!("Lucy › {}", lucy_core::friendly(raw))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucy_core::{SessionData, SessionId};
    #[test]
    fn bare_slash_lists_every_command() {
        let hits = command_matches("/");
        assert_eq!(hits.len(), COMMANDS.len(), "bare / must list all commands");
        assert!(hits.iter().any(|(c, _)| *c == "/settings"));
    }

    #[test]
    fn prefix_matches_then_falls_back_to_substring() {
        let names = |q: &str| {
            command_matches(q)
                .into_iter()
                .map(|(c, _)| c)
                .collect::<Vec<_>>()
        };
        let prefix = names("/se");
        assert!(prefix.contains(&"/settings"), "{prefix:?}");
        assert!(prefix.contains(&"/sessions"), "{prefix:?}");
        // No bare `/new` in a `/se` query: prefix stage filtered it out and
        // the substring stage only runs when the prefix stage found nothing.
        assert!(!prefix.contains(&"/new"), "{prefix:?}");

        let fuzzy = names("/ession");
        assert_eq!(fuzzy, vec!["/sessions"]);

        // Arguments end the suggestion list (nothing to complete).
        assert!(command_matches("/settings foo").is_empty());
        assert!(command_matches("hello").is_empty());
        assert!(command_matches("/zzz").is_empty());
    }

    #[test]
    fn the_automode_and_kill_switch_commands_are_discoverable() {
        // `/auto` was advertised in the approval dialog footer for as long as
        // that footer existed, with no handler behind it.
        let auto = command_matches("/au");
        assert_eq!(auto.len(), 1, "{auto:?}");
        assert_eq!(auto[0].0, "/auto");

        let stop = command_matches("/st");
        let stop_names: Vec<&str> = stop.iter().map(|(c, _)| *c).collect();
        assert!(stop_names.contains(&"/stop"), "{stop_names:?}");

        // Both are in the flat help list the F2 popup renders.
        assert!(COMMANDS.iter().any(|(c, _)| *c == "/auto"));
        assert!(COMMANDS.iter().any(|(c, _)| *c == "/stop"));
    }

    #[test]
    fn agent_slash_is_discoverable() {
        let hits = command_matches("/ag");
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].0, "/agent");
        assert!(hits[0].1.contains("<goal>"), "{}", hits[0].1);
    }

    #[test]
    fn the_mobile_bridge_switch_is_discoverable() {
        // `/serve` is how the bridge is turned on and off, so it must be as
        // findable as `/auto` and `/stop`.
        let hits = command_matches("/se");
        assert!(hits.iter().any(|(c, _)| *c == "/serve"), "{hits:?}");
        let names = COMMANDS.iter().any(|(c, _)| *c == "/serve");
        assert!(names, "/serve must be in the flat help list");
    }

    #[test]
    fn a_failed_turn_never_shows_the_provider_payload() {
        // The regression this whole path exists for: a 429 with a JSON body used
        // to reach the chat verbatim, because only 401-shaped text was rewritten.
        let raw = r#"google/gemini-2.5-flash returned HTTP 429 Too Many Requests: {"error":{"message":"Rate limit reached for gemini-2.5-flash","type":"rate_limit_error"}}"#;
        let line = friendly_error(raw);
        assert!(!line.contains('{'), "{line}");
        assert!(line.contains("Rate limit reached"), "{line}");
        assert!(line.contains("/settings"), "no next step: {line}");
    }

    #[test]
    fn a_transport_failure_in_a_turn_gets_its_own_remedy() {
        let line = friendly_error("error sending request: tcp connect error: Connection refused");
        assert!(line.contains("Lucy ›"), "{line}");
        assert!(line.contains("start the local server"), "{line}");
    }

    #[test]
    fn resolves_session_arg() {
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
