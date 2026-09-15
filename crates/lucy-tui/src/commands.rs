//! Slash commands (opencode-style): dispatch, session management, submit.
//!
//! `handle_slash` never touches the agent — commands run locally against
//! `LucyRuntime`. Plain text goes to the hierarchical loop via `submit_text`.

use std::sync::Arc;

use tokio::sync::mpsc;

use lucy_core::{AgentEvent, SessionMeta};
use lucy_runtime::LucyRuntime;

use super::model::{App, ChatMsg};
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
    ("/history", "show input history"),
    ("/status", "show session + runtime status"),
    ("/model", "show current model"),
    ("/auto", "auto-approve permissions — /auto [on|off]"),
    (
        "/approvals",
        "permission mode — /approvals [never|write|always]",
    ),
    ("/export", "export session — /export <file.json>"),
    ("/help", "show help (F1)"),
    ("/settings", "open settings (Ctrl+,)"),
    ("/quit", "quit lucy"),
];

/// Human-readable one-line label for a tool call start (`bash: …`, `read: …`).
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

pub(crate) fn command_hint(input: &str) -> Option<String> {
    let q = input.trim();
    if !q.starts_with('/') || q.contains(' ') {
        return None;
    }
    let mut hits: Vec<&&str> = COMMANDS
        .iter()
        .map(|(c, _)| c)
        .filter(|c| c.starts_with(q))
        .collect();
    if hits.is_empty() {
        // fuzzy: contains
        hits = COMMANDS
            .iter()
            .map(|(c, _)| c)
            .filter(|c| c.contains(&q[1..]))
            .collect();
    }
    if hits.is_empty() {
        None
    } else {
        Some(
            hits.into_iter()
                .take(5)
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("   "),
        )
    }
}

/// Handle a `/command`. Returns true when the input was a command (no agent submit).
pub(crate) async fn handle_slash(rt: &LucyRuntime, app: &mut App, raw: &str) -> bool {
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
            let mode = rt.approval_mode();
            let mode_note = match mode.as_str() {
                "never" => "auto (no permission popups — /auto off to re-enable)",
                "always" => "paranoid (asks for every tool)",
                _ => "default (asks for state-changing tools — /auto on for hands-free)",
            };
            app.push_msg(ChatMsg::system(format!(
                "Session: {} ({})\nMessages: {} · Model: {} · Dir: {}\nPermissions: {} ({mode_note})\n{usage_note}",
                meta.title,
                short_id(&meta.id.0.to_string()),
                meta.message_count,
                app.model_label,
                rt.sessions_dir().display(),
                mode,
            )));
            true
        }
        "/auto" => {
            // Hands-free switch the user asked for: `/auto` toggles,
            // `/auto on` enables (no permission popups), `/auto off` disables.
            let want = match arg.trim().to_ascii_lowercase().as_str() {
                "" => None,
                "on" | "enable" | "enabled" | "1" | "true" | "yes" | "y" => Some(true),
                "off" | "disable" | "disabled" | "0" | "false" | "no" | "n" => Some(false),
                other => {
                    app.push_msg(ChatMsg::system(format!(
                        "Usage: /auto [on|off] — unknown arg '{other}'"
                    )));
                    return true;
                }
            };
            let target = want.unwrap_or_else(|| !rt.is_auto());
            match rt.set_auto_approve(target) {
                Ok(mode) => {
                    app.config.approvals.mode = mode.clone();
                    if target {
                        app.push_msg(ChatMsg::lucy("Lucy › Auto-approve ON — I will run every step without asking (browser, shell, files). Use /auto off to re-enable permission popups.".into()));
                        app.status = "Auto-approve: ON (never ask)".into();
                    } else {
                        app.push_msg(ChatMsg::lucy(format!("Lucy › Auto-approve OFF — permission mode back to '{mode}' (I will ask before state-changing tools).")));
                        app.status = format!("Auto-approve: OFF ({mode})");
                    }
                }
                Err(e) => app.push_msg(ChatMsg::system(format!("Auto switch failed: {e}"))),
            }
            true
        }
        "/approvals" | "/permissions" | "/permission" => {
            // Full permission-mode control: show or set never|write|always.
            if arg.is_empty() {
                let mode = rt.approval_mode();
                app.push_msg(ChatMsg::lucy(format!(
                    "Lucy › Permissions: {mode}\n\n- never (auto) — run everything, never ask — /auto on\n- write (default) — ask before state-changing tools\n- always — ask before every tool\n\nSet with /approvals <never|write|always> or /auto [on|off]"
                )));
                return true;
            }
            match rt.set_approval_mode(&arg) {
                Ok(mode) => {
                    app.config.approvals.mode = mode.clone();
                    app.push_msg(ChatMsg::lucy(format!(
                        "Lucy › Permissions set to '{mode}'{}",
                        if mode == "never" {
                            " — auto-approve ON, I will not ask again"
                        } else {
                            ""
                        }
                    )));
                    app.status = format!("Permissions: {mode}");
                }
                Err(e) => app.push_msg(ChatMsg::system(format!(
                    "{e} — try /approvals never|write|always"
                ))),
            }
            true
        }
        "/model" => {
            if arg.is_empty() {
                let cur = if app.model_label.is_empty() {
                    "(unknown)".to_owned()
                } else {
                    app.model_label.clone()
                };
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
                "Lucy › Available commands\n\n- /help — show this help\n- /clear — clear session history\n- /model [name] — show or set model\n- /auto [on|off] — auto-approve permissions (hands-free computer control)\n- /approvals [never|write|always] — permission mode\n- /compact — trim history into a summary\n- /usage — show token usage\n- /doctor — run config checks\n\nKeys\n\n- F2 hold-to-talk voice\n- PgUp/PgDn scroll\n- Esc cancel/quit".to_owned(),
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

/// Slash handling when no runtime is available (degraded mode).
fn handle_slash_offline(app: &mut App, raw: &str) {
    let text = raw.trim();
    let mut parts = text.splitn(2, char::is_whitespace);
    let cmd = parts.next().unwrap_or("").to_ascii_lowercase();
    let arg = parts.next().unwrap_or("").trim().to_owned();
    match cmd.as_str() {
        "/help" | "/?" => {
            app.push_msg(ChatMsg::lucy(
                "Lucy › Available commands\n\n- /help — show this help\n- /clear — clear session history\n- /model [name] — show or set model\n- /auto [on|off] — auto-approve permissions\n- /approvals [never|write|always] — permission mode\n- /compact — trim history into a summary\n- /usage — show token usage\n- /doctor — run config checks\n\nKeys\n\n- F2 hold-to-talk voice\n- PgUp/PgDn scroll\n- Esc cancel/quit".to_owned(),
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
                let cur = if app.model_label.is_empty() {
                    "(unknown)".to_owned()
                } else {
                    app.model_label.clone()
                };
                app.push_msg(ChatMsg::lucy(format!("Lucy › Current model: {cur}")));
            } else {
                app.status = "Setup required before changing model.".into();
            }
        }
        "/compact" => {
            app.status = "Setup required before compact works.".into();
        }
        "/auto" | "/approvals" | "/permissions" | "/permission" => {
            app.status = "Setup required before changing permissions.".into();
        }
        "/usage" => {
            app.push_msg(ChatMsg::lucy(
                "Lucy › Usage — prompt: 0 tokens, completion: 0 tokens, total: 0 tokens".to_owned(),
            ));
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

pub(crate) async fn submit_text(
    rt: Option<&Arc<LucyRuntime>>,
    app: &mut App,
    agent_rx: &mut Option<mpsc::UnboundedReceiver<AgentEvent>>,
    text: String,
) {
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
    if let Some(rt) = rt {
        // Interrupt any running turn before starting a new one (opencode-style).
        if agent_rx.is_some() {
            rt.interrupt();
            *agent_rx = None;
            app.flush_stream();
        }
        // Instant feedback — the spinner row renders on the very next frame,
        // before any LLM event arrives.
        app.set_phase("Thinking", "sending to model");
        match rt.submit(text).await {
            Ok(rx) => *agent_rx = Some(rx),
            Err(e) => {
                app.mark_idle();
                app.status = format!("Error: {e}");
            }
        }
    } else {
        app.status =
            "Setup required: LLM API Key missing — open Settings (Ctrl+,) set keys or run: lucy config doctor"
                .into();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucy_core::{SessionData, SessionId};
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
