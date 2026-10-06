//! Lucy TUI: terminal chat interface over [`LucyRuntime`].
//!
//! Module layout:
//! - `model` — chat state (`App`, `ChatMsg`, streaming, activity phases).
//! - `commands` — slash commands, session actions, submit dispatch.
//! - `views` — all rendering (welcome/active screens, popups, input).
//! - `voice` — push-to-talk hotkey and hold-to-talk recording.
//! - `util` — text truncation, token formatting, cursor math.
//! - `settings` — settings overlay.
//!
//! This file only owns the main event loop: agent events, voice results,
//! keyboard input, and terminal setup/teardown.

mod commands;
mod companion;
mod model;
mod settings;
pub mod screen_ctx;
mod util;
mod views;
mod voice;

use std::io::{self, Stdout};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use lucy_config::{LucyConfig, ReasoningLevel};
use lucy_runtime::LucyRuntime;
use lucy_stt::GroqStt;
use ratatui::{Terminal, backend::CrosstermBackend};
use tokio::sync::mpsc;

use commands::{TaskEvent, refresh_sessions, submit_prompt};
use lucy_core::ApprovalDecision;
use model::{App, ChatMsg};
use util::{char_byte_idx, line_end_cursor, line_home_cursor, short_id};
use views::draw;
use voice::{Recording, is_voice_hotkey, stop_recording};

fn cleanup(t: &mut Terminal<CrosstermBackend<Stdout>>) -> anyhow::Result<()> {
    disable_raw_mode()?;
    execute!(t.backend_mut(), LeaveAlternateScreen)?;
    t.show_cursor()?;
    Ok(())
}

pub async fn run_voice(stt: Option<Arc<GroqStt>>) -> anyhow::Result<()> {
    let config = LucyConfig::load()?;
    // Try to create runtime but allow degraded mode if API key is missing (any brand)
    let (runtime_opt, runtime_error): (Option<Arc<LucyRuntime>>, Option<String>) =
        match LucyRuntime::new().await {
            Ok(r) => (Some(Arc::new(r)), None),
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("API key")
                    || msg.contains("OPENAI_API_KEY")
                    || msg.contains("ANTHROPIC_API_KEY")
                {
                    (None, Some(msg))
                } else {
                    (None, Some(format!("{msg} (run: lucy config doctor)")))
                }
            }
        };
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    // Taste the terminal for Kitty/Sixel before any events are read, so the
    // mascot can render full-detail pixels where it matters.
    lucy_mascot::init_image_support();
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let mut app = App::new(config);
    app.model_label = runtime_opt
        .as_ref()
        .map(|r| r.config().resolve_level_model(ReasoningLevel::L3))
        .unwrap_or_else(|| app.config.resolve_level_model(ReasoningLevel::L3));
    // Hydrate current session into the new chat model.
    if let Some(rt) = runtime_opt.as_ref() {
        let meta = rt.current_meta().await;
        app.session_title = meta.title.clone();
        app.session_id_short = short_id(&meta.id.0.to_string());
        app.session_count = meta.message_count;
        // The resumed transcript stays in the runtime as the agent's context,
        // but it is not preloaded into the view buffer — a returning user gets
        // the welcome screen too, and can open the old transcript via /sessions.
        if app.messages.is_empty() {
            app.status = "Ready — type /help for commands".into();
        }
        // Report servers that failed to start instead of letting them vanish.
        if !rt.mcp_problems.is_empty() {
            let n = rt.mcp_problems.len();
            app.push_msg(ChatMsg::system(format!(
                "{n} MCP server(s) failed to start — /doctor for details"
            )));
        }
        if rt.config().automode() {
            app.push_msg(ChatMsg::system(
                "Automode is on — Lucy runs without asking. /auto off to re-enable prompts, \
                 /stop or Ctrl+C to halt a run."
                    .into(),
            ));
        }
    }
    if let Some(err) = runtime_error.clone() {
        app.status =
            format!("Setup required: {err} — press Ctrl+, for settings or run: lucy config doctor");
    } else if stt.is_none() && app.messages.is_empty() {
        app.status = "Ready — voice optional (set GROQ_API_KEY) · /help for commands".into();
    }
    let (voice_tx, mut voice_rx) = mpsc::unbounded_channel::<Result<String, String>>();
    let (task_tx, mut task_rx) = mpsc::unbounded_channel::<TaskEvent>();
    // Toggle recording: None = idle, Some = actively recording.
    let mut recording: Option<Recording> = None;
    // Safety cap: auto-send after 60 seconds even if user forgets to press F2 again.
    const MAX_RECORD: Duration = Duration::from_secs(60);
    let result = loop {
        while let Ok(event) = task_rx.try_recv() {
            match event {
                TaskEvent::PushMsg(msg) => {
                    app.push_msg(msg);
                    app.pin();
                }
                TaskEvent::SetPhase { phase, detail } => {
                    app.set_phase(&phase, &detail);
                }
                TaskEvent::SetStatus(status) => {
                    app.status = status;
                }
                TaskEvent::SetSessionTitle(title) => {
                    app.session_title = title;
                }
                TaskEvent::SetSessionCount(count) => {
                    app.session_count = count;
                }
                TaskEvent::ApprovalRequest { id, name, input } => {
                    app.approval = Some(model::ApprovalDialog { id, name, input });
                    app.set_phase("Approval needed", "waiting for your decision");
                }
                TaskEvent::SetActiveModel(label) => {
                    app.active_model = label;
                }
                TaskEvent::Finished => {
                    // Refresh usage *before* stopping the clock: the final
                    // model call bills after its text is already on screen, so
                    // a read taken later would under-count the tokens this
                    // turn actually cost.
                    if let Some(rt) = runtime_opt.as_ref() {
                        app.usage = rt.usage();
                    }
                    app.mark_idle();
                    app.pin();
                    // Whiteboard: task completed → audible close of the loop.
                    lucy_stt::done_beep();
                }
            }
        }
        draw(&mut terminal, &app)?;
        if let Some(rt) = runtime_opt.as_ref() {
            app.usage = rt.usage();
        }
        while let Ok(result) = voice_rx.try_recv() {
            app.listening = false;
            match result {
                Ok(text) if !text.trim().is_empty() => {
                    let text = text.trim().to_owned();
                    submit_prompt(runtime_opt.as_ref(), &mut app, &task_tx, text).await;
                    // Paint the Thinking state immediately — otherwise the first
                    // frame only appears after the next 50ms poll, which feels
                    // like a freeze on slow terminals (foot).
                    draw(&mut terminal, &app)?;
                    if app.status == "quit-requested" {
                        break;
                    }
                }
                Ok(_) => app.status = "I didn't catch anything — try again".into(),
                Err(e) if e.contains("too short") || e.contains("no speech") => {
                    let ptt = app.config.voice.push_to_talk.to_ascii_uppercase();
                    lucy_stt::error_beep();
                    app.status = format!("Speak after pressing {ptt}, then press it again to send");
                }
                // Classified like every other failure: an STT error is usually an HTTP
                // status plus a provider sentence, and the status bar is one line.
                Err(e) => app.status = format!("Voice error: {}", lucy_core::friendly(&e)),
            }
        }
        if app.status == "quit-requested" {
            break Ok(());
        }
        // Safety cap: auto-send if user forgets to press F2 a second time.
        if let Some(rec) = &recording {
            if rec.started_at.elapsed() >= MAX_RECORD {
                stop_recording(&mut recording, stt.as_ref(), &mut app, &voice_tx);
            }
        }
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                // Only act on key-press events (ignore repeats and releases entirely).
                if key.kind != KeyEventKind::Press {
                    continue;
                }

                // Kill switch: Ctrl+C stops the run from ANY state, including
                // while an approval dialog is up (where Esc would only deny that
                // one prompt). This is the safety net automode relies on, since
                // automode never prompts.
                if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
                    if let Some(rt) = runtime_opt.as_ref() {
                        commands::stop_run(rt, &mut app);
                    }
                    continue;
                }

                // A planned command needs approval: the runtime is blocked
                // until the user answers with y / a / n.
                if let Some(dlg) = app.approval.clone() {
                    let decision = match key.code {
                        KeyCode::Char('y') | KeyCode::Char('Y') => {
                            Some(ApprovalDecision::AllowOnce)
                        }
                        KeyCode::Char('a') | KeyCode::Char('A') => {
                            Some(ApprovalDecision::AllowAlways)
                        }
                        KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                            Some(ApprovalDecision::Deny)
                        }
                        _ => None,
                    };
                    if let Some(decision) = decision {
                        if let Some(rt) = runtime_opt.as_ref() {
                            rt.approval_gate().resolve(&dlg.id, decision);
                            // "Always allow" only lived in memory, so the same
                            // tool re-prompted after every restart. Persist it.
                            if decision == ApprovalDecision::AllowAlways
                                && let Err(e) = rt.persist_always_allowed()
                            {
                                app.status = format!("Could not save always-allow: {e}");
                            }
                        }
                        app.approval = None;
                        app.status = match decision {
                            ApprovalDecision::AllowOnce => format!("Approved: {}", dlg.name),
                            ApprovalDecision::AllowAlways => {
                                format!("Always allowed: {}", dlg.name)
                            }
                            ApprovalDecision::Deny => format!("Denied: {}", dlg.name),
                        };
                    }
                    continue;
                }

                // Help overlay takes precedence (F2 opens, Esc/F2 closes).
                if app.show_help {
                    app.show_help = false;
                    app.status = "Ready".into();
                    continue;
                }

                // Sessions overlay.
                if app.show_sessions {
                    match key.code {
                        KeyCode::Esc => {
                            app.show_sessions = false;
                            app.status = "Ready".into();
                        }
                        KeyCode::Up => {
                            if !app.sessions.is_empty() {
                                app.sess_selected = app.sess_selected.saturating_sub(1);
                            }
                        }
                        KeyCode::Down => {
                            if !app.sessions.is_empty() {
                                app.sess_selected = (app.sess_selected + 1) % app.sessions.len();
                            }
                        }
                        KeyCode::Enter => {
                            if let Some(meta) = app.sessions.get(app.sess_selected).cloned() {
                                if let Some(rt) = runtime_opt.as_ref() {
                                    match rt.switch_session(&meta.id).await {
                                        Ok(switched) => {
                                            app.session_title = switched.title.clone();
                                            app.session_id_short =
                                                short_id(&switched.id.0.to_string());
                                            app.session_count = switched.message_count;
                                            app.load_turns(&rt.history().await);
                                            app.push_msg(ChatMsg::system(format!(
                                                "Switched to: {}",
                                                switched.title
                                            )));
                                            app.pin();
                                        }
                                        Err(e) => app.push_msg(ChatMsg::system(format!(
                                            "Switch failed: {e}"
                                        ))),
                                    }
                                }
                            }
                            app.show_sessions = false;
                            app.status = "Ready".into();
                        }
                        KeyCode::Char('d')
                            if !key.modifiers.intersects(
                                KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SUPER,
                            ) =>
                        {
                            if let Some(meta) = app.sessions.get(app.sess_selected).cloned() {
                                if let Some(rt) = runtime_opt.as_ref() {
                                    let _ = rt.delete_session(&meta.id).await;
                                    refresh_sessions(rt, &mut app).await;
                                }
                            }
                        }
                        KeyCode::Char('n')
                            if !key.modifiers.intersects(
                                KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SUPER,
                            ) =>
                        {
                            if let Some(rt) = runtime_opt.as_ref() {
                                if let Ok(meta) = rt.new_session(None).await {
                                    app.session_title = meta.title.clone();
                                    app.session_id_short = short_id(&meta.id.0.to_string());
                                    app.session_count = meta.message_count;
                                    app.load_turns(&[]);
                                    app.show_sessions = false;
                                    app.status = "Ready — new session".into();
                                }
                            }
                        }
                        _ => {}
                    }
                    continue;
                }

                // `/settings` — the six-section screen owns every key while it
                // is open, including its own async [Test]/[Save] buttons.
                if app.settings {
                    let action = app.settings_state.on_key(key, &mut app.config).await;
                    match action {
                        Some(settings::SettingsAction::Close) => {
                            app.settings = false;
                            app.status = "Settings closed".into();
                        }
                        Some(settings::SettingsAction::ModelsChanged) => {
                            // Track the Level 3 anchor: it is what the header
                            // shows and what the next turn will plan with.
                            app.model_label = app.config.resolve_level_model(ReasoningLevel::L3);
                        }
                        Some(settings::SettingsAction::ApprovalModeChanged) => {
                            let mode = app.config.approval_mode().to_owned();
                            if let Some(rt) = runtime_opt.as_ref() {
                                // Push into the live gate, else the new mode
                                // only takes effect on the next launch.
                                match rt.set_approval_mode(&mode) {
                                    Ok(_) => {
                                        app.status = format!(
                                            "Permissions: {}",
                                            app.config.approval_mode_label()
                                        );
                                        if mode == "never" {
                                            app.push_msg(ChatMsg::lucy(
                                                "Automode on — Lucy will run without asking. \
                                                 Stop any run with /stop, Esc, or Ctrl+C."
                                                    .into(),
                                            ));
                                        }
                                    }
                                    Err(e) => {
                                        app.status = format!("Could not apply permissions: {e}");
                                    }
                                }
                            }
                        }
                        None => {}
                    }
                    continue;
                }
                if key.code == KeyCode::Char(',') && key.modifiers.contains(KeyModifiers::CONTROL) {
                    app.settings = true;
                    app.settings_state = settings::SettingsState::open(&app.config);
                    continue;
                }
                // Ctrl+N: new session · Ctrl+O: sessions · F1: help
                if key.code == KeyCode::Char('n') && key.modifiers.contains(KeyModifiers::CONTROL) {
                    if app.busy {
                        app.status = "Task in progress — press Esc to cancel".into();
                        continue;
                    }
                    if let Some(rt) = runtime_opt.as_ref() {
                        match rt.new_session(None).await {
                            Ok(meta) => {
                                app.session_title = meta.title.clone();
                                app.session_id_short = short_id(&meta.id.0.to_string());
                                app.session_count = meta.message_count;
                                app.load_turns(&[]);
                                app.push_msg(ChatMsg::system(format!(
                                    "New session: {} ({})",
                                    meta.title, app.session_id_short
                                )));
                                app.pin();
                                app.status = "Ready — new session".into();
                            }
                            Err(e) => app.status = format!("New session failed: {e}"),
                        }
                    }
                    continue;
                }
                if key.code == KeyCode::Char('o') && key.modifiers.contains(KeyModifiers::CONTROL) {
                    if app.busy {
                        app.status = "Task in progress — press Esc to cancel".into();
                        continue;
                    }
                    if let Some(rt) = runtime_opt.as_ref() {
                        refresh_sessions(rt, &mut app).await;
                        app.show_sessions = true;
                        app.status =
                            "Sessions — ↑/↓ select · Enter switch · d delete · n new · Esc close"
                                .into();
                    }
                    continue;
                }
                // F1 = /new, F2 = help — the wireframe's bottom bar.
                if key.code == KeyCode::F(1) {
                    if app.busy {
                        app.status = "Task in progress — press Esc to cancel".into();
                    } else if let Some(rt) = runtime_opt.as_ref() {
                        match rt.new_session(None).await {
                            Ok(meta) => {
                                app.session_title = meta.title.clone();
                                app.session_id_short = short_id(&meta.id.0.to_string());
                                app.session_count = meta.message_count;
                                app.load_turns(&[]);
                                app.push_msg(ChatMsg::system(format!(
                                    "New session: {} ({})",
                                    meta.title, app.session_id_short
                                )));
                                app.pin();
                                app.status = "Ready — new session".into();
                            }
                            Err(e) => app.status = format!("New session failed: {e}"),
                        }
                    }
                    continue;
                }
                if key.code == KeyCode::F(2) {
                    app.show_help = !app.show_help;
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
                                recording = Some(Recording {
                                    cap,
                                    started_at: Instant::now(),
                                });
                                app.listening = true;
                                lucy_stt::ack_beep();
                                app.status =
                                    format!("🎙 Listening… press {ptt_label} again to send");
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
                if key.code == KeyCode::Enter
                    && (key.modifiers.contains(KeyModifiers::ALT)
                        || key.modifiers.contains(KeyModifiers::CONTROL))
                {
                    let bi = char_byte_idx(&app.input, app.cursor.min(app.input.chars().count()));
                    app.input.insert(bi, '\n');
                    app.cursor += 1;
                    app.hist_idx = None;
                    continue;
                }
                if let KeyCode::Char(c) = key.code {
                    if (c == 'j' || c == 'J')
                        && key.modifiers.contains(KeyModifiers::CONTROL)
                        && !key
                            .modifiers
                            .intersects(KeyModifiers::ALT | KeyModifiers::SUPER)
                    {
                        let bi =
                            char_byte_idx(&app.input, app.cursor.min(app.input.chars().count()));
                        app.input.insert(bi, '\n');
                        app.cursor += 1;
                        app.hist_idx = None;
                        continue;
                    }
                }

                match key.code {
                    KeyCode::Esc => {
                        if let Some(rec) = recording.take() {
                            drop(rec.cap);
                            app.listening = false;
                            app.status = "Cancelled".into();
                        } else if app.cmd_suggestions_visible() {
                            // Hide the list for this input (it returns on the
                            // next edit); Esc must not quit the app here.
                            app.cmd_suggest_dismissed = true;
                        } else if app.show_help {
                            app.show_help = false;
                        } else if app.busy {
                            if let Some(rt) = runtime_opt.as_ref() {
                                // Full stop, not just an interrupt: also drain
                                // pending approvals so the run cannot hang.
                                commands::stop_run(rt, &mut app);
                            } else {
                                app.set_phase("Cancelling", "stopping task");
                            }
                        } else {
                            break Ok(());
                        }
                    }
                    KeyCode::Up => {
                        if app.cmd_suggestions_visible() {
                            app.cmd_move_suggestion(-1);
                        } else if !app.history.is_empty() {
                            if app.hist_idx.is_none() {
                                app.hist_draft = app.input.clone();
                                let idx = app.history.len() - 1;
                                app.hist_idx = Some(idx);
                                app.input = app.history[idx].clone();
                                app.cursor = app.input.chars().count();
                                app.reset_cmd_suggestions();
                            } else if let Some(idx) = app.hist_idx {
                                if idx > 0 {
                                    let nidx = idx - 1;
                                    app.hist_idx = Some(nidx);
                                    app.input = app.history[nidx].clone();
                                    app.cursor = app.input.chars().count();
                                    app.reset_cmd_suggestions();
                                }
                            }
                        }
                    }
                    KeyCode::Down => {
                        if app.cmd_suggestions_visible() {
                            app.cmd_move_suggestion(1);
                        } else if let Some(idx) = app.hist_idx {
                            if idx + 1 < app.history.len() {
                                let nidx = idx + 1;
                                app.hist_idx = Some(nidx);
                                app.input = app.history[nidx].clone();
                                app.cursor = app.input.chars().count();
                                app.reset_cmd_suggestions();
                            } else {
                                app.hist_idx = None;
                                app.input = app.hist_draft.clone();
                                app.cursor = app.input.chars().count();
                                app.reset_cmd_suggestions();
                            }
                        }
                    }
                    KeyCode::Tab if app.cmd_suggestions_visible() => {
                        app.cmd_accept_suggestion();
                    }
                    KeyCode::PageUp => {
                        app.scroll = app.scroll.saturating_add(10);
                    }
                    KeyCode::PageDown => {
                        app.scroll = app.scroll.saturating_sub(10);
                    }
                    KeyCode::Home if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        app.scroll = usize::MAX;
                    }
                    KeyCode::End if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        app.pin();
                    }
                    KeyCode::Left => {
                        app.cursor = app.cursor.saturating_sub(1);
                    }
                    KeyCode::Right => {
                        let n = app.input.chars().count();
                        app.cursor = (app.cursor + 1).min(n);
                    }
                    KeyCode::Home => {
                        app.cursor = line_home_cursor(&app.input, app.cursor);
                    }
                    KeyCode::End => {
                        app.cursor = line_end_cursor(&app.input, app.cursor);
                    }
                    KeyCode::Char(c)
                        if key.modifiers.contains(KeyModifiers::CONTROL)
                            && c.to_ascii_lowercase() == 'a' =>
                    {
                        app.cursor = 0;
                    }
                    KeyCode::Char(c)
                        if key.modifiers.contains(KeyModifiers::CONTROL)
                            && c.to_ascii_lowercase() == 'e' =>
                    {
                        app.cursor = app.input.chars().count();
                    }
                    KeyCode::Char(c)
                        if !key.modifiers.intersects(
                            KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SUPER,
                        ) =>
                    {
                        let bi =
                            char_byte_idx(&app.input, app.cursor.min(app.input.chars().count()));
                        app.input.insert(bi, c);
                        app.cursor += 1;
                        app.hist_idx = None;
                        app.reset_cmd_suggestions();
                    }
                    KeyCode::Backspace => {
                        if app.cursor > 0 {
                            let count = app.input.chars().count();
                            let cur = app.cursor.min(count);
                            let bi = char_byte_idx(&app.input, cur - 1);
                            let ch_len = app.input[bi..]
                                .chars()
                                .next()
                                .map(|c| c.len_utf8())
                                .unwrap_or(0);
                            if ch_len > 0 {
                                app.input.drain(bi..bi + ch_len);
                            }
                            app.cursor = cur - 1;
                        }
                        app.hist_idx = None;
                        app.reset_cmd_suggestions();
                    }
                    KeyCode::Enter
                        if !key.modifiers.intersects(
                            KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SUPER,
                        ) && !app.input.trim().is_empty() =>
                    {
                        if app.busy {
                            app.status = "Task in progress — press Esc to cancel".into();
                            continue;
                        }
                        let text = match app.cmd_selected() {
                            // Popup open: Enter runs the highlighted command
                            // instead of submitting the partial prefix.
                            Some(cmd) => cmd.to_owned(),
                            None => app.input.trim().to_owned(),
                        };
                        app.input.clear();
                        app.cursor = 0;
                        // Slash and normal lines alike go to input history.
                        if app.history.last().is_none_or(|l| l != &text) {
                            app.history.push(text.clone());
                            if app.history.len() > 200 {
                                let excess = app.history.len() - 200;
                                app.history.drain(0..excess);
                            }
                        }
                        app.hist_idx = None;
                        app.hist_draft.clear();
                        app.reset_cmd_suggestions();
                        submit_prompt(runtime_opt.as_ref(), &mut app, &task_tx, text).await;
                        // Critical: submit_prompt sets busy/Thinking synchronously
                        // and spawns the LLM work. Paint that frame NOW so the
                        // welcome→active switch + spinner appear instantly on
                        // Enter instead of after the first LLM round-trip.
                        draw(&mut terminal, &app)?;
                        if app.status == "quit-requested" {
                            break Ok(());
                        }
                    }
                    _ => {}
                }
            }
        }
    };
    let cleanup_result = cleanup(&mut terminal);
    result.and(cleanup_result)
}

pub async fn run() -> anyhow::Result<()> {
    let cfg = LucyConfig::load()?;
    let stt = GroqStt::from_config(&cfg).ok().map(Arc::new);
    run_voice(stt).await
}
