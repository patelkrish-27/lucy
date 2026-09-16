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
mod model;
mod settings;
mod util;
mod views;
mod voice;

use std::io::{self, Stdout};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{
        disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
    },
};
use lucy_config::LucyConfig;
use lucy_core::{AgentEvent, ApprovalDecision};
use lucy_runtime::LucyRuntime;
use lucy_stt::GroqStt;
use ratatui::{backend::CrosstermBackend, Terminal};
use tokio::sync::mpsc;

use commands::{format_tool_finish, format_tool_start, refresh_sessions, submit_text};
use model::{App, ApprovalDialog, ChatMsg};
use util::{char_byte_idx, line_end_cursor, line_home_cursor, short_id};
use views::draw;
use voice::{is_voice_hotkey, stop_recording, Recording};

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
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let mut app = App::new(config);
    app.model_label = runtime_opt
        .as_ref()
        .map(|r| r.config().models.main.clone())
        .unwrap_or(app.config.models.main.clone());
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
    } else if stt.is_none() && app.messages.is_empty() {
        app.status = "Ready — voice optional (set GROQ_API_KEY) · /help for commands".into();
    }
    let (voice_tx, mut voice_rx) = mpsc::unbounded_channel::<Result<String, String>>();
    // Toggle recording: None = idle, Some = actively recording.
    let mut recording: Option<Recording> = None;
    // Safety cap: auto-send after 60 seconds even if user forgets to press F2 again.
    const MAX_RECORD: Duration = Duration::from_secs(60);
    let mut agent_rx: Option<mpsc::UnboundedReceiver<AgentEvent>> = None;
    let result = loop {
        draw(&mut terminal, &app)?;
        let mut agent_done = false;
        if let Some(rx) = agent_rx.as_mut() {
            loop {
                match rx.try_recv() {
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
            // Drop any stale transient triage note that was ever pushed to the feed.
            app.progress.retain(|p| p.trim() != "Understanding your request…");
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
    let cleanup_result = cleanup(&mut terminal);
    result.and(cleanup_result)
}

pub async fn run() -> anyhow::Result<()> {
    let cfg = LucyConfig::load()?;
    let stt = GroqStt::from_config(&cfg).ok().map(Arc::new);
    run_voice(stt).await
}
