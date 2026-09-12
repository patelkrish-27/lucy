use std::io::{self, Stdout};
use std::sync::Arc;
use std::time::Duration;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers, KeyboardEnhancementFlags,
        PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use lucy_stt::GroqStt;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Terminal,
};
use tokio::sync::mpsc;

const VOICE_HOTKEY: &str = "Super+C";

pub struct App {
    pub status: String,
    pub listening: bool,
    pub commands: Vec<String>,
    pub input: String,
}

impl Default for App {
    fn default() -> Self {
        Self {
            status: "Ready — press Super+C to speak".into(),
            listening: false,
            commands: Vec::new(),
            input: String::new(),
        }
    }
}

fn draw(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    app: &App,
) -> anyhow::Result<()> {
    terminal.draw(|frame| {
        let outer = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(5), Constraint::Length(3), Constraint::Length(2)])
            .split(frame.area());

        let history = if app.commands.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "No commands yet. Press Super+C and start speaking.",
                Style::default().add_modifier(Modifier::DIM),
            )))]
        } else {
            app.commands
                .iter()
                .map(|command| ListItem::new(Line::from(command.as_str())))
                .collect()
        };

        let history_block = List::new(history).block(
            Block::default()
                .title(" Command History ")
                .borders(Borders::ALL),
        );
        frame.render_widget(history_block, outer[0]);

        let status = if app.listening {
            "🎙  Listening… speak now, then pause"
        } else {
            &app.status
        };
        frame.render_widget(
            Paragraph::new(status)
                .block(Block::default().title(" Status ").borders(Borders::ALL))
                .wrap(Wrap { trim: true }),
            outer[1],
        );

        frame.render_widget(
            Paragraph::new(format!("[{}] voice   [Esc] quit", VOICE_HOTKEY))
                .style(Style::default().add_modifier(Modifier::DIM)),
            outer[2],
        );
    })?;
    Ok(())
}

fn cleanup(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
) -> anyhow::Result<()> {
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        PopKeyboardEnhancementFlags,
        LeaveAlternateScreen
    )?;
    terminal.show_cursor()?;
    Ok(())
}

/// Run Lucy's first usable voice TUI.
///
/// Super+C starts one live microphone utterance. After the user pauses,
/// Groq Whisper transcribes it and the resulting command is shown in the UI.
pub async fn run_voice(stt: Option<Arc<GroqStt>>) -> anyhow::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(
        stdout,
        EnterAlternateScreen,
        PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
    )?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let mut app = App::default();

    if stt.is_none() {
        app.status = "STT unavailable — set GROQ_API_KEY, then restart Lucy".into();
    }

    let (voice_tx, mut voice_rx) = mpsc::unbounded_channel::<Result<String, String>>();
    let mut voice_task_running = false;

    let result = loop {
        draw(&mut terminal, &app)?;

        while let Ok(result) = voice_rx.try_recv() {
            voice_task_running = false;
            app.listening = false;

            match result {
                Ok(text) if !text.trim().is_empty() => {
                    let text = text.trim().to_owned();
                    app.commands.push(format!("You  ›  {text}"));
                    app.status = format!("Command received: {text}");
                }
                Ok(_) => {
                    app.status = "I didn't catch anything — press Super+C and try again".into();
                }
                Err(error) => {
                    app.status = format!("Voice error: {error}");
                }
            }
        }

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }

                let super_c = key.code == KeyCode::Char('c')
                    && key.modifiers.contains(KeyModifiers::SUPER);

                if super_c && !voice_task_running {
                    match stt.as_ref() {
                        Some(stt) => {
                            let stt = Arc::clone(stt);
                            let tx = voice_tx.clone();
                            voice_task_running = true;
                            app.listening = true;
                            app.status = "Listening…".into();

                            tokio::spawn(async move {
                                let result = stt.listen_once().await.map_err(|e| e.to_string());
                                let _ = tx.send(result);
                            });
                        }
                        None => {
                            app.status =
                                "Voice is disabled — set GROQ_API_KEY and restart Lucy".into();
                        }
                    }
                    continue;
                }

                match key.code {
                    KeyCode::Esc => break Ok(()),
                    KeyCode::Char(c) if key.modifiers.is_empty() => app.input.push(c),
                    KeyCode::Backspace => {
                        app.input.pop();
                    }
                    KeyCode::Enter if !app.input.trim().is_empty() => {
                        let text = app.input.trim().to_owned();
                        app.commands.push(format!("You  ›  {text}"));
                        app.status = format!("Command received: {text}");
                        app.input.clear();
                    }
                    _ => {}
                }
            }
        }
    };

    let cleanup_result = cleanup(&mut terminal);
    result.and(cleanup_result)
}

/// Backwards-compatible entry point for callers that only need the TUI shell.
pub async fn run() -> anyhow::Result<()> {
    let stt = match GroqStt::from_env() {
        Ok(stt) => Some(Arc::new(stt)),
        Err(_) => None,
    };
    run_voice(stt).await
}
