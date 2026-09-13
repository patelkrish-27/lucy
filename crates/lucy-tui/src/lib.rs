use std::io::{self, Stdout};
use std::sync::Arc;
use std::time::Duration;

use crossterm::{
    event::{
        self, Event, KeyCode, KeyEventKind, KeyModifiers, KeyboardEnhancementFlags,
        PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use lucy_core::AgentEvent;
use lucy_runtime::LucyRuntime;
use lucy_stt::GroqStt;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph, Wrap},
    Terminal,
};
use tokio::sync::mpsc;

const VOICE_HOTKEY: &str = "Super+C / Ctrl+Space / F2";
const MASCOT: [&str; 9] = [
    "      ╭────────────────────╮      ",
    "      │                    │      ",
    "      │    ███      ███    │      ",
    "      │    ███      ███    │      ",
    "      │                    │      ",
    "      │       ╭────╮       │      ",
    "      │      ╰──────╯      │      ",
    "      │                    │      ",
    "      ╰────────────────────╯      ",
];

pub struct App {
    pub status: String,
    pub listening: bool,
    pub commands: Vec<String>,
    pub input: String,
}

impl Default for App {
    fn default() -> Self {
        Self {
            status: "Ready".into(),
            listening: false,
            commands: Vec::new(),
            input: String::new(),
        }
    }
}

fn draw(terminal: &mut Terminal<CrosstermBackend<Stdout>>, app: &App) -> anyhow::Result<()> {
    terminal.draw(|frame| {
        let area = frame.area();
        if app.commands.is_empty() {
            draw_welcome(frame, area, app);
        } else {
            draw_active(frame, area, app);
        }
    })?;
    Ok(())
}

fn draw_welcome(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App) {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),
            Constraint::Length(11),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(2),
        ])
        .split(area);

    let brand = Paragraph::new(Line::from(vec![
        Span::styled("LUCY", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw("  ·  your computer companion"),
    ]))
    .alignment(Alignment::Center);
    frame.render_widget(brand, vertical[0]);

    let mascot = Text::from(
        MASCOT
            .iter()
            .map(|line| Line::from(Span::raw(*line)))
            .collect::<Vec<_>>(),
    );
    frame.render_widget(Paragraph::new(mascot).alignment(Alignment::Center), vertical[1]);

    let status = if app.listening {
        "◉  Listening… speak now, then pause"
    } else {
        "What can I do for you?"
    };
    frame.render_widget(
        Paragraph::new(status)
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::NONE)),
        vertical[2],
    );

    render_input(frame, vertical[3], app, "  Tell Lucy what to do  ");

    let controls = format!(
        "[Enter] send    [{}] voice    [Esc] quit",
        VOICE_HOTKEY
    );
    frame.render_widget(
        Paragraph::new(controls)
            .alignment(Alignment::Center)
            .style(Style::default().add_modifier(Modifier::DIM)),
        vertical[4],
    );
}

fn draw_active(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App) {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Min(3), Constraint::Length(3), Constraint::Length(2)])
        .split(area);

    // Once the first command is submitted, Lucy collapses into a lightweight one-line header.
    let header = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(10), Constraint::Length(28)])
        .split(vertical[0]);
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("● LUCY", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw("  ·  computer companion"),
        ])),
        header[0],
    );
    frame.render_widget(
        Paragraph::new(if app.listening { "◉ Listening…" } else { &app.status })
            .alignment(Alignment::Right),
        header[1],
    );

    let history_lines = app.commands.iter().map(|command| {
        if let Some(rest) = command.strip_prefix("You › ") {
            Line::from(vec![
                Span::styled("You  ", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw(rest),
            ])
        } else {
            Line::from(command.as_str())
        }
    });
    let history = Paragraph::new(Text::from_iter(history_lines))
        .block(Block::default().title(" Activity ").borders(Borders::ROUNDED))
        .wrap(Wrap { trim: true });
    frame.render_widget(history, vertical[1]);

    render_input(frame, vertical[2], app, "  Ask Lucy  ");

    let controls = format!(
        "[Enter] send   [{}] voice   [Esc] quit",
        VOICE_HOTKEY
    );
    frame.render_widget(
        Paragraph::new(controls)
            .alignment(Alignment::Center)
            .style(Style::default().add_modifier(Modifier::DIM)),
        vertical[3],
    );
}

fn render_input(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App, title: &str) {
    let text = format!("> {}", app.input);
    frame.render_widget(
        Paragraph::new(text)
            .block(Block::default().title(title).borders(Borders::ROUNDED))
            .wrap(Wrap { trim: true }),
        area,
    );
    let cursor_x = area.x
        .saturating_add(2)
        .saturating_add(app.input.chars().count() as u16 + 2)
        .min(area.right().saturating_sub(1));
    frame.set_cursor_position((cursor_x, area.y.saturating_add(1)));
}

fn cleanup(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> anyhow::Result<()> {
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        PopKeyboardEnhancementFlags,
        LeaveAlternateScreen
    )?;
    terminal.show_cursor()?;
    Ok(())
}

pub async fn run_voice(stt: Option<Arc<GroqStt>>) -> anyhow::Result<()> {
    let runtime = Arc::new(LucyRuntime::new().await?);
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
        app.status = "STT unavailable — set GROQ_API_KEY to enable voice".into();
    }

    let (voice_tx, mut voice_rx) = mpsc::unbounded_channel::<Result<String, String>>();
    let mut voice_task_running = false;
    let mut agent_rx: Option<mpsc::UnboundedReceiver<AgentEvent>> = None;

    let result = loop {
        draw(&mut terminal, &app)?;
        let mut agent_done = false;

        if let Some(rx) = agent_rx.as_mut() {
            loop {
                match rx.try_recv() {
                    Ok(event) => match event {
                        AgentEvent::Status { message } => app.status = message,
                        AgentEvent::TextDelta { text } => {
                            app.commands.push(format!("Lucy › {text}"));
                            app.status = "Working…".into();
                        }
                        AgentEvent::ToolStarted { name, .. } => {
                            app.status = format!("Using {name}…")
                        }
                        AgentEvent::ToolFinished { .. }
                        | AgentEvent::History { .. }
                        | AgentEvent::Thinking { .. } => {}
                        AgentEvent::Error { message } => app.status = format!("Error: {message}"),
                        AgentEvent::Done => {
                            agent_done = true;
                            break;
                        }
                    },
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => {
                        agent_done = true;
                        break;
                    }
                }
            }
        }

        if agent_done {
            agent_rx = None;
            app.status = "Ready".into();
        }

        while let Ok(result) = voice_rx.try_recv() {
            voice_task_running = false;
            app.listening = false;
            match result {
                Ok(text) if !text.trim().is_empty() => {
                    let text = text.trim().to_owned();
                    app.commands.push(format!("You › {text}"));
                    app.status = "Executing…".into();
                    agent_rx = Some(runtime.submit(text).await?);
                }
                Ok(_) => app.status = "I didn't catch anything — try again".into(),
                Err(e) => app.status = format!("Voice error: {e}"),
            }
        }

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }

                let hotkey =
                    (key.code == KeyCode::Char('c')
                        && key.modifiers.contains(KeyModifiers::SUPER))
                        || (key.code == KeyCode::Char(' ')
                            && key.modifiers.contains(KeyModifiers::CONTROL))
                        || key.code == KeyCode::F(2)
                        || key.code == KeyCode::F(9)
                        || (key.code == KeyCode::Char('v')
                            && key.modifiers.contains(KeyModifiers::ALT))
                        || (key.code == KeyCode::Char('m')
                            && key.modifiers.contains(KeyModifiers::CONTROL));

                if hotkey && !voice_task_running {
                    if let Some(stt) = stt.as_ref() {
                        let stt = Arc::clone(stt);
                        let tx = voice_tx.clone();
                        voice_task_running = true;
                        app.listening = true;
                        app.status = "Listening…".into();
                        tokio::spawn(async move {
                            let _ = tx.send(stt.listen_once().await.map_err(|e| e.to_string()));
                        });
                    } else {
                        app.status = "Voice disabled — set GROQ_API_KEY".into();
                    }
                    continue;
                }

                match key.code {
                    KeyCode::Esc => break Ok(()),
                    KeyCode::Char(c)
                        if !key
                            .modifiers
                            .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SUPER) =>
                    {
                        app.input.push(c);
                    }
                    KeyCode::Backspace => {
                        app.input.pop();
                    }
                    KeyCode::Enter if !app.input.trim().is_empty() => {
                        let text = app.input.trim().to_owned();
                        app.input.clear();
                        app.commands.push(format!("You › {text}"));
                        app.status = "Executing…".into();
                        agent_rx = Some(runtime.submit(text).await?);
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
    let stt = GroqStt::from_env().ok().map(Arc::new);
    run_voice(stt).await
}
