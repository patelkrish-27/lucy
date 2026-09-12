use std::io;
use crossterm::{event::{self, Event, KeyCode}, execute, terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen}};
use ratatui::{backend::CrosstermBackend, widgets::{Block, Borders, Paragraph}, Terminal};
use tokio::sync::mpsc;
use lucy_core::AgentEvent;

pub struct App { pub input: String, pub status: String, pub transcript: Vec<String>, pub running: bool }
impl Default for App { fn default() -> Self { Self { input: String::new(), status: "Ready".into(), transcript: Vec::new(), running: false } } }

pub fn run(mut rx: Option<mpsc::UnboundedReceiver<AgentEvent>>) -> anyhow::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let mut app = App::default();
    loop {
        terminal.draw(|f| { let area = f.area(); let text = format!("Lucy\n\n{}\n\n> {}\n\n[Esc] quit", app.status, app.input); f.render_widget(Paragraph::new(text).block(Block::default().title("Lucy — Your AI Computer Buddy").borders(Borders::ALL)), area); })?;
        if let Some(receiver) = rx.as_mut() { while let Ok(event) = receiver.try_recv() { match event { AgentEvent::TextDelta { text } => app.transcript.push(text), AgentEvent::Status { message } => app.status = message, AgentEvent::Error { message } => app.status = message, AgentEvent::Done => app.running = false, _ => {} } } }
        if event::poll(std::time::Duration::from_millis(50))? { if let Event::Key(key) = event::read()? { match key.code { KeyCode::Esc => break, KeyCode::Char(c) => app.input.push(c), KeyCode::Backspace => { app.input.pop(); }, KeyCode::Enter => { app.status = format!("Queued: {}", app.input); app.input.clear(); app.running=true; }, _ => {} } } }
    }
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}
