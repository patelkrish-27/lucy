//! Lucy TUI views: calm, compact, terminal-first UI.
use std::io::Stdout;

use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Wrap},
    Terminal,
};

use super::model::{App, MsgKind};
use super::util::{cursor_row_col, format_tokens, short_id, truncate_model_label, truncate_one_line};
use crate::settings;

// Tokyo Night-inspired, intentionally muted. One accent carries focus.
const BG: Color = Color::Rgb(26, 27, 38);
const PANEL: Color = Color::Rgb(30, 32, 45);
const PANEL_HI: Color = Color::Rgb(36, 39, 55);
const BORDER: Color = Color::Rgb(60, 63, 82);
const TEXT: Color = Color::Rgb(192, 202, 220);
const TEXT_BRIGHT: Color = Color::Rgb(220, 225, 236);
const MUTED: Color = Color::Rgb(120, 127, 148);
const BLUE: Color = Color::Rgb(122, 162, 247);
const PURPLE: Color = Color::Rgb(187, 154, 247);
const GREEN: Color = Color::Rgb(158, 206, 106);
const ORANGE: Color = Color::Rgb(224, 175, 104);
const RED: Color = Color::Rgb(247, 118, 142);

pub(crate) fn draw(terminal: &mut Terminal<CrosstermBackend<Stdout>>, app: &App) -> anyhow::Result<()> {
    terminal.draw(|frame| {
        let area = frame.area();
        frame.render_widget(Block::default().style(Style::default().bg(BG)), area);
        if app.messages.is_empty() && app.streaming.is_empty() && !app.settings {
            draw_welcome(frame, area, app);
        } else {
            draw_active(frame, area, app);
        }
        if app.settings { settings::draw(frame, area, &app.config, app.settings_selected); }
        if app.show_sessions { draw_sessions_popup(frame, area, app); }
        if app.show_help { draw_help_popup(frame, area); }
        if app.approval.is_some() { draw_approval_popup(frame, area, app); }
    })?;
    Ok(())
}

fn top_bar(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App) {
    let parts = Layout::default().direction(Direction::Horizontal)
        .constraints([Constraint::Min(20), Constraint::Length(46)]).split(area);
    let title = if app.session_title.trim().is_empty() { "New session" } else { &app.session_title };
    let state = if app.listening { ("●", "Listening", GREEN) }
        else if app.busy { ("●", if app.phase.is_empty() { "Working" } else { &app.phase }, BLUE) }
        else { ("●", "Ready", GREEN) };
    frame.render_widget(Paragraph::new(Line::from(vec![
        Span::styled("✦ LUCY", Style::default().fg(PURPLE).add_modifier(Modifier::BOLD)),
        Span::styled("  ·  ", Style::default().fg(BORDER)),
        Span::styled(truncate_one_line(title, 52), Style::default().fg(TEXT_BRIGHT)),
    ])), parts[0]);
    let model = truncate_model_label(&app.model_label);
    let right = format!("{}  {}  ·  {}  ·  {}", state.0, state.1, if model.is_empty() { "model" } else { &model }, format_tokens(app.usage.total_tokens));
    frame.render_widget(Paragraph::new(right).style(Style::default().fg(MUTED)).alignment(Alignment::Right), parts[1]);
}

fn draw_welcome(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App) {
    let outer = centered(area, 118, area.height.saturating_sub(2));
    let rows = Layout::default().direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Min(8), Constraint::Length(4), Constraint::Length(1)]).split(outer);
    top_bar(frame, rows[0], app);

    let hero = Layout::default().direction(Direction::Vertical)
        .constraints([Constraint::Length(8), Constraint::Length(3), Constraint::Min(1)]).split(rows[1]);
    let mascot = vec![
        Line::from(Span::styled("      .-''''-.      ", Style::default().fg(PURPLE))),
        Line::from(Span::styled("    .'  ◡  ◡  '.    ", Style::default().fg(PURPLE))),
        Line::from(Span::styled("   /     ◡      \\    ", Style::default().fg(PURPLE))),
        Line::from(Span::styled("   '._       _.'    ", Style::default().fg(PURPLE))),
        Line::from(Span::styled("      '-----'       ", Style::default().fg(PURPLE))),
        Line::from(""),
        Line::from(Span::styled("Hi, I'm Lucy.", Style::default().fg(TEXT_BRIGHT).add_modifier(Modifier::BOLD))),
        Line::from(Span::styled("Your personal computer buddy. Tell me the outcome - I'll handle the steps.", Style::default().fg(MUTED))),
    ];
    frame.render_widget(Paragraph::new(Text::from(mascot)).alignment(Alignment::Center), hero[0]);

    let suggestions = Line::from(vec![
        Span::styled("Try  ", Style::default().fg(MUTED)),
        Span::styled("create a file", Style::default().fg(BLUE)), Span::styled("  ·  ", Style::default().fg(BORDER)),
        Span::styled("edit a video", Style::default().fg(BLUE)), Span::styled("  ·  ", Style::default().fg(BORDER)),
        Span::styled("draw an image", Style::default().fg(BLUE)), Span::styled("  ·  ", Style::default().fg(BORDER)),
        Span::styled("run a command", Style::default().fg(BLUE)),
    ]);
    frame.render_widget(Paragraph::new(suggestions).alignment(Alignment::Center), hero[1]);
    render_input(frame, hero[2], app, " Ask Lucy anything ");
    frame.render_widget(Paragraph::new("Enter send   ·   / commands   ·   F2 voice   ·   Ctrl+N new   ·   Ctrl+O sessions   ·   F1 help   ·   Esc quit")
        .alignment(Alignment::Center).style(Style::default().fg(MUTED)), rows[3]);
}

fn draw_active(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App) {
    let rows = Layout::default().direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Min(5), Constraint::Length(4), Constraint::Length(2)]).split(area);
    top_bar(frame, rows[0], app);

    let wide = area.width >= 120;
    let cols = if wide {
        Layout::default().direction(Direction::Horizontal).constraints([Constraint::Length(22), Constraint::Min(48), Constraint::Length(24)]).split(rows[1])
    } else {
        Layout::default().direction(Direction::Horizontal).constraints([Constraint::Length(18), Constraint::Min(42)]).split(rows[1])
    };
    draw_sidebar(frame, cols[0], app);
    draw_chat(frame, cols[1], app);
    if wide { draw_status_panel(frame, cols[2], app); }

    render_input(frame, rows[2], app, if app.input.starts_with('/') { " Command " } else { " Ask Lucy anything " });
    frame.render_widget(Paragraph::new("Enter send   ·   ↑/↓ history   ·   Ctrl+N new   ·   Ctrl+O sessions   ·   Ctrl+, settings   ·   F1 help   ·   Esc quit")
        .alignment(Alignment::Center).style(Style::default().fg(MUTED)), rows[3]);
}

fn draw_sidebar(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App) {
    let block = Block::default().borders(Borders::RIGHT).border_style(Style::default().fg(BORDER));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let title = if app.session_title.trim().is_empty() { "New session" } else { &app.session_title };
    let lines = vec![
        Line::from(Span::styled("SESSION", Style::default().fg(MUTED).add_modifier(Modifier::BOLD))),
        Line::from(Span::styled(truncate_one_line(title, inner.width.saturating_sub(2) as usize), Style::default().fg(TEXT_BRIGHT))),
        Line::from(Span::styled(format!("{} messages", app.session_count), Style::default().fg(MUTED))),
        Line::from(""),
        Line::from(Span::styled("CHAT", Style::default().fg(BLUE).add_modifier(Modifier::BOLD))),
        Line::from(Span::styled("  Ctrl+N  new session", Style::default().fg(TEXT))),
        Line::from(Span::styled("  Ctrl+O  sessions", Style::default().fg(TEXT))),
        Line::from(""),
        Line::from(Span::styled("LUCY", Style::default().fg(MUTED).add_modifier(Modifier::BOLD))),
        Line::from(Span::styled("  Ctrl+,  settings", Style::default().fg(TEXT))),
        Line::from(Span::styled("  /       commands", Style::default().fg(TEXT))),
    ];
    frame.render_widget(Paragraph::new(Text::from(lines)).wrap(Wrap { trim: true }), inner);
    if area.height > 12 {
        let quote = Rect { x: inner.x, y: inner.bottom().saturating_sub(5), width: inner.width, height: 5 };
        frame.render_widget(Paragraph::new("\"Small commands.\n  Big possibilities.\"\n\n  — Lucy").style(Style::default().fg(MUTED)), quote);
    }
}

fn draw_status_panel(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App) {
    let block = Block::default().borders(Borders::LEFT).border_style(Style::default().fg(BORDER));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let mut lines = vec![Line::from(Span::styled("STATUS", Style::default().fg(MUTED).add_modifier(Modifier::BOLD)))];
    if app.busy {
        let spinner = ["◐", "◓", "◑", "◒"][(app.busy_elapsed_ms() / 180 % 4) as usize];
        lines.push(Line::from(vec![Span::styled(format!("{} ", spinner), Style::default().fg(BLUE)), Span::styled(if app.phase.is_empty() { "Working" } else { &app.phase }, Style::default().fg(TEXT_BRIGHT).add_modifier(Modifier::BOLD))]));
        lines.push(Line::from(Span::styled(format!("{}s elapsed", app.busy_elapsed_ms() / 1000), Style::default().fg(MUTED))));
        if !app.phase_detail.is_empty() {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled("CURRENT", Style::default().fg(MUTED).add_modifier(Modifier::BOLD))));
            lines.push(Line::from(Span::styled(truncate_one_line(&app.phase_detail, inner.width.saturating_sub(1) as usize), Style::default().fg(TEXT))));
        }
    } else {
        lines.push(Line::from(vec![Span::styled("● ", Style::default().fg(GREEN)), Span::styled("Ready", Style::default().fg(GREEN))]));
        lines.push(Line::from(Span::styled("Waiting for your next task", Style::default().fg(MUTED))));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled("MODEL", Style::default().fg(MUTED).add_modifier(Modifier::BOLD))));
    lines.push(Line::from(Span::styled(truncate_model_label(&app.model_label), Style::default().fg(TEXT))));
    lines.push(Line::from(Span::styled(format!("{} tokens", format_tokens(app.usage.total_tokens)), Style::default().fg(MUTED))));
    if !app.progress.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled("RECENT", Style::default().fg(MUTED).add_modifier(Modifier::BOLD))));
        for p in app.progress.iter().rev().take(4).rev() {
            lines.push(Line::from(vec![Span::styled("› ", Style::default().fg(PURPLE)), Span::styled(truncate_one_line(p, inner.width.saturating_sub(3) as usize), Style::default().fg(MUTED))]));
        }
    }
    frame.render_widget(Paragraph::new(Text::from(lines)).wrap(Wrap { trim: true }), inner);
}

fn draw_chat(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App) {
    let block = Block::default().borders(Borders::ALL).border_style(Style::default().fg(BORDER))
        .title(Span::styled(if app.busy { " Activity · working " } else { " Activity " }, Style::default().fg(MUTED).add_modifier(Modifier::BOLD)));
    let inner = block.inner(area);
    let lines = build_chat_lines(app);
    let height = inner.height.max(1) as usize;
    let width = inner.width.max(1) as usize;
    let total = visual_rows(&lines, width);
    let max_scroll = total.saturating_sub(height);
    let scroll = app.scroll.min(max_scroll);
    let top = max_scroll.saturating_sub(scroll) as u16;
    frame.render_widget(Paragraph::new(Text::from(lines)).block(block).wrap(Wrap { trim: false }).scroll((top, 0)), area);
}

fn build_chat_lines(app: &App) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    for m in &app.messages {
        match m.kind {
            MsgKind::User => {
                out.push(Line::from(vec![Span::styled("You", Style::default().fg(BLUE).add_modifier(Modifier::BOLD)), Span::styled("  ", Style::default().fg(BORDER)), Span::styled(m.text.clone(), Style::default().fg(TEXT_BRIGHT))]));
            }
            MsgKind::Lucy => {
                out.push(Line::from(Span::styled("Lucy", Style::default().fg(PURPLE).add_modifier(Modifier::BOLD))));
                for line in format_reply_lines(&m.text) {
                    out.push(Line::from(vec![Span::styled("  ", Style::default().fg(BORDER)), Span::styled(line, Style::default().fg(TEXT))]));
                }
            }
            MsgKind::System => {
                for line in format_reply_lines(&m.text) { out.push(Line::from(Span::styled(format!("· {}", line), Style::default().fg(ORANGE)))); }
            }
            MsgKind::Tool => {
                let error = m.text.starts_with('✖');
                let done = m.text.starts_with('✔');
                let (icon, color) = if error { ("✕", RED) } else if done { ("✓", GREEN) } else { ("›", MUTED) };
                out.push(Line::from(vec![Span::styled(format!("{} ", icon), Style::default().fg(color)), Span::styled(truncate_one_line(&m.text, 180), Style::default().fg(MUTED))]));
            }
        }
        out.push(Line::from(""));
    }
    if !app.streaming.trim().is_empty() {
        out.push(Line::from(Span::styled("Lucy", Style::default().fg(PURPLE).add_modifier(Modifier::BOLD))));
        for line in format_reply_lines(app.streaming.trim()) { out.push(Line::from(vec![Span::styled("  ", Style::default().fg(BORDER)), Span::styled(line, Style::default().fg(TEXT))])); }
        out.push(Line::from(Span::styled("  ▍", Style::default().fg(BLUE))));
    } else if let Some(tool) = &app.tool_active {
        out.push(Line::from(vec![Span::styled("› ", Style::default().fg(BLUE)), Span::styled(format!("{} …", tool), Style::default().fg(TEXT))]));
    }
    if !app.progress.is_empty() {
        out.push(Line::from(""));
        for p in app.progress.iter().rev().take(5).rev() { out.push(Line::from(vec![Span::styled("  › ", Style::default().fg(PURPLE)), Span::styled(p.clone(), Style::default().fg(MUTED))])); }
    }
    if app.busy {
        let spinner = ["◐", "◓", "◑", "◒"][(app.busy_elapsed_ms() / 180 % 4) as usize];
        out.push(Line::from(vec![Span::styled(format!("{} {}", spinner, if app.phase.is_empty() { "Working" } else { &app.phase }), Style::default().fg(BLUE).add_modifier(Modifier::BOLD)), Span::styled(format!("  {}s", app.busy_elapsed_ms() / 1000), Style::default().fg(MUTED))]));
    }
    if out.is_empty() { out.push(Line::from(Span::styled("No activity yet. Tell Lucy what you want done.", Style::default().fg(MUTED)))); }
    out
}

fn visual_rows(lines: &[Line], width: usize) -> usize {
    let w = width.max(1) as u32;
    lines.iter().map(|line| { let n = line.width() as u32; if n == 0 { 1 } else { ((n + w - 1) / w).max(1) as usize } }).sum()
}

fn render_input(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App, title: &str) {
    let border = if app.listening { GREEN } else if app.busy { BLUE } else { BORDER };
    let block = Block::default().borders(Borders::ALL).border_style(Style::default().fg(border)).style(Style::default().bg(PANEL)).title(Span::styled(title, Style::default().fg(MUTED)));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let text = if app.input.is_empty() {
        if app.listening { "Speak now…" } else { "Describe what you want Lucy to do…" }
    } else { &app.input };
    frame.render_widget(Paragraph::new(text.to_owned()).style(Style::default().fg(if app.input.is_empty() { MUTED } else { TEXT_BRIGHT })).wrap(Wrap { trim: false }), inner);
    let (row, col) = cursor_row_col(&app.input, app.cursor);
    let x = inner.x.saturating_add(col as u16).min(inner.right().saturating_sub(1));
    let y = inner.y.saturating_add(row as u16).min(inner.bottom().saturating_sub(1));
    frame.set_cursor_position((x, y));
}

fn format_reply_lines(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut blank = false;
    for raw in text.replace("\r\n", "\n").lines() {
        let line = raw.trim_end();
        if line.is_empty() { if !blank { out.push(String::new()); } blank = true; continue; }
        let t = line.trim_start();
        if let Some(b) = t.strip_prefix("* ").or_else(|| t.strip_prefix("- ")) { out.push(format!("• {}", b)); }
        else { out.push(line.to_owned()); }
        blank = false;
    }
    while out.first().is_some_and(|s| s.is_empty()) { out.remove(0); }
    while out.last().is_some_and(|s| s.is_empty()) { out.pop(); }
    if out.is_empty() { out.push(String::new()); }
    out
}

fn centered(area: Rect, max_width: u16, max_height: u16) -> Rect {
    let width = area.width.min(max_width);
    let height = area.height.min(max_height);
    Rect { x: area.x + area.width.saturating_sub(width) / 2, y: area.y + area.height.saturating_sub(height) / 2, width, height }
}

fn draw_sessions_popup(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App) {
    let popup = centered(area, 88, area.height.saturating_sub(6));
    frame.render_widget(Clear, popup);
    let block = Block::default().borders(Borders::ALL).border_style(Style::default().fg(BLUE)).style(Style::default().bg(PANEL)).title(Span::styled(" Sessions · Enter switch · n new · d delete · Esc close ", Style::default().fg(TEXT_BRIGHT)));
    let inner = block.inner(popup);
    frame.render_widget(block, popup);
    if app.sessions.is_empty() { frame.render_widget(Paragraph::new("No saved sessions yet.").alignment(Alignment::Center), inner); return; }
    let items = app.sessions.iter().enumerate().map(|(i, s)| {
        let selected = i == app.sess_selected;
        let cur = short_id(&s.id.0.to_string()) == app.session_id_short;
        let style = if selected { Style::default().bg(PANEL_HI).fg(TEXT_BRIGHT) } else { Style::default().fg(TEXT) };
        let marker = if cur { "●" } else { "○" };
        ListItem::new(Line::from(Span::styled(format!("{} {:>2}. {}  ·  {} msgs", marker, i + 1, truncate_one_line(&s.title, 48), s.message_count), style))).style(style)
    }).collect::<Vec<_>>();
    frame.render_widget(List::new(items), inner);
}

fn draw_help_popup(frame: &mut ratatui::Frame<'_>, area: Rect) {
    let popup = centered(area, 82, area.height.saturating_sub(6));
    frame.render_widget(Clear, popup);
    let block = Block::default().borders(Borders::ALL).border_style(Style::default().fg(PURPLE)).style(Style::default().bg(PANEL)).title(Span::styled(" Lucy · keyboard shortcuts ", Style::default().fg(TEXT_BRIGHT)));
    let inner = block.inner(popup);
    frame.render_widget(block, popup);
    let lines = vec![
        Line::from(Span::styled("TASK", Style::default().fg(MUTED).add_modifier(Modifier::BOLD))),
        Line::from("  Enter       send task"), Line::from("  ↑ / ↓       history"), Line::from("  F2          push-to-talk voice"),
        Line::from(""), Line::from(Span::styled("SESSION", Style::default().fg(MUTED).add_modifier(Modifier::BOLD))),
        Line::from("  Ctrl+N      new session"), Line::from("  Ctrl+O      sessions"),
        Line::from(""), Line::from(Span::styled("OTHER", Style::default().fg(MUTED).add_modifier(Modifier::BOLD))),
        Line::from("  Ctrl+,      settings"), Line::from("  /           commands"), Line::from("  F1          help"), Line::from("  Esc         close / quit"),
    ];
    frame.render_widget(Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }), inner);
}

fn draw_approval_popup(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App) {
    let Some(dlg) = app.approval.as_ref() else { return };
    let popup = centered(area, 82, area.height.saturating_sub(8).min(24));
    frame.render_widget(Clear, popup);
    let block = Block::default().borders(Borders::ALL).border_style(Style::default().fg(ORANGE)).style(Style::default().bg(PANEL_HI)).title(Span::styled(" Permission needed ", Style::default().fg(ORANGE).add_modifier(Modifier::BOLD)));
    let inner = block.inner(popup);
    frame.render_widget(block, popup);
    let mut lines = vec![Line::from(vec![Span::styled("Lucy wants to use ", Style::default().fg(MUTED)), Span::styled(dlg.name.clone(), Style::default().fg(TEXT_BRIGHT).add_modifier(Modifier::BOLD))]), Line::from("")];
    for l in dlg.input.lines() { lines.push(Line::from(l.to_owned())); }
    lines.push(Line::from(""));
    lines.push(Line::from(vec![Span::styled("[y] ", Style::default().fg(GREEN).add_modifier(Modifier::BOLD)), Span::styled("allow once    ", Style::default().fg(TEXT)), Span::styled("[a] ", Style::default().fg(BLUE).add_modifier(Modifier::BOLD)), Span::styled("always    ", Style::default().fg(TEXT)), Span::styled("[n] ", Style::default().fg(RED).add_modifier(Modifier::BOLD)), Span::styled("deny", Style::default().fg(TEXT))]));
    frame.render_widget(Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }), inner);
}
