//! Rendering: welcome/active screens, chat lines, popups, input box.
//!
//! Pure view code — reads `App`, never mutates it.

use std::io::Stdout;

use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Wrap},
};

use super::commands::{COMMANDS, command_hint};
use super::model::{App, MsgKind};
use super::util::{
    cursor_row_col, format_tokens, short_id, truncate_model_label, truncate_one_line,
};
use crate::settings;

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

pub(crate) fn draw(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    app: &App,
) -> anyhow::Result<()> {
    terminal.draw(|frame| {
        let area = frame.area();
        if app.settings {
            if app.messages.is_empty() && app.streaming.is_empty() {
                draw_welcome(frame, area, app);
            } else {
                draw_active(frame, area, app);
            }
            settings::draw(frame, area, &app.config, app.settings_selected);
        } else if app.messages.is_empty() && app.streaming.is_empty() {
            draw_welcome(frame, area, app)
        } else {
            draw_active(frame, area, app)
        }
        if app.show_sessions {
            draw_sessions_popup(frame, area, app);
        }
        if app.show_help {
            draw_help_popup(frame, area);
        }
        if app.approval.is_some() {
            draw_approval_popup(frame, area, app);
        }
    })?;
    Ok(())
}

fn draw_approval_popup(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App) {
    let Some(dlg) = app.approval.as_ref() else {
        return;
    };
    let w = (area.width * 70 / 100)
        .max(20)
        .min(area.width.saturating_sub(2));
    let h = (area.height * 50 / 100)
        .max(10)
        .min(area.height.saturating_sub(2));
    let x = area.x + area.width.saturating_sub(w) / 2;
    let y = area.y + area.height.saturating_sub(h) / 2;
    let popup = Rect {
        x,
        y,
        width: w,
        height: h,
    };
    frame.render_widget(Clear, popup);
    let block = Block::default()
        .title(" Approval needed ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow));
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
    lines.push(Line::from(
        "[y] allow once   [a] always allow   [n] deny   (tip: /auto on = never ask)",
    ));
    frame.render_widget(
        Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }),
        inner,
    );
}

fn draw_welcome(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App) {
    let v = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),
            Constraint::Length(11),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
        ])
        .split(area);
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("LUCY", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw("  ·  your computer companion"),
        ]))
        .alignment(Alignment::Center),
        v[0],
    );
    let mascot = Text::from(
        MASCOT
            .iter()
            .map(|x| Line::from(Span::raw(*x)))
            .collect::<Vec<_>>(),
    );
    frame.render_widget(Paragraph::new(mascot).alignment(Alignment::Center), v[1]);
    let status = if app.listening {
        "◉  Listening… release to send"
    } else {
        "What can I do for you?"
    };
    frame.render_widget(Paragraph::new(status).alignment(Alignment::Center), v[2]);
    render_input(
        frame,
        v[3],
        app,
        "  Tell Lucy what to do — / for commands  ",
    );
    let ptt = app.config.voice.push_to_talk.to_ascii_uppercase();
    frame.render_widget(
        Paragraph::new(format!(
            "[Enter] send  [/] commands  [Hold {}] voice  [Ctrl+O] sessions  [F1] help  [Esc] quit",
            ptt
        ))
        .alignment(Alignment::Center)
        .style(Style::default().add_modifier(Modifier::DIM)),
        v[4],
    );
}

fn build_chat_lines(app: &App) -> Vec<Line<'static>> {
    let mut lines: Vec<Line> = Vec::new();
    for m in &app.messages {
        match m.kind {
            MsgKind::User => {
                lines.push(Line::from(vec![
                    Span::styled(
                        "You   ",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(m.text.clone(), Style::default().fg(Color::White)),
                ]));
            }
            MsgKind::Lucy => {
                for (i, bl) in format_reply_lines(&m.text).iter().enumerate() {
                    lines.push(render_body_line(bl, i == 0));
                }
            }
            MsgKind::System => {
                for bl in format_reply_lines(&m.text) {
                    lines.push(Line::from(vec![
                        Span::styled(
                            "· ",
                            Style::default()
                                .fg(Color::Yellow)
                                .add_modifier(Modifier::DIM),
                        ),
                        Span::styled(
                            bl,
                            Style::default()
                                .fg(Color::Yellow)
                                .add_modifier(Modifier::DIM),
                        ),
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
        for (i, bl) in format_reply_lines(app.streaming.trim()).iter().enumerate() {
            lines.push(render_body_line(bl, i == 0 && true));
        }
        lines.push(Line::from(Span::styled(
            "▍",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::DIM),
        )));
        lines.push(Line::from(""));
    } else if let Some(tool) = &app.tool_active {
        lines.push(Line::from(vec![
            Span::styled("⚙ ", Style::default().fg(Color::Magenta)),
            Span::styled(
                format!("Using {tool}…"),
                Style::default().fg(Color::Gray).add_modifier(Modifier::DIM),
            ),
        ]));
        lines.push(Line::from(""));
    }
    // Persistent progress feed: what the planner is doing / did.
    for p in &app.progress {
        lines.push(Line::from(vec![
            Span::styled(
                "› ",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::DIM),
            ),
            Span::styled(
                p.clone(),
                Style::default().fg(Color::Gray).add_modifier(Modifier::DIM),
            ),
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
        let label = if app.phase.trim().is_empty() {
            "Working".to_owned()
        } else {
            app.phase.clone()
        };
        let mut spans = vec![Span::styled(
            format!("{frame} {label}…"),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )];
        let secs = ms / 1000;
        if secs > 0 {
            spans.push(Span::styled(
                format!(" ({secs}s)"),
                Style::default().fg(Color::Gray).add_modifier(Modifier::DIM),
            ));
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
        lines.push(Line::from(Span::styled(
            "No messages yet — say hi or type /help.",
            Style::default().add_modifier(Modifier::DIM),
        )));
    }
    lines
}

/// Visual (wrapped) row count — the fix for "new chats not visible".
/// Ratatui's scroll offset operates on wrapped rows, but the old code counted
/// logical lines only, so after enough wrapping the computed bottom fell short
/// and the viewport got stuck above the newest messages.
fn visual_rows(lines: &[Line], inner_width: usize) -> usize {
    let w = inner_width.max(1) as u32;
    lines
        .iter()
        .map(|l| {
            let lw = l.width() as u32;
            if lw == 0 {
                1
            } else {
                ((lw + w - 1) / w).max(1) as usize
            }
        })
        .sum()
}

fn draw_active(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App) {
    let v = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Min(3),
            Constraint::Length(3),
            Constraint::Length(2),
        ])
        .split(area);
    let h = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(10), Constraint::Length(34)])
        .split(v[0]);
    let sess = if app.session_title.trim().is_empty() {
        "untitled".to_owned()
    } else {
        app.session_title.clone()
    };
    let left = format!("● LUCY  ·  {sess}");
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("● LUCY", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format!("  ·  {sess}")),
        ]))
        .style(Style::default()),
        h[0],
    );
    let _ = left;
    let right_text = if app.listening {
        "◉ Listening…".to_owned()
    } else {
        let m = truncate_model_label(&app.model_label);
        let tokens = format_tokens(app.usage.total_tokens);
        if m.is_empty() {
            format!("{tokens} · {}", app.status)
        } else {
            format!("{m} · {tokens} · {}", app.status)
        }
    };
    frame.render_widget(Paragraph::new(right_text).alignment(Alignment::Right), h[1]);

    let lines = build_chat_lines(app);
    let inner_h = v[1].height.saturating_sub(2) as usize;
    let inner_w = v[1].width.saturating_sub(2) as usize;
    let inner = inner_h.max(1);
    let total = visual_rows(&lines, inner_w.max(1));
    let max_scroll = total.saturating_sub(inner);
    let clamped = app.scroll.min(max_scroll);
    // Paragraph scroll = rows from top; offset-from-bottom = max - clamped.
    let scroll_top = max_scroll.saturating_sub(clamped) as u16;
    let title = if clamped > 0 {
        format!(" Activity · {} · ▲ scrolled (End to latest) ", sess)
    } else {
        format!(" Activity · {} ", sess)
    };
    frame.render_widget(
        Paragraph::new(Text::from(lines))
            .block(Block::default().title(title).borders(Borders::ALL))
            .wrap(Wrap { trim: false })
            .scroll((scroll_top, 0)),
        v[1],
    );

    // Slash-command autocomplete hint above the input.
    if app.input.starts_with('/') {
        if let Some(hint) = command_hint(&app.input) {
            let hint_area = Rect {
                x: v[2].x,
                y: v[2].y.saturating_sub(1),
                width: v[2].width,
                height: 1,
            };
            frame.render_widget(
                Paragraph::new(hint)
                    .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::DIM)),
                hint_area,
            );
        }
    }
    let input_title = if app.input.starts_with('/') {
        "  Command  "
    } else {
        "  Ask Lucy — / for commands  "
    };
    render_input(frame, v[2], app, input_title);
    frame.render_widget(Paragraph::new("[Enter] send   [/] commands   [Ctrl+N] new   [Ctrl+O] sessions   [F1] help   [Esc] quit").alignment(Alignment::Center).style(Style::default().add_modifier(Modifier::DIM)),v[3]);
}

fn draw_sessions_popup(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App) {
    let popup = Rect {
        x: area.width / 8,
        y: area.height / 8,
        width: area.width * 6 / 8,
        height: area.height * 6 / 8,
    };
    frame.render_widget(Clear, popup);
    let block = Block::default()
        .title(" Sessions — ↑/↓ select · Enter switch · d delete · n new · Esc close ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));
    frame.render_widget(block.clone(), popup);
    let inner = block.inner(popup);
    if app.sessions.is_empty() {
        frame.render_widget(
            Paragraph::new("No sessions yet — press n for a new one.").alignment(Alignment::Center),
            inner,
        );
        return;
    }
    let items: Vec<ListItem> = app
        .sessions
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let cur = short_id(&m.id.0.to_string()) == app.session_id_short;
            let style = if i == app.sess_selected {
                Style::default()
                    .bg(Color::Rgb(45, 45, 65))
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            let marker = if cur { "● " } else { "  " };
            let line = format!(
                "{marker}{:>2}. {}  [{} msgs]  {}",
                i + 1,
                truncate_one_line(&m.title, 40),
                m.message_count,
                short_id(&m.id.0.to_string())
            );
            ListItem::new(Line::from(vec![Span::styled(line, style)])).style(style)
        })
        .collect();
    frame.render_widget(List::new(items), inner);
}

fn draw_help_popup(frame: &mut ratatui::Frame<'_>, area: Rect) {
    let popup = Rect {
        x: area.width / 8,
        y: area.height / 8,
        width: area.width * 6 / 8,
        height: area.height * 6 / 8,
    };
    frame.render_widget(Clear, popup);
    let block = Block::default()
        .title(" Lucy help — Esc/F1 to close ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Green));
    frame.render_widget(block.clone(), popup);
    let inner = block.inner(popup);
    let mut lines = vec![
        Line::from(Span::styled(
            "Chat",
            Style::default().add_modifier(Modifier::BOLD),
        )),
        Line::from(
            "  Enter send · Shift+chars type · Up/Down input history · PgUp/PgDn scroll · End latest",
        ),
        Line::from(""),
        Line::from(Span::styled(
            "Sessions (opencode-style)",
            Style::default().add_modifier(Modifier::BOLD),
        )),
    ];
    for (c, d) in COMMANDS {
        lines.push(Line::from(vec![
            Span::styled(format!("  {c:<10}"), Style::default().fg(Color::Cyan)),
            Span::raw(*d),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "Keys",
        Style::default().add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(
        "  Ctrl+N new session · Ctrl+O sessions · Ctrl+, settings · F1 help · Esc cancel/quit",
    ));
    frame.render_widget(
        Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }),
        inner,
    );
}

/// Break a raw assistant reply into readable display lines:
/// - run-on `* **Item:** ...` bullets each get their own line
/// - `**` bold markers and backticks are stripped (no markdown in terminal)
/// - at most one blank line in a row, no leading/trailing blanks
fn format_reply_lines(text: &str) -> Vec<String> {
    let mut s = text.replace("\r\n", "\n");
    s = s.replace("* **", "\n• ");
    s = s.replace("**", "");
    s = s.replace('`', "");
    // A bullet left mid-line (e.g. "…done. • Next…") starts its own line.
    let mut lines: Vec<String> = Vec::new();
    for raw in s.split('\n') {
        let mut rest = raw.trim_end();
        // Split off any mid-line " • " continuations.
        loop {
            if let Some(idx) = rest.find(" • ") {
                let (head, tail) = rest.split_at(idx + 1); // tail starts with "• "
                let head = head.trim_end_matches([' ', '•']).trim_end();
                if !head.is_empty() {
                    lines.push(head.to_owned());
                }
                rest = tail.trim_start();
            } else {
                break;
            }
        }
        // Strip a leftover leading "* "/"- " bullet into "• ".
        let t = rest.trim_start();
        if let Some(b) = t.strip_prefix("* ").or_else(|| t.strip_prefix("- ")) {
            lines.push(format!("• {b}"));
        } else {
            lines.push(rest.trim_end().to_owned());
        }
    }
    // Collapse 2+ blank lines into one, trim ends.
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut blank = false;
    for l in lines {
        if l.trim().is_empty() {
            if !blank {
                out.push(String::new());
            }
            blank = true;
        } else {
            out.push(l);
            blank = false;
        }
    }
    while out.first().is_some_and(|l| l.is_empty()) {
        out.remove(0);
    }
    while out.last().is_some_and(|l| l.is_empty()) {
        out.pop();
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

fn render_body_line(line: &str, first: bool) -> Line<'static> {
    let prefix = if first { "Lucy  " } else { "      " };
    let pre = Span::styled(
        prefix.to_owned(),
        Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD),
    );
    if line.trim().is_empty() {
        return Line::from("");
    }
    if let Some(rest) = line.strip_prefix("• ") {
        // Bold the lead ("Title:") so items scan easily.
        if let Some(idx) = rest.find(": ") {
            let (title, body) = rest.split_at(idx + 1);
            return Line::from(vec![
                pre,
                Span::raw("• "),
                Span::styled(
                    title.to_owned(),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::raw(body.to_owned()),
            ]);
        }
        return Line::from(vec![pre, Span::raw("• "), Span::raw(rest.to_owned())]);
    }
    Line::from(vec![pre, Span::raw(line.to_owned())])
}

fn render_input(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App, title: &str) {
    let hint = if app.input.is_empty() && !title.contains("Command") {
        app.input.clone()
    } else {
        app.input.clone()
    };
    let _ = hint;
    frame.render_widget(
        Paragraph::new(format!("> {}", app.input))
            .block(Block::default().title(title).borders(Borders::ALL))
            .wrap(Wrap { trim: true }),
        area,
    );
    let (row, col) = cursor_row_col(&app.input, app.cursor);
    let row0 = area.y.saturating_add(1);
    let max_y = area.bottom().saturating_sub(1);
    let y = row0.saturating_add(row as u16).min(max_y);
    let base_x = if row == 0 {
        area.x.saturating_add(3)
    } else {
        area.x.saturating_add(1)
    };
    let x = base_x
        .saturating_add(col as u16)
        .min(area.right().saturating_sub(1));
    frame.set_cursor_position((x, y));
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn splits_run_on_bullets_and_strips_markers() {
        let raw = "As Lucy, my superpowers center on operation:* **Direct Control:** Execute shell. * **Files:** Read with `ripgrep`.";
        let lines = format_reply_lines(raw);
        assert!(
            lines.iter().any(|l| l.starts_with("• Direct Control:")),
            "got {lines:?}"
        );
        assert!(
            lines.iter().any(|l| l.starts_with("• Files:")),
            "got {lines:?}"
        );
        assert!(
            !lines.iter().any(|l| l.contains("**") || l.contains('`')),
            "got {lines:?}"
        );
    }
    #[test]
    fn collapses_blank_lines() {
        let lines = format_reply_lines("a\n\n\nb\n");
        assert_eq!(lines, vec!["a".to_owned(), "".to_owned(), "b".to_owned()]);
    }
    #[test]
    fn visual_rows_counts_wrapping() {
        let lines = vec![Line::from("a".repeat(100))];
        // width 10 -> 10 visual rows
        assert_eq!(visual_rows(&lines, 10), 10);
        assert_eq!(visual_rows(&[Line::from("")], 10), 1);
    }
}
