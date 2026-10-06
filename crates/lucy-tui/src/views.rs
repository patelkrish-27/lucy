//! Rendering: welcome/active screens, chat lines, popups, input box.
//!
//! Pure view code — reads `App`, never mutates it.

use std::io::Stdout;

use lucy_mascot as mascot;
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Wrap},
};

use super::commands::COMMANDS;
use super::model::{App, MascotSize, MsgKind, TurnTiming};
use super::util::{
    cursor_row_col, format_elapsed, format_rate, format_tokens, short_id, truncate_model_label,
    truncate_one_line,
};
use crate::settings;

/// Colour mode for the sprite, resolved once per frame. On a terminal without
/// truecolor the same pixels are quantised to the 256-colour palette, so she
/// still reads as a plush purple creature rather than disappearing.
fn mascot_colors() -> mascot::ColorMode {
    mascot::color_mode()
}

pub(crate) fn is_welcome_active(app: &App) -> bool {
    // The welcome screen stays up until the user actually talks to Lucy here.
    // System notices (MCP failures, automode) and an empty transcript must not
    // flip the view, or the welcome screen would never survive startup on a
    // resumed session.
    app.streaming.is_empty()
        && !app.busy
        && !app
            .messages
            .iter()
            .any(|m| matches!(m.kind, MsgKind::User | MsgKind::Lucy))
}

pub(crate) fn draw(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    app: &App,
) -> anyhow::Result<()> {
    terminal.draw(|frame| draw_frame(frame, app))?;
    Ok(())
}

/// The per-frame body of [`draw`], split out so tests can drive it with a
/// `TestBackend` instead of a real terminal.
pub(crate) fn draw_frame(frame: &mut ratatui::Frame<'_>, app: &App) {
    {
        let area = frame.area();
        if is_welcome_active(app) {
            draw_welcome(frame, area, app)
        } else {
            draw_active(frame, area, app)
        }
        if app.settings {
            let caret = settings::draw(frame, area, &app.settings_state, &app.config);
            if let Some(r) = caret {
                frame.set_cursor_position((r.x, r.y));
            }
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
        if app.companion.is_visible() {
            draw_companion_overlay(frame, area, app);
        }
    }
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
        "[y] allow once   [a] always allow   [n] deny   (/auto on = never ask again)",
    ));
    frame.render_widget(
        Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }),
        inner,
    );
}

/// Desktop companion overlay: a small panel pinned to the bottom-right with
/// the mascot avatar (painted by `lucy_mascot` via `DesktopCompanion`) and
/// the latest companion note. Rendered last so it floats above the chat.
pub(crate) fn draw_companion_overlay(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App) {
    let w = area.width.min(36).max(20).min(area.width.saturating_sub(2));
    let h = 10.min(area.height.saturating_sub(2)).max(6);
    let popup = Rect {
        x: area.x + area.width.saturating_sub(w + 1),
        y: area.y + area.height.saturating_sub(h + 1),
        width: w,
        height: h,
    };
    frame.render_widget(Clear, popup);
    let block = Block::default()
        .title(format!(" {} ", app.companion.state().label()))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Magenta));
    frame.render_widget(block.clone(), popup);
    let inner = block.inner(popup);
    if inner.width < 6 || inner.height < 2 {
        return;
    }
    // Avatar on the left, note on the right.
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(12), Constraint::Min(8)])
        .split(inner);
    app.companion.draw_avatar(
        frame.buffer_mut(),
        cols[0],
        app.mascot_mood(),
        app.mascot_ms(),
    );
    let note = app.companion.message();
    let text = if note.trim().is_empty() {
        "Companion ready — /companion hide to dismiss.".to_owned()
    } else {
        note.to_owned()
    };
    frame.render_widget(
        Paragraph::new(Text::from(text)).wrap(Wrap { trim: false }),
        cols[1],
    );
}

fn draw_welcome(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App) {
    // The sprite is the centrepiece, so it gets the slack: `Min` lets it grow
    // into whatever the terminal has, and the fixed rows below keep the input
    // and the status line where the eye expects them.
    let v = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Min(8),
            Constraint::Length(2),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(3),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(area);
    // While the command popup is open it takes over this screen: every line
    // above the input would only be *partially* covered, and a half-hidden
    // centered line reads as a truncated fragment rather than a missing one.
    let hide_for_popup = app.cmd_suggestions_visible();
    if !hide_for_popup {
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("LUCY", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw("  ·  your computer companion"),
            ]))
            .alignment(Alignment::Center),
            v[0],
        );
        draw_mascot(frame, v[1], app, true);
    }
    // Never leave a dead gap: if work started (busy) but the transcript is
    // still empty, the welcome screen itself must show the spinner + phase.
    // (Normally submit pushes the user bubble first so we flip to the active
    // view — this is the belt-and-braces path for slow terminals like foot.)
    if app.busy && !hide_for_popup {
        const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let ms = app.busy_elapsed_ms();
        let frame_glyph = FRAMES[((ms / 100) % FRAMES.len() as u128) as usize];
        let label = if app.phase.trim().is_empty() {
            "Working".to_owned()
        } else {
            app.phase.clone()
        };
        let mut spans = vec![Span::styled(
            format!("{frame_glyph} {label}…"),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )];
        let secs = ms / 1000;
        if secs > 0 {
            spans.push(Span::styled(
                format!(" ({})", format_elapsed(ms)),
                Style::default().fg(Color::Gray).add_modifier(Modifier::DIM),
            ));
        }
        if !app.phase_detail.trim().is_empty() {
            spans.push(Span::styled(
                format!(" — {}", truncate_one_line(&app.phase_detail, 90)),
                Style::default().fg(Color::Gray).add_modifier(Modifier::DIM),
            ));
        }
        frame.render_widget(
            Paragraph::new(Line::from(spans)).alignment(Alignment::Center),
            v[2],
        );
    } else if !hide_for_popup {
        let status = if app.listening {
            "◉  Listening… release to send"
        } else {
            "What can I do for you?"
        };
        frame.render_widget(Paragraph::new(status).alignment(Alignment::Center), v[2]);
    }
    render_status_bar(frame, v[3], app);
    render_rule(frame, v[4]);
    render_input(frame, v[5], app);
    render_rule(frame, v[6]);
    frame.render_widget(
        Paragraph::new(bottom_bar(app))
            .alignment(Alignment::Center)
            .style(Style::default().add_modifier(Modifier::DIM)),
        v[7],
    );
    // Last: the popup spans the rows the status lines occupy, so it has to be
    // painted after them.
    draw_cmd_suggestions(frame, area, v[5], app);
}

/// The bottom bar from the wireframe: `/settings`, `/new` (F1), `help` (F2),
/// and `voice` on the configured push-to-talk key.
pub(crate) fn bottom_bar(app: &App) -> String {
    let ptt = app.config.voice.push_to_talk.to_ascii_uppercase();
    let parts: Vec<(&str, &str)> = vec![
        ("/settings", "Ctrl+,"),
        ("/new", "F1"),
        ("help", "F2"),
        ("voice", &ptt),
    ];
    let mut spans: Vec<Span> = Vec::new();
    for (i, (label, key)) in parts.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled(
                "   ",
                Style::default().add_modifier(Modifier::DIM),
            ));
        }
        spans.push(Span::styled(
            *label,
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(
            format!(" ({key})"),
            Style::default().fg(Color::Gray).add_modifier(Modifier::DIM),
        ));
    }
    Line::from(spans).to_string()
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
        // What this turn cost, printed under the line the turn ended on.
        // Rendered per message, so it belongs to the turn above it even after
        // the conversation moves on — and never while that turn is still
        // running. Keyed off the message rather than off a message kind,
        // because a turn that ended without a reply still spent real time.
        if let Some(t) = m.timing.filter(|_| !app.busy) {
            lines.push(turn_timing_line(t));
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
        let ms = app.busy_elapsed_ms();
        // The live clock ticks with the same formatter the finished row
        // freezes with, so the number the user watched is the number reported.
        spans.push(Span::styled(
            format!(" ({})", format_elapsed(ms)),
            Style::default().fg(Color::Gray).add_modifier(Modifier::DIM),
        ));
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

/// The meta row under a finished reply — `5.2s · 9.6 tok/s`. Indented under the
/// body text so it reads as a footnote to the reply rather than a new message.
fn turn_timing_line(t: TurnTiming) -> Line<'static> {
    let dim = Style::default().fg(Color::Gray).add_modifier(Modifier::DIM);
    let mut spans = vec![Span::styled(
        format!("      {}", format_elapsed(t.elapsed_ms)),
        dim,
    )];
    if let Some(rate) = format_rate(t.tokens, t.elapsed_ms) {
        spans.push(Span::styled(format!(" · {rate}"), dim));
    }
    Line::from(spans)
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

/// How many columns the chat rail may take, or `0` for no rail at all.
///
/// The rail is a luxury: it only appears when the transcript can still hold a
/// readable line after losing the columns, and when the mascot is switched on.
/// Everything here is arithmetic on the area, not a device or a terminal
/// guesswork, so the same layout maths is testable without a screen.
fn rail_width(app: &App, chat_row: Rect) -> u16 {
    if !app.mascot_size.rail() {
        return 0;
    }
    let widest = match app.mascot_size {
        // `Large` asks for a proper portrait, so it gives up more columns.
        MascotSize::Large => 20,
        _ => 14,
    };
    let need = widest + 20;
    if chat_row.width < need {
        return 0;
    }
    // Never more than a third of the screen, however wide it gets.
    (chat_row.width / 3).min(widest)
}

/// Draw the sprite, centred in `area`, sized to fit.
fn draw_mascot(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App, hero: bool) {
    if area.width < 8 || area.height < 6 {
        return;
    }
    if hero && !app.mascot_size.hero() {
        return;
    }
    // `Small` means a small portrait, so the sprite gets a smaller stage rather
    // than a shrunken drawing: the painter scales cleanly, and a full-width
    // stage with a tiny creature in it would just waste the space.
    let stage = if hero && app.mascot_size == MascotSize::Small {
        let w = (area.width as f32 * 0.62) as u16;
        let h = (area.height as f32 * 0.62) as u16;
        Rect::new(
            area.x + (area.width - w) / 2,
            area.y + (area.height - h) / 2,
            w,
            h,
        )
    } else {
        area
    };
    let mood = app.mascot_mood();
    let (prev, alpha) = app.mascot_transition();
    if alpha < 1.0 {
        if let Some(prev) = prev {
            if !mascot::draw_image_protocol_blend(frame, stage, prev, mood, alpha) {
                mascot::draw_blend(
                    frame.buffer_mut(),
                    stage,
                    prev,
                    mood,
                    alpha,
                    app.mascot_ms(),
                    mascot_colors(),
                );
            }
            return;
        }
    }
    if !mascot::draw_image_protocol(frame, stage, mood) {
        mascot::draw(
            frame.buffer_mut(),
            stage,
            mood,
            app.mascot_ms(),
            mascot_colors(),
        );
    }

    // Under the hero: a caption in the mood's own colour, so the text agrees
    // with the face above it. The rail is too narrow for one, so it keeps the
    // chat box's full height.
    if !hero {
        return;
    }
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            mood.label(),
            Style::default()
                .fg(mood.accent())
                .add_modifier(Modifier::BOLD),
        )))
        .alignment(Alignment::Center),
        Rect::new(
            area.x,
            area.y + area.height.saturating_sub(1),
            area.width,
            1,
        ),
    );
}

fn draw_active(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App) {
    let v = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(3),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(area);
    // The sprite gets a rail down the left of the transcript when the terminal
    // is wide enough to spare the columns. Below the width threshold the chat
    // takes the full width again, so a narrow window never loses text to a
    // mascot.
    let rail = rail_width(app, v[0]);
    let chat = if rail > 0 {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(rail), Constraint::Min(20)])
            .split(v[0]);
        draw_mascot(frame, cols[0], app, false);
        cols[1]
    } else {
        v[0]
    };

    let lines = build_chat_lines(app);
    let inner_h = chat.height as usize;
    let inner_w = chat.width as usize;
    let total = visual_rows(&lines, inner_w.max(1));
    let max_scroll = total.saturating_sub(inner_h.max(1));
    let clamped = app.scroll.min(max_scroll);
    // Paragraph scroll = rows from top; offset-from-bottom = max - clamped.
    let scroll_top = max_scroll.saturating_sub(clamped) as u16;
    frame.render_widget(
        Paragraph::new(Text::from(lines))
            .wrap(Wrap { trim: false })
            .scroll((scroll_top, 0)),
        chat,
    );
    if clamped > 0 {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "▲ scrolled — End to latest",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::DIM),
            )))
            .alignment(Alignment::Right),
            Rect {
                x: chat.x,
                y: chat.y,
                width: chat.width,
                height: 1,
            },
        );
    }

    // Slash-command suggestions, drawn just above the input line.
    draw_cmd_suggestions(frame, area, v[3], app);
    render_status_bar(frame, v[1], app);
    render_rule(frame, v[2]);
    render_input(frame, v[3], app);
    render_rule(frame, v[4]);
    frame.render_widget(
        Paragraph::new(bottom_bar(app))
            .alignment(Alignment::Center)
            .style(Style::default().add_modifier(Modifier::DIM)),
        v[5],
    );
}

/// The active model + reasoning tier for the header. The tier is kept short
/// (`L1`/`L2`/`L3`) so the line fits the header column; the full tier name is
/// shown in the activity block title.
pub(crate) fn active_model_label(app: &App) -> String {
    let live = app.active_model.trim();
    if !live.is_empty() {
        // Drop the trailing " · Level N (…)" clause, keep the tier short name.
        let head = live
            .split(" · Level ")
            .next()
            .unwrap_or(live)
            .trim()
            .to_owned();
        return truncate_one_line(&head, 26);
    }
    truncate_model_label(&app.model_label)
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
        "  F1 new session (/new) · F2 this help (help) · Ctrl+N new · Ctrl+O sessions · Ctrl+, settings (model, provider) · Esc cancel/quit",
    ));
    lines.push(Line::from(
        "  Ctrl+C stop the running task immediately (works during a permission prompt)",
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

/// The `/` command suggestion popup: one row per match, `command` plus its
/// description, with the highlighted row in an inverted cyan bar. It grows
/// upward from the top of the input box so the caret and typed text stay
/// visible underneath.
fn draw_cmd_suggestions(frame: &mut ratatui::Frame<'_>, screen: Rect, input: Rect, app: &App) {
    let hits = app.cmd_suggestions();
    if hits.is_empty() {
        return;
    }
    let sel = app.cmd_suggest_sel.min(hits.len() - 1);

    // Width = longest "command  description", plus borders; never wider than
    // the input box, never taller than the room above it.
    let content = hits
        .iter()
        .map(|(c, d)| (c.chars().count() + 3 + d.chars().count()) as u16)
        .max()
        .unwrap_or(0);
    let width = content
        .saturating_add(2)
        .max(" commands ".len() as u16 + 4)
        .min(input.width);
    let want_h = hits.len() as u16 + 2;
    let room_above = input.y.saturating_sub(screen.y);
    let height = want_h.min(room_above.max(1)).max(3).min(screen.height);
    let y = input.y.saturating_sub(height);
    let area = Rect {
        x: input.x,
        y,
        width,
        height,
    };
    let inner_w = width.saturating_sub(2) as usize;
    // Scroll the window so the highlighted row is always on screen; a bare `/`
    // lists every command but only the rows that fit above the input are shown.
    let rows = height.saturating_sub(2) as usize;
    let scroll = sel.saturating_add(1).saturating_sub(rows);
    let end = (scroll + rows).min(hits.len());

    let lines: Vec<Line<'_>> = hits[scroll..end]
        .iter()
        .enumerate()
        .map(|(i, (cmd, desc))| {
            let is_sel = i + scroll == sel;
            let cmd_style = if is_sel {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Cyan)
            };
            let desc_style = if is_sel {
                Style::default().fg(Color::Black).bg(Color::Cyan)
            } else {
                Style::default().fg(Color::Gray).add_modifier(Modifier::DIM)
            };
            let desc_max = inner_w.saturating_sub(cmd.len() + 3);
            let shown = if desc_max == 0 {
                String::new()
            } else {
                truncate_one_line(desc, desc_max)
            };
            let used = cmd.len() + 3 + shown.chars().count();
            let pad = inner_w.saturating_sub(used);
            Line::from(vec![
                Span::styled(format!(" {cmd}"), cmd_style),
                Span::styled(format!("  {shown}"), desc_style),
                Span::styled(" ".repeat(pad), desc_style),
            ])
        })
        .collect();

    frame.render_widget(
        Paragraph::new(lines)
            .block(Block::default().title(" commands ").borders(Borders::ALL))
            .wrap(Wrap { trim: false }),
        area,
    );
}

/// Hermes-style status row above the composer:
/// `✦ model | ctx 12k | 5.2s | ◉ 0.4s` — mood-coloured star, active model,
/// token count, last turn's time, and the live clock of the running turn.
fn render_status_bar(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App) {
    let dim = Style::default().fg(Color::Gray).add_modifier(Modifier::DIM);
    let mood = app.mascot_mood();
    let mut spans = vec![
        Span::styled(
            "✦ ",
            Style::default()
                .fg(mood.accent())
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            truncate_one_line(&active_model_label(app), 24),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("  |  ", dim),
        Span::styled("ctx ", dim),
        Span::styled(
            if app.usage.total_tokens == 0 {
                "--".to_owned()
            } else {
                format_tokens(app.usage.total_tokens)
            },
            Style::default().fg(Color::White),
        ),
        Span::styled("  |  ", dim),
    ];
    // The frozen duration of the last finished turn — what `5s` is in the
    // reference UI — or `--` before anything has run.
    let last = app
        .messages
        .iter()
        .rev()
        .find_map(|m| m.timing.map(|t| t.elapsed_ms));
    spans.push(Span::styled(
        match last {
            Some(ms) => format_elapsed(ms),
            None => "--".to_owned(),
        },
        Style::default().fg(Color::White),
    ));
    spans.push(Span::styled("  |  ", dim));
    if app.busy {
        spans.push(Span::styled(
            format!("◉ {}", format_elapsed(app.busy_elapsed_ms())),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ));
    } else if app.listening {
        spans.push(Span::styled(
            "◉ Listening…",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        ));
    } else {
        spans.push(Span::styled("◉ 0s", dim));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// The horizontal rule that brackets the composer like the reference UI.
fn render_rule(frame: &mut ratatui::Frame<'_>, area: Rect) {
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            "─".repeat(area.width as usize),
            Style::default()
                .fg(Color::Rgb(120, 90, 70))
                .add_modifier(Modifier::DIM),
        ))),
        area,
    );
}

fn render_input(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App) {
    let prompt_style = if app.input.starts_with('/') {
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD)
    };
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("❯ ", prompt_style),
            Span::styled(app.input.clone(), Style::default().fg(Color::White)),
        ]))
        .wrap(Wrap { trim: false }),
        area,
    );
    let (row, col) = cursor_row_col(&app.input, app.cursor);
    let y = area
        .y
        .saturating_add(row as u16)
        .min(area.bottom().saturating_sub(1));
    // Row 0: the `❯ ` prefix shifts the text two cells right; continuation
    // rows start at the left edge, so no offset.
    let base_x = if row == 0 {
        area.x.saturating_add(2)
    } else {
        area.x
    };
    let x = base_x
        .saturating_add(col as u16)
        .min(area.right().saturating_sub(1));
    frame.set_cursor_position((x, y));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ChatMsg;
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
    #[test]
    #[ignore = "visual snapshot; run with --ignored --nocapture"]
    fn print_welcome_view() {
        let app = App::new(lucy_config::LucyConfig::default());
        let text = render_text(&app, 100, 30);
        println!("{text}");
    }
    #[test]
    #[ignore = "visual snapshot; run with --ignored --nocapture"]
    fn print_chat_view() {
        use ratatui::{Terminal, backend::TestBackend};
        let mut app = App::new(lucy_config::LucyConfig::default());
        app.session_title = "despacito".into();
        app.active_model = "Groq · pro · L3 · Level 3 (deep reasoning)".into();
        app.model_label = "groq/pro".into();
        app.status = "Ready".into();
        app.push_msg(ChatMsg::user("open yt & play this song".into()));
        app.push_msg(ChatMsg::system(
            "🧭 intent: no (93%) · reasoning level 3 (L3)".into(),
        ));
        app.push_msg(ChatMsg::tool(
            "Step 1/3: Open YouTube (browser_open)".into(),
        ));
        app.push_msg(ChatMsg::tool(
            "Step 2/3: Search for the song (browser_navigate)".into(),
        ));
        app.push_msg(ChatMsg::lucy(
            "✔ Goal completed: open yt & play this song (3 step(s))".into(),
        ));
        app.messages.last_mut().expect("reply").timing = Some(TurnTiming {
            elapsed_ms: 5_200,
            tokens: 50,
        });
        let backend = TestBackend::new(100, 26);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| draw_frame(frame, &app))
            .expect("draw");
        let buffer = terminal.backend().buffer().clone();
        println!(
            "{}",
            (0..buffer.area.height)
                .map(|y| {
                    (0..buffer.area.width)
                        .map(|x| buffer[(x, y)].symbol())
                        .collect::<String>()
                })
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    /// Render one frame to a test backend and return it as plain text.
    fn render_text(app: &App, w: u16, h: u16) -> String {
        use ratatui::{Terminal, backend::TestBackend};
        let backend = TestBackend::new(w, h);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal.draw(|frame| draw_frame(frame, app)).expect("draw");
        let buffer = terminal.backend().buffer().clone();
        (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn slash_popup_lists_matching_commands_above_the_input() {
        let mut app = App::new(lucy_config::LucyConfig::default());
        app.input = "/".into();
        let text = render_text(&app, 100, 34);
        // The popup's bordered title — the input box title reads
        // "Ask lucy — / for commands", so `┌ commands ` only comes from us.
        assert!(text.contains("┌ commands "), "popup title missing:\n{text}");
        assert!(
            text.contains("start a new session"),
            "/new missing:\n{text}"
        );
        assert!(text.contains("open settings (Ctrl+,)"), "{text}");

        // Prefix filtering swaps the list as you type.
        app.input = "/se".into();
        app.reset_cmd_suggestions();
        let text = render_text(&app, 100, 26);
        assert!(text.contains("open settings"), "/settings missing:\n{text}");
        assert!(text.contains("list sessions"), "/sessions missing:\n{text}");
        assert!(
            !text.contains("start a new session"),
            "/new must drop out of a /se query:\n{text}"
        );

        // No popup once the input is not a command in progress.
        app.input = String::new();
        app.reset_cmd_suggestions();
        let text = render_text(&app, 100, 26);
        assert!(!text.contains("┌ commands "), "{text}");
        assert!(!text.contains("start a new session"), "{text}");
    }

    #[test]
    #[ignore = "visual snapshot; run with --ignored --nocapture"]
    fn print_command_suggestions() {
        use ratatui::{Terminal, backend::TestBackend};
        let mut app = App::new(lucy_config::LucyConfig::default());
        app.session_title = "despacito".into();
        app.model_label = "groq/pro".into();
        app.input = "/se".into();
        app.cmd_suggest_sel = 1;
        let backend = TestBackend::new(100, 26);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| draw_frame(frame, &app))
            .expect("draw");
        let buffer = terminal.backend().buffer().clone();
        println!(
            "{}",
            (0..buffer.area.height)
                .map(|y| {
                    (0..buffer.area.width)
                        .map(|x| buffer[(x, y)].symbol())
                        .collect::<String>()
                })
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    #[test]
    #[ignore = "visual snapshot; run with --ignored --nocapture"]
    fn print_command_suggestions_all() {
        use ratatui::{Terminal, backend::TestBackend};
        let mut app = App::new(lucy_config::LucyConfig::default());
        app.session_title = "despacito".into();
        app.model_label = "groq/pro".into();
        app.input = "/".into();
        // Last entry: exercises the scroll window (17 commands, ~19 rows fit).
        app.cmd_suggest_sel = app.cmd_suggestions().len() - 1;
        let backend = TestBackend::new(100, 26);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| draw_frame(frame, &app))
            .expect("draw");
        let buffer = terminal.backend().buffer().clone();
        println!(
            "{}",
            (0..buffer.area.height)
                .map(|y| {
                    (0..buffer.area.width)
                        .map(|x| buffer[(x, y)].symbol())
                        .collect::<String>()
                })
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    #[test]
    fn a_turn_that_ended_without_a_reply_still_shows_its_time() {
        let mut app = App::new(lucy_config::LucyConfig::default());
        // A cancelled or failed run leaves a tool line rather than a reply.
        // The duration is real work, so it has to be visible — a stamp the
        // renderer only draws for replies would print nothing at all.
        app.push_msg(ChatMsg::tool("⚙ opened the app".into()));
        app.set_phase("Automating", "running the agentic loop");
        app.mark_idle();
        app.messages.last_mut().expect("tool line").timing = Some(TurnTiming {
            elapsed_ms: 2_400,
            tokens: 12,
        });
        let text = render_text(&app, 100, 26);
        assert!(text.contains("2.4s"), "timing dropped:\n{text}");
    }

    #[test]
    fn selected_row_is_highlighted_and_scrolled_into_view() {
        use ratatui::{Terminal, backend::TestBackend};
        let mut app = App::new(lucy_config::LucyConfig::default());
        app.input = "/".into();
        let total = app.cmd_suggestions().len();
        assert!(total > 9, "fixture needs a long list, got {total}");
        app.cmd_suggest_sel = total - 1; // /quit

        let backend = TestBackend::new(100, 14);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| draw_frame(frame, &app))
            .expect("draw");
        let buffer = terminal.backend().buffer().clone();

        // Every highlighted cell uses the cyan bar; find the row that has one.
        let rows_with_hl: Vec<u16> = (0..buffer.area.height)
            .filter(|y| (0..buffer.area.width).any(|x| buffer[(x, *y)].bg == Color::Cyan))
            .collect();
        assert_eq!(
            rows_with_hl.len(),
            1,
            "exactly one row should be highlighted, got {rows_with_hl:?}"
        );

        // Short terminal: the window must scroll so the last command is shown
        // rather than clipped off the top/bottom.
        let text: String = (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("/quit"), "selected /quit off-screen:\n{text}");
        assert!(
            !text.contains("start a new session"),
            "scrolled-out rows should not be shown:\n{text}"
        );
        // The row directly above the selected one must still be listed, so the
        // window scrolled rather than clipping everything above the cursor.
        // Derived from the table instead of a literal: a hardcoded command name
        // here went stale the moment that command was removed.
        let above = COMMANDS[COMMANDS.len() - 2].0;
        assert!(
            text.contains(above),
            "row above the selection ({above}) must still be listed:\n{text}"
        );
    }

    #[test]
    fn narrow_terminal_still_renders_the_popup() {
        let mut app = App::new(lucy_config::LucyConfig::default());
        app.input = "/".into();
        // 40x10: the popup is clamped to the input width and the room above.
        let text = render_text(&app, 40, 10);
        assert!(text.contains("/"), "{text}");
    }

    #[test]
    fn bottom_bar_lists_the_wireframe_actions() {
        let mut app = App::new(lucy_config::LucyConfig::default());
        app.config.voice.push_to_talk = "f2".into();
        let bar = bottom_bar(&app);
        for expected in [
            "/settings",
            "(Ctrl+,)",
            "/new",
            "(F1)",
            "help",
            "(F2)",
            "voice",
            "(F2)",
        ] {
            assert!(
                bar.contains(expected),
                "bottom bar missing {expected}: {bar}"
            );
        }
    }

    #[test]
    fn active_model_label_prefers_the_routed_tier_line() {
        let mut app = App::new(lucy_config::LucyConfig::default());
        app.model_label = "groq/flash".into();
        assert_eq!(active_model_label(&app), "groq/flash");
        app.active_model = "Groq · pro · L3 · Level 3 (deep reasoning)".into();
        let label = active_model_label(&app);
        assert!(label.contains("L3"), "{label}");
        // Truncated to a single header-sized line (26 chars + the ellipsis).
        assert!(label.chars().count() <= 27, "{label}");
        assert!(!label.contains('\n'), "{label}");
    }

    #[test]
    fn a_finished_turn_reports_its_own_time_and_rate() {
        let mut app = App::new(lucy_config::LucyConfig::default());
        app.push_msg(ChatMsg::user("hello".into()));
        app.push_msg(ChatMsg::lucy("Hello! How can I help?".into()));
        // Nothing has run yet: no row, so a fresh chat has no stale number.
        let text = render_text(&app, 100, 26);
        assert!(!text.contains("tok/s"), "row before any run:\n{text}");

        app.set_phase("Thinking", "sending to model");
        // A turn in flight must not print a timing for the reply it is still
        // working on.
        let text = render_text(&app, 100, 26);
        assert!(
            !text.lines().any(|l| l.contains("tok/s")),
            "row while busy:\n{text}"
        );

        // The run ends; the reply lands and the clock is frozen.
        app.mark_idle();
        // Pin the timing to fixed values: the view is under test, not the
        // wall-clock (which `model.rs` covers with its own test).
        app.messages.last_mut().expect("reply").timing = Some(TurnTiming {
            elapsed_ms: 5_200,
            tokens: 50,
        });
        let text = render_text(&app, 100, 26);
        assert!(text.contains("5.2s"), "elapsed missing:\n{text}");
        assert!(text.contains("9.6 tok/s"), "rate missing:\n{text}");

        // No tokens billed (a turn that never reached a model call): the time
        // still reports, the rate is not invented.
        app.messages.last_mut().expect("reply").timing = Some(TurnTiming {
            elapsed_ms: 187_000,
            tokens: 0,
        });
        let text = render_text(&app, 100, 26);
        assert!(text.contains("3m 7s"), "long run missing:\n{text}");
        assert!(!text.contains("tok/s"), "invented a rate:\n{text}");
    }

    #[test]
    fn the_row_lands_under_its_own_reply_not_the_newest_one() {
        let mut app = App::new(lucy_config::LucyConfig::default());
        // A short first reply and a long second one, so their rows land at
        // different offsets and a row rendered under the wrong reply is visible.
        for (reply, ms) in [("first reply", 1_000u128), ("second reply", 9_000)] {
            app.push_msg(ChatMsg::user(format!("ask about {reply}")));
            app.set_phase("Thinking", "sending to model");
            app.push_msg(ChatMsg::lucy(reply.into()));
            app.mark_idle();
            app.messages.last_mut().expect("reply").timing = Some(TurnTiming {
                elapsed_ms: ms,
                tokens: 10,
            });
        }
        let text = render_text(&app, 100, 26);
        let first = text.find("first reply").expect("first reply on screen");
        let second = text.find("second reply").expect("second reply on screen");
        assert!(first < second, "replies out of order:\n{text}");
        // Each number sits between its own reply and the next one.
        assert!(
            text[first..second].contains("1.0s"),
            "first row missing or misplaced:\n{text}"
        );
        assert!(
            text[second..].contains("9.0s"),
            "second row missing or misplaced:\n{text}"
        );
    }

    #[test]
    fn welcome_screen_deactivates_on_busy_or_messages() {
        let mut app = App::new(lucy_config::LucyConfig::default());
        assert!(is_welcome_active(&app), "fresh app should show welcome");

        app.set_phase("Thinking", "routing request");
        assert!(
            !is_welcome_active(&app),
            "busy app must transition away from welcome immediately"
        );

        app.mark_idle();
        assert!(is_welcome_active(&app));

        app.push_msg(ChatMsg::system("1 MCP server(s) failed to start".into()));
        assert!(
            is_welcome_active(&app),
            "system notices alone must not hide the welcome screen"
        );

        app.push_msg(ChatMsg::user("hello".into()));
        assert!(
            !is_welcome_active(&app),
            "app with messages must show active chat"
        );
    }
}
