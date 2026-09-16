//! Chat state: message model (`ChatMsg`), approval dialog, and the central
//! `App` state (input, scroll, streaming, activity phases, progress feed).

use std::collections::VecDeque;
use std::time::Instant;

use lucy_config::LucyConfig;
use lucy_core::{SessionMeta, TokenUsage, TurnMessage};

use super::util::truncate_one_line;

const MAX_MSGS: usize = 800;
/// Persistent work/progress notes ("Plan ready: 3 steps", "Step 1: …") —
/// kept in the transcript so the user sees what happened, not just the
/// transient spinner row.
const MAX_PROGRESS: usize = 8;

// ---- chat model (human/agent separation + streaming) ----
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsgKind {
    User,
    Lucy,
    System,
    Tool,
}

#[derive(Debug, Clone)]
pub struct ChatMsg {
    pub(crate) kind: MsgKind,
    pub(crate) text: String,
}

impl ChatMsg {
    pub(crate) fn user(t: String) -> Self {
        Self {
            kind: MsgKind::User,
            text: t,
        }
    }
    pub(crate) fn lucy(t: String) -> Self {
        Self {
            kind: MsgKind::Lucy,
            text: t,
        }
    }
    pub(crate) fn system(t: String) -> Self {
        Self {
            kind: MsgKind::System,
            text: t,
        }
    }
    pub(crate) fn tool(t: String) -> Self {
        Self {
            kind: MsgKind::Tool,
            text: t,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ApprovalDialog {
    pub id: String,
    pub name: String,
    pub input: String,
}

pub struct App {
    pub status: String,
    pub listening: bool,
    pub input: String,
    pub settings: bool,
    pub settings_selected: usize,
    pub config: LucyConfig,
    pub history: Vec<String>,
    pub hist_idx: Option<usize>,
    pub hist_draft: String,
    pub cursor: usize,
    /// Offset from bottom in *visual* (wrapped) rows. 0 = pinned to newest.
    pub scroll: usize,
    pub model_label: String,
    pub usage: TokenUsage,
    pub approval: Option<ApprovalDialog>,
    pub messages: Vec<ChatMsg>,
    pub streaming: String,
    pub tool_active: Option<String>,
    /// Live background-activity feedback (thinking / working / writing …).
    /// Rendered as an animated spinner row so the user always knows work
    /// is in flight — even before the first token arrives.
    pub busy: bool,
    pub phase: String,
    pub phase_detail: String,
    pub phase_since: Option<Instant>,
    pub progress: VecDeque<String>,
    pub session_title: String,
    pub session_id_short: String,
    pub session_count: usize,
    pub show_sessions: bool,
    pub sessions: Vec<SessionMeta>,
    pub sess_selected: usize,
    pub show_help: bool,
}

impl App {
    pub(crate) fn new(config: LucyConfig) -> Self {
        Self {
            status: "Ready".into(),
            listening: false,
            input: String::new(),
            settings: false,
            settings_selected: 0,
            config,
            history: Vec::new(),
            hist_idx: None,
            hist_draft: String::new(),
            cursor: 0,
            scroll: 0,
            model_label: String::new(),
            usage: TokenUsage::default(),
            approval: None,
            messages: Vec::new(),
            streaming: String::new(),
            tool_active: None,
            busy: false,
            phase: String::new(),
            phase_detail: String::new(),
            phase_since: None,
            progress: VecDeque::new(),
            session_title: "untitled".into(),
            session_id_short: String::new(),
            session_count: 0,
            show_sessions: false,
            sessions: Vec::new(),
            sess_selected: 0,
            show_help: false,
        }
    }

    pub(crate) fn is_pinned(&self) -> bool {
        self.scroll == 0
    }

    pub(crate) fn pin(&mut self) {
        self.scroll = 0;
    }

    /// Keep the viewport stable when new rows arrive while scrolled up.
    pub(crate) fn grow(&mut self, added_visual_rows: usize) {
        if !self.is_pinned() {
            self.scroll = self.scroll.saturating_add(added_visual_rows);
        }
    }

    pub(crate) fn push_msg(&mut self, msg: ChatMsg) {
        // Rough visual-row estimate for scroll stability (refined at draw time).
        let rows = msg.text.split('\n').count().max(1).saturating_add(1);
        self.grow(rows);
        self.messages.push(msg);
        if self.messages.len() > MAX_MSGS {
            let excess = self.messages.len() - MAX_MSGS;
            self.messages.drain(0..excess);
        }
    }

    pub(crate) fn append_stream(&mut self, delta: &str) {
        if self.streaming.is_empty() && self.is_pinned() {
            // stay pinned
        } else if !self.is_pinned() {
            self.grow(delta.split('\n').count().max(1));
        }
        self.streaming.push_str(delta);
    }

    pub(crate) fn flush_stream(&mut self) {
        let text = self.streaming.trim().to_owned();
        self.streaming.clear();
        self.tool_active = None;
        if !text.is_empty() {
            self.push_msg(ChatMsg::lucy(text));
        }
    }

    /// Mark background work as in-flight with a human phase label
    /// ("Thinking", "Writing", "Using bash", …) plus optional detail.
    /// Called the moment the user hits Enter so there is never a dead gap.
    pub(crate) fn set_phase(&mut self, phase: &str, detail: &str) {
        self.busy = true;
        self.phase = phase.to_owned();
        if !detail.trim().is_empty() {
            self.phase_detail = detail.trim().to_owned();
        }
        if self.phase_since.is_none() {
            self.phase_since = Some(Instant::now());
        }
        self.pin();
        self.status = if self.phase_detail.is_empty() {
            format!("{phase}…")
        } else {
            format!("{phase}… — {}", truncate_one_line(&self.phase_detail, 80))
        };
    }

    pub(crate) fn mark_idle(&mut self) {
        self.busy = false;
        self.phase.clear();
        self.phase_detail.clear();
        self.phase_since = None;
    }

    /// Persistent progress note from the hierarchical loop ("Plan ready: 3
    /// steps", "Step 2: open the browser", "Done: …"). Stays in the
    /// transcript; the newest note also drives the spinner + header status.
    pub(crate) fn push_progress(&mut self, msg: &str) {
        let m = msg.trim().to_owned();
        if m.is_empty() {
            return;
        }
        // Transient triage note — show only as spinner, never as persistent feed.
        if m == "Understanding your request…" {
            if self.phase.trim().is_empty() {
                self.phase = "Thinking".to_owned();
            }
            self.phase_detail = m.clone();
            self.busy = true;
            if self.phase_since.is_none() {
                self.phase_since = Some(Instant::now());
            }
            self.status = format!("{}… — {}", self.phase, truncate_one_line(&m, 80));
            return;
        }
        if self.progress.back().is_some_and(|l| l == &m) {
            return;
        }
        self.grow(1);
        self.progress.push_back(m.clone());
        while self.progress.len() > MAX_PROGRESS {
            self.progress.pop_front();
        }
        self.busy = true;
        if self.phase.trim().is_empty() {
            self.phase = "Working".to_owned();
        }
        self.phase_detail = m.clone();
        if self.phase_since.is_none() {
            self.phase_since = Some(Instant::now());
        }
        self.pin();
        self.status = format!("{}… — {}", self.phase, truncate_one_line(&m, 80));
    }

    pub(crate) fn busy_elapsed_ms(&self) -> u128 {
        self.phase_since.map(|t| t.elapsed().as_millis()).unwrap_or(0)
    }

    pub(crate) fn load_turns(&mut self, turns: &[TurnMessage]) {
        self.messages.clear();
        self.streaming.clear();
        self.tool_active = None;
        self.progress.clear();
        self.mark_idle();
        for m in turns {
            match m {
                TurnMessage::User(t) => self.messages.push(ChatMsg::user(t.clone())),
                TurnMessage::Assistant(turn) => {
                    if let Some(t) = turn.text.as_deref() {
                        if !t.trim().is_empty() {
                            self.messages.push(ChatMsg::lucy(t.to_owned()));
                        }
                    }
                    for c in &turn.tool_calls {
                        self.messages.push(ChatMsg::tool(format!(
                            "{} {}",
                            c.name,
                            truncate_one_line(&c.input.to_string(), 120)
                        )));
                    }
                }
                TurnMessage::Tool(res) => {
                    let preview = truncate_one_line(&res.output.to_string(), 160);
                    self.messages
                        .push(ChatMsg::tool(format!("{} → {}", res.name, preview)));
                }
            }
        }
        // Cap on load as well.
        if self.messages.len() > MAX_MSGS {
            let excess = self.messages.len() - MAX_MSGS;
            self.messages.drain(0..excess);
        }
        self.pin();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn streaming_appends_not_spams() {
        let mut app = App::new(LucyConfig::default());
        app.append_stream("hello ");
        app.append_stream("world");
        assert_eq!(app.streaming, "hello world");
        assert!(app.messages.is_empty());
        app.flush_stream();
        assert_eq!(app.messages.len(), 1);
        assert!(app.streaming.is_empty());
    }
    #[test]
    fn progress_feed_caps_and_dedups() {
        let mut app = App::new(LucyConfig::default());
        app.push_progress("Step 1: open the browser");
        app.push_progress("Step 1: open the browser"); // consecutive dup ignored
        assert_eq!(app.progress.len(), 1);
        assert!(app.busy);
        assert_eq!(app.phase_detail, "Step 1: open the browser");
        for i in 0..20 {
            app.push_progress(&format!("note {i}"));
        }
        assert_eq!(app.progress.len(), MAX_PROGRESS);
        assert!(app.progress.back().unwrap().starts_with("note "));
    }
    #[test]
    fn activity_phase_tracks_busy_until_idle() {
        let mut app = App::new(LucyConfig::default());
        assert!(!app.busy);
        app.set_phase("Thinking", "sending to model");
        assert!(app.busy);
        assert_eq!(app.phase, "Thinking");
        assert!(app.phase_since.is_some());
        app.set_phase("Writing", "");
        assert_eq!(app.phase, "Writing");
        // Empty detail must not wipe the previous detail.
        assert_eq!(app.phase_detail, "sending to model");
        app.mark_idle();
        assert!(!app.busy);
        assert!(app.phase_since.is_none());
    }
}
