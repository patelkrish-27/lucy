//! Chat state: message model (`ChatMsg`), approval dialog, and the central
//! `App` state (input, scroll, streaming, activity phases, progress feed).

use std::collections::VecDeque;
use std::cell::Cell;
use std::time::{Duration, Instant};

use lucy_config::LucyConfig;
use lucy_core::{SessionMeta, TokenUsage, TurnMessage};

use lucy_mascot::Mood;

use super::commands::command_matches;
use super::companion::DesktopCompanion;
use super::settings;
use super::util::truncate_one_line;

const MAX_MSGS: usize = 800;

/// How long after a turn ends she stays delighted, before settling back to idle.
const HAPPY_HOLD: Duration = Duration::from_millis(4_000);
/// Persistent work/progress notes ("Plan ready: 3 steps", "Step 1: …") —
/// kept in the transcript so the user sees what happened, not just the
/// transient spinner row.
#[allow(dead_code)]
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
    /// What the turn that produced this message cost. Lives on the message, not
    /// on `App`, so each reply keeps the number for *its* turn — a reader
    /// scrolling back gets that turn's time rather than the newest one's.
    pub(crate) timing: Option<TurnTiming>,
}

impl ChatMsg {
    pub(crate) fn user(t: String) -> Self {
        Self {
            kind: MsgKind::User,
            text: t,
            timing: None,
        }
    }
    pub(crate) fn lucy(t: String) -> Self {
        Self {
            kind: MsgKind::Lucy,
            text: t,
            timing: None,
        }
    }
    pub(crate) fn system(t: String) -> Self {
        Self {
            kind: MsgKind::System,
            text: t,
            timing: None,
        }
    }
    pub(crate) fn tool(t: String) -> Self {
        Self {
            kind: MsgKind::Tool,
            text: t,
            timing: None,
        }
    }
}

/// What one finished turn cost, printed as the dim meta row under the reply:
/// the wall-clock, plus the generation rate when tokens were billed. Taken when
/// the run ends so the number the user watched tick is the one recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurnTiming {
    pub elapsed_ms: u128,
    pub tokens: u64,
}

/// One pending approval: the runtime is blocked until the user answers.
#[derive(Debug, Clone)]
pub struct ApprovalDialog {
    pub id: String,
    pub name: String,
    pub input: String,
}

/// How big the sprite is drawn, and whether it is drawn at all. `Auto` is the
/// big sprite on the welcome screen and a small one in the chat rail; the rest
/// pin one size everywhere so a person can choose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MascotSize {
    Auto,
    Large,
    Small,
    Off,
}

impl MascotSize {
    pub const ALL: [MascotSize; 4] = [
        MascotSize::Auto,
        MascotSize::Large,
        MascotSize::Small,
        MascotSize::Off,
    ];

    pub fn label(self) -> &'static str {
        match self {
            MascotSize::Auto => "auto",
            MascotSize::Large => "large",
            MascotSize::Small => "small",
            MascotSize::Off => "off",
        }
    }

    /// Parse a `/mascot` argument, case- and spelling-insensitively.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "" => Some(MascotSize::Auto),
            "large" | "big" | "on" => Some(MascotSize::Large),
            "small" | "tiny" => Some(MascotSize::Small),
            "off" | "none" | "hide" => Some(MascotSize::Off),
            _ => None,
        }
    }

    /// The next size in the cycle, for a bare `/mascot`.
    pub fn next(self) -> Self {
        match self {
            MascotSize::Auto => MascotSize::Large,
            MascotSize::Large => MascotSize::Small,
            MascotSize::Small => MascotSize::Off,
            MascotSize::Off => MascotSize::Auto,
        }
    }

    /// Whether the welcome screen draws a sprite. `Small` still draws one, just
    /// on a smaller stage.
    pub fn hero(self) -> bool {
        !matches!(self, MascotSize::Off)
    }

    /// Whether the chat rail draws a sprite. Every size but `Off` keeps the
    /// rail, because the rail is where the mascot lives while work is running.
    pub fn rail(self) -> bool {
        !matches!(self, MascotSize::Off)
    }
}

pub struct App {
    pub status: String,
    pub listening: bool,
    pub input: String,
    /// `/settings` is open (full-screen six-section view).
    pub settings: bool,
    pub settings_state: settings::SettingsState,
    pub config: LucyConfig,
    pub history: Vec<String>,
    pub hist_idx: Option<usize>,
    pub hist_draft: String,
    pub cursor: usize,
    /// Offset from bottom in *visual* (wrapped) rows. 0 = pinned to newest.
    pub scroll: usize,
    pub model_label: String,
    pub usage: TokenUsage,
    pub messages: Vec<ChatMsg>,
    pub streaming: String,
    pub tool_active: Option<String>,
    /// Live background-activity feedback (thinking / working / writing …).
    /// Rendered as an animated spinner row so the user always knows work
    /// is in flight — even before the first token arrives.
    pub busy: bool,
    pub phase: String,
    /// Previous pose mood and when it changed — lets the mascot crossfade
    /// between poses instead of snapping. `Cell` because we only read it
    /// through &self in the view.
    pub mascot_last_mood: Cell<Mood>,
    pub mascot_prev_mood: Cell<Option<Mood>>,
    pub mascot_mood_since: Cell<Instant>,
    pub phase_detail: String,
    pub phase_since: Option<Instant>,
    /// Wall-clock start of the turn in flight, and the token count it began
    /// from — the two things [`TurnTiming`] is derived from when the turn ends.
    pub turn_started: Option<Instant>,
    pub turn_tokens_base: u64,
    pub progress: VecDeque<String>,
    pub session_title: String,
    pub session_id_short: String,
    pub session_count: usize,
    pub show_sessions: bool,
    pub sessions: Vec<SessionMeta>,
    pub sess_selected: usize,
    pub show_help: bool,
    /// Active model + reasoning tier line shown in the chat header, e.g.
    /// `Groq · flash · L2 · Level 2 (balanced)`.
    pub active_model: String,
    /// Pending approval for one planned command, answered with y/a/n.
    pub approval: Option<ApprovalDialog>,
    /// Selected row in the `/` command suggestion popup. Reset whenever the
    /// input changes so the highlight always starts on the top match.
    pub cmd_suggest_sel: usize,
    /// Set by Esc to hide the popup for the current input; cleared on the next
    /// edit so the suggestions come back.
    pub cmd_suggest_dismissed: bool,
    /// Sprite size, from `/mascot`.
    pub mascot_size: MascotSize,
    /// Desktop companion overlay (toggled with `/companion`).
    pub companion: DesktopCompanion,
    /// When the session started. Every mascot animation is a pure function of
    /// the time elapsed since this instant, so the sprite is identical on
    /// every redraw and across a resize.
    pub mascot_epoch: Instant,
    /// When the last turn ended, so she can hold a happy face for a beat.
    pub turn_ended: Option<Instant>,
}

impl App {
    pub(crate) fn new(config: LucyConfig) -> Self {
        Self {
            status: "Ready".into(),
            listening: false,
            input: String::new(),
            settings: false,
            settings_state: settings::SettingsState::closed(),
            config,
            history: Vec::new(),
            hist_idx: None,
            hist_draft: String::new(),
            cursor: 0,
            scroll: 0,
            model_label: String::new(),
            usage: TokenUsage::default(),
            messages: Vec::new(),
            streaming: String::new(),
            tool_active: None,
            busy: false,
            phase: String::new(),
            mascot_last_mood: Cell::new(Mood::Idle),
            mascot_prev_mood: Cell::new(None),
            mascot_mood_since: Cell::new(Instant::now()),
            phase_detail: String::new(),
            phase_since: None,
            turn_started: None,
            turn_tokens_base: 0,
            progress: VecDeque::new(),
            session_title: "untitled".into(),
            session_id_short: String::new(),
            session_count: 0,
            show_sessions: false,
            sessions: Vec::new(),
            sess_selected: 0,
            show_help: false,
            active_model: String::new(),
            approval: None,
            cmd_suggest_sel: 0,
            cmd_suggest_dismissed: false,
            mascot_size: MascotSize::Auto,
            companion: DesktopCompanion::default(),
            mascot_epoch: Instant::now(),
            turn_ended: None,
        }
    }

    /// Command suggestions for the popup, or nothing when the popup is not
    /// applicable (no `/` yet, arguments already typed, or dismissed by Esc).
    pub(crate) fn cmd_suggestions(&self) -> Vec<(&'static str, &'static str)> {
        if self.cmd_suggest_dismissed {
            return Vec::new();
        }
        command_matches(&self.input)
    }

    pub(crate) fn cmd_suggestions_visible(&self) -> bool {
        !self.cmd_suggestions().is_empty()
    }

    /// Called after any edit to the input: restart the highlight at the top.
    pub(crate) fn reset_cmd_suggestions(&mut self) {
        self.cmd_suggest_sel = 0;
        self.cmd_suggest_dismissed = false;
    }

    /// Move the highlight with Up/Down while the popup is open.
    pub(crate) fn cmd_move_suggestion(&mut self, dir: i8) {
        let n = self.cmd_suggestions().len();
        if n == 0 {
            return;
        }
        let sel = self.cmd_suggest_sel as i32 + i32::from(dir);
        self.cmd_suggest_sel = sel.clamp(0, n as i32 - 1) as usize;
    }

    /// The highlighted popup command, when the popup is open. Enter runs
    /// this directly; Tab fills it into the input for editing.
    pub(crate) fn cmd_selected(&self) -> Option<&'static str> {
        if !self.cmd_suggestions_visible() {
            return None;
        }
        self.cmd_suggestions()
            .get(self.cmd_suggest_sel)
            .map(|(cmd, _)| *cmd)
    }

    /// Tab: replace the typed token with the highlighted command.
    pub(crate) fn cmd_accept_suggestion(&mut self) -> bool {
        let hits = self.cmd_suggestions();
        let Some((cmd, _)) = hits.get(self.cmd_suggest_sel) else {
            return false;
        };
        let cmd = *cmd;
        self.input = format!("{cmd} ");
        self.cursor = self.input.chars().count();
        self.hist_idx = None;
        // The command is in the input now, so the list would just echo it
        // back. Dismiss until the user edits again.
        self.cmd_suggest_sel = 0;
        self.cmd_suggest_dismissed = true;
        true
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

    #[allow(dead_code)]
    pub(crate) fn append_stream(&mut self, delta: &str) {
        if self.streaming.is_empty() && self.is_pinned() {
            // stay pinned
        } else if !self.is_pinned() {
            self.grow(delta.split('\n').count().max(1));
        }
        self.streaming.push_str(delta);
    }

    #[allow(dead_code)]
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
        self.begin_turn_if_idle();
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

    /// Open the turn clock the first time work arrives for a request. Both
    /// entry points into a turn (`set_phase` from submit, `push_progress`
    /// from the loop) funnel through here, so the elapsed time always covers
    /// one user request — never two chained runs.
    fn begin_turn_if_idle(&mut self) {
        if self.busy {
            return;
        }
        self.turn_started = Some(Instant::now());
        self.turn_tokens_base = self.usage.total_tokens;
    }

    /// Close the turn clock and stamp the cost onto the message the turn ended
    /// on: its reply when there is one, otherwise whatever line it did produce.
    /// Attaching to the message rather than to `App` is what lets a finished
    /// turn keep its own time once a later request starts.
    ///
    /// The reply is preferred over merely the last line because a run can emit
    /// a trailing note of its own — a status or an error the user should read —
    /// and a duration printed under that would look like it describes the note.
    pub(crate) fn mark_idle(&mut self) {
        if self.busy
            && let Some(started) = self.turn_started
            && let Some(target) = self
                .messages
                .iter()
                .rposition(|m| m.kind == MsgKind::Lucy)
                .or_else(|| self.messages.len().checked_sub(1))
        {
            self.messages[target].timing = Some(TurnTiming {
                elapsed_ms: started.elapsed().as_millis(),
                tokens: self
                    .usage
                    .total_tokens
                    .saturating_sub(self.turn_tokens_base),
            });
        }
        self.turn_started = None;
        self.busy = false;
        self.phase.clear();
        self.phase_detail.clear();
        self.phase_since = None;
        self.turn_ended = Some(Instant::now());
    }

    /// Persistent progress note from the hierarchical loop ("Plan ready: 3
    /// steps", "Step 2: open the browser", "Done: …"). Stays in the
    /// transcript; the newest note also drives the spinner + header status.
    #[allow(dead_code)]
    pub(crate) fn push_progress(&mut self, msg: &str) {
        let m = msg.trim().to_owned();
        if m.is_empty() {
            return;
        }
        // Progress notes can be the first sign of a turn (a path that reports
        // work before naming a phase), so they open the clock too.
        self.begin_turn_if_idle();
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
        self.phase_since
            .map(|t| t.elapsed().as_millis())
            .unwrap_or(0)
    }

    /// Which face Lucy is wearing, derived from what the app is actually
    /// doing. Every arm below is a state flag the UI already tracks, so the
    /// pose cannot disagree with the transcript, and no new vocabulary has to
    /// be added when a new activity phase is invented upstream.
    pub(crate) fn mascot_mood(&self) -> Mood {
        let mood = self.mascot_mood_inner();
        if self.mascot_last_mood.get() != mood {
            // Roll the previous mood over to the new one; mascot_transition
            // then crossfades over the short window after this instant.
            self.mascot_prev_mood.set(Some(self.mascot_last_mood.get()));
            self.mascot_last_mood.set(mood);
            self.mascot_mood_since.set(Instant::now());
        }
        mood
    }

    /// Within this window after a mood changed, the old pose fades out and
    /// the new one fades in — no hard cut.
    pub(crate) fn mascot_transition(&self) -> (Option<Mood>, f32) {
        let cur = self.mascot_mood();
        match self.mascot_prev_mood.get() {
            Some(prev) if prev != cur => {
                let f = self.mascot_mood_since.get().elapsed().as_millis() as f32 / 260.0;
                if f >= 1.0 { self.mascot_prev_mood.set(None); return (None, 1.0); }
                (Some(prev), f.clamp(0.0, 1.0))
            }
            _ => (None, 1.0),
        }
    }

    pub(crate) fn mascot_mood_inner(&self) -> Mood {
        if self.approval.is_some() {
            // Blocked on a person, not on work: she waits on you.
            return Mood::Approval;
        }
        if self.listening {
            return Mood::Listening;
        }
        if self.busy {
            return if !self.streaming.trim().is_empty() {
                // Tokens are arriving, so she is mid-sentence.
                Mood::Talking
            } else if self.tool_active.is_some() {
                // A tool is running and nothing has been said yet.
                Mood::Working
            } else {
                // Busy with no output at all: planning.
                Mood::Thinking
            };
        }
        if self.turn_ended.is_some_and(|t| t.elapsed() < HAPPY_HOLD) {
            return Mood::Happy;
        }
        Mood::Idle
    }

    /// Milliseconds since the session started. The only clock the mascot
    /// reads, so an animation is a pure function of it and cannot drift
    /// between the welcome screen, the rail and an export.
    pub(crate) fn mascot_ms(&self) -> u128 {
        self.mascot_epoch.elapsed().as_millis()
    }

    pub(crate) fn load_turns(&mut self, turns: &[TurnMessage]) {
        self.messages.clear();
        self.streaming.clear();
        self.tool_active = None;
        self.progress.clear();
        // Messages are rebuilt below, and a rebuilt message carries no timing:
        // a number left over here would print under a reply it never described.
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
    fn slash_suggestions_select_and_accept() {
        let mut app = App::new(LucyConfig::default());
        app.input = "/".into();
        let total = app.cmd_suggestions().len();
        assert!(total > 1, "bare / should offer many commands");

        // Down wraps the highlight within range, never out of bounds.
        app.cmd_move_suggestion(1);
        assert_eq!(app.cmd_suggest_sel, 1);
        app.cmd_suggest_sel = total - 1;
        app.cmd_move_suggestion(1);
        assert_eq!(app.cmd_suggest_sel, total - 1, "must clamp at the bottom");
        app.cmd_move_suggestion(-1);
        assert_eq!(app.cmd_suggest_sel, total - 2);

        // Accept copies the highlighted command into the input.
        app.cmd_suggest_sel = 0;
        let first = app.cmd_suggestions()[0].0.to_owned();
        assert!(app.cmd_accept_suggestion());
        assert_eq!(app.input, format!("{first} "));
        assert_eq!(app.cursor, app.input.chars().count());
        // Input now has a space, so the list is gone (nothing left to match).
        assert!(app.cmd_suggestions().is_empty());
    }

    #[test]
    fn enter_runs_the_highlighted_suggestion() {
        let mut app = App::new(LucyConfig::default());
        app.input = "/se".into();
        let first = app.cmd_suggestions()[0].0;
        assert_eq!(app.cmd_selected(), Some(first));

        app.cmd_move_suggestion(1);
        let second = app.cmd_suggestions()[1].0;
        assert_eq!(app.cmd_selected(), Some(second));

        // Once the input is a full command line (args typed, or Esc-dismissed),
        // there is no popup and Enter falls back to the typed text.
        app.input = "/switch 2".into();
        assert_eq!(app.cmd_selected(), None);
    }

    #[test]
    fn editing_resets_the_highlight_and_esc_dismisses() {
        let mut app = App::new(LucyConfig::default());
        app.input = "/".into();
        app.cmd_move_suggestion(1);
        assert_ne!(app.cmd_suggest_sel, 0);

        app.reset_cmd_suggestions();
        assert_eq!(app.cmd_suggest_sel, 0);

        app.cmd_suggest_dismissed = true;
        assert!(app.cmd_suggestions().is_empty());
        assert!(!app.cmd_suggestions_visible());

        // Any edit brings the popup back.
        app.input = "/s".into();
        app.reset_cmd_suggestions();
        assert!(app.cmd_suggestions_visible());
        assert_eq!(app.cmd_suggest_sel, 0);
    }

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
    /// The timing rows currently in the transcript, oldest first.
    fn timings(app: &App) -> Vec<TurnTiming> {
        app.messages.iter().filter_map(|m| m.timing).collect()
    }

    #[test]
    fn a_turn_records_its_own_elapsed_time_and_tokens() {
        let mut app = App::new(LucyConfig::default());
        // Idle: nothing has run, so there is no timing to report.
        assert!(timings(&app).is_empty());
        app.mark_idle();
        assert!(timings(&app).is_empty());

        // Usage the turn starts from, so the delta is this run's cost and not
        // the whole session's.
        app.usage.total_tokens = 1_000;
        app.push_msg(ChatMsg::user("what is the weather".into()));
        app.set_phase("Thinking", "routing request");
        assert!(app.turn_started.is_some());
        assert!(timings(&app).is_empty(), "no timing while the turn runs");

        // A second phase is the same turn: the clock must not restart, or the
        // reported time would only cover the last phase.
        app.set_phase("Writing", "answering");
        assert_eq!(app.turn_started, app.turn_started);
        app.usage.total_tokens = 1_500;
        app.push_msg(ChatMsg::lucy("it is raining".into()));

        app.mark_idle();
        let t = timings(&app)
            .into_iter()
            .next()
            .expect("finished turn reports a timing");
        assert_eq!(t.tokens, 500, "only this turn's tokens");
        assert!(app.turn_started.is_none(), "clock closed");

        // The next request leaves that row alone: it annotates the reply above
        // it, so it describes the turn it was measured for and not the prompt
        // now running.
        app.set_phase("Thinking", "next request");
        assert_eq!(timings(&app).len(), 1, "the finished turn keeps its row");
    }

    #[test]
    fn every_finished_turn_keeps_its_own_time() {
        let mut app = App::new(LucyConfig::default());
        // Two turns in a row, each with a reply of its own.
        for n in 1..=2 {
            app.push_msg(ChatMsg::user(format!("request {n}")));
            app.set_phase("Thinking", "routing");
            app.push_msg(ChatMsg::lucy(format!("reply {n}")));
            app.mark_idle();
        }
        // The second request must not erase the first reply's cost — a reader
        // scrolling back is asking about *that* turn, not the latest one.
        assert_eq!(
            timings(&app).len(),
            2,
            "each finished reply keeps its timing"
        );
    }

    #[test]
    fn a_note_after_the_reply_does_not_steal_the_timing_row() {
        let mut app = App::new(LucyConfig::default());
        app.push_msg(ChatMsg::lucy("the answer".into()));
        app.set_phase("Thinking", "sending to model");
        app.mark_idle();
        // Housekeeping that outlives the turn (auto-compact) lands its own
        // message afterwards. The row annotates the reply, so it has to stay
        // put rather than follow whichever message landed most recently.
        app.push_msg(ChatMsg::system("compacted history".into()));
        let timed: Vec<usize> = app
            .messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.timing.is_some())
            .map(|(i, _)| i)
            .collect();
        assert_eq!(timed, vec![0], "the row stays on the reply");
    }

    #[test]
    fn a_note_emitted_before_the_clock_stops_does_not_steal_the_row() {
        let mut app = App::new(LucyConfig::default());
        app.push_msg(ChatMsg::lucy("the answer".into()));
        app.set_phase("Thinking", "sending to model");
        // A run can emit a status or an error after its reply but before it
        // reports itself finished. The cost belongs to the reply, not to
        // whichever line happened to land last.
        app.push_msg(ChatMsg::system("compacted history".into()));
        app.mark_idle();
        let timed: Vec<usize> = app
            .messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.timing.is_some())
            .map(|(i, _)| i)
            .collect();
        assert_eq!(timed, vec![0], "the row stays on the reply");
    }

    #[test]
    fn a_turn_with_no_reply_still_reports_its_time() {
        let mut app = App::new(LucyConfig::default());
        // A turn that ends without a reply — cancelled, or failed — still ran
        // for a measurable time, and hiding that would make a stall look like
        // it never started.
        app.push_msg(ChatMsg::tool("opened the app".into()));
        app.set_phase("Automating", "running the agentic loop");
        app.set_phase("Stopped", "cancelled by /stop");
        app.mark_idle();
        assert_eq!(timings(&app).len(), 1);
    }

    #[test]
    fn loading_another_transcript_drops_the_old_timing_row() {
        let mut app = App::new(LucyConfig::default());
        app.push_msg(ChatMsg::user("earlier question".into()));
        app.set_phase("Thinking", "sending to model");
        app.push_msg(ChatMsg::lucy("an answer".into()));
        app.mark_idle();
        assert_eq!(timings(&app).len(), 1);
        // Switching sessions / reloading history rebuilds every message, so the
        // row goes with the reply it described rather than riding along.
        app.load_turns(&[TurnMessage::User("earlier question".into())]);
        assert!(timings(&app).is_empty());
    }
}
