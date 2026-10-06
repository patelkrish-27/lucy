//! Screen context for the on-screen companion.
//!
//! A [`ScreenCapture`] is a point-in-time description of what is visible: the
//! active window's title and class plus a free-form summary of the observed
//! state (e.g. an accessibility-tree digest produced by the caller). A
//! [`ScreenContext`] holds the latest capture and renders it into a model
//! prompt next to the user's question.
//!
//! Deliberately generic: this module never inspects the *content* of the
//! question or the summary to decide what to do. No site, app, or task
//! vocabulary lives here — the model does the routing, this module only
//! carries the context (see AGENTS.md).

use std::time::{SystemTime, UNIX_EPOCH};

use lucy_config::CompanionConfig;

/// One observed screen state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScreenCapture {
    /// Title of the active window at capture time (`""` when unknown).
    pub window_title: String,
    /// Window class / app id at capture time (`""` when unknown).
    pub window_class: String,
    /// Free-form summary of the observed state. Produced by the caller;
    /// never interpreted here.
    pub screen_state: String,
    /// Unix seconds when the capture was taken (`0` = unspecified, which is
    /// what the pure [`ScreenContext::capture_from_summary`] constructor uses
    /// so it stays deterministic and testable).
    pub timestamp: u64,
}

impl ScreenCapture {
    pub fn new(
        window_title: impl Into<String>,
        window_class: impl Into<String>,
        screen_state: impl Into<String>,
        timestamp: u64,
    ) -> Self {
        Self {
            window_title: window_title.into(),
            window_class: window_class.into(),
            screen_state: screen_state.into(),
            timestamp,
        }
    }

    /// Current time as unix seconds; saturates to `0` before the epoch.
    pub fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}

/// The latest screen capture, if any, to attach to a turn.
#[derive(Debug, Clone, Default)]
pub struct ScreenContext {
    pub capture: Option<ScreenCapture>,
}

impl ScreenContext {
    /// No capture — [`Self::build_prompt`] returns the question verbatim.
    pub fn empty() -> Self {
        Self { capture: None }
    }

    pub fn from_capture(capture: ScreenCapture) -> Self {
        Self {
            capture: Some(capture),
        }
    }

    /// Pure constructor: deterministic in its arguments (timestamp `0`,
    /// class `""`) so tests need no clock. Prefer [`Self::capture_now`]
    /// for live captures.
    pub fn capture_from_summary(summary: &str, window_title: &str) -> Self {
        Self {
            capture: Some(ScreenCapture::new(window_title, "", summary, 0)),
        }
    }

    /// Live constructor: stamps the capture with the current time.
    pub fn capture_now(summary: &str, window_title: &str, window_class: &str) -> Self {
        Self {
            capture: Some(ScreenCapture::new(
                window_title,
                window_class,
                summary,
                ScreenCapture::now_secs(),
            )),
        }
    }

    pub fn with_window_class(mut self, class: &str) -> Self {
        if let Some(capture) = self.capture.as_mut() {
            capture.window_class = class.to_owned();
        }
        self
    }

    pub fn is_empty(&self) -> bool {
        self.capture.is_none()
    }

    /// Render the question with the latest capture attached, using default
    /// companion settings. Without a capture this is the question verbatim.
    pub fn build_prompt(&self, question: &str) -> String {
        self.build_prompt_with_config(question, &CompanionConfig::default())
    }

    /// Render the question with the latest capture attached, honouring
    /// `config`: a disabled companion (or one configured not to include
    /// screen context) yields the question verbatim, and the summary is
    /// truncated to the configured budget. The question itself is never
    /// altered or inspected.
    pub fn build_prompt_with_config(
        &self,
        question: &str,
        config: &CompanionConfig,
    ) -> String {
        let Some(capture) = self.capture.as_ref() else {
            return question.to_owned();
        };
        if !config.screens_allowed() {
            return question.to_owned();
        }
        let title = none_when_blank(&capture.window_title);
        let class = none_when_blank(&capture.window_class);
        let state = config.truncate_summary(&capture.screen_state);
        format!(
            "[Screen context]\n\
             Active window title: {title}\n\
             Window class: {class}\n\
             Captured at (unix secs): {ts}\n\
             Observed state:\n\
             {state}\n\
             \n\
             User request:\n\
             {question}",
            ts = capture.timestamp,
        )
    }
}

fn none_when_blank(raw: &str) -> &str {
    if raw.trim().is_empty() {
        "(unknown)"
    } else {
        raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_from_summary_stores_fields_and_stays_deterministic() {
        let first = ScreenContext::capture_from_summary("three items listed", "Notes");
        let second = ScreenContext::capture_from_summary("three items listed", "Notes");
        let capture = first.capture.as_ref().expect("capture present");
        assert_eq!(capture.screen_state, "three items listed");
        assert_eq!(capture.window_title, "Notes");
        // Pure constructor: no clock read, so repeated calls agree exactly.
        assert_eq!(capture.timestamp, 0);
        assert_eq!(
            first.capture.as_ref().unwrap(),
            second.capture.as_ref().unwrap()
        );
    }

    #[test]
    fn empty_context_has_no_capture() {
        assert!(ScreenContext::empty().is_empty());
        assert!(!ScreenContext::capture_from_summary("s", "w").is_empty());
    }

    #[test]
    fn prompt_without_capture_returns_question_verbatim() {
        for question in [
            "what is on screen?",
            "summarize the visible items",
            "  spaced out  ",
            "",
        ] {
            assert_eq!(ScreenContext::empty().build_prompt(question), question);
        }
    }

    #[test]
    fn prompt_embeds_both_context_and_question() {
        let ctx = ScreenContext::capture_from_summary("three items listed", "Notes")
            .with_window_class("notes-app");
        let out = ctx.build_prompt("summarize the visible items");
        assert!(out.contains("summarize the visible items"));
        assert!(out.contains("three items listed"));
        assert!(out.contains("Notes"));
        assert!(out.contains("notes-app"));
    }

    #[test]
    fn prompt_marks_blank_title_and_class_unknown() {
        let ctx = ScreenContext::capture_from_summary("state", "");
        let out = ctx.build_prompt("q");
        assert!(out.contains("(unknown)"));
        assert!(out.contains("state"));
        assert!(out.contains('q'));
    }

    #[test]
    fn prompt_without_screens_allowed_returns_question_verbatim() {
        let ctx = ScreenContext::capture_from_summary("three items listed", "Notes");
        for config in [
            CompanionConfig {
                enabled: false,
                ..CompanionConfig::default()
            },
            CompanionConfig {
                include_screen_context: false,
                ..CompanionConfig::default()
            },
        ] {
            assert_eq!(
                ctx.build_prompt_with_config("summarize this", &config),
                "summarize this"
            );
        }
    }

    #[test]
    fn prompt_truncates_state_to_budget_but_keeps_question() {
        let ctx = ScreenContext::capture_from_summary("abcdefghij", "Notes");
        let config = CompanionConfig {
            max_screen_chars: 4,
            ..CompanionConfig::default()
        };
        let out = ctx.build_prompt_with_config("summarize this", &config);
        assert!(out.contains("abcd"));
        assert!(!out.contains("abcdefghij"));
        assert!(out.contains("summarize this"));
    }

    #[test]
    fn live_capture_stamps_a_real_timestamp() {
        let ctx = ScreenContext::capture_now("state", "Notes", "notes-app");
        let ts = ctx.capture.as_ref().expect("capture present").timestamp;
        assert!(ts > 0);
    }

    #[test]
    fn companion_budget_truncation_is_char_boundary_safe() {
        let config = CompanionConfig {
            max_screen_chars: 2,
            ..CompanionConfig::default()
        };
        // Multi-byte chars must not be split: 2 chars, not 2 bytes.
        assert_eq!(config.truncate_summary("éclair"), "éc");
    }
}
