//! Desktop companion overlay: a small always-on-top mascot panel.
//!
//! Pure state + avatar painting. The avatar is rendered with
//! `lucy_mascot`, the same sprite the welcome screen and chat rail use —
//! never re-drawn here.

use lucy_mascot::{CompanionState, Mood};
use ratatui::{buffer::Buffer, layout::Rect};

use super::util::truncate_one_line;

/// A small mascot panel that can hover over the chat view.
pub struct DesktopCompanion {
    visible: bool,
    message: String,
    state: CompanionState,
}

impl DesktopCompanion {
    pub fn new() -> Self {
        Self {
            visible: false,
            message: String::new(),
            state: CompanionState::Idle,
        }
    }

    pub fn show(&mut self) {
        self.visible = true;
        if self.state == CompanionState::Idle {
            self.state = CompanionState::Listening;
        }
    }

    pub fn hide(&mut self) {
        self.visible = false;
        self.state = CompanionState::Idle;
    }

    pub fn toggle(&mut self) {
        self.visible = !self.visible;
        self.set_state(if self.visible {
            CompanionState::Listening
        } else {
            CompanionState::Idle
        });
    }

    pub fn is_visible(&self) -> bool {
        self.visible
    }

    /// Latest one-line note shown under the avatar. Showing the panel on a
    /// new note keeps it live without a separate `/companion show` round-trip.
    pub fn handle_message(&mut self, text: &str) {
        let t = text.trim();
        if t.is_empty() {
            return;
        }
        self.message = truncate_one_line(t, 140);
        self.visible = true;
        // A note is the mascot speaking: the pose should say so.
        self.set_state(CompanionState::Speaking);
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    /// Current companion lifecycle state (drives the mascot pose).
    pub fn state(&self) -> CompanionState {
        self.state
    }

    /// Set the lifecycle state explicitly (e.g. Thinking while a turn runs).
    pub fn set_state(&mut self, state: CompanionState) {
        self.state = state;
    }

    /// Paint the avatar with the shared mascot renderer. The companion's own
    /// lifecycle state drives the pose via `draw_companion`, so the face
    /// always agrees with what the panel is doing; `mood` is the fallback
    /// when the state maps to no dedicated art.
    pub fn draw_avatar(&self, buf: &mut Buffer, area: Rect, mood: Mood, t_ms: u128) {
        if area.width < 4 || area.height < 3 {
            return;
        }
        let mode = lucy_mascot::color_mode();
        // CompanionState owns the pose; fall back to the caller's mood only
        // when the state has no mapped art (defensive: mapping is total today).
        if lucy_mascot::CompanionState::from_mood(mood).is_some() {
            lucy_mascot::draw_companion(buf, area, self.state, t_ms, mode);
        } else {
            lucy_mascot::draw(buf, area, mood, t_ms, mode);
        }
    }
}

impl Default for DesktopCompanion {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toggles_visibility() {
        let mut c = DesktopCompanion::new();
        assert!(!c.is_visible());
        c.show();
        assert!(c.is_visible());
        c.hide();
        assert!(!c.is_visible());
        c.toggle();
        assert!(c.is_visible());
        c.toggle();
        assert!(!c.is_visible());
    }

    #[test]
    fn visibility_transitions_drive_the_mascot_pose() {
        use lucy_mascot::CompanionState;
        let mut c = DesktopCompanion::new();
        assert_eq!(c.state(), CompanionState::Idle);
        c.show();
        assert_eq!(c.state(), CompanionState::Listening);
        c.handle_message("done");
        assert_eq!(c.state(), CompanionState::Speaking);
        c.hide();
        assert_eq!(c.state(), CompanionState::Idle);
        c.toggle();
        assert!(c.is_visible());
        assert_eq!(c.state(), CompanionState::Listening);
        c.toggle();
        assert!(!c.is_visible());
        assert_eq!(c.state(), CompanionState::Idle);
    }

    #[test]
    fn a_note_shows_the_panel_and_is_capped_to_one_line() {        let mut c = DesktopCompanion::new();
        c.handle_message("  hello there  ");
        assert!(c.is_visible());
        assert_eq!(c.message(), "hello there");
        c.handle_message("");
        assert_eq!(c.message(), "hello there", "blank note keeps the old one");
        c.handle_message(&"x".repeat(500));
        assert!(c.message().chars().count() <= 141, "got {}", c.message());
    }
}
