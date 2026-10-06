//! A simple Markdown editor with slash commands, undo/redo, and cursor
//! management.
//!
//! The editor operates on a plain `String` buffer with a byte-offset cursor.
//! It is intentionally minimal: it provides the structural operations (insert,
//! delete, move, slash commands) that a TUI or MCP tool needs to drive editing,
//! without taking on a full text-rendering stack.

use anyhow::{Result, anyhow};

/// Whether the editor is accepting text input or navigating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditorMode {
    /// Navigation and command entry; typed characters are not inserted.
    Normal,
    /// Text is inserted at the cursor.
    Insert,
    /// A `/` command is being typed at the start of a line.
    Command,
}

/// Cursor movement direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

/// A Markdown editor with undo/redo and slash commands.
///
/// Slash commands are entered by typing `/` at the beginning of a line in
/// Command mode. Supported commands:
///
/// * `/title <text>` — set the page title (stored as the first `#` heading)
/// * `/tag <name>` — add a tag to the page's tag list
/// * `/save` — mark the content as saved (clears the dirty flag)
/// * `/delete` — clear the entire content buffer
/// * `/move <space>` — annotate a pending space move (stored in metadata)
#[derive(Debug, Clone)]
pub struct MarkdownEditor {
    content: String,
    cursor: usize,
    mode: EditorMode,
    dirty: bool,
    undo_stack: Vec<String>,
    redo_stack: Vec<String>,
    /// The page title, set via `/title`.
    pub title: String,
    /// Tags added via `/tag`.
    pub tags: Vec<String>,
    /// Pending space name set via `/move`.
    pub pending_space: Option<String>,
}

impl Default for MarkdownEditor {
    fn default() -> Self {
        Self::new()
    }
}

impl MarkdownEditor {
    /// Create a new empty editor in Normal mode.
    pub fn new() -> Self {
        Self {
            content: String::new(),
            cursor: 0,
            mode: EditorMode::Normal,
            dirty: false,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            title: String::new(),
            tags: Vec::new(),
            pending_space: None,
        }
    }

    /// Create an editor pre-loaded with content.
    pub fn with_content(content: &str) -> Self {
        let mut ed = Self::new();
        ed.content = content.to_string();
        ed.cursor = content.len();
        ed
    }

    /// The current editor mode.
    pub fn mode(&self) -> EditorMode {
        self.mode
    }

    /// Switch the editor mode.
    pub fn set_mode(&mut self, mode: EditorMode) {
        self.mode = mode;
    }

    /// Insert text at the cursor position.
    ///
    /// In Normal mode this is a no-op; text is only inserted in Insert mode.
    pub fn insert(&mut self, text: &str) {
        if self.mode != EditorMode::Insert {
            return;
        }
        self.push_undo();
        self.content.insert_str(self.cursor, text);
        self.cursor += text.len();
        self.dirty = true;
        self.redo_stack.clear();
    }

    /// Delete the character at the cursor (backspace semantics: delete the
    /// character before the cursor).
    pub fn delete(&mut self) {
        if self.cursor == 0 {
            return;
        }
        self.push_undo();
        self.cursor -= 1;
        self.content.remove(self.cursor);
        self.dirty = true;
        self.redo_stack.clear();
    }

    /// Delete the character after the cursor (delete-forward semantics).
    pub fn delete_forward(&mut self) {
        if self.cursor >= self.content.len() {
            return;
        }
        self.push_undo();
        self.content.remove(self.cursor);
        self.dirty = true;
        self.redo_stack.clear();
    }

    /// Move the cursor in the given direction.
    ///
    /// Left/Right move by one byte (sufficient for ASCII; callers that need
    /// grapheme-aware movement should operate on the content directly).
    /// Up/Down move to the same column on the previous/next line.
    pub fn move_cursor(&mut self, direction: Direction) {
        match direction {
            Direction::Left => {
                if self.cursor > 0 {
                    self.cursor -= 1;
                }
            }
            Direction::Right => {
                if self.cursor < self.content.len() {
                    self.cursor += 1;
                }
            }
            Direction::Up => {
                self.cursor = self.cursor_up();
            }
            Direction::Down => {
                self.cursor = self.cursor_down();
            }
        }
    }

    /// Execute a slash command.
    ///
    /// The command string should not include the leading `/`. Returns an
    /// error for unknown commands or missing arguments.
    pub fn slash_command(&mut self, command: &str) -> Result<()> {
        let (cmd, args) = match command.split_once(' ') {
            Some((c, a)) => (c, a.trim()),
            None => (command, ""),
        };

        match cmd {
            "title" => {
                if args.is_empty() {
                    return Err(anyhow!("/title requires a title argument"));
                }
                self.title = args.to_string();
                // Also set the first-line heading if it's a title line.
                self.set_title_heading(args);
                Ok(())
            }
            "tag" => {
                if args.is_empty() {
                    return Err(anyhow!("/tag requires a tag name"));
                }
                if !self.tags.iter().any(|t| t == args) {
                    self.tags.push(args.to_string());
                }
                Ok(())
            }
            "save" => {
                self.dirty = false;
                Ok(())
            }
            "delete" => {
                self.push_undo();
                self.content.clear();
                self.cursor = 0;
                self.dirty = true;
                self.redo_stack.clear();
                Ok(())
            }
            "move" => {
                if args.is_empty() {
                    return Err(anyhow!("/move requires a space name"));
                }
                self.pending_space = Some(args.to_string());
                Ok(())
            }
            _ => Err(anyhow!("unknown slash command: /{cmd}")),
        }
    }

    /// Get the current content buffer.
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Replace the entire content buffer.
    pub fn set_content(&mut self, content: &str) {
        self.push_undo();
        self.content = content.to_string();
        self.cursor = self.content.len();
        self.dirty = true;
        self.redo_stack.clear();
    }

    /// Render the content as Markdown.
    ///
    /// If a title is set and the content doesn't already start with a `#`
    /// heading, the title is prepended as an H1.
    pub fn render(&self) -> String {
        let mut out = String::new();
        if !self.title.is_empty() && !self.content.starts_with("# ") {
            out.push_str(&format!("# {}\n\n", self.title));
        }
        out.push_str(&self.content);
        out
    }

    /// Whether the content has been modified since the last save.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Undo the last operation, if any.
    pub fn undo(&mut self) {
        if let Some(prev) = self.undo_stack.pop() {
            self.redo_stack.push(self.content.clone());
            self.content = prev;
            self.cursor = self.cursor.min(self.content.len());
            self.dirty = true;
        }
    }

    /// Redo the last undone operation, if any.
    pub fn redo(&mut self) {
        if let Some(next) = self.redo_stack.pop() {
            self.undo_stack.push(self.content.clone());
            self.content = next;
            self.cursor = self.cursor.min(self.content.len());
            self.dirty = true;
        }
    }

    /// The current cursor position (byte offset).
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Set the cursor position (byte offset), clamped to content length.
    pub fn set_cursor(&mut self, pos: usize) {
        self.cursor = pos.min(self.content.len());
    }

    // -------------------------------------------------------------- internals

    fn push_undo(&mut self) {
        self.undo_stack.push(self.content.clone());
        // Cap the undo stack so a long editing session doesn't grow without
        // bound.
        const MAX_UNDO: usize = 100;
        if self.undo_stack.len() > MAX_UNDO {
            self.undo_stack.remove(0);
        }
    }

    fn cursor_up(&self) -> usize {
        let before = &self.content[..self.cursor];
        let line_start = before.rfind('\n').map(|i| i + 1).unwrap_or(0);
        let col = self.cursor - line_start;
        if line_start == 0 {
            return 0;
        }
        let prev_line_end = line_start - 1; // the \n
        let prev_line_start = self.content[..prev_line_end]
            .rfind('\n')
            .map(|i| i + 1)
            .unwrap_or(0);
        let prev_line_len = prev_line_end - prev_line_start;
        prev_line_start + col.min(prev_line_len)
    }

    fn cursor_down(&self) -> usize {
        let after = &self.content[self.cursor..];
        let Some(newline_offset) = after.find('\n') else {
            return self.content.len();
        };
        let line_end = self.cursor + newline_offset;
        let next_line_start = line_end + 1;
        if next_line_start >= self.content.len() {
            return self.content.len();
        }
        let before = &self.content[..self.cursor];
        let line_start = before.rfind('\n').map(|i| i + 1).unwrap_or(0);
        let col = self.cursor - line_start;
        let next_line_end = self.content[next_line_start..]
            .find('\n')
            .map(|i| next_line_start + i)
            .unwrap_or(self.content.len());
        let next_line_len = next_line_end - next_line_start;
        next_line_start + col.min(next_line_len)
    }

    fn set_title_heading(&mut self, title: &str) {
        if let Some(rest) = self.content.strip_prefix("# ") {
            // Replace the existing H1 title line.
            if let Some(newline) = rest.find('\n') {
                self.content = format!("# {}{}", title, &rest[newline..]);
            } else {
                self.content = format!("# {title}");
            }
        } else {
            // Prepend a new H1.
            self.content = format!("# {title}\n\n{}", self.content);
        }
        self.dirty = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_appends_text_at_cursor() {
        let mut ed = MarkdownEditor::new();
        ed.set_mode(EditorMode::Insert);
        ed.insert("hello");
        ed.insert(" world");
        assert_eq!(ed.content(), "hello world");
        assert!(ed.is_dirty());
    }

    #[test]
    fn insert_is_noop_in_normal_mode() {
        let mut ed = MarkdownEditor::new();
        ed.set_mode(EditorMode::Normal);
        ed.insert("hello");
        assert_eq!(ed.content(), "");
        assert!(!ed.is_dirty());
    }

    #[test]
    fn delete_removes_char_before_cursor() {
        let mut ed = MarkdownEditor::with_content("hello");
        ed.set_cursor(5);
        ed.delete();
        assert_eq!(ed.content(), "hell");
    }

    #[test]
    fn delete_forward_removes_char_after_cursor() {
        let mut ed = MarkdownEditor::with_content("hello");
        ed.set_cursor(0);
        ed.delete_forward();
        assert_eq!(ed.content(), "ello");
    }

    #[test]
    fn cursor_movement_left_right() {
        let mut ed = MarkdownEditor::with_content("abc");
        ed.set_cursor(3);
        ed.move_cursor(Direction::Left);
        assert_eq!(ed.cursor(), 2);
        ed.move_cursor(Direction::Right);
        assert_eq!(ed.cursor(), 3);
    }

    #[test]
    fn cursor_movement_up_down() {
        let mut ed = MarkdownEditor::with_content("ab\ncd");
        // Column is preserved across the line boundary, which is what makes
        // up/down usable: moving up from column 1 of "cd" lands on column 1 of
        // "ab" ('b', index 1), not on that line's first character.
        ed.set_cursor(4); // 'd', column 1
        ed.move_cursor(Direction::Up);
        assert_eq!(ed.cursor(), 1, "'b' — same column, previous line");
        ed.move_cursor(Direction::Down);
        assert_eq!(ed.cursor(), 4, "'d' — same column, back down");
    }

    #[test]
    fn moving_up_from_column_zero_lands_on_column_zero() {
        // The regression this pairs with: a cursor at the start of a line must
        // not gain a column on the way up, or repeated up-presses drift right.
        let mut ed = MarkdownEditor::with_content("ab\ncd");
        ed.set_cursor(3); // 'c', column 0
        ed.move_cursor(Direction::Up);
        assert_eq!(ed.cursor(), 0, "'a' — column 0 of the first line");
    }

    #[test]
    fn slash_command_title_sets_heading() {
        let mut ed = MarkdownEditor::new();
        ed.set_mode(EditorMode::Command);
        ed.slash_command("title My Page").expect("title command");
        assert_eq!(ed.title, "My Page");
        assert!(ed.content.starts_with("# My Page"));
    }

    #[test]
    fn slash_command_tag_adds_tag() {
        let mut ed = MarkdownEditor::new();
        ed.set_mode(EditorMode::Command);
        ed.slash_command("tag rust").expect("tag command");
        ed.slash_command("tag coding").expect("tag command");
        assert_eq!(ed.tags.len(), 2);
        assert!(ed.tags.contains(&"rust".to_string()));
    }

    #[test]
    fn slash_command_save_clears_dirty() {
        let mut ed = MarkdownEditor::new();
        ed.set_mode(EditorMode::Insert);
        ed.insert("text");
        assert!(ed.is_dirty());
        ed.set_mode(EditorMode::Command);
        ed.slash_command("save").expect("save command");
        assert!(!ed.is_dirty());
    }

    #[test]
    fn slash_command_delete_clears_content() {
        let mut ed = MarkdownEditor::with_content("some content");
        ed.set_mode(EditorMode::Command);
        ed.slash_command("delete").expect("delete command");
        assert_eq!(ed.content(), "");
        assert!(ed.is_dirty());
    }

    #[test]
    fn slash_command_move_sets_pending_space() {
        let mut ed = MarkdownEditor::new();
        ed.set_mode(EditorMode::Command);
        ed.slash_command("move Archive").expect("move command");
        assert_eq!(ed.pending_space.as_deref(), Some("Archive"));
    }

    #[test]
    fn slash_command_unknown_errors() {
        let mut ed = MarkdownEditor::new();
        ed.set_mode(EditorMode::Command);
        assert!(ed.slash_command("bogus").is_err());
    }

    #[test]
    fn undo_redo_roundtrip() {
        let mut ed = MarkdownEditor::new();
        ed.set_mode(EditorMode::Insert);
        ed.insert("hello");
        ed.undo();
        assert_eq!(ed.content(), "");
        ed.redo();
        assert_eq!(ed.content(), "hello");
    }

    #[test]
    fn render_prepends_title_when_missing() {
        let mut ed = MarkdownEditor::new();
        ed.title = "My Title".to_string();
        ed.content = "Some body text.".to_string();
        let rendered = ed.render();
        assert!(rendered.starts_with("# My Title\n\n"));
        assert!(rendered.contains("Some body text."));
    }

    #[test]
    fn render_does_not_double_title() {
        let mut ed = MarkdownEditor::new();
        ed.title = "My Title".to_string();
        ed.content = "# My Title\n\nBody".to_string();
        let rendered = ed.render();
        assert_eq!(rendered.matches("# My Title").count(), 1);
    }

    #[test]
    fn with_content_starts_at_end() {
        let ed = MarkdownEditor::with_content("hello");
        assert_eq!(ed.cursor(), 5);
        assert!(!ed.is_dirty());
    }
}
