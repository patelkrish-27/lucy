//! Small shared helpers: text truncation, token formatting, input cursor math.

pub(crate) fn truncate_one_line(s: &str, max: usize) -> String {
    let one: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if one.chars().count() > max {
        let mut t: String = one.chars().take(max).collect();
        t.push('…');
        t
    } else {
        one
    }
}

pub(crate) fn char_byte_idx(s: &str, char_idx: usize) -> usize {
    if char_idx == 0 {
        return 0;
    }
    s.char_indices()
        .nth(char_idx)
        .map(|(b, _)| b)
        .unwrap_or(s.len())
}

pub(crate) fn truncate_model_label(s: &str) -> String {
    if s.chars().count() > 16 {
        s.chars().take(16).collect()
    } else {
        s.to_owned()
    }
}

pub(crate) fn format_tokens(n: u64) -> String {
    if n < 1000 {
        format!("{n} tok")
    } else if n < 1_000_000 {
        let v = n as f64 / 1000.0;
        if v >= 100.0 {
            format!("{:.0}k", v)
        } else {
            format!("{:.1}k", v)
        }
    } else {
        let v = n as f64 / 1_000_000.0;
        format!("{:.1}M", v)
    }
}

pub(crate) fn line_home_cursor(input: &str, cursor: usize) -> usize {
    let chars: Vec<char> = input.chars().collect();
    let cur = cursor.min(chars.len());
    let mut i = cur;
    while i > 0 && chars[i - 1] != '\n' {
        i -= 1;
    }
    i
}

pub(crate) fn line_end_cursor(input: &str, cursor: usize) -> usize {
    let chars: Vec<char> = input.chars().collect();
    let cur = cursor.min(chars.len());
    let mut i = cur;
    while i < chars.len() && chars[i] != '\n' {
        i += 1;
    }
    i
}

pub(crate) fn cursor_row_col(input: &str, cursor: usize) -> (usize, usize) {
    let count = input.chars().count();
    let cur = cursor.min(count);
    let prefix: String = input.chars().take(cur).collect();
    let row = prefix.chars().filter(|&c| c == '\n').count();
    let col = prefix
        .rfind('\n')
        .map(|idx| prefix[idx + 1..].chars().count())
        .unwrap_or(cur);
    // NB: rfind on the char-collected prefix is byte-based but '\n' is 1 byte,
    // so slicing at idx+1 is always a char boundary.
    (row, col)
}

pub(crate) fn short_id(full: &str) -> String {
    full.chars().take(8).collect()
}
