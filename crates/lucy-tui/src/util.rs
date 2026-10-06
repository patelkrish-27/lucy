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

/// Wall-clock as a stopwatch reads it: tenths under a minute, then minutes
/// and whole seconds. Used both by the live spinner and by the frozen number
/// under a finished reply, so the timer the user watched tick is the one that
/// gets recorded.
pub(crate) fn format_elapsed(ms: u128) -> String {
    if ms < 60_000 {
        format!("{:.1}s", ms as f64 / 1000.0)
    } else {
        format!("{}m {}s", ms / 60_000, (ms % 60_000) / 1000)
    }
}

/// Generation rate for a finished turn. A run that billed no tokens (routing
/// failed, everything was cached, no model call at all) has no rate to report,
/// so the caller shows the time alone rather than a fabricated `0.0 tok/s`.
pub(crate) fn format_rate(tokens: u64, ms: u128) -> Option<String> {
    if tokens == 0 || ms == 0 {
        return None;
    }
    Some(format!("{:.1} tok/s", tokens as f64 / (ms as f64 / 1000.0)))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elapsed_reads_like_a_stopwatch() {
        for (ms, want) in [
            (0, "0.0s"),
            (40, "0.0s"),
            (940, "0.9s"),
            (1_000, "1.0s"),
            (5_200, "5.2s"),
            (42_700, "42.7s"),
            (59_999, "60.0s"),
            (60_000, "1m 0s"),
            (187_000, "3m 7s"),
            (3_600_000, "60m 0s"),
        ] {
            assert_eq!(format_elapsed(ms), want, "for {ms}ms");
        }
    }

    #[test]
    fn rate_is_tokens_per_second_and_absent_when_there_is_nothing_to_rate() {
        assert_eq!(format_rate(50, 5_200).as_deref(), Some("9.6 tok/s"));
        assert_eq!(format_rate(1_000, 1_000).as_deref(), Some("1000.0 tok/s"));
        // No tokens, or no time: a rate would be a division by nothing, so
        // the caller gets None and prints the duration by itself.
        assert!(format_rate(0, 5_200).is_none());
        assert!(format_rate(50, 0).is_none());
    }
}
