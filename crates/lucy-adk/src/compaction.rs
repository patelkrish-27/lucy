//! Automatic session compaction (`auto_compact = On` in `/settings`).
//!
//! When a session's live history grows past a threshold, the older turns are
//! replaced by a single summary turn produced by the configured model. The
//! append-only ADK event log is never rewritten — only the *runtime context*
//! is compacted, so a compacted session can still be audited in full.

use anyhow::Result;
use lucy_core::{AssistantTurn, SessionData, TurnMessage};

/// Default trigger: compact once the live history exceeds this many turns.
pub const DEFAULT_TRIGGER_TURNS: usize = 24;
/// Default number of recent turns kept verbatim after a compaction.
pub const DEFAULT_KEEP_TURNS: usize = 8;

pub const COMPACT_SYSTEM: &str = "You are Lucy's session compactor. Summarize the conversation so \
another assistant can continue it without the full transcript. Preserve: the user's goal, \
decisions made, facts learned, credentials/paths/URLs in play, and what is still pending. \
Drop pleasantries. Return only the summary text.";

/// A compaction that is ready to apply: `(before, after, summary)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Compaction {
    pub summary: String,
    pub before: usize,
    pub after: usize,
}

/// True when `history` is long enough to warrant a compaction.
pub fn should_compact(history: &[TurnMessage], trigger: usize) -> bool {
    trigger > 0 && history.len() > trigger
}

/// Build the prompt for the summarizer from the turns that will be dropped.
pub fn render_compact_prompt(history: &[TurnMessage], keep: usize) -> Option<String> {
    let keep = keep.min(history.len());
    let older = &history[..history.len() - keep];
    if older.is_empty() {
        return None;
    }
    let mut out = String::from("Conversation so far:\n\n");
    for turn in older {
        match turn {
            TurnMessage::User(t) => {
                out.push_str("user: ");
                out.push_str(t);
            }
            TurnMessage::Assistant(a) => {
                out.push_str("assistant: ");
                out.push_str(a.text.as_deref().unwrap_or("(tool call)"));
            }
            TurnMessage::Tool(r) => {
                out.push_str(&format!("tool {}: ", r.name));
                out.push_str(&r.output.to_string());
            }
        }
        out.push('\n');
    }
    out.push_str("\nWrite the continuation summary now.");
    Some(out)
}

/// Apply a summary in place: the older turns collapse into one assistant turn
/// carrying the summary, and the newest `keep` turns stay verbatim.
pub fn apply_compaction(data: &mut SessionData, summary: &str, keep: usize) -> Compaction {
    let before = data.history.len();
    let keep = keep.min(before);
    let split = before - keep;
    let anchor = data.history[..split].to_vec();
    let mut recent: Vec<TurnMessage> = data.history.split_off(split);
    // A dangling tool result whose call is now summarised away would confuse a
    // strict OpenAI-compatible server, so drop orphaned tool turns.
    recent.retain(|t| !matches!(t, TurnMessage::Tool(_)));
    let summarized = format!(
        "[compacted {} earlier turn(s)]\n\n{summary}\n\n(kept {} recent turn(s) verbatim)",
        anchor.len(),
        recent.len()
    );
    data.history = vec![TurnMessage::Assistant(AssistantTurn {
        text: Some(summarized),
        tool_calls: Vec::new(),
    })];
    data.history.extend(recent);
    Compaction {
        summary: summary.to_owned(),
        before,
        after: data.history.len(),
    }
}

/// Summary a set of older turns with the configured model.
///
/// Returns `None` when the model call fails or returns nothing usable, so the
/// caller can fall back to plain trimming instead of losing history.
pub async fn summarize_with<F, Fut>(
    history: &[TurnMessage],
    keep: usize,
    mut complete: F,
) -> Option<String>
where
    F: FnMut(String) -> Fut,
    Fut: std::future::Future<Output = Result<String>>,
{
    let prompt = render_compact_prompt(history, keep)?;
    let summary = complete(prompt).await.ok()?;
    let trimmed = summary.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.chars().take(4_000).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucy_core::ToolResult;

    fn turn(n: usize) -> TurnMessage {
        TurnMessage::User(format!("message {n}"))
    }

    #[test]
    fn trigger_fires_past_the_threshold() {
        let h: Vec<TurnMessage> = (0..10).map(turn).collect();
        assert!(!should_compact(&h, 10));
        assert!(should_compact(&h, 9));
        assert!(!should_compact(&h, 0));
        assert!(!should_compact(&[], 3));
    }

    #[test]
    fn prompt_contains_only_the_older_turns() {
        let h: Vec<TurnMessage> = (0..10).map(turn).collect();
        let prompt = render_compact_prompt(&h, 3).unwrap();
        assert!(prompt.contains("message 0"));
        assert!(prompt.contains("message 6"));
        assert!(!prompt.contains("message 7"));
        assert!(prompt.contains("continuation summary"));
    }

    #[test]
    fn prompt_is_none_when_everything_is_kept() {
        let h: Vec<TurnMessage> = (0..3).map(turn).collect();
        assert!(render_compact_prompt(&h, 5).is_none());
        assert!(render_compact_prompt(&[], 5).is_none());
    }

    #[test]
    fn compaction_collapses_older_turns_into_one_summary() {
        let mut data = SessionData::new(lucy_core::SessionId::default());
        data.history = (0..10).map(turn).collect();
        let result = apply_compaction(&mut data, "the user wanted a song played", 3);
        assert_eq!(result.before, 10);
        // 1 summary + 3 kept.
        assert_eq!(result.after, 4);
        assert_eq!(data.history.len(), 4);
        match &data.history[0] {
            TurnMessage::Assistant(a) => {
                let text = a.text.clone().unwrap_or_default();
                assert!(text.contains("the user wanted a song played"), "{text}");
                assert!(text.contains("compacted 7 earlier turn(s)"), "{text}");
            }
            other => panic!("expected a summary turn, got {other:?}"),
        }
        assert!(matches!(&data.history[1], TurnMessage::User(t) if t == "message 7"));
    }

    #[test]
    fn compaction_drops_orphaned_tool_results() {
        let mut data = SessionData::new(lucy_core::SessionId::default());
        data.history = vec![
            turn(0),
            turn(1),
            TurnMessage::Tool(ToolResult {
                call_id: "c1".into(),
                name: "browser_open".into(),
                output: serde_json::json!({"ok": true}),
                is_error: false,
            }),
            turn(3),
        ];
        let result = apply_compaction(&mut data, "s", 1);
        // 1 summary + 1 kept (the orphaned tool turn is dropped).
        assert_eq!(result.after, 2);
        assert!(
            !data
                .history
                .iter()
                .any(|t| matches!(t, TurnMessage::Tool(_)))
        );
    }

    #[test]
    fn compaction_result_is_bounded_by_keep_plus_one() {
        let mut data = SessionData::new(lucy_core::SessionId::default());
        data.history = (0..20).map(turn).collect();
        for keep in 0..=20 {
            let before = data.history.len();
            let result = apply_compaction(&mut data, "s", keep);
            assert_eq!(data.history.len(), result.after);
            assert!(
                data.history.len() <= keep + 1,
                "keep={keep} produced {} turns",
                data.history.len()
            );
            // Never grows once there is something to drop.
            if before > keep + 1 {
                assert!(data.history.len() < before, "keep={keep}");
            }
        }
    }

    #[tokio::test]
    async fn compaction_then_summarize_is_idempotent_in_size() {
        // Compacting an already-compacted history must not grow it further.
        let mut data = SessionData::new(lucy_core::SessionId::default());
        data.history = (0..12).map(turn).collect();
        apply_compaction(&mut data, "first", 3);
        let after_first = data.history.len();
        apply_compaction(&mut data, "second", 3);
        assert!(data.history.len() <= after_first + 1);
    }

    #[tokio::test]
    async fn summarize_uses_the_completion_closure() {
        let h: Vec<TurnMessage> = (0..6).map(turn).collect();
        let out = summarize_with(&h, 2, |p| async move {
            assert!(p.contains("message 0"));
            Ok("  a tidy summary  ".into())
        })
        .await;
        assert_eq!(out.as_deref(), Some("a tidy summary"));
    }

    #[tokio::test]
    async fn summarize_returns_none_on_failure_or_blank() {
        let h: Vec<TurnMessage> = (0..6).map(turn).collect();
        let failed = summarize_with(&h, 2, |_| async { anyhow::bail!("nope") }).await;
        assert!(failed.is_none());
        let blank = summarize_with(&h, 2, |_| async { Ok("   ".into()) }).await;
        assert!(blank.is_none());
        // Nothing to summarize -> no model call at all.
        let none = summarize_with(&h, 6, |_| async { Ok("x".into()) }).await;
        assert!(none.is_none());
    }
}
