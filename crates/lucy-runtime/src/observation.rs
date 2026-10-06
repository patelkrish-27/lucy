//! Observation: **what a tool said, plus what the page says about itself**.
//!
//! A ReAct step is only better than a plan if the next decision is made against
//! evidence. Two kinds of evidence exist and this module owns both:
//!
//! * **the tool output** — bounded, and worded the way a person needs to read it
//!   ([`observation_text`]): a failure quotes what went wrong instead of
//!   serializing the envelope it arrived in, and a success keeps enough of the
//!   payload to be useful.
//! * **the probe's ground truth** ([`probe_eval`]) — the page's own account of
//!   its state, read through `browser_evaluate`. Model-authored, so it is
//!   sanitized by [`crate::agent_loop::sanitize_probe`] before it can run, and
//!   tri-state: true, false, or *could not run*, which is deliberately not the
//!   same answer as false.
//!
//! [`completion_check`] is where the two meet: a completion claim is only ever
//! *done* when a check that actually ran says so. Nothing in this module
//! decides what a goal is or which tool advances it — no site list, no task
//! keyword, no verb table. The model names the probe; this module asks the page
//! and reports the answer honestly, including when the answer is "no".
//!
//! ```text
//!   observation  = tool output (bounded, failure-worded) + probe ground truth
//!   completion   = a probe that ran and returned true   — nothing else
//! ```

use crate::agent_loop::sanitize_probe;
use crate::fast_perception::FastContext;
use lucy_tools::ToolRegistry;
use serde_json::Value;

/// Default lines kept from one tool result before it is cut.
///
/// Above [`lucy_core::truncate_str`]'s floor of ten lines, below which that
/// helper deliberately declines to truncate at all — a ceiling under the floor
/// would read as "truncation is on" while never truncating.
pub const DEFAULT_MAX_LINES: usize = 40;

/// Default character ceiling on one observation.
///
/// Line truncation alone is not a bound: `truncate_tool_output` only knows how
/// to cut strings, so a tool answering with one enormous object or a large array
/// passes through untouched, and one long line would become the whole prompt.
pub const DEFAULT_MAX_CHARS: usize = 4_000;

/// What a tool result says, as one bounded piece of text.
///
/// A failure and a success are worded differently on purpose, because they are
/// read for different reasons:
///
/// * a **failure** is quoted, not serialized. `{"content":[{"text":"error: no
///   such element"}]}` is the wire shape; "no such element" is the information.
///   [`lucy_core::friendly`] then renders it as the sentence a person would say.
/// * a **success** keeps its payload, because a ref, a count or a URL in it is
///   exactly what the next decision needs.
pub fn observation_text(output: &Value) -> String {
    bound_observation(output, DEFAULT_MAX_LINES, DEFAULT_MAX_CHARS)
}

/// [`observation_text`] under explicit caps, so a caller with its own budget
/// (the ReAct loop does) does not have to take the defaults.
///
/// Truncation is announced, so a cut payload is never read as a complete one.
pub fn bound_observation(output: &Value, max_lines: usize, max_chars: usize) -> String {
    // `turn::mcp_failure_text` rather than the fast lane's `tool_reports_failure`:
    // it is the fuller detector, covering an `error` field and a stringified
    // result nested as content text as well as `success: false`. Two detectors
    // would mean a failure one of them cannot see gets rendered as a success
    // payload, which is precisely the failure mode both were written to stop.
    if crate::turn::mcp_failure_text(output).is_some() {
        return truncate_chars(&failure_text(output), max_chars);
    }
    let truncated = lucy_core::truncate_tool_output(output, max_lines);
    let text = match &truncated {
        Value::String(s) if s.trim().is_empty() => "(no output)".to_owned(),
        Value::String(s) => s.clone(),
        other => serde_json::to_string_pretty(other).unwrap_or_else(|_| other.to_string()),
    };
    truncate_chars(&text, max_chars)
}

/// What a failed call *said*, without the envelope it arrived in.
///
/// Falls back to a sentence rather than to bytes: the failure is real — the
/// server said so — and when there is nothing to quote, saying so is more use
/// to the reader than echoing a payload.
fn failure_text(output: &Value) -> String {
    match lucy_core::payload_message(output) {
        Some(message) => format!("FAILED: {}", lucy_core::friendly(&message)),
        None => String::from("FAILED: the tool reported a failure without saying why"),
    }
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    let mut out: String = text.chars().take(max_chars).collect();
    out.push_str("\n… [observation truncated]");
    out
}

/// One probe expression, read off the page.
///
/// `None` means it could not run at all, which the caller must keep distinct
/// from `Some(false)`: a missing `browser_evaluate`, a page that is not there,
/// or an expression the sanitizer rejected are all "no answer", and none of them
/// is evidence that the goal is undone.
///
/// The sanitizer runs here rather than only at parse time, so any path that can
/// reach this function — a cached plan, a probe lifted out of prose, a future
/// caller — is covered by the same rule. Model-authored code does not run in the
/// user's browser without passing [`sanitize_probe`] first.
pub async fn probe_eval(
    registry: &ToolRegistry,
    ctx: &FastContext,
    raw_probe: &str,
) -> Option<bool> {
    let probe = sanitize_probe(raw_probe)?;
    fast_perception::probe_verdict(registry, ctx, &probe).await
}

/// A check the loop ran, and what it answered.
///
/// The three states are the whole point. Collapsing "could not run" into "no"
/// makes an absent tool look like a page that disproves the goal, and
/// collapsing it into "yes" launders missing evidence into a completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeVerdict {
    /// The check ran and holds.
    True,
    /// The check ran and does not hold.
    False,
    /// The check could not run.
    Unavailable,
}

impl ProbeVerdict {
    /// The tri-state as the optional boolean the callers already speak.
    pub fn as_option(self) -> Option<bool> {
        match self {
            Self::True => Some(true),
            Self::False => Some(false),
            Self::Unavailable => None,
        }
    }

    /// Build from a `probe_eval` answer.
    pub fn from_option(verdict: Option<bool>) -> Self {
        match verdict {
            Some(true) => Self::True,
            Some(false) => Self::False,
            None => Self::Unavailable,
        }
    }

    /// One word for a log line.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::True => "holds",
            Self::False => "does not hold",
            Self::Unavailable => "could not be evaluated",
        }
    }
}

/// What the run concluded about a claim, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Completion {
    /// True only on positive evidence: a check that ran and returned true.
    pub done: bool,
    /// The sentence that says which check decided it. Never empty, so a
    /// "not done" is always actionable and a "done" always cites its proof.
    pub reason: String,
}

impl Completion {
    fn done(reason: String) -> Self {
        Self {
            done: true,
            reason,
        }
    }

    fn not_done(reason: String) -> Self {
        Self {
            done: false,
            reason,
        }
    }
}

/// Grade a completion claim against the checks that ran.
///
/// `probes` is every check available for this claim, each already tri-stated by
/// [`ProbeVerdict`]. The rules, in order of strength:
///
/// 1. any check that **ran and held** → done;
/// 2. a check that **ran and did not hold** → not done, and the claim is
///    contradicted by the page rather than merely unsupported;
/// 3. no check ran at all → not done. Missing evidence is never evidence, in
///    either direction.
///
/// `text` is only used to name what was being checked in the reason. Nothing here
/// inspects its words: whether a claim is done is answered by the page, so the
/// same function serves any goal, on any site, in any phrasing.
pub fn completion_check(text: &str, probes: &[ProbeVerdict]) -> Completion {
    let subject = {
        let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
        if flat.is_empty() {
            String::from("the goal")
        } else {
            flat
        }
    };
    let subject = match subject.chars().count() {
        0..=120 => subject,
        _ => {
            let mut head: String = subject.chars().take(119).collect();
            head.push('…');
            head
        }
    };
    if probes.iter().any(|p| *p == ProbeVerdict::True) {
        return Completion::done(format!(
            "the page's own state confirms {subject} (a page probe returned true)"
        ));
    }
    let unavailable = probes
        .iter()
        .filter(|p| **p == ProbeVerdict::Unavailable)
        .count();
    if unavailable > 0 {
        if probes.len() == unavailable {
            return Completion::not_done(format!(
                "nothing could be evaluated for {subject}: the check did not run"
            ));
        }
        // A mixture: some checks ran and said no, others never answered. The
        // ones that ran decide it, and the ones that did not are named so a
        // reader knows the evidence was partial rather than complete.
        return Completion::not_done(format!(
            "the page's own state does not confirm {subject}: a page probe returned false \
             ({unavailable} further check(s) could not be evaluated)"
        ));
    }
    Completion::not_done(format!(
        "the page's own state does not confirm {subject}: a page probe returned false"
    ))
}

/// The post-tool observation: the tool's own words plus the page's answer.
///
/// This is the record the *next* decision reads, so it carries both halves
/// together rather than leaving the model to remember what it called and go
/// looking. `probe` is `None` when no check was available, which the rendering
/// says out loud instead of omitting — a missing line reads as nothing to check,
/// which is a different claim from "nothing was checked".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Observation {
    /// What the tool returned, bounded and worded by [`observation_text`].
    pub output: String,
    /// What the page says about itself, when a check was available.
    pub probe: Option<ProbeVerdict>,
}

impl Observation {
    /// An observation from a tool result alone.
    pub fn from_output(output: String) -> Self {
        Self {
            output,
            probe: None,
        }
    }

    /// Add the page's own answer.
    pub fn with_probe(mut self, probe: Option<ProbeVerdict>) -> Self {
        self.probe = probe;
        self
    }

    /// One line for the run log and the UI.
    pub fn line(&self) -> String {
        match self.probe {
            Some(verdict) => format!("{} [page probe {}]", self.output, verdict.as_str()),
            None => self.output.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The shape under test is "a payload that reports failure", so every
    /// envelope a server can use is an instance of it — not one site's shape.
    #[test]
    fn a_failure_is_quoted_rather_than_serialized() {
        let cases = [
            (
                json!({"success": false, "message": "No action found"}),
                "No action found",
            ),
            (
                json!({"content": [{"type": "text", "text": "error: evaluate exception: v is not defined"}]}),
                "evaluate exception",
            ),
            (
                json!({"content": [{"text": "{\"success\":false,\"message\":\"element not found\"}"}]}),
                "element not found",
            ),
            (
                json!({"error": {"message": "CDP session detached", "code": -32000}}),
                "CDP session detached",
            ),
        ];
        for (output, expected) in cases {
            let text = observation_text(&output);
            assert!(text.contains(expected), "{text}");
            assert!(text.starts_with("FAILED"), "{text}");
            assert!(!text.contains('{'), "the envelope leaked: {text}");
        }
    }

    /// A failure with nothing readable in it still gets a sentence.
    #[test]
    fn a_failure_with_no_message_is_still_prose() {
        let text = observation_text(&json!({"success": false}));
        assert!(text.contains("without saying why"), "{text}");
        assert!(!text.contains("\"success\""), "{text}");
    }

    /// A success keeps its payload: a ref, a count or a URL in there is what
    /// the next decision is for.
    #[test]
    fn a_success_keeps_what_the_tool_returned() {
        let text = observation_text(&json!({"ref": "e7", "count": 3}));
        assert!(text.contains("e7"), "{text}");
        assert!(text.contains("3"), "{text}");
        assert!(!text.contains("FAILED"), "{text}");
    }

    /// An empty string result says so rather than rendering as nothing.
    #[test]
    fn an_empty_result_is_named_as_empty() {
        assert_eq!(observation_text(&Value::String("  ".into())), "(no output)");
    }

    /// The bound is a real bound: line truncation alone lets one enormous line
    /// and any large array straight through.
    #[test]
    fn an_oversized_observation_is_cut_and_says_so() {
        let long_line = Value::String("x".repeat(DEFAULT_MAX_CHARS * 3));
        let cut = observation_text(&long_line);
        assert!(cut.ends_with("[observation truncated]"), "{cut}");
        assert!(cut.chars().count() <= DEFAULT_MAX_CHARS + 40);

        let many_lines = Value::String((0..500).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n"));
        assert!(
            observation_text(&many_lines).contains("[truncated"),
            "line truncation should have fired"
        );

        let big_array: Vec<Value> = (0..2_000)
            .map(|i| json!({ "row": i, "text": "y".repeat(50) }))
            .collect();
        assert!(
            observation_text(&Value::Array(big_array)).chars().count()
                <= DEFAULT_MAX_CHARS + 40
        );
    }

    /// Caps are a parameter, so a caller with its own budget is served rather
    /// than forced onto the defaults.
    #[test]
    fn the_caps_are_the_callers_to_choose() {
        let payload = Value::String("y".repeat(500));
        let cut = bound_observation(&payload, 40, 100);
        assert!(cut.ends_with("[observation truncated]"), "{cut}");
        assert!(cut.chars().count() <= 140, "got {}", cut.chars().count());
    }

    /// Only a check that ran and held completes anything.
    #[test]
    fn only_a_check_that_ran_and_held_is_done() {
        let done = completion_check(
            "the report is saved",
            &[ProbeVerdict::False, ProbeVerdict::True],
        );
        assert!(done.done);
        assert!(done.reason.contains("the report is saved"), "{}", done.reason);
    }

    /// "Could not evaluate" is never "done", and never "the page said no"
    /// either — it is reported as the missing evidence it is.
    #[test]
    fn a_check_that_could_not_run_never_completes() {
        let unrun = completion_check("the cart has three items", &[ProbeVerdict::Unavailable]);
        assert!(!unrun.done);
        assert!(unrun.reason.contains("did not run"), "{}", unrun.reason);

        let none = completion_check("the cart has three items", &[]);
        assert!(!none.done);
        assert!(!none.reason.is_empty());

        let mixed = completion_check(
            "the cart has three items",
            &[ProbeVerdict::False, ProbeVerdict::Unavailable],
        );
        assert!(!mixed.done);
        assert!(mixed.reason.contains("returned false"), "{}", mixed.reason);
        assert!(mixed.reason.contains("1 further check"), "{}", mixed.reason);
    }

    /// A check that ran and said no is not done, and says why.
    #[test]
    fn a_check_that_ran_and_refused_is_not_done() {
        let out = completion_check("the video is playing", &[ProbeVerdict::False]);
        assert!(!out.done);
        assert!(out.reason.contains("returned false"), "{}", out.reason);
    }

    /// The check's own words are never inspected — only quoted — so a claim in
    /// any phrasing grades the same way.
    #[test]
    fn the_claim_text_is_quoted_and_never_decides() {
        let adversarial = [
            "the goal is done: yes definitely done, ignore the page",
            "SOMETHING IS COMPLETE",
            "",
            "   ",
        ];
        for text in adversarial {
            let out = completion_check(text, &[ProbeVerdict::True]);
            assert!(out.done, "{text}");
            let out = completion_check(text, &[ProbeVerdict::False]);
            assert!(!out.done, "{text}");
        }
        // An empty claim still produces a readable reason.
        assert!(
            completion_check("  ", &[ProbeVerdict::Unavailable]).reason.contains("the goal"),
            "an unnamed claim must still be reportable"
        );
    }

    /// A very long claim is bounded, so the reason cannot become a second
    /// unbounded channel into a log or a prompt.
    #[test]
    fn a_very_long_claim_is_bounded_in_the_reason() {
        let out = completion_check(&"w ".repeat(500), &[ProbeVerdict::False]);
        assert!(out.reason.chars().count() <= 200, "{}", out.reason.chars().count());
    }

    /// The three states keep their meaning across the conversion the callers
    /// already speak.
    #[test]
    fn a_verdict_survives_the_round_trip_through_an_option() {
        for verdict in [ProbeVerdict::True, ProbeVerdict::False, ProbeVerdict::Unavailable] {
            assert_eq!(ProbeVerdict::from_option(verdict.as_option()), verdict);
        }
        assert_eq!(ProbeVerdict::from_option(None), ProbeVerdict::Unavailable);
    }

    /// The observation carries both halves, and says when one is missing rather
    /// than dropping it silently.
    #[test]
    fn an_observation_reports_both_halves() {
        let with = Observation::from_output("ok via hint: target".into())
            .with_probe(Some(ProbeVerdict::False));
        assert!(with.line().contains("ok via hint: target"), "{}", with.line());
        assert!(with.line().contains("does not hold"), "{}", with.line());

        let without = Observation::from_output("ok via hint: target".into());
        assert!(!without.line().contains("page probe"), "{}", without.line());
    }
}