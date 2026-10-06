//! The fast lane: thin, typed wrappers over the `hyprfast` tools that route to
//! Decider-2B.
//!
//! Every function here makes **one** `ToolRegistry` call and normalizes the
//! result. Two reasons it is a module and not a pile of inline `registry.execute`
//! calls in the loop:
//!
//! 1. The agent loop is measured in fast calls and slow calls, so each fast call
//!    is logged to `model-calls.jsonl` as a `classification` record with
//!    `operation` = the hyprfast tool name. That is what makes the two-speed
//!    thesis measurable after the fact instead of asserted.
//! 2. The whole lane is reachable in tests through `&ToolRegistry`: a fake tool
//!    registered under the same name answers exactly as a real hyprfast server
//!    would, with no browser and no Decider.
//!
//! Nothing here decides anything. Perception, grounding, and verification are
//! all Decider's job; this module only asks and normalizes.

use anyhow::{Result, bail};
use lucy_core::{
    AgentEvent, ApprovalGate, InterruptSignal, ModelCallKind, ModelCallRecord, ToolContext,
    log_model_call,
};
use lucy_tools::ToolRegistry;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc::UnboundedSender;

/// The hyprfast tools the fast lane calls. Names are the bare catalog names;
/// [`resolve_fast_tool`] maps them onto however the registry registered them.
pub const HINT_SNAPSHOT: &str = "hint_snapshot";
pub const HINT_ACT: &str = "hint_act";
pub const HINT_BATCH: &str = "hint_batch";
pub const FIND: &str = "find";
pub const FIND_AND_CLICK: &str = "find_and_click";
pub const FIND_AND_TYPE: &str = "find_and_type";
pub const VERIFY: &str = "verify";
pub const VERIFY_ACTION: &str = "verify_action";
pub const WAIT_UNTIL: &str = "wait_until";
pub const BROWSER_TABS: &str = "browser_tabs";
pub const DESKTOP: &str = "desktop";
pub const BROWSER_OPEN: &str = "browser_open";
pub const BROWSER_EVALUATE: &str = "browser_evaluate";
/// Drives the tab that is already open. `browser_open` is the one that
/// launches a browser, and [`ensure_browser`] owns that decision.
pub const BROWSER_NAVIGATE: &str = "browser_navigate";
/// One key into whatever is focused in the current tab. The only way to
/// submit a form the fast lane typed into: YouTube's search box, for one, does
/// nothing on keystrokes alone, and a `Type` that reports success without a
/// following `Enter` leaves the goal unmet however true the act looked.
pub const BROWSER_PRESS_KEY: &str = "browser_press_key";

/// Where a bootstrap-launched browser lands when the goal named no destination
/// of its own.
///
/// Deliberately **not** a specific site. It used to be `youtube.com`, which
/// meant every run that did not name a site — a Reddit task, a docs lookup, a
/// booking flow — began by loading YouTube and then navigating away. That is
/// one task's destination baked in as a global default, which is what AGENTS.md
/// forbids.
///
/// `about:blank` is the honest choice: it asserts nothing about the task. The
/// start page only has to be reachable for CDP to come up; the planner supplies
/// the real destination in its first navigation step, and
/// [`crate::turn::PLAN_INSTRUCTIONS`] already requires it to carry a full URL.
pub const BROWSER_START_URL: &str = "about:blank";
/// Bound on each CDP probe, so an unreachable port costs milliseconds rather
/// than a socket timeout.
const CDP_PROBE_TIMEOUT_MS: u64 = 1_200;
/// How long a just-launched browser gets to start answering the probe before
/// the bootstrap calls it a failure. Polling only — the launch itself is never
/// repeated.
const CDP_LAUNCH_GRACE_MS: u64 = 6_000;
const CDP_LAUNCH_POLL_MS: u64 = 300;

/// Counts + latency for every fast-lane call, so the loop can report the split
/// that is the whole point of the design.
#[derive(Debug, Default)]
pub struct FastCallStats {
    calls: AtomicU64,
    failures: AtomicU64,
    latency_ms: AtomicU64,
}

impl FastCallStats {
    pub fn calls(&self) -> u64 {
        self.calls.load(Ordering::SeqCst)
    }
    pub fn failures(&self) -> u64 {
        self.failures.load(Ordering::SeqCst)
    }
    pub fn total_latency_ms(&self) -> u64 {
        self.latency_ms.load(Ordering::SeqCst)
    }
    fn record(&self, ok: bool, ms: u64) {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if !ok {
            self.failures.fetch_add(1, Ordering::SeqCst);
        }
        self.latency_ms.fetch_add(ms, Ordering::SeqCst);
    }
}

/// Shared state for one agent-loop run: where events go, how to count calls,
/// which approval gate authorizes a fast tool, and the interrupt every fast
/// call must observe.
pub struct FastContext {
    pub session_id: lucy_core::SessionId,
    pub events: Option<UnboundedSender<AgentEvent>>,
    pub interrupt: InterruptSignal,
    pub stats: Arc<FastCallStats>,
    /// The runtime's shared gate when the loop runs inside a `LucyRuntime`.
    /// `None` in tests and for callers that gate elsewhere.
    pub gate: Option<ApprovalGate>,
    /// Tools the hyprfast catalog marks `destructive`. Carried into the fast
    /// lane so it gates on the same signal as the blind executor; without it the
    /// two lanes disagreed and destructive steps ran unattended here.
    destructive_tools: std::sync::Arc<std::collections::HashSet<String>>,
    /// Outcome of the one browser bootstrap this run is allowed, once the first
    /// browser-domain fast call asks for it. Caching it is what makes the
    /// bootstrap idempotent: a live browser is not relaunched, and a launch
    /// that failed is not retried.
    browser: Mutex<Option<BrowserBootstrap>>,
}

impl FastContext {
    pub fn new(
        session_id: lucy_core::SessionId,
        events: Option<UnboundedSender<AgentEvent>>,
        interrupt: InterruptSignal,
    ) -> Self {
        Self {
            session_id,
            events,
            interrupt,
            stats: Arc::new(FastCallStats::default()),
            gate: None,
            destructive_tools: Arc::new(std::collections::HashSet::new()),
            browser: Mutex::new(None),
        }
    }

    /// The same context, gated by a live [`ApprovalGate`].
    pub fn with_gate(mut self, gate: ApprovalGate) -> Self {
        self.gate = Some(gate);
        self
    }

    /// Supply the hyprfast `destructive` flags so this lane prompts exactly
    /// where the blind executor does.
    pub fn with_destructive_tools(mut self, tools: std::collections::HashSet<String>) -> Self {
        self.destructive_tools = Arc::new(tools);
        self
    }

    /// Registry verdict OR hyprfast `destructive`, matching `turn::validate_step`.
    pub(crate) fn approval_required(
        &self,
        registry: &lucy_tools::ToolRegistry,
        tool_name: &str,
    ) -> bool {
        registry.requires_approval(tool_name) || self.is_destructive(tool_name)
    }

    /// True when hyprfast flags this tool destructive under either its bare or
    /// `mcp_hyprfast_`-prefixed name.
    pub fn is_destructive(&self, tool_name: &str) -> bool {
        let bare = tool_name
            .strip_prefix("mcp_hyprfast_")
            .or_else(|| tool_name.strip_prefix("mcp_computer_use_"))
            .or_else(|| tool_name.strip_prefix("mcp_"))
            .unwrap_or(tool_name);
        self.destructive_tools.contains(tool_name) || self.destructive_tools.contains(bare)
    }

    pub fn tool_context(&self, call_id: &str) -> ToolContext {
        ToolContext {
            session_id: self.session_id.clone(),
            tool_call_id: call_id.to_owned(),
            working_dir: None,
            execution_mode: lucy_core::ExecutionMode::Agent,
            events: self
                .events
                .clone()
                .unwrap_or_else(|| tokio::sync::mpsc::unbounded_channel().0),
            interrupt: self.interrupt.clone(),
        }
    }

    pub fn emit(&self, event: AgentEvent) {
        if let Some(tx) = &self.events {
            let _ = tx.send(event);
        }
    }

    pub fn interrupted(&self) -> bool {
        self.interrupt.is_set()
    }

    /// The bootstrap already decided for this run, if any.
    pub fn browser_bootstrap(&self) -> Option<BrowserBootstrap> {
        self.browser.lock().ok().and_then(|guard| guard.clone())
    }

    /// Claim the single bootstrap attempt for this run. `false` when another
    /// call already claimed it, so only one `browser_open` can ever happen.
    pub fn claim_browser_bootstrap(&self, outcome: BrowserBootstrap) -> bool {
        match self.browser.lock() {
            Ok(mut guard) => {
                if guard.is_some() {
                    return false;
                }
                *guard = Some(outcome);
                true
            }
            Err(_) => false,
        }
    }
}

/// A cheap liveness probe for the CDP HTTP endpoint. Injectable so a test can
/// decide what the world looks like without opening a socket.
pub type CdpProbe = Arc<dyn Fn() -> Pin<Box<dyn Future<Output = bool> + Send>> + Send + Sync>;

/// Wrap an async closure as a [`CdpProbe`].
pub fn probe_fn<F, Fut>(probe: F) -> CdpProbe
where
    F: Fn() -> Fut + Send + Sync + 'static,
    Fut: Future<Output = bool> + Send + 'static,
{
    Arc::new(move || Box::pin(probe()))
}

/// `host:port` of the CDP endpoint, honouring the same environment the rest of
/// the codebase and the planner prompt document.
pub fn cdp_endpoint() -> (String, u16) {
    let host = std::env::var("HYPRFAST_CDP_HOST")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "127.0.0.1".to_owned());
    let port = std::env::var("HYPRFAST_CDP_PORT")
        .ok()
        .and_then(|v| v.trim().parse::<u16>().ok())
        .unwrap_or(9222);
    (host, port)
}

/// The real probe: one `GET /json/version` against the CDP port. A TCP connect
/// plus the status line is the whole question — no browser, no JSON parse, no
/// model call.
pub fn cdp_probe() -> CdpProbe {
    probe_fn(|| async move {
        let (host, port) = cdp_endpoint();
        cdp_answers(&host, port).await
    })
}

/// Poll the probe for a bounded window. Used only to confirm a launch: the
/// browser is polled, never relaunched.
async fn cdp_comes_up(probe: &CdpProbe) -> bool {
    let deadline = tokio::time::Instant::now() + Duration::from_millis(CDP_LAUNCH_GRACE_MS);
    loop {
        if probe().await {
            return true;
        }
        if tokio::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(CDP_LAUNCH_POLL_MS)).await;
    }
}

async fn cdp_answers(host: &str, port: u16) -> bool {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let wait = Duration::from_millis(CDP_PROBE_TIMEOUT_MS);
    let Ok(Ok(mut stream)) =
        tokio::time::timeout(wait, tokio::net::TcpStream::connect((host, port))).await
    else {
        return false;
    };
    // HTTP/1.1 on purpose: Chromium's DevTools HTTP server closes the
    // connection without answering an HTTP/1.0 request, which would read as
    // "no browser" on a browser that is up and serving.
    let request = format!(
        "GET /json/version HTTP/1.1\r\nHost: {host}:{port}\r\nUser-Agent: lucy-agent-loop\r\nAccept: */*\r\nConnection: close\r\n\r\n"
    );
    if tokio::time::timeout(wait, stream.write_all(request.as_bytes()))
        .await
        .is_err()
    {
        return false;
    }
    let deadline = tokio::time::Instant::now() + wait;
    let mut head = Vec::with_capacity(64);
    let mut buf = [0u8; 64];
    while head.len() < 64 {
        let Ok(Ok(n)) = tokio::time::timeout_at(deadline, stream.read(&mut buf)).await else {
            return false;
        };
        if n == 0 {
            break;
        }
        head.extend_from_slice(&buf[..n]);
        if head.windows(2).any(|w| w == b"\r\n") {
            break;
        }
    }
    let head = String::from_utf8_lossy(&head);
    head.starts_with("HTTP/1.")
        && head
            .lines()
            .next()
            .is_some_and(|line| line.contains(" 200"))
}

/// Map a bare hyprfast tool name onto the name the registry registered it
/// under (`mcp_hyprfast_*`, `mcp_computer_use_*`, or bare). `None` when the
/// live catalog does not advertise it.
pub fn resolve_fast_tool(registry: &ToolRegistry, name: &str) -> Option<String> {
    crate::turn::resolve_plan_tool(registry, name)
}

/// What `hint_snapshot` reported, normalized to the two things the loop
/// actually uses: the clickable/typeable labels (fed back to the LLM on a
/// replan) and a fingerprint (anti-thrash).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ScreenState {
    /// `label` / `name` / `text` of each hint, in hint order. These are the
    /// session-local keys the hint pipeline resolves and clicks with.
    pub labels: Vec<String>,
    /// The same hints written the way a planner can read them: the element's
    /// accessible name/text, paired with its label (`despacito (S)`). Ordinals
    /// like `A`/`S`/`J` mean nothing to a model, so this — not `labels` — is
    /// what the summary shows.
    pub described: Vec<String>,
    pub count: usize,
    /// How the snapshot was produced (`decider`, `dom`, …) when reported.
    pub via: Option<String>,
    /// Free-form remainder, kept so a replan prompt can show something useful
    /// even for a tool whose shape we do not model.
    pub detail: String,
}

impl ScreenState {
    /// Stable hash of the observable state: url + hint labels + element names +
    /// counts. Two consecutive identical fingerprints mean the screen did not
    /// move, which is the signal that the loop is acting without effect.
    pub fn fingerprint(&self) -> u64 {
        let mut acc: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |s: &str| {
            for b in s.as_bytes() {
                acc ^= u64::from(*b);
                acc = acc.wrapping_mul(0x0000_0100_0000_01b3);
            }
            acc ^= 0xff;
            acc = acc.wrapping_mul(0x0000_0100_0000_01b3);
        };
        mix(&self.detail);
        mix(&self.via.clone().unwrap_or_default());
        mix(&self.count.to_string());
        for label in &self.labels {
            mix(label);
        }
        for described in &self.described {
            mix(described);
        }
        acc
    }

    /// One capped line for a replan prompt: what is on screen, briefly. Names
    /// lead, labels ride along as the disambiguator.
    pub fn summary(&self) -> String {
        let shown: &[String] = if self.described.is_empty() {
            &self.labels
        } else {
            &self.described
        };
        if shown.is_empty() {
            return if self.detail.is_empty() {
                "no interactive elements detected".to_owned()
            } else {
                crate::turn::summarize_output(&Value::String(self.detail.clone()))
            };
        }
        let head: Vec<String> = shown.iter().take(24).cloned().collect();
        let mut s = format!("{} element(s): {}", self.count, head.join(", "));
        if self.count > head.len() {
            s.push_str(&format!(", … +{}", self.count - head.len()));
        }
        crate::turn::summarize_output(&Value::String(s))
    }
}

/// A hint description short enough to be a label rather than part of the name.
const MAX_HINT_LABEL: usize = 4;
/// Below this a matched name is not a target, it is a stray glyph.
const MIN_ANCHOR_NAME: usize = 3;

/// Cap on a failure reason. It is joined from up to several payload fields and
/// reaches the run report, so an uncapped provider paragraph would dominate the
/// sentence the reader is actually trying to read.
const MAX_FAILURE_REASON_CHARS: usize = 200;
/// Below this an instruction word is noise, not an identity: "a", "in", "ft".
const MIN_ANCHOR_WORD: usize = 3;

/// The words a planner spends its `suggested_action` on that say nothing about
/// *which* element is meant: the verb, the grammar, and the nouns that name a
/// kind of thing rather than an instance of one.
///
/// They are stripped before anything is scored, so `click the video result
/// whose title contains Despacito` is judged on `despacito` alone and the
/// "Videos" filter tab cannot be the answer. `first`/`second`/`one`/`two`/`three`
/// are here for the ordinals the plan prompt already forbids — a real model
/// cannot count the screen either, and a name that happens to be `Next` must
/// not become the target of "the first result".
const ANCHOR_STOPWORDS: &[&str] = &[
    // the verbs the loop itself performs
    "click",
    "type",
    "press",
    "select",
    // the grammar around the target
    "the",
    "a",
    "an",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "their",
    "there",
    "here",
    "and",
    "or",
    "but",
    "so",
    "if",
    "then",
    "than",
    "when",
    "where",
    "which",
    "who",
    "while",
    "until",
    "with",
    "without",
    "within",
    "into",
    "onto",
    "upon",
    "about",
    "over",
    "under",
    "after",
    "before",
    "between",
    "of",
    "for",
    "from",
    "at",
    "on",
    "in",
    "to",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "has",
    "have",
    "had",
    "not",
    "please",
    "now",
    "just",
    "only",
    "also",
    "both",
    "each",
    "every",
    "other",
    "another",
    "such",
    "some",
    "too",
    "can",
    "could",
    "should",
    "would",
    "will",
    "may",
    "might",
    "must",
    "you",
    "i",
    "my",
    "we",
    "us",
    "whose",
    "containing",
    "contains",
    "include",
    "includes",
    "including",
    // nouns that name a kind of element rather than one element
    "link",
    "button",
    "field",
    "box",
    "element",
    "page",
    "item",
    "text",
    "name",
    "named",
    "label",
    "tab",
    "row",
    "card",
    "entry",
    "menu",
    "header",
    "footer",
    "toolbar",
    "icon",
    "image",
    "picture",
    "photo",
    "thumbnail",
    "song",
    "track",
    "option",
    "section",
    "window",
    "screen",
    "result",
    "results",
    "video",
    "videos",
    "search",
    // words that name a position rather than a thing: the ordinals the plan
    // prompt forbids, and the spatial ones a model writes anyway
    "first",
    "second",
    "third",
    "fourth",
    "one",
    "two",
    "three",
    "left",
    "right",
    "top",
    "bottom",
    "last",
    "next",
    "previous",
    "again",
];

/// Content words that name a *kind* of element rather than *which* one. They
/// still add to a name's score — a name carrying both the specific and the
/// generic word is a better answer — but a name justified by these alone is not
/// an anchor at all, because "the thing called Video" is a guess wearing a
/// target's clothes. `title` is the one that matters for the measured failure:
/// the instruction says "whose title contains Despacito", and every result row
/// on the page is a title.
const ANCHOR_GENERIC: &[&str] = &[
    "title",
    "titled",
    "thumbnail",
    "image",
    "picture",
    "photo",
    "song",
    "track",
    "row",
    "card",
    "entry",
    "icon",
    "widget",
    "control",
    "option",
    "section",
    "content",
    "thing",
    "things",
];

/// Strip the session-local hint label from a `ScreenState::described` entry:
/// `Luis Fonsi - Despacito ft. Daddy Yankee (Z)` is the name, `(Z)` is the key
/// the hint pipeline clicks with.
///
/// Only a group shaped like a hint label is stripped — one to four keypress
/// characters, no spaces — so a name that legitimately ends in a parenthetical
/// (`Despacito (Official Video)`) keeps it. The label is never part of what the
/// element is called, and leaving it in would let a two-letter key score
/// against a two-letter content word.
fn strip_hint_label(entry: &str) -> String {
    let entry = entry.trim();
    let Some(head) = entry.strip_suffix(')') else {
        return entry.to_owned();
    };
    let Some(open) = head.rfind('(') else {
        return entry.to_owned();
    };
    let label = &head[open + 1..];
    if label.is_empty()
        || label.chars().count() > MAX_HINT_LABEL
        || !label.chars().all(|c| c.is_ascii_alphanumeric())
    {
        return entry.to_owned();
    }
    head[..open].trim_end().to_owned()
}

/// The name on screen that `instruction` is most plausibly about, or `None`.
///
/// Pure, offline, and deliberately cheap: no model call, no screen read, no
/// state. That is the whole point — the loop needs a second opinion it can
/// afford on an attempt it has already spent.
///
/// The measured incident this exists for. `lucy ask "play despacito on
/// youtube"` reached the YouTube results page correctly and then burned its
/// whole budget there. The planner wrote "click the video result whose title
/// contains Despacito" *before* it had seen the page, so it described the
/// target instead of naming it, and Decider-2B — asked to turn that description
/// into one of the 32 names on screen — answered "All" (the filter tab) on the
/// first attempt and "Clear search query" (which wiped the query box) on the
/// second. Handed the same page and the quoted name
/// `Luis Fonsi - Despacito ft. Daddy Yankee 4 minutes, 42 seconds` it clicked
/// the right thing both times. The resolver was never the problem; the
/// instruction was, and the loop's answer to a bad instruction was to repeat
/// it byte for byte.
///
/// A name is scored by the total length of the instruction's *content* words it
/// contains, case-insensitively, either way round (so `video` finds `Videos`),
/// each word counted once. `None` for an instruction that says nothing
/// specific, and for a screen it cannot read a specific word off — so a caller
/// that gets nothing keeps exactly the behaviour it had.
pub fn anchor_to_visible_name(instruction: &str, described: &[String]) -> Option<String> {
    if described.is_empty() {
        return None;
    }
    let words = anchor_content_words(instruction);
    if words.is_empty() {
        return None;
    }
    let mut best: Option<(usize, String)> = None;
    for entry in described {
        let name = strip_hint_label(entry);
        let hay = name.to_lowercase();
        // A name that is nothing but a word the instruction already used is not
        // an anchor: the instruction already said `despacito`, so re-anchoring
        // to `despacito` hands the resolver back the token that just failed —
        // and on a results page that token is the search box holding the query.
        if name.chars().count() < MIN_ANCHOR_NAME || words.contains(&hay) {
            continue;
        }
        let mut score = 0usize;
        let mut specific = false;
        for word in &words {
            if !name_carries_word(&hay, word) {
                continue;
            }
            score += word.chars().count();
            specific |= !ANCHOR_GENERIC.contains(&word.as_str());
        }
        if score == 0 || !specific {
            continue;
        }
        // Earliest on screen wins a tie, not the longest name. On a results
        // page the first row that matches is the primary result, whereas
        // "longer name" is a proxy for "more words the instruction did not
        // say" and hands the tie to `Justin Bieber - Despacito (Lyrics /
        // Letra) ft. Luis Fonsi & Daddy Yankee 3 minutes, 51 seconds` over
        // `Luis Fonsi - Despacito ft. Daddy Yankee 4 minutes, 42 seconds` — a
        // different song, picked for being wordier. First match is also the
        // order the hint pipeline itself enumerates, so the anchor is the
        // element the resolver would have reached first anyway.
        if best
            .as_ref()
            .is_none_or(|(best_score, _)| score > *best_score)
        {
            best = Some((score, name));
        }
    }
    best.map(|(_, name)| name)
}

/// Does this lowercased name carry this lowercased instruction word? Symmetric
/// on purpose — `video` has to find `Videos` — but the reverse direction (a
/// name that is a fragment of a longer word) only counts when the name is at
/// least half the word, so `Des` is not evidence about `despacito` while `video`
/// is evidence about `Videos`.
fn name_carries_word(name: &str, word: &str) -> bool {
    if name.contains(word) {
        return true;
    }
    word.contains(name) && name.chars().count() * 2 >= word.chars().count()
}

/// The instruction's tokens minus everything in [`ANCHOR_STOPWORDS`],
/// lowercased, de-duplicated, and long enough to identify anything.
fn anchor_content_words(instruction: &str) -> Vec<String> {
    let mut words: Vec<String> = Vec::new();
    for token in instruction.split(|c: char| !c.is_alphanumeric()) {
        let word = token.to_lowercase();
        if word.chars().count() < MIN_ANCHOR_WORD || ANCHOR_STOPWORDS.contains(&word.as_str()) {
            continue;
        }
        if !words.contains(&word) {
            words.push(word);
        }
    }
    words
}

/// One candidate element resolved by Decider — "where do I click", without
/// clicking.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResolvedTarget {
    /// The label/name Decider matched.
    pub label: String,
    pub selector: Option<String>,
    pub rect: Option<(f64, f64, f64, f64)>,
    pub confidence: f64,
}

impl ResolvedTarget {
    /// True when the match is too weak to act on blind. Deliberately
    /// conservative: a wrong click on a live desktop is worse than a replan.
    pub fn is_confident(&self) -> bool {
        self.confidence >= MIN_RESOLVE_CONFIDENCE && !self.label.trim().is_empty()
    }
}

/// Below this, [`decide_target`] returns "ambiguous" and the loop replans.
pub const MIN_RESOLVE_CONFIDENCE: f64 = 0.55;

/// The outcome of one `hint_act` (or its `find_and_*` fallback).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ActionOutcome {
    pub success: bool,
    /// Which resolution tier fired (`hint`, `decider`, `vision`, …).
    pub tier: Option<String>,
    pub label: Option<String>,
    pub message: String,
}

impl ActionOutcome {
    /// One capped line for a replan prompt or the TUI log. A failure always
    /// carries its reason: the loop's log is the only place a reader learns why
    /// a step did not land.
    pub fn summary(&self) -> String {
        let head = if self.success { "ok" } else { "failed" };
        let mut s = String::new();
        match (&self.tier, &self.label) {
            (Some(tier), Some(label)) => s.push_str(&format!("{head} via {tier}: {label}")),
            (Some(tier), None) => s.push_str(&format!("{head} via {tier}")),
            (None, Some(label)) => s.push_str(&format!("{head}: {label}")),
            (None, None) => s.push_str(head),
        }
        if !self.success && !self.message.trim().is_empty() {
            s.push_str(&format!(": {}", self.message.trim()));
        }
        s
    }
}

/// The result of asking Decider whether something is now true.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyOutcome {
    Satisfied,
    NotSatisfied,
    Uncertain,
}

impl VerifyOutcome {
    pub fn is_settled(&self) -> bool {
        matches!(self, Self::Satisfied | Self::NotSatisfied)
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Satisfied => "satisfied",
            Self::NotSatisfied => "not satisfied",
            Self::Uncertain => "uncertain",
        }
    }
}

/// Run one fast-lane tool call: emit `ToolStarted`/`ToolFinished`, log it to
/// `model-calls.jsonl` as a classification call, and count it.
pub async fn call_fast(
    registry: &ToolRegistry,
    ctx: &FastContext,
    tool: &str,
    input: Value,
) -> Result<Value> {
    let Some(resolved) = resolve_fast_tool(registry, tool) else {
        let err = anyhow::anyhow!("'{tool}' is not in the live hyprfast catalog");
        record_call(ctx, tool, 0, Some(&err.to_string()));
        return Err(err);
    };
    let started = std::time::Instant::now();
    ctx.emit(AgentEvent::ToolStarted {
        id: format!("fast-{tool}"),
        name: resolved.clone(),
        input: input.clone(),
    });
    // Routine UI actions run unattended. `requires_approval` folds in the
    // hyprfast catalog's own `destructive` flag, matching the blind executor at
    // turn.rs — previously this path consulted only the registry, so a
    // destructive step was gated in one lane and waved through in the other.
    if let Some(gate) = &ctx.gate
        && gate.needs_approval(&resolved, ctx.approval_required(registry, &resolved))
    {
        // Raced against the interrupt: a kill switch that lands while the prompt
        // is open returns `None` and the call is abandoned, instead of the run
        // sitting out the 300s timeout on a dialog nobody is going to answer.
        let decision = gate
            .ask_cancellable(&ctx.interrupt, &format!("agent-{tool}"), &resolved, &input)
            .await;
        let (denied, err) = match decision {
            None => (true, format!("stopped: {resolved}")),
            Some(lucy_core::ApprovalDecision::Deny) => (true, format!("denied: {resolved}")),
            Some(_) => (false, String::new()),
        };
        if denied {
            let err = anyhow::anyhow!("{err}");
            record_call(
                ctx,
                tool,
                started.elapsed().as_millis() as u64,
                Some(&err.to_string()),
            );
            ctx.emit(AgentEvent::ToolFinished {
                id: format!("fast-{tool}"),
                name: resolved,
                output: Value::String(err.to_string()),
                is_error: true,
            });
            return Err(err);
        }
    }
    let call_ctx = ctx.tool_context(&format!("agent-{tool}"));
    let out = registry.execute(&resolved, input, call_ctx).await;
    let ms = started.elapsed().as_millis() as u64;
    let is_err = out.is_err() || tool_reports_failure(out.as_ref().ok());
    // A tool-level failure arrives inside an `Ok` payload; the log still gets
    // the reason hyprfast actually reported rather than a bare "failed".
    let (payload, err_text) = match &out {
        Ok(v) => (v.clone(), is_err.then(|| failure_reason(v))),
        Err(e) => (Value::String(e.to_string()), Some(e.to_string())),
    };
    ctx.emit(AgentEvent::ToolFinished {
        id: format!("fast-{tool}"),
        name: resolved,
        output: payload,
        is_error: is_err,
    });
    record_call(ctx, tool, ms, err_text.as_deref());
    out
}

/// Append one fast-lane record to the unified model-call log, so
/// `model-calls.jsonl` shows the LLM-vs-fast split directly.
fn record_call(ctx: &FastContext, tool: &str, ms: u64, err: Option<&str>) {
    let mut rec = ModelCallRecord::new(
        ModelCallKind::Tool,
        tool,
        "hyprfast",
        "decider-2b",
        ms,
    )
    .with_purpose("fast_lane");
    if let Some(e) = err {
        rec = rec.failed(e);
    }
    log_model_call(&rec);
    ctx.stats.record(err.is_none(), ms);
}

/// hyprfast reports tool-level failure inside an `Ok` payload; the loop must
/// not read that as success.
///
/// Crate-visible because "did this call actually work" has to be one answer:
/// the fast lane and the ReAct loop both read the same payload, and two
/// implementations of this convention is how one of them starts reporting
/// `hint-act`'s empty stdout as success.
pub(crate) fn tool_reports_failure(out: Option<&Value>) -> bool {
    let Some(out) = out else {
        return true;
    };
    if out.get("success") == Some(&Value::Bool(false)) {
        return true;
    }
    if let Some(content) = out.get("content").and_then(Value::as_array) {
        for item in content {
            if let Some(text) = item.get("text").and_then(Value::as_str)
                && text.trim_start().starts_with("error:")
            {
                return true;
            }
        }
    }
    false
}

/// What the run's single browser bootstrap did. `Failed` is terminal: the
/// launch is attempted once, and after that the fast call fails on its own so
/// the existing truthful reporting owns the error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BrowserBootstrap {
    /// CDP already answered, so nothing was launched.
    AlreadyRunning,
    /// CDP was down; `browser_open` launched a browser and CDP answered after.
    Launched { url: String },
    /// The goal is not a web goal, so the loop must not launch anything.
    NotNeeded,
    /// The launch was attempted once and CDP stayed down.
    Failed { reason: String },
}

impl BrowserBootstrap {
    pub fn is_failure(&self) -> bool {
        matches!(self, Self::Failed { .. })
    }

    /// One line for the run log.
    pub fn summary(&self) -> String {
        match self {
            Self::AlreadyRunning => "browser already listening on CDP".to_owned(),
            Self::Launched { url } => format!("no browser was listening — launched one at {url}"),
            Self::NotNeeded => "not a web goal — no browser launched".to_owned(),
            Self::Failed { reason } => format!("browser launch failed: {reason}"),
        }
    }
}

/// The deterministic precondition for launching anything: a web goal gets a
/// browser, a desktop goal does not. It must be cheap, offline, and never a
/// model call — a model call here would cost more than the whole fast lane.
///
/// Two classes of signal, deliberately separated:
///
/// * **Structural, built in.** A URL, a dotted domain, or the user naming the
///   medium ("browse", "search for"). These cannot go stale, because they are
///   properties of the *request*, not of a topic.
/// * **Topical, user-configured** (`harness.browser_goal_keywords`). "book a
///   flight" has no URL and no dotted domain, so something has to know that
///   "flight" is a web-ish noun. That list lives in config rather than in this
///   function: a hardcoded list of topics is a closed world that needs a code
///   change and a release for every new noun, which AGENTS.md forbids. The user
///   adds "recipe", "reservation", "invoice" and it works immediately.
///
/// The fallback direction is deliberate: an unrecognised web goal launches no
/// browser and the run says so, rather than silently opening one for a question
/// like "what is 2 + 2".
pub fn goal_needs_browser(goal: &str, topic_keywords: &[String]) -> bool {
    let low = goal.to_lowercase();
    const STRUCTURAL: &[&str] = &[
        "http://",
        "https://",
        ".com",
        ".org",
        ".net",
        ".io",
        "browser",
        "browse",
        "web",
        "website",
        "search on",
        "search for",
    ];
    if STRUCTURAL.iter().any(|needle| low.contains(needle)) {
        return true;
    }
    topic_keywords
        .iter()
        .map(|k| k.trim().to_lowercase())
        .filter(|k| !k.is_empty())
        .any(|k| low.contains(&k))
}

/// **FAST bootstrap.** Make sure a browser exists before the first fast-lane
/// browser call, because a web task cannot be attempted at all while CDP is
/// unreachable — every fast call would short-circuit on the same error.
///
/// The precondition is deterministic, not a model step: a web goal, probed
/// with one cheap HTTP request. Only then does it spend one `browser_open`
/// tool call, which goes through [`call_fast`] like every other fast call so it
/// is gated, evented, and logged to `model-calls.jsonl` as `fast_lane`.
///
/// `topic_keywords` is `harness.browser_goal_keywords` — see
/// [`goal_needs_browser`] for why topical words are configuration and not code.
///
/// `probe` is injectable; `None` uses the real CDP endpoint. The outcome is
/// cached on the context, so this is once per run and a failed launch is never
/// retried.
pub async fn ensure_browser(
    registry: &ToolRegistry,
    ctx: &FastContext,
    goal: &str,
    start_url: &str,
    topic_keywords: &[String],
    probe: Option<CdpProbe>,
) -> BrowserBootstrap {
    if let Some(done) = ctx.browser_bootstrap() {
        return done;
    }
    let probe = probe.unwrap_or_else(cdp_probe);
    let outcome = if !goal_needs_browser(goal, topic_keywords) {
        BrowserBootstrap::NotNeeded
    } else if probe().await {
        BrowserBootstrap::AlreadyRunning
    } else {
        let url = match start_url.trim() {
            "" => BROWSER_START_URL,
            u => u,
        };
        let attempt = call_fast(registry, ctx, BROWSER_OPEN, json!({ "url": url })).await;
        let reason = match &attempt {
            Ok(v) if tool_reports_failure(Some(v)) => failure_reason(v),
            Ok(_) => String::new(),
            Err(e) => e.to_string(),
        };
        if cdp_comes_up(&probe).await {
            BrowserBootstrap::Launched {
                url: url.to_owned(),
            }
        } else {
            let (host, port) = cdp_endpoint();
            BrowserBootstrap::Failed {
                reason: if reason.is_empty() {
                    format!("CDP at {host}:{port} still unreachable after browser_open")
                } else {
                    reason
                },
            }
        }
    };
    if !ctx.claim_browser_bootstrap(outcome.clone()) {
        // Another call claimed the single attempt first; its outcome is the
        // run's answer.
        return ctx.browser_bootstrap().unwrap_or(BrowserBootstrap::Failed {
            reason: "bootstrap claimed by another fast call".to_owned(),
        });
    }
    // Announced here rather than by the caller, so a launch is reported once
    // per run instead of once per attempt.
    if !matches!(
        outcome,
        BrowserBootstrap::AlreadyRunning | BrowserBootstrap::NotNeeded
    ) {
        ctx.emit(AgentEvent::Status {
            message: outcome.summary(),
        });
    }
    outcome
}

/// **FAST perceive.** One `hint_snapshot` call → what is clickable right now.
pub async fn perceive_screen(registry: &ToolRegistry, ctx: &FastContext) -> ScreenState {
    let out = match call_fast(registry, ctx, HINT_SNAPSHOT, json!({})).await {
        Ok(v) => v,
        Err(_) => return ScreenState::default(),
    };
    screen_state_from_hint_snapshot(&out)
}

/// Pure parser for `hint_snapshot` output, split out so the shape can be tested
/// without a registry.
pub fn screen_state_from_hint_snapshot(out: &Value) -> ScreenState {
    // The registry hands back hyprfast's MCP envelope, not the bare payload.
    // Without this the labels stay empty and the fingerprint is constant, which
    // makes the anti-thrash guard fire on a screen that really did change.
    let out = &unwrap_envelope(out);
    let hints = out
        .get("hints")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let (labels, described): (Vec<String>, Vec<String>) = hints
        .iter()
        .map(|h| (hint_label(h), hint_description(h)))
        .unzip();
    ScreenState {
        count: out
            .get("count")
            .and_then(Value::as_u64)
            .map(|c| c as usize)
            .unwrap_or(hints.len()),
        labels,
        described,
        via: out.get("via").and_then(Value::as_str).map(str::to_owned),
        detail: crate::turn::summarize_output(&strip_hints(out)),
    }
}

/// How long one element's name may get in a summary line, so one long video
/// title cannot crowd out the rest of the screen.
const HINT_NAME_MAX: usize = 48;

/// The clickable key: `label` first, because that is what the hint pipeline
/// resolves. Name and text are the fallbacks when a hint carries no key.
fn hint_label(h: &Value) -> String {
    first_text(h, &["label", "name", "text"])
        .unwrap_or_else(|| "(unlabelled)".to_owned())
        .trim()
        .to_owned()
}

/// What a planner can act on: the element's accessible name (or its text),
/// with the session-local label kept as a disambiguator — `despacito (S)`,
/// `Luis Fonsi - Despacito ft. Daddy Yankee (Y)`. A hint with no human
/// identity at all falls back to its label, so a line never goes blank.
fn hint_description(h: &Value) -> String {
    let label = first_text(h, &["label"]);
    let name = first_text(h, &["name"]).or_else(|| first_text(h, &["text"]));
    let name = name.map(|n| {
        let n = n.trim();
        if n.chars().count() > HINT_NAME_MAX {
            let mut t: String = n.chars().take(HINT_NAME_MAX - 1).collect();
            t.push('…');
            t
        } else {
            n.to_owned()
        }
    });
    match (name, label) {
        (Some(name), Some(label)) if !name.is_empty() && name != label => {
            format!("{name} ({label})")
        }
        (Some(name), _) if !name.is_empty() => name,
        (_, label) => label.unwrap_or_else(|| "(unlabelled)".to_owned()),
    }
}

/// First non-empty trimmed string among `keys`, reading a nested object for
/// `message`/`text` as well so `{"error": {"message": …}}` is not missed.
fn first_text(value: &Value, keys: &[&str]) -> Option<String> {
    for key in keys {
        let Some(found) = value.get(*key) else {
            continue;
        };
        let text = match found {
            Value::String(s) => Some(s.clone()),
            Value::Object(_) => first_text(found, &["message", "text", "reason"])
                .or_else(|| Some(found.to_string())),
            _ => None,
        };
        if let Some(t) = text.filter(|t| !t.trim().is_empty()) {
            return Some(t);
        }
    }
    None
}

/// `hint_snapshot` output minus the hint array, so the fingerprint and the
/// replan prompt do not carry two copies of every label.
fn strip_hints(out: &Value) -> Value {
    match out.as_object() {
        Some(o) => Value::Object(
            o.iter()
                .filter(|(k, _)| *k != "hints")
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        ),
        None => out.clone(),
    }
}

/// **FAST decide.** Resolve a natural-language target without touching the
/// screen. `find` first (semantic resolver with its own candidate filtering),
/// then `hint_resolve` for phrasings `find` cannot phrase-match.
pub async fn decide_target(
    registry: &ToolRegistry,
    ctx: &FastContext,
    instruction: &str,
) -> Option<ResolvedTarget> {
    let found = call_fast(registry, ctx, FIND, json!({ "query": instruction }))
        .await
        .ok()
        .and_then(|v| resolved_target_from_find(&v));

    if let Some(target) = found.as_ref().filter(|t| t.is_confident()) {
        return Some(target.clone());
    }
    let fallback = call_fast(
        registry,
        ctx,
        "hint_resolve",
        json!({ "instruction": instruction }),
    )
    .await
    .ok()
    .and_then(|v| resolved_target_from_find(&v));
    match (found, fallback) {
        (Some(f), Some(_)) if f.is_confident() => Some(f),
        (Some(f), Some(b)) => Some(if b.confidence > f.confidence { b } else { f }),
        (Some(f), None) => Some(f),
        (None, other) => other,
    }
}

/// Pure parser for `find` / `hint_resolve` output.
pub fn resolved_target_from_find(out: &Value) -> Option<ResolvedTarget> {
    // hyprfast nests its payload inside the MCP content envelope, sometimes
    // stringified: unwrap one or two levels before reading fields.
    let root = unwrap_envelope(out);
    let candidate = pick_candidate(&root)?;
    let label = ["label", "name", "text", "query", "selector"]
        .iter()
        .filter_map(|k| candidate.get(*k).and_then(Value::as_str))
        .find(|s| !s.trim().is_empty())
        .unwrap_or_default()
        .trim()
        .to_owned();
    let selector = candidate
        .get("selector")
        .or_else(|| candidate.get("ref"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let rect = candidate.get("rect").and_then(parse_rect);
    let confidence = ["confidence", "score", "probability"]
        .iter()
        .find_map(|k| candidate.get(*k).and_then(value_as_f64))
        .or_else(|| {
            ["confidence", "score", "probability"]
                .iter()
                .find_map(|k| root.get(*k).and_then(value_as_f64))
        })
        .unwrap_or(0.0);
    if label.is_empty() && selector.is_none() && rect.is_none() {
        return None;
    }
    Some(ResolvedTarget {
        label,
        selector,
        rect,
        confidence,
    })
}

fn pick_candidate(root: &Value) -> Option<Value> {
    for key in [
        "candidate",
        "resolved",
        "target",
        "match",
        "element",
        "best",
    ] {
        if let Some(v) = root.get(key)
            && v.is_object()
        {
            return Some(v.clone());
        }
    }
    if let Some(arr) = root.get("candidates").and_then(Value::as_array)
        && let Some(first) = arr.first()
    {
        return Some(first.clone());
    }
    if root.get("selector").is_some() || root.get("rect").is_some() {
        return Some(root.clone());
    }
    None
}

fn parse_rect(v: &Value) -> Option<(f64, f64, f64, f64)> {
    let o = v.as_object()?;
    let num = |k: &str| o.get(k).and_then(value_as_f64);
    match (num("x"), num("y"), num("width"), num("height")) {
        (Some(x), Some(y), Some(w), Some(h)) => Some((x, y, w, h)),
        (Some(x), Some(y), None, None) => Some((x, y, 0.0, 0.0)),
        _ => None,
    }
}

fn value_as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// Peel the MCP `{"content":[{"text":"<json>"}]}` envelope hyprfast returns.
pub fn unwrap_envelope(out: &Value) -> Value {
    let Some(items) = out.get("content").and_then(Value::as_array) else {
        return out.clone();
    };
    for item in items {
        let Some(text) = item.get("text").and_then(Value::as_str) else {
            continue;
        };
        let trimmed = text.trim();
        if let Ok(inner) = serde_json::from_str::<Value>(trimmed) {
            return inner;
        }
        if trimmed.starts_with('{') || trimmed.starts_with('[') {
            return json!({ "text": trimmed });
        }
    }
    out.clone()
}

/// Why a fast call failed, in the order a payload can carry the reason: an
/// explicit `error`/`message` field, then the MCP envelope's text (which is
/// where hyprfast puts its CLI error), then the envelope's error flags.
///
/// Never blank. A tool that fails without a message is the case this exists
/// for: `hint-act` can exit 0 with empty stdout, and `failed:  | find failed
/// or uncertain` tells a reader nothing.
pub fn failure_reason(out: &Value) -> String {
    let root = unwrap_envelope(out);
    let mut parts: Vec<String> = Vec::new();
    let mut take = |text: String| {
        let text = text.trim().to_owned();
        if !text.is_empty() && !parts.contains(&text) {
            parts.push(text);
        }
    };
    for source in [root.clone(), out.clone()] {
        for key in ["error", "message", "reason", "detail"] {
            if let Some(text) = first_text(&source, &[key]) {
                take(text);
            }
        }
    }
    if let Some(items) = out.get("content").and_then(Value::as_array) {
        for item in items {
            if let Some(text) = first_text(item, &["text"]) {
                take(text);
            }
        }
    }
    if parts.is_empty() {
        for source in [out, &root] {
            if source.get("isError") == Some(&Value::Bool(true))
                || source.get("is_error") == Some(&Value::Bool(true))
            {
                return "tool reported an error with no message".to_owned();
            }
        }
        // The old fallback quoted the serialized payload — a JSON blob inside a
        // sentence inside a status line, which is how a raw `{"content":…}` ended
        // up in the chat. The value is still logged by the caller; what the
        // reader gets is the fact that nothing was reported.
        return "no reason reported (the tool returned no readable message)".to_owned();
    }
    // Joined reasons are one line by construction, but a payload can carry a
    // paragraph per reason, and this string goes into the run report.
    let joined = parts.join(" | ");
    if joined.chars().count() > MAX_FAILURE_REASON_CHARS {
        let mut t: String = joined.chars().take(MAX_FAILURE_REASON_CHARS - 1).collect();
        t.push('…');
        return t;
    }
    joined
}

/// **FAST act.** One self-resolving interaction. `hint_act` is the primary
/// verb (it resolves the target itself, so the loop never needs a `ref`);
/// `find_and_click` / `find_and_type` are the fallbacks when the hint tiers
/// could not resolve.
pub async fn act(
    registry: &ToolRegistry,
    ctx: &FastContext,
    instruction: &str,
    action: ActKind,
    text: Option<&str>,
) -> Result<ActionOutcome> {
    // A key press has no on-screen target, so it skips `hint_act` and the
    // `find_and_*` ladder entirely: neither can press, and routing it through
    // them would spend three calls to fail at a job none of them can do.
    if let ActKind::Press = action {
        return press_key(registry, ctx, text.unwrap_or("Enter")).await;
    }
    let mut input = json!({ "instruction": instruction });
    match action {
        ActKind::Click => {}
        ActKind::Type => {
            input["action"] = json!("type");
            input["text"] = json!(text.unwrap_or_default());
        }
        // Unreachable: `Press` returned above. The payload is never built.
        ActKind::Press => {}
    }
    let out = call_fast(registry, ctx, HINT_ACT, input).await?;
    let mut outcome = action_outcome_from_hint_act(&out);
    if outcome.success {
        return Ok(outcome);
    }
    // Fallback ladder: the same instruction through the semantic resolver.
    let (fallback_tool, fallback_input) = match action {
        ActKind::Click => (FIND_AND_CLICK, json!({ "query": instruction })),
        ActKind::Type => (
            FIND_AND_TYPE,
            json!({ "query": instruction, "text": text.unwrap_or_default() }),
        ),
        // Also unreachable, and deliberately not paired with a fallback tool.
        ActKind::Press => (HINT_ACT, json!({ "instruction": instruction })),
    };
    match call_fast(registry, ctx, fallback_tool, fallback_input).await {
        Ok(v) => {
            let alt = action_outcome_from_hint_act(&v);
            if alt.success {
                return Ok(alt);
            }
            outcome.message = format!("{} | {}", outcome.message, alt.message);
        }
        Err(e) => {
            outcome.message = format!("{} | {e:#}", outcome.message);
        }
    }
    Ok(outcome)
}

/// The interaction verb for one fast-lane action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActKind {
    Click,
    Type,
    /// A single key, not a target. Needed because typing is not submitting:
    /// several of the most common goals (search YouTube, send a query) only
    /// act on Enter, and without this the loop has no verb for the step that
    /// makes the previous one mean anything.
    Press,
}

impl ActKind {
    pub fn parse(s: &str) -> Self {
        let s = s.trim();
        if s.eq_ignore_ascii_case("type") {
            Self::Type
        } else if matches!(
            s.to_ascii_lowercase().as_str(),
            "press" | "key" | "submit" | "enter"
        ) {
            // `submit` and `enter` are the two words a planner reaches for when
            // it means "commit what I just typed"; mapping them here is what
            // keeps that phrasing from falling through to a click on nothing.
            Self::Press
        } else {
            Self::Click
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Click => "click",
            Self::Type => "type",
            Self::Press => "press",
        }
    }
}

/// **FAST press.** One key into the focused element of the current tab.
pub async fn press_key(
    registry: &ToolRegistry,
    ctx: &FastContext,
    key: &str,
) -> Result<ActionOutcome> {
    // An empty key is the overwhelmingly common case (the planner wrote
    // "submit" with no key), and Enter is the key that submits.
    let key = match key.trim() {
        "" => "Enter",
        k => k,
    };
    let out = call_fast(registry, ctx, BROWSER_PRESS_KEY, json!({ "key": key })).await?;
    Ok(press_outcome(&out))
}

/// Normalizer for a key press, which cannot share
/// [`action_outcome_from_hint_act`]'s rule.
///
/// `browser_press_key` answers `{"pressed":"Enter","via":"Input"}` — no
/// `success` field, because the press either happened or the call errored. The
/// hint-act rule reads a missing `success` as failure, so reusing it here would
/// report a working Enter as a failed step. Success is therefore the absence of
/// an error marker, and only a marker makes it a failure.
pub fn press_outcome(out: &Value) -> ActionOutcome {
    let root = unwrap_envelope(out);
    // Both the unwrapped payload and the raw reply are consulted: a real
    // failure can arrive as an inner JSON object or as the envelope's plain
    // `error:` text, and only the raw value still has the latter.
    let failed = tool_reports_failure(Some(out))
        || root.get("isError") == Some(&Value::Bool(true))
        || root.get("is_error") == Some(&Value::Bool(true))
        || root.get("error").is_some_and(|e| !e.is_null())
        || root
            .get("message")
            .and_then(Value::as_str)
            .is_some_and(|m| m.trim_start().starts_with("error:"));
    let pressed = root
        .get("pressed")
        .or_else(|| root.get("key"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    ActionOutcome {
        success: !failed,
        // hyprfast reports which transport carried the key; keeping it means a
        // press that went to the window instead of the page is visible in the
        // log rather than inferred later.
        tier: root.get("via").and_then(Value::as_str).map(str::to_owned),
        label: pressed,
        message: if failed {
            failure_reason(out)
        } else {
            String::new()
        },
    }
}

/// Pure parser for `hint_act` / `find_and_*` output.
pub fn action_outcome_from_hint_act(out: &Value) -> ActionOutcome {
    let root = unwrap_envelope(out);
    let success = root
        .get("success")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        && !root
            .get("message")
            .and_then(Value::as_str)
            .is_some_and(|m| m.trim_start().starts_with("error:"));
    ActionOutcome {
        success,
        tier: root.get("tier").and_then(Value::as_str).map(str::to_owned),
        label: ["label", "name", "instruction"]
            .iter()
            .filter_map(|k| root.get(*k).and_then(Value::as_str))
            .find(|s| !s.trim().is_empty())
            .map(str::to_owned),
        // A failure must say why, so the reason comes from every place the
        // payload can carry one rather than from `message` alone.
        message: if success {
            root.get("message")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim()
                .to_owned()
        } else {
            failure_reason(out)
        },
    }
}

/// **FAST verify.** Ask Decider whether the objective's success check now holds.
/// `verify` with an `expected` string becomes `verify_action`.
pub async fn verify_step(
    registry: &ToolRegistry,
    ctx: &FastContext,
    query: &str,
    expected: Option<&str>,
) -> VerifyOutcome {
    let (tool, input) = match expected {
        Some(e) => (VERIFY_ACTION, json!({ "query": query, "expected": e })),
        None => (VERIFY, json!({ "query": query })),
    };
    match call_fast(registry, ctx, tool, input).await {
        Ok(v) => verify_outcome_from_payload(&v),
        Err(_) => VerifyOutcome::Uncertain,
    }
}

/// The destination a goal names outright: a URL the user actually typed.
///
/// This used to fall back to a ~40-entry name→URL table (`KNOWN_SITES`), so
/// "play despacito on youtube" resolved to youtube.com offline. The table is
/// gone, deliberately:
///
/// * It is a closed world. Every site the user names that is not in the table
///   silently got no destination, which is the exact failure mode AGENTS.md
///   exists to prevent.
/// * It duplicates a decision the planner already makes and makes better.
///   `PLAN_INSTRUCTIONS` requires the first navigation step to carry the *full
///   destination URL*, and the planner is a language model that knows thousands
///   of sites and none of them have to be listed here.
///
/// Only a URL the user wrote is treated as a fact worth acting on before the
/// planner is consulted — that is a determinism guarantee, not a site list.
pub fn site_url_for_goal(goal: &str) -> Option<String> {
    let goal = goal.trim().to_lowercase();
    let idx = goal.find("https://").or_else(|| goal.find("http://"))?;
    let tail = &goal[idx..];
    let end = tail
        .find(|c: char| c.is_whitespace() || c == '"' || c == '\'')
        .unwrap_or(tail.len());
    let url = tail[..end].trim_end_matches(['.', ',', ')']);
    if url.is_empty() {
        None
    } else {
        Some(url.to_owned())
    }
}

/// Host of the page the browser is on, read straight from the page.
pub async fn current_page_host(registry: &ToolRegistry, ctx: &FastContext) -> Option<String> {
    let value = call_fast(
        registry,
        ctx,
        BROWSER_EVALUATE,
        json!({ "expression": "location.hostname", "js": "location.hostname" }),
    )
    .await
    .ok()?;
    let root = unwrap_envelope(&value);
    let node = root.get("result").unwrap_or(&root);
    let text = match node {
        Value::String(s) => s.clone(),
        Value::Object(map) => map.get("value").and_then(Value::as_str)?.to_owned(),
        _ => return None,
    };
    let host = text.trim().to_ascii_lowercase();
    (!host.is_empty()).then_some(host)
}

/// Full URL of the page the browser is on, read straight from the page.
///
/// Unlike [`current_page_host`] this keeps the path and the original case: the
/// replanner's question is "which page am I on", and `/results?search_query=`
/// is what distinguishes a submitted search from a typed-but-unsubmitted one —
/// exactly the difference a run that stops before Enter cannot see from the
/// hostname.
pub async fn current_page_url(registry: &ToolRegistry, ctx: &FastContext) -> Option<String> {
    let value = call_fast(
        registry,
        ctx,
        BROWSER_EVALUATE,
        json!({ "expression": "location.href", "js": "location.href" }),
    )
    .await
    .ok()?;
    let root = unwrap_envelope(&value);
    let node = root.get("result").unwrap_or(&root);
    let text = match node {
        Value::String(s) => s.clone(),
        Value::Object(map) => map.get("value").and_then(Value::as_str)?.to_owned(),
        _ => return None,
    };
    // No lowercasing and no trimming of the address itself: a URL path is
    // case-sensitive, so `Wikipedia` and `wikipedia` can be different pages.
    let url = text.trim();
    (!url.is_empty()).then(|| url.to_owned())
}

/// Host of `url` (no scheme, no `www.`, no path), for comparing "already
/// there?" without string-matching whole addresses.
pub fn host_of_url(url: &str) -> String {
    let rest = url
        .trim()
        .split("://")
        .nth(1)
        .unwrap_or(url.trim())
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default()
        .rsplit('@')
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let rest = rest.strip_prefix("www.").unwrap_or(&rest);
    match rest.split_once(':') {
        Some((h, _)) => h.to_owned(),
        None => rest.to_owned(),
    }
}

/// FAST navigate: the one way a run changes page, in the same tab.
pub async fn navigate(registry: &ToolRegistry, ctx: &FastContext, url: &str) -> Result<()> {
    let url = url.trim();
    if url.is_empty() {
        bail!("navigate needs a url");
    }
    let url = if url.contains("://") {
        url.to_owned()
    } else {
        format!("https://{url}")
    };
    call_fast(registry, ctx, BROWSER_NAVIGATE, json!({ "url": url }))
        .await
        .map(|_| ())
}

/// Run a probe expression in the page and decide whether it holds.
///
/// The probe is a side-effect-free expression (see
/// [`crate::agent_loop::sanitize_probe`]) whose value is the page's own
/// account of itself — the same signal a developer would read off
/// devtools. `true` on any failure, because a probe that cannot run must
/// never be read as "the objective is done".
pub async fn probe_is_satisfied(registry: &ToolRegistry, ctx: &FastContext, probe: &str) -> bool {
    probe_verdict(registry, ctx, probe).await.unwrap_or(false)
}

/// The three states a probe can be in: it ran and is true, it ran and is false,
/// or it could not run at all.
///
/// "Could not run" has to stay distinct from "ran and said false". Collapsing
/// them makes a missing `browser_evaluate` tool look like a page whose state
/// disproves the goal, which is how a whole run with no JS tool available ends
/// up reported as a failure on evidence that was never collected.
pub async fn probe_verdict(
    registry: &ToolRegistry,
    ctx: &FastContext,
    probe: &str,
) -> Option<bool> {
    let expression =
        format!("(() => {{ try {{ return !!({probe}); }} catch (e) {{ return false; }} }})()");
    let value = call_fast(
        registry,
        ctx,
        BROWSER_EVALUATE,
        json!({ "expression": expression, "js": expression }),
    )
    .await
    .ok()?;
    // A transport-level failure is "could not run", not a false result.
    if value.get("error").is_some() && value.get("result").is_none() {
        return None;
    }
    Some(probe_result_is_true(&value))
}

/// Pure parser for a `browser_evaluate` reply carrying a probe's value.
///
/// CDP answers `{"result":{"type":"boolean","value":true}}`, sometimes
/// wrapped in the MCP content envelope, so both shapes are accepted. Anything
/// else — including an error, an object, or a string — is `false`.
pub fn probe_result_is_true(out: &Value) -> bool {
    let root = unwrap_envelope(out);
    let node = root.get("result").unwrap_or(&root);
    match node {
        Value::Bool(b) => *b,
        Value::Object(map) => map
            .get("value")
            .and_then(|v| match v {
                Value::Bool(b) => Some(*b),
                Value::String(s) => Some(s == "true"),
                _ => None,
            })
            .unwrap_or(false),
        Value::String(s) => s.trim() == "true",
        _ => false,
    }
}

/// Pure parser for `verify` / `verify_action` output.
pub fn verify_outcome_from_payload(out: &Value) -> VerifyOutcome {
    let root = unwrap_envelope(out);
    if root.get("success") == Some(&Value::Bool(false))
        || root
            .get("message")
            .and_then(Value::as_str)
            .is_some_and(|m| m.trim_start().starts_with("error:"))
    {
        return VerifyOutcome::NotSatisfied;
    }
    for key in ["result", "verdict", "status", "outcome", "answer"] {
        if let Some(v) = root.get(key) {
            return verdict_from_str(&v.to_string());
        }
    }
    if let Some(b) = root.get("satisfied").and_then(Value::as_bool) {
        return if b {
            VerifyOutcome::Satisfied
        } else {
            VerifyOutcome::NotSatisfied
        };
    }
    if let Some(b) = root.get("success").and_then(Value::as_bool) {
        return if b {
            VerifyOutcome::Satisfied
        } else {
            VerifyOutcome::NotSatisfied
        };
    }
    VerifyOutcome::Uncertain
}

fn verdict_from_str(v: &str) -> VerifyOutcome {
    let l = v.trim().trim_matches('"').to_ascii_lowercase();
    if l.contains("success") || l.contains("true") || l.contains("satisfied") {
        VerifyOutcome::Satisfied
    } else if l.contains("fail") || l.contains("false") || l.contains("not_satisfied") {
        VerifyOutcome::NotSatisfied
    } else {
        VerifyOutcome::Uncertain
    }
}

/// **FAST wait.** Poll until `query` holds or `timeout_ms` is spent.
///
/// The budget is split across escalating [`settle_slices`] rather than handed
/// to one `wait_until` call, so a page that settles early stops costing the
/// full budget. Returns `false` when the whole budget expires unproven, which
/// is the same answer the single blocking call gave — the caller cannot tell
/// the difference, and pays far less when the condition is already true.
pub async fn wait_for(
    registry: &ToolRegistry,
    ctx: &FastContext,
    query: &str,
    timeout_ms: u64,
) -> bool {
    for slice in settle_slices(timeout_ms) {
        if wait_once(registry, ctx, query, slice).await {
            return true;
        }
    }
    false
}

/// Split one wait budget into short polls that escalate, instead of one call
/// that blocks for the whole budget.
///
/// `wait_until` polls internally and returns only when the query holds or the
/// budget runs out, so a single call costs the full `timeout_ms` whenever the
/// condition is not met — and "not met" is the common case, because every
/// caller here waits on a page that may still be hydrating rather than on
/// something that is already true. Measured across a session, every `wait_until`
/// returned at its ceiling: 5.49s, 4.78s, 4.61s, 4.63s against a 4s budget.
///
/// Splitting the same budget into 250ms/750ms/1.5s/3s changes nothing about what
/// is proven — the total ceiling is identical, and the query is still checked
/// against the same page — but a page that settles in 200ms now costs 250ms
/// instead of 4s. It converts an unconditional sleep into a poll-then-act
/// schedule, so the common fast path stops paying for the slow path.
///
/// Returns nothing for a zero budget, so a caller cannot spin on empty slices.
fn settle_slices(total_ms: u64) -> Vec<u64> {
    const LADDER: [u64; 4] = [250, 750, 1_500, 3_000];
    let mut out = Vec::with_capacity(LADDER.len());
    let mut left = total_ms;
    for step in LADDER {
        if left == 0 {
            break;
        }
        let slice = step.min(left);
        out.push(slice);
        left -= slice;
    }
    // A budget larger than the ladder still gets polled rather than dropped, so
    // raising `wait_timeout_ms` in config lengthens the wait instead of
    // discarding the remainder.
    while left > 0 {
        let slice = left.min(LADDER[LADDER.len() - 1]);
        out.push(slice);
        left -= slice;
    }
    out
}

/// One `wait_until` call with its own budget. `true` when the query held.
async fn wait_once(
    registry: &ToolRegistry,
    ctx: &FastContext,
    query: &str,
    timeout_ms: u64,
) -> bool {
    let input = json!({ "query": query, "timeout_ms": timeout_ms });
    match call_fast(registry, ctx, WAIT_UNTIL, input).await {
        Ok(v) => {
            let root = unwrap_envelope(&v);
            if root.get("success") == Some(&Value::Bool(false)) {
                return false;
            }
            match verify_outcome_from_payload(&v) {
                VerifyOutcome::Satisfied => true,
                _ => {
                    let text = root.to_string().to_ascii_lowercase();
                    text.contains("\"matched\":true")
                        || text.contains("\"found\":true")
                        || text.contains("\"timed_out\":false")
                }
            }
        }
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::AtomicUsize;

    /// The destination is whatever URL the user wrote, and nothing else.
    ///
    /// The three "play despacito on youtube" / "watch it on yt" cases that used
    /// to be here are gone with the site table. Naming a site in prose is now
    /// not a destination at all: the planner writes that URL in its own first
    /// navigation step, so there is no list here that a new site has to be
    /// added to.
    /// The wait budget is a ceiling, and splitting it must not move the ceiling.
    ///
    /// This is the whole contract of the poll-then-act change: the loop still
    /// waits at most `wait_timeout_ms` for a page to settle, it just stops
    /// waiting the moment the page does. A schedule that summed to more than the
    /// budget would turn every slow page into a longer run than before.
    #[test]
    fn the_settle_schedule_never_exceeds_the_budget_it_splits() {
        for budget in [
            0u64,
            1,
            100,
            250,
            251,
            999,
            1_000,
            2_500,
            4_000,
            5_500,
            5_751,
            10_000,
            60_000,
            u64::MAX / 2,
        ] {
            let slices = settle_slices(budget);
            let total: u64 = slices.iter().sum();
            assert_eq!(total, budget, "budget {budget} split into {slices:?}");
            assert!(
                slices.iter().all(|s| *s > 0),
                "a zero slice would spin: {budget} -> {slices:?}"
            );
        }
    }

    /// A zero budget must produce no calls at all, not an empty slice.
    #[test]
    fn a_zero_budget_polls_nothing() {
        assert!(settle_slices(0).is_empty());
    }

    /// The first slice is the common case: a page that is already settled must
    /// not pay the whole budget, so the schedule has to start short.
    #[test]
    fn the_first_poll_is_short_so_a_settled_page_returns_at_once() {
        assert_eq!(settle_slices(4_000)[0], 250);
    }

    #[test]
    fn only_a_url_the_user_typed_is_a_destination() {
        assert_eq!(
            site_url_for_goal("open https://news.ycombinator.com and read the top post").as_deref(),
            Some("https://news.ycombinator.com")
        );
        assert_eq!(
            site_url_for_goal("go to https://en.wikipedia.org/wiki/Rust").as_deref(),
            Some("https://en.wikipedia.org/wiki/rust")
        );
        // A query string is part of the destination the user asked for.
        assert_eq!(
            site_url_for_goal("https://example.com/search?q=rust&page=2").as_deref(),
            Some("https://example.com/search?q=rust&page=2")
        );
        // Nothing typed, nothing to navigate to — including goals that name a
        // site in prose, which is the whole point of deleting the table.
        for goal in [
            "what is 2 + 2",
            "",
            "play a song on youtube",
            "search for rust on reddit",
            "find a cafe on google maps",
            "watch it on yt",
        ] {
            assert_eq!(site_url_for_goal(goal), None, "{goal} names no URL");
        }
    }

    /// Whatever used to guard the table's word-boundary matching now guards the
    /// URL scan, which is the only text scan left here.
    #[test]
    fn text_around_a_url_is_not_swallowed_into_it() {
        // Punctuation around a typed URL is not part of it.
        assert_eq!(
            site_url_for_goal("open https://example.com, then read it").as_deref(),
            Some("https://example.com")
        );
        assert_eq!(
            site_url_for_goal("(\"https://example.com/a\")").as_deref(),
            Some("https://example.com/a")
        );
        // A bare domain with no scheme is not a URL, so it is left to the
        // planner rather than guessed at here.
        assert_eq!(site_url_for_goal("open example.com"), None);
    }

    #[test]
    #[test]
    fn hosts_are_compared_without_scheme_www_or_port() {
        assert_eq!(host_of_url("https://www.youtube.com"), "youtube.com");
        assert_eq!(
            host_of_url("https://www.youtube.com/results?search_query=x"),
            "youtube.com"
        );
        assert_eq!(host_of_url("https://mail.google.com"), "mail.google.com");
        assert_eq!(host_of_url("https://youtube.com:443/watch"), "youtube.com");
        assert_eq!(host_of_url("http://127.0.0.1:11435/v1"), "127.0.0.1");
        assert_eq!(host_of_url(""), "");
    }

    #[test]
    fn fingerprint_changes_when_the_screen_changes() {
        let a = screen_state_from_hint_snapshot(&json!({
            "count": 2, "via": "decider",
            "hints": [{"label":"Play"}, {"label":"Queue"}]
        }));
        let b = screen_state_from_hint_snapshot(&json!({
            "count": 2, "via": "decider",
            "hints": [{"label":"Play"}, {"label":"Share"}]
        }));
        assert_eq!(a.fingerprint(), a.fingerprint(), "stable for one state");
        assert_ne!(
            a.fingerprint(),
            b.fingerprint(),
            "label change must move it"
        );
    }

    #[test]
    fn mcp_envelope_is_unwrapped_before_reading_hints() {
        // The registry returns hyprfast's `{"content":[{"text":"<json>"}]}`
        // envelope. Parsing it raw yields an empty screen and a fingerprint
        // that never moves, so the anti-thrash guard fires on a changed screen.
        let inner = json!({
            "count": 3, "via": "hint",
            "hints": [{"label":"A","name":"Search"},{"label":"B"},{"label":"C"}]
        });
        let enveloped = json!({
            "content": [{"text": inner.to_string(), "type": "text"}]
        });
        let s = screen_state_from_hint_snapshot(&enveloped);
        assert_eq!(s.count, 3, "count must survive the envelope");
        assert_eq!(s.via.as_deref(), Some("hint"));
        assert_eq!(s.labels, vec!["A", "B", "C"]);
        assert_eq!(
            s.fingerprint(),
            screen_state_from_hint_snapshot(&inner).fingerprint(),
            "enveloped and bare payloads describe the same screen"
        );
    }

    #[test]
    fn fingerprint_ignores_hint_ordering_only_via_counts() {
        let one = screen_state_from_hint_snapshot(&json!({"count": 1, "hints": [{"label":"A"}]}));
        let two = screen_state_from_hint_snapshot(&json!({"count": 1, "hints": [{"label":"A"}]}));
        assert_eq!(one.fingerprint(), two.fingerprint());
        let three = screen_state_from_hint_snapshot(&json!({"count": 3, "hints": [{"label":"A"}]}));
        assert_ne!(
            one.fingerprint(),
            three.fingerprint(),
            "count is part of state"
        );
    }

    #[test]
    fn labels_prefer_label_then_name_then_text() {
        let s = screen_state_from_hint_snapshot(&json!({"hints": [
            {"label": "Search", "text": "ignored"},
            {"name": "Submit"},
            {"text": "Queue up"},
            {"tag": "div"}
        ]}));
        assert_eq!(
            s.labels,
            vec!["Search", "Submit", "Queue up", "(unlabelled)"]
        );
        assert_eq!(s.count, 4);
        assert!(s.summary().contains("Search"));
    }

    #[test]
    fn empty_snapshot_is_a_fingerprintable_state_too() {
        let s = screen_state_from_hint_snapshot(&json!({"count": 0, "via": "dom"}));
        assert!(s.labels.is_empty());
        assert_eq!(s.fingerprint(), s.fingerprint());
        assert_ne!(s.fingerprint(), ScreenState::default().fingerprint());
        // Nothing to click, but the snapshot still reported how it looked.
        assert!(s.summary().contains("dom"), "{}", s.summary());
    }

    #[test]
    fn find_parses_a_candidate_with_confidence() {
        let t = resolved_target_from_find(&json!({
            "candidate": {"label": "Play button", "selector": "#play",
                          "rect": {"x": 10, "y": 20, "width": 30, "height": 40},
                          "confidence": 0.91}
        }))
        .unwrap();
        assert_eq!(t.label, "Play button");
        assert_eq!(t.selector.as_deref(), Some("#play"));
        assert_eq!(t.rect, Some((10.0, 20.0, 30.0, 40.0)));
        assert_eq!(t.confidence, 0.91);
        assert!(t.is_confident());
    }

    #[test]
    fn find_parses_through_the_mcp_content_envelope() {
        let inner = json!({"target": {"name": "Next page", "confidence": "0.77"}}).to_string();
        let t = resolved_target_from_find(&json!({"content": [{"text": inner}]})).unwrap();
        assert_eq!(t.label, "Next page");
        assert!((t.confidence - 0.77).abs() < 1e-9);
        assert!(t.is_confident());
    }

    #[test]
    fn a_low_confidence_target_is_reported_not_hidden() {
        let t = resolved_target_from_find(&json!({"candidate": {"label": "x", "confidence": 0.2}}))
            .unwrap();
        assert!(
            !t.is_confident(),
            "the loop needs to see it and choose to replan"
        );
    }

    #[test]
    fn find_with_nothing_resolvable_yields_none() {
        assert!(resolved_target_from_find(&json!({})).is_none());
        assert!(resolved_target_from_find(&json!({"candidates": []})).is_none());
    }

    #[test]
    fn hint_act_success_and_failure_are_distinguished() {
        let ok = action_outcome_from_hint_act(&json!({
            "success": true, "tier": "decider", "label": "Play", "message": "clicked"
        }));
        assert!(ok.success);
        assert_eq!(ok.summary(), "ok via decider: Play");
        let bad = action_outcome_from_hint_act(&json!({
            "success": false, "tier": "hint", "message": "No action found"
        }));
        assert!(!bad.success);
        assert!(bad.summary().contains("failed"));
    }

    #[test]
    fn an_error_message_never_reads_as_success() {
        let out = action_outcome_from_hint_act(&json!({
            "success": true, "message": "error: CDP unreachable"
        }));
        assert!(!out.success);
    }

    #[test]
    fn a_probe_value_is_read_from_every_reply_shape() {
        // Raw CDP result.
        assert!(probe_result_is_true(&json!({
            "result": {"type": "boolean", "value": true}
        })));
        assert!(!probe_result_is_true(&json!({
            "result": {"type": "boolean", "value": false}
        })));
        // Bare boolean, and the MCP content envelope around it.
        assert!(probe_result_is_true(&json!(true)));
        assert!(!probe_result_is_true(&json!(false)));
        assert!(probe_result_is_true(
            &json!({"content": [{"type": "text", "text": "{\"result\":{\"value\":true}}"}]})
        ));
        // A stringified boolean still reads, everything else does not.
        assert!(probe_result_is_true(&json!({"value": "true"})));
        assert!(!probe_result_is_true(&json!({"value": "false"})));
        assert!(!probe_result_is_true(&json!({"value": 1})));
        // An evaluate error must never read as "done".
        assert!(!probe_result_is_true(
            &json!({"content": [{"type": "text", "text": "error: boom"}]})
        ));
        assert!(!probe_result_is_true(&json!({"exceptionDetails": {}})));
    }

    #[test]
    fn verify_maps_the_three_states() {
        assert_eq!(
            verify_outcome_from_payload(&json!({"result": "success", "confidence": 0.9})),
            VerifyOutcome::Satisfied
        );
        assert_eq!(
            verify_outcome_from_payload(&json!({"verdict": "failure"})),
            VerifyOutcome::NotSatisfied
        );
        assert_eq!(
            verify_outcome_from_payload(&json!({"satisfied": false})),
            VerifyOutcome::NotSatisfied
        );
        assert_eq!(
            verify_outcome_from_payload(&json!({})),
            VerifyOutcome::Uncertain
        );
        assert_eq!(
            verify_outcome_from_payload(&json!({"message": "error: no browser"})),
            VerifyOutcome::NotSatisfied
        );
    }

    #[test]
    fn tool_failure_payloads_are_not_silent_successes() {
        assert!(tool_reports_failure(Some(&json!({"success": false}))));
        assert!(tool_reports_failure(Some(
            &json!({"content": [{"text": "error: x"}]})
        )));
        assert!(!tool_reports_failure(Some(
            &json!({"success": true, "count": 1})
        )));
        assert!(
            tool_reports_failure(None),
            "a call that returned nothing is not a success"
        );
    }

    #[test]
    fn act_kind_defaults_to_click() {
        assert_eq!(ActKind::parse("click"), ActKind::Click);
        assert_eq!(ActKind::parse("TYPE"), ActKind::Type);
        assert_eq!(ActKind::parse("nonsense"), ActKind::Click);
    }

    #[test]
    fn a_planner_can_ask_for_a_key_under_any_of_its_names() {
        for s in [
            "press", "key", "submit", "enter", "PRESS", " Enter ", "Submit",
        ] {
            assert_eq!(ActKind::parse(s), ActKind::Press, "{s} means press a key");
        }
        // Trimming and case are the only normalization; nothing else may widen
        // into Press, or an ordinary click verb would start eating keystrokes.
        assert_eq!(ActKind::parse("press enter now"), ActKind::Click);
        assert_eq!(ActKind::parse(""), ActKind::Click);
        assert_eq!(ActKind::as_str(&ActKind::Press), "press");
        assert_eq!(ActKind::as_str(&ActKind::Click), "click");
        assert_eq!(ActKind::as_str(&ActKind::Type), "type");
    }

    #[test]
    fn a_press_with_no_success_field_is_a_success() {
        // The real reply from hyprfast. `action_outcome_from_hint_act` would
        // call this a failure because it has no `success` field, which is
        // exactly why press gets its own normalizer.
        let out = press_outcome(&json!({"pressed": "Enter", "via": "Input"}));
        assert!(out.success);
        assert_eq!(out.label.as_deref(), Some("Enter"));
        assert_eq!(out.tier.as_deref(), Some("Input"));
        assert!(out.message.is_empty());
        // Even through the MCP envelope.
        let wrapped = json!({"content": [{"text": "{\"pressed\":\"Enter\",\"via\":\"Input\"}"}]});
        assert!(press_outcome(&wrapped).success);
    }

    #[test]
    fn a_press_that_reports_an_error_is_a_failure_with_a_reason() {
        let out = press_outcome(&json!({"error": "no focused element"}));
        assert!(!out.success);
        assert!(!out.message.trim().is_empty());
        assert!(!press_outcome(&json!({"success": false})).success);
        assert!(!press_outcome(&json!({"isError": true})).success);
        assert!(!press_outcome(&json!({"content": [{"text": "error: no tab"}]})).success);
        // A key named "error" is not an error report.
        assert!(press_outcome(&json!({"pressed": "Escape"})).success);
    }

    // ---- browser bootstrap -------------------------------------------------

    /// The default topic list, i.e. exactly what a user gets before editing
    /// anything. The point of these cases is that the *mechanism* is config:
    /// the first two goals are caught by structural signals that exist in every
    /// build, and the last three are caught only because those words are in
    /// `harness.browser_goal_keywords` — which the user can change.
    fn default_topics() -> Vec<String> {
        lucy_config::LucyConfig::default()
            .harness
            .browser_goal_keywords
    }

    #[test]
    fn structural_signals_need_no_configuration() {
        // Nothing here appears in the topic list, so these hold in any config.
        for goal in [
            "open https://example.com/pricing",
            "go to news.ycombinator.com and read the top post",
            "browse the docs site",
            "search for the linux kernel release notes",
            "look it up on the web",
        ] {
            assert!(
                goal_needs_browser(goal, &[]),
                "{goal} is web work on structure alone"
            );
        }
    }

    #[test]
    fn topical_words_come_from_configuration_not_code() {
        // With an empty list the topics are invisible to the code...
        for goal in ["book a flight to lisbon", "order a pizza", "check my bank"] {
            assert!(
                !goal_needs_browser(goal, &[]),
                "{goal} must not be web work with no topics configured"
            );
        }
        // ...and adding one is enough, with no code change.
        let topics = vec!["flight".to_owned(), "pizza".to_owned()];
        for goal in ["book a flight to lisbon", "order a pizza"] {
            assert!(goal_needs_browser(goal, &topics), "{goal} is web work");
        }
        assert!(!goal_needs_browser("check my bank", &topics));
        // The shipped default covers the same ground the old hardcoded list
        // did, so removing the hardcoding changed no behaviour.
        let defaults = default_topics();
        for goal in [
            "play a song on youtube",
            "search the reddit thread",
            "book a flight to lisbon",
            "buy the cheapest gpu on amazon",
            "check out tonight's news article",
            "add it to the cart and check out",
        ] {
            assert!(goal_needs_browser(goal, &defaults), "{goal} is web work");
        }
    }

    #[test]
    fn desktop_work_never_launches_a_browser() {
        for goal in [
            "arrange the three windows side by side",
            "move focus to the terminal",
            "take a screenshot of the desktop",
            "mute the volume",
        ] {
            assert!(
                !goal_needs_browser(goal, &default_topics()),
                "{goal} must not launch a browser"
            );
        }
    }

    #[test]
    fn an_empty_topic_entry_is_ignored_rather_than_matching_everything() {
        // A stray blank line in a hand-edited config must not turn every goal
        // into web work.
        let topics = vec![String::new(), "   ".to_owned()];
        assert!(!goal_needs_browser("arrange the windows", &topics));
        assert!(goal_needs_browser("open https://example.com", &topics));
    }

    fn browser_up() -> CdpProbe {
        probe_fn(|| async { true })
    }

    fn browser_down() -> CdpProbe {
        probe_fn(|| async { false })
    }

    #[tokio::test]
    async fn a_live_cdp_endpoint_costs_one_probe_and_no_launch() {
        // The empty catalog would fail any launch attempt, so reaching
        // `AlreadyRunning` proves no `browser_open` was spent.
        let registry = ToolRegistry::new();
        let ctx = FastContext::new(
            lucy_core::SessionId::default(),
            None,
            InterruptSignal::new(),
        );
        let outcome = ensure_browser(
            &registry,
            &ctx,
            "play despacito song on youtube",
            BROWSER_START_URL,
            &default_topics(),
            Some(browser_up()),
        )
        .await;
        assert_eq!(outcome, BrowserBootstrap::AlreadyRunning);
        assert_eq!(ctx.stats.calls(), 0, "a live browser is not relaunched");
    }

    #[tokio::test]
    async fn a_launch_that_does_not_work_is_never_retried() {
        let registry = ToolRegistry::new();
        let ctx = FastContext::new(
            lucy_core::SessionId::default(),
            None,
            InterruptSignal::new(),
        );
        let probe = Some(browser_down());
        let first = ensure_browser(
            &registry,
            &ctx,
            "play despacito song on youtube",
            BROWSER_START_URL,
            &default_topics(),
            probe.clone(),
        )
        .await;
        assert!(first.is_failure(), "{first:?}");
        // One attempted call, and the reason is hyprfast's, not a blank.
        assert_eq!(ctx.stats.calls(), 1, "{first:?}");
        let second = ensure_browser(
            &registry,
            &ctx,
            "play despacito song on youtube",
            BROWSER_START_URL,
            &default_topics(),
            probe,
        )
        .await;
        assert_eq!(first, second, "the single attempt is the run's answer");
        assert_eq!(ctx.stats.calls(), 1, "no second launch attempt");
    }

    #[tokio::test]
    async fn a_desktop_goal_never_touches_cdp() {
        // The probe is a counter here: a desktop goal must not even ask.
        let asked = Arc::new(AtomicUsize::new(0));
        let registry = ToolRegistry::new();
        let ctx = FastContext::new(
            lucy_core::SessionId::default(),
            None,
            InterruptSignal::new(),
        );
        let outcome = ensure_browser(
            &registry,
            &ctx,
            "arrange the three windows side by side",
            BROWSER_START_URL,
            &default_topics(),
            Some(probe_fn({
                let asked = asked.clone();
                move || {
                    let asked = asked.clone();
                    async move {
                        asked.fetch_add(1, Ordering::SeqCst);
                        false
                    }
                }
            })),
        )
        .await;
        assert_eq!(outcome, BrowserBootstrap::NotNeeded);
        assert_eq!(asked.load(Ordering::SeqCst), 0, "{outcome:?}");
    }

    // ---- Gap 2: names, not ordinals ---------------------------------------

    #[test]
    fn the_summary_names_the_element_and_keeps_the_label() {
        let s = screen_state_from_hint_snapshot(&json!({
            "count": 2, "via": "decider",
            "hints": [
                {"label": "S", "name": "despacito"},
                {"label": "Y", "name": "Luis Fonsi - Despacito ft. Daddy Yankee"}
            ]
        }));
        // The label is still there: it is the key the hint pipeline clicks with.
        assert_eq!(s.labels, vec!["S", "Y"]);
        // What a planner reads is the element's identity, label as suffix.
        assert_eq!(
            s.described,
            vec![
                "despacito (S)",
                "Luis Fonsi - Despacito ft. Daddy Yankee (Y)"
            ]
        );
        let summary = s.summary();
        assert!(summary.contains("despacito (S)"), "{summary}");
        assert!(
            !summary.contains("2 element(s): S, Y"),
            "bare ordinals are what made the old line useless: {summary}"
        );
    }

    #[test]
    fn a_hint_with_no_name_falls_back_to_its_label() {
        let s = screen_state_from_hint_snapshot(&json!({"hints": [
            {"label": "Play"},
            {"text": "Queue up"},
            {"tag": "div"}
        ]}));
        assert_eq!(s.labels, vec!["Play", "Queue up", "(unlabelled)"]);
        assert_eq!(s.described, vec!["Play", "Queue up", "(unlabelled)"]);
    }

    #[test]
    fn a_changed_element_name_moves_the_fingerprint() {
        let a = screen_state_from_hint_snapshot(&json!({
            "hints": [{"label": "A", "name": "Search"}]
        }));
        let b = screen_state_from_hint_snapshot(&json!({
            "hints": [{"label": "A", "name": "Search despacito"}]
        }));
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    // ---- Gap 3: a failure must say why ------------------------------------

    #[test]
    fn a_failed_hint_act_reports_the_underlying_reason() {
        // The real shape: `hint-act` wrote its error to stderr, exited 0, and
        // the MCP envelope carried it as content text only.
        let out = action_outcome_from_hint_act(&json!({
            "content": [{"type": "text", "text": "error: CDP connection failed: CDP unreachable"}],
            "isError": false
        }));
        assert!(!out.success);
        assert_eq!(out.message, "error: CDP connection failed: CDP unreachable");
        assert!(
            out.summary()
                .starts_with("failed: error: CDP connection failed"),
            "{}",
            out.summary()
        );
    }

    #[test]
    fn an_empty_tool_result_still_produces_a_reason() {
        // `hint-act` with the Decider disabled: exits 0, no stdout at all. A
        // blank prefix is the bug this replaces.
        let out = action_outcome_from_hint_act(&json!({"content": [{"text": ""}]}));
        assert!(!out.success);
        assert!(!out.message.is_empty(), "a failure may never be blank");
        assert!(out.summary().starts_with("failed:"), "{}", out.summary());
    }

    #[test]
    fn failure_reason_reads_every_place_a_payload_can_carry_one() {
        assert_eq!(
            failure_reason(&json!({"success": false, "error": "decider disabled"})),
            "decider disabled"
        );
        assert_eq!(
            failure_reason(&json!({"content": [{"text": "CDP unreachable"}]})),
            "CDP unreachable"
        );
        assert_eq!(
            failure_reason(&json!({"isError": true, "content": []})),
            "tool reported an error with no message"
        );
        // A nested JSON-RPC style error object, not just a string.
        assert!(
            failure_reason(&json!({"error": {"code": -32000, "message": "browser gone"}}))
                .contains("browser gone"),
            "{}",
            failure_reason(&json!({"error": {"code": -32000, "message": "browser gone"}}))
        );
        assert!(!failure_reason(&json!({})).is_empty());
    }

    #[test]
    fn a_failure_reason_never_quotes_the_payload() {
        // The regression: with nothing readable in the result, the old fallback
        // formatted the whole value into the sentence, so a raw envelope reached
        // the chat inside a status line.
        let reason = failure_reason(&json!({"content": [{"type": "text"}], "code": -32000}));
        assert!(!reason.contains('{'), "{reason}");
        assert!(!reason.contains("content"), "{reason}");
        assert!(reason.contains("no reason reported"), "{reason}");
    }

    #[test]
    fn a_very_long_failure_reason_is_capped() {
        let reason = failure_reason(&json!({"error": "e".repeat(4000)}));
        assert!(reason.chars().count() <= MAX_FAILURE_REASON_CHARS, "{}", reason.len());
    }

    #[test]
    fn a_tiered_failure_still_carries_its_message() {
        // A tier on its own used to swallow the reason entirely.
        let out = action_outcome_from_hint_act(&json!({
            "success": false, "tier": "hint", "message": "decider disabled"
        }));
        assert_eq!(out.summary(), "failed via hint: decider disabled");
        let ok = action_outcome_from_hint_act(&json!({
            "success": true, "tier": "hint", "label": "Play", "message": "clicked"
        }));
        assert_eq!(ok.summary(), "ok via hint: Play", "success stays terse");
    }

    // ---- Re-anchoring a vague instruction to a name on screen -------------

    /// The `described` list measured live on the YouTube results page for
    /// "despacito", verbatim: element name, hint label. This is the screen the
    /// loop got stuck on, and two of these labels are what the resolver actually
    /// clicked when handed the description instead of a name — `Q = "All"` on
    /// one run, `F = "Clear search query"` on the next, both wrong, on a page
    /// whose right answer was `Z`.
    const MEASURED_DESPACITO_RESULTS: &[&str] = &[
        "Guide (A)",
        "a (S)",
        "despacito (D)",
        "Clear search query (F)",
        "Search (G)",
        "Search with your voice (H)",
        "a (J)",
        "Settings (K)",
        "Sign in (L)",
        "All (Q)",
        "Shorts (W)",
        "Unwatched (E)",
        "Watched (R)",
        "Videos (T)",
        "Recently uploaded (Y)",
        "Live (U)",
        "Next (I)",
        "Search filters (O)",
        "4:42 Now playing (P)",
        "Luis Fonsi - Despacito ft. Daddy Yankee 4 minutes, 42 seconds (Z)",
        "Action menu (X)",
        "Go to channel LuisFonsiVEVO (C)",
        "Luis Fonsi (V)",
        "Mix (B)",
        "Mix - Luis Fonsi - Despacito ft. Daddy Yankee (N)",
        "Luis Fonsi - Despacito ft. Daddy Yankee · 4:42 (M)",
        "Luis Fonsi - No Me Doy Por Vencido · 3:54 (AA)",
        "Justin Bieber - Despacito (Lyrics / Letra) ft. Luis Fonsi & Daddy Yankee 3 minutes, 51 seconds (AD)",
    ];

    /// The planner's own wording for this objective, before it had seen the page.
    const VAGUE_DESPACITO_INSTRUCTION: &str =
        "click the video result whose title contains Despacito";

    /// The instruction that worked when it was measured, quoting the name.
    const NAMED_DESPACITO_INSTRUCTION: &str =
        "click the link named 'Luis Fonsi - Despacito ft. Daddy Yankee 4 minutes, 42 seconds'";

    fn measured_results() -> Vec<String> {
        MEASURED_DESPACITO_RESULTS
            .iter()
            .map(|s| (*s).to_owned())
            .collect()
    }

    #[test]
    fn a_vague_instruction_anchors_to_the_name_on_screen_not_to_a_filter_tab() {
        // The measured failure, whole. The planner described the target instead
        // of naming it, and the resolver answered "All" (Q) the first time and
        // "Clear search query" (F) the second.
        let described = measured_results();
        let anchored = anchor_to_visible_name(VAGUE_DESPACITO_INSTRUCTION, &described)
            .expect("Despacito is on the page, so this is anchorable");
        for wrong in [
            "All",
            "Videos",
            "Shorts",
            "Live",
            "Search",
            "Clear search query",
        ] {
            assert_ne!(
                anchored, wrong,
                "a filter tab is never the target: {anchored:?}"
            );
        }
        assert_eq!(
            anchored, "Luis Fonsi - Despacito ft. Daddy Yankee 4 minutes, 42 seconds",
            "the primary result, first on screen among the names that match, label stripped"
        );
    }

    #[test]
    fn a_vague_instruction_never_anchors_the_search_box_holding_the_query() {
        // `despacito (D)` is the search input. It scores the same word as the
        // result does and sits higher on the page, so without the "the
        // instruction already said this" rule it is the answer — and clicking it
        // re-focuses the search box instead of playing anything.
        let described = measured_results();
        let anchored =
            anchor_to_visible_name(VAGUE_DESPACITO_INSTRUCTION, &described).expect("anchorable");
        assert_ne!(anchored, "despacito", "{anchored:?}");
    }

    #[test]
    fn an_instruction_with_no_specific_word_anchors_to_nothing() {
        // A generic word cannot anchor on its own: with nothing but `button` and
        // `page` to go on, every element on the page ties and the safe answer is
        // no answer.
        let described = measured_results();
        for vague in [
            "click the button on the page",
            "click the link",
            "click the first video result",
            "click the search box",
            "type into the field",
            "",
        ] {
            assert_eq!(
                anchor_to_visible_name(vague, &described),
                None,
                "{vague:?} names nothing that distinguishes one element"
            );
        }
    }

    #[test]
    fn anchoring_a_quoted_name_still_resolves_and_an_empty_screen_is_not_an_error() {
        let described = measured_results();
        // The positive control: the instruction that worked when it was measured
        // must keep resolving to the same element.
        assert_eq!(
            anchor_to_visible_name(NAMED_DESPACITO_INSTRUCTION, &described).as_deref(),
            Some("Luis Fonsi - Despacito ft. Daddy Yankee 4 minutes, 42 seconds")
        );
        assert_eq!(
            anchor_to_visible_name("click the Despacito result", &[]),
            None,
            "an empty screen is not a failure, it is no answer"
        );
        assert_eq!(
            anchor_to_visible_name("click the Despacito result", &["a".to_owned()]),
            None,
            "a name below the minimum length is not a target"
        );
    }

    #[test]
    fn anchoring_is_deterministic_and_its_tiebreak_is_pinned() {
        let described = measured_results();
        let first = anchor_to_visible_name(VAGUE_DESPACITO_INSTRUCTION, &described);
        // Same input, same answer, every time — this is what makes it usable as
        // a retry, where a coin-flip answer is a coin-flip click.
        for _ in 0..5 {
            assert_eq!(
                anchor_to_visible_name(VAGUE_DESPACITO_INSTRUCTION, &described),
                first
            );
        }
        // The three Luis Fonsi / Justin rows all carry the same word, so the tie
        // is what decides the click. It is pinned to first-on-screen, which lands
        // on the primary result.
        let first = first.expect("anchorable");
        assert!(
            first.starts_with("Luis Fonsi - Despacito"),
            "the tie must land on a Luis Fonsi result: {first:?}"
        );
        // And it really is position doing the work, not the entry itself: the
        // same page in the other order picks the other first.
        let mut reversed = measured_results();
        reversed.reverse();
        assert_eq!(
            anchor_to_visible_name(VAGUE_DESPACITO_INSTRUCTION, &reversed).as_deref(),
            Some(
                "Justin Bieber - Despacito (Lyrics / Letra) ft. Luis Fonsi & Daddy Yankee 3 minutes, 51 seconds"
            ),
            "first-on-screen is the rule, so a reversed page picks the reversed first"
        );
    }

    #[test]
    fn a_generic_word_alone_is_never_enough_to_anchor() {
        // `title` is the one that matters: the measured instruction says "whose
        // title contains Despacito", and every result row on the page is a
        // title. A name matched only by the generic word is a guess, not an
        // anchor, however well it scores.
        let titles_only = vec![
            "Video title (Q)".to_owned(),
            "Title (W)".to_owned(),
            "Playlist thumbnail (E)".to_owned(),
        ];
        assert_eq!(
            anchor_to_visible_name("click the item whose title matches", &titles_only),
            None,
            "nothing here is identified by the generic word alone"
        );
        // One specific word on the page is enough, even beside a generic match.
        let with_a_match = vec!["Video title (Q)".to_owned(), "despacito mix (E)".to_owned()];
        assert_eq!(
            anchor_to_visible_name("click the despacito item", &with_a_match).as_deref(),
            Some("despacito mix"),
            "a name that merely CONTAINS the query is a target; one that IS it is the search box"
        );
    }

    #[test]
    fn a_name_ending_in_a_parenthetical_is_not_mistaken_for_a_hint_label() {
        // `(AD)` is a hint label and comes off. `(Lyrics / Letra)` is part of
        // the title and must stay, or the anchored instruction names a video
        // that does not exist.
        assert_eq!(
            strip_hint_label("Justin Bieber - Despacito (Lyrics / Letra) (AD)"),
            "Justin Bieber - Despacito (Lyrics / Letra)"
        );
        assert_eq!(
            strip_hint_label("Despacito (Official Video)"),
            "Despacito (Official Video)",
            "a long parenthetical is part of the name, not a label"
        );
        assert_eq!(strip_hint_label("Search (G)"), "Search");
        assert_eq!(strip_hint_label("Plain name"), "Plain name");
    }
}
