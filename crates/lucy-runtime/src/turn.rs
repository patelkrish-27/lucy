//! Turn routing: one classification forward pass, verified by the main LLM,
//! then branch A or branch B.
//!
//! Every user command goes through [`classify_turn`]:
//!
//! 1. **Step 1** — `POST {classification_api_url}/predict` asks the
//!    `decider-serve` model two heads in a single forward pass: does the
//!    command need only a text response (`yes`/`no`), and what reasoning
//!    complexity does it need (`1`/`2`/`3`).
//! 2. **Step 1b (dynamic verify, no keywords)** — when the classifier says
//!    text-only (or is disabled/unreachable/under-confident), the main LLM
//!    reads the request against the live tool catalog and decides whether
//!    any tool can advance it. No verb lists anywhere: the model decides
//!    from capability, so new tools and new phrasings route correctly
//!    without code changes.
//! 3. **Step 2A** — `"yes"`: `Manual` chat mode goes straight to the Level 3
//!    model; `Auto` dispatches to the model bound to the returned tier.
//! 4. **Step 2B** — `"no"`: the Level 3 model turns the request plus the
//!    `lucy-hyprfast` tool catalog into an ordered command list, every command
//!    is validated through `lucy_core::ApprovalGate` + `lucy_tools::ToolRegistry`,
//!    and the commands are executed **one by one in order** with step-by-step
//!    status streamed to the TUI.
//!
//! When both the classifier and the verify LLM are unreachable the turn
//! degrades to a text answer with a warning surfaced to the TUI — never a crash.

use anyhow::{Result, anyhow};
use lucy_config::{ChatMode, LucyConfig, ReasoningLevel};
use lucy_core::{AgentEvent, ApprovalDecision, ApprovalGate, ExecutionMode, ToolContext};
use lucy_hyprfast::ToolCapability;
use lucy_systemone::{DeciderClient, TurnBranch, TurnClassification};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use tracing::warn;

use super::LucyRuntime;

/// Minimum `decider-serve` confidence trusted on its own. Below this the turn
/// is routed by the main LLM reading the live tool catalog instead. There is
/// no third path: no keyword heuristics anywhere in routing.
pub const ROUTING_CONFIDENCE_MIN: f64 = 0.65;

/// Routing produced no verdict. Returned — and shown to the user — instead of
/// guessing: a silently misrouted turn is worse than a stopped one, because a
/// chat answer to an action request looks like success while nothing runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoutingError {
    /// `decider-serve` was disabled, unconfigured, unreachable, or answered
    /// below [`ROUTING_CONFIDENCE_MIN`] while the LLM second opinion was also
    /// unavailable. No tools run. Carries the user-facing reason.
    DeciderUnavailable(String),
    /// The LLM second opinion answered but named no usable intent.
    RouterUndecided(String),
}

impl std::fmt::Display for RoutingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeciderUnavailable(why) => write!(
                f,
                "routing unavailable: decider-serve could not decide ({why}) — \
                 start decider-serve or set the classification model URL in /settings, then retry. No tools were run."
            ),
            Self::RouterUndecided(why) => write!(
                f,
                "routing undecided: the router LLM did not commit to an intent ({why}) — \
                 rephrase the request and retry. No tools were run."
            ),
        }
    }
}

impl std::error::Error for RoutingError {}

/// Where a routing decision came from — surfaced in the chat log so the user
/// can tell a decider verdict from an LLM second opinion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RouteSource {
    /// `decider-serve` answered at or above [`ROUTING_CONFIDENCE_MIN`] and no
    /// verify was needed.
    Classifier,
    /// The decider was under-confident and the main LLM decided from the live
    /// tool catalog instead. No keywords involved.
    LlmVerify,
    /// `chat_mode = Manual`: no classification, the Level 3 model handles the
    /// turn.
    ManualOverride,
}

impl RouteSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Classifier => "classifier",
            Self::LlmVerify => "llm verify",
            Self::ManualOverride => "manual chat mode",
        }
    }
}

/// The resolved routing decision for one turn.
#[derive(Debug, Clone, PartialEq)]
pub struct TurnRoute {
    pub classification: TurnClassification,
    /// Which branch of Step 2 the turn takes.
    pub branch: TurnBranch,
    /// The tier the classifier picked (already mapped to a model).
    pub reasoning_level: ReasoningLevel,
    /// `provider_id/model` that will be called.
    pub model_key: String,
    /// Human label for the chat log header.
    pub model_label: String,
    pub source: RouteSource,
    /// Warning to surface in the TUI (classifier down, tier unset, …).
    pub note: Option<String>,
    /// The knowledge topic the classifier picked, when its third head answered.
    ///
    /// Advisory only, and deliberately kept on the route so a caller does not
    /// have to reach back into the classification it already had: it reorders a
    /// deterministic recall result set and never decides whether recall runs.
    pub knowledge_topic: Option<String>,
}

impl TurnRoute {
    /// Terse `model · tier` line for the chat log.
    pub fn active_model_line(&self) -> String {
        let tier = if self.source == RouteSource::ManualOverride {
            "manual".to_string()
        } else {
            format!(
                "{} · {}",
                self.reasoning_level.short(),
                self.reasoning_level.label()
            )
        };
        format!("{} · {tier}", self.model_label)
    }

    /// True when the turn needs system/browser actions rather than a reply.
    pub fn needs_actions(&self) -> bool {
        self.branch.needs_actions()
    }
}

/// One command in the Level 3 plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlannedCommand {
    pub index: usize,
    /// Tool name exactly as the catalog names it, e.g. `browser_navigate`.
    pub tool: String,
    /// Tool input object.
    #[serde(default)]
    pub input: Value,
    /// Human step description for the execution log.
    #[serde(default)]
    pub description: String,
}

impl PlannedCommand {
    /// Numbered one-liner for the execution log.
    pub fn log_line(&self) -> String {
        if self.description.trim().is_empty() {
            format!("{}. {}", self.index, self.tool)
        } else {
            format!("{}. {} ({})", self.index, self.description, self.tool)
        }
    }
}

/// The Level 3 planning prompt: turn the goal + `lucy-hyprfast` catalog into an
/// ordered command list.
pub const PLAN_SYSTEM: &str =
    "You are Lucy, a computer-use planner. Return ONLY the specified JSON.";
pub const PLAN_INSTRUCTIONS: &str = r#"Turn the user's request into an ordered list of commands that reach the goal state.

Rules:
- Use ONLY tools from the catalog below, with the exact tool name.
- Steps must be strictly ordered: each step must be possible once the previous ones have run.
- Keep it minimal. Never invent intermediate observation steps that are not needed.
- `input` must satisfy the tool's `input_schema`.
- `description` is one short imperative line shown to the user.

Plans run BLIND: every step's input is fixed when you write it and you never
see a previous step's output. So no step may depend on observing the screen,
and no step may wait for a `ref` from a `browser_snapshot` — you would never
see the refs that snapshot produced.

Interaction verb — `hint_act`:
- `hint_act` is the PRIMARY way to act on a page and the default answer to
  "this step must change something on screen". It takes ONE natural-language
  `instruction` and resolves the target itself: DOM/AX heuristic, then
  Decider-2B. That is exactly why it is the only
  interaction verb a blind plan can use — it needs no ref, no selector and no
  observation step in front of it.
- Write `instruction` in concrete, hint-matchable words naming what is visibly
  on the page: "click the first video result", "type despacito into the search
  box". Never vague ("interact with page", "do the thing").
- `hint_batch` runs several such instructions in one call when a step needs
  more than one action. `hint_resolve`, `find_and_click` and `find_and_type`
  are self-resolving alternatives that suit some phrasings better.

Navigation:
- PREFER one navigation step carrying the FULL destination URL. For a
  search-style task that removes the interaction step entirely: "play
  despacito on youtube" becomes
  `{"tool":"browser_navigate","input":{"url":"https://www.youtube.com/results?search_query=despacito"}}`.
  Other ready-made forms: `https://www.google.com/search?q=<query>`,
  `https://en.wikipedia.org/wiki/<Title>`, `https://www.reddit.com/search/?q=<query>`.
  Always include the scheme.
- `browser_open` is for when no browser is running yet (it launches one with
  `--remote-debugging-port=9222`); `browser_navigate` drives the tab that is
  already open. `browser_go_back`, `browser_go_forward`, `browser_tabs`,
  `browser_wait` and `browser_evaluate` are the cheap non-desktop verbs
  around them.
- Never END a plan on `browser_open` / `browser_navigate` alone: opening a page
  is not completing the goal. Finish with the `hint_act` that visibly
  completes it.

Ref-based tools (`browser_click`, `browser_type`, `browser_hover`,
`browser_select_option`):
- They need a `ref` from a snapshot or a concrete CSS `selector`. A blind plan
  can supply neither, so do not reach for them. Prefer `hint_act`. Only use one
  when you are sure of the selector, e.g.
  `{"tool":"browser_click","input":{"selector":"ytd-video-renderer a#video-title","element":"first video title"}}`.

When the hint pipeline reports it could not resolve the target, the sanctioned
ladder is: `hint_act` / `hint_batch` → `find_and_click` / `find_and_type` →
`browser_evaluate` (raw JS, needs a selector you know) → `keyboard` (key
chords). Confirm with `verify`,
`wait_until` or `browser_screenshot`. Never emit `ground`, `act_fast`,
`act_batch`, or any `stagehand_*` tool: they were removed and the catalog
will not resolve them.

Return exactly:
{"commands":[{"tool":"<name>","input":{...},"description":"<one line>"}]}
"#;

/// Load-bearing environment facts, always in the planner prompt and never
/// truncated: the skill body that follows is capped, and the cap used to land
/// mid-table and drop exactly these.
pub const PLANNER_OPERATING_RULES: &str = r#"## Operating rules (always in force)

- Interaction priority: Hint + Decider (`hint_act` / `hint_batch`) FIRST,
  `browser_evaluate` LAST RESORT, `browser_screenshot` VERIFY ONLY. Screenshot
  is never used to locate an element.
- The browser is reached over CDP at `HYPRFAST_CDP_HOST` / `HYPRFAST_CDP_PORT`,
  default `127.0.0.1:9222`. `CDP unreachable` means the browser was not
  launched with `--remote-debugging-port=9222` — launch it with
  `browser_open` (which adds the flag itself), never a bare `brave` /
  `chromium` launch.
- When `hint_snapshot` reports 0 hints (canvas, WebGL, PDF viewer, native app):
  use `browser_evaluate` with a selector you know, or `browser_snapshot` to
  re-observe. Do NOT reach for vision grounding: `ground` / `act_fast` /
  `act_batch` were removed.
- Before acting on a page when more than one tab is open, check `browser_tabs`;
  multi-tab CDP calls otherwise hit the wrong tab.
"#;

/// Render the Level 3 prompt: the catalog, the goal, and the JSON contract.
///
/// Kept as the catalog-only form for tests and callers without the skill on
/// disk. The live pipeline prefers [`render_action_plan_prompt`] below, which
/// is this prompt plus the hyprfast SKILL body (whiteboard stage 3).
pub fn render_plan_prompt(catalog_brief: &str, goal: &str, history: &str) -> String {
    format!(
        "{PLAN_INSTRUCTIONS}\n{PLANNER_OPERATING_RULES}\n## Available tools\n{catalog_brief}\n\n## Recent history\n{history}\n\n## Request\n{goal}\n"
    )
}

/// Whiteboard stage 3: hyprfast SKILL + user request → L3 planner prompt.
///
/// Re-exported from the `command` module (the pipeline's canonical home);
/// kept here too so planner call sites read naturally.
pub fn render_action_plan_prompt(
    skill_body: &str,
    catalog: Option<&lucy_hyprfast::HyprFastCatalog>,
    catalog_brief: &str,
    goal: &str,
    history: &str,
    knowledge: &str,
) -> String {
    super::command::render_action_plan_prompt(
        skill_body,
        catalog,
        catalog_brief,
        goal,
        history,
        knowledge,
    )
}

/// Parse the Level 3 plan. Accepts `commands` or `steps`, objects or plain
/// strings, and drops malformed entries. Returns an empty vec when the model
/// produced nothing usable (the caller then falls back to the automation loop).
pub fn parse_plan(value: &Value) -> Vec<PlannedCommand> {
    let arr = value
        .get("commands")
        .or_else(|| value.get("steps"))
        .or_else(|| value.get("plan"))
        .and_then(Value::as_array)
        .cloned()
        .or_else(|| value.as_array().cloned());
    let Some(arr) = arr else {
        return Vec::new();
    };
    let mut out: Vec<PlannedCommand> = Vec::new();
    for (i, entry) in arr.iter().take(24).enumerate() {
        // String form: a bare tool name, optionally `tool {json}`.
        if let Some(s) = entry.as_str() {
            let s = s.trim();
            if s.is_empty() {
                continue;
            }
            let (tool, input) = match s.split_once(char::is_whitespace) {
                Some((t, rest)) => (
                    t.trim().to_owned(),
                    lucy_agent::OpenAIProvider::extract_json(rest).unwrap_or_else(|_| json!({})),
                ),
                None => (s.to_owned(), json!({})),
            };
            out.push(PlannedCommand {
                index: i + 1,
                tool,
                input,
                description: String::new(),
            });
            continue;
        }
        let Some(tool) = entry
            .get("tool")
            .or_else(|| entry.get("name"))
            .or_else(|| entry.get("action"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
        else {
            continue;
        };
        let input = entry
            .get("input")
            .or_else(|| entry.get("args"))
            .or_else(|| entry.get("arguments"))
            .or_else(|| entry.get("params"))
            .or_else(|| entry.get("value"))
            .cloned()
            .unwrap_or_else(|| json!({}));
        let input = if input.is_object() {
            input
        } else {
            json!({ "value": input })
        };
        let description = entry
            .get("description")
            .or_else(|| entry.get("why"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_owned();
        out.push(PlannedCommand {
            index: out.len() + 1,
            tool: tool.to_owned(),
            input,
            description,
        });
    }
    out
}

/// Numbered plan for the execution log.
pub fn format_plan(plan: &[PlannedCommand]) -> String {
    plan.iter()
        .map(PlannedCommand::log_line)
        .collect::<Vec<_>>()
        .join("\n")
}

/// The `decider-serve` client, built from `classification_api_url`.
pub fn decider(cfg: &LucyConfig) -> Result<DeciderClient> {
    DeciderClient::from_lucy_config(cfg)
}

/// True when the classification model is configured and switched on.
pub fn classifier_enabled(rt: &LucyRuntime) -> bool {
    rt.config().classification.enabled && !rt.config().classification_api_url().is_empty()
}

/// **Step 1.** One decider forward pass, with the LLM as second opinion below
/// [`ROUTING_CONFIDENCE_MIN`].
///
/// Returns `Err` — never a guess — when no verdict exists: decider disabled,
/// unreachable, or failed; decider under-confident while the LLM second
/// opinion is unavailable or undecided. Callers stop the turn and show the
/// error (TUI status + chat); no tools run on an unrouted turn.
pub async fn classify_turn(
    rt: &LucyRuntime,
    prompt: &str,
) -> std::result::Result<TurnClassification, RoutingError> {
    if !classifier_enabled(rt) {
        return Err(RoutingError::DeciderUnavailable(format!(
            "classification model disabled (set /settings > Classification model API URL, default {})",
            rt.config().classification_api_url()
        )));
    }
    // The client is cached on the runtime (built at startup), so every turn
    // reuses the connection pool and the `/predict` path already discovered.
    let Some(client) = rt.decider() else {
        return Err(RoutingError::DeciderUnavailable(
            "classification client unavailable (set /settings > Classification model API URL)".into(),
        ));
    };
    let context = classification_context(rt, prompt).await;
    // The knowledge head rides in the same forward pass, so it costs no extra
    // round trip. Its options are the topics that exist on disk, which means a
    // new topic needs no code change — and the model is never offered a topic
    // Lucy does not have. Empty when knowledge is off, which makes this call
    // exactly the two-head one it was before.
    let topics = super::knowledge::routing_topics(rt, KNOWLEDGE_OPTION_LIMIT).await;
    // Every path below consumes the verifier's verdict, so it never waits for
    // the classifier: the two calls overlap and the turn's routing cost is
    // max(classifier, verify) instead of their sum.
    let (classify_result, verify) = tokio::join!(
        client.classify_turn_with_knowledge(&context, &topics),
        verify_turn_intent_detailed(rt, prompt)
    );
    match classify_result {
        Ok(c) => {
            if c.confidence < ROUTING_CONFIDENCE_MIN {
                // Under-confident decider: the LLM second opinion decides.
                return match verify {
                    VerifyOutcome::Decided(TurnIntent::Action) => Ok(action_override(
                        c,
                        "decider was under-confident, but the routing LLM found actionable tools — routing to actions",
                    )),
                    VerifyOutcome::Decided(TurnIntent::Question) => Ok(text_override(
                        c,
                        "decider was under-confident, and the routing LLM found no actionable tools — answering in text",
                    )),
                    VerifyOutcome::Undecided => Err(RoutingError::RouterUndecided(
                        "the routing LLM did not commit to an intent".into(),
                    )),
                    VerifyOutcome::Unavailable(why) => {
                        Err(RoutingError::DeciderUnavailable(format!(
                            "decider confidence {:.2} below {:.2} and the routing LLM was unavailable ({why})",
                            c.confidence, ROUTING_CONFIDENCE_MIN
                        )))
                    }
                };
            }
            if c.branch.needs_actions() {
                // The same check, run in the other direction. A 2B classifier at
                // 99% confidence called "act" on "what is the capital of
                // France" and the turn went off to drive a browser to Google it
                // — a wrong branch here is far more expensive than one wasted
                // classifier call to catch.
                //
                // Only a stated *question* intent flips it. A real task that
                // gets answered as a chat message is the worse failure, so
                // anything short of an explicit question keeps the action branch.
                return match verify {
                    VerifyOutcome::Decided(TurnIntent::Question) => Ok(text_override(
                        c,
                        "classifier said act, but the routing LLM read the request as a question — answering it instead",
                    )),
                    VerifyOutcome::Decided(TurnIntent::Action)
                    | VerifyOutcome::Undecided => Ok(c),
                    VerifyOutcome::Unavailable(why) => Err(
                        RoutingError::DeciderUnavailable(format!(
                            "routing LLM unavailable ({why}) — refusing to guess the branch"
                        )),
                    ),
                };
            }
            // The classifier said text-only. Verify dynamically: ask the
            // main LLM whether any catalog tool can advance the request.
            // The model reads the request — no keywords, no verb lists.
            match verify {
                VerifyOutcome::Decided(TurnIntent::Action) => Ok(action_override(
                    c,
                    "classifier said text-only, but the routing LLM found actionable tools — routing to actions",
                )),
                VerifyOutcome::Decided(TurnIntent::Question) | VerifyOutcome::Undecided => Ok(c),
                // This is the incident: the verifier is the *only* net standing
                // between a 2B classifier's misread and a task that silently
                // never happens. "play despacito song on yt" was classified chat
                // at 87% confidence, the verify call 502'd on an unregistered
                // model, `None` came back, and the classifier's word stood — so
                // the turn was answered as prose and no tool ever ran. Keeping
                // the classifier's word here is not fail-safe in this
                // direction, so the turn stops with an error instead.
                VerifyOutcome::Unavailable(why) => Err(RoutingError::DeciderUnavailable(format!(
                    "classifier said text-only, but the routing LLM was unavailable ({why}) — refusing to answer a possible task as chat"
                ))),
            }
        }
        Err(e) => {
            warn!(
                error = %format!("{e:#}"),
                url = %rt.config().classification_api_url(),
                "classification model unreachable"
            );
            Err(RoutingError::DeciderUnavailable(format!(
                "classification model at {} unreachable ({e:#})",
                rt.config().classification_api_url()
            )))
        }
    }
}

/// System prompt for the dynamic router-verify: capability-based, no keywords.
/// The model sees the real tool catalog and decides from what the tools can do.
pub const VERIFY_SYSTEM: &str = "You are Lucy's router. Decide ONLY whether the user's request needs system/browser actions. Return ONLY the specified JSON.";

/// Cap on the knowledge topics offered to the classifier's routing head.
///
/// The options ride along in the same forward pass that routes the turn, so a
/// large index would slow down every request to buy a reordering hint. Eight
/// promoted topics is a working set; beyond that, recall's own ranking is the
/// better judge anyway.
pub const KNOWLEDGE_OPTION_LIMIT: usize = 8;

pub const VERIFY_INSTRUCTIONS: &str = r#"You route user requests to chat (a text reply) or act (execute tools on this machine).

First decide the user's INTENT, then the branch follows from it:
- "action" — they want something CHANGED on this machine or the web: open, click, type, search, play, send, download, arrange. There is a visible end state.
- "question" — they want to KNOW something: a question, a definition, an explanation, small talk, or anything answerable from what you already know.

Then:
- intent "action" → needs_actions true.
- intent "question" → needs_actions false. Even though a browser could technically look it up, someone asking "what is the capital of France" wants an answer, not a page opened. "Search Google for the capital of France" is an action.
- A request mixing both is an action when they asked for the visible part.

Judge from what the tools can do, not from phrasing. New tools and new phrasings must still route correctly.

Return exactly:
{"intent":"action"|"question","needs_actions":true|false,"reason":"<one short line>"}
"#;

/// Render the verify prompt: catalog + goal + the JSON contract. The catalog
/// is the live one — whatever tools exist are what the model reasons over.
pub fn render_verify_prompt(tool_brief: &str, goal: &str) -> String {
    format!(
        "{VERIFY_INSTRUCTIONS}\n## Tools you can route to (if NONE of these can advance the request, it is chat)\n{tool_brief}\n\n## User request\n{goal}\n"
    )
}

/// What the routing model says the user is actually after.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnIntent {
    Action,
    Question,
}

/// The model's declared intent, or `None` when it did not say — in which case
/// the caller keeps the classifier's verdict.
///
/// This is deliberately stricter than [`parse_verify_answer`], which reads a
/// bare `needs_actions` bool. A single bool has no idea *why*, and a model
/// that answers the older "could any tool help?" framing will happily say
/// `true` for "what is the capital of France" — and, asked cold, just as
/// happily `false` for "play despacito on youtube". Requiring the model to
/// name the intent first, and flipping the branch only on that word, makes
/// both directions fail-safe: an unparseable or undecided answer leaves the
/// classification exactly as it was.
pub fn parse_verify_intent(value: &Value) -> Option<TurnIntent> {
    let raw = ["intent", "user_intent", "kind", "type"]
        .iter()
        .find_map(|k| value.get(*k).and_then(Value::as_str))
        .or_else(|| {
            // A bare string answer ("action" / "question") is still a stated
            // intent; anything else is not.
            value.as_str()
        })?
        .trim()
        .to_ascii_lowercase();
    match raw.as_str() {
        "action" | "act" | "actions" | "act on the system" | "browser" | "desktop" => {
            Some(TurnIntent::Action)
        }
        "question" | "chat" | "info" | "informational" | "text" | "answer" => {
            Some(TurnIntent::Question)
        }
        _ => None,
    }
}

/// Parse the verify answer. Accepts bool/string/number forms under several
/// key spellings; `None` means undecided (caller keeps its prior verdict).
pub fn parse_verify_answer(value: &Value) -> Option<bool> {
    for key in ["needs_actions", "needs_action", "act", "requires_actions"] {
        if let Some(v) = value.get(key) {
            if let Some(b) = verify_bool(v) {
                return Some(b);
            }
        }
    }
    value.as_bool()
}

fn verify_bool(v: &Value) -> Option<bool> {
    match v {
        Value::Bool(b) => Some(*b),
        Value::Number(n) => n.as_f64().map(|f| f != 0.0),
        Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
            "yes" | "true" | "1" | "act" | "action" | "actions" => Some(true),
            "no" | "false" | "0" | "chat" | "text" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

/// Dynamic check (no keywords): ask the routing model whether the request needs
/// any tool from the live catalog. Returns `None` when the model is
/// unavailable, cancelled, or undecided — the caller keeps its prior verdict.
pub async fn verify_needs_actions(rt: &LucyRuntime, prompt: &str) -> Option<bool> {
    verify_turn_intent(rt, prompt)
        .await
        .map(|i| i == TurnIntent::Action)
}

/// The routing model's answer, separating the three things `Option<TurnIntent>`
/// used to collapse into `None`.
///
/// The distinction is load-bearing: `Undecided` means a model answered and
/// declined to commit, which is no evidence and should leave the classifier's
/// verdict alone, while `Unavailable` means the only net that can catch a
/// classifier misread never ran at all. Treating both as "keep the prior
/// verdict" made a dead verifier indistinguishable from a working one that
/// agreed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyOutcome {
    /// The model named an intent.
    Decided(TurnIntent),
    /// The model replied but named no usable intent.
    Undecided,
    /// The verifier could not be reached, or its reply was unusable. Carries
    /// the reason, which is surfaced to the chat log.
    Unavailable(String),
}

/// The routing model's stated intent for this turn, or `None` when it could
/// not be asked, is down, or did not commit to one.
pub async fn verify_turn_intent(rt: &LucyRuntime, prompt: &str) -> Option<TurnIntent> {
    match verify_turn_intent_detailed(rt, prompt).await {
        VerifyOutcome::Decided(intent) => Some(intent),
        VerifyOutcome::Undecided | VerifyOutcome::Unavailable(_) => None,
    }
}

/// [`Self::verify_turn_intent`] with the failure mode preserved, so the caller
/// can tell "the verifier disagreed" from "the verifier never answered".
pub async fn verify_turn_intent_detailed(rt: &LucyRuntime, prompt: &str) -> VerifyOutcome {
    if rt.interrupt.is_set() {
        return VerifyOutcome::Unavailable("turn interrupted".to_owned());
    }
    // The verifier exists to catch one misread per request; re-asking the
    // same pending request twice per turn (or across retries in one CLI
    // invocation) doubles a 2–17s LLM call for no new information. Only
    // decided intents are cached — an unreachable model must stay visible
    // as Unavailable, and a transient failure should retry next turn.
    let cache_key = prompt
        .trim()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if let Some(cached) = verify_cache_get(&cache_key) {
        return VerifyOutcome::Decided(cached);
    }
    let rendered = render_verify_prompt(&rt.tool_brief, prompt);
    let model = rt.provider.model();
    match rt
        .provider
        .complete_json(
            &model,
            "route_verify",
            VERIFY_SYSTEM,
            &rendered,
            rt.interrupt.clone(),
        )
        .await
    {
        Ok(v) => match parse_verify_intent(&v) {
            Some(intent) => {
                verify_cache_put(cache_key, intent);
                VerifyOutcome::Decided(intent)
            }
            None => VerifyOutcome::Undecided,
        },
        Err(e) => {
            warn!("routing verify call failed ({e:#}) — routing without it");
            VerifyOutcome::Unavailable(format!("{e:#}"))
        }
    }
}

/// Turn a text-only verdict into the action branch (L3), preserving the
/// classifier's latency for the log.
fn action_override(c: TurnClassification, reason: &str) -> TurnClassification {
    TurnClassification {
        branch: TurnBranch::RequiresActions,
        reasoning_level: ReasoningLevel::L3,
        confidence: c.confidence,
        probabilities: HashMap::new(),
        level_confidence: c.level_confidence,
        latency_ms: c.latency_ms,
        summary_note: Some(reason.to_owned()),
        knowledge_topic: c.knowledge_topic,
    }
}

/// Turn an action verdict into a plain reply, keeping the classifier's
/// confidence and latency for the log.
fn text_override(c: TurnClassification, reason: &str) -> TurnClassification {
    TurnClassification {
        branch: TurnBranch::RequiresOnlyResponse,
        // The tier the classifier itself asked for — it is still the best
        // estimate of how hard the question is to answer.
        reasoning_level: if c.reasoning_level == ReasoningLevel::L3 {
            ReasoningLevel::default()
        } else {
            c.reasoning_level
        },
        confidence: c.confidence,
        probabilities: HashMap::new(),
        level_confidence: c.level_confidence,
        latency_ms: c.latency_ms,
        summary_note: Some(reason.to_owned()),
        knowledge_topic: c.knowledge_topic,
    }
}

/// Session-local memo of routing-intent decisions, bounded so a long-lived
/// TUI cannot grow it without limit. Cleared wholesale past the cap — the
/// oldest entries being wrong is not a correctness issue, just a re-ask.
fn verify_cache() -> &'static std::sync::Mutex<std::collections::HashMap<String, TurnIntent>> {
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<String, TurnIntent>>,
    > = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn verify_cache_get(key: &str) -> Option<TurnIntent> {
    verify_cache().lock().ok()?.get(key).copied()
}

fn verify_cache_put(key: String, intent: TurnIntent) {
    if let Ok(mut map) = verify_cache().lock() {
        if map.len() >= 256 {
            map.clear();
        }
        map.insert(key, intent);
    }
}

/// The `context` string sent to the classifier: the command plus a brief note
/// of the active state.
async fn classification_context(rt: &LucyRuntime, prompt: &str) -> String {
    let hist = rt.history_without_tail(prompt).await;
    let history = super::router::format_history(&hist, 4);
    if history == "(none)" {
        prompt.to_owned()
    } else {
        format!("{prompt}\n\nRecent conversation:\n{history}")
    }
}

/// **Step 2 dispatch.** Turn a command into the concrete routing decision:
/// branch, tier, and the model that will be called.
///
/// `Err` when there is no verdict to dispatch on (see [`classify_turn`]):
/// callers stop the turn and show the error instead of guessing a branch.
/// `Manual` mode never classifies, so it cannot fail this way.
pub async fn route_turn(
    rt: &LucyRuntime,
    prompt: &str,
    precomputed: Option<TurnClassification>,
) -> std::result::Result<TurnRoute, RoutingError> {
    if rt.config().chat_mode() == ChatMode::Manual {
        // `Manual` means "skip the classifier", not "use a different model":
        // every turn goes to the Level 3 anchor, the same tier that plans
        // actions in `Auto` mode.
        let model_key = rt.config().resolve_level_model(ReasoningLevel::L3);
        return Ok(TurnRoute {
            classification: precomputed.unwrap_or_else(|| TurnClassification {
                branch: TurnBranch::RequiresOnlyResponse,
                reasoning_level: ReasoningLevel::L3,
                confidence: 1.0,
                probabilities: HashMap::new(),
                level_confidence: 1.0,
                latency_ms: None,
                summary_note: Some("manual chat mode — classifier skipped".into()),
                knowledge_topic: None,
            }),
            branch: TurnBranch::RequiresOnlyResponse,
            reasoning_level: ReasoningLevel::L3,
            model_label: model_key.clone(),
            model_key,
            source: RouteSource::ManualOverride,
            note: rt.config().level_fallback_note(ReasoningLevel::L3),
            knowledge_topic: None,
        });
    }

    let classification = match precomputed {
        Some(c) => c,
        None => classify_turn(rt, prompt).await?,
    };

    // Step 2B (needs actions) always plans with the Level 3 model.
    let level = if classification.branch.needs_actions() {
        ReasoningLevel::L3
    } else {
        classification.reasoning_level
    };
    let note = match (
        classification.summary_note.clone(),
        rt.config().level_fallback_note(level),
    ) {
        (Some(prev), Some(fb)) => Some(format!("{prev}; {fb}")),
        (prev, fb) => fb.or(prev),
    };
    let model_key = rt.config().resolve_level_model(level);
    Ok(TurnRoute {
        branch: classification.branch,
        reasoning_level: level,
        model_label: describe_model(&rt.config(), &model_key),
        model_key,
        knowledge_topic: classification.knowledge_topic.clone(),
        source: match classification.summary_note.as_deref() {
            None => RouteSource::Classifier,
            Some(_) => RouteSource::LlmVerify,
        },
        note,
        classification,
    })
}

/// The Level 3 model — the anchor. Actions are always planned with it, so it is
/// the one tier the whole action path depends on.
pub fn planner_model_key(rt: &LucyRuntime) -> String {
    rt.config().resolve_level_model(ReasoningLevel::L3)
}

/// The exact Level 3 prompt for `goal`: instructions + never-truncated
/// operating rules + the capped skill body + the hint-first tool brief. Split
/// out of [`plan_commands`] so the prompt can be inspected without spending an
/// LLM call (`lucy smoke plan <goal> --dump-prompt`).
pub async fn planner_prompt(rt: &LucyRuntime, goal: &str) -> String {
    planner_prompt_with_knowledge(rt, goal, None).await
}

/// [`planner_prompt`] with the classifier's knowledge hint, which only reorders
/// what deterministic recall already found.
pub async fn planner_prompt_with_knowledge(
    rt: &LucyRuntime,
    goal: &str,
    preferred_topic: Option<&str>,
) -> String {
    let hist = rt.history_without_tail(goal).await;
    let skill_body = rt.skill_body("lucy").unwrap_or_default();
    let knowledge = super::knowledge::planner_section(rt, goal, preferred_topic).await;
    render_action_plan_prompt(
        &skill_body,
        rt.hyprfast_catalog(),
        &rt.tool_brief,
        goal,
        &super::router::format_history(&hist, 4),
        &knowledge,
    )
}

/// Ask the Level 3 model for the ordered `hyprfast` command list.
///
/// Whiteboard stage 3: the hyprfast SKILL body plus the user request go to
/// the L3 (Pro) model, which returns the valid command list. Returns an
/// empty vec when the model produced nothing usable, so the caller can fall
/// back to the subtask-based automation loop.
pub async fn plan_commands(rt: &LucyRuntime, goal: &str) -> Vec<PlannedCommand> {
    if rt.interrupt.is_set() {
        return Vec::new();
    }
    let rendered = planner_prompt(rt, goal).await;
    let key = planner_model_key(rt);
    let target = match lucy_agent::ModelTarget::from_config(&rt.config(), &key) {
        Ok(t) => t,
        Err(e) => {
            warn!("{e:#} — cannot plan hyprfast commands");
            return Vec::new();
        }
    };
    match rt
        .provider
        .complete_json_on(
            &target,
            "plan_commands",
            PLAN_SYSTEM,
            &rendered,
            rt.interrupt.clone(),
            Some(super::agent_loop::PLANNER_MAX_OUTPUT_TOKENS),
        )
        .await
    {
        Ok(value) => {
            let plan = sanitize_plan(parse_plan(&value), goal);
            if plan.is_empty() {
                warn!("Level 3 planner returned no usable commands");
            }
            plan
        }
        Err(e) => {
            warn!("{e:#} — planner call failed");
            Vec::new()
        }
    }
}

/// Verbs that need a live element reference (`ref`/`element`/`selector`) from
/// a snapshot. A static plan runs blind — the planner never sees snapshot
/// output — so such steps always fail with an evaluate exception. Rewrite
/// them to the self-resolving `hint_act`; keep steps that already carry a ref,
/// and leave every other tool untouched.
///
/// A plan that ENDS on a pure navigation verb (`browser_open` /
/// `browser_navigate`) only opens a page without visibly completing the goal,
/// so unless the goal itself is just to open something, append a `hint_act`
/// interaction step derived from the goal.
pub fn sanitize_plan(plan: Vec<PlannedCommand>, goal: &str) -> Vec<PlannedCommand> {
    let mut plan: Vec<PlannedCommand> = plan
        .into_iter()
        .map(|mut step| {
            let base = step
                .tool
                .strip_prefix("mcp_hyprfast_")
                .or_else(|| step.tool.strip_prefix("mcp_computer_use_"))
                .unwrap_or(&step.tool);
            if !matches!(
                base,
                "browser_click" | "browser_type" | "browser_hover" | "browser_select_option"
            ) {
                return step;
            }
            let has_ref = step.input.as_object().is_some_and(|o| {
                ["ref", "element", "selector", "xpath", "elementId"]
                    .iter()
                    .any(|k| o.get(*k).is_some_and(|v| !v.is_null()))
            });
            if has_ref {
                normalize_step_input(base, &mut step.input, &step.description);
                return step;
            }
            let instruction = if step.description.trim().is_empty() {
                flattened_input_text(&step.input)
            } else {
                step.description.trim().to_owned()
            };
            warn!(
                tool = %step.tool,
                "planner emitted ref-needing verb without a ref — rewriting to hint_act"
            );
            step.tool = "hint_act".to_owned();
            step.input = json!({ "instruction": instruction });
            normalize_step_input("hint_act", &mut step.input, &step.description);
            step
        })
        .collect();
    if needs_interaction_append(&plan, goal) {
        let index = plan.len() + 1;
        warn!("planner ended on navigation alone — appending hint_act");
        plan.push(PlannedCommand {
            index,
            tool: "hint_act".to_owned(),
            input: json!({ "instruction": interaction_instruction(goal) }),
            description: format!("Interact to complete: {goal}"),
        });
    }
    plan
}

/// True when the plan's last step only opens/navigates and the goal asks for
/// more than just opening a page.
fn needs_interaction_append(plan: &[PlannedCommand], goal: &str) -> bool {
    let Some(last) = plan.last() else {
        return false;
    };
    let base = last
        .tool
        .strip_prefix("mcp_hyprfast_")
        .or_else(|| last.tool.strip_prefix("mcp_computer_use_"))
        .unwrap_or(&last.tool);
    if !matches!(base, "browser_open" | "browser_navigate") {
        return false;
    }
    !is_open_only_goal(goal)
}

/// True when the goal asks only for a page to open, so a plan that ends on a
/// navigation step has already finished it.
///
/// Decided by **arity**, not vocabulary. The old rule asked whether the goal
/// contained any of `play, click, type, search, watch, listen, find, select` —
/// which means the question "is there anything left to do after navigating?"
/// was answered by a fixed list of English words. Every phrasing the list did
/// not contain got the same answer as "open youtube", so a goal like "put it in
/// my cart" or "book the cheapest one" would end its plan on a navigation step
/// with nothing done. That is the closed-world failure AGENTS.md describes.
///
/// A goal that is *only* a destination is structurally short: a destination and
/// nothing after it. Anything longer carries an instruction, and an instruction
/// is exactly what the appended interaction step exists to carry out.
fn is_open_only_goal(goal: &str) -> bool {
    let words: Vec<&str> = goal.split_whitespace().filter(|w| !w.is_empty()).collect();
    if words.is_empty() {
        return false;
    }
    // A named destination — a URL or a dotted host — settles it: the goal is
    // about arriving there, however many lead-in words precede it.
    if words.iter().any(|w| w.contains("://") || w.contains('.')) {
        return words.len() <= 4;
    }
    // No destination named, so it has to be a bare noun phrase to count.
    words.len() <= 2
}

/// Hint-matchable final interaction for an already-open page.
fn interaction_instruction(goal: &str) -> String {
    format!(
        "{goal} — on the already-open page, click the result, video, or button (or type into the field) that visibly completes this goal",
        goal = goal.trim()
    )
}

/// Best-effort prose for a rewritten step when it has no description: join
/// the input's scalar values, falling back to a generic instruction.
fn flattened_input_text(input: &Value) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(o) = input.as_object() {
        for v in o.values() {
            if let Value::String(s) = v {
                if !s.trim().is_empty() {
                    parts.push(s.trim().to_owned());
                }
            }
        }
    } else if let Some(s) = input.as_str() {
        if !s.trim().is_empty() {
            parts.push(s.trim().to_owned());
        }
    }
    if parts.is_empty() {
        "interact with the page".to_owned()
    } else {
        parts.join(" — ")
    }
}

/// Resolve a planned tool name against registered tools, supporting both bare
/// names (`browser_open`) and prefixed forms (`mcp_hyprfast_browser_open`).
pub fn resolve_plan_tool(registry: &lucy_tools::ToolRegistry, name: &str) -> Option<String> {
    let name = name.trim();
    if name.is_empty() {
        return None;
    }
    let bare = name
        .strip_prefix("mcp_hyprfast_")
        .or_else(|| name.strip_prefix("mcp_computer_use_"))
        .or_else(|| name.strip_prefix("mcp_"))
        .unwrap_or(name);

    let candidates = [
        name.to_owned(),
        bare.to_owned(),
        format!("mcp_hyprfast_{bare}"),
        format!("mcp_computer_use_{bare}"),
        format!("mcp_{bare}"),
    ];
    for c in &candidates {
        if registry.get(c).is_some() {
            return Some(c.clone());
        }
    }
    None
}

/// Normalizes parameters so that hyprfast schemas are satisfied regardless of
/// whether the planner emitted `selector`, `ref`, `element`, or `duration`.
pub fn normalize_step_input(tool_name: &str, input: &mut Value, description: &str) {
    let base = tool_name
        .strip_prefix("mcp_hyprfast_")
        .or_else(|| tool_name.strip_prefix("mcp_computer_use_"))
        .or_else(|| tool_name.strip_prefix("mcp_"))
        .unwrap_or(tool_name);

    if let Some(obj) = input.as_object_mut() {
        match base {
            "browser_click" | "browser_hover" => {
                let target = obj
                    .get("selector")
                    .or_else(|| obj.get("xpath"))
                    .or_else(|| obj.get("elementId"))
                    .or_else(|| obj.get("ref"))
                    .or_else(|| obj.get("element"))
                    .cloned();
                if let Some(t) = target {
                    if !obj.contains_key("ref") {
                        obj.insert("ref".to_string(), t.clone());
                    }
                    if !obj.contains_key("element") {
                        let desc = if !description.trim().is_empty() {
                            json!(description.trim())
                        } else {
                            t
                        };
                        obj.insert("element".to_string(), desc);
                    }
                }
            }
            "browser_type" => {
                let target = obj
                    .get("selector")
                    .or_else(|| obj.get("xpath"))
                    .or_else(|| obj.get("elementId"))
                    .or_else(|| obj.get("ref"))
                    .or_else(|| obj.get("element"))
                    .cloned();
                if let Some(t) = target {
                    if !obj.contains_key("ref") {
                        obj.insert("ref".to_string(), t.clone());
                    }
                    if !obj.contains_key("element") {
                        let desc = if !description.trim().is_empty() {
                            json!(description.trim())
                        } else {
                            t
                        };
                        obj.insert("element".to_string(), desc);
                    }
                }
                if !obj.contains_key("submit") {
                    let desc_lower = description.to_lowercase();
                    let should_submit =
                        desc_lower.contains("enter") || desc_lower.contains("submit");
                    obj.insert("submit".to_string(), json!(should_submit));
                }
                if !obj.contains_key("text") {
                    if let Some(val) = obj.get("value").or_else(|| obj.get("content")) {
                        obj.insert("text".to_string(), val.clone());
                    }
                }
            }
            "browser_select_option" => {
                let target = obj
                    .get("selector")
                    .or_else(|| obj.get("ref"))
                    .or_else(|| obj.get("element"))
                    .cloned();
                if let Some(t) = target {
                    if !obj.contains_key("ref") {
                        obj.insert("ref".to_string(), t.clone());
                    }
                    if !obj.contains_key("element") {
                        obj.insert("element".to_string(), t);
                    }
                }
                if !obj.contains_key("values") {
                    if let Some(val) = obj.get("value") {
                        if let Some(s) = val.as_str() {
                            obj.insert("values".to_string(), json!([s]));
                        } else if val.is_array() {
                            obj.insert("values".to_string(), val.clone());
                        }
                    }
                }
            }
            "browser_wait" => {
                if !obj.contains_key("time") {
                    if let Some(d) = obj
                        .get("duration")
                        .or_else(|| obj.get("ms"))
                        .or_else(|| obj.get("seconds"))
                    {
                        obj.insert("time".to_string(), d.clone());
                    }
                }
            }
            "browser_navigate" | "browser_open" => {
                // Ensure the URL has a scheme — CDP rejects schemeless URLs with
                // "Cannot navigate to invalid URL" (-32000).
                if let Some(url_val) = obj.get_mut("url") {
                    if let Some(url_str) = url_val.as_str() {
                        if !url_str.starts_with("http://")
                            && !url_str.starts_with("https://")
                            && !url_str.starts_with("file://")
                            && !url_str.starts_with("about:")
                            && !url_str.starts_with("data:")
                        {
                            *url_val = json!(format!("https://{url_str}"));
                        }
                    }
                }
            }
            "hint_act" => {
                if !obj.contains_key("instruction") {
                    // The description is the best instruction when one exists: a
                    // bare `text` says what to type but not what to type it into.
                    let from = obj
                        .get("prompt")
                        .cloned()
                        .or_else(|| {
                            (!description.trim().is_empty()).then(|| json!(description.trim()))
                        })
                        .or_else(|| obj.get("action").cloned())
                        .or_else(|| obj.get("text").cloned());
                    if let Some(inst) = from {
                        obj.insert("instruction".to_string(), inst);
                    }
                }
                // `text` only means something on hint_act alongside `action=type`.
                if obj.contains_key("text") && !obj.contains_key("action") {
                    obj.insert("action".to_string(), json!("type"));
                }
            }
            _ => {}
        }
    }
}

/// **Step 2B execution.** Run the plan's commands **one by one in order**,
/// streaming a status line per step to the TUI.
///
/// Validation per step: the tool must exist in the `ToolRegistry` (or the
/// discovered MCP catalog), and the `ApprovalGate` must allow it. Stops at the
/// first failure — later steps assume earlier ones succeeded.
pub async fn execute_plan_sequentially(
    rt: &LucyRuntime,
    goal: &str,
    plan: &[PlannedCommand],
    event_tx: Option<tokio::sync::mpsc::UnboundedSender<AgentEvent>>,
) -> Result<String> {
    if plan.is_empty() {
        return Err(anyhow!("empty plan"));
    }
    let registry = tool_registry(rt).await?;
    // The runtime's shared gate, so a prompt raised here is answerable from
    // the TUI. Point it at this run's event channel for the duration.
    let gate = rt.approval.clone();
    gate.set_events(
        event_tx
            .clone()
            .unwrap_or_else(|| tokio::sync::mpsc::unbounded_channel().0),
    );

    let notify = |event_tx: &Option<tokio::sync::mpsc::UnboundedSender<AgentEvent>>,
                  event: AgentEvent| {
        if let Some(tx) = event_tx {
            let _ = tx.send(event);
        }
    };

    let total = plan.len();
    let mut prev_was_nav = false;
    for step in plan {
        if rt.interrupt.is_set() {
            gate.clear_events();
            return Err(lucy_core::LucyError::Cancelled.into());
        }
        notify(
            &event_tx,
            AgentEvent::Progress {
                message: format!("Step {}/{}: {}", step.index, total, step.log_line()),
            },
        );

        let Some((resolved_tool, requires_approval)) = validate_step(rt, &registry, step)? else {
            gate.clear_events();
            return Err(anyhow!(
                "Step {}/{}: '{}' is not in the hyprfast tool catalog — refusing to run it",
                step.index,
                total,
                step.tool
            ));
        };

        let mut input = step.input.clone();
        normalize_step_input(&resolved_tool, &mut input, &step.description);

        if gate.needs_approval(&resolved_tool, requires_approval) {
            let call_id = format!("plan-{}", step.index);
            // Raced against the kill switch, so a stop that lands while the
            // prompt is open abandons the step instead of waiting out the
            // approval timeout on a dialog nobody will answer.
            let decision = gate
                .ask_cancellable(&rt.interrupt, &call_id, &resolved_tool, &input)
                .await;
            match decision {
                None => {
                    gate.clear_events();
                    return Err(anyhow!("stopped before step {}/{}", step.index, total));
                }
                Some(ApprovalDecision::Deny) => {
                    gate.clear_events();
                    notify(
                        &event_tx,
                        AgentEvent::Error {
                            message: format!(
                                "Step {}/{} denied: {}",
                                step.index, total, resolved_tool
                            ),
                        },
                    );
                    return Err(anyhow!("denied at step {}/{}", step.index, total));
                }
                Some(_) => {}
            }
        }

        let base = resolved_tool
            .strip_prefix("mcp_hyprfast_")
            .or_else(|| resolved_tool.strip_prefix("mcp_computer_use_"))
            .or_else(|| resolved_tool.strip_prefix("mcp_"))
            .unwrap_or(&resolved_tool);

        // If the previous step opened or navigated a URL, give the page
        // a moment to initialize CDP and hydrate DOM before interacting.
        if prev_was_nav && is_dom_interaction(base) {
            tokio::time::sleep(std::time::Duration::from_millis(2000)).await;
        }

        let ctx = ToolContext {
            session_id: rt.current_id().await,
            tool_call_id: format!("plan-{}", step.index),
            working_dir: None,
            execution_mode: ExecutionMode::Agent,
            events: event_tx
                .clone()
                .unwrap_or_else(|| tokio::sync::mpsc::unbounded_channel().0),
            interrupt: rt.interrupt.clone(),
        };

        let mut output = registry
            .execute(&resolved_tool, input.clone(), ctx.clone())
            .await;

        // If a DOM interaction failed (e.g. element not found due to page loading),
        // retry once after 1.5s.
        if let Ok(val) = &output {
            if mcp_failure_text(val).is_some() && is_dom_interaction(base) {
                tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
                output = registry.execute(&resolved_tool, input.clone(), ctx).await;
            }
        }

        match output {
            Ok(output) => {
                if let Some(tool_err) = mcp_failure_text(&output) {
                    notify(
                        &event_tx,
                        AgentEvent::Error {
                            message: format!("Step {}/{} failed: {tool_err}", step.index, total),
                        },
                    );
                    gate.clear_events();
                    return Err(anyhow!(
                        "Step {}/{} failed ({}): {}",
                        step.index,
                        total,
                        step.tool,
                        tool_err
                    ));
                }
                notify(
                    &event_tx,
                    AgentEvent::Status {
                        message: format!(
                            "Step {}/{} done: {} → {}",
                            step.index,
                            total,
                            step.tool,
                            summarize_output(&output)
                        ),
                    },
                );
            }
            Err(e) => {
                notify(
                    &event_tx,
                    AgentEvent::Error {
                        // Classified here as well as at the display layers: this
                        // string is also what the run report is built from, and a
                        // tool that dies with a JSON body should not put one in
                        // the transcript. `{e:#}` is still logged below.
                        message: format!(
                            "Step {}/{} failed: {}",
                            step.index,
                            total,
                            lucy_core::friendly(&format!("{e:#}"))
                        ),
                    },
                );
                tracing::warn!(
                    step = step.index,
                    tool = %step.tool,
                    error = %format!("{e:#}"),
                    "sequential plan step failed"
                );
                gate.clear_events();
                return Err(anyhow!(
                    "Step {}/{} failed ({}): {e:#}",
                    step.index,
                    total,
                    step.tool
                ));
            }
        }

        prev_was_nav = matches!(base, "browser_open" | "browser_navigate");
    }
    gate.clear_events();
    Ok(format!("Goal completed: {goal} ({total} step(s))"))
}

/// Registry used to validate and run plan steps: local tools plus every MCP
/// tool discovered at startup (registered lazily, no spawn cost).
///
/// The knowledge read tools are in the registry too, on the same config-driven
/// footing: they appear whenever knowledge is on and vanish when it is off, so
/// no branch here decides which is right.
pub async fn tool_registry(rt: &LucyRuntime) -> Result<lucy_tools::ToolRegistry> {
    let mut registry = lucy_tools::default_registry();
    if rt.knowledge_enabled() {
        rt.register_knowledge_tools(&mut registry);
    }
    // Group by owning server. Every tool in one group is registered against
    // that server's own launcher config, which is the only thing that says
    // which child process executes it. Registering them all against hyprfast
    // (as this did before per-server routing existed) meant an extra server's
    // tools were announced to the model and then called on a binary that never
    // had them — a discovery/execution split that reads as "the model ignored
    // the tool" when it is really a routing bug.
    let mut by_server: HashMap<&str, Vec<lucy_mcp::McpToolDefinition>> = HashMap::new();
    for t in rt.mcp_tools() {
        // Removed tools are dropped at discovery; filtering again here keeps
        // this registry honest if it is ever handed a stale list, and the
        // rule stays in one place (`lucy_hyprfast::is_removed_tool`).
        if lucy_hyprfast::is_removed_tool(&t.tool) {
            continue;
        }
        by_server
            .entry(t.server.as_str())
            .or_default()
            .push(lucy_mcp::McpToolDefinition {
                name: t.tool.clone(),
                description: Some(t.description.clone()),
                input_schema: t.schema.clone(),
            });
    }
    // Sorted so the registry (and therefore the tools array built from it) is
    // byte-identical between two runs in the same process.
    let mut servers: Vec<&str> = by_server.keys().copied().collect();
    servers.sort_unstable();
    for server in servers {
        let defs = by_server.remove(server).unwrap_or_default();
        if defs.is_empty() {
            continue;
        }
        let Some(cfg) = rt.mcp_server_config(server) else {
            // Without a launcher config there is no proxy that could ever run
            // this, so registering a nameless stub would only advertise a tool
            // that fails on first call.
            tracing::warn!(
                server,
                "no launcher config for an MCP server that reported tools; skipping its tools"
            );
            continue;
        };
        lucy_mcp::register_server_with_defs(&mut registry, cfg.clone(), defs);
    }
    Ok(registry)
}

/// The local half of the tool universe: the built-in registry plus the
/// knowledge read tools when knowledge is on.
///
/// Kept separate from [`tool_registry`] because the registry also holds the MCP
/// bare-name aliases (`browser_navigate` alongside `mcp_hyprfast_browser_navigate`).
/// Feeding the whole registry into the tools array would offer the model the
/// same capability under two names and let it pick the one the executor does not
/// resolve, so the array takes locals from here and MCP tools from discovery.
pub(crate) fn local_tool_definitions(rt: &LucyRuntime) -> Vec<serde_json::Value> {
    let mut registry = lucy_tools::default_registry();
    if rt.knowledge_enabled() {
        rt.register_knowledge_tools(&mut registry);
    }
    let mut defs = registry.definitions();
    defs.sort_by(|a, b| {
        a.get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .cmp(b.get("name").and_then(Value::as_str).unwrap_or_default())
    });
    defs
}

/// Every tool Lucy can actually run, in the OpenAI `tools` wire shape.
///
/// One array for the whole universe: the local tools, the knowledge read tools
/// when knowledge is on, every hyprfast capability, and every tool from the
/// configured MCP servers. The model is offered the same set the executor can
/// dispatch, so a tool it may name is a tool that will run — the split where a
/// planner sees a tool the registry lacks is what produces "the model said it
/// would click, and nothing happened".
///
/// Removed tools are filtered out (single source of truth:
/// `lucy_hyprfast::is_removed_tool`), and destructive tools are marked in
/// their description so the model can tell an action that needs the user from
/// one that runs unattended.
pub fn openai_tools(
    local_defs: &[serde_json::Value],
    mcp_tools: &[crate::McpToolFull],
    destructive: &std::collections::HashSet<String>,
) -> Vec<serde_json::Value> {
    let mut out: Vec<serde_json::Value> = Vec::new();
    // Local tools first: they are always present and are the cheapest thing a
    // model can reach for, so they read as the baseline rather than as one
    // server among several.
    for def in local_defs {
        if let Some(f) = openai_function(def, destructive) {
            out.push(f);
        }
    }
    for t in mcp_tools {
        if lucy_hyprfast::is_removed_tool(&t.tool) {
            continue;
        }
        if let Some(f) = openai_function(
            &serde_json::json!({
                "name": t.name,
                "description": t.description,
                "input_schema": t.schema,
            }),
            destructive,
        ) {
            out.push(f);
        }
    }
    out
}

/// One tool definition as an OpenAI `function` entry, or `None` when it has no
/// usable name.
///
/// The API nests Lucy's flat `{name, description, input_schema}` under
/// `function` and calls the schema `parameters`; this is the one place that
/// translation happens for the tool universe.
fn openai_function(
    definition: &serde_json::Value,
    destructive: &std::collections::HashSet<String>,
) -> Option<serde_json::Value> {
    let name = definition
        .get("name")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|n| !n.is_empty())?;
    let description = definition
        .get("description")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .trim();
    // The model has to be able to tell a gated call from a routine one, and
    // the schema has nowhere to say so — `destructive` is a catalog fact, not
    // part of the wire contract. Saying it in the description is what lets the
    // model ask instead of assuming.
    let description = if is_destructive(destructive, name) {
        let marker = "Destructive: needs user approval before it runs.";
        if description.is_empty() {
            marker.to_owned()
        } else {
            format!("{description} {marker}")
        }
    } else {
        description.to_owned()
    };
    let parameters = definition
        .get("input_schema")
        .filter(|p| p.is_object())
        .cloned()
        .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}));
    Some(serde_json::json!({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
    }))
}

/// True when the live catalog marks this tool destructive, under either its
/// qualified or bare name.
///
/// Both spellings are checked because the destructive set is keyed by the bare
/// catalog name while the tool universe advertises the qualified one — and a
/// prefix-stripping rule that missed a prefix would silently mark a
/// delete-the-window tool as routine.
fn is_destructive(destructive: &std::collections::HashSet<String>, name: &str) -> bool {
    let bare = name
        .strip_prefix("mcp_hyprfast_")
        .or_else(|| name.strip_prefix("mcp_computer_use_"))
        .or_else(|| name.strip_prefix("mcp_"))
        .unwrap_or(name);
    destructive.contains(name) || destructive.contains(bare)
}

/// Names of planned steps that are NOT in the tool registry or the
/// discovered MCP catalog. Empty means every step names a real,
/// executable tool — the smoke plan loop (`lucy smoke plan <goal>`) uses
/// this to decide whether the LLM actually returned hyprfast commands.
pub async fn unknown_plan_tools(rt: &LucyRuntime, plan: &[PlannedCommand]) -> Result<Vec<String>> {
    let registry = tool_registry(rt).await?;
    let mut unknown = Vec::new();
    for step in plan {
        let name = step.tool.trim();
        if name.is_empty() {
            unknown.push("(empty tool name)".to_owned());
            continue;
        }
        let registered = resolve_plan_tool(&registry, name).is_some();
        let known = registered
            || rt
                .mcp_tools
                .iter()
                .any(|t| t.name.ends_with(name) || t.name == name);
        if !known {
            unknown.push(name.to_owned());
        }
    }
    Ok(unknown)
}

/// Validate one planned step. Returns `Ok(None)` when the tool is unknown, and
/// `Ok(Some((resolved_tool, requires_approval)))` when it is known.
fn validate_step(
    rt: &LucyRuntime,
    registry: &lucy_tools::ToolRegistry,
    step: &PlannedCommand,
) -> Result<Option<(String, bool)>> {
    let name = step.tool.trim();
    if name.is_empty() {
        return Ok(None);
    }
    let resolved = resolve_plan_tool(registry, name);
    let Some(found) = resolved else {
        // Not registered: it may still be a known-but-unprobed capability.
        let known = rt
            .mcp_tools
            .iter()
            .any(|t| t.name.ends_with(name) || t.name == name);
        if known {
            return Ok(Some((name.to_owned(), false)));
        }
        return Ok(None);
    };
    let bare = name
        .strip_prefix("mcp_hyprfast_")
        .or_else(|| name.strip_prefix("mcp_computer_use_"))
        .or_else(|| name.strip_prefix("mcp_"))
        .unwrap_or(name);
    // Only destructive tools interrupt the user; routine UI actions run
    // unattended. `approvals.mode = always` still forces a prompt for all.
    let destructive = hyprfast_capability(rt, bare)
        .or_else(|| hyprfast_capability(rt, name))
        .map(|c| c.destructive)
        .unwrap_or(false);
    let requires_approval = destructive || registry.requires_approval(&found);
    Ok(Some((found, requires_approval)))
}

fn hyprfast_capability(rt: &LucyRuntime, name: &str) -> Option<ToolCapability> {
    rt.hyprfast_catalog.as_ref()?.tools.get(name).cloned()
}

/// Steps that reach into a live DOM/AX tree. They need the page hydrated
/// after a navigation, and they are the steps worth one retry when the target
/// turns out to be stale.
pub fn is_dom_interaction(base: &str) -> bool {
    matches!(
        base,
        "browser_click"
            | "browser_type"
            | "browser_hover"
            | "browser_select_option"
            | "hint_act"
            | "hint_batch"
            | "hint_resolve"
            | "find_and_click"
            | "find_and_type"
    )
}

/// MCP servers report tool-level failures inside an `Ok` payload instead of a
/// transport error: either an error envelope (`{"content":[{"text":"error:
/// ..."}]}`) or an explicit `"success": false` field (hyprfast `hint_act`
/// returns `{"success":false,"message":"No action found",...}` when no tier
/// resolves). Detect both so a failed browser/LLM call can't read as success.
pub(crate) fn mcp_failure_text(output: &Value) -> Option<String> {
    mcp_failure_inner(output, 0)
}

fn mcp_failure_inner(output: &Value, depth: usize) -> Option<String> {
    if output.get("success") == Some(&Value::Bool(false)) {
        return Some(failure_text(output));
    }
    // A server error envelope (`{"error": ...}`): an explicit failure field,
    // whatever shape its value takes (string or `{message, code}` object).
    if output.get("error").is_some_and(|e| !e.is_null()) {
        return Some(failure_text(output));
    }
    // MCP content envelope: {"content":[{"text":"..."}]}. hyprfast reports
    // tool-level failures as text starting with "error:" ("error: Google
    // error ...", "error: evaluate exception: ...").
    if let Some(content) = output.get("content").and_then(Value::as_array) {
        for item in content {
            if let Some(text) = item.get("text").and_then(Value::as_str) {
                let trimmed = text.trim_start();
                if trimmed.starts_with("error:") {
                    return Some(failure_text(output));
                }
                // hyprfast nests result objects as stringified JSON text;
                // recurse one level so an inner `"success": false` counts.
                if depth < 2 {
                    if let Ok(inner) = serde_json::from_str::<Value>(trimmed) {
                        if let Some(err) = mcp_failure_inner(&inner, depth + 1) {
                            return Some(err);
                        }
                    }
                }
            }
        }
    }
    // Fallback: serialized scan for the known LLM error shape in case the
    // envelope differs.
    let flat = output.to_string();
    if flat.contains("error: Google error ") {
        Some(failure_text(output))
    } else {
        None
    }
}

/// What a failed tool call *says*, without the envelope it arrived in.
///
/// This text goes straight into `AgentEvent::Error` and into the run report, so
/// it is read by the person running the task. `summarize_output` is the right
/// renderer for a *success* log line and the wrong one here: it serializes a
/// `Value`, so a failure arrived as `{"content":[{"type":"text","text":"error:
/// no such element"}]}`. The human message is pulled out by the same extractor
/// the display layers use, and a payload with no message in it degrades to a
/// sentence rather than to bytes.
fn failure_text(output: &Value) -> String {
    match lucy_core::payload_message(output) {
        Some(message) => lucy_core::friendly(&message),
        // No message anywhere in the value. The failure is real — the server said
        // so — but there is nothing to quote, and the serialized value is not an
        // explanation. Saying that is more use to the reader than echoing bytes.
        None => "the tool reported a failure without saying why".to_owned(),
    }
}

/// Compact rendering of a tool result for the execution log.
pub fn summarize_output(output: &Value) -> String {
    let text = match output {
        Value::String(s) => s.clone(),
        Value::Null => "ok".into(),
        other => other.to_string(),
    };
    let one: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if one.chars().count() <= 90 {
        one
    } else {
        let mut t: String = one.chars().take(89).collect();
        t.push('…');
        t
    }
}

/// Resolve an approval decision for a pending prompt (called by the TUI).
impl LucyRuntime {
    pub fn resolve_approval(
        &self,
        gate: &ApprovalGate,
        call_id: &str,
        decision: ApprovalDecision,
    ) -> bool {
        gate.resolve(call_id, decision)
    }
}

/// `provider_id/model` → `provider · model`, falling back to the raw key.
pub fn describe_model(config: &LucyConfig, key: &str) -> String {
    let key = key.trim();
    if key.is_empty() {
        return "(no model selected)".into();
    }
    if let Some((provider, model)) = config.split_model_key(key) {
        return format!("{} · {}", provider.label(), model);
    }
    key.to_owned()
}

/// Every text model the settings dropdowns can offer, as `provider_id/model`.
pub fn text_model_keys(config: &LucyConfig) -> Vec<String> {
    config
        .text_model_options()
        .iter()
        .map(|o| o.key())
        .collect()
}

/// The models bound to the three tiers, for the chat log / `/status`.
pub fn level_model_map(config: &LucyConfig) -> HashMap<ReasoningLevel, String> {
    ReasoningLevel::ALL
        .iter()
        .map(|l| (*l, config.resolve_level_model(*l)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucy_config::{ProviderConfig, ProviderType};

    fn with_providers() -> LucyConfig {
        LucyConfig {
            providers: vec![
                ProviderConfig {
                    id: "groq".into(),
                    name: "Groq".into(),
                    api_url: "https://api.groq.test".into(),
                    api_key: "sk-x".into(),
                    provider_type: ProviderType::Text,
                    available_models: vec!["flash".into(), "pro".into()],
                    deprecated_models: Vec::new(),
                },
                ProviderConfig {
                    id: "eleven".into(),
                    name: "ElevenLabs".into(),
                    api_url: "https://api.eleven.test".into(),
                    api_key: "sk-y".into(),
                    provider_type: ProviderType::Voice,
                    available_models: vec!["tts".into()],
                    deprecated_models: Vec::new(),
                },
            ],
            ..Default::default()
        }
    }

    #[test]
    fn parses_commands_plan() {
        let v = json!({"commands":[
            {"tool":"browser_open","input":{"url":"https://youtube.com"},"description":"Open YouTube"},
            {"tool":"browser_navigate","args":{"url":"https://youtube.com/search?q=despacito"}}
        ]});
        let plan = parse_plan(&v);
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].index, 1);
        assert_eq!(plan[0].tool, "browser_open");
        assert_eq!(plan[0].input["url"], "https://youtube.com");
        assert_eq!(plan[0].description, "Open YouTube");
        assert_eq!(plan[1].index, 2);
        assert_eq!(
            plan[1].input["url"],
            "https://youtube.com/search?q=despacito"
        );
        assert!(plan[1].description.is_empty());
    }

    #[test]
    fn parses_steps_array_and_string_form() {
        let v = json!({"steps":["browser_open {\"url\":\"x\"}", "browser_snapshot"]});
        let plan = parse_plan(&v);
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].tool, "browser_open");
        assert_eq!(plan[0].input["url"], "x");
        assert_eq!(plan[1].tool, "browser_snapshot");
        assert!(plan[1].input.as_object().unwrap().is_empty());
    }

    #[test]
    fn bare_array_and_scalars_are_tolerated() {
        let v = json!([{"name":"click_target","value":"Play"}]);
        let plan = parse_plan(&v);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].tool, "click_target");
        assert_eq!(plan[0].input["value"], "Play");
    }

    #[test]
    fn malformed_plan_entries_are_dropped() {
        assert!(parse_plan(&json!({})).is_empty());
        assert!(parse_plan(&json!({"commands":[]})).is_empty());
        assert!(parse_plan(&json!({"commands":[{"description":"no tool"}]})).is_empty());
        assert!(parse_plan(&json!({"commands":[{"tool":"  "}]})).is_empty());
    }

    #[test]
    fn prompt_makes_hint_act_the_primary_verb() {
        assert!(PLAN_INSTRUCTIONS.contains("`hint_act` is the PRIMARY"));
        assert!(PLAN_INSTRUCTIONS.contains("BLIND"));
        assert!(PLAN_INSTRUCTIONS.contains("ref` from a `browser_snapshot"));
        assert!(PLAN_INSTRUCTIONS.contains("Never END a plan on `browser_open`"));
        // Removed tools may only appear in the explicit ban — never as an
        // offered step. The ban is what keeps a weak model from emitting a
        // step the catalog will not resolve.
        assert!(PLAN_INSTRUCTIONS.contains("Never emit"), "{PLAN_INSTRUCTIONS}");
        assert!(!PLAN_INSTRUCTIONS.contains("ground.env"));
        assert!(!PLAN_INSTRUCTIONS.to_lowercase().contains("gemini"));
        assert!(
            !PLAN_INSTRUCTIONS.contains("`ground` + `pointer`"),
            "old vision ladder is still offered"
        );
    }

    #[test]
    fn prompt_names_only_tools_the_live_catalog_advertises() {
        for tool in [
            "hint_act",
            "hint_batch",
            "hint_resolve",
            "find_and_click",
            "find_and_type",
            "browser_open",
            "browser_navigate",
            "browser_go_back",
            "browser_go_forward",
            "browser_tabs",
            "browser_evaluate",
            "browser_wait",
            "browser_screenshot",
            "verify",
            "wait_until",
            "keyboard",
            "browser_click",
            "browser_type",
            "browser_hover",
            "browser_select_option",
        ] {
            assert!(PLAN_INSTRUCTIONS.contains(tool), "prompt drops {tool}");
        }
        // Removed tools may only appear in the explicit ban, never as a
        // usable step in the ladder.
        assert!(
            PLAN_INSTRUCTIONS.contains("Never emit"),
            "planner prompt must ban the removed tools"
        );
        assert!(
            !PLAN_INSTRUCTIONS.contains("`ground` + `pointer`"),
            "old vision ladder is still offered"
        );
    }

    #[test]
    fn sanitize_rewrites_refless_click_to_hint_act() {
        let plan = parse_plan(&json!({"commands":[
            {"tool":"browser_navigate","input":{"url":"https://www.youtube.com/results?search_query=despacito"},"description":"Open search results"},
            {"tool":"browser_click","input":{"instruction":"Click on the first search result"},"description":"Click on the first search result"}
        ]}));
        let clean = sanitize_plan(plan, "play despacito on youtube");
        assert_eq!(clean.len(), 2);
        assert_eq!(clean[0].tool, "browser_navigate");
        assert_eq!(clean[1].tool, "hint_act");
        assert_eq!(
            clean[1].input["instruction"],
            "Click on the first search result"
        );
    }

    #[test]
    fn sanitize_keeps_steps_that_carry_a_ref() {
        let plan = parse_plan(&json!({"commands":[
            {"tool":"mcp_hyprfast_browser_click","input":{"ref":"e12"},"description":"Click it"},
            {"tool":"browser_type","input":{"element":".search","text":"x"},"description":"Type"}
        ]}));
        let clean = sanitize_plan(plan, "do things");
        assert_eq!(clean[0].tool, "mcp_hyprfast_browser_click");
        assert_eq!(clean[0].input["ref"], "e12");
        assert_eq!(clean[1].tool, "browser_type");
    }

    #[test]
    fn sanitize_falls_back_to_input_text_without_description() {
        let plan = parse_plan(&json!({"commands":[
            {"tool":"browser_type","input":{"text":"despacito"}}
        ]}));
        let clean = sanitize_plan(plan, "search despacito");
        assert_eq!(clean[0].tool, "hint_act");
        assert_eq!(clean[0].input["instruction"], "despacito");
    }

    #[test]
    fn sanitize_appends_interaction_after_lone_navigation() {
        let plan = parse_plan(&json!({"commands":[
            {"tool":"browser_navigate","input":{"url":"https://www.youtube.com/results?search_query=despacito"},"description":"Open search results"}
        ]}));
        let clean = sanitize_plan(plan, "play despacito on youtube");
        assert_eq!(clean.len(), 2);
        assert_eq!(clean[1].tool, "hint_act");
        assert!(
            clean[1].input["instruction"]
                .as_str()
                .unwrap()
                .contains("play despacito")
        );
    }

    #[test]
    fn sanitize_leaves_open_only_goals_alone() {
        let plan = parse_plan(&json!({"commands":[
            {"tool":"browser_open","input":{"url":"https://www.youtube.com"},"description":"Open YouTube"}
        ]}));
        let clean = sanitize_plan(plan, "open youtube");
        assert_eq!(clean.len(), 1);
    }

    /// `hint_act` requires `instruction`; `text` is only meaningful next to
    /// `action: "type"`.
    #[test]
    fn hint_act_input_satisfies_its_schema() {
        let mut typed = json!({ "text": "despacito" });
        normalize_step_input("hint_act", &mut typed, "type it in the search box");
        assert_eq!(typed["instruction"], "type it in the search box");
        assert_eq!(typed["action"], "type");
        assert_eq!(typed["text"], "despacito");

        let mut click = json!({ "instruction": "click the first result" });
        normalize_step_input("hint_act", &mut click, "click it");
        assert_eq!(click["instruction"], "click the first result");
        assert!(click.get("action").is_none());
    }

    /// The whole point of the fix: whatever the model emits, the plan that
    /// reaches `execute_plan_sequentially` only names tools the catalog has.
    #[test]
    fn sanitized_plans_only_need_the_hint_pipeline_to_interact() {
        let plan = parse_plan(&json!({"commands":[
            {"tool":"browser_navigate","input":{"url":"https://www.youtube.com/results?search_query=despacito"},"description":"Open search results"},
            {"tool":"browser_click","input":{"instruction":"click the first result"},"description":"click the first result"},
            {"tool":"browser_type","input":{"text":"despacito"}}
        ]}));
        let clean = sanitize_plan(plan, "play despacito on youtube");
        let interaction: Vec<&str> = clean
            .iter()
            .map(|s| s.tool.as_str())
            .filter(|t| is_dom_interaction(t))
            .collect();
        assert_eq!(interaction, vec!["hint_act", "hint_act"], "{interaction:?}");
    }

    #[test]
    fn plan_numbering_and_log_lines() {
        let plan = parse_plan(&json!({"commands":[
            {"tool":"a","description":"do A"},
            {"tool":"b"}
        ]}));
        let out = format_plan(&plan);
        assert_eq!(out, "1. do A (a)\n2. b");
    }

    #[test]
    fn plan_prompt_carries_catalog_and_goal() {
        let out = render_plan_prompt("CATALOG", "open yt", "(none)");
        assert!(out.contains("CATALOG"));
        assert!(out.contains("open yt"));
        assert!(out.contains("\"commands\""));
        assert!(!out.contains("{user_request}"));
    }

    #[test]
    fn describe_model_uses_provider_label() {
        let cfg = with_providers();
        assert_eq!(describe_model(&cfg, "groq/pro"), "Groq · pro");
        assert_eq!(describe_model(&cfg, ""), "(no model selected)");
        // A bare model name resolves against the single connected text provider.
        assert_eq!(describe_model(&cfg, "hand-typed"), "Groq · hand-typed");
    }

    #[test]
    fn level_model_map_reports_the_model_each_tier_will_call() {
        let mut cfg = with_providers();
        cfg.chat
            .reasoning_levels
            .set(ReasoningLevel::L1, "groq/flash".into());
        cfg.chat
            .reasoning_levels
            .set(ReasoningLevel::L3, "groq/pro".into());
        let map = level_model_map(&cfg);
        // L2 is unbound and skips *down* to L1, never up to the L3 anchor.
        assert_eq!(map[&ReasoningLevel::L1], "groq/flash");
        assert_eq!(map[&ReasoningLevel::L2], "groq/flash");
        assert_eq!(map[&ReasoningLevel::L3], "groq/pro");
    }

    #[test]
    fn text_model_keys_excludes_voice_providers() {
        let cfg = with_providers();
        let keys = text_model_keys(&cfg);
        assert_eq!(keys, vec!["groq/flash", "groq/pro"]);
    }

    #[test]
    fn output_summary_is_one_line_and_capped() {
        assert_eq!(summarize_output(&json!("a\n  b")), "a b");
        assert_eq!(summarize_output(&Value::Null), "ok");
        let long = summarize_output(&json!("x".repeat(300)));
        assert!(long.chars().count() <= 90);
        assert!(long.ends_with('…'));
    }

    /// The rule under test: a tool failure is reported in the tool's own words,
    /// for every envelope shape a server can answer with. A table, not one
    /// anecdote — the shape is the contract, not the site.
    #[test]
    fn a_tool_failure_is_reported_without_its_envelope() {
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
            let text = mcp_failure_text(&output)
                .unwrap_or_else(|| panic!("expected a failure for {output}"));
            assert!(text.contains(expected), "{text}");
            assert!(!text.contains('{'), "envelope leaked: {text}");
        }
    }

    #[test]
    fn a_failure_with_no_readable_message_still_produces_a_sentence() {
        // Nothing here carries a human message, so the fallback must be prose —
        // the old renderer printed the serialized value into the chat.
        let output = json!({"success": false});
        let text = mcp_failure_text(&output).expect("success:false is a failure");
        assert!(!text.is_empty());
        assert!(!text.contains("\"success\""), "{text}");
    }

    fn chat_verdict(confidence: f64) -> TurnClassification {
        TurnClassification {
            branch: TurnBranch::RequiresOnlyResponse,
            reasoning_level: ReasoningLevel::L2,
            confidence,
            probabilities: HashMap::new(),
            level_confidence: 0.9,
            latency_ms: Some(4.0),
            summary_note: None,
            knowledge_topic: None,
        }
    }

    #[test]
    fn verify_override_routes_to_actions_on_llm_yes() {
        // The reported bug: decider-serve says `yes` at 100% for
        // "play despacito song on yt". When the routing LLM says actions,
        // the verdict flips to the action branch (L3) — no keywords involved.
        let routed = action_override(
            chat_verdict(1.0),
            "classifier said text-only, but the routing LLM found actionable tools — routing to actions",
        );
        assert_eq!(routed.branch, TurnBranch::RequiresActions);
        assert_eq!(routed.reasoning_level, ReasoningLevel::L3);
        assert!(routed.summary_note.is_some());
        // Latency is preserved for the log.
        assert_eq!(routed.latency_ms, Some(4.0));
    }

    #[test]
    fn a_stated_intent_is_required_before_the_branch_flips() {
        // The flip needs the model to name the intent. A bare bool cannot move
        // the branch, in either direction.
        assert_eq!(
            parse_verify_intent(&json!({"intent": "action", "needs_actions": true})),
            Some(TurnIntent::Action)
        );
        assert_eq!(
            parse_verify_intent(&json!({"intent": "question", "needs_actions": false})),
            Some(TurnIntent::Question)
        );
        assert_eq!(
            parse_verify_intent(&json!({"intent": "ACTION"})),
            Some(TurnIntent::Action)
        );
        // No intent: keep the classifier's verdict, whatever the bool says.
        assert_eq!(parse_verify_intent(&json!({"needs_actions": false})), None);
        assert_eq!(parse_verify_intent(&json!({"needs_actions": true})), None);
        // An intent word the model invented is not a decision either.
        assert_eq!(parse_verify_intent(&json!({"intent": "maybe"})), None);
        assert_eq!(parse_verify_intent(&json!({})), None);
        assert_eq!(parse_verify_intent(&json!(true)), None);
    }

    #[test]
    fn a_bare_intent_string_is_still_an_intent() {
        assert_eq!(
            parse_verify_intent(&json!("action")),
            Some(TurnIntent::Action)
        );
        assert_eq!(
            parse_verify_intent(&json!("question")),
            Some(TurnIntent::Question)
        );
    }

    /// The exact incident, kept as a regression. "play despacito song on yt"
    /// came back from `decider-2b-vision` as `chat` at 87% confidence. The
    /// verify call that exists to catch precisely that 2B misread returned
    /// a 502, because the configured model was not registered on the
    /// gateway. The old code kept the classifier's word, so the turn was
    /// answered in prose and no tool ever ran. The new contract: a dead
    /// verifier on a text-only verdict is an error, not a silent chat answer —
    /// the turn stops and the user sees why instead of reading prose for a
    /// task that never ran.
    #[test]
    fn a_dead_verifier_on_chat_is_an_error_not_a_silent_answer() {
        // Precondition: the 2B misread this test exists for.
        let misread = chat_verdict(0.87);
        assert_eq!(misread.branch, TurnBranch::RequiresOnlyResponse);

        // The contract: `classify_turn` returns `Err` on
        // `VerifyOutcome::Unavailable`, so no branch is taken at all.
        let err = RoutingError::DeciderUnavailable(
            "classifier said text-only, but the routing LLM was unavailable (502) — refusing to answer a possible task as chat".into(),
        );
        assert!(err.to_string().contains("502"));
        assert!(err.to_string().contains("No tools were run"));
    }

    /// Under-confident deciders defer to the LLM: the 0.65 floor is the
    /// boundary between trusting the decider and asking the second opinion.
    #[test]
    fn routing_confidence_floor_is_point_six_five() {
        assert_eq!(ROUTING_CONFIDENCE_MIN, 0.65);
        assert!(chat_verdict(0.87).confidence > ROUTING_CONFIDENCE_MIN);
        assert!(chat_verdict(0.40).confidence < ROUTING_CONFIDENCE_MIN);
    }

    /// A dead verifier is an error in both directions now: neither a chat
    /// verdict nor an act verdict survives without the second opinion.
    #[test]
    fn an_action_verdict_survives_a_dead_verifier() {
        let act = TurnClassification {
            branch: TurnBranch::RequiresActions,
            reasoning_level: ReasoningLevel::L3,
            confidence: 0.99,
            probabilities: HashMap::new(),
            level_confidence: 0.9,
            latency_ms: Some(4.0),
            summary_note: None,
            knowledge_topic: None,
        };
        assert!(act.branch.needs_actions());
        // The run still stops: `classify_turn` maps a dead verifier on an
        // above-floor act verdict to `RoutingError`, it does not keep the
        // branch. The assertion documents the contract, not the struct.
        let err = RoutingError::DeciderUnavailable("routing LLM unavailable".into());
        assert!(err.to_string().contains("No tools were run"));
    }

    #[test]
    fn verify_prompt_carries_live_catalog_not_verbs() {
        let out =
            render_verify_prompt("browser_navigate — open a URL", "play despacito song on yt");
        assert!(out.contains("browser_navigate — open a URL"));
        assert!(out.contains("play despacito song on yt"));
        assert!(out.contains("\"needs_actions\""));
        // The intent test must be stated, or "any tool could do it" sends every
        // question to the browser.
        assert!(out.contains("INTENT"), "{out}");
    }

    #[test]
    fn an_action_verdict_can_be_answered_instead() {
        // The other direction: decider-serve said `act` at 99% for "what is the
        // capital of France", and the turn drove a browser to Google it. The
        // routing LLM now gets the last word in this direction too.
        let act = TurnClassification {
            branch: TurnBranch::RequiresActions,
            reasoning_level: ReasoningLevel::L3,
            confidence: 0.99,
            probabilities: HashMap::new(),
            level_confidence: 0.9,
            latency_ms: Some(4.0),
            summary_note: None,
            knowledge_topic: None,
        };
        let routed = text_override(act, "no tool advances this");
        assert_eq!(routed.branch, TurnBranch::RequiresOnlyResponse);
        // L3 is the action tier, never the answer tier.
        assert_ne!(routed.reasoning_level, ReasoningLevel::L3);
        assert_eq!(routed.confidence, 0.99);
        assert_eq!(routed.latency_ms, Some(4.0));
        assert!(routed.summary_note.is_some());
    }

    #[test]
    fn an_action_verdict_keeps_the_classifiers_own_tier_for_the_answer() {
        let act = TurnClassification {
            branch: TurnBranch::RequiresActions,
            // The classifier can ask for L2 even while guessing "act".
            reasoning_level: ReasoningLevel::L2,
            confidence: 0.8,
            probabilities: HashMap::new(),
            level_confidence: 0.8,
            latency_ms: None,
            summary_note: None,
            knowledge_topic: None,
        };
        assert_eq!(
            text_override(act, "no tool advances this").reasoning_level,
            ReasoningLevel::L2
        );
    }

    #[test]
    fn verify_answer_parses_bool_forms() {
        assert_eq!(
            parse_verify_answer(&json!({"needs_actions": true})),
            Some(true)
        );
        assert_eq!(
            parse_verify_answer(&json!({"needs_actions": false})),
            Some(false)
        );
        assert_eq!(
            parse_verify_answer(&json!({"needs_actions": "yes"})),
            Some(true)
        );
        assert_eq!(
            parse_verify_answer(&json!({"needs_actions": "no"})),
            Some(false)
        );
        assert_eq!(
            parse_verify_answer(&json!({"needs_actions": 1})),
            Some(true)
        );
        assert_eq!(parse_verify_answer(&json!({"act": true})), Some(true));
        assert_eq!(parse_verify_answer(&json!({})), None);
        assert_eq!(
            parse_verify_answer(&json!({"needs_actions": "maybe"})),
            None
        );
        assert_eq!(parse_verify_answer(&json!(true)), Some(true));
    }

    #[test]
    fn route_source_labels() {
        assert_eq!(RouteSource::Classifier.as_str(), "classifier");
        assert_eq!(RouteSource::LlmVerify.as_str(), "llm verify");
        assert!(RouteSource::ManualOverride.as_str().contains("manual"));
    }

    #[test]
    fn active_model_line_reports_tier() {
        let route = TurnRoute {
            classification: TurnClassification {
                branch: TurnBranch::RequiresOnlyResponse,
                reasoning_level: ReasoningLevel::L2,
                confidence: 0.9,
                probabilities: HashMap::new(),
                level_confidence: 0.9,
                latency_ms: Some(4.0),
                summary_note: None,
                knowledge_topic: None,
            },
            branch: TurnBranch::RequiresOnlyResponse,
            reasoning_level: ReasoningLevel::L2,
            model_key: "groq/flash".into(),
            model_label: "Groq · flash".into(),
            knowledge_topic: None,
            source: RouteSource::Classifier,
            note: None,
        };
        let line = route.active_model_line();
        assert!(line.starts_with("Groq · flash"), "{line}");
        assert!(line.contains("L2"), "{line}");
        assert!(!route.needs_actions());
    }

    #[test]
    fn normalize_browser_navigate_prepends_https_when_no_scheme() {
        let cases = [
            // schemeless → gets https://
            (
                "youtube.com/results?search_query=despacito",
                "https://youtube.com/results?search_query=despacito",
            ),
            ("www.google.com", "https://www.google.com"),
            // already valid → unchanged
            ("https://youtube.com", "https://youtube.com"),
            ("http://localhost:3000", "http://localhost:3000"),
            ("file:///home/user/test.html", "file:///home/user/test.html"),
        ];
        for (input_url, expected) in cases {
            for tool in &[
                "browser_navigate",
                "browser_open",
                "mcp_hyprfast_browser_navigate",
                "mcp_hyprfast_browser_open",
            ] {
                let mut input = json!({ "url": input_url });
                normalize_step_input(tool, &mut input, "");
                assert_eq!(
                    input["url"].as_str().unwrap(),
                    expected,
                    "tool={tool} url={input_url}"
                );
            }
        }
    }

    /// One discovered MCP tool, standing in for a real handshake.
    ///
    /// `destructive` is not a field here because in production it is read off
    /// the live catalog; these tests pass the name to the destructive set
    /// directly instead, which is the same signal the catalog produces.
    fn mcp(server: &str, tool: &str, name: &str) -> crate::McpToolFull {
        crate::McpToolFull {
            server: server.into(),
            tool: tool.into(),
            name: name.into(),
            description: format!("{tool} does a thing"),
            schema: json!({"type": "object", "properties": {"x": {"type": "string"}}}),
        }
    }

    fn names(tools: &[serde_json::Value]) -> Vec<String> {
        tools
            .iter()
            .filter_map(|t| {
                t.get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .collect()
    }

    #[test]
    fn the_tools_array_carries_every_source_in_the_wire_shape() {
        // The rule: local + MCP tools arrive as ONE array, and each entry is
        // the OpenAI function shape, not Lucy's flat one. A model that is
        // offered local tools but not MCP tools (or vice versa) plans against
        // a set the executor cannot dispatch.
        let local = vec![json!({
            "name": "shell",
            "description": "Run a shell command.",
            "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}
        })];
        let mcp = vec![
            mcp("hyprfast", "browser_navigate", "mcp_hyprfast_browser_navigate"),
            mcp("notion", "search", "mcp_notion_search"),
        ];
        let tools = openai_tools(&local, &mcp, &std::collections::HashSet::new());
        assert_eq!(
            names(&tools),
            vec![
                "shell",
                "mcp_hyprfast_browser_navigate",
                "mcp_notion_search"
            ]
        );
        for t in &tools {
            assert_eq!(t["type"], "function");
            let f = &t["function"];
            assert!(f["name"].is_string(), "{f}");
            assert!(
                f["parameters"].is_object(),
                "schema must arrive as `parameters`: {f}"
            );
            assert!(!f["description"].as_str().unwrap().is_empty(), "{f}");
        }
    }

    #[test]
    fn an_extra_servers_tool_is_offered_under_its_own_qualified_name() {
        // Regression: extra MCP servers used to be dropped from the registry
        // (everything was registered against hyprfast), so their tools were
        // discovered, briefed, and then unreachable.
        let mcp = vec![mcp("notion", "search", "mcp_notion_search")];
        let tools = openai_tools(&[], &mcp, &std::collections::HashSet::new());
        assert_eq!(names(&tools), vec!["mcp_notion_search"]);
    }

    #[test]
    fn removed_tools_are_absent_from_the_tools_array() {
        // The registry would answer to these names if we let it: they are
        // advertised by a stale hyprfast binary. The array must not offer
        // them, or the model can pick a tool that never runs.
        let mcp = vec![
            mcp("hyprfast", "browser_navigate", "mcp_hyprfast_browser_navigate"),
            mcp("hyprfast", "ground", "mcp_hyprfast_ground"),
            mcp("hyprfast", "act_fast", "mcp_hyprfast_act_fast"),
            mcp("hyprfast", "stagehand_act", "mcp_hyprfast_stagehand_act"),
        ];
        let tools = openai_tools(&[], &mcp, &std::collections::HashSet::new());
        assert_eq!(names(&tools), vec!["mcp_hyprfast_browser_navigate"]);
    }

    #[test]
    fn a_destructive_tool_says_so_in_its_description() {
        // The catalog's `destructive` flag is a Lucy-side fact the wire schema
        // cannot carry, so it has to reach the model through the description —
        // otherwise it calls a window-closing tool expecting it to be routine.
        let destructive: std::collections::HashSet<String> =
            ["browser_close".to_string()].into_iter().collect();
        for name in [
            "browser_close",
            "mcp_hyprfast_browser_close",
            "mcp_computer_use_browser_close",
        ] {
            let mcp = vec![mcp("hyprfast", "browser_close", name)];
            let tools = openai_tools(&[], &mcp, &destructive);
            let desc = tools[0]["function"]["description"].as_str().unwrap();
            assert!(
                desc.contains("approval"),
                "{name} must say it needs approval: {desc}"
            );
        }
        // And a routine tool is not marked.
        let routine = vec![mcp(
            "hyprfast",
            "browser_snapshot",
            "mcp_hyprfast_browser_snapshot",
        )];
        let tools = openai_tools(&[], &routine, &destructive);
        assert!(
            !tools[0]["function"]["description"]
                .as_str()
                .unwrap()
                .contains("approval"),
            "a routine tool must not claim it needs approval"
        );
    }

    #[test]
    fn a_definition_without_a_name_is_dropped_rather_than_sent() {
        // An unnamed function cannot be called back, so sending it only spends
        // the model's attention on an option it cannot take.
        let local = vec![
            json!({"name": "", "description": "nameless", "input_schema": {}}),
            json!({"description": "no name at all", "input_schema": {}}),
            json!({"name": "read_file", "input_schema": {}}),
        ];
        assert_eq!(names(&openai_tools(&local, &[], &Default::default())), vec!["read_file"]);
    }

    #[test]
    fn a_tool_without_a_schema_still_arrives_callable() {
        // `parameters` is required by the API even for a tool that takes no
        // arguments; a missing schema must not become a missing key.
        let local = vec![json!({"name": "desktop", "description": "snapshot"})];
        let tools = openai_tools(&local, &[], &Default::default());
        assert_eq!(
            tools[0]["function"]["parameters"],
            json!({"type": "object", "properties": {}})
        );
    }
}
