//! The two-speed agent loop, driven end to end with a stub model and a fake
//! hyprfast tool registry.
//!
//! Nothing here touches a browser, Decider-2B, or a real endpoint: the slow
//! lane is a [`StubProvider`] that counts its calls, and the fast lane is a set
//! of `Tool` impls registered under the real hyprfast names and driven by a
//! tiny screen script. That is enough to prove the properties the design rests
//! on — a healthy task costs 1–2 slow calls, a stuck objective buys exactly one
//! replan, an unmoving screen stops the loop instead of thrashing, and the act
//! cap plus the interrupt both bind.

use async_trait::async_trait;
use lucy_core::{AgentEvent, InterruptSignal, Tool, ToolContext};
use lucy_runtime::agent_loop::{
    AGENT_PLAN_INSTRUCTIONS, AGENT_REPLAN_INSTRUCTIONS, AgentBudget, AgentDeps, Objective,
    ObjectiveExit, parse_objectives, render_replan_prompt, run_agent_loop_with,
};
use lucy_runtime::fast_perception::{self, ActKind};
use lucy_tools::ToolRegistry;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// One shared default config, so the borrowed `AgentDeps.config` outlives each
/// harness call.
fn default_config() -> &'static lucy_config::LucyConfig {
    static CONFIG: OnceLock<lucy_config::LucyConfig> = OnceLock::new();
    CONFIG.get_or_init(lucy_config::LucyConfig::default)
}

// ---------------------------------------------------------------------------
// The scripted screen
// ---------------------------------------------------------------------------

/// The screen the fake hyprfast reports, and the predicates a `verify` call
/// will treat as satisfied while it is showing.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Frame {
    labels: Vec<String>,
    satisfied: Vec<String>,
}

fn frame(labels: &[&str], satisfied: &[&str]) -> Frame {
    Frame {
        labels: labels.iter().map(|s| (*s).to_owned()).collect(),
        satisfied: satisfied.iter().map(|s| (*s).to_owned()).collect(),
    }
}

/// A timeline of screens. `hint_snapshot` advances it; every other fast tool
/// reads wherever it currently is. A frame with a repeat count of 1 means the
/// next `hint_snapshot` moves on — that is how a test makes the screen *move*
/// (no thrash) or *hold* (thrash).
#[derive(Debug)]
struct Script {
    frames: Vec<(Frame, usize)>,
    cursor: AtomicUsize,
    seen: Mutex<Vec<String>>,
}

impl Script {
    fn new(frames: Vec<(Frame, usize)>) -> Arc<Self> {
        Arc::new(Self {
            frames,
            cursor: AtomicUsize::new(0),
            seen: Mutex::new(Vec::new()),
        })
    }

    /// Move one step along the timeline.
    fn advance(&self) -> Frame {
        let mut pos = self.cursor.fetch_add(1, Ordering::SeqCst);
        for (f, times) in &self.frames {
            if pos < *times {
                return f.clone();
            }
            pos -= times;
        }
        // Past the end: hold the last frame forever.
        self.frames.last().map(|(f, _)| f.clone()).unwrap_or(Frame {
            labels: Vec::new(),
            satisfied: Vec::new(),
        })
    }

    /// The frame as of the most recent `advance`, without moving. Before the
    /// first snapshot this is the first frame: nothing has been perceived yet,
    /// so the fake must not answer a later one.
    fn peek(&self) -> Frame {
        let pos = self.cursor.load(Ordering::SeqCst).saturating_sub(1);
        let mut left = pos;
        for (f, times) in &self.frames {
            if left < *times {
                return f.clone();
            }
            left -= times;
        }
        self.frames.last().map(|(f, _)| f.clone()).unwrap_or(Frame {
            labels: Vec::new(),
            satisfied: Vec::new(),
        })
    }

    fn record(&self, tool: &str) {
        self.seen
            .lock()
            .expect("script poisoned")
            .push(tool.to_owned());
    }

    fn calls(&self, tool: &str) -> usize {
        self.seen
            .lock()
            .expect("script poisoned")
            .iter()
            .filter(|t| *t == tool)
            .count()
    }
}

/// How one fake tool answers.
enum Behaviour {
    /// `hint_snapshot`: advance the timeline, report the frame's elements.
    Snapshot,
    /// `verify` / `wait_until`: answer the call's `query` against the current
    /// frame's `satisfied` set, without moving.
    Verdict,
    /// Interaction tools: report a successful self-resolving action.
    Act,
    /// Target grounding: a confident candidate.
    Resolve,
    /// `browser_open`: bring the faked CDP endpoint up, the way the real one
    /// does, then report success.
    Launch(Arc<AtomicBool>),
    /// Fail at the transport level.
    Fail(String),
}

struct FakeTool {
    name: &'static str,
    behaviour: Behaviour,
    script: Option<Arc<Script>>,
}

impl FakeTool {
    fn snapshot(name: &'static str, script: &Arc<Script>) -> Arc<dyn Tool> {
        Arc::new(Self {
            name,
            behaviour: Behaviour::Snapshot,
            script: Some(script.clone()),
        })
    }
    fn scripted(name: &'static str, behaviour: Behaviour, script: &Arc<Script>) -> Arc<dyn Tool> {
        Arc::new(Self {
            name,
            behaviour,
            script: Some(script.clone()),
        })
    }
}

#[async_trait]
impl Tool for FakeTool {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "fake hyprfast tool"
    }
    fn parameters_schema(&self) -> Value {
        json!({})
    }
    fn requires_approval(&self) -> bool {
        false
    }
    async fn execute(&self, input: Value, _ctx: ToolContext) -> anyhow::Result<Value> {
        if let Some(script) = &self.script {
            script.record(self.name);
        }
        let script = self.script.clone();
        Ok(match &self.behaviour {
            Behaviour::Snapshot => {
                let f = script.expect("snapshot needs a script").advance();
                snapshot_payload(&f)
            }
            Behaviour::Verdict => {
                let f = script.expect("verdict needs a script").peek();
                verdict_payload(&f, &input)
            }
            Behaviour::Act => json!({"success": true, "tier": "decider", "label": "target"}),
            Behaviour::Resolve => json!({"candidate": {"label": "target", "confidence": 0.9}}),
            Behaviour::Launch(up) => {
                up.store(true, Ordering::SeqCst);
                json!({"success": true, "url": input.get("url").cloned().unwrap_or(Value::Null)})
            }
            Behaviour::Fail(msg) => return Err(anyhow::anyhow!("{msg}")),
        })
    }
}

fn snapshot_payload(f: &Frame) -> Value {
    json!({
        "count": f.labels.len(),
        "via": "decider",
        "hints": f.labels.iter().map(|l| json!({"label": l, "tag": "button"})).collect::<Vec<_>>(),
    })
}

/// `verify` is honest: it says the predicate holds only while the current frame
/// lists it. `wait_until` reports the same, so an unchanging screen also times
/// out truthfully.
fn verdict_payload(f: &Frame, input: &Value) -> Value {
    let query = input
        .get("query")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let holds = f.satisfied.iter().any(|s| {
        let s = s.to_ascii_lowercase();
        query.contains(&s) || s.contains(&query)
    });
    json!({"success": true, "result": if holds { "success" } else { "failure" }})
}

/// A registry whose `browser_navigate` and `location.hostname` behave like a
/// real page: navigating switches the host, and `browser_evaluate` answers from
/// the caller's map. Lets a test exercise PHASE 0's site landing for real.
fn registry_navigable(
    script: &Arc<Script>,
    js_result: Arc<Mutex<HashMap<String, bool>>>,
    host: Arc<Mutex<String>>,
) -> ToolRegistry {
    let mut r = registry_with_js_host(script, js_result, host.clone());
    r.register_arc(Arc::new(NavTool {
        host,
        visited: Arc::new(Mutex::new(Vec::new())),
    }));
    r
}

/// [`registry_with_js`] whose `browser_evaluate` also answers
/// `location.hostname`, which PHASE 0 uses to decide whether it is already on
/// the goal's site. Without it PHASE 0 navigates on every run.
fn registry_with_js_host(
    script: &Arc<Script>,
    js_result: Arc<Mutex<HashMap<String, bool>>>,
    host: Arc<Mutex<String>>,
) -> ToolRegistry {
    let mut r = registry(script);
    r.register_arc(Arc::new(JsProbeTool {
        results: js_result,
        host: Some(host),
    }));
    r
}

struct NavTool {
    host: Arc<Mutex<String>>,
    visited: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl Tool for NavTool {
    fn name(&self) -> &str {
        "browser_navigate"
    }
    fn description(&self) -> &str {
        "fake browser_navigate"
    }
    fn parameters_schema(&self) -> Value {
        json!({})
    }
    fn requires_approval(&self) -> bool {
        false
    }
    async fn execute(&self, input: Value, _ctx: ToolContext) -> anyhow::Result<Value> {
        let url = input
            .get("url")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_owned();
        self.visited
            .lock()
            .expect("visited poisoned")
            .push(url.clone());
        // "https://en.wikipedia.org/wiki/Main_Page" -> "en.wikipedia.org"
        if let Some(rest) = url.split("://").nth(1) {
            let host = rest.split(['/', '?', '#']).next().unwrap_or_default();
            *self.host.lock().expect("host poisoned") = host.to_owned();
        }
        Ok(json!({"url": url}))
    }
}

/// A registry wired to one [`Script`], registered under the real hyprfast
/// names the fast lane calls.
fn registry(script: &Arc<Script>) -> ToolRegistry {
    let mut r = ToolRegistry::new();
    r.register_arc(FakeTool::snapshot("hint_snapshot", script));
    r.register_arc(FakeTool::scripted("verify", Behaviour::Verdict, script));
    r.register_arc(FakeTool::scripted("wait_until", Behaviour::Verdict, script));
    r.register_arc(FakeTool::scripted("hint_act", Behaviour::Act, script));
    r.register_arc(FakeTool::scripted("find", Behaviour::Resolve, script));
    r.register_arc(FakeTool::scripted(
        "hint_resolve",
        Behaviour::Resolve,
        script,
    ));
    r.register_arc(FakeTool::scripted("find_and_click", Behaviour::Act, script));
    r.register_arc(FakeTool::scripted("find_and_type", Behaviour::Act, script));
    r
}

fn plan_two() -> Value {
    json!({"objectives":[
        {"description":"Click the first video result",
         "success_check":"the video is playing",
         "suggested_action":"click the first video result"},
        {"description":"Confirm playback started",
         "success_check":"the player shows a pause button",
         "suggested_action":"click the pause button"}
    ]})
}

/// A registry whose tools all fail: the loop must degrade, not hang or claim
/// success.
fn broken_registry() -> ToolRegistry {
    let mut r = ToolRegistry::new();
    for name in [
        "hint_snapshot",
        "verify",
        "wait_until",
        "hint_act",
        "find",
        "hint_resolve",
        "find_and_click",
        "find_and_type",
    ] {
        r.register_arc(Arc::new(FakeTool {
            name,
            behaviour: Behaviour::Fail("CDP unreachable".into()),
            script: None,
        }));
    }
    r
}

fn harness<'a>(
    stub: &'a Arc<lucy_agent::testing::StubProvider>,
    registry: &'a ToolRegistry,
    budget: AgentBudget,
) -> (AgentDeps<'a>, InterruptSignal) {
    harness_with_probe(stub, registry, budget, browser_up())
}

/// A registry whose `browser_evaluate` answers from a caller-supplied
/// predicate, so a test can decide what the page's own state actually is
/// independently of what the scripted screen claims.
fn registry_with_js(
    script: &Arc<Script>,
    js_result: Arc<Mutex<HashMap<String, bool>>>,
) -> ToolRegistry {
    let mut r = registry(script);
    r.register_arc(Arc::new(JsProbeTool {
        results: js_result,
        host: None,
    }));
    r
}

#[async_trait]
impl Tool for JsProbeTool {
    fn name(&self) -> &str {
        "browser_evaluate"
    }
    fn description(&self) -> &str {
        "fake browser_evaluate"
    }
    fn parameters_schema(&self) -> Value {
        json!({})
    }
    fn requires_approval(&self) -> bool {
        false
    }
    async fn execute(&self, input: Value, _ctx: ToolContext) -> anyhow::Result<Value> {
        let expr = input
            .get("expression")
            .or_else(|| input.get("js"))
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_owned();
        // PHASE 0 asks for the bare `location.hostname`; a probe asks a question
        // ABOUT the hostname. Only the bare form is answered from the host
        // state, so a probe still goes through the caller's answer map.
        let bare_hostname = expr.trim().trim_end_matches(';') == "location.hostname";
        if bare_hostname && let Some(host) = self.host.as_ref() {
            let h = host.lock().expect("host poisoned").clone();
            return Ok(json!({"result": {"type": "string", "value": h}}));
        }
        // Match on a distinctive substring so one fake can answer several
        // different probes.
        let answer = self
            .results
            .lock()
            .expect("js results poisoned")
            .iter()
            .find(|(needle, _)| expr.contains(needle.as_str()))
            .map(|(_, v)| *v)
            .unwrap_or(false);
        Ok(json!({"result": {"type": "boolean", "value": answer}}))
    }
}

struct JsProbeTool {
    results: Arc<Mutex<HashMap<String, bool>>>,
    /// When set, `location.hostname` is answered from here — PHASE 0 uses that
    /// expression to decide whether the browser is already on the goal's site.
    host: Option<Arc<Mutex<String>>>,
}

fn harness_with_probe<'a>(
    stub: &'a Arc<lucy_agent::testing::StubProvider>,
    registry: &'a ToolRegistry,
    budget: AgentBudget,
    probe: lucy_runtime::fast_perception::CdpProbe,
) -> (AgentDeps<'a>, InterruptSignal) {
    let interrupt = InterruptSignal::new();
    let deps = AgentDeps {
        knowledge: None,
        page_observations: Default::default(),
        provider: stub.as_ref(),
        registry,
        budget,
        interrupt: interrupt.clone(),
        events: None,
        model_key: "stub/model".into(),
        approval: None,
        cdp_probe: Some(probe),
        destructive_tools: Default::default(),
        config: default_config(),
    };
    (deps, interrupt)
}

fn deps_with_events<'a>(
    stub: &'a Arc<lucy_agent::testing::StubProvider>,
    registry: &'a ToolRegistry,
    budget: AgentBudget,
    tx: Option<tokio::sync::mpsc::UnboundedSender<AgentEvent>>,
) -> AgentDeps<'a> {
    AgentDeps {
        knowledge: None,
        page_observations: Default::default(),
        provider: stub.as_ref(),
        registry,
        budget,
        interrupt: InterruptSignal::new(),
        events: tx,
        model_key: "stub/model".into(),
        approval: None,
        cdp_probe: Some(browser_up()),
        destructive_tools: Default::default(),
        config: default_config(),
    }
}

/// A CDP probe that always answers "a browser is already listening", so the
/// shared harness never opens a socket or launches anything: these tests are
/// about the loop, not the bootstrap. The bootstrap tests below inject
/// `browser_down()` instead.
fn browser_up() -> lucy_runtime::fast_perception::CdpProbe {
    lucy_runtime::fast_perception::probe_fn(|| async { true })
}

fn browser_down() -> lucy_runtime::fast_perception::CdpProbe {
    lucy_runtime::fast_perception::probe_fn(|| async { false })
}

/// A probe that reads the endpoint's state, so a `browser_open` that brings it
/// up is visible to the very next probe — the real sequence.
fn endpoint_state(up: Arc<AtomicBool>) -> lucy_runtime::fast_perception::CdpProbe {
    lucy_runtime::fast_perception::probe_fn(move || {
        let up = up.clone();
        async move { up.load(Ordering::SeqCst) }
    })
}

// ---------------------------------------------------------------------------
// The core claim: a healthy task is 1-2 slow calls, however many fast steps.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_healthy_task_costs_one_slow_call_and_completes() {
    // Objective 1's check is false on the first screen and true after the
    // interaction, so the loop must act and re-verify — not assume.
    let script = Script::new(vec![
        (frame(&["Search", "Queue"], &[]), 1),
        (frame(&["Pause"], &["the video is playing"]), 1),
        (frame(&["Pause"], &["the player shows a pause button"]), 40),
    ]);
    let registry = registry(&script);
    let stub = lucy_agent::testing::StubProvider::new().push_json(plan_two());
    let (deps, _intr) = harness(&stub, &registry, AgentBudget::default());

    let out = run_agent_loop_with(&deps, "play despacito")
        .await
        .expect("loop runs");

    assert_eq!(
        stub.calls(),
        1,
        "a healthy task spends exactly one slow call: {:?}",
        stub.purposes()
    );
    assert_eq!(stub.purposes(), vec!["agent_plan"]);
    assert!(out.complete, "everything verified: {out}");
    assert_eq!(out.stats.objectives_done, 2);
    assert!(out.summary.contains("Goal completed"), "{out}");
    assert!(
        out.stats.fast_calls >= 4,
        "many fast calls, one slow call: {:?}",
        out.stats
    );
    assert_eq!(out.stats.recoveries, 0, "no replan was needed");
    // Three interactions for two objectives: the first one acts, re-perceives
    // (the page only reports playback on the next snapshot), and acts again
    // before its check goes green. The blind plan would have had exactly one
    // frozen click and no way to notice it did nothing.
    assert_eq!(
        script.calls("hint_act"),
        3,
        "act, re-observe, re-act — not one frozen click"
    );
    assert!(
        script.calls("hint_snapshot") >= 3,
        "the screen was re-observed after every act: {}",
        script.calls("hint_snapshot")
    );
}

#[tokio::test]
async fn a_healthy_task_never_spends_more_than_two_slow_calls() {
    let script = Script::new(vec![(
        frame(
            &["Pause"],
            &["the video is playing", "the player shows a pause button"],
        ),
        40,
    )]);
    let registry = registry(&script);
    let stub = lucy_agent::testing::StubProvider::new().push_json(plan_two());
    let (deps, _intr) = harness(&stub, &registry, AgentBudget::default());
    let out = run_agent_loop_with(&deps, "play despacito")
        .await
        .expect("loop runs");
    assert!(out.complete, "{out}");
    assert!(
        stub.calls() <= 2,
        "the latency thesis caps a healthy task at 2 slow calls, saw {}",
        stub.calls()
    );
}

#[tokio::test]
async fn an_uncertain_verdict_gets_one_wait_and_one_reverify() {
    // The check is false first (so one wait is spent), then true.
    let script = Script::new(vec![
        (frame(&["Search"], &[]), 1),
        (
            frame(
                &["Pause"],
                &["the video is playing", "the player shows a pause button"],
            ),
            40,
        ),
    ]);
    let registry = registry(&script);
    let stub = lucy_agent::testing::StubProvider::new().push_json(plan_two());
    let (deps, _intr) = harness(&stub, &registry, AgentBudget::default());
    let out = run_agent_loop_with(&deps, "play despacito")
        .await
        .expect("loop runs");
    assert!(out.complete, "{out}");
    assert_eq!(
        stub.calls(),
        1,
        "the wait absorbed the uncertainty, no replan"
    );
    assert!(
        script.calls("wait_until") <= 2,
        "at most one wait per objective, saw {}",
        script.calls("wait_until")
    );
}

// ---------------------------------------------------------------------------
// Deviation buys exactly one replan.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_stuck_objective_triggers_a_replan_and_the_loop_reports_partial() {
    // The screen never changes, so the anti-thrash guard escalates instead of
    // clicking forever, and the replan cannot help either.
    let script = Script::new(vec![(frame(&["Search", "Queue"], &[]), 100_000)]);
    let registry = registry(&script);
    let stub = lucy_agent::testing::StubProvider::new()
        .push_json(plan_two())
        .push_json(json!({"objectives":[
            {"description":"Open the result directly",
             "success_check":"the video is playing",
             "suggested_action":"click the video thumbnail"}
        ]}));
    // One recovery allowed, so "exactly one replan" is exact rather than
    // "however many the default allows".
    let mut budget = AgentBudget::default();
    budget.max_recoveries = 1;
    budget.max_act_steps = 12;
    let (deps, _intr) = harness(&stub, &registry, budget);

    let out = run_agent_loop_with(&deps, "play despacito")
        .await
        .expect("loop runs");

    assert_eq!(
        stub.calls(),
        2,
        "plan + exactly one replan: {:?}",
        stub.purposes()
    );
    assert_eq!(stub.purposes(), vec!["agent_plan", "agent_replan"]);
    assert_eq!(out.stats.recoveries, 1);
    assert!(!out.complete, "the goal never verified: {out}");
    assert!(out.summary.contains("Partial result"), "{out}");
    assert!(
        out.summary.contains("Stopped because"),
        "a partial report must say why it stopped: {out}"
    );
}

#[tokio::test]
async fn a_replan_that_succeeds_is_reported_as_a_completion() {
    // Every objective is stuck on an unchanging screen until the replan hands
    // back one objective whose check the scripted screen then satisfies.
    // The last frame lists the goal's own words so the whole-goal check can
    // corroborate the replan. A replan that genuinely reaches the end state is
    // a real recovery and must be reported as one — but it is the goal-level
    // check, not the per-objective ratio, that distinguishes "recovered" from
    // "replanned into easier steps that did not reach the goal".
    let script = Script::new(vec![(
        frame(&["Pause"], &["the end state is reached", "finish the task"]),
        100_000,
    )]);
    let registry = registry(&script);
    let stub = lucy_agent::testing::StubProvider::new()
        .push_json(json!({"objectives":[
            {"description":"Stuck step","success_check":"never happens",
             "suggested_action":"click nothing"},
            {"description":"Also stuck","success_check":"never happens either",
             "suggested_action":"click nothing"}
        ]}))
        .push_json(json!({"objectives":[
            {"description":"Reach the end state","success_check":"the end state is reached",
             "suggested_action":"click done"}
        ]}));
    let mut budget = AgentBudget::default();
    budget.max_act_steps = 16;
    let (deps, _intr) = harness(&stub, &registry, budget);

    let out = run_agent_loop_with(&deps, "finish the task")
        .await
        .expect("loop runs");
    assert_eq!(
        stub.calls(),
        2,
        "one replan, then done: {:?}",
        stub.purposes()
    );
    assert!(out.complete, "the replan path reached the end: {out}");
    assert!(out.summary.contains("Goal completed"), "{out}");
}

#[tokio::test]
async fn the_replan_budget_is_respected() {
    // Replans keep failing, so the loop must stop buying slow calls.
    let script = Script::new(vec![(frame(&["Same"], &[]), 100_000)]);
    let registry = registry(&script);
    let stub = lucy_agent::testing::StubProvider::new()
        .push_json(json!({"objectives":[
            {"description":"A","success_check":"a1","suggested_action":"click a"},
            {"description":"B","success_check":"b1","suggested_action":"click b"},
            {"description":"C","success_check":"c1","suggested_action":"click c"}
        ]}))
        .push_json(json!({"objectives":[
            {"description":"A2","success_check":"a1","suggested_action":"click a"}
        ]}))
        .push_json(json!({"objectives":[
            {"description":"B2","success_check":"b1","suggested_action":"click b"}
        ]}));
    let mut budget = AgentBudget::default();
    budget.max_act_steps = 20;
    let (deps, _intr) = harness(&stub, &registry, budget);

    let out = run_agent_loop_with(&deps, "impossible")
        .await
        .expect("loop runs");
    // 1 plan + at most max_recoveries (2) replans, and never a 4th slow call.
    assert_eq!(stub.calls(), 3, "{:?}", stub.purposes());
    assert!(out.stats.recoveries <= 2, "{out}");
    assert!(out.summary.contains("Partial result"), "{out}");
}

#[tokio::test]
async fn replan_on_failure_off_stops_after_the_plan() {
    let script = Script::new(vec![(frame(&["Same"], &[]), 100_000)]);
    let registry = registry(&script);
    let stub = lucy_agent::testing::StubProvider::new().push_json(plan_two());
    let mut budget = AgentBudget::default();
    budget.replan_on_failure = false;
    budget.max_act_steps = 12;
    let (deps, _intr) = harness(&stub, &registry, budget);

    let out = run_agent_loop_with(&deps, "play despacito")
        .await
        .expect("loop runs");
    assert_eq!(
        stub.calls(),
        1,
        "no replan was permitted: {:?}",
        stub.purposes()
    );
    assert!(!out.complete, "{out}");
    assert!(out.summary.contains("Partial result"), "{out}");
}

#[tokio::test]
async fn an_empty_replan_reply_ends_the_run_instead_of_looping() {
    // The model saying "impossible from here" must be accepted, not retried.
    let script = Script::new(vec![(frame(&["Same"], &[]), 100_000)]);
    let registry = registry(&script);
    let stub = lucy_agent::testing::StubProvider::new()
        .push_json(plan_two())
        .push_json(json!({"objectives": []}));
    let mut budget = AgentBudget::default();
    budget.max_act_steps = 12;
    let (deps, _intr) = harness(&stub, &registry, budget);

    let out = run_agent_loop_with(&deps, "play despacito")
        .await
        .expect("loop runs");
    assert_eq!(stub.calls(), 2, "one replan, then it accepted 'give up'");
    assert!(!out.complete, "{out}");
}

// ---------------------------------------------------------------------------
// Anti-thrash: the fingerprint must actually stop repeats.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn an_unmoving_screen_stops_the_loop_instead_of_repeating() {
    let script = Script::new(vec![(frame(&["Same"], &[]), 100_000)]);
    let registry = registry(&script);
    let stub = lucy_agent::testing::StubProvider::new().push_json(json!({"objectives":[
        {"description":"Never lands","success_check":"never",
         "suggested_action":"click the same thing"}
    ]}));
    // Replanning disabled so this measures only the fast-lane repeat guard, and
    // a huge act budget so only the fingerprint can stop it.
    let mut budget = AgentBudget {
        replan_on_failure: false,
        ..AgentBudget::default()
    };
    budget.max_act_steps = 1_000;
    let (deps, _intr) = harness(&stub, &registry, budget);

    let out = run_agent_loop_with(&deps, "thrash")
        .await
        .expect("loop runs");
    assert!(
        script.calls("hint_act") <= 2,
        "an unchanged fingerprint must bound the repeats, saw {} acts out of a 1000-step budget",
        script.calls("hint_act")
    );
    assert!(!out.complete, "{out}");
    assert!(
        out.summary.contains("screen did not change"),
        "the report must name the thrash: {out}"
    );
}

#[tokio::test]
async fn a_changing_screen_is_never_mistaken_for_thrash() {
    // Each attempt sees a different page, so the repeat counter stays at zero
    // and the loop keeps acting.
    let script = Script::new(vec![
        (frame(&["One"], &[]), 1),
        (frame(&["Two"], &[]), 1),
        (frame(&["Three"], &["the video is playing"]), 1),
        (frame(&["Four"], &["the video is playing"]), 40),
    ]);
    let registry = registry(&script);
    let stub = lucy_agent::testing::StubProvider::new().push_json(json!({"objectives":[
        {"description":"Play it","success_check":"the video is playing",
         "suggested_action":"click play"}
    ]}));
    let (deps, _intr) = harness(&stub, &registry, AgentBudget::default());
    let out = run_agent_loop_with(&deps, "play").await.expect("loop runs");
    assert!(out.complete, "{out}");
    assert!(
        script.calls("hint_act") >= 2,
        "a moving screen earns retries, saw {}",
        script.calls("hint_act")
    );
}

// ---------------------------------------------------------------------------
// Bounds: the act cap and the interrupt.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_total_act_step_cap_is_honored() {
    let frames: Vec<(Frame, usize)> = (0..400)
        .map(|i| (frame(&[&format!("page-{i}")], &[]), 1))
        .collect();
    let script = Script::new(frames);
    let registry = registry(&script);
    let stub = lucy_agent::testing::StubProvider::new().push_json(json!({"objectives":[
        {"description":"A","success_check":"never","suggested_action":"click a"},
        {"description":"B","success_check":"never","suggested_action":"click b"},
        {"description":"C","success_check":"never","suggested_action":"click c"},
        {"description":"D","success_check":"never","suggested_action":"click d"}
    ]}));
    let mut budget = AgentBudget::default();
    budget.max_act_steps = 5;
    budget.replan_on_failure = false;
    let (deps, _intr) = harness(&stub, &registry, budget);

    let out = run_agent_loop_with(&deps, "runaway")
        .await
        .expect("loop runs");
    assert_eq!(
        out.stats.act_steps, 5,
        "the cap is reached and not exceeded"
    );
    assert_eq!(
        stub.calls(),
        1,
        "a spent act budget buys no replan: {:?}",
        stub.purposes()
    );
    assert!(out.summary.contains("Partial result"), "{out}");
    assert!(
        out.summary.contains("budget"),
        "the report must name the bound it hit: {out}"
    );
}

#[tokio::test]
async fn an_interrupt_before_the_plan_cancels_without_any_slow_call() {
    let script = Script::new(vec![(frame(&["A"], &["x"]), 40)]);
    let registry = registry(&script);
    let stub = lucy_agent::testing::StubProvider::new().push_json(plan_two());
    let (deps, intr) = harness(&stub, &registry, AgentBudget::default());
    intr.fire();

    let err = run_agent_loop_with(&deps, "cancel me")
        .await
        .expect_err("cancelled");
    assert_eq!(err.to_string(), "cancelled", "{err:#}");
    assert_eq!(stub.calls(), 0, "no planning happened");
    assert_eq!(script.calls("hint_snapshot"), 0, "no fast call happened");
}

/// Cancels the run from inside the first act, so the loop must bail at its
/// next interrupt check rather than working through the remaining objectives.
struct CancellingTool {
    interrupt: InterruptSignal,
}

#[async_trait]
impl Tool for CancellingTool {
    fn name(&self) -> &str {
        "hint_act"
    }
    fn description(&self) -> &str {
        "acts, then cancels the run"
    }
    fn parameters_schema(&self) -> Value {
        json!({})
    }
    fn requires_approval(&self) -> bool {
        false
    }
    async fn execute(&self, _input: Value, _ctx: ToolContext) -> anyhow::Result<Value> {
        self.interrupt.fire();
        Ok(json!({"success": true, "tier": "decider", "label": "acted"}))
    }
}

#[tokio::test]
async fn an_interrupt_mid_run_cancels_immediately() {
    let script = Script::new(vec![(frame(&["A"], &[]), 100_000)]);
    let mut registry = registry(&script);
    let intr = InterruptSignal::new();
    // Replace the interaction tool with one that cancels as it acts.
    registry.register_arc(Arc::new(CancellingTool {
        interrupt: intr.clone(),
    }));

    let stub = lucy_agent::testing::StubProvider::new().push_json(json!({"objectives":[
        {"description":"One","success_check":"never","suggested_action":"click one"},
        {"description":"Two","success_check":"never","suggested_action":"click two"},
        {"description":"Three","success_check":"never","suggested_action":"click three"},
        {"description":"Four","success_check":"never","suggested_action":"click four"}
    ]}));
    let deps = AgentDeps {
        knowledge: None,
        page_observations: Default::default(),
        provider: stub.as_ref(),
        registry: &registry,
        budget: AgentBudget {
            replan_on_failure: false,
            ..AgentBudget::default()
        },
        interrupt: intr,
        events: None,
        model_key: "stub/model".into(),
        approval: None,
        cdp_probe: Some(browser_up()),
        destructive_tools: Default::default(),
        config: default_config(),
    };

    match run_agent_loop_with(&deps, "cancel midway").await {
        Err(e) => assert_eq!(e.to_string(), "cancelled", "{e:#}"),
        Ok(outcome) => panic!("expected cancellation, got: {}", outcome.summary),
    }
    assert_eq!(stub.calls(), 1, "no replan was bought after the cancel");
}

// ---------------------------------------------------------------------------
// Degradation: no planner output, no fast lane at all.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn an_unusable_plan_reply_degrades_to_acting_on_the_goal() {
    let script = Script::new(vec![(
        frame(&["Pause"], &["play despacito on youtube"]),
        40,
    )]);
    let registry = registry(&script);
    let stub = lucy_agent::testing::StubProvider::new().push_json(json!({"nonsense": true}));
    let (deps, _intr) = harness(&stub, &registry, AgentBudget::default());

    let out = run_agent_loop_with(&deps, "play despacito on youtube")
        .await
        .expect("loop runs");
    assert_eq!(
        stub.calls(),
        1,
        "the fallback objective needs no second call"
    );
    assert!(out.complete, "{out}");
    assert_eq!(script.calls("hint_act"), 1);
}

#[tokio::test]
async fn a_planner_outage_degrades_without_failing_the_run() {
    // No JSON answer queued at all: the loop must still act on the goal.
    let script = Script::new(vec![(frame(&["Pause"], &["play despacito"]), 40)]);
    let registry = registry(&script);
    let stub = lucy_agent::testing::StubProvider::new();
    let (deps, _intr) = harness(&stub, &registry, AgentBudget::default());

    let out = run_agent_loop_with(&deps, "play despacito")
        .await
        .expect("a planner outage must not fail the run");
    assert_eq!(stub.calls(), 1, "the failed call is still counted");
    assert!(out.complete, "{out}");
}

#[tokio::test]
async fn a_dead_fast_lane_is_reported_not_silently_skipped() {
    let registry = broken_registry();
    let stub = lucy_agent::testing::StubProvider::new().push_json(plan_two());
    let mut budget = AgentBudget::default();
    budget.replan_on_failure = false;
    let (deps, _intr) = harness(&stub, &registry, budget);

    let out = run_agent_loop_with(&deps, "nothing works")
        .await
        .expect("loop runs");
    assert!(!out.complete, "{out}");
    assert!(out.summary.contains("Partial result"), "{out}");
    assert_eq!(out.stats.fast_failures, out.stats.fast_calls);
    assert_eq!(stub.calls(), 1);
}

#[tokio::test]
async fn a_registry_with_no_hyprfast_tools_at_all_terminates() {
    // Not one fast tool registered: every call fails at resolve time.
    let registry = ToolRegistry::new();
    let stub = lucy_agent::testing::StubProvider::new().push_json(plan_two());
    let mut budget = AgentBudget::default();
    budget.replan_on_failure = false;
    let (deps, _intr) = harness(&stub, &registry, budget);

    let out = run_agent_loop_with(&deps, "nothing registered")
        .await
        .expect("loop runs");
    assert!(!out.complete, "{out}");
    assert!(out.stats.fast_calls > 0, "the attempts were still counted");
    assert!(out.summary.contains("Partial result"), "{out}");
}

// ---------------------------------------------------------------------------
// Browser bootstrap: a web goal gets a browser, a desktop goal does not, and a
// launch is attempted at most once per run.
// ---------------------------------------------------------------------------

/// The shared registry plus a `browser_open` that brings `up` up, so the probe
/// sequence is exactly the real one: down, launch, up.
fn registry_with_browser(script: &Arc<Script>, up: Arc<AtomicBool>) -> ToolRegistry {
    let mut r = registry(script);
    r.register_arc(FakeTool::scripted(
        "browser_open",
        Behaviour::Launch(up),
        script,
    ));
    r
}

#[tokio::test]
async fn a_web_goal_launches_a_browser_before_the_first_fast_call() {
    // Nothing is listening when the run starts — the case that made every fast
    // call short-circuit on `CDP unreachable` before the loop ever opened a
    // browser.
    let up = Arc::new(AtomicBool::new(false));
    let script = Script::new(vec![(
        frame(
            &["Pause"],
            &["the video is playing", "the player shows a pause button"],
        ),
        40,
    )]);
    let registry = registry_with_browser(&script, up.clone());
    let stub = lucy_agent::testing::StubProvider::new().push_json(plan_two());
    let (deps, _intr) =
        harness_with_probe(&stub, &registry, AgentBudget::default(), endpoint_state(up));

    let out = run_agent_loop_with(&deps, "play despacito song on youtube")
        .await
        .expect("loop runs");

    assert_eq!(
        script.calls("browser_open"),
        1,
        "the loop opened a browser instead of failing every fast call"
    );
    assert!(out.complete, "{out}");
    assert_eq!(stub.calls(), 1, "the launch cost no slow call: {out}");
}

#[tokio::test]
async fn the_browser_is_launched_once_for_the_whole_run() {
    let up = Arc::new(AtomicBool::new(false));
    let script = Script::new(vec![(
        frame(
            &["Pause"],
            &[
                "one is done",
                "two is done",
                "three is done",
                "the last thing is done",
            ],
        ),
        40,
    )]);
    let registry = registry_with_browser(&script, up.clone());
    let stub = lucy_agent::testing::StubProvider::new().push_json(json!({"objectives":[
        {"description":"One","success_check":"one is done","suggested_action":"click one"},
        {"description":"Two","success_check":"two is done","suggested_action":"click two"},
        {"description":"Three","success_check":"three is done","suggested_action":"click three"},
        {"description":"Four","success_check":"the last thing is done",
         "suggested_action":"click four"}
    ]}));
    let (deps, _intr) =
        harness_with_probe(&stub, &registry, AgentBudget::default(), endpoint_state(up));

    let out = run_agent_loop_with(&deps, "search youtube for despacito")
        .await
        .expect("loop runs");

    assert!(out.complete, "{out}");
    assert_eq!(out.stats.objectives_done, 4, "{out}");
    assert_eq!(
        script.calls("browser_open"),
        1,
        "four objectives, one browser: the bootstrap is once per run"
    );
}

#[tokio::test]
async fn a_launch_that_does_not_bring_cdp_up_is_not_retried() {
    // The tool answers, but the endpoint stays down. The launch must not loop:
    // the fast calls carry on and the existing reporting owns the failure.
    let up = Arc::new(AtomicBool::new(false));
    let script = Script::new(vec![(frame(&["Pause"], &["the video is playing"]), 40)]);
    let registry = registry_with_browser(&script, up.clone());
    let stub = lucy_agent::testing::StubProvider::new().push_json(json!({"objectives":[
        {"description":"One","success_check":"the video is playing",
         "suggested_action":"click one"},
        {"description":"Two","success_check":"the video is playing",
         "suggested_action":"click two"}
    ]}));
    let mut budget = AgentBudget::default();
    budget.max_act_steps = 12;
    let (deps, _intr) = harness_with_probe(&stub, &registry, budget, browser_down());

    let out = run_agent_loop_with(&deps, "play despacito on youtube")
        .await
        .expect("loop runs");

    assert_eq!(script.calls("browser_open"), 1, "{out}");
    assert!(
        out.complete,
        "a failed bootstrap is not fatal to the fast lane: {out}"
    );
}

#[tokio::test]
async fn a_live_browser_is_never_relaunched() {
    let up = Arc::new(AtomicBool::new(false));
    let script = Script::new(vec![(
        frame(
            &["Pause"],
            &["the video is playing", "the player shows a pause button"],
        ),
        40,
    )]);
    let registry = registry_with_browser(&script, up.clone());
    let stub = lucy_agent::testing::StubProvider::new().push_json(plan_two());
    let (deps, _intr) = harness_with_probe(&stub, &registry, AgentBudget::default(), browser_up());

    let out = run_agent_loop_with(&deps, "play despacito on youtube")
        .await
        .expect("loop runs");

    assert!(out.complete, "{out}");
    assert_eq!(
        script.calls("browser_open"),
        0,
        "a browser was already there — launching a second one would be a new window"
    );
}

#[tokio::test]
async fn a_desktop_goal_never_launches_a_browser() {
    let up = Arc::new(AtomicBool::new(false));
    let script = Script::new(vec![(
        frame(&["Window"], &["the windows are arranged"]),
        40,
    )]);
    let registry = registry_with_browser(&script, up.clone());
    let stub = lucy_agent::testing::StubProvider::new().push_json(json!({"objectives":[
        {"description":"Arrange the windows","success_check":"the windows are arranged",
         "suggested_action":"arrange the windows"}
    ]}));
    let (deps, _intr) =
        harness_with_probe(&stub, &registry, AgentBudget::default(), browser_down());

    let out = run_agent_loop_with(&deps, "arrange the three windows side by side")
        .await
        .expect("loop runs");

    assert!(out.complete, "{out}");
    assert_eq!(
        script.calls("browser_open"),
        0,
        "nothing here is web work, so nothing may be launched"
    );
}

#[tokio::test]
async fn the_launch_is_a_fast_lane_call_and_is_said_out_loud_once() {
    // The screen never moves, so the first objective burns several attempts:
    // that is what makes "announced once per run" a real claim.
    let up = Arc::new(AtomicBool::new(false));
    let script = Script::new(vec![(frame(&["Same"], &[]), 100_000)]);
    let registry = registry_with_browser(&script, up.clone());
    let stub = lucy_agent::testing::StubProvider::new().push_json(json!({"objectives":[
        {"description":"One","success_check":"never happens","suggested_action":"click one"},
        {"description":"Two","success_check":"never happens either","suggested_action":"click two"}
    ]}));
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AgentEvent>();
    let interrupt = InterruptSignal::new();
    let mut budget = AgentBudget::default();
    budget.replan_on_failure = false;
    let deps = AgentDeps {
        knowledge: None,
        page_observations: Default::default(),
        provider: stub.as_ref(),
        registry: &registry,
        budget,
        interrupt,
        events: Some(tx.clone()),
        model_key: "stub/model".into(),
        approval: None,
        cdp_probe: Some(endpoint_state(up)),
        destructive_tools: Default::default(),
        config: default_config(),
    };

    let out = run_agent_loop_with(&deps, "play despacito on youtube")
        .await
        .expect("loop runs");
    drop(tx);

    let mut statuses = Vec::new();
    let mut started = 0usize;
    while let Ok(event) = rx.try_recv() {
        match event {
            AgentEvent::Status { message } => statuses.push(message),
            AgentEvent::ToolStarted { name, .. } if name.ends_with("browser_open") => started += 1,
            _ => {}
        }
    }
    assert_eq!(started, 1, "one launch, bracketed like any other fast call");
    let launches = statuses
        .iter()
        .filter(|s| s.contains("no browser was listening"))
        .count();
    assert_eq!(
        launches, 1,
        "a launch is announced once per run, not once per attempt: {statuses:?}"
    );
    assert!(
        out.stats.fast_calls as usize > script.calls("hint_snapshot"),
        "the launch is counted in the fast lane: {out}"
    );
}

// ---------------------------------------------------------------------------
// Observability: events stream, and the split is reported.
// ---------------------------------------------------------------------------

/// A `hint_snapshot` reporting the real shape hyprfast returns: a session-local
/// label next to the element's accessible name.
struct NamedScreen {
    hints: Vec<(String, String)>,
}

#[async_trait]
impl Tool for NamedScreen {
    fn name(&self) -> &str {
        "hint_snapshot"
    }
    fn description(&self) -> &str {
        "fake screen with real hint labels and names"
    }
    fn parameters_schema(&self) -> Value {
        json!({})
    }
    fn requires_approval(&self) -> bool {
        false
    }
    async fn execute(&self, _input: Value, _ctx: ToolContext) -> anyhow::Result<Value> {
        Ok(json!({
            "count": self.hints.len(),
            "via": "decider",
            "hints": self.hints.iter()
                .map(|(label, name)| json!({"label": label, "name": name}))
                .collect::<Vec<_>>(),
        }))
    }
}

#[tokio::test]
async fn what_the_planner_is_told_about_the_screen_is_names_not_ordinals() {
    // `A, S, D, F, G, H, J, K` told the replanner nothing. What it needs is
    // `despacito (S)`.
    let script = Script::new(vec![(frame(&["Pause"], &["the video is playing"]), 40)]);
    let mut registry = registry(&script);
    registry.register_arc(Arc::new(NamedScreen {
        hints: vec![
            ("S".into(), "despacito".into()),
            ("Y".into(), "Luis Fonsi - Despacito ft. Daddy Yankee".into()),
        ],
    }));
    let stub = lucy_agent::testing::StubProvider::new().push_json(json!({"objectives":[
        {"description":"Play it","success_check":"the video is playing",
         "suggested_action":"click the first video result"}
    ]}));
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AgentEvent>();
    let deps = deps_with_events(&stub, &registry, AgentBudget::default(), Some(tx.clone()));

    let out = run_agent_loop_with(&deps, "play despacito song on youtube")
        .await
        .expect("loop runs");
    drop(tx);

    let mut statuses = Vec::new();
    while let Ok(event) = rx.try_recv() {
        if let AgentEvent::Status { message } = event {
            statuses.push(message);
        }
    }
    let perception = statuses
        .iter()
        .find(|s| s.contains("element(s)"))
        .expect("the loop describes the screen it sees");
    assert!(
        perception.contains("despacito (S)"),
        "names, not ordinals: {perception}"
    );
    assert!(
        perception.contains("Luis Fonsi - Despacito ft. Daddy Yankee (Y)"),
        "{perception}"
    );
    assert!(
        !perception.contains("element(s): S, Y"),
        "the old label-first line must be gone: {perception}"
    );
    assert!(out.complete, "{out}");
}

#[tokio::test]
async fn the_run_streams_progress_events_to_the_tui() {
    let script = Script::new(vec![(
        frame(
            &["Pause"],
            &["the video is playing", "the player shows a pause button"],
        ),
        40,
    )]);
    let registry = registry(&script);
    let stub = lucy_agent::testing::StubProvider::new().push_json(plan_two());
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AgentEvent>();
    let deps = deps_with_events(&stub, &registry, AgentBudget::default(), Some(tx.clone()));

    let out = run_agent_loop_with(&deps, "play despacito")
        .await
        .expect("loop runs");
    drop(tx);
    let mut started = 0usize;
    let mut finished = 0usize;
    let mut progress = 0usize;
    let mut statuses = 0usize;
    while let Ok(event) = rx.try_recv() {
        match event {
            AgentEvent::ToolStarted { .. } => started += 1,
            AgentEvent::ToolFinished { .. } => finished += 1,
            AgentEvent::Progress { .. } => progress += 1,
            AgentEvent::Status { .. } => statuses += 1,
            _ => {}
        }
    }
    assert!(
        started > 0 && started == finished,
        "every fast call is bracketed"
    );
    assert!(progress > 0, "objective progress is streamed");
    assert!(statuses > 0, "status lines are streamed");
    assert!(
        out.summary.contains("fast call(s)"),
        "the closing report states the split: {out}"
    );
    assert!(out.summary.contains("slow (LLM) call(s)"), "{out}");
    assert!(out.summary.contains("replan(s)"), "{out}");
}

// ---------------------------------------------------------------------------
// Prompt contract: the loop's real interface to the model.
// ---------------------------------------------------------------------------

#[test]
fn the_prompts_ask_for_objectives_not_tool_calls() {
    for prompt in [AGENT_PLAN_INSTRUCTIONS, AGENT_REPLAN_INSTRUCTIONS] {
        assert!(prompt.contains("success_check"), "{prompt}");
        assert!(prompt.contains("suggested_action"), "{prompt}");
        assert!(prompt.contains("objectives"), "{prompt}");
    }
    assert!(AGENT_REPLAN_INSTRUCTIONS.contains("empty"));
    assert!(AGENT_PLAN_INSTRUCTIONS.contains("2 to 6"));
}

/// The `action` contract has a third value. A planner that says `press` and
/// forgets the key still submits, because that is the only thing anyone asks
/// for with that verb.
#[test]
fn a_press_objective_carries_its_key() {
    let objs = parse_objectives(&json!({"objectives":[
        {"description":"submit the query","success_check":"results are showing",
         "suggested_action":"press enter to submit","action":"press"},
        {"description":"dismiss the dialog","success_check":"the dialog is gone",
         "suggested_action":"press escape","action":"press","key":"Escape"},
        {"description":"submit the other way","success_check":"results are showing",
         "suggested_action":"press enter","action":"press","text":"Enter"}
    ]}));
    assert_eq!(objs[0].action, ActKind::Press);
    assert_eq!(
        objs[0].text.as_deref(),
        Some("Enter"),
        "a press with no key is a submit"
    );
    // `key` is the natural name for a press argument, and `text` works too.
    assert_eq!(objs[1].text.as_deref(), Some("Escape"));
    assert_eq!(objs[2].text.as_deref(), Some("Enter"));
}

/// The plan prompt used to tell the model that typing a query IS the search,
/// which is false on YouTube and is the direct cause of the run that typed
/// three times and never searched.
#[test]
fn the_plan_prompt_does_not_claim_typing_a_query_searches() {
    assert!(
        !AGENT_PLAN_INSTRUCTIONS.contains("Entering a search query IS the search"),
        "the rule that caused the failure must be gone"
    );
    assert!(
        AGENT_PLAN_INSTRUCTIONS.contains("presses Enter once"),
        "the automatic submit has to be stated or the model will not rely on it: {}",
        AGENT_PLAN_INSTRUCTIONS
    );
    for contract in ["\"click\"|\"type\"|\"press\"", "END state"] {
        assert!(
            AGENT_PLAN_INSTRUCTIONS.contains(contract),
            "the action contract must state {contract}"
        );
    }
    assert!(
        AGENT_REPLAN_INSTRUCTIONS.contains("Page URL"),
        "the replanner has to be told the url line is the real page: {}",
        AGENT_REPLAN_INSTRUCTIONS
    );
}

/// The probe rules have to be stated as *rules*, not illustrated with one site's
/// DOM.
///
/// These prompts used to carry three worked examples — a media probe, a search
/// probe and a page-text probe, each written against one video site's markup,
/// plus a blacklist of the unguarded forms that once produced a false "✔ Goal
/// completed". Every one of those lines was knowledge that only helped that one
/// task and that every other task had to read, which is what AGENTS.md forbids.
/// So the examples are gone and what is asserted here is the general principle
/// they were teaching, which is what actually generalises.
#[test]
fn the_prompts_teach_probe_rules_rather_than_one_sites_dom() {
    for prompt in [AGENT_PLAN_INSTRUCTIONS, AGENT_REPLAN_INSTRUCTIONS] {
        assert!(
            prompt.contains("already-true probe")
                || prompt.contains("ALREADY TRUE")
                || prompt.contains("already true")
                || prompt.contains("already-true"),
            "the loop's refusal to count an already-true probe has to be stated as a fact, \
             not left for the model to discover: {prompt}"
        );
    }
    let plan = AGENT_PLAN_INSTRUCTIONS;
    // A probe must discriminate before from after.
    assert!(
        plan.contains("FALSE on the page")
            || plan.contains("false on the page")
            || plan.contains("must be FALSE"),
        "the vacuous-probe rule has to be stated: {plan}"
    );
    // Establish it from the page's own state, not from the element being acted
    // on — that is the general form of every removed example.
    assert!(
        plan.contains("Establish that falsity from the page's own state"),
        "the discriminator rule has to be stated: {plan}"
    );
    // A field's value is not evidence that the field did anything.
    assert!(
        plan.contains("not the same as the field doing anything")
            || plan.contains("not merely resemble it"),
        "the typed-but-not-submitted rule has to be stated: {plan}"
    );
    // A probe must be JS, and prose is silently dropped.
    assert!(
        AGENT_REPLAN_INSTRUCTIONS.contains("single JavaScript expression"),
        "the replanner has to be told prose is not a probe: {}",
        AGENT_REPLAN_INSTRUCTIONS
    );
}

/// The regression this whole probe apparatus exists for, kept as a check that
/// the *rule* is still taught: a probe that is already true before the action
/// cannot distinguish "Lucy did this" from "it was like that already", and Lucy
/// refuses to count one.
#[test]
fn the_probe_rules_survive_without_naming_a_site() {
    let plan = AGENT_PLAN_INSTRUCTIONS;
    for banned in [
        "youtube",
        "YouTube",
        "despacito",
        "Despacito",
        "search_query=",
        "wob_loc",
        "ytd-video-renderer",
    ] {
        assert!(
            !plan.contains(banned),
            "the plan prompt is carrying one task's site knowledge again ({banned})"
        );
    }
    // The general form of what those examples taught is still present.
    assert!(
        plan.contains("location"),
        "the URL-state discriminator is taught"
    );
    assert!(
        plan.contains("already on screen before you act"),
        "the 'the element is already there' case is taught generally"
    );
}

/// A page URL the loop could not read has to read as `unknown` rather than
/// being left out: a desktop run has no browser, and the model has to be able
/// to tell that apart from a page whose address failed to load.
#[test]
fn the_replan_prompt_states_an_unreadable_page_url() {
    let screen = fast_perception::screen_state_from_hint_snapshot(
        &json!({"count": 1, "hints": [{"label": "S", "name": "despacito"}]}),
    );
    let exit = ObjectiveExit::AttemptsExhausted;
    let with_url = render_replan_prompt(
        "play despacito",
        "type despacito into the search box",
        0,
        1,
        Some("https://www.youtube.com/results?search_query=despacito"),
        &screen,
        &exit,
        "'type despacito into the search box' did not verify",
        &[],
    );
    assert!(
        with_url.contains("https://www.youtube.com/results?search_query=despacito"),
        "{with_url}"
    );
    let without = render_replan_prompt(
        "arrange the windows",
        "arrange the windows",
        0,
        1,
        None,
        &screen,
        &exit,
        "deterministic attempts ran out",
        &[],
    );
    assert!(
        without.contains("## Page URL"),
        "the line is rendered even with no page: {without}"
    );
    let after_url = without
        .split_once("## Page URL")
        .expect("the url line is always rendered")
        .1;
    assert!(
        after_url
            .trim_start()
            .starts_with("(where the browser is right now)\nunknown\n"),
        "an unreadable url is stated, not omitted: {without}"
    );
}

#[test]
fn an_objective_needs_no_tool_name_to_be_actionable() {
    // A parsed objective with no `suggested_action` grounds its own description.
    let objs = parse_objectives(&json!({"objectives":[
        {"description":"the account menu is open","success_check":"menu visible"}
    ]}));
    assert_eq!(objs.len(), 1);
    let o: Objective = objs.into_iter().next().expect("one objective");
    assert_eq!(o.instruction(), "the account menu is open");
    assert!(o.suggested_action.is_empty());
}

// ---------------------------------------------------------------------------
// Submit recovery: typing is not searching.
//
// The real "play despacito on youtube" run typed the query into YouTube's
// search box three times and failed three times. YouTube does not search on
// typing — the text sits in the field until it is submitted — and nothing in
// the loop could submit it, so the objective could not verify from any amount
// of typing. One Enter fixes that case, and these pin the bounds on it.
// ---------------------------------------------------------------------------

/// A `browser_press_key` that submits: pressing the key flips the page's own
/// state, which is what the objective's probe reads.
///
/// `fail` and `submits` are separate on purpose — a key that goes through and a
/// tool that reports an error for it is a real shape (the transport answered
/// late, the page navigated anyway), and the loop has to be able to tell the
/// objective apart from the press it uses to rescue it.
struct PressKeyTool {
    /// The probe answer, flipped when the key submits.
    js: Arc<Mutex<HashMap<String, bool>>>,
    needle: &'static str,
    presses: Arc<AtomicUsize>,
    /// Report the press as an error, so `press_key` returns `Err`.
    fail: bool,
    /// Submit the focused field anyway.
    submits: bool,
}

#[async_trait]
impl Tool for PressKeyTool {
    fn name(&self) -> &str {
        "browser_press_key"
    }
    fn description(&self) -> &str {
        "fake browser_press_key that submits the focused field"
    }
    fn parameters_schema(&self) -> Value {
        json!({})
    }
    fn requires_approval(&self) -> bool {
        false
    }
    async fn execute(&self, input: Value, _ctx: ToolContext) -> anyhow::Result<Value> {
        self.presses.fetch_add(1, Ordering::SeqCst);
        let key = input
            .get("key")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        // The submit only takes effect if the key was the one that submits.
        if self.submits && key.trim().eq_ignore_ascii_case("Enter") {
            self.js
                .lock()
                .expect("js results poisoned")
                .insert(self.needle.to_owned(), true);
        }
        if self.fail {
            return Err(anyhow::anyhow!("no focused element to press"));
        }
        Ok(json!({"pressed": key, "via": "Input"}))
    }
}

/// The shared registry plus a `browser_press_key`. One `SearchPage` per test
/// holds the page state the fake press acts on, so a test reads as a
/// description of the page rather than a list of arguments.
struct SearchPage {
    js: Arc<Mutex<HashMap<String, bool>>>,
    presses: Arc<AtomicUsize>,
}

impl SearchPage {
    /// A page that has not searched: the probe answers false until a key is
    /// pressed into it.
    fn unsearched() -> Arc<Self> {
        Arc::new(Self {
            js: Arc::new(Mutex::new(HashMap::from([(
                PROBE_NEEDLE.to_string(),
                false,
            )]))),
            presses: Arc::new(AtomicUsize::new(0)),
        })
    }

    fn presses(&self) -> usize {
        self.presses.load(Ordering::SeqCst)
    }

    /// The same page, with a `browser_evaluate` that also records what it was
    /// asked. A test that needs to see how many calls the loop spent on the
    /// probe — the vacuous-probe guard spends exactly one per objective, and
    /// none at all when there is no probe — uses this instead of `registry`.
    fn counted(&self, script: &Arc<Script>) -> (ToolRegistry, Arc<ProbePage>) {
        let page = Arc::new(ProbePage::new(self.js.clone()));
        let mut r = registry(script);
        r.register_arc(page.clone());
        r.register_arc(Arc::new(PressKeyTool {
            js: self.js.clone(),
            needle: PROBE_NEEDLE,
            presses: self.presses.clone(),
            fail: false,
            submits: true,
        }));
        (r, page)
    }

    /// `presses` with this behaviour, the real shape being a key that submits
    /// the focused field.
    fn registry(&self, script: &Arc<Script>, fail: bool, submits: bool) -> ToolRegistry {
        let mut r = registry_with_js(script, self.js.clone());
        r.register_arc(Arc::new(PressKeyTool {
            js: self.js.clone(),
            needle: PROBE_NEEDLE,
            presses: self.presses.clone(),
            fail,
            submits,
        }));
        r
    }
}

/// The plan the failing run actually produced: one objective that types the
/// query and asks for the results.
fn plan_type_into_search() -> Value {
    json!({"objectives":[
        {"description":"type despacito into the search box",
         "success_check":"search results for despacito are displayed",
         "success_probe":"document.querySelectorAll('a#video-title,ytd-video-renderer').length > 0",
         "suggested_action":"type despacito into the search box",
         "action":"type","text":"despacito"}
    ]})
}

/// The distinctive part of the results probe, so the fake `browser_evaluate`
/// can tell it from every other expression the loop runs.
const PROBE_NEEDLE: &str = "a#video-title";

/// The page does not move while the query is only typed, so the objective has
/// to burn its attempts rather than passing on the first look. The last frame
/// is the results page the submit produces.
fn search_page() -> Arc<Script> {
    Script::new(vec![
        (frame(&["Search", "Queue"], &[]), 1),
        (frame(&["Search", "Queue", "despacito"], &[]), 1),
        (frame(&["Search", "Queue", "despacito"], &[]), 1),
        (
            frame(
                &["despacito (Y)"],
                &[
                    "search results for despacito are displayed",
                    "play despacito on youtube",
                ],
            ),
            40,
        ),
    ])
}

/// A screen that changes on every attempt, so a bounded-retry assertion is
/// about the recovery and not about the anti-thrash guard firing first.
fn moving_screen() -> Arc<Script> {
    Script::new(vec![
        (frame(&["One"], &[]), 1),
        (frame(&["Two"], &[]), 1),
        (frame(&["Three"], &[]), 1),
        (frame(&["Four"], &[]), 40),
    ])
}

#[tokio::test]
async fn typing_a_query_that_never_searches_is_recovered_by_one_enter() {
    let script = search_page();
    let page = SearchPage::unsearched();
    let registry = page.registry(&script, false, true);
    let stub = lucy_agent::testing::StubProvider::new().push_json(plan_type_into_search());
    let (deps, _intr) = harness(&stub, &registry, AgentBudget::default());

    let out = run_agent_loop_with(&deps, "play despacito on youtube")
        .await
        .expect("loop runs");

    assert_eq!(
        page.presses(),
        1,
        "typing landed in the field and one submit finished the job: {out}"
    );
    assert_eq!(
        out.stats.objectives_done, 1,
        "the query was actually searched: {out}"
    );
    assert!(
        out.summary.contains("Goal completed"),
        "a submitted search is a completed search: {out}"
    );
    // The point of the recovery: the objective ended on the press, not after
    // typing the same query three times into a box that never searched.
    assert!(
        script.calls("hint_act") <= 2,
        "the recovery stops the retry loop early, saw {} type(s): {out}",
        script.calls("hint_act")
    );
    assert_eq!(
        stub.purposes(),
        vec!["agent_plan"],
        "a submitted search needs no slow call: {out}"
    );
}

#[tokio::test]
async fn the_submit_recovery_is_attempted_at_most_once_per_objective() {
    // The press never lands and the probe never flips, so every attempt
    // fails. The recovery must still be spent only once: a second Enter on a
    // page that did not submit just types a newline and costs a fast call.
    let script = moving_screen();
    let page = SearchPage::unsearched();
    let registry = page.registry(&script, true, false);
    let stub = lucy_agent::testing::StubProvider::new().push_json(plan_type_into_search());
    let mut budget = AgentBudget::default();
    budget.replan_on_failure = false;
    let (deps, _intr) = harness(&stub, &registry, budget);

    let out = run_agent_loop_with(&deps, "play despacito on youtube")
        .await
        .expect("loop runs");

    assert_eq!(
        page.presses(),
        1,
        "one recovery per objective, not one per attempt: {out}"
    );
    assert!(
        !out.complete,
        "a press that never lands leaves the query unsubmitted: {out}"
    );
    assert_eq!(
        out.stats.objectives_failed,
        vec!["type despacito into the search box".to_string()],
        "the objective failed, and a failed recovery must not hide that: {out}"
    );
    assert!(out.summary.contains("Partial result"), "{out}");
}

#[tokio::test]
async fn a_press_reported_as_failed_does_not_itself_fail_the_objective() {
    // The key went through and the tool reported an error for it, so
    // `press_key` returns `Err` and the recovery skips its own re-verify. The
    // objective is decided by the page's state on the next attempt, not by the
    // return value of the recovery that tried to help it.
    let script = search_page();
    let page = SearchPage::unsearched();
    let registry = page.registry(&script, true, true);
    let stub = lucy_agent::testing::StubProvider::new().push_json(plan_type_into_search());
    let (deps, _intr) = harness(&stub, &registry, AgentBudget::default());

    let out = run_agent_loop_with(&deps, "play despacito on youtube")
        .await
        .expect("a failed recovery must not fail the run");

    assert_eq!(page.presses(), 1, "{out}");
    assert_eq!(
        out.stats.objectives_done, 1,
        "the query was submitted, whatever the press reported: {out}"
    );
    assert!(out.complete, "the page's own state says it worked: {out}");
}

#[tokio::test]
async fn a_click_objective_is_never_given_a_submit_recovery() {
    // The recovery is for "the field has the text and the page did not
    // move". A click that did not land is a different failure, and Enter on
    // whatever happens to be focused afterwards would be a second blind
    // action nobody planned.
    let script = moving_screen();
    let page = SearchPage::unsearched();
    let registry = page.registry(&script, false, false);
    let stub = lucy_agent::testing::StubProvider::new().push_json(json!({"objectives":[
        {"description":"click the video result",
         "success_check":"the video is playing",
         "success_probe":"Array.from(document.querySelectorAll('video')).some(v => !v.paused)",
         "suggested_action":"click the video result"}
    ]}));
    let mut budget = AgentBudget::default();
    budget.replan_on_failure = false;
    let (deps, _intr) = harness(&stub, &registry, budget);

    let out = run_agent_loop_with(&deps, "play despacito on youtube")
        .await
        .expect("loop runs");

    assert_eq!(page.presses(), 0, "a click objective gets no submit: {out}");
    assert!(!out.complete, "{out}");
    assert!(
        script.calls("hint_act") >= 2,
        "the click was still retried normally: {out}"
    );
}

#[tokio::test]
async fn a_press_objective_presses_its_own_key_and_gets_no_recovery() {
    // `action: "press"` with no key presses Enter, and then the loop must not
    // press Enter again behind its back.
    let script = search_page();
    let page = SearchPage::unsearched();
    let registry = page.registry(&script, false, true);
    let stub = lucy_agent::testing::StubProvider::new().push_json(json!({"objectives":[
        {"description":"submit the query",
         "success_check":"search results for despacito are displayed",
         "success_probe":"document.querySelectorAll('a#video-title,ytd-video-renderer').length > 0",
         "suggested_action":"press enter to submit the search box",
         "action":"press"}
    ]}));
    let (deps, _intr) = harness(&stub, &registry, AgentBudget::default());

    let out = run_agent_loop_with(&deps, "play despacito on youtube")
        .await
        .expect("loop runs");

    assert_eq!(
        page.presses(),
        1,
        "the objective pressed once and the loop did not press behind it: {out}"
    );
    assert!(
        out.stats.objectives_done == 1,
        "the press submitted the query: {out}"
    );
}

#[tokio::test]
async fn the_replan_prompt_carries_the_page_the_browser_is_actually_on() {
    // The replan for the real failing run said "Play the Despacito video from
    // the search results" and clicked on the home page: the prompt never said
    // which page the browser was on. A `browser_evaluate` that answers
    // `location.href` proves the URL reaches the model.
    let script = Script::new(vec![(frame(&["Same"], &[]), 100_000)]);
    let r = registry_with_url(&script, "https://www.youtube.com/");
    let stub = lucy_agent::testing::StubProvider::new()
        .push_json(json!({"objectives":[
            {"description":"Never lands","success_check":"never",
             "suggested_action":"click the same thing"}
        ]}))
        .push_json(json!({"objectives": []}));
    let (deps, _intr) = harness(&stub, &r, AgentBudget::default());

    let out = run_agent_loop_with(&deps, "play despacito on youtube")
        .await
        .expect("loop runs");

    assert_eq!(stub.purposes(), vec!["agent_plan", "agent_replan"], "{out}");
    let replan_prompt = stub
        .seen()
        .into_iter()
        .find(|(purpose, _)| purpose == "agent_replan")
        .map(|(_, prompt)| prompt)
        .expect("a replan call was made");
    assert!(
        replan_prompt.contains("## Page URL") && replan_prompt.contains("https://www.youtube.com/"),
        "the replan prompt must name the page the browser is on: {replan_prompt}"
    );
}

/// The shared registry plus a `browser_evaluate` that reports a real address
/// for `location.href`, which is what the replan prompt now reads.
fn registry_with_url(script: &Arc<Script>, url: &str) -> ToolRegistry {
    let mut r = registry(script);
    r.register_arc(Arc::new(UrlPageTool {
        url: url.to_owned(),
    }));
    r
}

/// Answers `location.href` and `location.hostname` with a real address, so the
/// loop's page reads are the ones a real browser gives. Every other expression
/// is false, which is the honest answer for a page that has not moved.
struct UrlPageTool {
    url: String,
}

#[async_trait]
impl Tool for UrlPageTool {
    fn name(&self) -> &str {
        "browser_evaluate"
    }
    fn description(&self) -> &str {
        "fake browser_evaluate that reports a page URL"
    }
    fn parameters_schema(&self) -> Value {
        json!({})
    }
    fn requires_approval(&self) -> bool {
        false
    }
    async fn execute(&self, input: Value, _ctx: ToolContext) -> anyhow::Result<Value> {
        let expr = input
            .get("expression")
            .or_else(|| input.get("js"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        if expr.contains("location.href") {
            return Ok(json!({"result": {"type": "string", "value": self.url}}));
        }
        if expr.contains("location.hostname") {
            return Ok(json!({"result": {"type": "string", "value": "www.youtube.com"}}));
        }
        Ok(json!({"result": {"type": "boolean", "value": false}}))
    }
}

// ---------------------------------------------------------------------------
// Completion honesty.
//
// A replan replaces the remaining objectives, so the done/total ratio can be
// reshaped into a clean sweep by a run that never achieved the goal. These
// pin the rule that a failure stays on the books.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_replan_into_easier_steps_never_reports_completion() {
    // The real shape of the "play despacito on youtube" run: objective 1
    // fails, the replan replaces it with two trivial steps that both verify,
    // and the ratio comes out 3/3 — while the goal-level check, which is what
    // actually looks at the goal, is unconfirmed.
    let script = Script::new(vec![(
        frame(
            &["Search", "Submit", "Type"],
            &[
                "the search box contains 'despacito'",
                "the query was submitted",
            ],
        ),
        100_000,
    )]);
    let registry = registry(&script);
    let stub = lucy_agent::testing::StubProvider::new()
        .push_json(json!({"objectives":[
            {"description":"type despacito into the search box",
             "success_check":"the search box contains 'despacito'",
             "suggested_action":"type despacito into the search box","action":"type",
             "text":"despacito"},
            {"description":"click the video result whose title contains Despacito",
             "success_check":"the video is playing",
             "suggested_action":"click the video result"}
        ]}))
        // The replan drifts to the easy halves of the task and drops the step
        // that would have started playback.
        .push_json(json!({"objectives":[
            {"description":"Enter Despacito into the Search field",
             "success_check":"the search box contains 'despacito'",
             "suggested_action":"type despacito into the search box","action":"type",
             "text":"despacito"},
            {"description":"Submit the search query to show video results",
             "success_check":"the query was submitted",
             "suggested_action":"click the search button"}
        ]}));
    let mut budget = AgentBudget::default();
    budget.max_recoveries = 1;
    budget.max_act_steps = 24;
    let (deps, _intr) = harness(&stub, &registry, budget);

    let out = run_agent_loop_with(&deps, "play despacito on youtube")
        .await
        .expect("loop runs");

    assert!(
        !out.complete,
        "the video never played, so the run must not claim completion: {out}"
    );
    assert!(
        !out.summary.contains("Goal completed"),
        "the report must not say the goal completed: {out}"
    );
    assert!(out.summary.contains("Partial result"), "{out}");
    // The objective that actually failed has to be named. It is no longer in
    // the plan after the replan, so only the failure record still knows it.
    assert!(
        out.summary
            .contains("click the video result whose title contains Despacito"),
        "the failed objective is still outstanding and must be reported: {out}"
    );
    assert_eq!(
        out.stats.objectives_failed.len(),
        1,
        "exactly the one objective that never verified: {:?}",
        out.stats.objectives_failed
    );
}

#[tokio::test]
async fn a_healthy_run_still_reports_completion() {
    // The other side of the rule: recording failures must not make completion
    // unreachable. No objective fails here, so the run is still a completion.
    // Both objectives' checks go green as the screen advances, so nothing ever
    // fails and the whole plan verifies.
    let script = Script::new(vec![
        (frame(&["Search", "Queue"], &[]), 1),
        (frame(&["Pause"], &["the video is playing"]), 1),
        (frame(&["Pause"], &["the player shows a pause button"]), 40),
    ]);
    let registry = registry(&script);
    let stub = lucy_agent::testing::StubProvider::new().push_json(plan_two());
    let (deps, _intr) = harness(&stub, &registry, AgentBudget::default());

    let out = run_agent_loop_with(&deps, "play despacito")
        .await
        .expect("loop runs");

    assert!(
        out.complete,
        "nothing failed, so this is a completion: {out}"
    );
    assert!(out.summary.contains("Goal completed"), "{out}");
    assert!(out.stats.objectives_failed.is_empty());
}

#[tokio::test]
async fn a_stale_vision_verdict_cannot_outvote_the_last_objective_probe() {
    // The real shape of the false completion on "play despacito on youtube":
    // every objective's own check passed on the strength of a 2B model reading
    // a page of search results, while the page's own state said nothing was
    // playing. Nothing ever failed, so the failure record was empty -- the
    // only honest evidence was the last objective's probe, and the vision
    // verdict was being preferred over it.
    let script = Script::new(vec![(
        frame(
            &["Search", "despacito"],
            &["the search box contains despacito", "the video is playing"],
        ),
        100_000,
    )]);
    // Every objective's own check is satisfied by the scripted screen, but the
    // final "is it actually playing" probe is false. The first objective's
    // probe is false too, so what verifies it is the vision model reading the
    // screen — which is the whole point of this test, and is also what keeps
    // the vacuous-probe guard out of it.
    let js = Arc::new(Mutex::new(HashMap::from([
        ("value.toLowerCase".to_string(), false),
        ("the video is playing".to_string(), true),
        ("!m.paused".to_string(), false),
    ])));
    let registry = registry_with_js(&script, js);
    let stub = lucy_agent::testing::StubProvider::new().push_json(json!({"objectives":[
        {"description":"type despacito into the search box",
         "success_check":"the search box contains despacito",
         "success_probe":"document.querySelector('input')?.value.toLowerCase().includes('despacito')",
         "suggested_action":"type despacito into the search box","action":"type","text":"despacito"},
        {"description":"ensure the video is playing",
         "success_check":"the video is playing",
         "success_probe":"Array.from(document.querySelectorAll('video,audio')).some(m => !m.paused)",
         "suggested_action":"click the play button"}
    ]}));
    let (deps, _intr) = harness(&stub, &registry, AgentBudget::default());

    let out = run_agent_loop_with(&deps, "play despacito on youtube")
        .await
        .expect("loop runs");

    assert!(
        out.stats.objectives_failed.is_empty(),
        "nothing failed here — both objectives verified on the vision verdict: {out}"
    );
    assert_eq!(out.stats.objectives_done, 2, "both checks passed: {out}");
    assert!(
        !out.complete,
        "the page says nothing is playing, so this must not be a completion: {out}"
    );
    assert!(!out.summary.contains("Goal completed"), "{out}");
    assert!(
        out.summary.contains("Partial result"),
        "the report must be honest about the end state: {out}"
    );
}

// ---------------------------------------------------------------------------
// Vacuous probes: evidence that was already there cannot end an objective.
//
// The measured incident. "play despacito on youtube" printed
// "✔ Goal completed: play despacito on youtube" after 14449ms, 13 fast calls,
// one slow call and no replan. An independent CDP check five seconds later —
// none of Lucy's own code — found a YouTube SEARCH RESULTS page with a 400x225
// inline player at currentTime 0. The plan's first objective proved "a search
// ran" with `document.querySelectorAll('a[href*="watch"],ytd-video-renderer')
// .length > 0`, and on the untouched home page, before any click or keystroke,
// that expression is already true: 44 `a[href*="watch"]` links sit in the
// sidebar. The objective was marked done the moment the query was typed, the
// run walked the rest of the plan on a page it never left, and only Lucy's own
// page probes ever said otherwise.
// ---------------------------------------------------------------------------

/// The planner's "a search ran" probe, verbatim from that plan. It counts
/// watch links and never looks at the address, which is the whole defect: the
/// number is 44 on a home page that has not searched.
const WATCH_LINK_PROBE: &str =
    "document.querySelectorAll('a[href*=\"watch\"],ytd-video-renderer').length > 0";

/// The distinctive part of [`WATCH_LINK_PROBE`], so the fake
/// `browser_evaluate` can answer it without answering every other expression.
const WATCH_LINK_NEEDLE: &str = "a[href*=\"watch\"]";

/// The other half of the incident: the bare "is it playing" probe, which any
/// inline preview player on a results page trips.
const PLAYING_PROBE: &str =
    "Array.from(document.querySelectorAll('video')).some(m => !m.paused && m.currentTime > 0)";

/// The distinctive part of [`PLAYING_PROBE`].
const PLAYING_NEEDLE: &str = "!m.paused";

/// The YouTube home page the incident measured, and what the 2B screen check
/// said about it: the search results are "displayed" and the video is "playing"
/// because the page has a suggestion list and a preview player. Both checks are
/// satisfied, so without the guard every objective here verifies twice over.
fn home_page() -> Arc<Script> {
    Script::new(vec![(
        frame(
            &["Search", "despacito", "Pause"],
            &["search results are displayed", "the video is playing"],
        ),
        100_000,
    )])
}

/// A `browser_evaluate` that answers every expression it is asked and records
/// what it was asked, so a test can see exactly how many calls the run spent on
/// a probe. Probe answers are looked up by a distinctive substring, the way
/// [`JsProbeTool`] does, so one page can hold several probes at once.
struct ProbePage {
    answers: Arc<Mutex<HashMap<String, bool>>>,
    asked: Arc<Mutex<Vec<String>>>,
    url: String,
}

impl ProbePage {
    fn new(answers: Arc<Mutex<HashMap<String, bool>>>) -> Self {
        Self {
            answers,
            asked: Arc::new(Mutex::new(Vec::new())),
            url: "https://www.youtube.com/".to_owned(),
        }
    }

    /// A page whose probes are all false until a test says otherwise, which is
    /// the honest answer for a page that has not moved.
    fn blank() -> Arc<Self> {
        Arc::new(Self::new(Arc::new(Mutex::new(HashMap::new()))))
    }

    fn answer(&self, needle: &str, value: bool) {
        self.answers
            .lock()
            .expect("js results poisoned")
            .insert(needle.to_owned(), value);
    }

    fn registry(&self, script: &Arc<Script>) -> ToolRegistry {
        let mut r = registry(script);
        r.register_arc(Arc::new(ProbePage {
            answers: self.answers.clone(),
            asked: self.asked.clone(),
            url: self.url.clone(),
        }));
        r
    }

    /// The same page, with an interaction that makes one probe start holding.
    /// A click on a play button starting playback is the honest shape: the
    /// probe was false when Lucy arrived and true only because it acted, which
    /// is the difference the guard is measuring.
    fn registry_where_acting_plays(&self, script: &Arc<Script>, needle: &str) -> ToolRegistry {
        let mut r = self.registry(script);
        r.register_arc(Arc::new(ActingStartsPlayback {
            answers: self.answers.clone(),
            needle: needle.to_owned(),
        }));
        r
    }

    /// Every expression the loop evaluated, in order.
    fn asked(&self) -> Vec<String> {
        self.asked.lock().expect("asked poisoned").clone()
    }

    /// How many times one probe was evaluated. One per objective is the whole
    /// point: the vacuous-probe guard is a single measurement, not one per
    /// attempt.
    fn times_asked(&self, needle: &str) -> usize {
        self.asked().iter().filter(|e| e.contains(needle)).count()
    }

    /// Everything except the loop's own page-address reads, which happen for
    /// reasons that have nothing to do with an objective's probe.
    fn probes_asked(&self) -> Vec<String> {
        self.asked()
            .into_iter()
            .filter(|e| !e.contains("location.hostname") && !e.contains("location.href"))
            .collect()
    }
}

#[async_trait]
impl Tool for ProbePage {
    fn name(&self) -> &str {
        "browser_evaluate"
    }
    fn description(&self) -> &str {
        "fake browser_evaluate that records what it was asked"
    }
    fn parameters_schema(&self) -> Value {
        json!({})
    }
    fn requires_approval(&self) -> bool {
        false
    }
    async fn execute(&self, input: Value, _ctx: ToolContext) -> anyhow::Result<Value> {
        let expr = input
            .get("expression")
            .or_else(|| input.get("js"))
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_owned();
        self.asked
            .lock()
            .expect("asked poisoned")
            .push(expr.clone());
        if expr.contains("location.hostname") {
            return Ok(json!({"result": {"type": "string", "value": "www.youtube.com"}}));
        }
        if expr.contains("location.href") {
            return Ok(json!({"result": {"type": "string", "value": self.url}}));
        }
        let answer = self
            .answers
            .lock()
            .expect("js results poisoned")
            .iter()
            .find(|(needle, _)| expr.contains(needle.as_str()))
            .map(|(_, v)| *v)
            .unwrap_or(false);
        Ok(json!({"result": {"type": "boolean", "value": answer}}))
    }
}

/// A `hint_act` that starts playback, so the media probe goes from false to
/// true because of the action rather than in spite of it.
struct ActingStartsPlayback {
    answers: Arc<Mutex<HashMap<String, bool>>>,
    needle: String,
}

#[async_trait]
impl Tool for ActingStartsPlayback {
    fn name(&self) -> &str {
        "hint_act"
    }
    fn description(&self) -> &str {
        "fake hint_act that starts playback"
    }
    fn parameters_schema(&self) -> Value {
        json!({})
    }
    fn requires_approval(&self) -> bool {
        false
    }
    async fn execute(&self, _input: Value, _ctx: ToolContext) -> anyhow::Result<Value> {
        self.answers
            .lock()
            .expect("js results poisoned")
            .insert(self.needle.clone(), true);
        Ok(json!({"success": true, "tier": "decider", "label": "play"}))
    }
}

#[tokio::test]
async fn a_probe_that_was_already_true_cannot_end_an_objective() {
    // Objective 1 is the incident verbatim: its probe counts watch links, the
    // home page already carries 44 of them, and the screen check agrees the
    // results are displayed. So both oracles say the search happened, and the
    // only thing that can tell the truth is that the answer was already yes.
    let script = home_page();
    let page = ProbePage::blank();
    page.answer(WATCH_LINK_NEEDLE, true);
    let registry = page.registry(&script);
    let stub = lucy_agent::testing::StubProvider::new().push_json(json!({"objectives":[
        {"description":"Search for \"despacito\"",
         "success_check":"search results are displayed",
         "success_probe":WATCH_LINK_PROBE,
         "suggested_action":"type despacito into the search box",
         "action":"type","text":"despacito"},
        {"description":"Play the video",
         "success_check":"the video is playing",
         "success_probe":PLAYING_PROBE,
         "suggested_action":"click the video result"}
    ]}));
    let (deps, _intr) = harness(&stub, &registry, AgentBudget::default());

    let out = run_agent_loop_with(&deps, "play despacito on youtube")
        .await
        .expect("loop runs");

    assert_eq!(
        out.stats.objectives_done, 1,
        "only the goal objective is earned evidence: {out}"
    );
    assert!(
        !out.complete,
        "a probe that was true before Lucy acted is not a completion: {out}"
    );
    assert!(!out.summary.contains("Goal completed"), "{out}");
    assert_eq!(
        out.stats.objectives_failed,
        vec!["Search for \"despacito\"".to_string()],
        "the objective is on the failure record: {out}"
    );
    assert_eq!(
        out.stats.vacuous_probes, 1,
        "counted, not quietly dropped: {out}"
    );
    assert!(
        out.stats.summary().contains("1 vacuous probe(s)"),
        "the user sees the number: {}",
        out.stats.summary()
    );
    assert!(
        out.summary.contains("1 vacuous probe(s)"),
        "and it is in the report itself: {out}"
    );
    assert!(
        out.summary.contains("Search for \"despacito\""),
        "the report names what is not done: {out}"
    );
    // One measurement for the whole objective, taken before the first action —
    // and the action never happened, because there was nothing to prove.
    assert_eq!(
        page.times_asked(WATCH_LINK_NEEDLE),
        1,
        "one pre-check per objective, not one per attempt: {:?}",
        page.asked()
    );
    assert_eq!(
        out.stats.act_steps, 1,
        "the vacuous objective never acted: {out}"
    );
}

#[tokio::test]
async fn a_redundant_later_objective_is_a_success_that_is_still_booked_as_vacuous() {
    // The other side of the guard, and the false negative it caused. A real run
    // of "play despacito on youtube" re-anchored its way onto
    // /watch?v=kJQP7kiw5Fk, YouTube auto-started the video, and the plan's
    // LAST objective — "Play the video", whose probe is `!m.paused &&
    // m.currentTime > 0` — was already true before Lucy touched it, because
    // opening the video had just done the playing. The guard called that a
    // failure and the run printed "Partial result" with the video playing.
    //
    // So the run below is shaped like that one: objective 1 earns its evidence
    // honestly, and that action is also what makes objective 2's probe true, so
    // objective 2 has nothing left to prove. Redundant is not failed.
    // Both objectives carry the same end-state probe, which is exactly the
    // shape the real plan had: a planner that cannot see the page yet writes
    // "open the video" and "play the video" as two steps against the same
    // observable state. Objective 1's probe is FALSE when Lucy arrives and its
    // click makes it true — a real transition. Objective 2's is then already
    // true, and the work it was asked to do is the work objective 1 just did.
    let script = home_page();
    let page = ProbePage::blank();
    let registry = page.registry_where_acting_plays(&script, PLAYING_NEEDLE);
    let stub = lucy_agent::testing::StubProvider::new().push_json(json!({"objectives":[
        {"description":"Open the Despacito video",
         "success_check":"the video is playing",
         "success_probe":PLAYING_PROBE,
         "suggested_action":"click the video result whose title contains Despacito"},
        {"description":"Play the Despacito video",
         "success_check":"the video is playing",
         "success_probe":PLAYING_PROBE,
         "suggested_action":"click the play button"}
    ]}));
    let (deps, _intr) = harness(&stub, &registry, AgentBudget::default());

    let out = run_agent_loop_with(&deps, "play despacito on youtube")
        .await
        .expect("loop runs");

    assert_eq!(
        out.stats.objectives_done, 2,
        "the redundant objective is a success, not a failure: {out}"
    );
    assert!(
        out.stats.objectives_failed.is_empty(),
        "nothing failed — the goal was reached: {out}"
    );
    assert_eq!(
        out.stats.vacuous_probes, 1,
        "counted exactly once, and only for the objective that arrived pre-satisfied: {out}"
    );
    assert!(
        out.complete,
        "the goal check confirms the run, so the redundant objective is not \
         held against it: {out}"
    );
    assert!(out.summary.contains("Goal completed"), "{out}");
    assert!(
        out.stats.summary().contains("1 vacuous probe(s)"),
        "and the number is still on the transcript, because the evidence was \
         pre-satisfied even though the outcome was right: {}",
        out.stats.summary()
    );
}

#[tokio::test]
async fn an_unproven_objective_buys_no_slow_call() {
    // Two vacuous objectives and nothing else on the page. A replan is
    // affordable here (one recovery in the budget), so a call that took the
    // fast-lane exit and spent a slow one anyway would show up in the counts
    // rather than in a "budget spent" message.
    let script = home_page();
    let page = ProbePage::blank();
    page.answer(WATCH_LINK_NEEDLE, true);
    let registry = page.registry(&script);
    let stub = lucy_agent::testing::StubProvider::new().push_json(json!({"objectives":[
        {"description":"Search for \"despacito\"",
         "success_check":"search results are displayed",
         "success_probe":WATCH_LINK_PROBE,
         "suggested_action":"type despacito into the search box",
         "action":"type","text":"despacito"},
        {"description":"Play the video",
         "success_check":"the video is playing",
         "success_probe":WATCH_LINK_PROBE,
         "suggested_action":"click the video result"}
    ]}));
    let mut budget = AgentBudget::default();
    budget.max_recoveries = 1;
    let (deps, _intr) = harness(&stub, &registry, budget);

    let out = run_agent_loop_with(&deps, "play despacito on youtube")
        .await
        .expect("loop runs");

    assert_eq!(
        stub.calls(),
        1,
        "the plan and nothing else: an unproven objective is not a deviation to replan: {:?}",
        stub.purposes()
    );
    assert_eq!(stub.purposes(), vec!["agent_plan"]);
    assert_eq!(out.stats.llm_calls, 1, "{out}");
    assert_eq!(
        out.stats.recoveries, 0,
        "the replan counter never moved: {out}"
    );
    assert_eq!(
        out.stats.vacuous_probes, 2,
        "both objectives were unproven: {out}"
    );
    assert_eq!(
        out.stats.act_steps, 0,
        "an objective that cannot be proven is never acted on: {out}"
    );
}

#[tokio::test]
async fn a_probe_that_is_false_before_the_act_still_completes() {
    // The healthy path through the same guard. The results probe answers false
    // on the page Lucy starts on and true only after the submit, so the
    // objective is earned — the guard has to stay out of this run.
    let script = search_page();
    let search = SearchPage::unsearched();
    let (registry, page) = search.counted(&script);
    let stub = lucy_agent::testing::StubProvider::new().push_json(plan_type_into_search());
    let (deps, _intr) = harness(&stub, &registry, AgentBudget::default());

    let out = run_agent_loop_with(&deps, "play despacito on youtube")
        .await
        .expect("loop runs");

    assert!(out.complete, "an earned probe is still a completion: {out}");
    assert_eq!(
        out.stats.objectives_done, 1,
        "the query was really searched: {out}"
    );
    assert_eq!(
        out.stats.vacuous_probes, 0,
        "a probe that starts false is not a vacuous one: {out}"
    );
    assert_eq!(out.stats.objectives_failed, Vec::<String>::new(), "{out}");
    assert!(out.summary.contains("Goal completed"), "{out}");
    assert_eq!(search.presses(), 1, "the submit still happened: {out}");
    assert!(
        page.probes_asked().len() >= 2,
        "the pre-check plus the verifications it was protecting: {:?}",
        page.probes_asked()
    );
}

#[tokio::test]
async fn an_objective_with_no_probe_is_untouched_by_the_guard() {
    // Nothing to falsify, so the guard must be inert: the vision path decides
    // the objective as it always did, and not one fast call is spent asking a
    // question that has no answer.
    let script = Script::new(vec![(
        frame(&["Pause"], &["the player shows a pause button"]),
        100_000,
    )]);
    let page = ProbePage::blank();
    let registry = page.registry(&script);
    let stub = lucy_agent::testing::StubProvider::new().push_json(json!({"objectives":[
        {"description":"Confirm playback started",
         "success_check":"the player shows a pause button",
         "suggested_action":"click the play button"}
    ]}));
    let (deps, _intr) = harness(&stub, &registry, AgentBudget::default());

    let out = run_agent_loop_with(&deps, "play despacito on youtube")
        .await
        .expect("loop runs");

    assert!(out.complete, "the vision path still decides this: {out}");
    assert_eq!(out.stats.objectives_done, 1, "{out}");
    assert_eq!(
        out.stats.vacuous_probes, 0,
        "there was no probe at all: {out}"
    );
    assert!(
        page.probes_asked().is_empty(),
        "the guard spent no call on a probe that does not exist: {:?}",
        page.asked()
    );
    assert_eq!(
        script.calls("hint_act"),
        1,
        "one act and one check, exactly as before: {out}"
    );
}

/// A `browser_evaluate` that answers a fixed expression `true` for the first
/// `flips_after` asks and `false` from then on.
///
/// The real run needed exactly this and nothing else. "open wikipedia" is a
/// one-objective plan, and at the moment Lucy landed on the site the planner's
/// probe still held — so the objective was satisfied by ARRIVING (0 vacuous
/// probes). Minutes later the same probe read false, because Wikipedia's
/// tagline is "Wikipedia, the free encyclopedia" and the planner had written
/// `document.body.innerText.includes('The Free Encyclopedia')`. The whole-goal
/// check read that second answer and turned a finished task into
/// "Partial result". Nothing about the page or the goal changed between the two
/// evaluations except the page's own text finishing hydrating, and a static
/// answer map cannot express "true, then false" — hence this.
struct ProbeFlipsAfter {
    expression: String,
    flips_after: usize,
    asks: Arc<AtomicUsize>,
    host: Option<Arc<Mutex<String>>>,
}

#[async_trait]
impl Tool for ProbeFlipsAfter {
    fn name(&self) -> &str {
        "browser_evaluate"
    }
    fn description(&self) -> &str {
        "fake browser_evaluate whose one expression stops holding partway through"
    }
    fn parameters_schema(&self) -> Value {
        json!({})
    }
    fn requires_approval(&self) -> bool {
        false
    }
    async fn execute(&self, input: Value, _ctx: ToolContext) -> anyhow::Result<Value> {
        let expr = input
            .get("expression")
            .or_else(|| input.get("js"))
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_owned();
        let bare_hostname = expr.trim().trim_end_matches(';') == "location.hostname";
        if bare_hostname && let Some(host) = self.host.as_ref() {
            let h = host.lock().expect("host poisoned").clone();
            return Ok(json!({"result": {"type": "string", "value": h}}));
        }
        // Substring, not equality: the probe reaches the tool after
        // `sanitize_probe` has had its way with the text, so the expression as
        // written in the plan is not necessarily the expression as evaluated.
        // Matching on the tagline clause is what this fixture is for.
        if !expr.contains(&self.expression) {
            return Ok(json!({"result": {"type": "boolean", "value": false}}));
        }
        let n = self.asks.fetch_add(1, Ordering::SeqCst);
        Ok(json!({"result": {"type": "boolean", "value": n < self.flips_after}}))
    }
}

#[tokio::test]
async fn landing_on_the_goal_site_outranks_a_clause_that_stopped_holding_afterwards() {
    // The false negative the whole-goal check had for the simplest task there
    // is. "open wikipedia" is a ONE-objective plan that PHASE 0 satisfies by
    // navigating, and the planner wrote this probe for it:
    //   location.hostname.includes('wikipedia.org') && document.title
    //     .includes('Wikipedia') && document.body.innerText
    //     .includes('The Free Encyclopedia')
    // Wikipedia's live tagline is "Wikipedia, the free encyclopedia" — the
    // lowercase "free encyclopedia" — so that third clause is false on the very
    // page Lucy navigated to. The run printed "Partial result" for a page it was
    // demonstrably on, and the page really is the goal.
    //
    // The probe never ran against the live page: the planner wrote it before
    // Lucy had read the site, and nothing can have read this wording off a
    // screen. So the fixture answers `title.includes` true and
    // `The Free Encyclopedia` false, which is exactly what the real page does.
    const PLANNERS_PROBE: &str = "location.hostname.includes('wikipedia.org') && document.title.includes('Wikipedia') && document.body.innerText.includes('The Free Encyclopedia')";

    let script = Script::new(vec![(frame(&["Search"], &[]), 100_000)]);
    let host = Arc::new(Mutex::new(String::new()));
    // Holds for the objective's own pre-check — which is what lets Lucy settle
    // the objective by arriving — and stops holding before the whole-goal
    // check reads it. One expression, no map, so there is no iteration order to
    // depend on. `location.hostname` still comes from `host`, which is what
    // makes PHASE 0 navigate.
    let asks = Arc::new(AtomicUsize::new(0));
    let mut registry = registry(&script);
    registry.register_arc(Arc::new(ProbeFlipsAfter {
        expression: "The Free Encyclopedia".to_owned(),
        flips_after: 1,
        asks,
        host: Some(host.clone()),
    }));
    registry.register_arc(Arc::new(NavTool {
        host,
        visited: Arc::new(Mutex::new(Vec::new())),
    }));
    let stub = lucy_agent::testing::StubProvider::new().push_json(json!({"objectives":[
        {"description":"Open Wikipedia main page",
         "success_check":"the Wikipedia main page is displayed",
         "success_probe":PLANNERS_PROBE,
         "suggested_action":"click the link titled 'English'"}
    ]}));
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AgentEvent>();
    let deps = deps_with_events(&stub, &registry, AgentBudget::default(), Some(tx.clone()));

    let out = run_agent_loop_with(&deps, "open https://en.wikipedia.org/wiki/Main_Page")
        .await
        .expect("loop runs");
    let statuses: Vec<String> = std::iter::from_fn(|| rx.try_recv().ok())
        .filter_map(|e| match e {
            AgentEvent::Status { message } => Some(message),
            _ => None,
        })
        .collect();
    drop(tx);

    assert!(
        statuses
            .iter()
            .any(|m| m.contains("going to https://en.wikipedia.org")),
        "Lucy must have navigated for this to be the case at all: {statuses:?}"
    );
    assert_eq!(
        out.stats.objectives_done, 1,
        "the single objective was satisfied by arriving: {out}"
    );
    assert!(
        out.complete,
        "a clause the planner never read off a screen cannot contradict Lucy \
         actually being on the page the goal named: {out}"
    );
    assert!(out.summary.contains("Goal completed"), "{out}");
    assert!(
        out.stats.objectives_failed.is_empty(),
        "and nothing failed: {out}"
    );
}

#[tokio::test]
async fn a_vacuous_probe_blocks_the_goal_check_from_rescuing_the_run() {
    // The exact false positive. The last objective's own probe — the one the
    // goal check consults — is true, so `goal_says_done` would carry the run
    // to a "✔ Goal completed" over a page that never moved. One vacuous probe
    // earlier in the run is what has to stop it.
    let script = home_page();
    let page = ProbePage::blank();
    page.answer(WATCH_LINK_NEEDLE, true);
    let registry = page.registry_where_acting_plays(&script, PLAYING_NEEDLE);
    let stub = lucy_agent::testing::StubProvider::new().push_json(json!({"objectives":[
        {"description":"Search for \"despacito\"",
         "success_check":"search results are displayed",
         "success_probe":WATCH_LINK_PROBE,
         "suggested_action":"type despacito into the search box",
         "action":"type","text":"despacito"},
        {"description":"Play the video",
         "success_check":"the video is playing",
         "success_probe":PLAYING_PROBE,
         "suggested_action":"click the video result"}
    ]}));
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AgentEvent>();
    let deps = deps_with_events(&stub, &registry, AgentBudget::default(), Some(tx.clone()));

    let out = run_agent_loop_with(&deps, "play despacito on youtube")
        .await
        .expect("loop runs");
    drop(tx);

    // Only the FIRST objective is unproven. The second one's media probe starts
    // false and only holds because Lucy clicked play, so the goal check is
    // reading earned evidence when it says satisfied.
    assert_eq!(out.stats.vacuous_probes, 1, "{out}");
    assert_eq!(out.stats.objectives_done, 1, "{out}");
    // The page-level check really did say satisfied, and it is the last
    // objective's own probe that said it. That is what makes this the incident
    // and not a run that simply failed: the gate on `goal_says_done` is the
    // only thing between this and a false completion, so the test has to
    // prove the rescue was live and was declined.
    let progress: Vec<String> = std::iter::from_fn(|| rx.try_recv().ok())
        .filter_map(|e| match e {
            AgentEvent::Progress { message } => Some(message),
            _ => None,
        })
        .collect();
    assert!(
        progress
            .iter()
            .any(|m| m.contains("overall goal check: satisfied")),
        "the goal probe answered true, which is exactly why the guard has to weigh: {progress:?}"
    );
    assert!(
        !out.complete,
        "evidence that was never earned cannot rescue a run: {out}"
    );
    assert!(!out.summary.contains("Goal completed"), "{out}");
    assert!(out.summary.contains("Partial result"), "{out}");
    assert!(
        out.summary.contains("1 vacuous probe(s)"),
        "and the reason the goal check was not believed is in the report: {out}"
    );
}

#[tokio::test]
async fn arriving_at_a_goal_that_is_only_a_site_is_a_completion() {
    // "open wikipedia" is complete the moment Lucy is on wikipedia. PHASE 0
    // gets there before the plan runs, so the plan's single objective has a
    // probe that is already true. Reported as a vacuous probe it used to print
    // "Partial result ... 0/1 objective(s) verified" for a task that plainly
    // worked.
    let script = Script::new(vec![(
        frame(
            &["Main Page", "Search"],
            &["the Wikipedia main page is displayed"],
        ),
        100_000,
    )]);
    let host = Arc::new(Mutex::new(String::new()));
    // The objective's probe and the whole-goal check both read the URL. The
    // wrapper the loop wraps probes in still contains the expression, so a
    // substring match on it answers both.
    let js = Arc::new(Mutex::new(HashMap::from([
        ("wikipedia.org".to_string(), true),
        ("hostname".to_string(), true),
    ])));
    let registry = registry_navigable(&script, js, host.clone());
    let stub = lucy_agent::testing::StubProvider::new().push_json(json!({"objectives":[
        {"description":"Open the Wikipedia main page",
         "success_check":"the Wikipedia main page is displayed",
         "success_probe":"location.hostname.includes('wikipedia.org')",
         "suggested_action":"click the Wikipedia link"}
    ]}));
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AgentEvent>();
    let deps = deps_with_events(&stub, &registry, AgentBudget::default(), Some(tx.clone()));

    let out = run_agent_loop_with(&deps, "open https://en.wikipedia.org/wiki/Main_Page")
        .await
        .expect("loop runs");
    drop(tx);

    // The exemption is reachable only on a run where PHASE 0 really navigated,
    // and nothing about the outcome below proves that on its own — so the run
    // is asked. Without this the test would still pass if PHASE 0 stopped
    // landing on sites entirely, which is the one change that would make the
    // exemption unreachable rather than wrong.
    let statuses: Vec<String> = std::iter::from_fn(|| rx.try_recv().ok())
        .filter_map(|e| match e {
            AgentEvent::Status { message } => Some(message),
            _ => None,
        })
        .collect();
    assert!(
        statuses
            .iter()
            .any(|m| m.contains("going to https://en.wikipedia.org")),
        "PHASE 0 landed on the goal's site, so the exemption is live: {statuses:?}"
    );
    assert_eq!(
        *host.lock().expect("host poisoned"),
        "en.wikipedia.org",
        "and the browser is really on it: {statuses:?}"
    );

    assert!(
        out.complete,
        "landing on the only site the goal names IS the task: {out}"
    );
    assert!(out.summary.contains("Goal completed"), "{out}");
    assert_eq!(
        out.stats.vacuous_probes, 0,
        "arriving is not a vacuous probe: {out}"
    );
    assert_eq!(
        out.stats.objectives_done, 1,
        "the objective is Satisfied, not merely un-counted: {out}"
    );
    assert!(
        out.stats.objectives_failed.is_empty(),
        "nothing failed — the navigation was Lucy's action for it: {out}"
    );
}

#[tokio::test]
async fn a_navigated_multi_objective_plan_still_reports_the_vacuous_probe() {
    // The other half of the guard, and the case the exemption is not allowed
    // to touch. PHASE 0 DOES navigate — "play despacito on youtube" names
    // youtube.com and the browser starts somewhere else, so the exemption is
    // live — and the plan has two objectives, so arriving on the YouTube home
    // page is not what the first of them does. Its probe counts the 44
    // `a[href*="watch"]` links that were in the sidebar before Lucy clicked
    // anything, exactly as in the measured incident, so it proves nothing about
    // searching and the objective stays on the books.
    let script = home_page();
    let host = Arc::new(Mutex::new(String::new()));
    // One needle only: the fake answers a probe by the first map key its
    // expression contains, so two keys that could both match would make the
    // answer depend on HashMap iteration order.
    let js = Arc::new(Mutex::new(HashMap::from([(
        WATCH_LINK_NEEDLE.to_string(),
        true,
    )])));
    let registry = registry_navigable(&script, js, host.clone());
    let stub = lucy_agent::testing::StubProvider::new().push_json(json!({"objectives":[
        {"description":"Search for \"despacito\"",
         "success_check":"search results are displayed",
         "success_probe":WATCH_LINK_PROBE,
         "suggested_action":"type despacito into the search box",
         "action":"type","text":"despacito"},
        {"description":"Play the video",
         "success_check":"the video is playing",
         "success_probe":PLAYING_PROBE,
         "suggested_action":"click the video result"}
    ]}));
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AgentEvent>();
    let deps = deps_with_events(&stub, &registry, AgentBudget::default(), Some(tx.clone()));

    // The goal names its destination as a URL, which is what makes PHASE 0
    // navigate. It used to be the prose "play despacito on youtube", relying on
    // the deleted site table to resolve the name; the goal is unchanged in
    // substance, but a URL the user typed is now the only thing that counts as
    // a destination before the planner runs.
    let out = run_agent_loop_with(&deps, "play despacito on https://www.youtube.com")
        .await
        .expect("loop runs");
    drop(tx);

    // The same precondition the exemption needs, asserted the same way: this
    // run navigated, so `site_navigations` is 1 and the only thing standing
    // between this objective and a false "Goal completed" is the
    // `objectives.len() == 1` clause.
    let statuses: Vec<String> = std::iter::from_fn(|| rx.try_recv().ok())
        .filter_map(|e| match e {
            AgentEvent::Status { message } => Some(message),
            _ => None,
        })
        .collect();
    assert!(
        statuses
            .iter()
            .any(|m| m.contains("going to https://www.youtube.com")),
        "this run navigated, which is what makes the guard's single-objective \
         clause the only thing protecting it: {statuses:?}"
    );

    assert_eq!(
        out.stats.vacuous_probes, 1,
        "the first objective's probe was already true: {out}"
    );
    assert_eq!(
        out.stats.objectives_done, 1,
        "only the second objective is earned evidence: {out}"
    );
    assert_eq!(
        out.stats.objectives_failed,
        vec!["Search for \"despacito\"".to_string()],
        "and the vacuous one is on the failure record: {out}"
    );
    assert!(
        !out.complete,
        "a probe that was true before Lucy acted is not a completion: {out}"
    );
    assert!(!out.summary.contains("Goal completed"), "{out}");
    assert!(out.summary.contains("Partial result"), "{out}");
}

#[tokio::test]
async fn only_the_first_objective_claims_the_navigated_exemption() {
    // `objectives_done == 0` is load-bearing, and a replan is the only way to
    // reach it. The plan starts as ONE objective — which is what makes
    // `navigated` true — but its probe starts false, so the objective is
    // genuinely attempted, fails on a screen that never changes, and the
    // replan hands back two more. The first of those may still be explained by
    // Lucy's own navigation; the second cannot, because the page has moved on
    // and a probe that is already true really is proving nothing.
    //
    // It also pins that `navigated` is read ONCE, before the loop: the revised
    // plan has two objectives, and a flag recomputed per iteration would have
    // gone false here and left the first revised objective unproven too.
    let script = Script::new(vec![(frame(&["Main Page", "Search"], &[]), 100_000)]);
    let host = Arc::new(Mutex::new(String::new()));
    // Distinct needles, none of them a substring of another probe's, so which
    // key answers which expression cannot depend on HashMap iteration order.
    let js = Arc::new(Mutex::new(HashMap::from([
        ("wikipedia.org".to_string(), true),
        ("dataset.stage".to_string(), true),
    ])));
    let registry = registry_navigable(&script, js, host.clone());
    let stub = lucy_agent::testing::StubProvider::new()
        .push_json(json!({"objectives":[
            {"description":"Open the Wikipedia main page",
             "success_check":"the Wikipedia main page is displayed",
             "success_probe":"document.querySelector('h1') === null",
             "suggested_action":"click the Wikipedia link"}
        ]}))
        .push_json(json!({"objectives":[
            {"description":"Land on the article",
             "success_check":"the article is open",
             "success_probe":"location.hostname.includes('wikipedia.org')",
             "suggested_action":"click the article link"},
            {"description":"Confirm the article finished loading",
             "success_check":"the article finished loading",
             "success_probe":"document.querySelector('main')?.dataset.stage === 'done'",
             "suggested_action":"wait for the article"}
        ]}));
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AgentEvent>();
    let deps = deps_with_events(&stub, &registry, AgentBudget::default(), Some(tx.clone()));

    let out = run_agent_loop_with(&deps, "open https://en.wikipedia.org/wiki/Main_Page")
        .await
        .expect("loop runs");
    drop(tx);

    let statuses: Vec<String> = std::iter::from_fn(|| rx.try_recv().ok())
        .filter_map(|e| match e {
            AgentEvent::Status { message } => Some(message),
            _ => None,
        })
        .collect();
    assert!(
        statuses
            .iter()
            .any(|m| m.contains("going to https://en.wikipedia.org")),
        "PHASE 0 navigated, which is the whole precondition: {statuses:?}"
    );
    assert_eq!(
        stub.purposes(),
        vec!["agent_plan", "agent_replan"],
        "the one-objective plan failed and was replanned into two: {out}"
    );

    // Both revised objectives end satisfied — the first by Lucy's own
    // navigation, the second because the plan left it redundant — so the run
    // verifies its whole revised plan. What tells them apart is the counter
    // below, and that is the point: `objectives_done == 0` on the navigation
    // clause is what keeps the first objective from being booked as vacuous,
    // and it can only be observed because the flag outlived the plan length it
    // was computed from.
    assert_eq!(
        out.stats.objectives_done, 2,
        "the revised plan is satisfied in full: {out}"
    );
    assert_eq!(
        out.stats.vacuous_probes, 1,
        "the second is on the books as vacuous, the first is not: {out}"
    );
    assert_eq!(
        out.stats.objectives_failed,
        vec!["Open the Wikipedia main page".to_string()],
        "only the objective that genuinely failed is on the record — the objective \
         the navigation was allowed to satisfy, and the one the plan left redundant, \
         are both successes: {out}"
    );
    assert!(
        !out.complete,
        "a vacuous probe on the books disqualifies the run: {out}"
    );
    assert!(!out.summary.contains("Goal completed"), "{out}");
    assert!(out.summary.contains("Partial result"), "{out}");
    assert!(
        out.summary.contains("1 vacuous probe"),
        "and the redundant objective is still on the books as pre-satisfied, so \
         the transcript never hides that its evidence was not earned: {out}"
    );
}

// ---------------------------------------------------------------------------
// Re-anchoring: a description that does not verify gets one retry as a name.
//
// The measured "play despacito on youtube" failure, in full. The loop reached
// the YouTube results page — typing, Enter, the probe, all correct — and then
// spent its entire budget there:
//
//   ℹ click the video result whose title contains Despacito — 32 element(s): …
//   ℹ attempt 1/3 …: ok via hint: Q
//   ℹ attempt 2/3: failed: …
//   ℹ replan budget spent (2 of 2)
//   ⚠ Partial result … Stopped because: screen did not change after 2 attempt(s)
//
// Q is "All", a filter tab. Re-run on the same page, the same instruction
// picked F, "Clear search query", which wiped the query box. The right answer
// was Z — "Luis Fonsi - Despacito ft. Daddy Yankee 4 minutes, 42 seconds" —
// and handing the resolver THAT string clicked it. So the resolver was fine and
// the instruction was the problem, and the loop's answer to a bad instruction
// was to send the identical bytes again and let the same wrong element be
// picked. These pin the replacement: one retry, quoting what is on screen.
// ---------------------------------------------------------------------------

/// The names the measured page showed, with their hint labels — the twenty the
/// failure transcript is quotable from. `hint_snapshot` serves them plus one
/// changing element so the anti-thrash guard cannot stop the run before the
/// retry it is supposed to be measuring.
const MEASURED_NAMES: &[(&str, &str)] = &[
    ("A", "Guide"),
    ("S", "a"),
    ("D", "despacito"),
    ("F", "Clear search query"),
    ("G", "Search"),
    ("H", "Search with your voice"),
    ("K", "Settings"),
    ("L", "Sign in"),
    ("Q", "All"),
    ("W", "Shorts"),
    ("T", "Videos"),
    ("Y", "Recently uploaded"),
    ("U", "Live"),
    ("I", "Next"),
    (
        "Z",
        "Luis Fonsi - Despacito ft. Daddy Yankee 4 minutes, 42 seconds",
    ),
    ("X", "Action menu"),
    ("C", "Go to channel LuisFonsiVEVO"),
    ("B", "Mix"),
    ("M", "Luis Fonsi - Despacito ft. Daddy Yankee · 4:42"),
];

/// The name the measured run needed and the measured resolver acted on, as the
/// page spells it. `hint_description` clips what Lucy sees to `HINT_NAME_MAX`,
/// so the anchored instruction quotes a prefix of this — the assertions below
/// are written against the prefix so a change to the clip width does not
/// quietly turn them into a different test.
/// The video link the results page spells out, and enough of it to survive
/// `hint_description`'s clip width. The assertions below are written against
/// the prefix so a change to that width does not quietly turn them into a
/// different test.
const DESPACITO_VIDEO_PREFIX: &str = "Luis Fonsi - Despacito ft. Daddy Yankee";

/// The anchored instruction's shape, without the name in it.
const ANCHORED_PREFIX: &str = "click the element named '";

/// The planner's wording for the objective, written before it had ever seen the
/// page — a description, because it had nothing else.
const VAGUE_ACTION: &str = "click the video result whose title contains Despacito";

/// The measured results page: the real names, in the real order, with one
/// element that changes per snapshot so a bounded-retry assertion is about the
/// re-anchoring and not about the fingerprint guard firing first.
struct ResultsPage {
    tick: AtomicUsize,
}

#[async_trait]
impl Tool for ResultsPage {
    fn name(&self) -> &str {
        "hint_snapshot"
    }
    fn description(&self) -> &str {
        "the measured YouTube results page, with one element that moves"
    }
    fn parameters_schema(&self) -> Value {
        json!({})
    }
    fn requires_approval(&self) -> bool {
        false
    }
    async fn execute(&self, _input: Value, _ctx: ToolContext) -> anyhow::Result<Value> {
        // A live page is never twice identical: the lazy image placeholder
        // resolves, a spinner stops. Without this the guard would end the run
        // on attempt three and the retry would never be spent.
        let n = self.tick.fetch_add(1, Ordering::SeqCst);
        let mut hints: Vec<Value> = MEASURED_NAMES
            .iter()
            .map(|(label, name)| json!({"label": label, "name": name}))
            .collect();
        hints.push(json!({"label": format!("z{n}"), "name": format!("loading {n}")}));
        Ok(json!({"count": hints.len(), "via": "decider", "hints": hints}))
    }
}

/// A `hint_act` that records every instruction it is given, reports success for
/// all of them — the resolver did click something, which is what the loop saw —
/// and only moves the page when the instruction quotes `landed_on`.
///
/// The first shape is the incident: `ok via hint: Q`. The second is what a
/// quoted name buys, measured on the same page.
struct RecordingAct {
    instructions: Arc<Mutex<Vec<String>>>,
    probe: Arc<Mutex<HashMap<String, bool>>>,
    /// The name the click has to quote for the page to actually move.
    landed_on: Option<&'static str>,
    /// When set, an instruction that does NOT quote `landed_on` fails outright
    /// instead of reporting a click. That is the real resolver's behaviour on
    /// the YouTube results page: a description it cannot map to a name returns
    /// "all tiers exhausted", so the act reports failure rather than clicking
    /// the wrong thing.
    fail_unquoted: bool,
}

impl RecordingAct {
    fn instructions(&self) -> Vec<String> {
        self.instructions
            .lock()
            .expect("instructions poisoned")
            .clone()
    }
}

#[async_trait]
impl Tool for RecordingAct {
    fn name(&self) -> &str {
        "hint_act"
    }
    fn description(&self) -> &str {
        "fake hint_act that records its instructions and lands on a named element"
    }
    fn parameters_schema(&self) -> Value {
        json!({})
    }
    fn requires_approval(&self) -> bool {
        false
    }
    async fn execute(&self, input: Value, _ctx: ToolContext) -> anyhow::Result<Value> {
        let instruction = input
            .get("instruction")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        self.instructions
            .lock()
            .expect("instructions poisoned")
            .push(instruction.clone());
        if self
            .landed_on
            .as_deref()
            .is_some_and(|name| instruction.contains(name))
        {
            self.probe
                .lock()
                .expect("js results poisoned")
                .insert(PLAYING_NEEDLE.to_owned(), true);
            return Ok(json!({"success": true, "tier": "hint", "label": "clicked"}));
        }
        if self.fail_unquoted {
            return Ok(json!({"success": false, "tier": "uncertain", "label": null,
                             "message": "find failed or uncertain, not clicked"}));
        }
        Ok(json!({"success": true, "tier": "hint", "label": "clicked"}))
    }
}

/// A `browser_press_key` that reports the press landed, so a `press`
/// objective's act genuinely succeeds — which is the precondition the
/// re-anchoring would fire on if it were not gated to clicks.
struct LandedPress {
    presses: Arc<AtomicUsize>,
}

#[async_trait]
impl Tool for LandedPress {
    fn name(&self) -> &str {
        "browser_press_key"
    }
    fn description(&self) -> &str {
        "fake browser_press_key that reports the press landed"
    }
    fn parameters_schema(&self) -> Value {
        json!({})
    }
    fn requires_approval(&self) -> bool {
        false
    }
    async fn execute(&self, input: Value, _ctx: ToolContext) -> anyhow::Result<Value> {
        self.presses.fetch_add(1, Ordering::SeqCst);
        let key = input
            .get("key")
            .and_then(Value::as_str)
            .unwrap_or("Enter")
            .to_owned();
        Ok(json!({"pressed": key, "via": "Input"}))
    }
}

/// The measured results page, a probe that only holds once the right element is
/// clicked, and a `hint_act` that records what it was asked.
fn results_page(landed_on: Option<&'static str>) -> (ToolRegistry, Arc<RecordingAct>) {
    results_page_with(landed_on, false)
}

/// [`results_page`], with the fake act optionally refusing any instruction that
/// does not quote `landed_on` — the shape a real resolver takes when the
/// description it is given is not a name it can find.
fn results_page_with(
    landed_on: Option<&'static str>,
    fail_unquoted: bool,
) -> (ToolRegistry, Arc<RecordingAct>) {
    let probe = Arc::new(Mutex::new(HashMap::new()));
    let still = Script::new(vec![(frame(&[], &[]), 1_000)]);
    let mut r = ToolRegistry::new();
    r.register_arc(Arc::new(ResultsPage {
        tick: AtomicUsize::new(0),
    }));
    r.register_arc(Arc::new(JsProbeTool {
        results: probe.clone(),
        host: None,
    }));
    // `verify` and `wait_until` are the honest ones: nothing on this page ever
    // satisfies the vision check, so the probe is the only thing that can end
    // the objective — which is what makes "the anchored click verified" mean
    // something rather than "the script got bored".
    for name in ["verify", "wait_until"] {
        r.register_arc(FakeTool::scripted(name, Behaviour::Verdict, &still));
    }
    let act = Arc::new(RecordingAct {
        instructions: Arc::new(Mutex::new(Vec::new())),
        probe,
        landed_on,
        fail_unquoted,
    });
    r.register_arc(act.clone());
    (r, act)
}

/// A single click objective the resolver cannot ground. The failing run's own
/// objective, kept verbatim as the incident fixture.
fn an_unresolvable_click_objective() -> Value {
    json!({"objectives":[
        {"description":"click the video result whose title contains Despacito",
         "success_check":"the video is playing",
         "success_probe":PLAYING_PROBE,
         "suggested_action":VAGUE_ACTION}
    ]})
}

#[tokio::test]
async fn a_click_that_the_resolver_cannot_resolve_is_still_retried_as_the_name_on_screen() {
    // The gate bug, kept honest. This run re-anchoring was conditional on the
    // previous act having reported SUCCESS — so an objective whose act failed
    // outright never got a second, better-aimed attempt. A real run hit exactly
    // that: on the YouTube results page the resolver could not map "click the
    // video result whose title contains Despacito" to any of the 31 names, so
    // `hint_act` and its `find_and_click` fallback both returned
    // "find failed or uncertain, not clicked", `outcome.success` was false, and
    // the loop re-sent the identical description for all three attempts and the
    // whole replan budget behind it. Handed the quoted name instead, the same
    // resolver clicked Z and the video played (`currentTime` 2.51s, measured).
    //
    // A failure to resolve is the strongest possible reason to change the
    // instruction, so the fake act here REFUSES anything that does not quote
    // the target's real name.
    let (registry, act) = results_page_with(Some(DESPACITO_VIDEO_PREFIX), true);
    let stub =
        lucy_agent::testing::StubProvider::new().push_json(an_unresolvable_click_objective());
    let mut budget = AgentBudget::default();
    budget.replan_on_failure = false;
    let (deps, _intr) = harness(&stub, &registry, budget);

    let out = run_agent_loop_with(&deps, "play despacito on youtube")
        .await
        .expect("loop runs");

    let instructions = act.instructions();
    assert_eq!(
        instructions.first().map(String::as_str),
        Some(VAGUE_ACTION),
        "the first attempt is the planner's own description: {instructions:?}"
    );
    assert!(
        instructions.len() >= 2 && instructions[1].contains(DESPACITO_VIDEO_PREFIX),
        "the retry must quote the name the resolver can actually find, even though \
         the first act FAILED rather than clicking the wrong thing: {instructions:?}"
    );
    assert_eq!(
        out.stats.objectives_done, 1,
        "and that retry is what opened the video: {out}"
    );
    assert!(out.complete, "{out}");
}

#[tokio::test]
async fn a_click_that_acts_but_does_not_verify_is_retried_as_the_name_on_screen() {
    // The whole incident as a test. The first attempt is the vague description
    // and it lands on nothing; the second quotes the on-screen name and opens
    // the video. Without the re-anchoring this run reports the partial failure
    // the transcript shows.
    let (registry, act) = results_page(Some(DESPACITO_VIDEO_PREFIX));
    let stub =
        lucy_agent::testing::StubProvider::new().push_json(an_unresolvable_click_objective());
    let mut budget = AgentBudget::default();
    budget.replan_on_failure = false;
    let (deps, _intr) = harness(&stub, &registry, budget);

    let out = run_agent_loop_with(&deps, "play despacito on youtube")
        .await
        .expect("loop runs");

    assert_eq!(
        out.stats.objectives_done, 1,
        "the re-anchored click is what opened the video: {out}"
    );
    assert!(out.complete, "{out}");
    let instructions = act.instructions();
    assert_eq!(
        instructions.first().map(String::as_str),
        Some(VAGUE_ACTION),
        "the first attempt is the planner's own instruction: {instructions:?}"
    );
    let retry = instructions
        .get(1)
        .expect("the objective was retried")
        .clone();
    assert!(
        retry.starts_with(ANCHORED_PREFIX) && retry.contains(DESPACITO_VIDEO_PREFIX),
        "the retry quotes the name the page actually shows: {retry:?}"
    );
    for wrong in ["All", "Videos", "Shorts", "Live", "despacito'"] {
        assert!(
            !retry.contains(&format!("{ANCHORED_PREFIX}{wrong}")),
            "the tab and the search box are not the target: {retry:?}"
        );
    }
    // And only that one retry: the objective ended on it, so a third attempt
    // would have been the loop buying another chance it did not need.
    assert_eq!(instructions.len(), 2, "{instructions:?}");
}

#[tokio::test]
async fn the_re_anchoring_happens_at_most_once_per_objective() {
    // Nothing on the page ever matches the anchor, so every attempt fails and
    // the loop spends all three. The re-anchoring is a single correction, not a
    // mode: once it has been tried, attempt three is the planner's instruction
    // again, exactly as it would have been before this feature existed.
    let (registry, act) = results_page(Some("a name that is not on this page"));
    let stub =
        lucy_agent::testing::StubProvider::new().push_json(an_unresolvable_click_objective());
    let mut budget = AgentBudget::default();
    budget.replan_on_failure = false;
    let (deps, _intr) = harness(&stub, &registry, budget);

    let out = run_agent_loop_with(&deps, "play despacito on youtube")
        .await
        .expect("loop runs");

    assert!(!out.complete, "nothing on the page ever verified: {out}");
    let instructions = act.instructions();
    assert_eq!(
        instructions.len(),
        3,
        "three attempts, so the bound is measured against a run that had one: {instructions:?}"
    );
    let anchored: Vec<&String> = instructions
        .iter()
        .filter(|i| i.starts_with(ANCHORED_PREFIX))
        .collect();
    assert_eq!(
        anchored.len(),
        1,
        "one re-anchored attempt, not one per attempt: {instructions:?}"
    );
    assert_eq!(
        *anchored[0], instructions[1],
        "and it is the second attempt: {instructions:?}"
    );
    assert_eq!(
        instructions.last().map(String::as_str),
        Some(VAGUE_ACTION),
        "the last attempt is the planner's own instruction again: {instructions:?}"
    );
}

#[tokio::test]
async fn the_user_sees_which_element_a_vague_instruction_was_re_anchored_to() {
    // The transcript is how a user finds out why a different thing got clicked.
    let (registry, _act) = results_page(Some(DESPACITO_VIDEO_PREFIX));
    let stub =
        lucy_agent::testing::StubProvider::new().push_json(an_unresolvable_click_objective());
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AgentEvent>();
    let deps = deps_with_events(&stub, &registry, AgentBudget::default(), Some(tx.clone()));

    let _out = run_agent_loop_with(&deps, "play despacito on youtube")
        .await
        .expect("loop runs");
    drop(tx);

    let statuses: Vec<String> = std::iter::from_fn(|| rx.try_recv().ok())
        .filter_map(|e| match e {
            AgentEvent::Status { message } => Some(message),
            _ => None,
        })
        .collect();
    let line = statuses
        .iter()
        .find(|s| s.contains("re-anchoring"))
        .unwrap_or_else(|| panic!("the re-anchoring has to be said out loud: {statuses:?}"));
    assert!(line.contains(VAGUE_ACTION), "the vague half: {line}");
    assert!(
        line.contains(DESPACITO_VIDEO_PREFIX),
        "the element half: {line}"
    );
}

#[tokio::test]
async fn a_type_objective_is_never_re_anchored_into_a_click() {
    // The dangerous shape: the same vague description on a `type` objective.
    // Re-anchoring it would replace "type despacito into the search box" with
    // a CLICK on a video link, which is not a worse version of the step, it is
    // a different step. So a `type` keeps its own instruction on every attempt.
    let (registry, act) = results_page(Some(DESPACITO_VIDEO_PREFIX));
    let presses = Arc::new(AtomicUsize::new(0));
    let mut r = registry;
    r.register_arc(Arc::new(LandedPress {
        presses: presses.clone(),
    }));
    let stub = lucy_agent::testing::StubProvider::new().push_json(json!({"objectives":[
        {"description":"type despacito into the search box",
         "success_check":"search results for despacito are displayed",
         "success_probe":PLAYING_PROBE,
         "suggested_action":"type despacito into the video result search box",
         "action":"type","text":"despacito"}
    ]}));
    let mut budget = AgentBudget::default();
    budget.replan_on_failure = false;
    let (deps, _intr) = harness(&stub, &r, budget);

    let out = run_agent_loop_with(&deps, "play despacito on youtube")
        .await
        .expect("loop runs");

    assert!(!out.complete, "the probe never held: {out}");
    let instructions = act.instructions();
    assert!(
        instructions.len() >= 2,
        "the type objective really was retried: {instructions:?}"
    );
    for instruction in &instructions {
        assert_eq!(
            instruction, "type despacito into the video result search box",
            "a type objective must never be rewritten into a click: {instructions:?}"
        );
    }
    assert!(
        presses.load(Ordering::SeqCst) >= 1,
        "and it was still submitted with Enter rather than clicked: {out}"
    );
}

#[tokio::test]
async fn a_press_objective_is_never_re_anchored() {
    // Same gate on the other verb. A press has no target to re-aim — the key IS
    // the step — and "press enter to submit the despacito search" rewritten as a
    // click on the search box would be a step nobody planned.
    let (registry, act) = results_page(Some(DESPACITO_VIDEO_PREFIX));
    let presses = Arc::new(AtomicUsize::new(0));
    let mut r = registry;
    r.register_arc(Arc::new(LandedPress {
        presses: presses.clone(),
    }));
    let stub = lucy_agent::testing::StubProvider::new().push_json(json!({"objectives":[
        {"description":"submit the despacito search",
         "success_check":"search results for despacito are displayed",
         "success_probe":PLAYING_PROBE,
         "suggested_action":"press enter to submit the despacito search",
         "action":"press"}
    ]}));
    let mut budget = AgentBudget::default();
    budget.replan_on_failure = false;
    let (deps, _intr) = harness(&stub, &r, budget);

    let out = run_agent_loop_with(&deps, "play despacito on youtube")
        .await
        .expect("loop runs");

    assert!(!out.complete, "the probe never held: {out}");
    assert!(
        presses.load(Ordering::SeqCst) >= 2,
        "the press was retried on its own: {out}"
    );
    assert!(
        act.instructions().is_empty(),
        "a press never reaches hint_act, so it can never be re-anchored: {:?}",
        act.instructions()
    );
}

/// The prompt has to say the loop does this, or the model will keep trying to
/// guess a string it has never seen — which is what the description was.
#[test]
fn the_plan_prompt_states_that_a_description_is_enough() {
    assert!(
        AGENT_PLAN_INSTRUCTIONS.contains("re-anchors that instruction once"),
        "the re-anchoring has to be stated as a fact the model can rely on: {}",
        AGENT_PLAN_INSTRUCTIONS
    );
    assert!(
        AGENT_PLAN_INSTRUCTIONS.contains("Do NOT try to guess an exact string"),
        "and the corollary — describing the target is enough: {}",
        AGENT_PLAN_INSTRUCTIONS
    );
}
