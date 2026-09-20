//! Lucy harness v2: minimal-call, zero-ambiguity computer-use loop.
//!
//! Implements `docs/lucy-harness-architecture.md`:
//! - immutable session [`GoalObject`] (§7) that no replan may drop (§4, §4.6)
//! - deterministic browser [`preflight_browser`] (§4.2, 0 LLM calls)
//! - full-sequence native `tool_calls` plans ending in observation (§4.3)
//! - [`ExecutionTrace`] tool runtime without per-step LLM verifies (§4.4)
//! - [`deterministic_recovery`] table before any Recovery call (§4.5)
//! - scoped recovery + terminal verifier (`complete`|`recover` only, §4.6–§4.7)
//! - batch-tool preference + launch suppression via offered-schema filtering
//!   (§11.4.1, §6), backed by `HyprFastCatalog::planner_tool_set`
//! - strict response validation (§7) and per-task LLM call budgets (§10.8).

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use lucy_config::BrowserConfig;
use lucy_core::ToolCall;
use lucy_hyprfast::HyprFastCatalog;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// §7 Session goal object — immutable for the run.
// ---------------------------------------------------------------------------

/// Immutable acceptance test for the whole run, created once by the
/// Router+Planner and read by recovery + verifier. Never rewritten.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GoalObject {
    pub goal_id: String,
    pub goal_statement: String,
    pub success_condition: String,
    pub created_at: String,
    pub domains: Vec<String>,
}

impl GoalObject {
    pub fn new(
        goal_statement: impl Into<String>,
        success_condition: impl Into<String>,
        domains: Vec<String>,
    ) -> Result<Self> {
        let goal_statement = goal_statement.into();
        let success_condition = success_condition.into();
        if goal_statement.trim().is_empty() {
            return Err(anyhow!("goal_statement must not be empty"));
        }
        if success_condition.trim().is_empty() {
            return Err(anyhow!("success_condition must not be empty"));
        }
        Ok(Self {
            goal_id: uuid::Uuid::new_v4().to_string(),
            goal_statement,
            success_condition,
            created_at: iso8601_now(),
            domains,
        })
    }
}

/// Minimal UTC ISO-8601 clock without extra dependencies.
fn iso8601_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    iso8601_from_secs(secs)
}

fn iso8601_from_secs(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = (secs % 86_400) as i64;
    let (y, m, d) = civil_from_days(days + 719_468);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y,
        m,
        d,
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// Howard Hinnant's civil_from_days; valid for the full u64 epoch range.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

// ---------------------------------------------------------------------------
// §4.2 Deterministic preflight (code, 0 LLM calls).
// ---------------------------------------------------------------------------

/// Preflight outcome for one domain group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreflightState {
    Ready,
    Failed(String),
}

/// True when something answers on the CDP port (TCP connect succeeds).
pub fn cdp_port_responds(port: u16) -> bool {
    let addr = format!("127.0.0.1:{port}");
    std::net::TcpStream::connect_timeout(
        &addr.parse().unwrap_or("127.0.0.1:9222".parse().unwrap()),
        Duration::from_millis(250),
    )
    .is_ok()
}

/// True when a process command line carries the expected debug-port flag.
pub fn has_debug_flag(args: &str, port: u16) -> bool {
    args.contains(&format!("--remote-debugging-port={port}"))
        || args.contains("--remote-debugging-port")
}

/// Parse `ps -eo args` output for the first line mentioning a known browser
/// binary. Returns (binary, full_args). Pure for testability.
pub fn parse_browser_process(
    ps_output: &str,
    binaries: &[String],
) -> Option<(String, String)> {
    for line in ps_output.lines() {
        let lower = line.to_ascii_lowercase();
        for bin in binaries {
            let b = bin.to_ascii_lowercase();
            if lower.contains(b.as_str()) {
                return Some((bin.clone(), line.to_owned()));
            }
        }
    }
    None
}

/// Candidate binaries in priority order: explicit binary first, then the
/// configured fallbacks (§4.5: consult the config list, never LLM-guess).
pub fn browser_candidates(cfg: &BrowserConfig, resolved: &str) -> Vec<String> {
    let mut out = vec![resolved.to_owned()];
    for b in &cfg.fallback_binaries {
        if !out.iter().any(|x| x == b) {
            out.push(b.clone());
        }
    }
    out
}

/// Best-effort running-browser discovery via `ps`. Returns (binary, args).
pub fn find_running_browser(binaries: &[String]) -> Option<(String, String)> {
    let out = std::process::Command::new("ps")
        .args(["-eo", "args"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    parse_browser_process(&text, binaries)
}

/// Launch the browser detached with the CDP flag; returns immediately while
/// the caller polls the port (§4.2: no race — poll, don't check once).
pub fn launch_browser(binary: &str, port: u16, extra_args: &[String]) -> Result<()> {
    let mut cmd = std::process::Command::new(binary);
    cmd.arg(format!("--remote-debugging-port={port}"));
    for a in extra_args {
        cmd.arg(a);
    }
    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        cmd.arg("--no-first-run");
        unsafe {
            cmd.pre_exec(|| {
                libc_setsid();
                Ok(())
            });
        }
    }
    cmd.spawn().map(|_| ()).map_err(|e| {
        anyhow!("failed to launch browser binary '{binary}': {e}")
    })
}

#[cfg(unix)]
fn libc_setsid() {
    unsafe {
        // Minimal setsid without pulling in a libc crate: raw syscall
        // number for setsid is 112 on x86_64/aarch64 Linux.
        #[cfg(target_arch = "x86_64")]
        const SYS_SETSID: i64 = 112;
        #[cfg(not(target_arch = "x86_64"))]
        const SYS_SETSID: i64 = 112;
        core::arch::asm!(
            "syscall",
            in("rax") SYS_SETSID,
            out("rcx") _, out("r11") _,
            lateout("rax") _,
        );
    }
}

/// Async CDP readiness poll: backgrounds launch, then polls the port rather
/// than checking once immediately after `&` (§4.2 race fix).
pub async fn poll_cdp_ready(port: u16, timeout: Duration, interval: Duration) -> bool {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if cdp_port_responds(port) {
            return true;
        }
        if tokio::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(interval).await;
    }
}

/// §4.2 browser preflight state machine: check → repair-if-needed →
/// poll-until-ready → READY/FAILED. Idempotent: one syscall when healthy.
pub async fn preflight_browser(
    cfg: &BrowserConfig,
    resolved_binary: &str,
) -> PreflightState {
    if cdp_port_responds(cfg.cdp_port) {
        return PreflightState::Ready;
    }
    let candidates = browser_candidates(cfg, resolved_binary);
    match find_running_browser(&candidates) {
        Some((bin, args)) if !has_debug_flag(&args, cfg.cdp_port) => {
            // Running without the flag: relaunch the same binary with it.
            let _ = std::process::Command::new("pkill")
                .args(["-f", &bin])
                .output();
            if launch_browser(&bin, cfg.cdp_port, &cfg.launch_args).is_err() {
                // Fall through to candidate walk below on launch failure.
            } else if poll_cdp_ready(
                cfg.cdp_port,
                Duration::from_secs(cfg.launch_timeout_secs.max(1)),
                Duration::from_millis(250),
            )
            .await
            {
                return PreflightState::Ready;
            }
        }
        Some(_) => {
            // Process claims the flag but the port is dark: transient —
            // poll once more before relaunching.
            let settled = poll_cdp_ready(
                cfg.cdp_port,
                Duration::from_secs(2),
                Duration::from_millis(250),
            )
            .await;
            if settled {
                return PreflightState::Ready;
            }
        }
        None => {}
    }
    // No usable process: walk the config-driven binary list in order.
    for bin in &candidates {
        if launch_browser(bin, cfg.cdp_port, &cfg.launch_args).is_ok()
            && poll_cdp_ready(
                cfg.cdp_port,
                Duration::from_secs(cfg.launch_timeout_secs.max(1)),
                Duration::from_millis(250),
            )
            .await
        {
            return PreflightState::Ready;
        }
    }
    PreflightState::Failed(format!(
        "browser did not come up with CDP on port {} after launch",
        cfg.cdp_port
    ))
}

/// Domain-scoped preflight: only browser-ish domains trigger the browser
/// state machine; every other domain is READY by default (§4.2).
pub async fn preflight_for_domains(
    domains: &[String],
    cfg: &BrowserConfig,
    resolved_binary: &str,
) -> PreflightState {
    let needs_browser = domains
        .iter()
        .any(|d| matches!(d.to_ascii_lowercase().as_str(), "browser" | "stagehand" | "hints" | "vision"));
    if !needs_browser {
        return PreflightState::Ready;
    }
    preflight_browser(cfg, resolved_binary).await
}

// ---------------------------------------------------------------------------
// §4.3 Planner output: parse + validate.
// ---------------------------------------------------------------------------

/// Router+Planner text envelope (§11.1 Step 2). The native `tool_calls`
/// array is read from the provider response, not from this JSON.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct RouterEnvelope {
    #[serde(default)]
    pub mode: String,
    #[serde(default)]
    pub reply: Option<String>,
    #[serde(default)]
    pub goal_statement: Option<String>,
    #[serde(default)]
    pub success_condition: Option<String>,
}

/// Parse the planner text envelope. Falls back to deriving the goal from
/// the user prompt when the model returns bare prose (reject-and-retry is
/// handled by the caller; this never fails).
pub fn parse_router_envelope(text: Option<&str>, fallback_prompt: &str) -> RouterEnvelope {
    let fallback_goal = fallback_prompt.split_whitespace().collect::<Vec<_>>().join(" ");
    let fallback_goal = fallback_goal.chars().take(200).collect::<String>();
    let Some(raw) = text.map(str::trim).filter(|s| !s.is_empty()) else {
        return RouterEnvelope {
            mode: "act".into(),
            reply: None,
            goal_statement: Some(if fallback_goal.is_empty() {
                "Complete the user's request.".into()
            } else {
                fallback_goal.clone()
            }),
            success_condition: Some(if fallback_goal.is_empty() {
                "The user's requested outcome is observably true.".into()
            } else {
                format!("The requested outcome is observably true: {fallback_goal}")
            }),
        };
    };
    if let Ok(v) = serde_json::from_str::<Value>(raw)
        && let Ok(env) = serde_json::from_value::<RouterEnvelope>(v)
    {
        return normalize_envelope(env, &fallback_goal);
    }
    if let Some(obj) = largest_balanced_object(raw)
        && let Ok(v) = serde_json::from_str::<Value>(&obj)
        && let Ok(env) = serde_json::from_value::<RouterEnvelope>(v)
    {
        return normalize_envelope(env, &fallback_goal);
    }
    // Bare prose with no JSON: treat as chat (router) — never hallucinate
    // an act plan from prose.
    RouterEnvelope {
        mode: "chat".into(),
        reply: Some(raw.to_owned()),
        goal_statement: None,
        success_condition: None,
    }
}

fn normalize_envelope(mut env: RouterEnvelope, fallback_goal: &str) -> RouterEnvelope {
    env.mode = env.mode.to_ascii_lowercase();
    if env.mode != "chat" && env.mode != "act" {
        env.mode = if env.goal_statement.is_some() || env.success_condition.is_some() {
            "act".into()
        } else {
            "chat".into()
        };
    }
    if env.mode == "act" {
        if env.goal_statement.as_deref().map(str::trim).unwrap_or("").is_empty() {
            env.goal_statement = Some(if fallback_goal.is_empty() {
                "Complete the user's request.".into()
            } else {
                fallback_goal.to_owned()
            });
        }
        if env.success_condition.as_deref().map(str::trim).unwrap_or("").is_empty() {
            env.success_condition = Some(format!(
                "The requested outcome is observably true: {}",
                env.goal_statement.clone().unwrap_or_default()
            ));
        }
    }
    env
}

fn largest_balanced_object(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut best: Option<(usize, usize)> = None;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'{' {
            i += 1;
            continue;
        }
        let mut depth = 0i32;
        let mut in_str = false;
        let mut esc = false;
        let mut j = i;
        while j < bytes.len() {
            let b = bytes[j];
            if in_str {
                if esc {
                    esc = false;
                } else if b == b'\\' {
                    esc = true;
                } else if b == b'"' {
                    in_str = false;
                }
            } else if b == b'"' {
                in_str = true;
            } else if b == b'{' {
                depth += 1;
            } else if b == b'}' {
                depth -= 1;
                if depth == 0 {
                    match best {
                        Some((_, len)) if j + 1 - i <= len => {}
                        _ => best = Some((i, j + 1 - i)),
                    }
                    break;
                }
            }
            j += 1;
        }
        i += 1;
    }
    best.map(|(s, l)| text[s..s + l].to_owned())
}

/// Name-heuristic observation check used when no catalog entry exists
/// (e.g. local `read_file`/`shell` outputs that already are observations).
pub fn name_looks_like_observation(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.contains("snapshot")
        || n.contains("extract")
        || n.contains("observe")
        || n.contains("get_ui")
        || n.contains("ui_tree")
        || n.contains("find_element")
        || n.contains("screenshot")
        || n.contains("read_file")
        || n.contains("list_dir")
        || n.contains("search_files")
        || n == "shell"
        || n == "git"
}

/// Name-heuristic launch check (mirrors `is_launch_tool` without catalog).
pub fn name_looks_like_launch(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.contains("browser_open") || n.contains("browser_launch") || n.contains("launch")
}

/// Validate a planner/recovery `tool_calls` array (§4.3 + §4.6 invariants):
/// non-empty, all args are objects, no launch when a session already
/// exists, and the final step is a read-only observation.
pub fn validate_tool_plan(
    calls: &[ToolCall],
    catalog: Option<&HyprFastCatalog>,
    browser_ready: bool,
) -> Result<()> {
    if calls.is_empty() {
        return Err(anyhow!("planner returned an empty tool_calls array"));
    }
    for c in calls {
        if c.name.trim().is_empty() {
            return Err(anyhow!("planner returned a tool call with an empty name"));
        }
        if !c.input.is_object() {
            return Err(anyhow!(
                "planner tool call '{}' arguments must be a JSON object",
                c.name
            ));
        }
        if browser_ready && name_looks_like_launch(&c.name) {
            return Err(anyhow!(
                "planner offered a launch tool ('{}') while a usable session exists; 'Open X' must reuse the session",
                c.name
            ));
        }
    }
    let last = calls.last().expect("non-empty");
    let is_obs = match catalog.and_then(|c| c.capability_for_mcp_name(&last.name)) {
        Some(cap) => {
            cap.read_only
                || lucy_hyprfast::is_observation_tool(cap)
                || name_looks_like_observation(&last.name)
        }
        None => name_looks_like_observation(&last.name),
    };
    if !is_obs {
        return Err(anyhow!(
            "planner sequence must end in a read-only observation step, got '{}'",
            last.name
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// §4.4 Tool runtime trace.
// ---------------------------------------------------------------------------

/// One executed step inside an [`ExecutionTrace`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StepResult {
    pub tool: String,
    pub status: String,
    #[serde(default)]
    pub result: Value,
}

/// Full ordered execution of one planner/recovery `tool_calls` array (code,
/// 0 LLM calls). `deviated == true` means deterministic recovery (§4.5)
/// must run before any Recovery LLM call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExecutionTrace {
    pub steps: Vec<StepResult>,
    #[serde(default)]
    pub duration_ms: u64,
    #[serde(default)]
    pub deviated: bool,
}

impl ExecutionTrace {
    pub fn new(steps: Vec<StepResult>, duration_ms: u64) -> Self {
        let deviated = steps.iter().any(|s| s.status != "ok");
        Self {
            steps,
            duration_ms,
            deviated,
        }
    }

    /// Final observation for the verifier: last successful result, else the
    /// last error payload (§4.3: the plan ends in observation, so this is
    /// already in hand with no extra round-trip).
    pub fn final_observation(&self) -> Value {
        for s in self.steps.iter().rev() {
            if s.status == "ok" {
                return s.result.clone();
            }
        }
        self.steps.last().map(|s| s.result.clone()).unwrap_or(Value::Null)
    }

    pub fn completed_summary(&self) -> String {
        self.steps
            .iter()
            .filter(|s| s.status == "ok")
            .map(|s| s.tool.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

// ---------------------------------------------------------------------------
// §4.5 Deterministic recovery table (first line of defense, 0 LLM calls).
// ---------------------------------------------------------------------------

/// Deterministic fix for a deviated step. Only `Escalate` costs an LLM call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryFix {
    /// Retry the same step once (timeouts get 2x in the caller).
    RetrySame,
    /// Wait for the page to settle, then retry once.
    WaitThenRetry,
    /// Re-run preflight (§4.2), then retry the same step once.
    RelaunchBrowserThenRetry,
    /// Not a known shape (or the deterministic retry already failed):
    /// escalate to the scoped Recovery LLM call.
    Escalate,
}

fn contains_any(hay: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| hay.contains(n))
}

/// Classify a step error into the §4.5 table. `page_loading` should be true
/// when `document.readyState != complete` was observed alongside the error.
pub fn deterministic_recovery(error_text: &str, page_loading: bool) -> RecoveryFix {
    let e = error_text.to_ascii_lowercase();
    if contains_any(
        &e,
        &["cdp unreachable", "cdp not", "websocket", "target closed", "browser has been closed", "no browser", "connection refused", "net::err_connection_refused"],
    ) || (e.contains("127.0.0.1:9222") || e.contains("localhost:9222"))
        && contains_any(&e, &["refused", "unreachable", "closed", "econnrefused"])
    {
        return RecoveryFix::RelaunchBrowserThenRetry;
    }
    if contains_any(&e, &["element not found", "no such element", "elementnotfound", "selector did not match", "could not find element", "node not found"])
        && (page_loading || contains_any(&e, &["loading", "readyState", "not yet rendered"]))
    {
        return RecoveryFix::WaitThenRetry;
    }
    if contains_any(&e, &["timeout", "timed out", "deadline exceeded", "etimeout"]) {
        return RecoveryFix::RetrySame;
    }
    if contains_any(&e, &["process not found", "no such process", "binary not found", "command not found", "failed to launch", "spawn"]) {
        return RecoveryFix::RelaunchBrowserThenRetry;
    }
    RecoveryFix::Escalate
}

// ---------------------------------------------------------------------------
// §4.6–§4.7 Recovery + verifier contracts.
// ---------------------------------------------------------------------------

/// Terminal verifier decision. There is no third "continue and do nothing"
/// branch: `recover` always carries the next action's reason (§4.7).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifierDecision {
    pub complete: bool,
    pub evidence: String,
    pub unmet_reason: Option<String>,
}

/// Strict validation of the verifier's JSON (§7 + §11.4.2). Rejects
/// `{"decision":"continue",...}`, empty plans, and `recover` without a
/// non-null next action.
pub fn parse_verifier(value: &Value) -> Result<VerifierDecision> {
    let decision = value
        .get("decision")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("verifier output must contain a 'decision' string"))?;
    let evidence = value
        .get("evidence")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("verifier output must contain non-empty 'evidence'"))?;
    match decision {
        "complete" => Ok(VerifierDecision {
            complete: true,
            evidence: evidence.to_owned(),
            unmet_reason: None,
        }),
        "recover" => {
            let reason = value
                .get("unmet_reason")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    anyhow!("verifier 'recover' must include a non-null 'unmet_reason' next action")
                })?;
            Ok(VerifierDecision {
                complete: false,
                evidence: evidence.to_owned(),
                unmet_reason: Some(reason.to_owned()),
            })
        }
        other => Err(anyhow!(
            "verifier decision must be 'complete' or 'recover', got '{other}' — there is no 'continue' branch"
        )),
    }
}

/// Render the scoped-recovery user message (§11.2) with fixed context.
pub fn recovery_user_message(
    goal: &GoalObject,
    completed_summary: &str,
    failed_step: &str,
    error_detail: &Value,
) -> String {
    let completed = if completed_summary.trim().is_empty() {
        "(none — the first step failed)"
    } else {
        completed_summary
    };
    format!(
        "Original goal: {goal}\nSuccess condition (unchanged): {cond}\nSteps already completed successfully: {completed}\nThe step that deviated: {failed}\nObserved error/state: {err}",
        goal = goal.goal_statement,
        cond = goal.success_condition,
        failed = failed_step,
        err = error_detail,
    )
}

/// Render the verifier user message (§11.3) with fixed context.
pub fn verifier_user_message(success_condition: &str, final_observation: &Value) -> String {
    let mut obs = final_observation.to_string();
    if obs.len() > 6000 {
        obs.truncate(6000);
        obs.push_str("…[truncated]");
    }
    format!("Success condition: {success_condition}\nFinal observation: {obs}")
}

/// Recovery invariant (§4.6, enforced by the runtime not the prompt): the
/// replacement sequence must be non-empty, end in observation, and must not
/// silently substitute the goal (a bare relaunch without follow-up work is
/// rejected — "browser is now open" is not the goal).
pub fn validate_recovery_calls(
    calls: &[ToolCall],
    catalog: Option<&HyprFastCatalog>,
    browser_ready: bool,
    goal: &GoalObject,
) -> Result<()> {
    validate_tool_plan(calls, catalog, browser_ready)?;
    if calls.len() == 1 && name_looks_like_launch(&calls[0].name) {
        return Err(anyhow!(
            "recovery must lead back to the success condition ('{}'), not stop at environment fix-up",
            goal.success_condition
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// §10.8 Call budgets — the number that keeps the system honest.
// ---------------------------------------------------------------------------

/// Per-task LLM call counter. Routine single-app tasks must stay within
/// `limit` (§3: 2 healthy, 3–4 with one genuine deviation).
#[derive(Debug, Clone)]
pub struct CallBudget {
    pub limit: usize,
    pub calls: usize,
    pub labels: Vec<String>,
}

impl CallBudget {
    pub fn new(limit: usize) -> Self {
        Self {
            limit: limit.max(1),
            calls: 0,
            labels: Vec::new(),
        }
    }

    pub fn record(&mut self, label: impl Into<String>) {
        self.calls += 1;
        self.labels.push(label.into());
    }

    pub fn over_budget(&self) -> bool {
        self.calls > self.limit
    }
}

/// Compact capability summary for the Router prompt (§4.1: domain names +
/// counts only, never full schemas).
pub fn capability_summary_line(catalog: &HyprFastCatalog) -> String {
    catalog.domain_summary_line()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn browser_cfg() -> BrowserConfig {
        BrowserConfig {
            binary: None,
            fallback_binaries: vec!["brave".into(), "google-chrome".into()],
            cdp_port: 19222,
            launch_timeout_secs: 1,
            launch_args: vec![],
        }
    }

    fn call(name: &str) -> ToolCall {
        ToolCall {
            id: "1".into(),
            name: name.into(),
            input: serde_json::json!({}),
        }
    }

    #[test]
    fn goal_rejects_empty_fields() {
        assert!(GoalObject::new("", "x", vec![]).is_err());
        assert!(GoalObject::new("x", "  ", vec![]).is_err());
        let g = GoalObject::new("Play Despacito", "video is playing", vec!["browser".into()]).unwrap();
        assert!(!g.goal_id.is_empty());
        assert!(g.created_at.ends_with('Z'));
    }

    #[test]
    fn iso8601_epoch_is_sane() {
        assert_eq!(iso8601_from_secs(0), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn parses_browser_process_lines() {
        let ps = "  PID ARGS\n  123 /usr/bin/brave --remote-debugging-port=9222\n  456 /usr/bin/code\n";
        let found = parse_browser_process(ps, &["brave".into(), "code".into()]).unwrap();
        assert_eq!(found.0, "brave");
        assert!(has_debug_flag(&found.1, 9222));
        assert!(!has_debug_flag("/usr/bin/code --new-window", 9222));
    }

    #[test]
    fn candidates_put_resolved_first_without_dupes() {
        let cfg = browser_cfg();
        let c = browser_candidates(&cfg, "google-chrome");
        assert_eq!(c[0], "google-chrome");
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn envelope_parses_act_and_chat() {
        let act = parse_router_envelope(
            Some(r#"{"mode":"act","goal_statement":"Play X","success_condition":"video playing"}"#),
            "fallback",
        );
        assert_eq!(act.mode, "act");
        assert_eq!(act.goal_statement.unwrap(), "Play X");
        let chat = parse_router_envelope(Some(r#"{"mode":"chat","reply":"hi"}"#), "fallback");
        assert_eq!(chat.mode, "chat");
        let prose = parse_router_envelope(Some("Hello there"), "fallback");
        assert_eq!(prose.mode, "chat");
        let missing = parse_router_envelope(None, "do the thing");
        assert_eq!(missing.mode, "act");
        assert!(missing.goal_statement.unwrap().contains("do the thing"));
    }

    #[test]
    fn plan_must_end_in_observation() {
        assert!(validate_tool_plan(&[call("mcp_hyprfast_browser_snapshot")], None, false).is_ok());
        assert!(validate_tool_plan(&[call("mcp_hyprfast_browser_navigate")], None, false).is_err());
        assert!(validate_tool_plan(&[], None, false).is_err());
    }

    #[test]
    fn plan_rejects_launch_when_session_ready() {
        let calls = vec![call("mcp_hyprfast_browser_launch"), call("mcp_hyprfast_browser_snapshot")];
        assert!(validate_tool_plan(&calls, None, true).is_err());
        assert!(validate_tool_plan(&calls, None, false).is_ok());
    }

    #[test]
    fn recovery_table_routes_known_signals() {
        assert_eq!(
            deterministic_recovery("CDP unreachable: connection refused 127.0.0.1:9222", false),
            RecoveryFix::RelaunchBrowserThenRetry
        );
        assert_eq!(
            deterministic_recovery("element not found while page loading readyState", true),
            RecoveryFix::WaitThenRetry
        );
        assert_eq!(
            deterministic_recovery("tool timed out after 30s", false),
            RecoveryFix::RetrySame
        );
        assert_eq!(
            deterministic_recovery("process not found for launch", false),
            RecoveryFix::RelaunchBrowserThenRetry
        );
        assert_eq!(
            deterministic_recovery("weird unknown shape", false),
            RecoveryFix::Escalate
        );
    }

    #[test]
    fn verifier_allows_only_complete_or_recover() {
        let ok = parse_verifier(&serde_json::json!({"decision":"complete","evidence":"video title matches"})).unwrap();
        assert!(ok.complete);
        let rec = parse_verifier(&serde_json::json!({"decision":"recover","evidence":"no video","unmet_reason":"page shows search, nothing playing"})).unwrap();
        assert!(!rec.complete);
        assert!(parse_verifier(&serde_json::json!({"decision":"continue"})).is_err());
        assert!(parse_verifier(&serde_json::json!({"decision":"recover","evidence":"x"})).is_err());
        assert!(parse_verifier(&serde_json::json!({"decision":"recover","evidence":"x","unmet_reason":"  "})).is_err());
    }

    #[test]
    fn recovery_rejects_bare_relaunch() {
        let goal = GoalObject::new("Play X", "video is playing", vec!["browser".into()]).unwrap();
        assert!(validate_recovery_calls(&[call("mcp_hyprfast_browser_launch")], None, false, &goal).is_err());
        assert!(validate_recovery_calls(
            &[call("mcp_hyprfast_browser_navigate"), call("mcp_hyprfast_browser_snapshot")],
            None,
            false,
            &goal
        )
        .is_ok());
    }

    #[test]
    fn trace_final_observation_prefers_last_ok() {
        let t = ExecutionTrace::new(
            vec![
                StepResult { tool: "a".into(), status: "ok".into(), result: serde_json::json!({"v":1}) },
                StepResult { tool: "b".into(), status: "ok".into(), result: serde_json::json!({"v":2}) },
            ],
            10,
        );
        assert!(!t.deviated);
        assert_eq!(t.final_observation(), serde_json::json!({"v":2}));
        let bad = ExecutionTrace::new(
            vec![StepResult { tool: "a".into(), status: "error".into(), result: serde_json::json!({"error":"x"}) }],
            5,
        );
        assert!(bad.deviated);
    }

    #[test]
    fn budget_flags_overruns() {
        let mut b = CallBudget::new(4);
        for i in 0..4 {
            b.record(format!("call{i}"));
        }
        assert!(!b.over_budget());
        b.record("one-too-many");
        assert!(b.over_budget());
    }
}
