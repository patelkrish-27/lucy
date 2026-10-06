use anyhow::{Context, Result, anyhow};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};
use tracing::{info, warn};

use crate::metrics::BrowserMetrics;

pub const SNAPSHOT_JS: &str = include_str!("snapshot.js");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserAction {
    pub id: String,
    #[serde(default)]
    pub node: Option<i64>,
    pub kind: String,
    pub role: Option<String>,
    pub label: String,
    pub value: Option<String>,
    pub current_value: Option<String>,
    pub checked: Option<String>,
    pub expanded: Option<String>,
    pub hint: Option<String>,
    pub rect: Option<Rect>,
    pub delta: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserPageSnapshot {
    pub url: String,
    pub title: String,
    pub w: u64,
    pub h: u64,
    pub text: String,
    pub actions: Vec<BrowserAction>,
    pub marker: Value,
    pub page_key: Value,
    #[serde(default)]
    pub guards: HashMap<String, Value>,
    #[serde(default)]
    pub omitted_actions: usize,
}

type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// Lucy's CDP session against the browser hyprfast owns.
///
/// This client never launches a browser. hyprfast is the single launcher:
/// `hyprfast browser open <url>` probes the CDP port, launches with an
/// isolated profile when it is closed, and appends
/// [`HYPRFAST_BROWSER_ARGS_ENV`] to whatever it spawns. Lucy's job is to make
/// sure that launcher has run, then attach. A second spawn path here would
/// mean a second browser with a second profile — and whichever lost the race
/// would silently drive the wrong instance.
pub struct BrowserCdpClient {
    ws: Arc<Mutex<Option<WsStream>>>,
    request_id: AtomicU64,
    target_id: Arc<Mutex<Option<String>>>,
    cdp_port: u16,
    hyprfast_cmd: String,
    launch_timeout: Duration,
    launch_args: Vec<String>,
    metrics: Arc<BrowserMetrics>,
}

impl BrowserCdpClient {
    pub fn new(cdp_port: u16) -> Self {
        Self::from_browser_config(cdp_port, "hyprfast", &lucy_config::BrowserConfig::default())
    }

    /// Build a client that delegates launching to `hyprfast_cmd`.
    pub fn from_browser_config(
        cdp_port: u16,
        hyprfast_cmd: &str,
        browser: &lucy_config::BrowserConfig,
    ) -> Self {
        Self {
            ws: Arc::new(Mutex::new(None)),
            request_id: AtomicU64::new(1),
            target_id: Arc::new(Mutex::new(None)),
            cdp_port,
            hyprfast_cmd: hyprfast_cmd.to_string(),
            launch_timeout: Duration::from_secs(browser.launch_timeout_secs.max(1)),
            launch_args: browser.launch_args.clone(),
            metrics: Arc::new(BrowserMetrics::default()),
        }
    }

    /// Share one honesty-harness counter set (P0) with this client.
    /// Counts from `connect()` (daemon start aside) accumulate into it.
    pub fn with_metrics(mut self, metrics: Arc<BrowserMetrics>) -> Self {
        self.metrics = metrics;
        self
    }

    /// Live P0 counters for this client's CDP session.
    pub fn metrics(&self) -> &Arc<BrowserMetrics> {
        &self.metrics
    }

    /// Ask hyprfast to open the browser, and let *it* decide whether that
    /// means launching or navigating what is already running.
    ///
    /// hyprfast owns the launch command line, the isolated profile and the
    /// CDP-readiness poll, so lucy contributes two things only: the URL, and
    /// the environment that keeps the launch consistent with lucy's config —
    /// the CDP port (which port to open) and the launch flags (GPU policy).
    /// Both are passed as env because the launcher is a separate process that
    /// lucy does not link against.
    async fn open_via_hyprfast(&self, url: &str) -> std::process::Output {
        use std::process::Stdio;
        let mut cmd = tokio::process::Command::new(&self.hyprfast_cmd);
        cmd.args(["browser", "open", url])
            .env(lucy_config::HYPRFAST_CDP_HOST_ENV, "127.0.0.1")
            .env(
                lucy_config::HYPRFAST_CDP_PORT_ENV,
                self.cdp_port.to_string(),
            )
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(false);
        // Unset means "browser defaults" on the hyprfast side, which is not
        // what an explicitly emptied list means. Passing it through either
        // way keeps one code path.
        if let Some(flags) = lucy_config::join_flags(&self.launch_args) {
            cmd.env(lucy_config::HYPRFAST_BROWSER_ARGS_ENV, flags);
        }
        cmd.output().await.unwrap_or_else(|e| {
            // A failed spawn has no Output to return; synthesise one so the
            // caller's diagnostics report the spawn error instead of a
            // misleading success-shaped value.
            use std::os::unix::process::ExitStatusExt;
            std::process::Output {
                status: std::process::ExitStatus::from_raw(127 << 8),
                stdout: Vec::new(),
                stderr: format!("failed to run `{} browser open`: {e}", self.hyprfast_cmd)
                    .into_bytes(),
            }
        })
    }

    /// Connect to the CDP endpoint hyprfast manages.
    /// If target_url is provided, creates or navigates to that URL.
    pub async fn connect(&self, target_url: Option<&str>) -> Result<()> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_millis(1500))
            .build()?;

        let base_url = format!("http://127.0.0.1:{}", self.cdp_port);

        // 0.9: best-effort start of the persistent browser-runtime daemon so
        // `hint-*` uses the single-WS path instead of
        // degraded direct mode. Never fail connect if the daemon is missing.
        {
            use std::process::Stdio;
            let mut cmd = tokio::process::Command::new(&self.hyprfast_cmd);
            cmd.args(["browser-runtime", "start"])
                .env(lucy_config::HYPRFAST_CDP_HOST_ENV, "127.0.0.1")
                .env(
                    lucy_config::HYPRFAST_CDP_PORT_ENV,
                    self.cdp_port.to_string(),
                )
                .stdout(Stdio::null())
                .stderr(Stdio::null());
            // The daemon respawns the browser on crash; without this it would
            // come back with different process flags than the launch below.
            if let Some(flags) = lucy_config::join_flags(&self.launch_args) {
                cmd.env(lucy_config::HYPRFAST_BROWSER_ARGS_ENV, flags);
            }
            let start = cmd.output();
            let _ = tokio::time::timeout(Duration::from_secs(8), start).await;
        }

        // 1. Check whether a browser is already listening. If not, hyprfast is
        // the one that launches it — see `open_via_hyprfast`.
        let version_url = format!("{base_url}/json/version");
        let initial_probe_err = match http.get(&version_url).send().await {
            Ok(_) => None,
            Err(e) => Some(e.to_string()),
        };
        if initial_probe_err.is_some() {
            let url = target_url.unwrap_or("about:blank");
            let profile_dir = hyprfast_profile_dir(self.cdp_port);
            // The pref write has to happen before the launch, not after: Brave
            // reads hardware_acceleration_mode at startup, so a write racing
            // the spawn can land on a browser that already decided. hyprfast
            // creates the profile dir itself, but creating it here first makes
            // the ordering deterministic.
            let pref_state =
                disable_hardware_acceleration_pref(&profile_dir).unwrap_or_else(|e| format!("{e}"));
            info!(
                port = self.cdp_port,
                url = %url,
                profile = %profile_dir.display(),
                accel_pref = %pref_state,
                initial_err = ?initial_probe_err,
                "CDP port not reachable, delegating launch to hyprfast"
            );

            // hyprfast polls the same endpoint internally and returns when the
            // browser is up, so a failure here is already a diagnosis. Its
            // stderr is the useful part: "a Brave already running on the
            // default profile swallows the launch request" is the single most
            // common cause and is only visible from the launcher.
            let out = self.open_via_hyprfast(url).await;
            let launcher_out = format!(
                "stdout=[{}] stderr=[{}]",
                truncate_for_log(&String::from_utf8_lossy(&out.stdout)),
                truncate_for_log(&String::from_utf8_lossy(&out.stderr))
            );
            info!(port = self.cdp_port, status = ?out.status, "hyprfast browser open returned");

            // Second, cheap confirmation. hyprfast waits for the *page*, which
            // can outlast the launch timeout on a heavy site; the port itself
            // should already answer, and this separates "launch failed" from
            // "page still loading".
            let deadline = tokio::time::Instant::now() + self.launch_timeout;
            let mut attempts = 0u32;
            let mut last_poll_err = String::from("no attempts");
            let mut ready = false;
            while tokio::time::Instant::now() < deadline {
                tokio::time::sleep(Duration::from_millis(250)).await;
                attempts += 1;
                match http.get(&version_url).send().await {
                    Ok(_) => {
                        ready = true;
                        break;
                    }
                    Err(e) => last_poll_err = e.to_string(),
                }
            }
            if !ready {
                let diag = collect_cdp_diagnostics(
                    self.cdp_port,
                    &self.hyprfast_cmd,
                    &profile_dir,
                    &format!(
                        "post-launch-poll-failed attempts={attempts} last_err={last_poll_err} launcher={launcher_out}"
                    ),
                    Some(&last_poll_err),
                    initial_probe_err.as_deref(),
                );
                write_cdp_log(self.cdp_port, &diag);
                warn!(port = self.cdp_port, attempts, last_err = %last_poll_err, diag = %diag, "CDP port still unreachable after `hyprfast browser open`");
                return Err(anyhow!(
                    "`{cmd} browser open {url}` returned but CDP port {port} is still closed \
                     (attempts={attempts} last_err={last_poll_err}). Diagnostics: {diag}. \
                     Full log: /tmp/lucy-cdp-{port}.log",
                    cmd = self.hyprfast_cmd,
                    url = url,
                    port = self.cdp_port,
                ));
            }
            info!(
                port = self.cdp_port,
                attempts, "CDP port ready via hyprfast"
            );
        }

        // 2. Discover or create a dedicated owned tab (P1-4).
        // Tolerant dedup: if same host+path exists (YouTube only search_query),
        // reuse that tab but ALWAYS navigate it to target_url instead of just
        // activating. Otherwise create a new background tab via about:blank and
        // Page.navigate. Isolation comes from the dedicated --user-data-dir
        // profile (port-scoped) created above; no shared default profile.
        // 2. Dedicated owned tab. Prefer a tab already sitting on the target URL —
        // that is normally the one hyprfast's `browser open` just navigated,
        // since it is the only launcher — so attaching here does not leave a
        // duplicate behind. Otherwise own a fresh tab and navigate it.
        let (ws_url, navigate_url): (String, Option<String>) = if let Some(url) = target_url {
            let list_url = format!("{base_url}/json/list");
            // Tab dedup: tolerant host+path reuse.
            let mut reused: Option<(String, String)> = None;
            if let Ok(res) = http.get(&list_url).send().await {
                if let Ok(tabs) = res.json::<Vec<Value>>().await {
                    for t in &tabs {
                        let is_page = t.get("type").and_then(Value::as_str) == Some("page");
                        let tab_url = t.get("url").and_then(Value::as_str).unwrap_or("");
                        if is_page
                            && crate::hyprfast_browser::same_tab_for_reuse(tab_url, url)
                            && let (Some(tid), Some(ws)) = (
                                t.get("id").and_then(Value::as_str),
                                t.get("webSocketDebuggerUrl").and_then(Value::as_str),
                            )
                        {
                            reused = Some((tid.to_string(), ws.to_string()));
                            break;
                        }
                    }
                }
            }
            if let Some((tid, ws)) = reused {
                info!("Reusing existing browser tab for {url} (tolerant dedup, will navigate)");
                // Activate the reused tab (best-effort, awaited) then navigate via CDP below.
                let _ = http
                    .get(format!("{base_url}/json/activate/{tid}"))
                    .send()
                    .await;
                *self.target_id.lock().await = Some(tid);
                (ws, Some(url.to_string()))
            } else {
                // Dedicated owned tab: always create via about:blank then navigate.
                let blank_url = format!("{base_url}/json/new?about:blank");
                let res = http.put(&blank_url).send().await.with_context(|| {
                    format!(
                        "CDP tab-create PUT (about:blank) failed on port {} (target={url})",
                        self.cdp_port
                    )
                })?;
                let tab: Value = res.json().await.with_context(|| {
                    format!(
                        "CDP tab-create response was not JSON on port {}",
                        self.cdp_port
                    )
                })?;
                if let Some(tid) = tab.get("id").and_then(Value::as_str) {
                    *self.target_id.lock().await = Some(tid.to_string());
                }
                let ws = tab
                    .get("webSocketDebuggerUrl")
                    .and_then(Value::as_str)
                    .map(String::from)
                    .ok_or_else(|| {
                        anyhow!(
                            "missing webSocketDebuggerUrl in new tab response (port {}) tab={tab}",
                            self.cdp_port
                        )
                    })?;
                (ws, Some(url.to_string()))
            }
        } else {
            let list_url = format!("{base_url}/json/list");
            let res =
                http.get(&list_url).send().await.with_context(|| {
                    format!("CDP tab-list GET failed on port {}", self.cdp_port)
                })?;
            let tabs: Vec<Value> = res.json().await.with_context(|| {
                format!(
                    "CDP tab-list response was not JSON on port {}",
                    self.cdp_port
                )
            })?;
            let page_tab = tabs
                .into_iter()
                .find(|t| t.get("type").and_then(Value::as_str) == Some("page"))
                .ok_or_else(|| {
                    anyhow!(
                        "no open page target found in browser (port {})",
                        self.cdp_port
                    )
                })?;
            if let Some(tid) = page_tab.get("id").and_then(Value::as_str) {
                *self.target_id.lock().await = Some(tid.to_string());
            }
            let ws = page_tab
                .get("webSocketDebuggerUrl")
                .and_then(Value::as_str)
                .map(String::from)
                .ok_or_else(|| {
                    anyhow!(
                        "missing webSocketDebuggerUrl in target page (port {})",
                        self.cdp_port
                    )
                })?;
            (ws, None)
        };

        info!(ws_url = %ws_url, "Connecting to Chrome DevTools Protocol WebSocket");
        let (ws_stream, _) = connect_async(&ws_url).await
            .with_context(|| format!("failed to establish WebSocket connection to {ws_url} (cdp_port={})", self.cdp_port))
            .map_err(|e| {
                // The same resolver the launch path uses. It used to build a
                // `lucy-brave-{port}` directory that no launch ever created, so
                // this reported a profile that could not exist.
                let profile_dir = hyprfast_profile_dir(self.cdp_port);
                let diag = collect_cdp_diagnostics(
                    self.cdp_port, "<ws-connect>", &profile_dir,
                    "ws-connect-failed", Some(&e.to_string()), None,
                );
                write_cdp_log(self.cdp_port, &diag);
                warn!(port = self.cdp_port, error = %e, diag = %diag, "CDP WebSocket connect failed");
                anyhow!("{e}\nDiagnostics: {diag}\nFull log: /tmp/lucy-cdp-{}.log", self.cdp_port)
            })?;

        *self.ws.lock().await = Some(ws_stream);

        // 2b. Dedicated owned tab navigation: if we created or reused a tab,
        // always Page.navigate to target_url (tolerant dedup ensures we don't
        // just activate a stale URL). Best-effort: log but don't fail connect.
        if let Some(nav_url) = navigate_url.as_deref() {
            // Ensure the target is attached before navigate.
            let nav_res = self.call("Page.navigate", json!({ "url": nav_url })).await;
            match nav_res {
                Ok(_) => info!(url = %nav_url, "Page.navigate issued for owned tab"),
                Err(e) => warn!(url = %nav_url, error = %e, "Page.navigate failed (best-effort)"),
            }
        }

        // 3. Initialize metrics & focus emulation
        self.call(
            "Emulation.setDeviceMetricsOverride",
            json!({
                "width": 1280,
                "height": 800,
                "deviceScaleFactor": 1,
                "mobile": false
            }),
        )
        .await?;

        self.call(
            "Emulation.setFocusEmulationEnabled",
            json!({ "enabled": true }),
        )
        .await?;

        // Wait for initial page load (readyState + meaningful SPA content).
        // Best-effort: never fail connect just because a heavy SPA is slow.
        #[allow(deprecated)]
        let _ = self.wait_for_load(Duration::from_secs(20)).await;

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Settle-lite helpers (P1-3): pure logic, unit-testable without CDP.
    // -------------------------------------------------------------------------

    /// JS that resolves after two animation frames (~32ms at 60fps).
    /// Used by `wait_for_settle` via `Runtime.evaluate` with `awaitPromise`.
    pub fn settle_js_expression() -> &'static str {
        "new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(()=>r(true))))"
    }

    /// JS that returns `true` if a visible `[role=option]` exists in a
    /// listbox/combobox suggestion popup (checks bounding rect + visibility).
    pub fn suggestions_js_expression() -> &'static str {
        r#"(() => { const sel='[role="option"], [role="listbox"] [role="option"]'; const nodes=document.querySelectorAll(sel); for(const n of nodes){ if(!n.isConnected) continue; const r=n.getBoundingClientRect(); if(r.width===0&&r.height===0) continue; if(n.checkVisibility && !n.checkVisibility({checkVisibilityCSS:true, checkOpacity:true})) continue; const cs=getComputedStyle(n); if(cs.display==='none'||cs.visibility==='hidden'||cs.opacity==='0') continue; return true; } const ac=document.querySelector('[aria-controls]'); if(ac){ const id=ac.getAttribute('aria-controls'); if(id){ const lb=document.getElementById(id); if(lb){ const o=lb.querySelector('[role="option"]'); if(o){ const r=o.getBoundingClientRect(); if(r.width>0||r.height>0) return true; } } } } return false; })()"#
    }

    /// JS that returns the count of resource entries in the last 500ms.
    /// Cheap network-idle heuristic; no CDP Network domain needed.
    pub fn network_idle_js() -> &'static str {
        "(() => { const e=performance.getEntriesByType('resource'); const now=performance.now(); return e.filter(x=>now - x.startTime < 500).length; })()"
    }

    /// Pure helper: is the evaluated `document.readyState` value `"complete"`?
    pub fn is_ready_state_complete(value: &Value) -> bool {
        value.as_str() == Some("complete")
    }

    /// Pure helper: did the suggestions probe return visible options?
    pub fn suggestions_visible_from_value(value: &Value) -> bool {
        value.as_bool().unwrap_or(false)
    }

    /// Settle-lite: fast readyState==complete + 2×rAF (~50ms).
    ///
    /// Burns at most 1-2 CDP polls per step (vs 5-6 for `wait_for_load`'s
    /// double-stable text/inter/href loop). Intended for per-step use in
    /// `automation.rs`; `wait_for_load` remains for cold-nav `connect()`.
    pub async fn wait_for_settle(&self, timeout: Duration) -> Result<()> {
        let t0 = tokio::time::Instant::now();
        let deadline = t0 + timeout;

        // Phase 1: wait for readyState == "complete" (fast 30ms poll).
        // If the page is already complete this is a single evaluate.
        while tokio::time::Instant::now() < deadline {
            self.metrics.wait_polls.fetch_add(1, Ordering::SeqCst);
            match self.evaluate("document.readyState").await {
                Ok(v) if Self::is_ready_state_complete(&v) => break,
                Ok(_) => {}
                Err(_) => {
                    // Best-effort: transient CDP errors should not abort settle.
                }
            }
            if tokio::time::Instant::now() >= deadline {
                break;
            }
            let remaining = deadline - tokio::time::Instant::now();
            if remaining.is_zero() {
                break;
            }
            tokio::time::sleep(remaining.min(Duration::from_millis(30))).await;
        }

        let elapsed = t0.elapsed();
        if elapsed >= timeout {
            self.metrics
                .wait_ms
                .fetch_add(elapsed.as_millis() as u64, Ordering::SeqCst);
            return Ok(());
        }

        // Phase 2: two animation frames or 50ms – lets layout/paint settle
        // without double-stable text/href polling (the 17-63 polls burn).
        let remaining = timeout - elapsed;
        // Cap rAF wait so settle never exceeds caller's timeout; 350ms lets
        // the promise resolve on 60fps (~32ms) plus headroom, but yields to
        // the caller deadline.
        let raf_timeout = remaining.min(Duration::from_millis(350));
        let raf_expr = Self::settle_js_expression();
        match tokio::time::timeout(raf_timeout, self.evaluate(raf_expr)).await {
            Ok(Ok(_)) => {}
            Ok(Err(_)) => {
                // JS error – fall back to a short sleep so we still yield
                // one frame to the compositor.
                let fallback = remaining.min(Duration::from_millis(50));
                // Only sleep what remains after the failed evaluate.
                let after_eval = t0.elapsed();
                if after_eval < timeout {
                    let left = (timeout - after_eval).min(fallback);
                    if !left.is_zero() {
                        tokio::time::sleep(left).await;
                    }
                }
            }
            Err(_) => {
                // rAF promise timed out (e.g. background throttling) – treat
                // as settled; we've already waited raf_timeout.
            }
        }

        self.metrics
            .wait_ms
            .fetch_add(t0.elapsed().as_millis() as u64, Ordering::SeqCst);
        Ok(())
    }

    /// Wait up to `timeout` (recommended 200ms) for a visible combobox
    /// suggestion popup (`[role=option]` in an open listbox/aria-controls).
    /// Polls every ~25ms; returns `true` if suggestions became visible,
    /// `false` on timeout. Always `Ok` (best-effort).
    pub async fn wait_for_suggestions(&self, timeout: Duration) -> Result<bool> {
        let t0 = tokio::time::Instant::now();
        let deadline = t0 + timeout;
        let expr = Self::suggestions_js_expression();
        loop {
            self.metrics.wait_polls.fetch_add(1, Ordering::SeqCst);
            match self.evaluate(expr).await {
                Ok(v) if Self::suggestions_visible_from_value(&v) => {
                    self.metrics
                        .wait_ms
                        .fetch_add(t0.elapsed().as_millis() as u64, Ordering::SeqCst);
                    return Ok(true);
                }
                _ => {}
            }
            if tokio::time::Instant::now() >= deadline {
                self.metrics
                    .wait_ms
                    .fetch_add(t0.elapsed().as_millis() as u64, Ordering::SeqCst);
                return Ok(false);
            }
            let remaining = deadline - tokio::time::Instant::now();
            tokio::time::sleep(remaining.min(Duration::from_millis(25))).await;
        }
    }

    /// Cheap network-idle stub: sleep up to `idle_ms` (capped at 100ms) then
    /// check `performance.getEntriesByType('resource')` for recent loads in
    /// the last 500ms. Best-effort; always `Ok`.
    ///
    /// A real Network-domain drain would require enabling `Network` and
    /// tracking `requestWillBeSent`/`loadingFinished`; this keeps CDP calls
    /// to ~1 per invocation.
    pub async fn wait_for_network_idle(&self, idle_ms: Duration) -> Result<()> {
        let t0 = tokio::time::Instant::now();
        let sleep_dur = idle_ms.min(Duration::from_millis(100));
        if !sleep_dur.is_zero() {
            tokio::time::sleep(sleep_dur).await;
        }
        self.metrics.wait_polls.fetch_add(1, Ordering::SeqCst);
        let _ = self.evaluate(Self::network_idle_js()).await;
        self.metrics
            .wait_ms
            .fetch_add(t0.elapsed().as_millis() as u64, Ordering::SeqCst);
        Ok(())
    }

    /// Block until the page is loaded and settled.
    ///
    /// `document.readyState == "complete"` fires early on SPAs (YouTube, etc.)
    /// while content is still streaming in, so this also waits for meaningful
    /// content (visible text / interactive elements) to be present *and*
    /// stable across two consecutive polls. Best-effort: returns `Ok` on
    /// timeout so callers can snapshot whatever exists instead of hard-failing.
    #[deprecated(
        note = "use wait_for_settle for per-step settle-lite; retained for cold-nav connect()"
    )]
    pub async fn wait_for_load(&self, timeout: Duration) -> Result<()> {
        let t0 = tokio::time::Instant::now();
        let deadline = t0 + timeout;
        // (text_len, interactive_count, href) from the previous stable poll.
        let mut last: Option<(i64, i64, String)> = None;
        let mut stable_polls = 0u32;

        loop {
            self.metrics.wait_polls.fetch_add(1, Ordering::SeqCst);
            let state = self
                .evaluate(
                    r#"({rs: document.readyState,
                        body: !!document.body,
                        text: (document.body && document.body.innerText ? document.body.innerText.length : 0),
                        inter: document.querySelectorAll('a[href],button,input,textarea,select,[role="button"],[role="link"],[role="searchbox"],[role="combobox"],[role="textbox"]').length,
                        href: location.href})"#,
                )
                .await;
            if let Ok(v) = state {
                let ready = v.get("rs").and_then(Value::as_str) == Some("complete");
                let body = v.get("body").and_then(Value::as_bool).unwrap_or(false);
                let text = v.get("text").and_then(Value::as_i64).unwrap_or(0);
                let inter = v.get("inter").and_then(Value::as_i64).unwrap_or(0);
                let href = v
                    .get("href")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let has_content = body && ready && (text > 80 || inter >= 4);

                if has_content {
                    if last
                        .as_ref()
                        .is_some_and(|(t, i, h)| *t == text && *i == inter && *h == href)
                    {
                        stable_polls += 1;
                    } else {
                        stable_polls = 0;
                    }
                    last = Some((text, inter, href));
                    // Two consecutive stable polls (~300ms apart) = settled.
                    if stable_polls >= 1 {
                        self.metrics
                            .wait_ms
                            .fetch_add(t0.elapsed().as_millis() as u64, Ordering::SeqCst);
                        return Ok(());
                    }
                } else {
                    stable_polls = 0;
                    last = None;
                }
            }

            if tokio::time::Instant::now() >= deadline {
                warn!("wait_for_load timed out after {:?}", timeout);
                self.metrics.wait_timeouts.fetch_add(1, Ordering::SeqCst);
                self.metrics
                    .wait_ms
                    .fetch_add(t0.elapsed().as_millis() as u64, Ordering::SeqCst);
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(300)).await;
        }
    }

    /// Dispatch a raw CDP method call and wait for its matching response.
    pub async fn call(&self, method: &str, params: Value) -> Result<Value> {
        self.metrics.cdp_calls.fetch_add(1, Ordering::SeqCst);
        let id = self.request_id.fetch_add(1, Ordering::SeqCst);
        let req = json!({
            "id": id,
            "method": method,
            "params": params,
        });

        let mut lock = self.ws.lock().await;
        let ws = lock
            .as_mut()
            .ok_or_else(|| anyhow!("CDP client not connected"))?;

        ws.send(tokio_tungstenite::tungstenite::Message::Text(
            req.to_string().into(),
        ))
        .await?;

        // Read frames until response with matching id arrives
        let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
        while tokio::time::Instant::now() < deadline {
            let msg = tokio::time::timeout(Duration::from_secs(5), ws.next())
                .await
                .map_err(|_| anyhow!("CDP response timeout for method {method}"))?
                .ok_or_else(|| anyhow!("CDP connection closed"))??;

            if let tokio_tungstenite::tungstenite::Message::Text(text) = msg {
                if let Ok(resp) = serde_json::from_str::<Value>(&text) {
                    if resp.get("id").and_then(Value::as_u64) == Some(id) {
                        if let Some(err) = resp.get("error") {
                            return Err(anyhow!("CDP error for {method}: {err}"));
                        }
                        return Ok(resp.get("result").cloned().unwrap_or(Value::Null));
                    }
                }
            }
        }

        Err(anyhow!("CDP call {method} timed out"))
    }

    /// Evaluate an expression in page context and return the result value.
    pub async fn evaluate(&self, expression: &str) -> Result<Value> {
        self.metrics.evaluates.fetch_add(1, Ordering::SeqCst);
        let res = self
            .call(
                "Runtime.evaluate",
                json!({
                    "expression": expression,
                    "returnByValue": true,
                    "awaitPromise": true
                }),
            )
            .await?;

        if let Some(details) = res.get("exceptionDetails") {
            return Err(anyhow!("JS evaluation error: {details}"));
        }

        Ok(res
            .get("result")
            .and_then(|r| r.get("value"))
            .cloned()
            .unwrap_or(Value::Null))
    }

    /// Atomically observe the current DOM and action space using snapshot.js.
    pub async fn observe(&self) -> Result<BrowserPageSnapshot> {
        self.metrics.snapshots.fetch_add(1, Ordering::SeqCst);
        let val = self
            .evaluate(SNAPSHOT_JS)
            .await
            .context("failed to execute snapshot.js in browser")?;

        if val.is_null() {
            return Err(anyhow!("snapshot.js returned null (document.body missing)"));
        }

        let snapshot: BrowserPageSnapshot = serde_json::from_value(val)
            .context("failed to parse BrowserPageSnapshot from snapshot.js output")?;
        Ok(snapshot)
    }

    /// Execute an indexed action chosen by Laya or policy on the page.
    ///
    /// Always waits for the page to finish loading first: acting on a
    /// half-loaded DOM is what produced "covered, hidden, or stale" failures.
    /// The hit-test guard is then retried (not single-shot) so a briefly
    /// covered/detached node gets a chance to settle instead of aborting.
    pub async fn act(&self, action: &BrowserAction, text: Option<&str>) -> Result<()> {
        let kind = &action.kind;

        if kind == "wait" {
            tokio::time::sleep(Duration::from_millis(200)).await;
            return Ok(());
        }

        // Never act on a loading page.
        #[allow(deprecated)]
        {
            self.wait_for_load(Duration::from_secs(15)).await?;
        }

        if kind == "scroll" {
            let delta = action.delta.unwrap_or(560);
            self.call(
                "Input.dispatchMouseEvent",
                json!({
                    "type": "mouseWheel",
                    "x": 600,
                    "y": 400,
                    "deltaX": 0,
                    "deltaY": delta,
                }),
            )
            .await?;
            return Ok(());
        }

        // Target verification via guard evaluator (verifies node is connected and hit-testable)
        let node_id = action
            .node
            .ok_or_else(|| anyhow!("missing node ID for action {}", action.id))?;
        let action_json = serde_json::to_string(action)?;
        let eval_str = format!(
            r#"(action => {{
              const e=window.__jevFast?.nodes.get(action.node);
              if (!e?.isConnected || e.matches(':disabled') || e.closest('[aria-disabled="true"],[inert]') ||
                  !e.checkVisibility({{checkOpacity:true,checkVisibilityCSS:true}})) return null;
              if (action.kind==='fill' && (e.readOnly || e.getAttribute('aria-readonly')==='true')) return null;
              const r=e.getBoundingClientRect(), x=r.x+r.width/2, y=r.y+r.height/2;
              if (!r.width || !r.height || x<0 || y<0 || x>=innerWidth || y>=innerHeight) return null;
              if (!e.contains(document.elementFromPoint(x,y))) return null;
              if (action.kind==='select') {{
                if (e.tagName!=='SELECT' || ![...e.options].some(o=>o.value===action.value &&
                    !o.disabled && !o.closest('optgroup[disabled]'))) return null;
                e.value=action.value;
                e.dispatchEvent(new Event('input',{{bubbles:true}}));
                e.dispatchEvent(new Event('change',{{bubbles:true}}));
              }}
              return {{x,y}};
            }})({})"#,
            action_json
        );

        let target_pos = {
            let actionable_deadline = tokio::time::Instant::now() + Duration::from_secs(8);
            let mut pos = self.evaluate(&eval_str).await?;
            while pos.is_null() {
                self.metrics
                    .guard_extra_polls
                    .fetch_add(1, Ordering::SeqCst);
                if tokio::time::Instant::now() >= actionable_deadline {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(250)).await;
                pos = self.evaluate(&eval_str).await?;
            }
            pos
        };
        if target_pos.is_null() {
            self.metrics.stale_aborts.fetch_add(1, Ordering::SeqCst);
            return Err(anyhow!(
                "Action target node {} is covered, hidden, or stale after waiting for page load",
                node_id
            ));
        }

        if kind == "select" {
            // Already dispatched DOM change events in evaluation
            return Ok(());
        }

        let x = target_pos.get("x").and_then(Value::as_f64).unwrap_or(0.0);
        let y = target_pos.get("y").and_then(Value::as_f64).unwrap_or(0.0);

        // Click element
        for event in ["mousePressed", "mouseReleased"] {
            self.call(
                "Input.dispatchMouseEvent",
                json!({
                    "type": event,
                    "x": x,
                    "y": y,
                    "button": "left",
                    "clickCount": 1,
                }),
            )
            .await?;
        }

        if kind == "fill" {
            // Select all existing text: ctrl+a
            self.call(
                "Input.dispatchKeyEvent",
                json!({
                    "type": "keyDown",
                    "key": "a",
                    "code": "KeyA",
                    "modifiers": 2, // Ctrl on Linux
                    "commands": ["selectAll"]
                }),
            )
            .await?;
            self.call(
                "Input.dispatchKeyEvent",
                json!({
                    "type": "keyUp",
                    "key": "a",
                    "code": "KeyA",
                    "modifiers": 2,
                }),
            )
            .await?;

            // Insert new text
            let str_to_type = text.unwrap_or_else(|| action.value.as_deref().unwrap_or(""));
            if !str_to_type.is_empty() {
                self.call("Input.insertText", json!({ "text": str_to_type }))
                    .await?;
                // Dispatch Enter key
                self.call(
                    "Input.dispatchKeyEvent",
                    json!({
                        "type": "rawKeyDown",
                        "key": "Enter",
                        "code": "Enter",
                        "windowsVirtualKeyCode": 13,
                    }),
                )
                .await?;
                self.call(
                    "Input.dispatchKeyEvent",
                    json!({
                        "type": "keyUp",
                        "key": "Enter",
                        "code": "Enter",
                        "windowsVirtualKeyCode": 13,
                    }),
                )
                .await?;
            }
        }

        Ok(())
    }

    /// Dispatch a bare Enter keypress to the focused element (CDP Input domain).
    ///
    /// Used after `hint-act type` fills, which insert text but never submit.
    /// Mirrors the trailing Enter in [`Self::act`]'s fill branch so
    /// search-style fields actually submit instead of leaving the policy to
    /// burn its step budget on unverified DONE.
    pub async fn press_enter(&self) -> Result<()> {
        self.call(
            "Input.dispatchKeyEvent",
            json!({
                "type": "rawKeyDown",
                "key": "Enter",
                "code": "Enter",
                "windowsVirtualKeyCode": 13,
            }),
        )
        .await?;
        self.call(
            "Input.dispatchKeyEvent",
            json!({
                "type": "keyUp",
                "key": "Enter",
                "code": "Enter",
                "windowsVirtualKeyCode": 13,
            }),
        )
        .await?;
        Ok(())
    }

    /// Close the active tab if owned.
    pub async fn close(&self) {
        if let Some(target) = self.target_id.lock().await.take() {
            let http = reqwest::Client::new();
            let close_url = format!("http://127.0.0.1:{}/json/close/{target}", self.cdp_port);
            let _ = http.get(&close_url).send().await;
        }
        *self.ws.lock().await = None;
    }
}

/// Which of these names resolve on PATH, as a diagnostic string.
///
/// Also answers "is this an absolute path that exists", because config allows
/// a full path where a bare command name is expected.
fn which_binary(binary: &str) -> String {
    if std::path::Path::new(binary).exists() {
        return binary.to_string();
    }
    match std::process::Command::new("sh")
        .args(["-c", "command -v -- \"$1\" 2>&1", "lucy", binary])
        .output()
    {
        Ok(o) => {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if s.is_empty() {
                format!("not-on-PATH ({binary})")
            } else {
                s
            }
        }
        Err(e) => format!("which-failed: {e}"),
    }
}

/// The profile directory hyprfast launches the CDP browser with.
///
/// Resolution must match hyprfast's own `browser_runtime::launch::profile_dir`
/// exactly, because lucy writes a preference into this directory before the
/// launch and hyprfast decides the directory at launch: if the two disagree,
/// lucy edits a profile nobody opens, which is the exact failure the
/// preference write exists to prevent.
///
/// Re-derived here rather than queried from the running browser because the
/// write has to happen *before* launch, when there is no browser to ask. The
/// rules it has to honour, and why:
///
/// - **Never `temp_dir()`.** A tmpfs is RAM, so every login and cookie is gone
///   after a reboot; a user who signs in once would find it forgotten.
/// - **Never the browser's default data directory.** Chromium 136+ silently
///   ignores `--remote-debugging-port` there, so no port opens at all.
///
/// Override with `HYPRFAST_USER_DATA_DIR` (`~/` expanded), else
/// `$XDG_DATA_HOME/hyprfast/browser-profile`, else
/// `~/.local/share/hyprfast/browser-profile`.
fn hyprfast_profile_dir(_port: u16) -> std::path::PathBuf {
    profile_dir_from(
        std::env::var(lucy_config::HYPRFAST_USER_DATA_DIR_ENV).ok(),
        std::env::var("XDG_DATA_HOME").ok(),
        std::env::var("HOME").ok(),
    )
}

/// Resolution of [`hyprfast_profile_dir`] with the environment passed in.
///
/// Split out so it is a pure function of its arguments and can be tested
/// without mutating process env, which is shared with every other test in
/// this binary.
fn profile_dir_from(
    override_dir: Option<String>,
    xdg_data_home: Option<String>,
    home: Option<String>,
) -> std::path::PathBuf {
    if let Some(raw) = override_dir.filter(|v| !v.trim().is_empty()) {
        return expand_home(raw.trim(), home.as_deref());
    }
    let base = match xdg_data_home.filter(|v| !v.trim().is_empty()) {
        Some(dir) => expand_home(dir.trim(), home.as_deref()),
        _ => expand_home("~/.local/share", home.as_deref()),
    };
    base.join("hyprfast").join("browser-profile")
}

/// Expand a leading `~/` to `$HOME`.
///
/// A path in an environment variable is almost always typed by a person, and
/// `~` is what they type — but no shell is involved here, so without this the
/// literal `~` becomes a directory name relative to the working directory.
/// Only a bare `~/` is expanded; `~someone` needs a passwd lookup this does
/// not do, and leaving it alone is better than creating a literal `~someone`.
fn expand_home(raw: &str, home: Option<&str>) -> std::path::PathBuf {
    let Some(rest) = raw.strip_prefix("~/") else {
        return raw.into();
    };
    match home.filter(|h| !h.is_empty()) {
        Some(home) => std::path::Path::new(home).join(rest),
        // No HOME: keep the tilde visible in the path we report rather than
        // silently picking a location.
        None => raw.into(),
    }
}

/// Turn off "use graphics acceleration when available" in the profile hyprfast
/// is about to launch.
///
/// `--disable-gpu` alone already stops GPU rasterisation for the process, so
/// this is not load-bearing for correctness. It is here so the *setting* agrees
/// with the flag: Brave persists `hardware_acceleration_mode` back to
/// `Default/Preferences` on shutdown, and a browser whose stored pref still
/// says "on" reports itself as accelerated to anything that inspects the
/// profile — including the next run.
///
/// Ordering matters: the browser must not be running, and the write must land
/// before the launch, because the pref is read at startup. Returns a short
/// human-readable state for the log rather than erroring the caller — a
/// profile that cannot be written must not stop the run, the flag still holds.
fn disable_hardware_acceleration_pref(profile_dir: &std::path::Path) -> Result<String> {
    let prefs_path = profile_dir.join("Default").join("Preferences");
    if let Some(parent) = prefs_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow!("create-dir-failed ({}): {e}", parent.display()))?;
    }
    // Read-modify-write: Brave owns this file's other ~200 keys, and a
    // wholesale replacement would reset every preference it has ever stored.
    let mut root: Value = match std::fs::read_to_string(&prefs_path) {
        Ok(text) => serde_json::from_str(&text).unwrap_or_else(|e| {
            // A truncated or corrupt file is Brave's to overwrite on next
            // start; an empty object keeps the write going instead of failing.
            warn!(path = %prefs_path.display(), error = %e, "prefs unreadable, rewriting from empty object");
            json!({})
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => json!({}),
        Err(e) => {
            return Err(anyhow!(
                "read-failed ({}): {e}",
                prefs_path.display()
            ));
        }
    };
    if !root.is_object() {
        root = json!({});
    }
    let obj = root.as_object_mut().expect("just replaced with an object");
    let entry = obj
        .entry("hardware_acceleration_mode".to_string())
        .or_insert_with(|| json!({}));
    if !entry.is_object() {
        *entry = json!({});
    }
    entry
        .as_object_mut()
        .expect("just replaced with an object")
        .insert("enabled".to_string(), json!(false));

    let text = serde_json::to_string(&root).map_err(|e| anyhow!("serialize-failed: {e}"))?;
    std::fs::write(&prefs_path, text)
        .map_err(|e| anyhow!("write-failed ({}): {e}", prefs_path.display()))?;
    Ok(format!("written {}", prefs_path.display()))
}

fn truncate_for_log(s: &str) -> String {
    s.chars().take(600).collect::<String>().replace('\n', " | ")
}

/// Best-effort synchronous diagnostics for a CDP connection failure.
/// Never panics; every probe has a fallback string. Used both in the
/// returned error and in the persistent `/tmp/lucy-cdp-{port}.log`.
///
/// `launcher` is the hyprfast command, not a browser binary: hyprfast is what
/// spawns the browser, so "is hyprfast on PATH" is the question worth asking
/// when no endpoint appears.
fn collect_cdp_diagnostics(
    port: u16,
    launcher: &str,
    profile_dir: &std::path::Path,
    stage: &str,
    last_err: Option<&str>,
    initial_err: Option<&str>,
) -> String {
    let listener = std::process::Command::new("sh")
        .args([
            "-c",
            "ss -tlnp 2>/dev/null | grep -E ':\"$1\"(\\s|$)' || echo no-listener",
            "lucy",
            &port.to_string(),
        ])
        .output()
        .map(|o| {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            s.chars().take(300).collect::<String>()
        })
        .unwrap_or_else(|e| format!("ss-failed: {e}"));

    let procs = std::process::Command::new("sh")
        .args([
            "-c",
            "pgrep -a -f 'brave|chrome|chromium' 2>/dev/null | head -5 || echo no-browser-proc",
            "lucy",
        ])
        .output()
        .map(|o| {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            s.chars().take(600).collect::<String>()
        })
        .unwrap_or_else(|e| format!("pgrep-failed: {e}"));

    let profile_state = if profile_dir.exists() {
        match std::fs::read_dir(profile_dir) {
            Ok(mut entries) => {
                let n = entries.by_ref().take(6).count();
                format!("exists entries~{n}")
            }
            Err(e) => format!("exists but unreadable: {e}"),
        }
    } else {
        "missing".to_string()
    };

    // Direct TCP check (independent of reqwest) to distinguish
    // connection-refused (nothing listening) from HTTP-level failures.
    let tcp = match std::net::TcpStream::connect_timeout(
        &format!("127.0.0.1:{port}")
            .parse()
            .unwrap_or_else(|_| "127.0.0.1:1".parse().unwrap()),
        std::time::Duration::from_millis(800),
    ) {
        Ok(_) => "tcp-open".to_string(),
        Err(e) => format!("tcp-failed: {e}"),
    };

    format!(
        "stage={stage} launcher={launcher} which={} profile={}({}) tcp={} listener=[{}] initial_err=[{}] last_err=[{}] procs=[{}]",
        which_binary(launcher),
        profile_dir.display(),
        profile_state,
        tcp,
        listener,
        initial_err.unwrap_or("none"),
        last_err.unwrap_or("none"),
        procs.replace('\n', " | ")
    )
}

fn write_cdp_log(port: u16, diag: &str) {
    let path = format!("/tmp/lucy-cdp-{port}.log");
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let line = format!("{ts} port={port} {diag}\n");
    if let Some(parent) = std::path::Path::new(&path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = f.write_all(line.as_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn profile_dir_honours_an_explicit_override() {
        assert_eq!(
            profile_dir_from(
                Some("/srv/lucy/profile".into()),
                Some("/xdg".into()),
                Some("/home/u".into()),
            ),
            std::path::PathBuf::from("/srv/lucy/profile")
        );
    }

    #[test]
    fn profile_dir_defaults_under_the_data_home() {
        assert_eq!(
            profile_dir_from(None, Some("/xdg".into()), Some("/home/u".into())),
            std::path::PathBuf::from("/xdg/hyprfast/browser-profile")
        );
    }

    /// The regression this change exists for: the profile used to resolve
    /// under `temp_dir()`, which is RAM on a normal Linux install, so every
    /// login was gone after a reboot.
    #[test]
    fn profile_dir_is_not_temporary() {
        let dir = profile_dir_from(None, Some("/xdg".into()), Some("/home/u".into()));
        let tmp = std::env::temp_dir();
        assert!(
            !dir.starts_with(&tmp),
            "profile must not live under {} or logins die on reboot: {}",
            tmp.display(),
            dir.display()
        );
    }

    /// A user setting this by hand types a tilde, and no shell is involved to
    /// expand it.
    #[test]
    fn a_tilde_in_the_override_is_expanded_to_home() {
        assert_eq!(
            profile_dir_from(Some("~/p".into()), None, Some("/home/u".into())),
            std::path::PathBuf::from("/home/u/p")
        );
        assert_eq!(
            profile_dir_from(None, None, Some("/home/u".into())),
            std::path::PathBuf::from("/home/u/.local/share/hyprfast/browser-profile")
        );
    }

    #[test]
    fn another_users_home_is_left_alone() {
        // `~someone` needs a passwd lookup this does not do; creating a
        // literal `~someone` directory would be worse.
        assert_eq!(
            expand_home("~someone/p", Some("/home/u")),
            std::path::PathBuf::from("~someone/p")
        );
    }

    #[test]
    fn an_empty_override_falls_through_to_the_default() {
        for blank in ["", "   "] {
            assert_eq!(
                profile_dir_from(
                    Some(blank.into()),
                    Some("/xdg".into()),
                    Some("/home/u".into())
                ),
                profile_dir_from(None, Some("/xdg".into()), Some("/home/u".into())),
                "{blank:?} should defer to the default"
            );
        }
    }

    /// The default has to be the same on every port, or logging in on one run
    /// hands back a different, empty profile on the next.
    #[test]
    fn the_default_profile_does_not_vary_by_port() {
        let a = hyprfast_profile_dir(9222);
        let b = hyprfast_profile_dir(9333);
        assert_eq!(a, b);
        assert!(a.ends_with("hyprfast/browser-profile"), "{a:?}");
    }

    #[test]
    fn test_settle_js_expression_contains_double_raf() {
        let js = BrowserCdpClient::settle_js_expression();
        assert!(
            js.contains("requestAnimationFrame"),
            "settle js must use rAF"
        );
        // Two nested rAF calls
        assert_eq!(
            js.matches("requestAnimationFrame").count(),
            2,
            "must have exactly 2 rAFs"
        );
        assert!(js.contains("new Promise"), "must be a Promise");
    }

    #[test]
    fn test_is_ready_state_complete() {
        assert!(BrowserCdpClient::is_ready_state_complete(&json!(
            "complete"
        )));
        assert!(!BrowserCdpClient::is_ready_state_complete(&json!(
            "loading"
        )));
        assert!(!BrowserCdpClient::is_ready_state_complete(&json!(
            "interactive"
        )));
        assert!(!BrowserCdpClient::is_ready_state_complete(&json!(null)));
        assert!(!BrowserCdpClient::is_ready_state_complete(&json!(42)));
    }

    #[test]
    fn test_suggestions_js_expression_contains_role_option() {
        let js = BrowserCdpClient::suggestions_js_expression();
        assert!(js.contains(r#"[role="option"]"#), "must query role=option");
        assert!(
            js.contains("aria-controls") || js.contains("listbox"),
            "must handle aria-controls/listbox"
        );
        assert!(
            js.contains("getBoundingClientRect"),
            "must check visibility via bbox"
        );
    }

    #[test]
    fn test_suggestions_visible_from_value() {
        assert!(BrowserCdpClient::suggestions_visible_from_value(&json!(
            true
        )));
        assert!(!BrowserCdpClient::suggestions_visible_from_value(&json!(
            false
        )));
        assert!(!BrowserCdpClient::suggestions_visible_from_value(&json!(
            null
        )));
        assert!(!BrowserCdpClient::suggestions_visible_from_value(&json!(1)));
    }

    #[test]
    fn test_network_idle_js_contains_performance() {
        let js = BrowserCdpClient::network_idle_js();
        assert!(
            js.contains("performance.getEntriesByType"),
            "must use performance API"
        );
        assert!(js.contains("resource"), "must query resource entries");
        assert!(js.contains("500"), "must check last 500ms");
    }

    #[test]
    fn test_suggestions_js_no_heavy_polling_markers() {
        // Guard: settle/suggestions JS must not contain the heavy
        // wait_for_load markers (innerText length, href polling).
        let settle = BrowserCdpClient::settle_js_expression();
        let sugg = BrowserCdpClient::suggestions_js_expression();
        for js in [settle, sugg] {
            assert!(
                !js.contains("innerText"),
                "settle-lite must not poll innerText"
            );
            assert!(
                !js.contains("location.href"),
                "settle-lite must not poll href"
            );
        }
    }
}
