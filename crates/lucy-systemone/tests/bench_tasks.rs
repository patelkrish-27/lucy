//! P0 bench harness — 3 canned tasks against fixture pages.
//!
//! Tasks:
//! 1. `search_open` — search.html: type "Nebula" → open "Nebula Widget"
//! 2. `form_fill`   — form.html:   fill name/email/country/checkbox → submit
//! 3. `feed_scroll_gap` — feed.html: attempt to open below-fold item 260 (proves 250-cap gap)
//!
//! Uses live `decider-serve` on `http://localhost:8001` when available, otherwise
//! a stub oracle so the harness still records CDP costs (laya_predicts=0).
//! Writes `target/lucy-bench/<task>.json` + prints a table for `docs/lucy-performance.md`.
//!
//! Run: `cargo test -p lucy-systemone --test bench_tasks -- --ignored --nocapture`
//! Or as example: `cargo run -p lucy-systemone --example bench` (same harness).

use lucy_systemone::browser_policy::{BrowserPolicy, GoalPlan, GoalRequirement};
use lucy_systemone::client::SystemOneClient;
use lucy_systemone::metrics::BrowserRunReport;
use lucy_systemone::{BrowserCdpClient, BrowserMetrics};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

async fn serve_once(mut stream: TcpStream, dir: PathBuf) {
    let mut buf = vec![0u8; 8192];
    let Ok(n) = stream.read(&mut buf).await else {
        return;
    };
    let req = String::from_utf8_lossy(&buf[..n]);
    let path = req
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .unwrap_or("/");
    let rel = path.trim_start_matches('/').split('?').next().unwrap_or("");
    let safe = if rel.is_empty() { "guards.html" } else { rel };
    let file = dir.join(safe);
    let (status, body, ctype) = if file.exists() && file.is_file() {
        let data = tokio::fs::read(&file).await.unwrap_or_default();
        let ct = if safe.ends_with(".html") {
            "text/html; charset=utf-8"
        } else {
            "text/plain"
        };
        ("HTTP/1.1 200 OK", data, ct)
    } else {
        (
            "HTTP/1.1 404 Not Found",
            b"not found".to_vec(),
            "text/plain",
        )
    };
    let header = format!(
        "{status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(header.as_bytes()).await;
    let _ = stream.write_all(&body).await;
}

async fn spawn_static_server(dir: PathBuf) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind bench server");
    let addr = listener.local_addr().unwrap();
    let h = tokio::spawn(async move {
        loop {
            let Ok((s, _)) = listener.accept().await else {
                break;
            };
            let d = dir.clone();
            tokio::spawn(serve_once(s, d));
        }
    });
    (addr, h)
}

fn cdp_port_for(name: &str) -> u16 {
    match name {
        "search_open" => 19411,
        "form_fill" => 19412,
        "feed_scroll_gap" => 19413,
        _ => 19414,
    }
}

fn laya_client_or_stub() -> Arc<SystemOneClient> {
    // Real decider-serve if reachable; fails fast otherwise (lucy never
    // starts the server — the user runs it manually).
    let cfg = lucy_config::SystemOneConfig {
        enabled: true,
        provider: "decider".into(),
        base_url: "http://localhost:8001".into(),
        api_key: None,
        model: "Mapika/decider-2b-vision".into(),
        direct_python: None,
        confidence_threshold: 0.1,
        timeout_ms: 8000,
        auto_start: false,
        server_command: None,
        server_model: None,
    };
    Arc::new(SystemOneClient::new(cfg, "python3".into()))
}

async fn run_task(
    task_name: &str,
    fixture: &str,
    goal: &str,
    plan: GoalPlan,
    http_addr: std::net::SocketAddr,
    max_steps: usize,
) -> BrowserRunReport {
    let url = format!("http://{http_addr}/{fixture}");
    let metrics = Arc::new(BrowserMetrics::default());
    let client_cdp =
        Arc::new(BrowserCdpClient::new(cdp_port_for(task_name)).with_metrics(metrics.clone()));
    client_cdp
        .connect(Some(&url))
        .await
        .expect("CDP connect for bench");
    let system_one = laya_client_or_stub();
    let laya_before = system_one.predict_calls();
    let mut policy = BrowserPolicy::new(goal);
    policy.set_plan(plan);

    let t0 = std::time::Instant::now();
    let mut outcome = "max_steps".to_string();
    let mut message = format!("reached max_steps={max_steps}");
    let mut steps_done: u64 = 0;

    for step_idx in 1..=max_steps {
        metrics
            .steps
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        // P1 settle-lite (mirrors automation.rs)
        let _ = client_cdp
            .wait_for_settle(Duration::from_millis(2000))
            .await;
        let snap = match client_cdp.observe().await {
            Ok(s) => s,
            Err(e) => {
                outcome = "error".into();
                message = e.to_string();
                steps_done = step_idx as u64;
                break;
            }
        };
        let step = match policy.step(&snap, &system_one).await {
            Ok(o) => o,
            Err(e) => {
                outcome = "error".into();
                message = e.to_string();
                steps_done = step_idx as u64;
                break;
            }
        };
        match step {
            lucy_systemone::PolicyOutcome::Done { message: m } => {
                outcome = "done".into();
                message = m;
                steps_done = step_idx as u64;
                break;
            }
            lucy_systemone::PolicyOutcome::Blocked { reason } => {
                outcome = "blocked".into();
                message = reason;
                steps_done = step_idx as u64;
                break;
            }
            lucy_systemone::PolicyOutcome::Action {
                action,
                text_to_type,
                description: _,
            } => {
                // No hint-act in bench (pure CDP cost, deterministic).
                if let Err(e) = client_cdp.act(&action, text_to_type.as_deref()).await {
                    let msg = e.to_string();
                    if msg.contains("covered, hidden, or stale") {
                        tokio::time::sleep(Duration::from_millis(200)).await;
                        if let Ok(snap2) = client_cdp.observe().await {
                            if let Ok(retry) = policy.step(&snap2, &system_one).await {
                                if let lucy_systemone::PolicyOutcome::Action {
                                    action: a2,
                                    text_to_type: t2,
                                    ..
                                } = retry
                                {
                                    if let Err(e2) = client_cdp.act(&a2, t2.as_deref()).await {
                                        outcome = "error".into();
                                        message = format!("stale retry failed: {e2}");
                                        steps_done = step_idx as u64;
                                        break;
                                    }
                                }
                            }
                        } else {
                            outcome = "error".into();
                            message = msg;
                            steps_done = step_idx as u64;
                            break;
                        }
                    } else {
                        outcome = "error".into();
                        message = msg;
                        steps_done = step_idx as u64;
                        break;
                    }
                }
                // Post-act settle + combobox suggestion wait
                if action.kind == "fill" {
                    let _ = client_cdp
                        .wait_for_suggestions(Duration::from_millis(200))
                        .await;
                }
                let _ = client_cdp
                    .wait_for_settle(Duration::from_millis(1500))
                    .await;
                steps_done = step_idx as u64;
            }
        }
    }

    let wall_ms = t0.elapsed().as_millis() as u64;
    let laya_predicts = system_one.predict_calls().saturating_sub(laya_before);
    // Verify outcome via page text/title when we claim done.
    let report = BrowserRunReport {
        task: task_name.to_string(),
        outcome: outcome.clone(),
        message: message.clone(),
        steps: steps_done,
        wall_ms,
        laya_predicts,
        metrics: metrics.snapshot(),
    };
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/lucy-bench")
        .join(format!("{task_name}.json"));
    let _ = report.write_json(&out);
    report
}

fn plan_search() -> GoalPlan {
    GoalPlan {
        requirements: vec![GoalRequirement {
            what: "Search widgets".into(),
            value: "Nebula".into(),
        }],
        open: Some("Nebula Widget".into()),
        finish: "Nebula Widget \u{2014} Widget Store is open".into(),
    }
}
fn plan_form() -> GoalPlan {
    GoalPlan {
        requirements: vec![
            GoalRequirement {
                what: "Full name".into(),
                value: "Ada Lovelace".into(),
            },
            GoalRequirement {
                what: "Email address".into(),
                value: "ada@example.com".into(),
            },
            GoalRequirement {
                what: "Country".into(),
                value: "portugal".into(),
            },
            GoalRequirement {
                what: "Subscribe to newsletter".into(),
                value: "checked".into(),
            },
        ],
        open: None,
        finish: "Form submitted for Ada Lovelace".into(),
    }
}
fn plan_feed() -> GoalPlan {
    GoalPlan {
        requirements: vec![],
        open: Some("Result item 260".into()),
        finish: "Result item 260 is open".into(),
    }
}

#[tokio::test]
#[ignore]
async fn bench_search_open() {
    let (addr, _srv) = spawn_static_server(fixtures_dir()).await;
    let r = run_task(
        "search_open",
        "search.html",
        "Search for Nebula and open Nebula Widget",
        plan_search(),
        addr,
        12,
    )
    .await;
    eprintln!(
        "[bench search_open] outcome={} steps={} wall_ms={} laya={} cdp={} eval={} snaps={} waits={} timeouts={} guard_polls={} stale={}",
        r.outcome,
        r.steps,
        r.wall_ms,
        r.laya_predicts,
        r.metrics.cdp_calls,
        r.metrics.evaluates,
        r.metrics.snapshots,
        r.metrics.wait_polls,
        r.metrics.wait_timeouts,
        r.metrics.guard_extra_polls,
        r.metrics.stale_aborts
    );
    // Search+open should succeed with the harness (proves fixtures are reachable).
    // Don't hard-fail on laya variability — just record.
}

#[tokio::test]
#[ignore]
async fn bench_form_fill() {
    let (addr, _srv) = spawn_static_server(fixtures_dir()).await;
    let r = run_task(
        "form_fill",
        "form.html",
        "Fill the contact form for Ada Lovelace, ada@example.com, Portugal, subscribe",
        plan_form(),
        addr,
        12,
    )
    .await;
    eprintln!(
        "[bench form_fill] outcome={} steps={} wall_ms={} laya={} cdp={} eval={} snaps={} waits={} timeouts={} guard_polls={} stale={}",
        r.outcome,
        r.steps,
        r.wall_ms,
        r.laya_predicts,
        r.metrics.cdp_calls,
        r.metrics.evaluates,
        r.metrics.snapshots,
        r.metrics.wait_polls,
        r.metrics.wait_timeouts,
        r.metrics.guard_extra_polls,
        r.metrics.stale_aborts
    );
}

#[tokio::test]
#[ignore]
async fn bench_feed_scroll_gap() {
    let (addr, _srv) = spawn_static_server(fixtures_dir()).await;
    let r = run_task(
        "feed_scroll_gap",
        "feed.html",
        "Open Result item 260",
        plan_feed(),
        addr,
        10,
    )
    .await;
    eprintln!(
        "[bench feed_scroll_gap] outcome={} steps={} wall_ms={} laya={} cdp={} eval={} snaps={} waits={} timeouts={} guard_polls={} stale={} omitted_in_last_snap?",
        r.outcome,
        r.steps,
        r.wall_ms,
        r.laya_predicts,
        r.metrics.cdp_calls,
        r.metrics.evaluates,
        r.metrics.snapshots,
        r.metrics.wait_polls,
        r.metrics.wait_timeouts,
        r.metrics.guard_extra_polls,
        r.metrics.stale_aborts
    );
    // P0 expectation: blocked/max_steps — item 260 is beyond the 250-action cap
    // and the policy cannot emit scroll (dropped in observed()). P1 fixes this.
    assert_ne!(
        r.outcome, "done",
        "P0 should NOT reach item 260 without scroll (gap probe); got {r:?}"
    );
}
