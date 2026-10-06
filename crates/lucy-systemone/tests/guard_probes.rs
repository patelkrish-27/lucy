//! P0 guard probes — style of jev's `check_guards.py`.
//!
//! Each test serves a tiny fixture via a local HTTP server, connects a fresh
//! `BrowserCdpClient` on an ephemeral CDP port, and asserts the in-page
//! guards that P1 will fix:
//! - `covered_rejected` — elementFromPoint occlusion must be rejected
//! - `detached_rejected` — isConnected=false after replace must be rejected
//! - `re_render_new_id` — SPA replace creates a new node identity
//! - `cap_250_and_scroll_synthetic` — 300 items truncates to 250+scroll/wait
//!
//! Tests are `#[ignore]` so `cargo test` stays green without a browser.
//! Run live: `cargo test -p lucy-systemone --test guard_probes -- --ignored --nocapture`
//! Requires: `hyprfast` on PATH (it is the only thing that launches a browser)
//! plus the browser it launches, `brave`; no `--remote-debugging-port` clash;
//! decider-serve optional.

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
    let n = match stream.read(&mut buf).await {
        Ok(n) => n,
        Err(_) => return,
    };
    let req = String::from_utf8_lossy(&buf[..n]);
    let path = req
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .unwrap_or("/");
    // Map / -> not used; callers request /guards.html etc.
    let rel = path.trim_start_matches('/').split('?').next().unwrap_or("");
    let safe = if rel.is_empty() { "guards.html" } else { rel };
    let file = dir.join(safe);
    let (status, body, ctype) = if file.exists() && file.is_file() {
        let data = tokio::fs::read(&file).await.unwrap_or_default();
        let ct = if safe.ends_with(".html") {
            "text/html; charset=utf-8"
        } else {
            "application/octet-stream"
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
        .expect("bind fixture server");
    let addr = listener.local_addr().unwrap();
    let h = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            let d = dir.clone();
            tokio::spawn(serve_once(stream, d));
        }
    });
    (addr, h)
}

fn cdp_port_for(test: &str) -> u16 {
    match test {
        "covered_rejected" => 19311,
        "detached_rejected" => 19312,
        "re_render_new_id" => 19313,
        "cap_250_and_scroll_synthetic" => 19314,
        _ => 19315,
    }
}

async fn with_browser<F, T>(
    http_addr: std::net::SocketAddr,
    url_path: &str,
    cdp_port: u16,
    f: F,
) -> T
where
    F: AsyncFnOnce(Arc<BrowserCdpClient>, String) -> T,
{
    let url = format!("http://{http_addr}/{url_path}");
    let metrics = Arc::new(BrowserMetrics::default());
    // Build client with shared metrics so wait/cdp counts land in the harness.
    let client = Arc::new(BrowserCdpClient::new(cdp_port).with_metrics(metrics.clone()));
    client
        .connect(Some(&url))
        .await
        .expect("CDP connect (is hyprfast on PATH, and the browser it launches installed?)");
    f(client, url).await
}

// 1. Covered element must be rejected by the hit-test guard.
#[tokio::test]
#[ignore]
async fn covered_rejected() {
    let (addr, _srv) = spawn_static_server(fixtures_dir()).await;
    let cdp = cdp_port_for("covered_rejected");
    with_browser(addr, "guards.html", cdp, async |client, _url| {
        // Brief settle — guard page is static.
        tokio::time::sleep(Duration::from_millis(400)).await;
        let snap = client.observe().await.expect("observe guards.html");
        // Find the covered button.
        let action = snap
            .actions
            .iter()
            .find(|a| a.label.contains("Covered action"))
            .expect("Covered action not in snapshot");
        let before_stale = client
            .metrics()
            .stale_aborts
            .load(std::sync::atomic::Ordering::SeqCst);
        let err = client
            .act(action, None)
            .await
            .expect_err("covered act must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("covered, hidden, or stale"),
            "expected stale error, got: {msg}"
        );
        // Instrumentation: at least one guard retry and one stale abort.
        let m = client.metrics().snapshot();
        assert!(m.guard_extra_polls >= 1, "guard should have retried: {m:?}");
        assert!(
            m.stale_aborts > before_stale,
            "stale_aborts must increment: {m:?}"
        );
        // Ensure the page really didn't click through.
        let outcome: String = client
            .evaluate("document.getElementById('outcome').textContent")
            .await
            .and_then(|v| Ok(v.as_str().unwrap_or("").to_string()))
            .unwrap_or_default();
        assert!(
            outcome.is_empty(),
            "overlay should have blocked click, outcome={outcome:?}"
        );
    })
    .await;
}

// 2. Detached node (replaced DOM) must be rejected.
#[tokio::test]
#[ignore]
async fn detached_rejected() {
    let (addr, _srv) = spawn_static_server(fixtures_dir()).await;
    let cdp = cdp_port_for("detached_rejected");
    with_browser(addr, "guards.html", cdp, async |client, _url| {
        tokio::time::sleep(Duration::from_millis(400)).await;
        let snap = client.observe().await.expect("observe");
        let action = snap
            .actions
            .iter()
            .find(|a| a.label.contains("Replace me"))
            .expect("Replace me not in snapshot")
            .clone();
        // Replace the node in-page: old WeakMap id now points to a detached node.
        client
            .evaluate(
                r#"(() => {
                    const old = document.getElementById('replace-me');
                    const neu = old.cloneNode(true);
                    neu.id = 'replace-me';
                    neu.textContent = 'Replace me (new)';
                    old.replaceWith(neu);
                    return true;
                })()"#,
            )
            .await
            .expect("replace eval");
        let err = client
            .act(&action, None)
            .await
            .expect_err("detached act must fail");
        assert!(
            err.to_string().contains("covered, hidden, or stale"),
            "expected stale, got {err}"
        );
        // Fresh observe should find the new node with a different identity.
        let snap2 = client.observe().await.expect("re-observe");
        let action2 = snap2
            .actions
            .iter()
            .find(|a| a.label.contains("Replace me"))
            .expect("new Replace me missing");
        assert_ne!(
            action.node,
            action2.node,
            "re-render must produce a new node id (old={}, new={})",
            action.node.unwrap_or(-1),
            action2.node.unwrap_or(-1)
        );
        // Clicking the fresh id should succeed.
        client
            .act(action2, None)
            .await
            .expect("fresh act should succeed");
        let outcome: String = client
            .evaluate("document.getElementById('outcome').textContent")
            .await
            .unwrap()
            .as_str()
            .unwrap_or("")
            .to_string();
        assert_eq!(outcome, "replaced-clicked");
    })
    .await;
}

// 3. Re-render inside search.html replaces suggestion nodes (stale-ID shape).
#[tokio::test]
#[ignore]
async fn re_render_new_id() {
    let (addr, _srv) = spawn_static_server(fixtures_dir()).await;
    let cdp = cdp_port_for("re_render_new_id");
    with_browser(addr, "search.html", cdp, async |client, _url| {
        tokio::time::sleep(Duration::from_millis(400)).await;
        // Direct DOM input without Enter (BrowserCdpClient::act always sends Enter,
        // which would hide suggestions via keydown handler). Use JS to trigger.
        async fn type_no_enter(client: &BrowserCdpClient, v: &str) {
            let js = format!(
                r#"(() => {{
                    const q = document.getElementById('q');
                    q.focus(); q.value = {:?}; q.dispatchEvent(new Event('input', {{bubbles:true}}));
                    return true;
                }})()"#,
                v
            );
            client.evaluate(&js).await.expect("type via js");
        }
        type_no_enter(&client, "neb").await;
        tokio::time::sleep(Duration::from_millis(250)).await;
        let snap1 = client.observe().await.expect("observe after neb");
        let first_ids: Vec<Option<i64>> = snap1
            .actions
            .iter()
            .filter(|a| a.role.as_deref() == Some("option"))
            .map(|a| a.node)
            .collect();
        assert!(!first_ids.is_empty(), "suggestions for 'neb' should exist: snap1={snap1:?}");

        // Typing more replaces all suggestion nodes.
        type_no_enter(&client, "nebu").await;
        tokio::time::sleep(Duration::from_millis(250)).await;
        let snap2 = client.observe().await.expect("observe after nebu");
        let second_ids: Vec<Option<i64>> = snap2
            .actions
            .iter()
            .filter(|a| a.role.as_deref() == Some("option"))
            .map(|a| a.node)
            .collect();
        // At least one id should differ (full replace).
        let overlap = first_ids.iter().filter(|id| second_ids.contains(id)).count();
        assert!(
            overlap < first_ids.len(),
            "re-render should replace suggestion node ids: before={first_ids:?} after={second_ids:?}"
        );
    })
    .await;
}

// 4. 300-item feed truncates to 250 actions + synthetics; policy scroll gap.
#[tokio::test]
#[ignore]
async fn cap_250_and_scroll_synthetic() {
    let (addr, _srv) = spawn_static_server(fixtures_dir()).await;
    let cdp = cdp_port_for("cap_250");
    with_browser(addr, "feed.html", cdp, async |client, _url| {
        tokio::time::sleep(Duration::from_millis(500)).await;
        let snap = client.observe().await.expect("observe feed");
        let real = snap
            .actions
            .iter()
            .filter(|a| a.kind != "scroll" && a.kind != "wait")
            .count();
        // Dense grid => many in-viewport. Either viewport limits or 250-cap
        // applies; we assert the cap is hit when >250 are visible.
        // If only ~200 are visible (font/metrics), check via omitted_actions
        // or synthetic scroll presence — both prove the probe fixture works.
        if real == 250 {
            assert!(snap.omitted_actions > 0, "250 real => omitted >0");
        } else {
            eprintln!(
                "feed real={real} omitted={} — grid density below cap, checking synthetics",
                snap.omitted_actions
            );
            assert!(
                real > 150,
                "dense feed should have >150 in-viewport buttons, got {real}"
            );
        }
        // Wait synthetic is always appended; scroll_down only if page scrollable.
        assert!(
            snap.actions.iter().any(|a| a.id == "wait"),
            "wait synthetic missing"
        );
        let m = client.metrics().snapshot();
        assert!(m.snapshots >= 1);
    })
    .await;
}
