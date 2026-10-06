//! hyprfast 0.9 browser executor: hint-key overlay + daemon.
//!
//! Replaces cached `__jevFast` node-ID retries (which produced
//! `covered, hidden, or stale` aborts on SPA re-render, e.g. lucy_testing 102
//! node 9) with fresh-snapshot-per-call semantics:
//! - `hint_snapshot` / `hint_act` / `hint_batch`: Vimium-primary,
//!   heuristic first, Decider-2B only on ambiguity.
//! - `browser_runtime start/status`: single-WS daemon owner (fixes multi-tab
//!   ambiguity + WS drops). All wrappers shell out to the `hyprfast` CLI so
//!   Lucy tracks the installed 0.9 binary instead of reimplementing CDP.
//!
//! Every call is best-effort with structured errors: callers must fall back
//! to direct CDP (`BrowserCdpClient`) when hint misses. Vision grounding
//! (`ground` / `act_fast` / `act_batch`) was removed: those tools need a
//! Gemini key Lucy no longer uses, so they are never called and never offered
//! to the planner (see `lucy_hyprfast::is_removed_tool`).

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HintRect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hint {
    pub label: String,
    #[serde(default)]
    pub tag: String,
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub selector: String,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub rect: Option<HintRect>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HintSnapshot {
    #[serde(default)]
    pub count: usize,
    #[serde(default)]
    pub hints: Vec<Hint>,
}

impl HintSnapshot {
    /// Heuristic pick without LLM: most word overlap with instruction.
    /// Returns the hint label (e.g. "A") or None.
    pub fn heuristic_pick(&self, instruction: &str) -> Option<String> {
        let words: Vec<String> = instruction
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 3)
            .map(|w| w.to_string())
            .collect();
        if words.is_empty() {
            return None;
        }
        let mut best: Option<(&Hint, usize)> = None;
        for h in &self.hints {
            let hay = format!("{} {} {} {}", h.name, h.text, h.role, h.tag).to_lowercase();
            let score = words.iter().filter(|w| hay.contains(w.as_str())).count();
            if score > 0 && best.map(|(_, s)| score > s).unwrap_or(true) {
                best = Some((h, score));
            }
        }
        best.map(|(h, _)| h.label.clone())
    }
}

/// Run `hyprfast` with args, return parsed JSON stdout.
/// hyprfast prints `warning: ...` lines to stderr (degraded mode notices) —
/// those are ignored; stdout must be JSON. CLI errors surface as Err with
/// stderr tail included.
async fn hyprfast_json(hyprfast_cmd: &str, args: &[&str]) -> Result<Value> {
    let out = Command::new(hyprfast_cmd)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .with_context(|| format!("failed to execute hyprfast {}", args.join(" ")))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let tail: String = stderr.lines().rev().take(3).collect::<Vec<_>>().join(" | ");
        return Err(anyhow!("hyprfast {} failed: {}", args.join(" "), tail));
    }
    let val: Value = serde_json::from_slice(&out.stdout).context("hyprfast stdout was not JSON")?;
    Ok(val)
}

/// Best-effort: start the persistent browser-runtime daemon.
/// Returns Ok even if already running; Err only if the start command itself failed.
pub async fn ensure_browser_runtime(hyprfast_cmd: &str) -> Result<()> {
    // Fast path: status succeeds => daemon alive.
    if hyprfast_json(hyprfast_cmd, &["browser-runtime", "status"])
        .await
        .is_ok()
    {
        return Ok(());
    }
    info!("browser-runtime daemon missing, starting via hyprfast");
    let res = tokio::time::timeout(
        Duration::from_secs(10),
        hyprfast_json(hyprfast_cmd, &["browser-runtime", "start"]),
    )
    .await;
    match res {
        Ok(Ok(_)) => {
            info!("browser-runtime daemon started");
            Ok(())
        }
        Ok(Err(e)) => {
            warn!(error = %e, "browser-runtime start failed (continuing in degraded direct mode)");
            Err(e)
        }
        Err(_) => {
            warn!("browser-runtime start timed out (continuing in degraded direct mode)");
            Err(anyhow!("browser-runtime start timed out"))
        }
    }
}

/// `hyprfast hint-snapshot` — fresh overlay scan (no cached IDs).
pub async fn hint_snapshot(hyprfast_cmd: &str) -> Result<HintSnapshot> {
    let val = hyprfast_json(hyprfast_cmd, &["hint-snapshot"]).await?;
    let snap: HintSnapshot =
        serde_json::from_value(val).context("failed to parse hint-snapshot output")?;
    Ok(snap)
}

/// Fused `hyprfast hint-act "instruction" --action click|type --text "..."`.
/// Returns the raw JSON result. Callers treat a miss (`error` / `not found` /
/// `no action`) as a fallback signal, not a fatal error.
pub async fn hint_act(
    hyprfast_cmd: &str,
    instruction: &str,
    action: &str,
    text: Option<&str>,
) -> Result<Value> {
    let mut args: Vec<&str> = vec!["hint-act", instruction, "--action", action];
    let text_owned;
    if let Some(t) = text {
        text_owned = t.to_string();
        args.push("--text");
        args.push(text_owned.as_str());
    }
    // Own the strings so the borrowed args outlive the call.
    let owned: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let refs: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();
    hyprfast_json(hyprfast_cmd, &refs).await
}

/// `hyprfast hint-batch '<steps JSON>'` — one snapshot + parallel dispatches.
pub async fn hint_batch(hyprfast_cmd: &str, steps_json: &str) -> Result<Value> {
    hyprfast_json(hyprfast_cmd, &["hint-batch", steps_json]).await
}

/// Compare two URLs for tab reuse: same host + same path tolerant dedup.
/// For YouTube, only `search_query` param matters (other params volatile);
/// for all other hosts, query differences are ignored — caller will navigate
/// the reused tab to the target URL. This prevents duplicate tabs while still
/// ensuring the tab ends up at the correct URL (P1-4 owned-tab + tolerant dedup).
pub fn same_tab_for_reuse(existing_url: &str, target_url: &str) -> bool {
    let parse = |u: &str| -> Option<(String, String, String)> {
        let after_scheme = u.split("://").nth(1).unwrap_or(u);
        let (host_path, query) = match after_scheme.split_once('?') {
            Some((h, q)) => (h, q),
            None => (after_scheme, ""),
        };
        let (host, path) = match host_path.split_once('/') {
            Some((h, p)) => (h, p),
            None => (host_path, ""),
        };
        Some((host.to_lowercase(), path.to_lowercase(), query.to_string()))
    };
    let (eh, ep, eq) = match parse(existing_url) {
        Some(v) => v,
        None => return false,
    };
    let (th, tp, tq) = match parse(target_url) {
        Some(v) => v,
        None => return false,
    };
    if eh != th || ep != tp {
        return false;
    }
    // YouTube: only search_query matters; other params are volatile.
    if th.contains("youtube") || eh.contains("youtube") {
        let qparam = |q: &str| {
            q.split('&')
                .find_map(|kv| kv.strip_prefix("search_query="))
                .unwrap_or("")
                .to_string()
        };
        let eq_q = qparam(&eq);
        let tq_q = qparam(&tq);
        // If either side has a search_query, require equality of that param.
        if !eq_q.is_empty() || !tq_q.is_empty() {
            return eq_q == tq_q;
        }
        // Neither has search_query: host+path equality is enough.
        return true;
    }
    // Tolerant dedup for non-YouTube: host+path equality is sufficient;
    // caller will navigate the reused tab to the exact target URL.
    true
}

/// True if two URLs share the same host and path (case-insensitive),
/// ignoring query entirely. Used for tolerant dedup fallback.
pub fn same_host_path(existing_url: &str, target_url: &str) -> bool {
    let host_path = |u: &str| -> Option<(String, String)> {
        let after_scheme = u.split("://").nth(1).unwrap_or(u);
        let host_path = after_scheme
            .split_once('?')
            .map(|(h, _)| h)
            .unwrap_or(after_scheme);
        let (host, path) = match host_path.split_once('/') {
            Some((h, p)) => (h, p),
            None => (host_path, ""),
        };
        Some((host.to_lowercase(), path.to_lowercase()))
    };
    match (host_path(existing_url), host_path(target_url)) {
        (Some((eh, ep)), Some((th, tp))) => eh == th && ep == tp,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_tab_exact() {
        assert!(same_tab_for_reuse(
            "https://www.youtube.com/results?search_query=despacito",
            "https://www.youtube.com/results?search_query=despacito"
        ));
    }

    #[test]
    fn test_same_tab_youtube_query_match_ignores_extra_params() {
        assert!(same_tab_for_reuse(
            "https://www.youtube.com/results?search_query=despacito&sp=foo",
            "https://www.youtube.com/results?search_query=despacito"
        ));
        assert!(!same_tab_for_reuse(
            "https://www.youtube.com/results?search_query=despacito",
            "https://www.youtube.com/results?search_query=bohemian"
        ));
    }

    #[test]
    fn test_same_tab_different_path() {
        assert!(!same_tab_for_reuse(
            "https://www.youtube.com/",
            "https://www.youtube.com/results?search_query=despacito"
        ));
        assert!(!same_tab_for_reuse(
            "https://news.ycombinator.com/",
            "https://www.youtube.com/results?search_query=despacito"
        ));
    }

    #[test]
    fn test_same_tab_tolerant_non_youtube_query_ignored() {
        // Tolerant dedup: same host+path with different query should reuse for non-YouTube
        assert!(same_tab_for_reuse(
            "https://www.google.com/travel/flights?query=tokyo",
            "https://www.google.com/travel/flights?query=paris"
        ));
        assert!(same_tab_for_reuse(
            "https://example.com/a?x=1",
            "https://example.com/a?x=2"
        ));
        // Different path still false
        assert!(!same_tab_for_reuse(
            "https://www.google.com/travel/flights",
            "https://www.google.com/search?q=flights"
        ));
    }

    #[test]
    fn test_same_host_path() {
        assert!(same_host_path(
            "https://www.google.com/travel/flights?x=1",
            "https://www.google.com/travel/flights?y=2"
        ));
        assert!(!same_host_path(
            "https://www.google.com/travel/flights",
            "https://www.google.com/search"
        ));
        assert!(same_host_path(
            "https://example.com/a",
            "https://example.com/a?query=123"
        ));
    }

    #[test]
    fn test_heuristic_pick() {
        let snap = HintSnapshot {
            count: 2,
            hints: vec![
                Hint {
                    label: "A".into(),
                    tag: "button".into(),
                    role: "button".into(),
                    name: "Guide".into(),
                    selector: "#button".into(),
                    text: "Guide".into(),
                    rect: None,
                },
                Hint {
                    label: "S".into(),
                    tag: "a".into(),
                    role: "link".into(),
                    name: "Despacito Luis Fonsi video".into(),
                    selector: "a#video-title".into(),
                    text: "Despacito".into(),
                    rect: None,
                },
            ],
        };
        assert_eq!(
            snap.heuristic_pick("click despacito video").as_deref(),
            Some("S")
        );
        assert!(snap.heuristic_pick("xyzzy nonexistent zzz").is_none());
    }
}
