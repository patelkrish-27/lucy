//! Honesty harness: live counters for the browser decision loop (P0).
//!
//! Every number in `docs/lucy-performance.md` comes from here — no estimates.
//! [`BrowserMetrics`] is bumped at the exact instrumentation points it names:
//! - `cdp_calls` — every [`crate::browser_cdp::BrowserCdpClient::call`]
//! - `evaluates` — every `Runtime.evaluate` via `evaluate()`
//! - `snapshots` — every `observe()` (atomic `snapshot.js` run)
//! - `wait_polls` / `wait_timeouts` / `wait_ms` — every `wait_for_load()`
//!   poll iteration, full-timeout burn, and elapsed wait time
//! - `guard_extra_polls` / `stale_aborts` — `act()` hit-test guard retries
//!   and terminal "covered, hidden, or stale" failures
//! - `hint_calls` / `hint_fallbacks` — `hyprfast hint-act` attempts and misses
//! - `steps` — decision-cycle iterations in `run_browser_loop`
//!
//! [`BrowserRunReport`] is the per-task record the benchmark harness
//! (`tests/bench_tasks.rs`) writes to `target/lucy-bench/<task>.json`.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// Live counters for one browser session. All atomics so the CDP client,
/// the policy loop, and the test harness can share one instance.
#[derive(Debug, Default)]
pub struct BrowserMetrics {
    pub cdp_calls: AtomicU64,
    pub evaluates: AtomicU64,
    pub snapshots: AtomicU64,
    pub wait_polls: AtomicU64,
    pub wait_timeouts: AtomicU64,
    pub wait_ms: AtomicU64,
    pub steps: AtomicU64,
    pub stale_aborts: AtomicU64,
    pub guard_extra_polls: AtomicU64,
    pub hint_calls: AtomicU64,
    pub hint_fallbacks: AtomicU64,
}

impl BrowserMetrics {
    pub fn snapshot(&self) -> BrowserMetricsSnapshot {
        let load = |a: &AtomicU64| a.load(Ordering::SeqCst);
        BrowserMetricsSnapshot {
            cdp_calls: load(&self.cdp_calls),
            evaluates: load(&self.evaluates),
            snapshots: load(&self.snapshots),
            wait_polls: load(&self.wait_polls),
            wait_timeouts: load(&self.wait_timeouts),
            wait_ms: load(&self.wait_ms),
            steps: load(&self.steps),
            stale_aborts: load(&self.stale_aborts),
            guard_extra_polls: load(&self.guard_extra_polls),
            hint_calls: load(&self.hint_calls),
            hint_fallbacks: load(&self.hint_fallbacks),
        }
    }
}

/// Point-in-time copy of [`BrowserMetrics`]. Serializable for bench reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserMetricsSnapshot {
    pub cdp_calls: u64,
    pub evaluates: u64,
    pub snapshots: u64,
    pub wait_polls: u64,
    pub wait_timeouts: u64,
    pub wait_ms: u64,
    pub steps: u64,
    pub stale_aborts: u64,
    pub guard_extra_polls: u64,
    pub hint_calls: u64,
    pub hint_fallbacks: u64,
}

/// Per-task benchmark record: outcome + wall time + model calls + CDP costs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserRunReport {
    pub task: String,
    pub outcome: String,
    pub message: String,
    pub steps: u64,
    pub wall_ms: u64,
    pub laya_predicts: u64,
    pub metrics: BrowserMetricsSnapshot,
}

impl BrowserRunReport {
    pub fn write_json(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }
}
