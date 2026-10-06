//! Lazy ownership of the heavy [`LucyRuntime`].
//!
//! Everything the gateway does beyond pairing and session browsing needs a
//! runtime: session store, ADK memory, MCP handshakes, the model provider.
//! Building one is expensive (spawns/attaches hyprfast, opens SQLite), and an
//! idle phone bridge should not pay for it.
//!
//! So the host is a state machine:
//!
//! ```text
//! Unloaded ──ensure()──▶ Loading ──ok──▶ Loaded
//!    ▲                                    │
//!    └────────── idle timeout ────────────┘
//! ```
//!
//! `ensure()` is what a task submission calls; every other request that can be
//! answered from cheap state (sessions list, health) reads through
//! [`Host::with_loaded`] and reports "not loaded" rather than forcing a load.
//! A background reaper drops the runtime after `idle_unload_secs` with no
//! connected clients, which also drops MCP children because the clients are
//! `kill_on_drop`.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use anyhow::{Result, anyhow};
use lucy_config::LucyConfig;
use tokio::sync::{Mutex, RwLock};

use lucy_runtime::LucyRuntime;

/// Run an open sequence that is not provably `Send` to completion on a blocking
/// thread, with the caller's runtime handle so any Tokio IO resources it
/// registers still belong to the caller's runtime.
///
/// Why this exists: sqlx 0.8's `Executor` impl has a higher-ranked lifetime
/// that rustc cannot prove for the ADK openers (`adk-session` 2.2.0 and
/// `adk-memory` 2.2.0 `migrate()`), so `LucyRuntime::new()` compiles in place
/// but fails the `Send` bound the moment it is boxed, spawned, or used in an
/// axum handler. The non-`Send` future never crosses a thread boundary as a
/// future object — it is created and driven inside the blocking thread, and
/// only its `Send` output crosses back.
pub async fn offload<R, F, Fut>(f: F) -> Result<R>
where
    R: Send + 'static,
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = Result<R>>,
{
    let handle = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || handle.block_on(f()))
        .await
        .map_err(|e| anyhow!("runtime builder thread failed: {e}"))?
}

/// How a [`Host`] builds its runtime. Kept as a small enum rather than a
/// `dyn Fn() -> BoxFuture` because boxing `LucyRuntime::new()` runs into the
/// sqlx higher-ranked lifetime limitation described on [`offload`].
pub enum RuntimeSource {
    /// The real runtime: spawns/attaches hyprfast, opens SQLite.
    Production,
    /// A test runtime, already built. `Host` takes ownership on first load so
    /// it cannot be handed out twice.
    Fixed(Option<LucyRuntime>),
}

impl RuntimeSource {
    async fn build(&mut self) -> Result<LucyRuntime> {
        match self {
            RuntimeSource::Production => offload(LucyRuntime::new).await,
            RuntimeSource::Fixed(slot) => slot
                .take()
                .ok_or_else(|| anyhow!("test runtime already consumed")),
        }
    }
}

/// The gateway's view of the runtime, including when it was last touched.
pub struct Host {
    runtime: RwLock<Option<Arc<LucyRuntime>>>,
    /// Serializes concurrent first-loads so two tasks arriving together build
    /// one runtime, not two MCP child sets.
    loading: Mutex<()>,
    /// Unix seconds of the last task/client activity; the reaper compares
    /// against `idle_unload_secs`.
    last_active: AtomicU64,
    /// Connected WebSocket clients. Non-zero pins the runtime: the phone is
    /// watching, so unloading would be visible as a stall.
    clients: AtomicU64,
    loaded_ever: AtomicBool,
    config: LucyConfig,
    source: Mutex<RuntimeSource>,
}

impl Host {
    pub fn new(config: LucyConfig, source: RuntimeSource) -> Self {
        Self {
            runtime: RwLock::new(None),
            loading: Mutex::new(()),
            last_active: AtomicU64::new(now_secs()),
            clients: AtomicU64::new(0),
            loaded_ever: AtomicBool::new(false),
            config,
            source: Mutex::new(source),
        }
    }

    /// Production host: builds a real [`LucyRuntime`] on first use.
    pub fn production(config: LucyConfig) -> Self {
        Self::new(config, RuntimeSource::Production)
    }

    pub fn is_loaded(&self) -> bool {
        self.runtime
            .try_read()
            .map(|g| g.is_some())
            .unwrap_or(false)
    }

    pub fn loaded_ever(&self) -> bool {
        self.loaded_ever.load(Ordering::SeqCst)
    }

    pub fn idle_unload_secs(&self) -> u64 {
        self.config.gateway.idle_unload_secs
    }

    /// Current runtime if loaded, without triggering a load.
    pub fn current(&self) -> Option<Arc<LucyRuntime>> {
        self.runtime.try_read().ok().and_then(|g| g.clone())
    }

    /// Run `f` against the loaded runtime, or `None` when it is unloaded.
    pub async fn with_loaded<R>(&self, f: impl FnOnce(&Arc<LucyRuntime>) -> R) -> Option<R> {
        let guard = self.runtime.read().await;
        guard.as_ref().map(f)
    }

    /// Mark activity. Every client frame and server event calls this so the
    /// reaper measures *use*, not uptime.
    pub fn touch(&self) {
        self.last_active.store(now_secs(), Ordering::SeqCst);
    }

    pub fn client_connected(&self) {
        self.clients.fetch_add(1, Ordering::SeqCst);
        self.touch();
    }

    pub fn client_disconnected(&self) {
        // Saturating: a mismatched disconnect must not wrap the counter.
        let _ = self
            .clients
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                Some(n.saturating_sub(1))
            });
        self.touch();
    }

    pub fn clients(&self) -> u64 {
        self.clients.load(Ordering::SeqCst)
    }

    /// Load the runtime if needed and return a clone of the handle. Concurrent
    /// callers wait on `loading` and then observe the loaded value.
    ///
    /// This is the *only* path that allocates the expensive runtime; every
    /// cheap operation must go through [`Self::with_loaded`] instead.
    pub async fn ensure(&self) -> Result<Arc<LucyRuntime>> {
        if let Some(rt) = self.current() {
            self.touch();
            return Ok(rt);
        }
        let _guard = self.loading.lock().await;
        // Another caller may have won the race while we waited.
        if let Some(rt) = self.current() {
            self.touch();
            return Ok(rt);
        }
        tracing::info!("gateway: loading Lucy runtime");
        let mut source = self.source.lock().await;
        let built = source.build().await?;
        drop(source);
        let rt = Arc::new(built);
        *self.runtime.write().await = Some(rt.clone());
        self.loaded_ever.store(true, Ordering::SeqCst);
        self.touch();
        tracing::info!("gateway: runtime loaded");
        Ok(rt)
    }

    /// Drop the runtime if it is loaded and has been idle long enough with no
    /// clients. Returns true when something was unloaded.
    pub async fn reap_if_idle(&self, now: u64) -> bool {
        let timeout = self.config.gateway.idle_unload_secs;
        if timeout == 0 || self.clients() > 0 || !self.is_loaded() {
            return false;
        }
        let last = self.last_active.load(Ordering::SeqCst);
        if now.saturating_sub(last) < timeout {
            return false;
        }
        let mut guard = self.runtime.write().await;
        if guard.take().is_some() {
            tracing::info!(
                idle_secs = now.saturating_sub(last),
                "gateway: unloading idle runtime"
            );
            return true;
        }
        false
    }

    /// Force-unload now (used by shutdown paths and tests).
    pub async fn unload(&self) {
        let _ = self.runtime.write().await.take();
    }
}

impl std::fmt::Debug for Host {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Host")
            .field("loaded", &self.is_loaded())
            .field("clients", &self.clients())
            .field("idle_unload_secs", &self.idle_unload_secs())
            .finish()
    }
}

pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A source that always fails: the point is counting load attempts, not
    /// producing a runtime.
    fn failing_source() -> RuntimeSource {
        // `Fixed(None)` fails on first load and never fabricates one.
        RuntimeSource::Fixed(None)
    }

    #[tokio::test]
    async fn an_idle_host_has_no_runtime_and_never_built_one() {
        let host = Host::new(LucyConfig::default(), failing_source());
        assert!(!host.is_loaded());
        assert!(!host.loaded_ever());
        assert!(host.current().is_none());
        assert_eq!(host.with_loaded(|_| ()).await, None);
    }

    #[tokio::test]
    async fn ensure_attempts_a_load_only_when_asked() {
        let host = Host::new(LucyConfig::default(), failing_source());
        assert!(host.ensure().await.is_err());
        assert!(!host.is_loaded());
        assert!(!host.loaded_ever());
    }

    #[tokio::test]
    async fn a_disabled_idle_timeout_never_reaps() {
        let mut cfg = LucyConfig::default();
        cfg.gateway.idle_unload_secs = 0;
        let host = Host::new(cfg, failing_source());
        assert!(!host.reap_if_idle(now_secs() + 10_000).await);
    }

    #[tokio::test]
    async fn an_active_client_pins_the_runtime() {
        let mut cfg = LucyConfig::default();
        cfg.gateway.idle_unload_secs = 1;
        let host = Host::new(cfg, failing_source());
        host.client_connected();
        assert_eq!(host.clients(), 1);
        assert!(!host.reap_if_idle(now_secs() + 1_000).await);
        host.client_disconnected();
        assert_eq!(host.clients(), 0);
        host.client_disconnected();
        assert_eq!(host.clients(), 0, "an extra disconnect must not underflow");
    }
}
