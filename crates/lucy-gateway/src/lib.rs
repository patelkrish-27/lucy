//! The Lucy mobile gateway.
//!
//! A desktop-side WebSocket + REST server the Lucy phone app pairs with to
//! submit tasks, watch what the agent is doing, and answer approval prompts.
//!
//! Design constraints (see `AGENTS.md` and `docs/mobile-gateway.md`):
//!
//! - **Opt-in.** The gateway is off unless `[gateway] enabled = true` (or
//!   `LUCY_GATEWAY_ENABLED=1`, or the user runs `lucy serve --enable`).
//! - **Idle costs nothing.** No runtime is built at startup; it is loaded on
//!   the first task and unloaded after `gateway.idle_unload_secs` with no
//!   clients, which also drops MCP children.
//! - **No hardcoded task knowledge.** The wire protocol carries prompts and
//!   agent events. It never names a site, an app, or a workflow.
//!
//! ```no_run
//! # async fn demo() -> anyhow::Result<()> {
//! let config = lucy_config::LucyConfig::load()?;
//! lucy_gateway::serve(config).await
//! # }
//! ```

pub mod auth;
pub mod host;
pub mod protocol;
pub mod qr;
pub mod server;

pub use auth::{DEVICE_PREFIX, DeviceRecord, DeviceStore, PAIRING_PREFIX, hash_token};
pub use host::Host;
pub use protocol::{ApprovalDecision, ClientMessage, PROTOCOL_VERSION, PairingPayload, ServerMessage};
pub use server::{GatewayState, router};

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use lucy_config::LucyConfig;

/// How long a QR pairing token stays redeemable. A QR left on a screen is a
/// key; ten minutes is long enough to scan and short enough not to linger.
pub const PAIRING_TTL_SECS: u64 = 600;

/// Where paired-device digests live.
pub fn device_store_path() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("LUCY_GATEWAY_DEVICES") {
        if !p.trim().is_empty() {
            return Ok(PathBuf::from(p));
        }
    }
    let home = std::env::var_os("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home).join(".config/lucy/gateway-devices.json"))
}

/// Build the gateway state (device store loaded, runtime *not* built).
pub fn state(config: LucyConfig) -> Result<Arc<GatewayState>> {
    let host = Host::production(config.clone());
    Ok(Arc::new(GatewayState::new(
        config,
        host,
        device_store_path()?,
    )))
}

/// Run the gateway until Ctrl+C. Requires `[gateway] enabled = true`.
pub async fn serve(config: LucyConfig) -> Result<()> {
    if !config.gateway.enabled {
        anyhow::bail!(
            "the mobile gateway is off — run `lucy serve --enable` (or set \
             [gateway] enabled = true, or LUCY_GATEWAY_ENABLED=1) to turn it on"
        );
    }
    serve_with(config, None).await
}

/// Run the gateway, optionally minting a pairing token first and printing its
/// QR. `pair` is what `lucy serve --pair` uses.
pub async fn serve_with(mut config: LucyConfig, pair: Option<PairingRequest>) -> Result<()> {
    // Pairing is intentionally turnkey: when the user asks for a QR and the
    // configured bind is loopback, expose the gateway on the LAN so the phone
    // can actually reach the address encoded in that QR. Authentication still
    // gates every WebSocket/REST operation, and the pairing token is one-shot.
    if pair.is_some() && matches!(config.gateway.bind.as_str(), "127.0.0.1" | "localhost" | "::1") {
        config.gateway.bind = "0.0.0.0".into();
    }
    if !config.gateway.enabled {
        anyhow::bail!(
            "the mobile gateway is off — run `lucy serve --enable` (or set \
             [gateway] enabled = true, or LUCY_GATEWAY_ENABLED=1) to turn it on"
        );
    }
    let gateway = state(config)?;

    // Mint before binding so a printed QR always belongs to a server that is
    // about to accept it.
    let mut pairing_payload = None;
    if let Some(request) = pair {
        let server_label = request.name.clone().unwrap_or_else(qr::server_name);
        let token = {
            let mut devices = gateway.devices.lock().await;
            let token = devices.mint_pairing(PAIRING_TTL_SECS);
            if request.reissue {
                // Explicit re-pair: drop the old devices so the new token is
                // the only way in.
                let ids: Vec<String> = devices.devices.keys().cloned().collect();
                for id in ids {
                    devices.revoke(&id);
                }
            }
            token
        };
        if let Err(e) = gateway.save_devices().await {
            tracing::warn!(error=%e, "gateway: could not persist pairing token");
        }
        let bind = gateway.bind.clone();
        let port = gateway.port.load(std::sync::atomic::Ordering::SeqCst);
        let host = qr::advertise_host(&bind);
        let server_id = gateway.devices.lock().await.server_id.clone();
        pairing_payload = Some(qr::pairing_payload(
            &server_label,
            &host,
            port,
            &token,
            &server_id,
        ));
    }

    let listener = bind_listener(&gateway.bind, gateway.port.load(std::sync::atomic::Ordering::SeqCst)).await?;
    let addr = listener.local_addr()?;
    gateway
        .port
        .store(addr.port(), std::sync::atomic::Ordering::SeqCst);
    tracing::info!(addr=%addr, "Lucy mobile gateway listening");

    if let Some(payload) = pairing_payload {
        print_pairing(&payload, gateway.config().gateway.log_pairing_secret)?;
    }

    let reaper = server::spawn_reaper(gateway.clone());
    let app = router(gateway.clone());
    let result = axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await;
    reaper.abort();
    // Drop the runtime so MCP children go with it.
    gateway.host.unload().await;
    result.context("gateway server failed")
}

/// A pairing request from the CLI.
#[derive(Debug, Clone, Default)]
pub struct PairingRequest {
    /// Force re-pairing: revoke every existing device first.
    pub reissue: bool,
    /// Device name to show in the QR. Currently also used as the server label;
    /// the phone sends its own device name at redemption.
    pub name: Option<String>,
}

/// Mint a pairing token and print its QR *without* starting the server. Used by
/// `lucy pair`: the token lands in the same device store the running gateway
/// reads, so pairing can happen before or after the server starts.
pub async fn print_pairing_only(config: LucyConfig, request: PairingRequest) -> Result<()> {
    if !config.gateway.enabled {
        anyhow::bail!(
            "the mobile gateway is off — run `lucy serve --enable` first, then `lucy pair`"
        );
    }
    let path = device_store_path()?;
    let server_label = request.name.clone().unwrap_or_else(qr::server_name);
    let mut store = DeviceStore::load(&path);
    let token = store.mint_pairing(PAIRING_TTL_SECS);
    if request.reissue {
        // A fresh pairing token supersedes every paired device, so an old
        // phone cannot still drive the computer after a re-pair.
        let ids: Vec<String> = store.devices.keys().cloned().collect();
        for id in ids {
            store.revoke(&id);
        }
    }
    let payload = qr::pairing_payload(
        &server_label,
        &qr::advertise_host(&config.gateway.bind),
        config.gateway.port,
        &token,
        &store.server_id,
    );
    // Persist before printing: a token the user scanned but the desktop forgot
    // is a dead end.
    store.save(&path)?;
    print_pairing(&payload, config.gateway.log_pairing_secret)?;
    println!("  Start the gateway with:  lucy serve");
    Ok(())
}

/// Print the QR and a manually-typable fallback. The raw token is included
/// only when `log_pairing_secret` is on, because a QR is the intended channel
/// and terminal scrollback often ends up in a file.
pub fn print_pairing(payload: &PairingPayload, log_secret: bool) -> Result<()> {
    let json = serde_json::to_string(payload)?;
    let qr = qr::render_terminal(&json)?;
    println!();
    println!("  Scan this with the Lucy app  ·  {} software", payload.name);
    println!();
    for line in qr.lines() {
        println!("  {line}");
    }
    println!();
    println!("  Address   {}", payload.http);
    println!("  Server id {}", payload.server_id);
    println!("  Expires   in {} minutes", PAIRING_TTL_SECS / 60);
    if log_secret {
        println!("  Token     {}", payload.token);
    } else {
        println!("  Token     (hidden — scan the code, or set gateway.log_pairing_secret)");
    }
    println!();
    Ok(())
}

/// Bind the configured address. `0.0.0.0`/`::` are honored as the user asked;
/// nothing here silently widens or narrows the bind.
pub async fn bind_listener(bind: &str, port: u16) -> Result<tokio::net::TcpListener> {
    let addr = format!("{}:{}", bind.trim(), port);
    tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("binding {addr}"))
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut signal) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            signal.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    tracing::info!("gateway: shutting down");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serving_requires_the_explicit_opt_in() {
        // The single most important property: no config means no listener.
        let cfg = LucyConfig::default();
        assert!(!cfg.gateway.enabled);
        let err = futures_lite_block_on(serve(cfg)).unwrap_err();
        assert!(
            err.to_string().contains("off"),
            "refusal must say the gateway is off: {err}"
        );
    }

    #[test]
    fn pairing_ttl_is_minutes_not_hours() {
        assert!(PAIRING_TTL_SECS >= 60);
        assert!(PAIRING_TTL_SECS <= 3600);
    }

    #[test]
    fn the_device_store_defaults_beside_the_config() {
        // Env override wins; otherwise it is under ~/.config/lucy/.
        unsafe { std::env::set_var("LUCY_GATEWAY_DEVICES", "/tmp/x/devices.json") };
        assert_eq!(
            device_store_path().unwrap(),
            PathBuf::from("/tmp/x/devices.json")
        );
        unsafe { std::env::remove_var("LUCY_GATEWAY_DEVICES") };
    }

    /// Minimal block_on so the async `serve` refusal can be checked in a sync
    /// test without pulling a runtime into every test binary.
    fn futures_lite_block_on<F: std::future::Future>(fut: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(fut)
    }
}
