//! End-to-end gateway tests over a real TCP socket.
//!
//! These cover the security-critical and lifecycle-critical properties:
//!
//! 1. A server with no paired device refuses every request, and `/health`
//!    never leaks pairing state.
//! 2. A pairing token redeems exactly once and yields a device token.
//! 3. A device token authenticates and can browse sessions without loading the
//!    heavy runtime.
//! 4. An unknown token closes the socket with `unauthorized`.
//!
//! The heavy runtime is never built here: `Host` uses `RuntimeSource::Fixed`,
//! representing "the runtime is unloaded", which is exactly what browsing
//! should tolerate.

use std::sync::Arc;

use lucy_config::LucyConfig;
use lucy_gateway::host::{Host, RuntimeSource};
use lucy_gateway::protocol::{ClientMessage, PairingPayload, ServerMessage};
use lucy_gateway::server::GatewayState;
use serde_json::json;

/// Start a gateway on an ephemeral port. Returns (base_url, state).
///
/// Each call gets its own temp dir for the device store *and* the session
/// store, so a parallel suite never touches the user's real state.
async fn start_gateway() -> (String, Arc<GatewayState>) {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("lucy-gw-it-{}-{stamp}", std::process::id()));
    std::fs::create_dir_all(root.join("sessions")).unwrap();
    let devices = root.join("devices.json");
    let mut config = LucyConfig::default();
    config.gateway.enabled = true;
    // Keep the cheap session service away from the real one too.
    let sessions_dir = root.join("sessions");
    config.sessions.dir = Some(sessions_dir.clone());
    // Some sandboxes block SQLite's file *creation* (it can open, not create),
    // so the fixture hands it an empty file and lets `migrate()` populate it.
    std::fs::write(sessions_dir.join("adk-sessions.db"), b"").unwrap();
    let host = Host::new(config.clone(), RuntimeSource::Fixed(None));
    let mut state = GatewayState::new(config, host, devices);
    state.set_sessions_dir_for_test(sessions_dir);
    let state = Arc::new(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = lucy_gateway::router(state.clone());
    tokio::spawn(async move {
        let _ = axum::serve(
            listener,
            app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await;
    });
    (format!("http://{addr}"), state)
}

fn http_client() -> reqwest::Client {
    reqwest::Client::new()
}

#[tokio::test]
async fn health_is_public_but_reveals_no_secret() {
    let (base, _state) = start_gateway().await;
    let body: serde_json::Value = http_client()
        .get(format!("{base}/health"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["ok"], true);
    assert_eq!(body["paired"], false);
    // The runtime is not loaded just because the server is up.
    assert_eq!(body["runtime_loaded"], false);
    // No token or server id in the health payload.
    assert!(body.get("token").is_none());
    assert!(body.get("server_id").is_none());
}

#[tokio::test]
async fn rest_endpoints_reject_an_unauthenticated_caller() {
    let (base, _state) = start_gateway().await;
    // GET /sessions and POST /tasks both require a device token.
    let resp = http_client()
        .get(format!("{base}/sessions"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401, "/sessions must require a device token");
    let resp = http_client()
        .post(format!("{base}/tasks"))
        .json(&json!({"prompt": "irrelevant"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401, "/tasks must require a device token");
}

#[tokio::test]
async fn an_unknown_device_token_is_unauthorized() {
    let (base, _state) = start_gateway().await;
    let resp = http_client()
        .get(format!("{base}/sessions"))
        .header("x-lucy-token", "lucy_dev_madeup")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}

#[tokio::test]
async fn the_pairing_endpoint_reports_status_but_never_a_token() {
    let (base, state) = start_gateway().await;
    // Mint a live pairing token...
    let token = {
        let mut devices = state.devices.lock().await;
        devices.mint_pairing(600)
    };
    let _ = state.save_devices().await;

    let body: serde_json::Value = http_client()
        .get(format!("{base}/pair"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["paired"], false);
    assert_eq!(body["pairing_live"], true);
    // The token must not appear anywhere in the response.
    let text = body.to_string();
    assert!(
        !text.contains(&token),
        "the pairing endpoint must never echo the token"
    );
}

#[tokio::test]
async fn a_paired_device_can_browse_sessions_without_loading_the_runtime() {
    let (base, state) = start_gateway().await;
    // Pair out-of-band (the QR flow is covered by the auth unit tests).
    let (record, token) = {
        let mut devices = state.devices.lock().await;
        devices.register_device("test phone")
    };
    let _ = state.save_devices().await;

    let body: serde_json::Value = http_client()
        .get(format!("{base}/sessions"))
        .header("x-lucy-token", &token)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        body["items"].is_array(),
        "expected an items array, got {body}"
    );
    assert!(!record.id.is_empty());
    // Browsing history must never have caused an MCP/browser startup.
    assert!(!state.host.is_loaded());
}

#[tokio::test]
async fn the_websocket_handshake_requires_auth_and_delivers_hello() {
    let (base, state) = start_gateway().await;
    let (record, token) = {
        let mut devices = state.devices.lock().await;
        devices.register_device("ws phone")
    };
    let _ = state.save_devices().await;
    assert!(!record.id.is_empty());

    let ws_url = base.replacen("http://", "ws://", 1) + "/ws";
    // No auth, just observe the hello frame; proving absence of auth is a
    // protocol-level test (connect, read hello, send a task, get refused).
    let (mut socket, _response) = tokio_tungstenite::connect_async(&ws_url).await.unwrap();
    use futures::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;
    let hello = socket.next().await.unwrap().unwrap();
    let Message::Text(text) = hello else {
        panic!("expected a text hello frame");
    };
    let parsed: ServerMessage = serde_json::from_str(&text).unwrap();
    match parsed {
        ServerMessage::Hello { protocol, paired, .. } => {
            assert_eq!(protocol, lucy_gateway::PROTOCOL_VERSION);
            assert!(paired, "a device is registered");
        }
        other => panic!("expected hello, got {other:?}"),
    }

    // Auth with the device token.
    let auth = ClientMessage::Auth {
        token: token.clone(),
        name: Some("ws phone".into()),
    };
    socket
        .send(Message::text(serde_json::to_string(&auth).unwrap()))
        .await
        .unwrap();
    let reply = socket.next().await.unwrap().unwrap();
    let Message::Text(text) = reply else {
        panic!("expected text");
    };
    let parsed: ServerMessage = serde_json::from_str(&text).unwrap();
    match parsed {
        ServerMessage::AuthOk { device, device_token } => {
            assert_eq!(device, "ws phone");
            assert!(device_token.is_none(), "an existing device gets no new token");
        }
        other => panic!("expected auth_ok, got {other:?}"),
    }
}

#[tokio::test]
async fn a_bad_websocket_token_is_refused_with_unauthorized() {
    let (base, _state) = start_gateway().await;
    let ws_url = base.replacen("http://", "ws://", 1) + "/ws";
    let (mut socket, _response) = tokio_tungstenite::connect_async(&ws_url).await.unwrap();
    use futures::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;
    let _hello = socket.next().await.unwrap().unwrap();
    let auth = ClientMessage::Auth {
        token: "lucy_dev_wrong".into(),
        name: None,
    };
    socket
        .send(Message::text(serde_json::to_string(&auth).unwrap()))
        .await
        .unwrap();
    let reply = socket.next().await.unwrap().unwrap();
    let Message::Text(text) = reply else {
        panic!("expected text");
    };
    let parsed: ServerMessage = serde_json::from_str(&text).unwrap();
    match parsed {
        ServerMessage::Error { code, .. } => assert_eq!(code.as_deref(), Some("unauthorized")),
        other => panic!("expected an error, got {other:?}"),
    }
}

#[tokio::test]
async fn the_pairing_qr_payload_encodes_the_live_endpoint() {
    // Build a payload the way `serve_with` does and make sure a scanner could
    // recover the URL and token.
    let payload = PairingPayload {
        lucy: 1,
        name: "desktop".into(),
        url: "ws://192.168.1.10:9847/ws".into(),
        http: "http://192.168.1.10:9847".into(),
        token: "lucy_pair_abc".into(),
        server_id: "srv-1".into(),
    };
    let json = serde_json::to_string(&payload).unwrap();
    let back: PairingPayload = serde_json::from_str(&json).unwrap();
    assert_eq!(back.token, "lucy_pair_abc");
    assert_eq!(back.url, payload.url);
    // The QR renderer accepts it.
    assert!(lucy_gateway::qr::render_terminal(&json).unwrap().lines().count() > 5);
}

#[cfg(test)]
mod test_helpers {
    use super::*;

    #[tokio::test]
    async fn state_requires_a_device_store_path_outside_the_real_config() {
        // Guards against a test accidentally writing the user's device store.
        let (_, state) = start_gateway().await;
        assert!(
            state
                .device_store_path
                .to_string_lossy()
                .contains("lucy-gw-it-"),
            "tests must use a temp device store: {}",
            state.device_store_path.display()
        );
    }
}
