//! The HTTP/WebSocket server.
//!
//! Routes:
//!
//! | Method | Path | Auth | Purpose |
//! |---|---|---|---|
//! | GET | `/health` | no | liveness + runtime state |
//! | GET | `/pair` | no | pairing info (never a token) |
//! | GET | `/sessions` | yes | session list |
//! | GET | `/sessions/:id` | yes | one session |
//! | GET | `/ws` | handshake | live channel |
//! | POST | `/tasks` | yes | one-shot task submit (no socket) |
//!
//! `/ws` is the real integration: the app holds one socket, receives a `hello`,
//! sends `auth`, then submits tasks and receives the agent's event stream.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path as AxumPath, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};
use tokio::sync::{Mutex, mpsc};
use uuid::Uuid;

use lucy_adk::LucySessionService;
use lucy_config::LucyConfig;
use lucy_core::{AgentEvent, SessionId};

use crate::auth::DeviceStore;
use crate::host::{Host, now_secs};
use crate::protocol::{ApprovalDecision, ClientMessage, PROTOCOL_VERSION, ServerMessage};

/// Shared server state.
pub struct GatewayState {
    pub host: Host,
    pub devices: Mutex<DeviceStore>,
    pub device_store_path: PathBuf,
    /// Cheap session service, opened lazily and independently of the heavy
    /// runtime so browsing sessions never spawns MCP children.
    sessions: Mutex<Option<Arc<LucySessionService>>>,
    /// Overrides the session directory. Used by tests so a suite never touches
    /// the real `~/.local/state/lucy/sessions`. Written before the server
    /// starts, so a plain field is enough (and `try_lock` races are impossible).
    sessions_dir_override: Option<PathBuf>,
    pub bind: String,
    pub port: std::sync::atomic::AtomicU16,
    config: LucyConfig,
    /// Live task count, for `/health` and to make "one task at a time" a
    /// deliberate refusal rather than a race.
    running_task: Mutex<Option<String>>,
}

impl GatewayState {
    pub fn new(config: LucyConfig, host: Host, device_store_path: PathBuf) -> Self {
        let devices = DeviceStore::load(&device_store_path);
        let bind = config.gateway.bind.clone();
        let port = config.gateway.port;
        Self {
            host,
            devices: Mutex::new(devices),
            device_store_path,
            sessions: Mutex::new(None),
            sessions_dir_override: None,
            bind,
            port: std::sync::atomic::AtomicU16::new(port),
            config,
            running_task: Mutex::new(None),
        }
    }

    /// Point the cheap session service at a directory (tests).
    ///
    /// Must be called before the server starts; it does not create the
    /// directory, which is the opener's job.
    pub fn set_sessions_dir_for_test(&mut self, dir: PathBuf) {
        self.sessions_dir_override = Some(dir);
    }

    pub fn config(&self) -> &LucyConfig {
        &self.config
    }

    /// Persist the device store, keeping memory and disk in step.
    pub async fn save_devices(&self) -> anyhow::Result<()> {
        let guard = self.devices.lock().await;
        guard.save(&self.device_store_path)
    }

    /// Open (or return) the cheap session service.
    ///
    /// Deliberately *not* the runtime's service: a phone that only wants to
    /// browse history must not cause an MCP handshake or a browser attach.
    ///
    /// The opener runs through [`crate::host::offload`] because the ADK
    /// `open()` future is not provably `Send` (see that function for the sqlx
    /// higher-ranked lifetime detail).
    pub async fn sessions(&self) -> anyhow::Result<Arc<LucySessionService>> {
        let mut guard = self.sessions.lock().await;
        if let Some(s) = guard.as_ref() {
            return Ok(s.clone());
        }
        let dir = match &self.sessions_dir_override {
            Some(dir) => dir.clone(),
            None => self
                .config
                .sessions
                .dir
                .clone()
                .or_else(|| std::env::var("LUCY_SESSIONS_DIR").ok().map(PathBuf::from))
                .unwrap_or_else(|| {
                    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
                        .join(".local/state/lucy/sessions")
                }),
        };
        let service = Arc::new(
            crate::host::offload(move || async move {
                // The opener creates the directory itself, but doing it here
                // keeps a failed `create_dir_all` inside the same error chain.
                if let Some(parent) = dir.parent() {
                    tokio::fs::create_dir_all(parent).await.ok();
                }
                LucySessionService::open(dir).await
            })
            .await?,
        );
        *guard = Some(service.clone());
        Ok(service)
    }

    /// True when this gateway has at least one paired device.
    pub async fn paired(&self) -> bool {
        self.devices.lock().await.is_paired()
    }

    async fn begin_task(&self, task_id: &str) -> bool {
        let mut guard = self.running_task.lock().await;
        if guard.is_some() {
            return false;
        }
        *guard = Some(task_id.to_string());
        true
    }

    async fn end_task(&self) {
        *self.running_task.lock().await = None;
    }

    async fn running_task(&self) -> Option<String> {
        self.running_task.lock().await.clone()
    }
}

/// Build the router. Public so tests can drive it with `axum::serve` on an
/// ephemeral port.
pub fn router(state: Arc<GatewayState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/pair", get(pair_info))
        .route("/sessions", get(list_sessions))
        .route("/sessions/{id}", get(get_session))
        .route("/tasks", post(submit_task))
        .route("/ws", get(ws_upgrade))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// REST
// ---------------------------------------------------------------------------

async fn health(State(state): State<Arc<GatewayState>>) -> impl IntoResponse {
    let running = state.running_task().await;
    Json(json!({
        "ok": true,
        "server": "lucy-gateway",
        "protocol": PROTOCOL_VERSION,
        "paired": state.paired().await,
        "runtime_loaded": state.host.is_loaded(),
        "clients": state.host.clients(),
        "running_task": running,
    }))
}

/// Pairing *status* only. The token exists in the QR and nowhere this endpoint
/// can leak: a browser tab on the same machine must not be able to pair itself.
async fn pair_info(State(state): State<Arc<GatewayState>>) -> impl IntoResponse {
    let devices = state.devices.lock().await;
    Json(json!({
        "paired": devices.is_paired(),
        "pairing_live": devices.pairing_live(),
        "server_id": devices.server_id,
        "devices": devices.devices.values().map(|d| json!({
            "id": d.id,
            "name": d.name,
            "created_at": d.created_at,
            "last_seen_at": d.last_seen_at,
        })).collect::<Vec<_>>(),
    }))
}

/// Extract a bearer token from either `Authorization: Bearer …` or
/// `X-Lucy-Token:` (the app uses the header form because WebSocket clients in
/// some runtimes cannot set an Authorization header).
fn bearer(headers: &HeaderMap) -> Option<String> {
    if let Some(v) = headers.get("x-lucy-token").and_then(|v| v.to_str().ok()) {
        if !v.trim().is_empty() {
            return Some(v.trim().to_string());
        }
    }
    let v = headers.get("authorization")?.to_str().ok()?;
    let token = v.strip_prefix("Bearer ").or_else(|| v.strip_prefix("bearer "))?;
    (!token.trim().is_empty()).then(|| token.trim().to_string())
}

/// Resolve the caller's device, touching its last-seen time on the way.
async fn authorize(state: &GatewayState, headers: &HeaderMap) -> Option<String> {
    let token = bearer(headers)?;
    let mut devices = state.devices.lock().await;
    let record = devices.authenticate(&token)?;
    devices.touch(&record.id);
    Some(record.id)
}

async fn list_sessions(
    State(state): State<Arc<GatewayState>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if authorize(&state, &headers).await.is_none() {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "unauthorized"}))).into_response();
    }
    let sessions = match state.sessions().await {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("session store unavailable: {e:#}")})),
            )
                .into_response();
        }
    };
    match sessions.list().await {
        Ok(items) => Json(json!({"items": items})).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

async fn get_session(
    State(state): State<Arc<GatewayState>>,
    headers: HeaderMap,
    AxumPath(id): AxumPath<String>,
) -> impl IntoResponse {
    if authorize(&state, &headers).await.is_none() {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "unauthorized"}))).into_response();
    }
    let Some(session_id) = parse_session_id(&id) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "invalid session id"})),
        )
            .into_response();
    };
    let sessions = match state.sessions().await {
        Ok(s) => s,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("session store unavailable: {e:#}")})),
            )
                .into_response();
        }
    };
    match sessions.load(&session_id).await {
        Ok(session) => Json(json!({"session": session})).into_response(),
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

/// One-shot submit for clients that do not hold a socket. The agent runs to
/// completion and the final summary is returned; live events are not streamed.
async fn submit_task(
    State(state): State<Arc<GatewayState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    if authorize(&state, &headers).await.is_none() {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "unauthorized"}))).into_response();
    }
    let Some(prompt) = body.get("prompt").and_then(|p| p.as_str()) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "missing prompt"})),
        )
            .into_response();
    };
    let prompt = prompt.trim().to_string();
    if prompt.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "prompt must not be empty"})),
        )
            .into_response();
    }
    let task_id = format!("rest-{}", Uuid::new_v4());
    if !state.begin_task(&task_id).await {
        return (
            StatusCode::CONFLICT,
            Json(json!({"error": "another task is already running"})),
        )
            .into_response();
    }
    let result = run_task_once(&state, &prompt).await;
    state.end_task().await;
    match result {
        Ok((summary, complete)) => {
            Json(json!({"summary": summary, "complete": complete})).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

/// Run one task with no live stream, collecting nothing but the outcome.
async fn run_task_once(state: &Arc<GatewayState>, prompt: &str) -> anyhow::Result<(String, bool)> {
    let rt = state.host.ensure().await?;
    rt.save_user_message(prompt.to_string()).await?;
    // No routing verdict: the decider is unavailable/undecided. Stop here —
    // no tools run on an unrouted turn.
    let classification = match rt.classify_turn(prompt).await {
        Ok(c) => c,
        Err(e) => {
            let summary = format!("✖ {e}");
            rt.save_assistant_text(summary.clone()).await?;
            return Ok((summary, false));
        }
    };
    let route = match rt.route_turn_with(prompt, Some(classification)).await {
        Ok(r) => r,
        Err(e) => {
            let summary = format!("✖ {e}");
            rt.save_assistant_text(summary.clone()).await?;
            return Ok((summary, false));
        }
    };
    if !route.needs_actions() {
        let hint = route.knowledge_topic.clone();
        let reply = rt.answer_turn_with_knowledge(prompt, &route, hint.as_deref()).await?;
        rt.save_assistant_text(reply.clone()).await?;
        return Ok((reply, true));
    }
    let (tx, mut rx) = mpsc::unbounded_channel::<AgentEvent>();
    let runner_rt = rt.clone();
    let goal = prompt.to_string();
    let runner = tokio::spawn(async move {
        runner_rt
            .execute_goal_outcome(&goal, Some(&route), Some(tx))
            .await
    });
    // No UI on this path: approve read-only tools, deny the rest. A caller
    // that needs approvals must use the socket.
    while let Some(event) = rx.recv().await {
        if let AgentEvent::ApprovalRequest { id, .. } = event {
            rt.approval_gate()
                .resolve(&id, lucy_core::ApprovalDecision::Deny);
        }
    }
    match runner.await {
        Ok(Ok(outcome)) => {
            rt.save_assistant_text(outcome.summary.clone()).await?;
            Ok((outcome.summary, outcome.complete))
        }
        Ok(Err(e)) => {
            let summary = format!("Goal not completed: {e}");
            rt.save_assistant_text(summary.clone()).await?;
            Ok((summary, false))
        }
        Err(e) => Err(anyhow::anyhow!("task failed: {e}")),
    }
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

async fn ws_upgrade(
    State(state): State<Arc<GatewayState>>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    let _ = headers;
    ws.on_upgrade(move |socket| async move {
        connection_loop(state, socket).await;
    })
}

/// Everything one socket does, in order: hello, auth, then messages.
async fn connection_loop(state: Arc<GatewayState>, mut socket: WebSocket) {
    let paired = state.paired().await;
    let hello = ServerMessage::Hello {
        protocol: PROTOCOL_VERSION,
        server: crate::qr::server_name(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        paired,
        runtime_loaded: state.host.is_loaded(),
    };
    state.host.client_connected();
    if socket
        .send(Message::text(hello.to_json()))
        .await
        .is_err()
    {
        state.host.client_disconnected();
        return;
    }

    // Auth must be the first client frame; anything else closes the socket.
    let mut authed: Option<String> = None;
    while authed.is_none() {
        let Some(Ok(msg)) = socket.recv().await else {
            state.host.client_disconnected();
            return;
        };
        let Message::Text(text) = msg else {
            continue;
        };
        let Ok(client) = serde_json::from_str::<ClientMessage>(&text) else {
            let _ = send(&mut socket, &ServerMessage::Error {
                message: "expected an auth message first".into(),
                code: Some("auth_required".into()),
            })
            .await;
            break;
        };
        match client {
            ClientMessage::Auth { token, name } => {
                let device_name = name.unwrap_or_else(|| "phone".into());
                let mut devices = state.devices.lock().await;
                if let Some(record) = devices.authenticate(&token) {
                    devices.touch(&record.id);
                    drop(devices);
                    if let Err(e) = state.save_devices().await {
                        tracing::warn!(error=%e, "gateway: could not persist device last-seen");
                    }
                    authed = Some(record.id.clone());
                    let _ = send(&mut socket, &ServerMessage::AuthOk {
                        device: record.name,
                        device_token: None,
                    })
                    .await;
                } else if let Some((record, device_token)) =
                    devices.redeem_pairing(&token, &device_name)
                {
                    let record_id = record.id.clone();
                    let record_name = record.name.clone();
                    drop(devices);
                    if let Err(e) = state.save_devices().await {
                        let _ = send(&mut socket, &ServerMessage::Error {
                            message: format!("paired, but could not save the device: {e}"),
                            code: Some("persist_failed".into()),
                        })
                        .await;
                        break;
                    }
                    authed = Some(record_id);
                    // The only time the long-lived token crosses the wire.
                    let _ = send(&mut socket, &ServerMessage::AuthOk {
                        device: record_name,
                        device_token: Some(device_token),
                    })
                    .await;
                } else {
                    drop(devices);
                    let _ = send(&mut socket, &ServerMessage::Error {
                        message: if paired {
                            "unknown device token — pair again from the desktop".into()
                        } else {
                            "not paired — run `lucy serve --pair` on the desktop".into()
                        },
                        code: Some("unauthorized".into()),
                    })
                    .await;
                    break;
                }
            }
            ClientMessage::Ping => {
                let _ = send(&mut socket, &ServerMessage::Pong).await;
            }
            _ => {
                let _ = send(&mut socket, &ServerMessage::Error {
                    message: "expected an auth message first".into(),
                    code: Some("auth_required".into()),
                })
                .await;
                break;
            }
        }
    }

    let Some(device_id) = authed else {
        state.host.client_disconnected();
        return;
    };
    state.host.touch();

    // From here on the socket owns a task runner. The runner forwards agent
    // events into `event_tx`; the select loop multiplexes socket input and
    // those events.
    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<ServerMessage>();
    let mut current_task: Option<String> = None;

    loop {
        tokio::select! {
            incoming = socket.recv() => {
                let Some(Ok(msg)) = incoming else { break };
                let Message::Text(text) = msg else { continue };
                let Ok(client) = serde_json::from_str::<ClientMessage>(&text) else {
                    let _ = send(&mut socket, &ServerMessage::Error {
                        message: "unparseable message".into(),
                        code: Some("bad_request".into()),
                    }).await;
                    continue;
                };
                state.host.touch();
                match client {
                    ClientMessage::Auth { .. } => {
                        let _ = send(&mut socket, &ServerMessage::Error {
                            message: "already authenticated".into(),
                            code: Some("already_authed".into()),
                        }).await;
                    }
                    ClientMessage::Ping => {
                        let _ = send(&mut socket, &ServerMessage::Pong).await;
                    }
                    ClientMessage::Task { id, prompt } => {
                        if current_task.is_some() {
                            let _ = send(&mut socket, &ServerMessage::Error {
                                message: "a task is already running; stop it or wait".into(),
                                code: Some("busy".into()),
                            }).await;
                            continue;
                        }
                        let task_id = id.unwrap_or_else(|| Uuid::new_v4().to_string());
                        if !state.begin_task(&task_id).await {
                            let _ = send(&mut socket, &ServerMessage::Error {
                                message: "another client is running a task".into(),
                                code: Some("busy".into()),
                            }).await;
                            continue;
                        }
                        current_task = Some(task_id.clone());
                        let _ = send(&mut socket, &ServerMessage::TaskStarted {
                            task_id: task_id.clone(),
                            prompt: prompt.clone(),
                        }).await;
                        spawn_task(state.clone(), task_id, prompt, event_tx.clone());
                    }
                    ClientMessage::Approval { id, decision } => {
                        let decision: lucy_core::ApprovalDecision = decision.into();
                        let resolved = state
                            .host
                            .with_loaded(|rt| rt.approval_gate().resolve(&id, decision))
                            .await;
                        if resolved == Some(true) {
                            if decision == lucy_core::ApprovalDecision::AllowAlways {
                                // Persist the always-allow set like the TUI does.
                                let _ = state.host.with_loaded(|rt| rt.persist_always_allowed()).await;
                            }
                        } else {
                            let _ = send(&mut socket, &ServerMessage::Error {
                                message: format!("no pending approval with id {id}"),
                                code: Some("unknown_approval".into()),
                            }).await;
                        }
                    }
                    ClientMessage::Stop => {
                        state.host.with_loaded(|rt| rt.interrupt()).await;
                        // Deny anything still pending so the blocked ask()
                        // returns instead of waiting out its timeout.
                        state.host.with_loaded(|rt| {
                            for id in rt.approval_gate().pending_ids() {
                                rt.approval_gate().resolve(&id, lucy_core::ApprovalDecision::Deny);
                            }
                        }).await;
                    }
                    ClientMessage::SessionsList => {
                        match state.sessions().await {
                            Ok(sessions) => match sessions.list().await {
                                Ok(items) => {
                                    let current = current_session_hint(&state).await;
                                    let _ = send(&mut socket, &ServerMessage::Sessions { items, current }).await;
                                }
                                Err(e) => {
                                    let _ = send(&mut socket, &ServerMessage::Error {
                                        message: e.to_string(),
                                        code: Some("sessions_failed".into()),
                                    }).await;
                                }
                            },
                            Err(e) => {
                                let _ = send(&mut socket, &ServerMessage::Error {
                                    message: e.to_string(),
                                    code: Some("sessions_failed".into()),
                                }).await;
                            }
                        }
                    }
                    ClientMessage::SessionsGet { id } => {
                        let session = match parse_session_id(&id) {
                            Some(sid) => match state.sessions().await {
                                Ok(sessions) => sessions.load(&sid).await.map_err(|e| e.to_string()),
                                Err(e) => Err(e.to_string()),
                            },
                            None => Err("invalid session id".to_string()),
                        };
                        match session {
                            Ok(session) => {
                                let _ = send(&mut socket, &ServerMessage::Session { session: Box::new(session) }).await;
                            }
                            Err(message) => {
                                let _ = send(&mut socket, &ServerMessage::Error { message, code: Some("sessions_failed".into()) }).await;
                            }
                        }
                    }
                    ClientMessage::SessionsNew { title } => {
                        match state.host.ensure().await {
                            Ok(rt) => match rt.new_session(title).await {
                                Ok(meta) => {
                                    let _ = send(&mut socket, &ServerMessage::Sessions {
                                        items: rt.list_sessions().await.unwrap_or_default(),
                                        current: Some(meta.id.0.to_string()),
                                    }).await;
                                }
                                Err(e) => {
                                    let _ = send(&mut socket, &ServerMessage::Error { message: e.to_string(), code: Some("sessions_failed".into()) }).await;
                                }
                            },
                            Err(e) => {
                                let _ = send(&mut socket, &ServerMessage::Error { message: e.to_string(), code: Some("sessions_failed".into()) }).await;
                            }
                        }
                    }
                    ClientMessage::SessionsSwitch { id } => {
                        let sid = parse_session_id(&id);
                        match (state.host.ensure().await, sid) {
                            (Ok(rt), Some(sid)) => match rt.switch_session(&sid).await {
                                Ok(meta) => {
                                    let _ = send(&mut socket, &ServerMessage::Sessions {
                                        items: rt.list_sessions().await.unwrap_or_default(),
                                        current: Some(meta.id.0.to_string()),
                                    }).await;
                                }
                                Err(e) => {
                                    let _ = send(&mut socket, &ServerMessage::Error { message: e.to_string(), code: Some("sessions_failed".into()) }).await;
                                }
                            },
                            (Err(e), _) => {
                                let _ = send(&mut socket, &ServerMessage::Error { message: e.to_string(), code: Some("runtime_failed".into()) }).await;
                            }
                            (_, None) => {
                                let _ = send(&mut socket, &ServerMessage::Error { message: "invalid session id".into(), code: Some("bad_request".into()) }).await;
                            }
                        }
                    }
                }
            }
            event = event_rx.recv() => {
                let Some(event) = event else { break };
                let is_terminal = matches!(event, ServerMessage::TaskDone { .. });
                if send(&mut socket, &event).await.is_err() {
                    break;
                }
                state.host.touch();
                if is_terminal {
                    current_task = None;
                    state.end_task().await;
                }
            }
        }
    }

    // Socket closed with a run in flight: stop it, because there is no one to
    // answer its approval prompts.
    if current_task.is_some() {
        state.host.with_loaded(|rt| rt.interrupt()).await;
        state.host.with_loaded(|rt| {
            for id in rt.approval_gate().pending_ids() {
                rt.approval_gate().resolve(&id, lucy_core::ApprovalDecision::Deny);
            }
        }).await;
        state.end_task().await;
    }
    state.host.client_disconnected();
    let _ = device_id;
}

/// Best-effort current session id for the `sessions` list frame. Loads the
/// runtime only if it is already up; otherwise reports nothing rather than
/// forcing a load.
async fn current_session_hint(state: &GatewayState) -> Option<String> {
    match state.host.current() {
        Some(rt) => Some(rt.current_id().await.0.to_string()),
        None => None,
    }
}

fn parse_session_id(id: &str) -> Option<SessionId> {
    Uuid::parse_str(id.trim()).ok().map(SessionId)
}

async fn send(socket: &mut WebSocket, message: &ServerMessage) -> Result<(), ()> {
    socket
        .send(Message::text(message.to_json()))
        .await
        .map_err(|_| ())
}

/// Spawn the agent run for one task and translate its `AgentEvent`s into
/// protocol frames.
fn spawn_task(
    state: Arc<GatewayState>,
    task_id: String,
    prompt: String,
    out: mpsc::UnboundedSender<ServerMessage>,
) {
    tokio::spawn(async move {
        let finish = |summary: String, complete: bool| ServerMessage::TaskDone {
            task_id: task_id.clone(),
            summary,
            complete,
        };

        let rt = match state.host.ensure().await {
            Ok(rt) => rt,
            Err(e) => {
                let _ = out.send(ServerMessage::Error {
                    message: format!("could not start the computer runtime: {e}"),
                    code: Some("runtime_failed".into()),
                });
                let _ = out.send(finish(format!("could not start: {e}"), false));
                return;
            }
        };

        if let Err(e) = rt.save_user_message(prompt.clone()).await {
            let _ = out.send(ServerMessage::Error {
                message: format!("could not save the task: {e}"),
                code: Some("persist_failed".into()),
            });
            let _ = out.send(finish(format!("could not save: {e}"), false));
            return;
        }

        let _ = out.send(ServerMessage::Status {
            message: "Understanding your request…".into(),
        });
        let classification = match rt.classify_turn(&prompt).await {
            Ok(c) => c,
            // No routing verdict: surface it on the socket and stop — no
            // tools run on an unrouted turn.
            Err(e) => {
                let msg = format!("✖ {e}");
                let _ = rt.save_assistant_text(msg.clone()).await;
                let _ = out.send(ServerMessage::Error {
                    message: msg.clone(),
                    code: Some("routing_unavailable".into()),
                });
                let _ = out.send(finish(msg, false));
                return;
            }
        };
        let _ = out.send(ServerMessage::Thinking {
            text: classification.summary(),
        });
        let route = match rt.route_turn_with(&prompt, Some(classification)).await {
            Ok(r) => r,
            Err(e) => {
                let msg = format!("✖ {e}");
                let _ = rt.save_assistant_text(msg.clone()).await;
                let _ = out.send(ServerMessage::Error {
                    message: msg.clone(),
                    code: Some("routing_unavailable".into()),
                });
                let _ = out.send(finish(msg, false));
                return;
            }
        };
        if let Some(note) = &route.note {
            let _ = out.send(ServerMessage::Status {
                message: note.clone(),
            });
        }

        // Chat-only turn: no agent loop, just the reply.
        if !route.needs_actions() {
            let hint = route.knowledge_topic.clone();
            match rt.answer_turn_with_knowledge(&prompt, &route, hint.as_deref()).await {
                Ok(reply) => {
                    let _ = rt.save_assistant_text(reply.clone()).await;
                    let _ = out.send(ServerMessage::TextDelta { text: reply.clone() });
                    let _ = out.send(finish(reply, true));
                }
                Err(e) => {
                    // The phone shows this string verbatim, so it goes through the
                    // same classifier the TUI uses: a provider body must not be
                    // the whole error the user reads on their handset.
                    let friendly = lucy_core::friendly(&format!("{e:#}"));
                    let _ = rt.save_assistant_text(friendly.clone()).await;
                    let _ = out.send(ServerMessage::Error {
                        message: friendly.clone(),
                        code: Some("model_failed".into()),
                    });
                    let _ = out.send(finish(friendly, false));
                }
            }
            return;
        }

        // Act: stream the agent's own events through the socket.
        let (tx, mut rx) = mpsc::unbounded_channel::<AgentEvent>();
        let runner_rt = rt.clone();
        let goal = prompt.clone();
        let run_route = route.clone();
        let runner = tokio::spawn(async move {
            runner_rt
                .execute_goal_outcome(&goal, Some(&run_route), Some(tx))
                .await
        });

        let mut last_error: Option<String> = None;
        while let Some(event) = rx.recv().await {
            match event {
                AgentEvent::Status { message } => {
                    let _ = out.send(ServerMessage::Status { message });
                }
                AgentEvent::Progress { message } => {
                    let _ = out.send(ServerMessage::Progress { message });
                }
                AgentEvent::Thinking { text } => {
                    let _ = out.send(ServerMessage::Thinking { text });
                }
                AgentEvent::ToolStarted { id, name, input } => {
                    let _ = out.send(ServerMessage::ToolStarted { id, name, input });
                }
                AgentEvent::ToolFinished {
                    id,
                    name,
                    output,
                    is_error,
                } => {
                    let _ = out.send(ServerMessage::ToolFinished {
                        id,
                        name,
                        output,
                        is_error,
                    });
                }
                AgentEvent::ApprovalRequest { id, name, input } => {
                    let _ = out.send(ServerMessage::ApprovalRequest { id, name, input });
                }
                AgentEvent::TextDelta { text } => {
                    let _ = out.send(ServerMessage::TextDelta { text });
                }
                AgentEvent::Error { message } => {
                    // `last_error` keeps the raw text for the log-side summary;
                    // the socket gets the reader-facing line.
                    last_error = Some(message.clone());
                    let _ = out.send(ServerMessage::Error {
                        message: lucy_core::friendly(&message),
                        code: Some("task_error".into()),
                    });
                }
                AgentEvent::History { .. } | AgentEvent::Done => {}
            }
        }

        match runner.await {
            Ok(Ok(outcome)) => {
                let _ = rt.save_assistant_text(outcome.summary.clone()).await;
                let _ = out.send(finish(outcome.summary, outcome.complete));
            }
            Ok(Err(e)) => {
                // Both arms of this match used to build the identical string,
                // which read as if `last_error` were consulted. It is not: the
                // returned error is the authority, and the raw text (including
                // whatever `last_error` held) goes to the log while the phone
                // reads the classified line.
                tracing::warn!(
                    error = %format!("{e:#}"),
                    last_error = ?last_error,
                    "goal not completed"
                );
                let summary = format!(
                    "Goal not completed: {}",
                    lucy_core::friendly(&format!("{e:#}"))
                );
                let _ = rt.save_assistant_text(summary.clone()).await;
                let _ = out.send(finish(summary, false));
            }
            Err(e) => {
                let summary = format!("task failed: {}", lucy_core::friendly(&format!("{e:#}")));
                let _ = rt.save_assistant_text(summary.clone()).await;
                let _ = out.send(finish(summary, false));
            }
        }
    });
}

/// Spawn the idle reaper. Returns a handle that stops it when dropped.
pub fn spawn_reaper(state: Arc<GatewayState>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            state.host.reap_if_idle(now_secs()).await;
        }
    })
}

/// Approve/deny map used by tests and `--pair` bookkeeping; kept public so an
/// embedder can reason about decisions without duplicating the enum.
pub fn decision_names() -> HashMap<&'static str, ApprovalDecision> {
    [
        ("allow_once", ApprovalDecision::AllowOnce),
        ("allow_always", ApprovalDecision::AllowAlways),
        ("deny", ApprovalDecision::Deny),
    ]
    .into_iter()
    .collect()
}
