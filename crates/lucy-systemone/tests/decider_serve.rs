//! End-to-end checks for the `decider-serve` client against a stub server:
//! the `/health` probe and the single-forward-pass `/predict` turn routing,
//! including the alternate-path fallback and the unreachable-server path.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// One request the stub saw.
#[derive(Debug, Clone)]
struct Captured {
    method: String,
    path: String,
    content_type: Option<String>,
    body: String,
}

/// A routing stub: answers `health_json` on `health_path` and `predict_json`
/// on `predict_path`, and 404s everything else (so the client's path sweep is
/// genuinely exercised).
struct Stub {
    base: String,
    requests: mpsc::Receiver<Captured>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl Stub {
    fn start(
        health_path: &'static str,
        health_json: &'static str,
        predict_path: &'static str,
        predict_json: &'static str,
    ) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind stub");
        let port = listener.local_addr().expect("local addr").port();
        let (tx, rx) = mpsc::channel();
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_thread = stop.clone();
        thread::spawn(move || {
            for stream in listener.incoming() {
                if stop_thread.load(std::sync::atomic::Ordering::SeqCst) {
                    return;
                }
                let Ok(stream) = stream else { continue };
                let tx = tx.clone();
                thread::spawn(move || {
                    serve(
                        stream,
                        health_path,
                        health_json,
                        predict_path,
                        predict_json,
                        tx,
                    )
                });
            }
        });
        Self {
            base: format!("http://127.0.0.1:{port}"),
            requests: rx,
            stop,
        }
    }

    fn url(&self) -> String {
        self.base.clone()
    }

    /// Next captured request, or `None` when none arrives within `timeout`.
    fn next_request(&self, timeout: Duration) -> Option<Captured> {
        self.requests.recv_timeout(timeout).ok()
    }
}

impl Drop for Stub {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::SeqCst);
        let _ = TcpStream::connect(self.base.trim_start_matches("http://"));
    }
}

fn serve(
    mut stream: TcpStream,
    health_path: &'static str,
    health_json: &'static str,
    predict_path: &'static str,
    predict_json: &'static str,
    tx: mpsc::Sender<Captured>,
) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() || request_line.is_empty() {
        return;
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_owned();
    let path = parts.next().unwrap_or_default().to_owned();
    let mut content_type = None;
    let mut length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() || line.trim().is_empty() {
            break;
        }
        let lower = line.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            length = v.trim().parse().unwrap_or(0);
        }
        if lower.starts_with("content-type:") {
            content_type = Some(line[line.find(':').expect("colon") + 1..].trim().to_owned());
        }
    }
    let mut body = vec![0u8; length];
    if length > 0 {
        use std::io::Read;
        let _ = reader.read_exact(&mut body);
    }
    let body = String::from_utf8_lossy(&body).into_owned();
    let _ = tx.send(Captured {
        method,
        path: path.clone(),
        content_type,
        body: body.clone(),
    });

    let (status, payload) = if path == health_path {
        ("200 OK", health_json)
    } else if path == predict_path {
        ("200 OK", predict_json)
    } else {
        ("404 Not Found", r#"{"error":"not found"}"#)
    };
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
        payload.len()
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

fn client(base: &str) -> lucy_systemone::DeciderClient {
    lucy_systemone::DeciderClient::new(lucy_config::ClassificationConfig {
        enabled: true,
        classification_api_url: base.into(),
        timeout_ms: 5_000,
        model: String::new(),
        confidence_threshold: 0.1,
    })
    .expect("client")
}

const HEALTH: &str = r#"{"status":"ok","model":"Mapika/decider-2b-vision","device":"cuda:0","dtype":"bfloat16","vram":{"total_gb":24.0,"used_gb":7.25},"loaded":true}"#;

#[tokio::test]
async fn health_reads_the_documented_payload() {
    let stub = Stub::start("/health", HEALTH, "/predict", r#"{"answers":{}}"#);
    let health = client(&stub.url()).health().await.expect("health ok");
    assert!(health.is_ok());
    assert_eq!(health.model, "Mapika/decider-2b-vision");
    assert_eq!(health.device, "cuda:0");
    assert_eq!(health.dtype, "bfloat16");
    assert!(health.loaded);
    let summary = health.summary();
    assert!(summary.contains("used_gb 7.25"), "{summary}");

    let seen = stub.next_request(Duration::from_secs(5)).expect("captured");
    assert_eq!(seen.method, "GET");
    assert_eq!(seen.path, "/health");
}

#[tokio::test]
async fn probe_reports_model_device_and_vram() {
    let stub = Stub::start(
        "/health",
        HEALTH,
        "/predict",
        r#"{"answers":{"requires_only_response":{"choice":"yes","confidence":0.97}}}"#,
    );
    let probe = client(&stub.url()).probe().await.expect("probe ok");
    assert!(probe.predict_ok, "{}", probe.predict_note);
    let summary = probe.summary();
    assert!(summary.contains("Mapika/decider-2b-vision"), "{summary}");
    assert!(summary.contains("device cuda:0"), "{summary}");
    assert!(summary.contains("dtype bfloat16"), "{summary}");
    assert!(summary.contains("loaded"), "{summary}");
}

#[tokio::test]
async fn classify_turn_sends_the_two_question_array_in_one_forward_pass() {
    let stub = Stub::start(
        "/health",
        HEALTH,
        "/predict",
        r#"{"answers":{
             "requires_only_response":{"choice":"no","confidence":0.93,"probabilities":{"yes":0.07,"no":0.93}},
             "reasoning_level":{"choice":"3","confidence":0.88}},
           "latency_ms":31.4}"#,
    );
    let c = client(&stub.url())
        .classify_turn("open yt & play this song")
        .await
        .expect("classify ok");

    // Branch B + the deepest tier.
    assert_eq!(c.branch, lucy_systemone::TurnBranch::RequiresActions);
    assert!(c.branch.needs_actions());
    assert_eq!(c.reasoning_level, lucy_config::ReasoningLevel::L3);
    assert!((c.confidence - 0.93).abs() < 1e-6);
    assert!((c.probabilities["yes"] - 0.07).abs() < 1e-6);
    assert!((c.level_confidence - 0.88).abs() < 1e-6);

    // Exactly one request, and it is the documented array schema.
    let seen = stub.next_request(Duration::from_secs(5)).expect("captured");
    assert_eq!(seen.method, "POST");
    assert_eq!(seen.path, "/predict");
    assert_eq!(
        seen.content_type.as_deref(),
        Some("application/json"),
        "the server requires application/json"
    );
    let body: serde_json::Value = serde_json::from_str(&seen.body).expect("json body");
    assert_eq!(body["context"], "open yt & play this song");
    let questions = body["questions"].as_array().expect("array of questions");
    assert_eq!(questions.len(), 2, "both heads in one forward pass");
    assert_eq!(questions[0]["id"], "q1");
    assert_eq!(questions[0]["question"], "what kind of request is this");
    assert_eq!(
        questions[0]["options"],
        serde_json::json!([
            "chat = the user wants a conversational reply, information, or explanation",
            "act = the user wants an action performed (open app, play media, control device, etc.)"
        ])
    );
    assert_eq!(questions[1]["id"], "q2");
    assert_eq!(
        questions[1]["question"],
        "what reasoning complexity level is required for this command"
    );
    // No vision context was supplied, so no image keys are sent.
    assert!(body.get("image").is_none());
    assert!(body.get("image_url").is_none());
}

#[tokio::test]
async fn a_text_only_command_routes_to_the_fast_tier() {
    let stub = Stub::start(
        "/health",
        HEALTH,
        "/predict",
        r#"{"answers":{
             "requires_only_response":{"choice":"yes","confidence":0.99},
             "reasoning_level":{"choice":"1","confidence":0.7}}}"#,
    );
    let c = client(&stub.url())
        .classify_turn("what is a monad?")
        .await
        .expect("classify ok");
    assert_eq!(c.branch, lucy_systemone::TurnBranch::RequiresOnlyResponse);
    assert!(!c.branch.needs_actions());
    assert_eq!(c.reasoning_level, lucy_config::ReasoningLevel::L1);
}

#[tokio::test]
async fn an_older_server_on_decide_is_found_by_the_path_sweep() {
    // Only `/decide` answers: `/predict` 404s, so the client must fall through
    // and pin the working path for later turns.
    let stub = Stub::start(
        "/health",
        HEALTH,
        "/decide",
        r#"{"answers":{"requires_only_response":"no","reasoning_level":3}}"#,
    );
    // The same client is reused, exactly as the runtime caches it.
    let cached = client(&stub.url());
    let c = cached
        .classify_turn("open the browser")
        .await
        .expect("classify ok via the fallback path");
    assert_eq!(c.branch, lucy_systemone::TurnBranch::RequiresActions);
    assert_eq!(c.reasoning_level, lucy_config::ReasoningLevel::L3);

    // The first request hit `/predict` and 404'd…
    let first = stub.next_request(Duration::from_secs(5)).expect("captured");
    assert_eq!(first.path, "/predict");
    // …the second hit `/decide`.
    let second = stub.next_request(Duration::from_secs(5)).expect("captured");
    assert_eq!(second.path, "/decide");

    // The working path is pinned on the client: a follow-up turn costs one
    // request, not another 404 sweep.
    cached
        .classify_turn("again")
        .await
        .expect("second classify ok");
    let third = stub.next_request(Duration::from_secs(5)).expect("captured");
    assert_eq!(third.path, "/decide", "the path that worked is reused");
}

#[tokio::test]
async fn a_dead_classifier_produces_a_heuristic_verdict_not_an_error() {
    // Nothing listens on the discard port.
    let c = lucy_systemone::heuristic_turn_classification("open yt & play this song");
    let client = client("http://127.0.0.1:9");
    // The client itself surfaces the error; the runtime turns it into the
    // heuristic verdict plus a warning (covered in lucy-runtime's unit tests).
    let err = client.classify_turn("hello").await.expect_err("dead port");
    assert!(
        format!("{err:#}").contains("127.0.0.1:9"),
        "the URL must be named in the error: {err:#}"
    );
    assert_eq!(c.reasoning_level, lucy_config::ReasoningLevel::L3);
}

#[tokio::test]
async fn a_server_that_answers_without_answers_is_an_error() {
    let stub = Stub::start("/health", HEALTH, "/predict", r#"{"latency_ms":4}"#);
    let err = client(&stub.url())
        .classify_turn("hello")
        .await
        .expect_err("no answers object must fail");
    assert!(format!("{err:#}").contains("answers"), "{err:#}");
}

#[tokio::test]
async fn an_unloaded_checkpoint_is_reported_but_not_a_hard_failure() {
    let stub = Stub::start(
        "/health",
        r#"{"status":"ok","model":"Mapika/decider-2b-vision","device":"cuda:0","dtype":"bfloat16","vram":{"total_gb":24.0,"used_gb":0.0},"loaded":false}"#,
        "/predict",
        r#"{"answers":{"requires_only_response":"yes","reasoning_level":2}}"#,
    );
    let probe = client(&stub.url())
        .probe()
        .await
        .expect("probe still works");
    assert!(!probe.health.is_ok(), "loaded=false must not report ok");
    assert!(probe.predict_ok);
    assert!(
        probe.summary().contains("not loaded"),
        "{}",
        probe.summary()
    );
}
