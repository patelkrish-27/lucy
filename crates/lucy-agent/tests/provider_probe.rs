//! End-to-end checks for the connected-provider flow against a stub
//! OpenAI-compatible server: `Test Connection` (`GET {api_url}/models`) and
//! `Save` (persist provider + discovered models).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::thread;

/// A request the stub server saw.
#[derive(Debug, Clone)]
struct Captured {
    method: String,
    path: String,
    auth: Option<String>,
}

/// One-shot stub OpenAI-compatible server. Returns its base URL, a receiver of
/// captured requests, and a shutdown handle.
struct StubServer {
    base: String,
    requests: mpsc::Receiver<Captured>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl StubServer {
    /// `body` is returned for every request; `status_line` is the HTTP status.
    fn start(status_line: &'static str, body: &'static str) -> Self {
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
                    serve_one(stream, status_line, body, tx);
                });
            }
        });
        Self {
            base: format!("http://127.0.0.1:{port}"),
            requests: rx,
            stop,
        }
    }
}

impl Drop for StubServer {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::SeqCst);
        // Unblock the accept loop with one throwaway connection.
        let _ = TcpStream::connect(self.base.trim_start_matches("http://"));
    }
}

fn serve_one(
    mut stream: TcpStream,
    status_line: &'static str,
    body: &'static str,
    tx: mpsc::Sender<Captured>,
) {
    let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() || request_line.is_empty() {
        return;
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_owned();
    let path = parts.next().unwrap_or_default().to_owned();
    let mut auth = None;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() || line.trim().is_empty() {
            break;
        }
        let lower = line.to_ascii_lowercase();
        if lower.starts_with("authorization:") {
            auth = Some(line[line.find(':').expect("colon") + 1..].trim().to_owned());
        }
    }
    let _ = tx.send(Captured { method, path, auth });
    let response = format!(
        "HTTP/1.1 {status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
    let mut sink = Vec::new();
    let _ = stream.take(0).read_to_end(&mut sink);
}

/// Redirect `LucyConfig` persistence at a throwaway file for the whole test
/// binary, so these tests can never write to the user's real
/// `~/.config/lucy/config.toml`.
fn isolate_config() {
    use std::sync::OnceLock;
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("lucy-provider-probe-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp config dir");
        let path = dir.join("config.toml");
        // SAFETY: single-threaded init, before any other thread reads it.
        unsafe { std::env::set_var("LUCY_CONFIG", &path) };
    });
}

fn provider(url: &str, key: &str) -> lucy_config::ProviderConfig {
    lucy_config::ProviderConfig {
        id: "groq".into(),
        name: "Groq".into(),
        api_url: url.into(),
        api_key: key.into(),
        provider_type: lucy_config::ProviderType::Text,
        available_models: Vec::new(),
        deprecated_models: Vec::new(),
    }
}

#[tokio::test]
async fn test_connection_hits_models_and_discovers_the_model_list() {
    let server = StubServer::start(
        "200 OK",
        r#"{"object":"list","data":[{"id":"llama-3.3-70b"},{"id":"flash"},{"id":"flash"}]}"#,
    );
    let health = lucy_agent::test_connection(&provider(&server.base, "sk-test"))
        .await
        .expect("probe must succeed");

    // Deduped + sorted.
    assert_eq!(health.available_models, vec!["flash", "llama-3.3-70b"]);
    // `/v1` is appended when the base URL has no version segment.
    assert_eq!(health.models_endpoint, format!("{}/v1/models", server.base));
    assert_eq!(health.provider_type, lucy_config::ProviderType::Text);

    let seen = server.requests.recv().expect("request captured");
    assert_eq!(seen.method, "GET");
    assert_eq!(seen.path, "/v1/models");
    assert_eq!(seen.auth.as_deref(), Some("Bearer sk-test"));
}

#[tokio::test]
async fn test_connection_respects_an_existing_v1_suffix() {
    let server = StubServer::start("200 OK", r#"{"data":[{"id":"m"}]}"#);
    let mut p = provider(&format!("{}/v1", server.base), "sk-test");
    let health = lucy_agent::test_connection(&p).await.expect("probe ok");
    assert_eq!(health.models_endpoint, format!("{}/v1/models", server.base));
    p.api_url = format!("{}/", server.base);
    let trimmed = lucy_agent::test_connection(&p).await.expect("probe ok");
    assert_eq!(
        trimmed.api_url, server.base,
        "a trailing slash is stripped before storing"
    );
}

#[tokio::test]
async fn unauthorized_is_reported_with_an_actionable_hint() {
    let server = StubServer::start("401 Unauthorized", r#"{"error":"invalid api key"}"#);
    let err = lucy_agent::test_connection(&provider(&server.base, "sk-bad"))
        .await
        .expect_err("401 must fail");
    let msg = lucy_agent::friendly_probe_error(&err);
    assert!(msg.contains("check the API Key"), "{msg}");
}

#[tokio::test]
async fn an_empty_model_list_is_a_failure() {
    // A 200 that advertises nothing cannot populate the dropdowns, so it must
    // not be reported as a working connection.
    let server = StubServer::start("200 OK", r#"{"data":[]}"#);
    let err = lucy_agent::test_connection(&provider(&server.base, "sk-test"))
        .await
        .expect_err("empty model list must fail");
    assert!(err.to_string().contains("listed no models"), "{err:#}");
}

#[tokio::test]
async fn save_provider_probes_then_persists_with_its_models() {
    isolate_config();
    let server = StubServer::start("200 OK", r#"{"data":[{"id":"pro"},{"id":"flash"}]}"#);
    let mut cfg = lucy_config::LucyConfig::default();
    let stored = lucy_agent::save_provider(&mut cfg, &provider(&server.base, "sk-test"))
        .await
        .expect("save must probe first");
    assert_eq!(stored.available_models, vec!["flash", "pro"]);
    assert_eq!(cfg.providers.len(), 1);
    assert_eq!(cfg.providers[0].available_models, vec!["flash", "pro"]);
    assert_eq!(cfg.providers[0].api_key, "sk-test");
    // The dropdowns are now populated.
    let options = cfg.text_model_options();
    assert_eq!(options.len(), 2);
    assert_eq!(options[0].key(), "groq/flash");
}

#[tokio::test]
async fn voice_providers_only_populate_the_voice_dropdowns() {
    isolate_config();
    let server = StubServer::start("200 OK", r#"{"data":[{"id":"tts-v2"}]}"#);
    let mut cfg = lucy_config::LucyConfig::default();
    let mut p = provider(&server.base, "sk-test");
    p.id = String::new();
    p.provider_type = lucy_config::ProviderType::Voice;
    lucy_agent::save_provider(&mut cfg, &p)
        .await
        .expect("save ok");
    assert!(cfg.text_model_options().is_empty());
    assert_eq!(cfg.voice_model_options().len(), 1);
    assert_eq!(cfg.voice_model_options()[0].key(), "groq/tts-v2");
}

#[tokio::test]
async fn saving_never_touches_the_users_real_config() {
    isolate_config();
    let server = StubServer::start("200 OK", r#"{"data":[{"id":"m"}]}"#);
    let mut cfg = lucy_config::LucyConfig::default();
    lucy_agent::save_provider(&mut cfg, &provider(&server.base, "sk-test"))
        .await
        .expect("save ok");
    let path = lucy_config::config_path().expect("config path");
    assert!(
        path.to_string_lossy().contains("lucy-provider-probe-"),
        "tests must not persist to {}",
        path.display()
    );
}

#[tokio::test]
async fn an_unreachable_endpoint_names_the_url_in_the_error() {
    // Nothing listens on the discard port.
    let err = lucy_agent::test_connection(&provider("http://127.0.0.1:9", "sk-test"))
        .await
        .expect_err("unreachable must fail");
    let msg = format!("{err:#}");
    assert!(msg.contains("127.0.0.1:9"), "{msg}");
    assert!(
        lucy_agent::friendly_probe_error(&err).contains("reachable"),
        "{msg}"
    );
}
