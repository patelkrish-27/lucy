//! The legacy text endpoint syncs its live `/models` into the provider list,
//! and a stale compiled-in default gets rebound to a live model.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::thread;

use lucy_config::LucyConfig;

struct StubServer {
    base: String,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl StubServer {
    fn start(status_line: &'static str, body: &'static str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind stub");
        let port = listener.local_addr().expect("local addr").port();
        let (tx, _rx) = mpsc::channel::<()>();
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
            stop,
        }
    }
}

impl Drop for StubServer {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::SeqCst);
        let _ = TcpStream::connect(self.base.trim_start_matches("http://"));
    }
}

fn serve_one(
    mut stream: TcpStream,
    status_line: &'static str,
    body: &'static str,
    _tx: mpsc::Sender<()>,
) {
    let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
    let mut request_line = String::new();
    let _ = reader.read_line(&mut request_line);
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() || line.trim().is_empty() {
            break;
        }
    }
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
/// binary so the sync never writes the user's real config.
fn isolate_config() {
    use std::sync::OnceLock;
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("lucy-endpoint-sync-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp config dir");
        let path = dir.join("config.toml");
        // SAFETY: single-threaded init, before any other thread reads it.
        unsafe { std::env::set_var("LUCY_CONFIG", &path) };
    });
}

#[tokio::test]
async fn a_sync_registers_the_endpoint_and_rebinds_a_stale_compiled_in_default() {
    isolate_config();
    let server = StubServer::start(
        "200 OK",
        r#"{"object":"list","data":[
            {"id":"gemini-flash","object":"model","available":true},
            {"id":"gemini-pro","object":"model","available":true},
            {"id":"gemini-web","object":"model","deprecated":true}
        ]}"#,
    );
    let mut cfg = LucyConfig::default();
    cfg.models.text_base_url = Some(format!("{}/v1", server.base));
    cfg.models.text_api_key = Some("k".into());
    // The compiled-in default, still persisted from before the rename.
    cfg.models.default_text = lucy_config::DEFAULT_TEXT_MODEL.into();

    let health = lucy_agent::sync_text_endpoint(&mut cfg)
        .await
        .expect("sync")
        .expect("endpoint configured");

    assert_eq!(
        health.available_models,
        vec!["gemini-flash", "gemini-pro", "gemini-web"]
    );
    assert_eq!(health.deprecated_models, vec!["gemini-web"]);
    assert_eq!(cfg.providers.len(), 1);
    assert_eq!(
        cfg.providers[0].available_models,
        vec!["gemini-flash", "gemini-pro", "gemini-web"]
    );
    // The stale compiled-in default was deprecated → rebound to a live model.
    assert_eq!(cfg.models.default_text, "gemini-flash");
    assert_eq!(cfg.default_text_model(), "gemini-flash");

    // Same response again: nothing structural changes, provider stays one
    // entry and the rebound default survives.
    let mut cfg2 = cfg.clone();
    cfg2.models.default_text = "gemini-flash".into();
    lucy_agent::sync_text_endpoint(&mut cfg2)
        .await
        .expect("second sync");
    assert_eq!(cfg2.providers.len(), 1);
    assert_eq!(cfg2.models.default_text, "gemini-flash");
}

#[tokio::test]
async fn a_sync_keeps_an_explicit_non_default_selection() {
    isolate_config();
    let server = StubServer::start(
        "200 OK",
        r#"{"data":[{"id":"alpha","object":"model"},{"id":"beta","object":"model"}]}"#,
    );
    let mut cfg = LucyConfig::default();
    cfg.models.text_base_url = Some(format!("{}/v1", server.base));
    cfg.models.text_api_key = Some("k".into());
    cfg.models.default_text = "beta".into();

    lucy_agent::sync_text_endpoint(&mut cfg)
        .await
        .expect("sync");
    assert_eq!(cfg.models.default_text, "beta");
    assert_eq!(cfg.default_text_model(), "beta");
}
