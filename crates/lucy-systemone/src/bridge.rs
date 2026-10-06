use anyhow::{Context, Result, anyhow};
use serde_json::Value;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::types::{PredictRequest, PredictResponse};

/// Python embedded script that runs the Laya Router as a high-speed stdio daemon.
const LAYA_DAEMON_SCRIPT: &str = r#"
import os, sys, json, time
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

try:
    import torch
    from laya import Router
except ImportError as e:
    sys.stdout.write(json.dumps({"status": "error", "message": f"import error: {e}"}) + "\n")
    sys.stdout.flush()
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    router = Router(device=device, max_loaded=1)
except Exception as e:
    sys.stdout.write(json.dumps({"status": "error", "message": f"router init error: {e}"}) + "\n")
    sys.stdout.flush()
    sys.exit(1)

sys.stdout.write(json.dumps({"status": "ready", "device": device}) + "\n")
sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        state = req.get("state", {})
        questions = req.get("questions", {})
        model = req.get("model")
        t0 = time.perf_counter()
        kwargs = {}
        if model:
            kwargs["model"] = model
        res = router.predict(state, questions, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000.0
        res["latency_ms"] = elapsed
        sys.stdout.write(json.dumps(res) + "\n")
    except Exception as e:
        sys.stdout.write(json.dumps({"error": str(e), "answers": {}}) + "\n")
    sys.stdout.flush()
"#;

struct DaemonProcess {
    child: Child,
    stdin: ChildStdin,
    reader: tokio::io::Lines<BufReader<ChildStdout>>,
}

/// Direct local driver running `laya` via a persistent Python daemon process.
#[derive(Clone)]
pub struct LayaDaemonBridge {
    python_path: PathBuf,
    process: Arc<Mutex<Option<DaemonProcess>>>,
}

impl LayaDaemonBridge {
    pub fn new(python_path: impl Into<PathBuf>) -> Self {
        Self {
            python_path: python_path.into(),
            process: Arc::new(Mutex::new(None)),
        }
    }

    /// Spawns the python daemon if not already running.
    async fn ensure_started(&self, lock: &mut Option<DaemonProcess>) -> Result<()> {
        if lock.is_some() {
            return Ok(());
        }

        info!(python = %self.python_path.display(), "Spawning Laya System-1 daemon bridge");
        let mut cmd = Command::new(&self.python_path);
        cmd.arg("-u")
            .arg("-c")
            .arg(LAYA_DAEMON_SCRIPT)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());

        let mut child = cmd.spawn().with_context(|| {
            format!(
                "failed to spawn python daemon at {}",
                self.python_path.display()
            )
        })?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("failed to get child stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("failed to get child stdout"))?;
        let mut reader = BufReader::new(stdout).lines();

        // Read initial handshake line with timeout
        let handshake_future = reader.next_line();
        let ready_res = tokio::time::timeout(Duration::from_secs(45), handshake_future)
            .await
            .map_err(|_| anyhow!("timeout waiting for Laya daemon startup"))??;

        let line =
            ready_res.ok_or_else(|| anyhow!("Laya daemon exited unexpectedly on startup"))?;
        let init_status: Value = serde_json::from_str(&line)
            .with_context(|| format!("invalid JSON handshake from Laya daemon: {line}"))?;

        if init_status.get("status").and_then(|v| v.as_str()) != Some("ready") {
            let msg = init_status
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error");
            return Err(anyhow!("Laya daemon failed to initialize: {msg}"));
        }

        let dev = init_status
            .get("device")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        info!(device = %dev, "Laya daemon ready");
        *lock = Some(DaemonProcess {
            child,
            stdin,
            reader,
        });
        Ok(())
    }

    /// Predict decision through direct local daemon.
    pub async fn predict(&self, req: &PredictRequest) -> Result<PredictResponse> {
        let mut lock = self.process.lock().await;
        self.ensure_started(&mut lock).await?;

        let daemon = lock.as_mut().unwrap();
        let payload = serde_json::to_string(req)? + "\n";

        if let Err(e) = daemon.stdin.write_all(payload.as_bytes()).await {
            warn!(error = %e, "Writing to Laya daemon failed; restarting daemon");
            *lock = None;
            self.ensure_started(&mut lock).await?;
            let daemon2 = lock.as_mut().unwrap();
            daemon2.stdin.write_all(payload.as_bytes()).await?;
            daemon2.stdin.flush().await?;
        } else {
            daemon.stdin.flush().await?;
        }

        let daemon = lock.as_mut().unwrap();
        let line_opt =
            match tokio::time::timeout(Duration::from_secs(60), daemon.reader.next_line()).await {
                Ok(Ok(opt)) => opt,
                Ok(Err(e)) => {
                    *lock = None;
                    return Err(anyhow!("Failed to read response from Laya daemon: {e}"));
                }
                Err(_) => {
                    warn!("Laya daemon inference timed out after 60s; restarting");
                    let _ = daemon.child.kill().await;
                    *lock = None;
                    return Err(anyhow!("Laya daemon inference timed out"));
                }
            };

        let line = line_opt.ok_or_else(|| anyhow!("Laya daemon closed stdout"))?;
        let resp_val: Value = serde_json::from_str(&line)
            .with_context(|| format!("invalid JSON response from Laya daemon: {line}"))?;

        if let Some(err) = resp_val.get("error").and_then(Value::as_str) {
            return Err(anyhow!("Laya inference error: {err}"));
        }

        let parsed: PredictResponse = serde_json::from_value(resp_val)?;
        Ok(parsed)
    }

    /// Stop daemon process cleanly.
    pub async fn shutdown(&self) {
        let mut lock = self.process.lock().await;
        if let Some(mut d) = lock.take() {
            let _ = d.child.kill().await;
        }
    }
}
