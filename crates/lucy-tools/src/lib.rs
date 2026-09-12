mod risk;

pub use risk::{assess, gate, GateDecision, RiskAssessment, RiskContext, RiskFinding, RiskLevel};

use std::{collections::HashMap, sync::Arc, time::Duration};
use anyhow::{anyhow, Result};
use lucy_core::*;
use serde_json::Value;
use tokio::io::AsyncReadExt;

const DEFAULT_OUTPUT_LIMIT: usize = 64 * 1024;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

pub struct ToolRegistry { tools: HashMap<String, Arc<dyn Tool>> }

impl ToolRegistry {
    pub fn new() -> Self { Self { tools: HashMap::new() } }
    pub fn register<T: Tool + 'static>(&mut self, tool: T) { self.tools.insert(tool.name().to_string(), Arc::new(tool)); }
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> { self.tools.get(name).cloned() }
    pub fn definitions(&self) -> Vec<Value> {
        self.tools.values().map(|t| serde_json::json!({
            "name": t.name(),
            "description": t.description(),
            "input_schema": t.parameters_schema(),
        })).collect()
    }
    pub async fn execute(&self, name: &str, input: Value, ctx: ToolContext) -> Result<Value> {
        if ctx.interrupt.is_set() { return Err(LucyError::Cancelled.into()); }
        let tool = self.get(name).ok_or_else(|| anyhow!(LucyError::ToolNotFound(name.to_string())))?;
        tool.execute(input, ctx).await
    }
}
impl Default for ToolRegistry { fn default() -> Self { Self::new() } }

pub struct ShellTool {
    pub output_limit: usize,
    pub timeout: Duration,
}
impl Default for ShellTool { fn default() -> Self { Self { output_limit: DEFAULT_OUTPUT_LIMIT, timeout: DEFAULT_TIMEOUT } } }

#[async_trait::async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &str { "shell" }
    fn description(&self) -> &str { "Run a shell command in Lucy's working directory." }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "justification": {"type": "string", "description": "Required when Lucy asks you to justify a risky command"},
                "intent": {"type": "string", "description": "Why this tool call is needed"}
            },
            "required": ["command", "intent"]
        })
    }
    async fn execute(&self, input: Value, ctx: ToolContext) -> Result<Value> {
        if ctx.interrupt.is_set() { return Err(LucyError::Cancelled.into()); }
        let command = input.get("command").and_then(Value::as_str)
            .ok_or_else(|| anyhow!(LucyError::InvalidInput("command is required".into())))?;
        let justification = input.get("justification").and_then(Value::as_str);
        let working_dir = ctx.working_dir.clone().unwrap_or(std::env::current_dir()?);
        let assessment = assess(command, &RiskContext::from_env(Some(working_dir.clone())));
        match gate(&assessment, justification) {
            GateDecision::Allow => {}
            GateDecision::Reflect(message) => {
                return Err(anyhow!(LucyError::InvalidInput(format!("RISK_REFLECTION_REQUIRED\n{message}\nProvide a substantive `justification` and retry."))));
            }
            GateDecision::Deny(message) => {
                return Err(anyhow!(LucyError::InvalidInput(format!("RISK_DENIED\n{message}"))));
            }
        }

        let mut child = tokio::process::Command::new("sh")
            .arg("-lc").arg(command)
            .current_dir(&working_dir)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()?;

        let stdout = child.stdout.take().ok_or_else(|| anyhow!("failed to capture stdout"))?;
        let stderr = child.stderr.take().ok_or_else(|| anyhow!("failed to capture stderr"))?;
        let limit = self.output_limit;
        let out_task = tokio::spawn(async move { read_limited(stdout, limit).await });
        let err_task = tokio::spawn(async move { read_limited(stderr, limit).await });

        tokio::select! {
            status = child.wait() => {
                let status = status?;
                let stdout = out_task.await??;
                let stderr = err_task.await??;
                Ok(serde_json::json!({
                    "status": status.code(),
                    "success": status.success(),
                    "stdout": stdout,
                    "stderr": stderr,
                    "truncated": false
                }))
            }
            _ = ctx.interrupt.notified() => {
                let _ = child.kill().await;
                Err(LucyError::Cancelled.into())
            }
            _ = tokio::time::sleep(self.timeout) => {
                let _ = child.kill().await;
                Err(anyhow!("shell tool timed out after {} seconds", self.timeout.as_secs()))
            }
        }
    }
}

async fn read_limited<R: tokio::io::AsyncRead + Unpin>(mut reader: R, limit: usize) -> Result<String> {
    let mut buf = Vec::with_capacity(limit.min(8192));
    let mut chunk = [0u8; 8192];
    loop {
        let n = reader.read(&mut chunk).await?;
        if n == 0 { break; }
        let remaining = limit.saturating_sub(buf.len());
        buf.extend_from_slice(&chunk[..n.min(remaining)]);
        if buf.len() >= limit { break; }
    }
    Ok(String::from_utf8_lossy(&buf).into_owned())
}
