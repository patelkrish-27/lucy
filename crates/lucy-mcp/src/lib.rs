use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use lucy_core::*;
use lucy_tools::ToolRegistry;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::HashMap, process::Stdio, sync::{Arc, atomic::{AtomicU64, Ordering}}};
use tokio::{io::{AsyncBufReadExt, AsyncWriteExt, BufReader}, process::{Child, ChildStdin}, sync::Mutex};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig { pub name: String, pub command: String, #[serde(default)] pub args: Vec<String>, #[serde(default)] pub env: HashMap<String, String> }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDefinition { pub name: String, pub description: Option<String>, pub input_schema: Value }

struct Session { child: Child, stdin: ChildStdin, reader: BufReader<tokio::process::ChildStdout> }
pub struct StdioMcpClient { config: McpServerConfig, session: Mutex<Option<Session>>, next_id: AtomicU64 }
impl StdioMcpClient {
    pub fn new(config: McpServerConfig) -> Arc<Self> { Arc::new(Self { config, session: Mutex::new(None), next_id: AtomicU64::new(1) }) }
    async fn ensure_connected(&self) -> Result<()> {
        let mut guard = self.session.lock().await;
        if guard.is_some() { return Ok(()); }
        let mut cmd = tokio::process::Command::new(&self.config.command);
        cmd.args(&self.config.args).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null());
        for (k, v) in &self.config.env { cmd.env(k, v); }
        let mut child = cmd.spawn().context("failed to spawn MCP server")?;
        let stdin = child.stdin.take().ok_or_else(|| anyhow!("missing MCP stdin"))?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("missing MCP stdout"))?;
        let mut session = Session { child, stdin, reader: BufReader::new(stdout) };
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        write_request(&mut session.stdin, id, "initialize", json!({"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"lucy","version":"0.3.0"}})).await?;
        read_response(&mut session.reader, id).await?;
        session.stdin.write_all(b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n").await?;
        *guard = Some(session);
        Ok(())
    }
    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        self.ensure_connected().await?;
        let mut guard = self.session.lock().await;
        let session = guard.as_mut().ok_or_else(|| anyhow!("MCP session unavailable"))?;
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        if let Err(e) = write_request(&mut session.stdin, id, method, params).await { *guard = None; return Err(e); }
        match read_response(&mut session.reader, id).await {
            Ok(v) => Ok(v),
            Err(e) => { let _ = session.child.kill().await; *guard = None; Err(e) }
        }
    }
    pub async fn list_tools(&self) -> Result<Vec<McpToolDefinition>> {
        let result = self.request("tools/list", json!({})).await?;
        Ok(result.get("tools").and_then(Value::as_array).map(|tools| tools.iter().map(|x| McpToolDefinition { name: x["name"].as_str().unwrap_or_default().to_owned(), description: x["description"].as_str().map(str::to_owned), input_schema: x["inputSchema"].clone() }).collect()).unwrap_or_default())
    }
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<Value> { self.request("tools/call", json!({"name": name, "arguments": arguments})).await }
}

async fn write_request(stdin: &mut ChildStdin, id: u64, method: &str, params: Value) -> Result<()> {
    let req = json!({"jsonrpc":"2.0","id":id,"method":method,"params":params});
    stdin.write_all(format!("{}\n", req).as_bytes()).await?;
    stdin.flush().await?;
    Ok(())
}
async fn read_response(reader: &mut BufReader<tokio::process::ChildStdout>, id: u64) -> Result<Value> {
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line).await? == 0 { return Err(anyhow!("MCP server closed stdout")); }
        let value: Value = serde_json::from_str(line.trim()).context("invalid MCP JSON-RPC response")?;
        if value.get("id").and_then(Value::as_u64) == Some(id) {
            if let Some(error) = value.get("error") { return Err(anyhow!("MCP error: {}", error)); }
            return Ok(value.get("result").cloned().unwrap_or(Value::Null));
        }
    }
}

struct McpToolProxy { client: Arc<StdioMcpClient>, definition: McpToolDefinition, full_name: String }
#[async_trait]
impl Tool for McpToolProxy {
    fn name(&self) -> &str { &self.full_name }
    fn description(&self) -> &str { self.definition.description.as_deref().unwrap_or("MCP tool") }
    fn parameters_schema(&self) -> Value { self.definition.input_schema.clone() }
    async fn execute(&self, input: Value, _ctx: ToolContext) -> Result<Value> { self.client.call_tool(&self.definition.name, input).await }
}

pub async fn register_server(registry: &mut ToolRegistry, config: McpServerConfig) -> Result<usize> {
    let client = StdioMcpClient::new(config.clone());
    let defs = client.list_tools().await?;
    let mut count = 0;
    for definition in defs {
        let full_name = format!("mcp_{}_{}", sanitize(&config.name), sanitize(&definition.name));
        registry.register_arc(Arc::new(McpToolProxy { client: client.clone(), definition, full_name }));
        count += 1;
    }
    Ok(count)
}
fn sanitize(s: &str) -> String { s.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect() }

pub fn load_config() -> Result<Vec<McpServerConfig>> {
    let path = std::env::var("LUCY_MCP_CONFIG").map(std::path::PathBuf::from).unwrap_or_else(|_| std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into())).join(".config/lucy/mcp.toml"));
    if !path.exists() { return Ok(Vec::new()); }
    #[derive(Deserialize)] struct Config { #[serde(default)] servers: Vec<McpServerConfig> }
    Ok(toml::from_str(&std::fs::read_to_string(path)?)?.servers)
}
