use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use lucy_core::*;
use lucy_tools::ToolRegistry;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin},
    sync::Mutex,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Value,
}

/// Maximum time to wait for a single MCP response line. Without this a hung
/// server (e.g. `npx` fetching a package) blocks startup forever.
/// LLM-driven browser tools (`hint_act`) legitimately take 40-90s per
/// call (measured 40s for one act), so this must comfortably exceed that.
const MCP_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);
struct Session {
    child: Child,
    stdin: ChildStdin,
    reader: BufReader<tokio::process::ChildStdout>,
}
pub struct StdioMcpClient {
    config: McpServerConfig,
    session: Mutex<Option<Session>>,
    next_id: AtomicU64,
}
impl StdioMcpClient {
    pub fn new(config: McpServerConfig) -> Arc<Self> {
        Arc::new(Self {
            config,
            session: Mutex::new(None),
            next_id: AtomicU64::new(1),
        })
    }
    async fn ensure_connected(&self) -> Result<()> {
        let mut guard = self.session.lock().await;
        if guard.is_some() {
            return Ok(());
        }
        let mut cmd = tokio::process::Command::new(&self.config.command);
        cmd.args(&self.config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        // Never leave orphaned MCP children (e.g. `node`) behind when the
        // client is dropped after a one-shot list_tools() at startup.
        cmd.kill_on_drop(true);
        for (k, v) in &self.config.env {
            cmd.env(k, v);
        }
        let mut child = cmd.spawn().context("failed to spawn MCP server")?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("missing MCP stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("missing MCP stdout"))?;
        let mut session = Session {
            child,
            stdin,
            reader: BufReader::new(stdout),
        };
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        write_request(&mut session.stdin, id, "initialize", json!({"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"lucy","version":"0.3.0"}})).await?;
        read_response(&mut session.reader, id).await?;
        session
            .stdin
            .write_all(b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n")
            .await?;
        *guard = Some(session);
        Ok(())
    }
    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        self.ensure_connected().await?;
        let mut guard = self.session.lock().await;
        let session = guard
            .as_mut()
            .ok_or_else(|| anyhow!("MCP session unavailable"))?;
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        if let Err(e) = write_request(&mut session.stdin, id, method, params).await {
            *guard = None;
            return Err(e);
        }
        match read_response(&mut session.reader, id).await {
            Ok(v) => Ok(v),
            Err(e) => {
                let _ = session.child.kill().await;
                *guard = None;
                Err(e)
            }
        }
    }
    pub async fn list_tools(&self) -> Result<Vec<McpToolDefinition>> {
        let result = self.request("tools/list", json!({})).await?;
        Ok(result
            .get("tools")
            .and_then(Value::as_array)
            .map(|tools| {
                tools
                    .iter()
                    .map(|x| McpToolDefinition {
                        name: x["name"].as_str().unwrap_or_default().to_owned(),
                        description: x["description"].as_str().map(str::to_owned),
                        input_schema: x["inputSchema"].clone(),
                    })
                    .collect()
            })
            .unwrap_or_default())
    }
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<Value> {
        self.request("tools/call", json!({"name": name, "arguments": arguments}))
            .await
    }
}

async fn write_request(stdin: &mut ChildStdin, id: u64, method: &str, params: Value) -> Result<()> {
    let req = json!({"jsonrpc":"2.0","id":id,"method":method,"params":params});
    stdin.write_all(format!("{}\n", req).as_bytes()).await?;
    stdin.flush().await?;
    Ok(())
}
async fn read_response(
    reader: &mut BufReader<tokio::process::ChildStdout>,
    id: u64,
) -> Result<Value> {
    let mut line = String::new();
    loop {
        line.clear();
        let n = tokio::time::timeout(MCP_READ_TIMEOUT, reader.read_line(&mut line))
            .await
            .context("MCP server timed out")??;
        if n == 0 {
            return Err(anyhow!("MCP server closed stdout"));
        }
        let value: Value =
            serde_json::from_str(line.trim()).context("invalid MCP JSON-RPC response")?;
        if value.get("id").and_then(Value::as_u64) == Some(id) {
            if let Some(error) = value.get("error") {
                return Err(anyhow!("MCP error: {}", error));
            }
            return Ok(value.get("result").cloned().unwrap_or(Value::Null));
        }
    }
}

struct McpToolProxy {
    client: Arc<StdioMcpClient>,
    definition: McpToolDefinition,
    full_name: String,
}
#[async_trait]
impl Tool for McpToolProxy {
    fn name(&self) -> &str {
        &self.full_name
    }
    fn description(&self) -> &str {
        self.definition.description.as_deref().unwrap_or("MCP tool")
    }
    fn parameters_schema(&self) -> Value {
        self.definition.input_schema.clone()
    }
    async fn execute(&self, input: Value, _ctx: ToolContext) -> Result<Value> {
        self.client.call_tool(&self.definition.name, input).await
    }
}

pub async fn register_server(
    registry: &mut ToolRegistry,
    config: McpServerConfig,
) -> Result<usize> {
    let client = StdioMcpClient::new(config.clone());
    let defs = client.list_tools().await?;
    Ok(register_server_with_defs(registry, config, defs))
}

/// Register pre-fetched tool definitions WITHOUT spawning the server.
///
/// The proxy client connects lazily on the first actual tool call
/// (`ensure_connected`), so startup pays zero process-spawn cost. Use this
/// with definitions obtained from discovery or the on-disk defs cache.
pub fn register_server_with_defs(
    registry: &mut ToolRegistry,
    config: McpServerConfig,
    defs: Vec<McpToolDefinition>,
) -> usize {
    let client = StdioMcpClient::new(config.clone());
    let mut count = 0;
    for definition in defs {
        let base_name = format!(
            "mcp_{}_{}",
            sanitize(&config.name),
            sanitize(&definition.name)
        );
        // Never let an MCP server overwrite an existing tool. Sanitization can
        // collapse distinct names (for example "foo-bar" and "foo_bar"), and a
        // malicious/buggy server could otherwise shadow a built-in or another
        // server's tool.
        let mut full_name = base_name.clone();
        let mut suffix = 2usize;
        while registry.get(&full_name).is_some() {
            full_name = format!("{base_name}_{suffix}");
            suffix = suffix.saturating_add(1);
        }
        let bare_name = definition.name.clone();
        registry.register_arc(Arc::new(McpToolProxy {
            client: client.clone(),
            definition: definition.clone(),
            full_name: full_name.clone(),
        }));
        if registry.get(&bare_name).is_none() {
            registry.register_arc(Arc::new(McpToolProxy {
                client: client.clone(),
                definition,
                full_name: bare_name,
            }));
        }
        count += 1;
    }
    count
}
fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

fn split_command_args(input: &str) -> Option<Vec<String>> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut quote = None;
    let mut escaped = false;
    for ch in input.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }
        match ch {
            '\\' => escaped = true,
            '"' | '\'' if quote == Some(ch) => quote = None,
            '"' | '\'' if quote.is_none() => quote = Some(ch),
            c if c.is_whitespace() && quote.is_none() => {
                if !current.is_empty() {
                    args.push(std::mem::take(&mut current));
                }
            }
            c => current.push(c),
        }
    }
    if escaped || quote.is_some() {
        return None;
    }
    if !current.is_empty() {
        args.push(current);
    }
    Some(args)
}

pub fn computer_use_config() -> McpServerConfig {
    McpServerConfig {
        name: "computer_use".into(),
        command: std::env::var("LUCY_COMPUTER_USE_COMMAND").unwrap_or_else(|_| "npx".into()),
        args: std::env::var("LUCY_COMPUTER_USE_ARGS")
            .ok()
            .and_then(|v| split_command_args(&v))
            .unwrap_or_else(|| vec!["-y".into(), "@zavora-ai/computer-use-mcp".into()]),
        env: HashMap::new(),
    }
}

pub fn load_config() -> Result<Vec<McpServerConfig>> {
    let path = std::env::var("LUCY_MCP_CONFIG")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
                .join(".config/lucy/mcp.toml")
        });
    let mut servers = if path.exists() {
        #[derive(Deserialize)]
        struct Config {
            #[serde(default)]
            servers: Vec<McpServerConfig>,
        }
        toml::from_str::<Config>(&std::fs::read_to_string(path)?)?.servers
    } else {
        Vec::new()
    };
    let enabled = std::env::var("LUCY_COMPUTER_USE_ENABLED")
        .map(|v| v != "0" && v.to_ascii_lowercase() != "false")
        .unwrap_or(true);
    if enabled
        && !servers
            .iter()
            .any(|s| s.name.eq_ignore_ascii_case("computer_use"))
    {
        servers.push(computer_use_config());
    }
    Ok(servers)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn def(name: &str) -> McpToolDefinition {
        McpToolDefinition {
            name: name.into(),
            description: Some("test tool".into()),
            input_schema: serde_json::json!({"type": "object"}),
        }
    }
    #[test]
    fn command_args_preserve_quoted_values() {
        assert_eq!(
            split_command_args(r#"node "file with spaces.js" --name "Lucy Agent""#).unwrap(),
            vec![
                "node",
                "file with spaces.js",
                "--name",
                "Lucy Agent"
            ]
        );
        assert!(split_command_args(r#"node "unterminated"#).is_none());
    }

    #[test]
    fn mcp_name_collisions_never_overwrite_tools() {
        let mut registry = lucy_tools::ToolRegistry::new();
        let config = McpServerConfig {
            name: "server".into(),
            command: "definitely-not-a-real-binary-xyz".into(),
            args: vec![],
            env: HashMap::new(),
        };
        let n = register_server_with_defs(
            &mut registry,
            config,
            vec![def("foo-bar"), def("foo_bar")],
        );
        assert_eq!(n, 2);
        assert!(registry.get("mcp_server_foo_bar").is_some());
        assert!(registry.get("mcp_server_foo_bar_2").is_some());
        assert!(registry.get("foo-bar").is_some());
        assert!(registry.get("foo_bar").is_some());
    }

    #[test]
    fn registers_prefetched_defs_without_spawning() {
        // Must not spawn anything: command does not exist, so any spawn
        // attempt would fail. Proxies connect lazily on first execute().
        let mut registry = lucy_tools::ToolRegistry::new();
        let config = McpServerConfig {
            name: "computer_use".into(),
            command: "definitely-not-a-real-binary-xyz".into(),
            args: vec![],
            env: HashMap::new(),
        };
        let n =
            register_server_with_defs(&mut registry, config, vec![def("click"), def("type_text")]);
        assert_eq!(n, 2);
        assert!(registry.get("mcp_computer_use_click").is_some());
        assert!(registry.get("mcp_computer_use_type_text").is_some());
    }
}
