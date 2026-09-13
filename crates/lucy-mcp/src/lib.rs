use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use lucy_tools::ToolRegistry;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig { pub name: String, pub command: String, #[serde(default)] pub args: Vec<String> }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDefinition { pub name: String, pub description: Option<String>, pub input_schema: Value }

pub struct StdioMcpClient {
    config: McpServerConfig,
}

impl StdioMcpClient {
    pub fn new(config: McpServerConfig) -> Self {
        Self { config }
    }

    pub async fn list_tools(&self) -> Result<Vec<McpToolDefinition>> {
        let mut child = Command::new(&self.config.command)
            .args(&self.config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("failed to spawn MCP server process")?;

        let mut stdin = child.stdin.take().ok_or_else(|| anyhow::anyhow!("failed to open stdin"))?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow::anyhow!("failed to open stdout"))?;
        let mut reader = BufReader::new(stdout);

        let init_req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": { "name": "lucy-mcp", "version": "0.1.0" }
            }
        });

        let line = format!("{}\n", serde_json::to_string(&init_req)?);
        stdin.write_all(line.as_bytes()).await?;

        let mut response_line = String::new();
        reader.read_line(&mut response_line).await?;

        let list_req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        });

        let line = format!("{}\n", serde_json::to_string(&list_req)?);
        stdin.write_all(line.as_bytes()).await?;

        response_line.clear();
        reader.read_line(&mut response_line).await?;

        let parsed: Value = serde_json::from_str(&response_line).unwrap_or_default();
        let mut tools = Vec::new();
        if let Some(tool_list) = parsed["result"]["tools"].as_array() {
            for item in tool_list {
                let name = item["name"].as_str().unwrap_or_default().to_string();
                let description = item["description"].as_str().map(|s| s.to_string());
                let input_schema = item["inputSchema"].clone();
                tools.push(McpToolDefinition { name, description, input_schema });
            }
        }

        let _ = child.kill().await;
        Ok(tools)
    }
}

pub fn register_mcp_tools(_registry: &mut ToolRegistry, _tools: Vec<McpToolDefinition>) -> Result<()> { Ok(()) }
