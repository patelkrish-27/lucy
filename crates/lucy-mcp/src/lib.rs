use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use lucy_tools::ToolRegistry;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig { pub name: String, pub command: String, #[serde(default)] pub args: Vec<String> }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDefinition { pub name: String, pub description: Option<String>, pub input_schema: Value }

pub trait McpTransport: Send + Sync { fn server_name(&self) -> &str; }
pub fn register_mcp_tools(_registry: &mut ToolRegistry, _tools: Vec<McpToolDefinition>) -> Result<()> { Ok(()) }
