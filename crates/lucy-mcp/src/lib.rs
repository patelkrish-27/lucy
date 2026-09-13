use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use lucy_core::*;
use lucy_tools::ToolRegistry;
use serde::{Deserialize,Serialize};
use serde_json::{json,Value};
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt,AsyncWriteExt,BufReader};
use tokio::process::Command;

#[derive(Debug,Clone,Serialize,Deserialize)] pub struct McpServerConfig{pub name:String,pub command:String,#[serde(default)]pub args:Vec<String>,#[serde(default)]pub env:std::collections::HashMap<String,String>}
#[derive(Debug,Clone,Serialize,Deserialize)] pub struct McpToolDefinition{pub name:String,pub description:Option<String>,pub input_schema:Value}

async fn call_mcp(config:&McpServerConfig,method:&str,params:Value)->Result<Value>{
 let mut cmd=Command::new(&config.command);cmd.args(&config.args);for(k,v)in &config.env{cmd.env(k,v);}let mut child=cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().context("failed to spawn MCP server")?;let mut stdin=child.stdin.take().ok_or_else(||anyhow!("missing MCP stdin"))?;let stdout=child.stdout.take().ok_or_else(||anyhow!("missing MCP stdout"))?;let mut reader=BufReader::new(stdout);
 let init=json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"lucy","version":"0.2.0"}}});stdin.write_all(format!("{}\n",init).as_bytes()).await?;let mut line=String::new();reader.read_line(&mut line).await?;
 stdin.write_all(b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n").await?;
 let req=json!({"jsonrpc":"2.0","id":2,"method":method,"params":params});stdin.write_all(format!("{}\n",req).as_bytes()).await?;line.clear();reader.read_line(&mut line).await?;let v:Value=serde_json::from_str(&line).context("invalid MCP JSON-RPC response")?;let _=child.kill().await;if let Some(e)=v.get("error"){return Err(anyhow!("MCP error: {}",e));}Ok(v.get("result").cloned().unwrap_or(Value::Null))
}

pub struct StdioMcpClient{pub config:McpServerConfig}
impl StdioMcpClient{pub fn new(config:McpServerConfig)->Self{Self{config}}pub async fn list_tools(&self)->Result<Vec<McpToolDefinition>>{let r=call_mcp(&self.config,"tools/list",json!({})).await?;Ok(r["tools"].as_array().unwrap_or(&Vec::new()).iter().map(|x|McpToolDefinition{name:x["name"].as_str().unwrap_or_default().to_owned(),description:x["description"].as_str().map(str::to_owned),input_schema:x["inputSchema"].clone()}).collect())}}

struct McpToolProxy{server:McpServerConfig,definition:McpToolDefinition,full_name:String}
#[async_trait] impl Tool for McpToolProxy{fn name(&self)->&str{&self.full_name}fn description(&self)->&str{self.definition.description.as_deref().unwrap_or("MCP tool")}fn parameters_schema(&self)->Value{self.definition.input_schema.clone()}async fn execute(&self,input:Value,_ctx:ToolContext)->Result<Value>{call_mcp(&self.server,"tools/call",json!({"name":self.definition.name,"arguments":input})).await}}

pub async fn register_server(registry:&mut ToolRegistry,config:McpServerConfig)->Result<usize>{let client=StdioMcpClient::new(config.clone());let defs=client.list_tools().await?;let mut n=0;for d in defs{let full=format!("mcp_{}_{}",config.name.replace(|c:char|!c.is_ascii_alphanumeric(),'_'),d.name.replace(|c:char|!c.is_ascii_alphanumeric(),'_'));registry.register_arc(std::sync::Arc::new(McpToolProxy{server:config.clone(),definition:d,full_name:full}));n+=1;}Ok(n)}

pub fn load_config()->Result<Vec<McpServerConfig>>{let path=std::env::var("LUCY_MCP_CONFIG").map(std::path::PathBuf::from).unwrap_or_else(|_|std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_|".".into())).join(".config/lucy/mcp.toml"));if !path.exists(){return Ok(Vec::new())}let s=std::fs::read_to_string(path)?;#[derive(Deserialize)]struct C{#[serde(default)]servers:Vec<McpServerConfig>}Ok(toml::from_str::<C>(&s)?.servers)}

pub fn register_mcp_tools(_registry:&mut ToolRegistry,_tools:Vec<McpToolDefinition>)->Result<()> {Ok(())}
