use anyhow::{anyhow,Context,Result};
use async_trait::async_trait;
use lucy_core::*;
use lucy_tools::ToolRegistry;
use serde::{Deserialize,Serialize};
use serde_json::{json,Value};
use std::{collections::HashMap,process::Stdio,sync::Arc};
use tokio::io::{AsyncBufReadExt,AsyncWriteExt,BufReader};
use tokio::process::Command;

#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct McpServerConfig{pub name:String,pub command:String,#[serde(default)]pub args:Vec<String>,#[serde(default)]pub env:HashMap<String,String>}
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct McpToolDefinition{pub name:String,pub description:Option<String>,pub input_schema:Value}

async fn call_mcp(config:&McpServerConfig,method:&str,params:Value)->Result<Value>{
 let mut cmd=Command::new(&config.command);cmd.args(&config.args);for(k,v)in &config.env{cmd.env(k,v);}let mut child=cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().context("failed to spawn MCP server")?;
 let mut stdin=child.stdin.take().ok_or_else(||anyhow!("missing MCP stdin"))?;let stdout=child.stdout.take().ok_or_else(||anyhow!("missing MCP stdout"))?;let mut reader=BufReader::new(stdout);
 let init=json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"lucy","version":"0.2.0"}}});stdin.write_all(format!("{}\n",init).as_bytes()).await?;let mut line=String::new();reader.read_line(&mut line).await?;
 stdin.write_all(b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n").await?;
 let req=json!({"jsonrpc":"2.0","id":2,"method":method,"params":params});stdin.write_all(format!("{}\n",req).as_bytes()).await?;line.clear();reader.read_line(&mut line).await?;let value:Value=serde_json::from_str(&line).context("invalid MCP JSON-RPC response")?;let _=child.kill().await;if let Some(error)=value.get("error"){return Err(anyhow!("MCP error: {}",error));}Ok(value.get("result").cloned().unwrap_or(Value::Null))
}

pub struct StdioMcpClient{config:McpServerConfig}
impl StdioMcpClient{pub fn new(config:McpServerConfig)->Self{Self{config}}pub async fn list_tools(&self)->Result<Vec<McpToolDefinition>>{let result=call_mcp(&self.config,"tools/list",json!({})).await?;Ok(result.get("tools").and_then(Value::as_array).map(|tools|tools.iter().map(|x|McpToolDefinition{name:x["name"].as_str().unwrap_or_default().to_owned(),description:x["description"].as_str().map(str::to_owned),input_schema:x["inputSchema"].clone()}).collect()).unwrap_or_default())}}

struct McpToolProxy{server:McpServerConfig,definition:McpToolDefinition,full_name:String}
#[async_trait]impl Tool for McpToolProxy{fn name(&self)->&str{&self.full_name}fn description(&self)->&str{self.definition.description.as_deref().unwrap_or("MCP tool")}fn parameters_schema(&self)->Value{self.definition.input_schema.clone()}async fn execute(&self,input:Value,_ctx:ToolContext)->Result<Value>{call_mcp(&self.server,"tools/call",json!({"name":self.definition.name,"arguments":input})).await}}

pub async fn register_server(registry:&mut ToolRegistry,config:McpServerConfig)->Result<usize>{let defs=StdioMcpClient::new(config.clone()).list_tools().await?;let mut count=0;for definition in defs{let full_name=format!("mcp_{}_{}",config.name.replace(|c:char|!c.is_ascii_alphanumeric(),"_"),definition.name.replace(|c:char|!c.is_ascii_alphanumeric(),"_"));registry.register_arc(Arc::new(McpToolProxy{server:config.clone(),definition,full_name}));count+=1;}Ok(count)}

pub fn load_config()->Result<Vec<McpServerConfig>>{let path=std::env::var("LUCY_MCP_CONFIG").map(std::path::PathBuf::from).unwrap_or_else(|_|std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_|".".into())).join(".config/lucy/mcp.toml"));if !path.exists(){return Ok(Vec::new())}#[derive(Deserialize)]struct Config{#[serde(default)]servers:Vec<McpServerConfig>}Ok(toml::from_str(&std::fs::read_to_string(path)?)?.servers)}
