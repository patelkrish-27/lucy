use anyhow::{Context, Result};
use lucy_mcp::{McpServerConfig, McpToolDefinition, StdioMcpClient};
use serde::{Deserialize, Serialize};
use std::{collections::{BTreeMap, HashMap}, path::PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Domain { Browser, Desktop, Vision, Tasks, Clipboard, Excalidraw, Stagehand, Hints, System, Unknown }
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Capability { Observe, Click, Type, Keyboard, Pointer, Window, Launch, Navigate, Extract, Act, Batch, Task, Clipboard, Draw, Wait, Bindings, Ground, Unknown }
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Operation { Observe, Act, Query, Manage, Transform, Unknown }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCapability { pub name:String,pub description:String,pub input_schema:serde_json::Value,pub domain:Domain,pub capabilities:Vec<Capability>,pub operation:Operation,pub read_only:bool,pub destructive:bool,pub batchable:bool }
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HyprFastCatalog { pub tools:HashMap<String,ToolCapability> }
impl HyprFastCatalog {
 pub fn from_tools(tools:Vec<McpToolDefinition>)->Self{let mut catalog=Self::default();for tool in tools{let c=classify(&tool);catalog.tools.insert(c.name.clone(),c);}catalog}
 pub fn len(&self)->usize{self.tools.len()}
 pub fn is_empty(&self)->bool{self.tools.is_empty()}
 pub fn by_domain(&self,domain:Domain)->Vec<&ToolCapability>{self.tools.values().filter(|t|t.domain==domain).collect()}
 pub fn summary(&self)->BTreeMap<String,usize>{let mut out=BTreeMap::new();for tool in self.tools.values(){*out.entry(format!("{:?}",tool.domain)).or_insert(0)+=1;}out}
 pub async fn discover(config:McpServerConfig)->Result<Self>{let tools=StdioMcpClient::new(config).list_tools().await.context("failed to discover HyprFast MCP tools")?;Ok(Self::from_tools(tools))}
 pub async fn discover_default()->Result<Self>{Self::discover(default_config()).await}
 pub fn cache_path()->PathBuf{std::env::var("LUCY_HYPRFAST_CACHE").map(PathBuf::from).unwrap_or_else(|_|PathBuf::from(std::env::var("HOME").unwrap_or_else(|_|".".into())).join(".local/state/lucy/hyprfast-catalog.json"))}
 pub async fn save(&self)->Result<()>{let path=Self::cache_path();if let Some(parent)=path.parent(){tokio::fs::create_dir_all(parent).await?;}tokio::fs::write(path,serde_json::to_vec_pretty(self)?).await?;Ok(())}
 pub async fn load()->Result<Self>{Ok(serde_json::from_slice(&tokio::fs::read(Self::cache_path()).await?)?)}
}
pub fn default_config()->McpServerConfig{McpServerConfig{name:"hyprfast".into(),command:"hyprfast".into(),args:vec!["mcp".into()],env:Default::default()}}
fn classify(tool:&McpToolDefinition)->ToolCapability{
 let name=tool.name.to_ascii_lowercase();let desc=tool.description.clone().unwrap_or_default().to_ascii_lowercase();let text=format!("{} {}",name,desc);
 let domain=if has(&text,&["browser_","browser "]){Domain::Browser}else if has(&text,&["stagehand"]){Domain::Stagehand}else if has(&text,&["hint_"]){Domain::Hints}else if has(&text,&["screenshot","ground","vision"]){Domain::Vision}else if has(&text,&["clipboard"]){Domain::Clipboard}else if has(&text,&["excalidraw"]){Domain::Excalidraw}else if has(&text,&["task_"]){Domain::Tasks}else if has(&text,&["hypr","desktop","window","pointer","keyboard","click_ui","ui_"]){Domain::Desktop}else{Domain::System};
 let mut capabilities=Vec::new();for(terms,cap)in [(&["screenshot","observe","read","inspect","query"][..],Capability::Observe),(&["click","press"][..],Capability::Click),(&["type","fill"][..],Capability::Type),(&["keyboard","key_"][..],Capability::Keyboard),(&["pointer","mouse"][..],Capability::Pointer),(&["window","workspace"][..],Capability::Window),(&["launch"][..],Capability::Launch),(&["navigate","goto","url"][..],Capability::Navigate),(&["extract","text","content"][..],Capability::Extract),(&["act","action"][..],Capability::Act),(&["batch"][..],Capability::Batch),(&["task_"][..],Capability::Task),(&["clipboard"][..],Capability::Clipboard),(&["excalidraw","draw"][..],Capability::Draw),(&["wait"][..],Capability::Wait),(&["bind"][..],Capability::Bindings),(&["ground"][..],Capability::Ground)]{if has(&text,terms){capabilities.push(cap)}}if capabilities.is_empty(){capabilities.push(Capability::Unknown)}
 let read_only=matches!(domain,Domain::Vision)||has(&text,&["screenshot","inspect","get","list","find","query"]);let destructive=has(&text,&["close","kill","delete","remove","destroy","shutdown","logout"]);let batchable=has(&text,&["batch","act_fast"]);let operation=if read_only{Operation::Observe}else if has(&text,&["list","get","find","query","screenshot"]){Operation::Query}else if has(&text,&["draw","clipboard","task_"]){Operation::Manage}else if has(&text,&["extract","ground"]){Operation::Transform}else{Operation::Act};
 ToolCapability{name:tool.name.clone(),description:tool.description.clone().unwrap_or_default(),input_schema:tool.input_schema.clone(),domain,capabilities,operation,read_only,destructive,batchable}
}
fn has(text:&str,terms:&[&str])->bool{terms.iter().any(|term|text.contains(term))}
#[cfg(test)]mod tests{use super::*;fn tool(name:&str,description:&str)->McpToolDefinition{McpToolDefinition{name:name.into(),description:Some(description.into()),input_schema:serde_json::json!({"type":"object"})}}#[test]fn categorizes_tools(){let c=HyprFastCatalog::from_tools(vec![tool("browser_click","click browser element"),tool("screenshot","capture desktop"),tool("act_batch","execute batch"),tool("task_init","start task")]);assert_eq!(c.tools["browser_click"].domain,Domain::Browser);assert!(c.tools["browser_click"].capabilities.contains(&Capability::Click));assert_eq!(c.tools["screenshot"].domain,Domain::Vision);assert!(c.tools["act_batch"].batchable);assert_eq!(c.tools["task_init"].domain,Domain::Tasks);}}
