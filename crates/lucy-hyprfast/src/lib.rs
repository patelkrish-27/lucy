use anyhow::{Context, Result};
use lucy_mcp::{McpServerConfig, McpToolDefinition, StdioMcpClient};
use serde::{Deserialize, Serialize};
use std::{collections::{BTreeMap, HashMap, HashSet}, path::PathBuf};

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
 pub fn capability_for_mcp_name(&self,name:&str)->Option<&ToolCapability>{self.tools.values().find(|tool|full_name(&tool.name)==name||tool.name==name)}
 pub async fn discover(config:McpServerConfig)->Result<Self>{let client=StdioMcpClient::new(config);let tools=client.list_tools().await.context("failed to discover HyprFast MCP tools")?;Ok(Self::from_tools(tools))}
 pub async fn discover_default()->Result<Self>{Self::discover(default_config()).await}
 pub fn cache_path()->PathBuf{std::env::var("LUCY_HYPRFAST_CACHE").map(PathBuf::from).unwrap_or_else(|_|PathBuf::from(std::env::var("HOME").unwrap_or_else(|_|".".into())).join(".local/state/lucy/hyprfast-catalog.json"))}
 pub async fn save(&self)->Result<()>{let path=Self::cache_path();if let Some(parent)=path.parent(){tokio::fs::create_dir_all(parent).await?;}tokio::fs::write(path,serde_json::to_vec_pretty(self)?).await?;Ok(())}
 pub async fn load()->Result<Self>{Ok(serde_json::from_slice(&tokio::fs::read(Self::cache_path()).await?)?)}

 pub fn route(&self,prompt:&str)->Route {
   let text=prompt.to_ascii_lowercase();
   let mut scored:Vec<(i32,&ToolCapability)>=self.tools.values().map(|t|(score(&text,t),t)).collect();
   scored.sort_by(|a,b|b.0.cmp(&a.0).then_with(||a.1.name.cmp(&b.1.name)));
   let mut selected=HashSet::new();
   for (s,t) in scored.iter().take(8) { if *s > 0 { selected.insert(full_name(&t.name)); } }
   if selected.is_empty() { for t in self.tools.values().filter(|t| !t.destructive).take(4) { selected.insert(full_name(&t.name)); } }
   Route { candidates:selected.into_iter().collect(), strategy:strategy_for(&text), fast_path:is_fast_path(&text) }
 }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Route { pub candidates:Vec<String>, pub strategy:String, pub fast_path:bool }
fn full_name(name:&str)->String{format!("mcp_hyprfast_{}",sanitize(name))}
fn sanitize(s:&str)->String{s.chars().map(|c|if c.is_ascii_alphanumeric(){c}else{'_'}).collect()}
fn score(text:&str,t:&ToolCapability)->i32{
 let mut s=0;
 let has=|terms:&[&str]|terms.iter().any(|x|text.contains(x));
 match t.domain { Domain::Browser=>{if has(&["browser","website","web","url","page","tab"]){s+=30}}, Domain::Desktop=>{if has(&["desktop","window","workspace","app","application","monitor"]){s+=30}}, Domain::Vision=>{if has(&["screenshot","screen","see","look","visual"]){s+=35}}, Domain::Hints=>{if has(&["click","select","press","element"]){s+=18}}, Domain::Stagehand=>{if has(&["complex","workflow","browser task"]){s+=16}}, _=>{} }
 for c in &t.capabilities { s+=match c { Capability::Click if has(&["click","select","press"] )=>28, Capability::Type if has(&["type","write","enter","fill","search"] )=>25, Capability::Navigate if has(&["open","go to","navigate","visit","url"] )=>28, Capability::Window if has(&["window","workspace","move","focus"] )=>28, Capability::Launch if has(&["open","launch","start"] )=>26, Capability::Observe if has(&["see","inspect","find","what","screenshot"] )=>24, Capability::Extract if has(&["read","extract","get","find"] )=>20, Capability::Batch if has(&["batch","many","multiple"] )=>35, Capability::Act if has(&["do","execute","perform","act"] )=>12, _=>0 }; }
 if t.batchable && has(&["fast","quick","multiple","many"]){s+=18}; if t.read_only && has(&["what","check","show","see"]){s+=8}; if t.destructive && !has(&["close","kill","delete","remove","shutdown"]){s-=50};
 s
}
fn strategy_for(text:&str)->String{if text.contains("screenshot")||text.contains("screen"){"direct-observe".into()}else if text.contains("click")||text.contains("select")||text.contains("press"){"hint-first".into()}else if text.contains("browser")||text.contains("website")||text.contains("url"){"browser-hint-ax-stagehand-vision".into()}else{"capability-first".into()}}
fn is_fast_path(text:&str)->bool{["screenshot","take a screenshot","open firefox","open chrome","launch firefox","launch chrome"].iter().any(|x|text.trim()==*x)}

pub fn default_config()->McpServerConfig{McpServerConfig{name:"hyprfast".into(),command:"hyprfast".into(),args:vec!["mcp".into()],env:Default::default()}}
fn classify(tool:&McpToolDefinition)->ToolCapability{
 let name=tool.name.to_ascii_lowercase();let desc=tool.description.clone().unwrap_or_default().to_ascii_lowercase();let text=format!("{} {}",name,desc);
 let domain=if has(&text,&["browser_","browser "]){Domain::Browser}else if has(&text,&["stagehand"]){Domain::Stagehand}else if has(&text,&["hint_"]){Domain::Hints}else if has(&text,&["screenshot","ground","vision"]){Domain::Vision}else if has(&text,&["clipboard"]){Domain::Clipboard}else if has(&text,&["excalidraw"]){Domain::Excalidraw}else if has(&text,&["task_"]){Domain::Tasks}else if has(&text,&["hypr","desktop","window","pointer","keyboard","click_ui","ui_"]){Domain::Desktop}else{Domain::System};
 let mut capabilities=Vec::new();for(terms,cap)in [(&["screenshot","observe","read","inspect","query"][..],Capability::Observe),(&["click","press"][..],Capability::Click),(&["type","fill"][..],Capability::Type),(&["keyboard","key_"][..],Capability::Keyboard),(&["pointer","mouse"][..],Capability::Pointer),(&["window","workspace"][..],Capability::Window),(&["launch"][..],Capability::Launch),(&["navigate","goto","url"][..],Capability::Navigate),(&["extract","text","content"][..],Capability::Extract),(&["act","action"][..],Capability::Act),(&["batch"][..],Capability::Batch),(&["task_"][..],Capability::Task),(&["clipboard"][..],Capability::Clipboard),(&["excalidraw","draw"][..],Capability::Draw),(&["wait"][..],Capability::Wait),(&["bind"][..],Capability::Bindings),(&["ground"][..],Capability::Ground)]{if has(&text,terms){capabilities.push(cap)}}if capabilities.is_empty(){capabilities.push(Capability::Unknown)}
 let read_only=matches!(domain,Domain::Vision)||has(&text,&["screenshot","inspect","get","list","find","query"]);let destructive=has(&text,&["close","kill","delete","remove","destroy","shutdown","logout"]);let batchable=has(&text,&["batch","act_fast"]);let operation=if read_only{Operation::Observe}else if has(&text,&["list","get","find","query","screenshot"]){Operation::Query}else if has(&text,&["draw","clipboard","task_"]){Operation::Manage}else if has(&text,&["extract","ground"]){Operation::Transform}else{Operation::Act};
 ToolCapability{name:tool.name.clone(),description:tool.description.clone().unwrap_or_default(),input_schema:tool.input_schema.clone(),domain,capabilities,operation,read_only,destructive,batchable}
}
fn has(text:&str,terms:&[&str])->bool{terms.iter().any(|term|text.contains(term))}
#[cfg(test)]mod tests{use super::*;fn tool(name:&str,description:&str)->McpToolDefinition{McpToolDefinition{name:name.into(),description:Some(description.into()),input_schema:serde_json::json!({"type":"object"})}}#[test]fn categorizes_tools(){let c=HyprFastCatalog::from_tools(vec![tool("browser_click","click browser element"),tool("screenshot","capture desktop"),tool("act_batch","execute batch"),tool("task_init","start task")]);assert_eq!(c.tools["browser_click"].domain,Domain::Browser);assert!(c.tools["browser_click"].capabilities.contains(&Capability::Click));assert_eq!(c.tools["screenshot"].domain,Domain::Vision);assert!(c.tools["act_batch"].batchable);assert_eq!(c.tools["task_init"].domain,Domain::Tasks);}#[test]fn routes_browser_click(){let c=HyprFastCatalog::from_tools(vec![tool("browser_click","click browser element"),tool("screenshot","capture desktop"),tool("desktop_window","move workspace window")]);let r=c.route("click the browser button");assert!(r.candidates.iter().any(|x|x.contains("browser_click")));assert_eq!(r.strategy,"hint-first");}}