use anyhow::{Context, Result};
use lucy_mcp::{McpServerConfig, McpToolDefinition, StdioMcpClient};
use serde::{Deserialize, Serialize};
use std::{collections::{BTreeMap, HashMap, HashSet}, path::PathBuf, process::Command};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Domain { Browser, Desktop, Vision, Tasks, Clipboard, Excalidraw, Stagehand, Hints, System, Unknown }
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Capability { Observe, Click, Type, Keyboard, Pointer, Window, Launch, Navigate, Extract, Act, Batch, Task, Clipboard, Draw, Wait, Bindings, Ground, Unknown }
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Operation { Observe, Act, Query, Manage, Transform, Unknown }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCapability { pub name:String,pub description:String,pub input_schema:serde_json::Value,pub domain:Domain,pub capabilities:Vec<Capability>,pub operation:Operation,pub read_only:bool,pub destructive:bool,pub batchable:bool,pub semantic:bool }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Environment {
    pub os: String,
    pub desktop: String,
    pub display_server: String,
    pub wayland: bool,
    pub hyprland: bool,
    pub hyprctl: bool,
    pub accessibility: bool,
    pub clipboard: bool,
    pub screenshot: bool,
}

impl Default for Environment {
    fn default() -> Self { Self::detect() }
}

impl Environment {
    pub fn detect() -> Self {
        let os = if cfg!(target_os = "linux") { "linux" } else if cfg!(target_os = "windows") { "windows" } else if cfg!(target_os = "macos") { "macos" } else { "unknown" }.into();
        let desktop_name = std::env::var("XDG_CURRENT_DESKTOP").unwrap_or_default().to_ascii_lowercase();
        let session = std::env::var("XDG_SESSION_TYPE").unwrap_or_default().to_ascii_lowercase();
        let wayland = session == "wayland" || std::env::var_os("WAYLAND_DISPLAY").is_some();
        let hyprland = std::env::var_os("HYPRLAND_INSTANCE_SIGNATURE").is_some()
            || desktop_name.contains("hyprland")
            || std::env::var_os("HYPRLAND_CMD").is_some();
        let desktop = if hyprland { "hyprland" } else if !desktop_name.is_empty() { desktop_name } else { "unknown" }.into();
        let display_server = if wayland { "wayland" } else if session == "x11" || std::env::var_os("DISPLAY").is_some() { "x11" } else { "unknown" }.into();
        let hyprctl = hyprland && command_exists("hyprctl");
        let accessibility = std::env::var_os("AT_SPI_BUS_ADDRESS").is_some()
            || std::env::var("LUCY_ACCESSIBILITY").map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false);
        let clipboard = command_exists("wl-copy") || command_exists("xclip") || command_exists("xsel");
        let screenshot = command_exists("grim") || command_exists("gnome-screenshot") || command_exists("scrot") || command_exists("hyprshot");
        Self { os, desktop, display_server, wayland, hyprland, hyprctl, accessibility, clipboard, screenshot }
    }

    pub fn context(&self) -> String {
        format!("OS={} desktop={} display_server={} wayland={} hyprland={} hyprctl={} accessibility={} clipboard={} screenshot={}", self.os, self.desktop, self.display_server, self.wayland, self.hyprland, self.hyprctl, self.accessibility, self.clipboard, self.screenshot)
    }
}

fn command_exists(command: &str) -> bool {
    Command::new(command).arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HyprFastCatalog {
    pub tools:HashMap<String,ToolCapability>,
    #[serde(default)]
    pub environment: Environment,
}

impl HyprFastCatalog {
 pub fn from_tools(tools:Vec<McpToolDefinition>)->Self{let mut catalog=Self{tools:HashMap::new(),environment:Environment::detect()};for tool in tools{let c=classify(&tool);catalog.tools.insert(c.name.clone(),c);}catalog}
 pub fn len(&self)->usize{self.tools.len()}
 pub fn is_empty(&self)->bool{self.tools.is_empty()}
 pub fn by_domain(&self,domain:Domain)->Vec<&ToolCapability>{self.tools.values().filter(|t|t.domain==domain).collect()}
 pub fn summary(&self)->BTreeMap<String,usize>{let mut out=BTreeMap::new();for tool in self.tools.values(){*out.entry(format!("{:?}",tool.domain)).or_insert(0)+=1;}out}
 pub fn capability_for_mcp_name(&self,name:&str)->Option<&ToolCapability>{self.tools.values().find(|tool|full_name(&tool.name)==name||tool.name==name)}
 pub async fn discover(config:McpServerConfig)->Result<Self>{
   let client=StdioMcpClient::new(config);let mut tools=client.list_tools().await.context("failed to discover HyprFast MCP tools")?;
   if std::env::var("LUCY_COMPUTER_USE_ENABLED").map(|v|v!="0"&&v.to_ascii_lowercase()!="false").unwrap_or(true){
      let computer=StdioMcpClient::new(computer_use_config());
      match computer.list_tools().await{Ok(defs)=>{for mut d in defs{d.name=format!("computer_use_{}",d.name);tools.push(d);}tracing::info!("ADK Computer Use MCP merged into capability catalog");},Err(error)=>tracing::warn!(error=%error,"ADK Computer Use unavailable; continuing with HyprFast only")}
   }
   Ok(Self::from_tools(tools))
 }
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
  pub fn route_domain(&self,category:&str,goal:&str)->Route{
    let category_l=category.to_ascii_lowercase();let text=goal.to_ascii_lowercase();
    let domain=match category_l.as_str(){"browser"=>Domain::Browser,"stagehand"=>Domain::Stagehand,"hints"=>Domain::Hints,"vision"=>Domain::Vision,"clipboard"=>Domain::Clipboard,"excalidraw"=>Domain::Excalidraw,"tasks"=>Domain::Tasks,"desktop"=>Domain::Desktop,"system"=>Domain::System,_=>Domain::Unknown};
    if domain!=Domain::Unknown{
      let mut tools:Vec<&ToolCapability>=self.by_domain(domain);
      let explicit_launch=has(&text,&["launch browser","start browser","open browser","new browser window","new browser"]);
      if domain==Domain::Browser && !explicit_launch {tools.retain(|t|!t.capabilities.contains(&Capability::Launch));}
      if domain==Domain::Desktop {tools.sort_by(|a,b|{let sa=desktop_score(&text,a,self);let sb=desktop_score(&text,b,self);sb.cmp(&sa).then_with(||a.name.cmp(&b.name))});}
      else {tools.sort_by(|a,b|score(&text,b).cmp(&score(&text,a)).then_with(||a.name.cmp(&b.name)));}
      let candidates=tools.into_iter().filter(|t|if domain==Domain::Desktop{desktop_score(&text,t,self)>0}else{score(&text,t)>0}).take(8).map(|t|full_name(&t.name)).collect::<Vec<_>>();
      if !candidates.is_empty(){let strategy=if domain==Domain::Browser&&has(&text,&["youtube","song","music","play","watch","listen","search"]){"direct-browser-navigation".into()}else if domain==Domain::Desktop{desktop_strategy(&text,self)}else{strategy_for(&text)};return Route{candidates,strategy,fast_path:false};}
    }
    self.route(goal)
  }
  pub fn context_for(&self,route:&Route)->String{let mut out=String::new();out.push_str(&format!("Lucy capability catalog: {} tools, strategy={}, fast_path={}\n",self.tools.len(),route.strategy,route.fast_path));out.push_str(&format!("Environment: {}\n",self.environment.context()));out.push_str("Routing policy: on Hyprland prefer semantic Computer Use window/app/workspace actions; use hyprctl only when the requested desktop state is explicitly Hyprland-specific; never expose OS commands to the model as the primary interface.\n");out.push_str("Domain summary:\n");for(k,v)in self.summary(){out.push_str(&format!("  {}: {}\n",k,v));}out.push_str("Candidates:\n");for cand in &route.candidates{if let Some(cap)=self.capability_for_mcp_name(cand){out.push_str(&format!("  {} [{:?}] - {}\n",cand,cap.domain,cap.description));}else{out.push_str(&format!("  {} (unknown)\n",cand));}}out}
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Route { pub candidates:Vec<String>, pub strategy:String, pub fast_path:bool }
fn full_name(name:&str)->String{if name.starts_with("computer_use_"){format!("mcp_computer_use_{}",sanitize(name.trim_start_matches("computer_use_")))}else{format!("mcp_hyprfast_{}",sanitize(name))}}
fn sanitize(s:&str)->String{s.chars().map(|c|if c.is_ascii_alphanumeric(){c}else{'_'}).collect()}
fn score(text:&str,t:&ToolCapability)->i32{let mut s=0;let has=|terms:&[&str]|terms.iter().any(|x|text.contains(x));match t.domain{Domain::Browser=>{if has(&["browser","website","web","url","page","tab","youtube","video","music","song","play","watch","listen"]){s+=30}},Domain::Desktop=>{if has(&["desktop","window","workspace","app","application","monitor"]){s+=30}},Domain::Vision=>{if has(&["screenshot","screen","see","look","visual"]){s+=35}},Domain::Hints=>{if has(&["click","select","press","element"]){s+=18}},Domain::Stagehand=>{if has(&["complex","workflow","browser task"]){s+=16}},_=>{}}for c in &t.capabilities{s+=match c{Capability::Click if has(&["click","select","press"])= >28,Capability::Type if has(&["type","write","enter","fill"])= >25,Capability::Navigate if has(&["open","go to","navigate","visit","url","play","watch","listen","search"])= >28,Capability::Window if has(&["window","workspace","move","focus"])= >28,Capability::Launch if has(&["launch","open","start","play"])= >26,Capability::Observe if has(&["see","inspect","find","what","screenshot"])= >24,Capability::Extract if has(&["read","extract","get","find"])= >20,Capability::Batch if has(&["batch","many","multiple"])= >35,Capability::Act if has(&["do","execute","perform","act"])= >12,_=>0};}if t.batchable&&has(&["fast","quick","multiple","many"]){s+=18};if t.read_only&&has(&["what","check","show","see"]){s+=8};if t.destructive&&!has(&["close","kill","delete","remove","shutdown"]){s-=50};s}
fn desktop_score(text:&str,t:&ToolCapability,env:&Environment)->i32{let n=t.name.to_ascii_lowercase();let mut s=0;if t.semantic{s+=20;}if env.hyprland {if n.contains("window")||n.contains("space")||n.contains("workspace"){s+=25;}if n.contains("application")||n.contains("app_"){s+=15;}}if n.contains("find_element")||n.contains("ui_tree"){s+=80*if text.contains("find")||text.contains("inspect")||text.contains("element")||text.contains("button")||text.contains("field"){1}else{0}}if n.contains("press_button")||n.contains("click_element"){s+=90*if text.contains("click")||text.contains("press")||text.contains("button"){1}else{0}}if n.contains("menu"){s+=90*if text.contains("menu")||text.contains("settings")||text.contains("preferences"){1}else{0}}if n.contains("form"){s+=90*if text.contains("form")||text.contains("fill")||text.contains("field"){1}else{0}}if n.contains("application")||n.contains("app_"){s+=70*if text.contains("open")||text.contains("launch")||text.contains("app"){1}else{0}}if n.contains("window")||n.contains("space")||n.contains("display"){s+=70*if text.contains("window")||text.contains("workspace")||text.contains("focus"){1}else{0}}if n.contains("screenshot"){s+=80*if text.contains("screenshot")||text.contains("screen")||text.contains("see")||text.contains("look"){1}else{0}}if n.contains("clipboard"){s+=80*if text.contains("clipboard")||text.contains("copy")||text.contains("paste"){1}else{0}}if n.contains("type")||n.contains("key"){s+=60*if text.contains("type")||text.contains("write")||text.contains("enter"){1}else{0}}if n.contains("scroll")||n.contains("drag")||n.contains("mouse")||n.contains("pointer"){s+=40*if text.contains("scroll")||text.contains("drag")||text.contains("mouse"){1}else{0}}if t.read_only{s+=5}if t.destructive&&!has(text,&["close","delete","kill","shutdown","remove"]){s-=100}s}
fn desktop_strategy(text:&str,env:&Environment)->String{if env.hyprland&&has(text,&["workspace","window","move","focus"]){"hyprland-semantic-desktop".into()}else if has(text,&["menu","settings","preferences"]){"semantic-menu".into()}else if has(text,&["form","fill","field","input"]){"semantic-form".into()}else if has(text,&["screenshot","screen","see","look"]){"visual-observe".into()}else if has(text,&["window","workspace","focus","app","application"]){"app-window-semantic".into()}else{"accessibility-first".into()}}
fn strategy_for(text:&str)->String{if text.contains("screenshot")||text.contains("screen"){"direct-observe".into()}else if text.contains("click")||text.contains("select")||text.contains("press"){"hint-first".into()}else if text.contains("browser")||text.contains("website")||text.contains("url"){"browser-hint-ax-stagehand-vision".into()}else{"capability-first".into()}}
fn is_fast_path(text:&str)->bool{["screenshot","take a screenshot","open firefox","open chrome","launch firefox","launch chrome"].iter().any(|x|text.trim()==*x)}
pub fn default_config()->McpServerConfig{McpServerConfig{name:"hyprfast".into(),command:"hyprfast".into(),args:vec!["mcp".into()],env:Default::default()}}
pub fn computer_use_config()->McpServerConfig{McpServerConfig{name:"computer_use".into(),command:std::env::var("LUCY_COMPUTER_USE_COMMAND").unwrap_or_else(|_|"npx".into()),args:std::env::var("LUCY_COMPUTER_USE_ARGS").map(|v|v.split_whitespace().map(str::to_owned).collect()).unwrap_or_else(|_|vec!["-y".into(),"@zavora-ai/computer-use-mcp".into()]),env:Default::default()}}
fn classify(tool:&McpToolDefinition)->ToolCapability{let name=tool.name.to_ascii_lowercase();let desc=tool.description.clone().unwrap_or_default().to_ascii_lowercase();let text=format!("{} {}",name,desc);let domain=if has(&text,&["browser_","browser "]){Domain::Browser}else if has(&text,&["stagehand"]){Domain::Stagehand}else if has(&text,&["hint_"]){Domain::Hints}else if has(&text,&["screenshot","ground","vision"]){if name.starts_with("computer_use_"){Domain::Desktop}else{Domain::Vision}}else if has(&text,&["clipboard"]){Domain::Clipboard}else if has(&text,&["excalidraw"]){Domain::Excalidraw}else if has(&text,&["task_"]){Domain::Tasks}else if has(&text,&["computer_use_","computer use","hypr","desktop","window","pointer","keyboard","click_ui","ui_","application","menu","form"]){Domain::Desktop}else{Domain::System};let mut capabilities=Vec::new();for(terms,cap)in[(&["screenshot","observe","read","inspect","query"][..],Capability::Observe),(&["click","press"][..],Capability::Click),(&["type","fill"][..],Capability::Type),(&["keyboard","key_"][..],Capability::Keyboard),(&["pointer","mouse"][..],Capability::Pointer),(&["window","workspace","space"][..],Capability::Window),(&["launch","open_application","discover_application"][..],Capability::Launch),(&["navigate","goto","url"][..],Capability::Navigate),(&["extract","text","content","element"][..],Capability::Extract),(&["act","action"][..],Capability::Act),(&["batch"][..],Capability::Batch),(&["task_"][..],Capability::Task),(&["clipboard"][..],Capability::Clipboard),(&["excalidraw","draw"][..],Capability::Draw),(&["wait"][..],Capability::Wait),(&["bind"][..],Capability::Bindings),(&["ground"][..],Capability::Ground)]{if has(&text,terms){capabilities.push(cap)}}if capabilities.is_empty(){capabilities.push(Capability::Unknown)}let read_only=matches!(domain,Domain::Vision)||has(&text,&["screenshot","inspect","get","list","find","query","focused","frontmost"]);let destructive=has(&text,&["close","kill","delete","remove","destroy","shutdown","logout"]);let batchable=has(&text,&["batch","act_fast"]);let operation=if read_only{Operation::Observe}else if has(&text,&["list","get","find","query","screenshot"]){Operation::Query}else if has(&text,&["draw","clipboard","task_"]){Operation::Manage}else if has(&text,&["extract","ground"]){Operation::Transform}else{Operation::Act};let semantic=has(&text,&["accessibility","semantic","element","button","form","menu","ui tree","application","window"]);ToolCapability{name:tool.name.clone(),description:tool.description.clone().unwrap_or_default(),input_schema:tool.input_schema.clone(),domain,capabilities,operation,read_only,destructive,batchable,semantic}}
fn has(text:&str,terms:&[&str])->bool{terms.iter().any(|term|text.contains(term))}
#[cfg(test)]mod tests{use super::*;fn tool(name:&str,description:&str)->McpToolDefinition{McpToolDefinition{name:name.into(),description:Some(description.into()),input_schema:serde_json::json!({"type":"object"})}}#[test]fn categorizes_tools(){let c=HyprFastCatalog::from_tools(vec![tool("browser_click","click browser element"),tool("screenshot","capture desktop"),tool("act_batch","execute batch"),tool("task_init","start task")]);assert_eq!(c.tools["browser_click"].domain,Domain::Browser);assert!(c.tools["browser_click"].capabilities.contains(&Capability::Click));assert_eq!(c.tools["screenshot"].domain,Domain::Vision);assert!(c.tools["act_batch"].batchable);assert_eq!(c.tools["task_init"].domain,Domain::Tasks);}#[test]fn routes_browser_click(){let c=HyprFastCatalog::from_tools(vec![tool("browser_click","click browser element"),tool("screenshot","capture desktop"),tool("desktop_window","move workspace window")]);let r=c.route("click the browser button");assert!(r.candidates.iter().any(|x|x.contains("browser_click")));assert_eq!(r.strategy,"hint-first");}#[test]fn routes_play_song_to_browser_tools(){let c=HyprFastCatalog::from_tools(vec![tool("browser_launch","launch browser"),tool("browser_navigate","navigate browser to url"),tool("browser_click","click browser element"),tool("screenshot","capture desktop"),tool("read_file","read a file")]);let r=c.route("play sammi meri waar song");assert!(r.candidates.iter().any(|x|x.contains("browser_navigate")),"play-song must route to browser tools, got {:?}",r.candidates);}#[test]fn domain_route_does_not_offer_launch_for_youtube_flow(){let c=HyprFastCatalog::from_tools(vec![tool("browser_launch","launch browser"),tool("browser_navigate","navigate browser to url"),tool("browser_click","click browser element"),tool("browser_type","type into browser"),tool("browser_snapshot","observe browser")]);let r=c.route_domain("browser","navigate to YouTube and search for Boom Shaka Laka");assert!(!r.candidates.iter().any(|x|x.contains("browser_launch")));assert!(r.candidates.iter().any(|x|x.contains("browser_navigate")));}#[test]fn domain_route_allows_explicit_browser_launch(){let c=HyprFastCatalog::from_tools(vec![tool("browser_launch","launch browser"),tool("browser_navigate","navigate browser to url")]);let r=c.route_domain("browser","launch browser");assert!(r.candidates.iter().any(|x|x.contains("browser_launch")));}#[test]fn routes_desktop_semantically(){let c=HyprFastCatalog::from_tools(vec![tool("computer_use_find_element","find accessibility element"),tool("computer_use_press_button","press a button"),tool("computer_use_left_click","click coordinates")]);let r=c.route_domain("desktop","click the Save button");assert!(r.candidates.iter().any(|x|x.contains("mcp_computer_use_press_button")));assert_eq!(r.strategy,"accessibility-first");}}
