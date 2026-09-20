use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::{env, fs, path::{Path, PathBuf}};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LucyConfig { pub general: GeneralConfig, pub models: ModelConfig, pub planner: PlannerConfig, pub voice: VoiceConfig, pub hyprfast: HyprFastConfig, pub appearance: AppearanceConfig, pub sessions: SessionConfig, pub approvals: ApprovalConfig, pub browser: BrowserConfig, pub harness: HarnessConfig }
#[derive(Debug, Clone, Serialize, Deserialize)] #[serde(default)] pub struct GeneralConfig { pub startup_screen:String, pub compact_after_command:bool }
#[derive(Debug, Clone, Serialize, Deserialize)] #[serde(default)] pub struct ModelConfig {
    pub main:String,
    // Legacy generic (fallback for both)
    pub base_url:Option<String>,
    #[serde(default, skip_serializing_if="Option::is_none")] pub api_key:Option<String>,
    // Main-model overrides. The same main model is used for planning, tool selection, and recovery.
    #[serde(default, skip_serializing_if="Option::is_none")] pub main_api_key:Option<String>,
    #[serde(default, skip_serializing_if="Option::is_none")] pub main_base_url:Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)] #[serde(default)] pub struct PlannerConfig { pub max_subtasks:usize, pub max_depth:usize, pub verify_state:bool, pub parallel:bool, pub replan_on_failure:bool }
#[derive(Debug, Clone, Serialize, Deserialize)] #[serde(default)] pub struct VoiceConfig { pub provider:String, pub model:String, pub language:Option<String>, pub push_to_talk:String, #[serde(default, skip_serializing_if="Option::is_none")] pub api_key:Option<String> }
#[derive(Debug, Clone, Serialize, Deserialize)] #[serde(default)] pub struct HyprFastConfig { pub command:String, pub args:Vec<String>, pub max_candidates:usize, pub batching:bool, pub parallel:bool, pub verify_actions:bool }
#[derive(Debug, Clone, Serialize, Deserialize)] #[serde(default)] pub struct AppearanceConfig { pub theme:String, pub animations:bool, pub activity_verbosity:String }
#[derive(Debug, Clone, Serialize, Deserialize)] #[serde(default)] pub struct SessionConfig { pub file:Option<PathBuf>, pub dir:Option<PathBuf>, pub resume:bool, pub max_history:usize }
#[derive(Debug, Clone, Serialize, Deserialize)] #[serde(default)] pub struct ApprovalConfig { pub mode:String }
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BrowserConfig {
    /// Explicit browser binary. When None, resolved once at startup via
    /// `which brave || which google-chrome || which chromium ...`, never
    /// guessed by the LLM per-run (§4.2).
    pub binary: Option<String>,
    /// Ordered fallbacks consulted deterministically when the primary
    /// binary is missing (§4.5). No LLM guessing.
    pub fallback_binaries: Vec<String>,
    /// CDP port the harness polls for readiness.
    pub cdp_port: u16,
    /// Seconds to poll the CDP port after launch before reporting FAILED.
    pub launch_timeout_secs: u64,
    /// Extra flags appended on every managed launch.
    pub launch_args: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HarnessConfig {
    /// §3 budget: alert when a single-domain, single-app task exceeds this
    /// many LLM calls. Routine tasks must stay within 4.
    pub max_llm_calls_single_task: usize,
    /// Max scoped-recovery LLM calls per run (genuine deviations only).
    pub max_recoveries: usize,
    /// Max deterministic retries per step before escalating to recovery.
    pub max_step_retries: usize,
}
impl Default for GeneralConfig{fn default()->Self{Self{startup_screen:"mascot".into(),compact_after_command:true}}}
impl Default for ModelConfig{fn default()->Self{Self{
    main:"openchat".into(),
    base_url:None, api_key:None,
    main_api_key:None,
    main_base_url:Some("https://api.openchat.ai/v1".into()),
}}}
impl Default for PlannerConfig{fn default()->Self{Self{max_subtasks:32,max_depth:8,verify_state:true,parallel:true,replan_on_failure:true}}}
impl Default for VoiceConfig{fn default()->Self{Self{provider:"groq".into(),model:"whisper-large-v3-turbo".into(),language:None,push_to_talk:"f2".into(),api_key:None}}}
impl Default for HyprFastConfig{fn default()->Self{Self{command:"hyprfast".into(),args:vec!["mcp".into()],max_candidates:8,batching:true,parallel:true,verify_actions:true}}}
impl Default for AppearanceConfig{fn default()->Self{Self{theme:"lucy".into(),animations:true,activity_verbosity:"normal".into()}}}
impl Default for SessionConfig{fn default()->Self{Self{file:None,dir:None,resume:true,max_history:100}}}
impl Default for ApprovalConfig{fn default()->Self{Self{mode:"write".into()}}}
impl Default for BrowserConfig{fn default()->Self{Self{binary:None,fallback_binaries:vec!["brave".into(),"brave-browser".into(),"google-chrome".into(),"chromium".into(),"chromium-browser".into()],cdp_port:9222,launch_timeout_secs:8,launch_args:vec![]}}}
impl Default for HarnessConfig{fn default()->Self{Self{max_llm_calls_single_task:4,max_recoveries:2,max_step_retries:1}}}
impl Default for LucyConfig{fn default()->Self{Self{general:Default::default(),models:Default::default(),planner:Default::default(),voice:Default::default(),hyprfast:Default::default(),appearance:Default::default(),sessions:Default::default(),approvals:Default::default(),browser:Default::default(),harness:Default::default()}}}
impl LucyConfig {
 pub fn path()->Result<PathBuf>{if let Ok(p)=env::var("LUCY_CONFIG"){return Ok(PathBuf::from(p));}let home=env::var_os("HOME").context("HOME is not set")?;Ok(PathBuf::from(home).join(".config/lucy/config.toml"))}
 pub fn load()->Result<Self>{let path=Self::path()?;let mut cfg=if path.exists(){let text=fs::read_to_string(&path).with_context(||format!("reading {}",path.display()))?;toml::from_str::<Self>(&text).with_context(||format!("parsing {}",path.display()))?}else{Self::default()};cfg.apply_env()?;cfg.validate()?;Ok(cfg)}
 pub fn save(&self)->Result<()>{self.validate()?;let path=Self::path()?;if let Some(parent)=path.parent(){fs::create_dir_all(parent)?;}fs::write(&path,toml::to_string_pretty(self)?)?;Ok(())}
 pub fn init_if_missing(&self)->Result<()>{if !Self::path()?.exists(){self.save()?;}Ok(())}
 pub fn get(&self,key:&str)->Result<String>{let value=toml::Value::try_from(self)?;let mut cur=&value;for part in key.split('.') {cur=cur.get(part).ok_or_else(||anyhow::anyhow!("unknown config key: {key}"))?;}Ok(cur.to_string().trim_matches('"').to_string())}
 pub fn set(&mut self,key:&str,raw:&str)->Result<()>{
        let mut value=toml::Value::try_from(&*self)?;
        let parts:Vec<_>=key.split('.').filter(|p|!p.is_empty()).collect();
        if parts.is_empty(){bail!("empty config key");}
        let mut cur=&mut value;
        for part in &parts[..parts.len()-1]{cur=cur.get_mut(*part).ok_or_else(||anyhow::anyhow!("unknown config key: {key}"))?;}
        let last=parts[parts.len()-1];
        // Handle optional fields that may be None and thus missing from Value — create as String if missing
        let new_val = if let Some(old)=cur.get(last).cloned(){
            parse_value(raw,&old)?
        } else {
            // Missing key (e.g. Option<String> that was None) — treat as string
            toml::Value::String(raw.to_owned())
        };
        if let Some(table)=cur.as_table_mut(){
            table.insert(last.to_string(), new_val);
        } else {
            bail!("config key parent is not a table: {key}");
        }
        *self=value.try_into()?;
        self.validate()
    }
  pub fn reset(&mut self){*self=Self::default();}
   pub fn validate(&self)->Result<()>{if self.planner.max_subtasks==0||self.planner.max_subtasks>256{bail!("planner.max_subtasks must be between 1 and 256");}if self.planner.max_depth==0||self.planner.max_depth>64{bail!("planner.max_depth must be between 1 and 64");}if self.hyprfast.max_candidates==0||self.hyprfast.max_candidates>68{bail!("hyprfast.max_candidates must be between 1 and 68");}if self.sessions.max_history==0{bail!("sessions.max_history must be greater than 0");}if self.browser.cdp_port==0{bail!("browser.cdp_port must be non-zero");}if self.harness.max_llm_calls_single_task==0||self.harness.max_llm_calls_single_task>32{bail!("harness.max_llm_calls_single_task must be between 1 and 32");}match self.approvals.mode.as_str(){"never"|"write"|"always"=>{},_=>bail!("approvals.mode must be never|write|always")}Ok(())}
 fn apply_env(&mut self)->Result<()>{
        if let Some(v)=env::var_os("OPENAI_MODEL"){self.models.main=v.to_string_lossy().into_owned();}
        // Legacy generic base_url
        if let Some(v)=env::var_os("OPENAI_BASE_URL"){self.models.base_url=Some(v.to_string_lossy().into_owned());}
        if self.models.base_url.is_none(){
            if let Some(v)=env::var_os("ANTHROPIC_BASE_URL").or_else(||env::var_os("LLM_BASE_URL")).or_else(||env::var_os("LUCY_BASE_URL")){
                self.models.base_url=Some(v.to_string_lossy().into_owned());
            }
        }
        // Per-model base_url overrides
        if self.models.main_base_url.is_none(){
            if let Ok(v)=env::var("OPENCHAT_BASE_URL").or_else(|_| env::var("LUCY_MAIN_BASE_URL")).or_else(|_| env::var("MAIN_BASE_URL")){
                if !v.trim().is_empty(){ self.models.main_base_url=Some(v); }
            }
        }
        // Generic LLM API key: accept any brand via endpoint + key. Priority: config file > env
        if self.models.api_key.is_none(){
            for key in ["LUCY_API_KEY","OPENAI_API_KEY","ANTHROPIC_API_KEY","GEMINI_API_KEY","LLM_API_KEY","MISTRAL_API_KEY","OPENROUTER_API_KEY"]{
                if let Ok(v)=env::var(key){ if !v.trim().is_empty(){ self.models.api_key=Some(v); break; } }
            }
        }
        // Main-model API key.
        if self.models.main_api_key.is_none(){
            for key in ["LUCY_MAIN_API_KEY","OPENCHAT_API_KEY","OPENAI_API_KEY","MAIN_API_KEY"]{
                if let Ok(v)=env::var(key){ if !v.trim().is_empty(){ self.models.main_api_key=Some(v); break; } }
            }
        }
        // Fallback: if per-model not set but generic is, clone generic
        if self.models.main_api_key.is_none(){ if let Some(v)=self.models.api_key.clone(){ self.models.main_api_key=Some(v); } }
        if self.models.main_base_url.is_none(){ if let Some(v)=self.models.base_url.clone(){ self.models.main_base_url=Some(v); } }

        if self.voice.api_key.is_none(){
            for key in ["GROQ_API_KEY","LUCY_STT_API_KEY","STT_API_KEY"]{
                if let Ok(v)=env::var(key){ if !v.trim().is_empty(){ self.voice.api_key=Some(v); break; } }
            }
        }
        if let Some(v)=env::var_os("LUCY_STT_MODEL"){self.voice.model=v.to_string_lossy().into_owned();}
        if let Some(v)=env::var_os("LUCY_STT_LANGUAGE"){self.voice.language=Some(v.to_string_lossy().into_owned());}
        if let Some(v)=env::var_os("LUCY_HYPRFAST_MAX_CANDIDATES"){self.hyprfast.max_candidates=v.to_string_lossy().parse()?;}
        if let Ok(v)=env::var("LUCY_BROWSER_BINARY"){ if !v.trim().is_empty(){ self.browser.binary=Some(v); } }
        if let Ok(v)=env::var("LUCY_BROWSER_CDP_PORT"){ if !v.trim().is_empty(){ self.browser.cdp_port=v.parse()?; } }
        Ok(())
    }
    /// Returns the effective LLM API key from config or env (any brand) — generic fallback
    pub fn llm_api_key(&self)->Option<String>{
        self.main_api_key().or_else(||{
            self.models.api_key.clone().filter(|v|!v.trim().is_empty()).or_else(||{
                for key in ["LUCY_API_KEY","OPENAI_API_KEY","ANTHROPIC_API_KEY","GEMINI_API_KEY","LLM_API_KEY","MISTRAL_API_KEY","OPENROUTER_API_KEY"]{
                    if let Ok(v)=env::var(key){ if !v.trim().is_empty(){ return Some(v); } }
                }
                None
            })
        })
    }
    pub fn llm_base_url(&self)->Option<String>{
        self.models.base_url.clone().or_else(||{
            env::var("OPENAI_BASE_URL").ok().filter(|v|!v.is_empty()).or_else(||env::var("LLM_BASE_URL").ok()).or_else(||env::var("LUCY_BASE_URL").ok())
        })
    }
    /// Main-model configuration helpers.
    pub fn main_api_key(&self)->Option<String>{
        self.models.main_api_key.clone().filter(|v|!v.trim().is_empty()).or_else(||{
            for key in ["LUCY_MAIN_API_KEY","OPENCHAT_API_KEY","OPENAI_API_KEY"]{
                if let Ok(v)=env::var(key){ if !v.trim().is_empty(){ return Some(v); } }
            }
            self.models.api_key.clone().filter(|v|!v.trim().is_empty())
        })
    }
    pub fn main_base_url(&self)->Option<String>{
        self.models.main_base_url.clone().filter(|v|!v.trim().is_empty()).or_else(||{
            for key in ["OPENCHAT_BASE_URL","LUCY_MAIN_BASE_URL","MAIN_BASE_URL"]{
                if let Ok(v)=env::var(key){ if !v.trim().is_empty(){ return Some(v); } }
            }
            self.models.base_url.clone().filter(|v|!v.trim().is_empty()).or_else(|| self.llm_base_url())
        })
    }
    pub fn stt_api_key(&self)->Option<String>{
        self.voice.api_key.clone().filter(|v|!v.trim().is_empty()).or_else(||{
            for key in ["GROQ_API_KEY","LUCY_STT_API_KEY","STT_API_KEY"]{
                if let Ok(v)=env::var(key){ if !v.trim().is_empty(){ return Some(v); } }
            }
            None
        })
    }
    /// §4.2: resolve the browser binary once from config, never per-run by
    /// the LLM. Order: explicit config → LUCY_BROWSER_BINARY → `which`
    /// over the fallback list → first fallback name as last resort.
    pub fn resolve_browser_binary(&self)->String{
        if let Some(b)=self.browser.binary.clone().filter(|v|!v.trim().is_empty()){
            return b;
        }
        for cand in &self.browser.fallback_binaries{
            if command_exists(cand){ return cand.clone(); }
        }
        self.browser.fallback_binaries.first().cloned().unwrap_or_else(||"brave".into())
    }
    pub fn cdp_port(&self)->u16{ self.browser.cdp_port }
}
fn parse_value(raw:&str,old:&toml::Value)->Result<toml::Value>{if matches!(old,toml::Value::String(_)){return Ok(toml::Value::String(raw.to_owned()));}raw.parse::<toml::Value>().map_err(|e| anyhow::anyhow!("invalid value: {e}"))}
pub fn doctor()->Vec<(&'static str,bool,String)>{
    let config_status = match LucyConfig::load(){
        Ok(c)=>{
            let hypr_ok = command_exists(&c.hyprfast.command);
            let main_key = c.main_api_key();
            let main_ok = main_key.is_some();
            let stt_key = c.stt_api_key();
            let stt_ok = stt_key.is_some();
            let main_endpoint = c.main_base_url().unwrap_or_else(||"https://api.openchat.ai/v1".into());
            let mask = |k:String| if k.len()>8 { format!("{}...{} ({} chars)", &k[..4], &k[k.len()-4..], k.len()) } else { "***".into() };
            let reachability = |endpoint:&str|->String{
                let trunc = |e:String|{let t=e.chars().take(80).collect::<String>();format!(" — unreachable ({t})")};
                let Some((scheme,rest))=endpoint.split_once("://")else{return trunc("invalid url".into())};
                let default_port=match scheme{"https"=>443,"http"=>80,_=>return trunc("unsupported scheme".into())};
                let hostport=rest.split('/').next().unwrap_or("");let hostport=hostport.rsplit('@').next().unwrap_or(hostport);
                let (host,port)=if let Some(br)=hostport.strip_prefix('['){
                    match br.split_once(']'){Some((h,r))=>{let p=r.strip_prefix(':').unwrap_or("").parse::<u16>().unwrap_or(default_port);(h.to_string(),if r.is_empty()||r.starts_with(':')&&r[1..].parse::<u16>().is_ok(){p}else{default_port})},None=>return trunc("invalid host".into())}
                }else if hostport.matches(':').count()>1{(hostport.to_string(),default_port)}
                else{match hostport.rsplit_once(':'){Some((h,p))if !h.is_empty()&&!p.is_empty()=>match p.parse::<u16>(){Ok(n)=>(h.to_string(),n),Err(_)=>(hostport.to_string(),default_port)},_=>(hostport.to_string(),default_port)}};
                if host.is_empty(){return trunc("invalid host".into())}
                use std::net::{TcpStream,ToSocketAddrs};use std::time::Duration;
                let addrs:Vec<_>=match format!("{host}:{port}").to_socket_addrs(){Ok(a)=>a.collect(),Err(e)=>return trunc(e.to_string())};
                if addrs.is_empty(){return trunc("no addresses".into())}
                let mut last_err="connection failed".to_string();for a in addrs{match TcpStream::connect_timeout(&a,Duration::from_secs(3)){Ok(s)=>{drop(s);return " — reachable".into()},Err(e)=>{last_err=e.to_string();}}}
                trunc(last_err)
            };
            let main_reach=reachability(&main_endpoint);
            let main_detail = if main_ok { format!("set {} — endpoint {}", mask(main_key.unwrap()), main_endpoint) } else { "missing — Settings > Main API Key (OpenChat) or env OPENCHAT_API_KEY / LUCY_MAIN_API_KEY".into() };
            vec![
                ("Config",true,LucyConfig::path().map(|p|p.display().to_string()).unwrap_or_default()),
                ("Main model",true,format!("{} (OpenChat)", c.models.main.clone())),
                ("Main Endpoint",true,format!("{main_endpoint}{main_reach}")),
                ("Main API Key",main_ok,main_detail),
                ("HyprFast",hypr_ok,if hypr_ok {c.hyprfast.command.clone()} else {format!("{} (not found - install hyprfast)", c.hyprfast.command)}),
                ("STT API Key",stt_ok,if stt_ok {"set (voice enabled)".into()} else {"not set — voice disabled (set Voice API Key or GROQ_API_KEY)".into()}),
            ]
        },
        Err(e)=>vec![("Config",false,e.to_string())],
    };
    config_status
}
fn command_exists(command:&str)->bool{if command.is_empty(){return false;}std::process::Command::new("sh").args(["-c","command -v -- \"$1\" >/dev/null 2>&1","lucy",command]).status().map(|s|s.success()).unwrap_or(false)}
pub fn config_path()->Result<PathBuf>{LucyConfig::path()}
pub fn config_exists()->Result<bool>{Ok(Path::new(&LucyConfig::path()?).exists())}
