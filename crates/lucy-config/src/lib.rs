use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::{env, fs, path::{Path, PathBuf}};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LucyConfig { pub general: GeneralConfig, pub models: ModelConfig, pub planner: PlannerConfig, pub voice: VoiceConfig, pub hyprfast: HyprFastConfig, pub appearance: AppearanceConfig, pub sessions: SessionConfig }
#[derive(Debug, Clone, Serialize, Deserialize)] #[serde(default)] pub struct GeneralConfig { pub startup_screen:String, pub compact_after_command:bool }
#[derive(Debug, Clone, Serialize, Deserialize)] #[serde(default)] pub struct ModelConfig {
    pub main:String,
    #[serde(alias="planner")] pub hyprfast_command:String,
    // Legacy generic (fallback for both)
    pub base_url:Option<String>,
    #[serde(default, skip_serializing_if="Option::is_none")] pub api_key:Option<String>,
    // Per-model overrides — allows OpenChat for main and Gemini for cheap
    #[serde(default, skip_serializing_if="Option::is_none")] pub main_api_key:Option<String>,
    #[serde(default, skip_serializing_if="Option::is_none")] pub main_base_url:Option<String>,
    #[serde(default, skip_serializing_if="Option::is_none")] pub cheap_api_key:Option<String>,
    #[serde(default, skip_serializing_if="Option::is_none")] pub cheap_base_url:Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)] #[serde(default)] pub struct PlannerConfig { pub max_subtasks:usize, pub max_depth:usize, pub verify_state:bool, pub parallel:bool, pub replan_on_failure:bool }
#[derive(Debug, Clone, Serialize, Deserialize)] #[serde(default)] pub struct VoiceConfig { pub provider:String, pub model:String, pub language:Option<String>, pub push_to_talk:String, #[serde(default, skip_serializing_if="Option::is_none")] pub api_key:Option<String> }
#[derive(Debug, Clone, Serialize, Deserialize)] #[serde(default)] pub struct HyprFastConfig { pub command:String, pub args:Vec<String>, pub max_candidates:usize, pub batching:bool, pub parallel:bool, pub verify_actions:bool }
#[derive(Debug, Clone, Serialize, Deserialize)] #[serde(default)] pub struct AppearanceConfig { pub theme:String, pub animations:bool, pub activity_verbosity:String }
#[derive(Debug, Clone, Serialize, Deserialize)] #[serde(default)] pub struct SessionConfig { pub file:Option<PathBuf>, pub resume:bool, pub max_history:usize }
impl Default for GeneralConfig{fn default()->Self{Self{startup_screen:"mascot".into(),compact_after_command:true}}}
impl Default for ModelConfig{fn default()->Self{Self{
    main:"openchat".into(),
    hyprfast_command:"gemini-3.5-flash-lite".into(),
    base_url:None, api_key:None,
    main_api_key:None,
    main_base_url:Some("https://api.openchat.ai/v1".into()),
    cheap_api_key:None,
    cheap_base_url:Some("https://generativelanguage.googleapis.com/v1beta/openai/".into()),
}}}
impl Default for PlannerConfig{fn default()->Self{Self{max_subtasks:32,max_depth:8,verify_state:true,parallel:true,replan_on_failure:true}}}
impl Default for VoiceConfig{fn default()->Self{Self{provider:"groq".into(),model:"whisper-large-v3-turbo".into(),language:None,push_to_talk:"super+c".into(),api_key:None}}}
impl Default for HyprFastConfig{fn default()->Self{Self{command:"hyprfast".into(),args:vec!["mcp".into()],max_candidates:8,batching:true,parallel:true,verify_actions:true}}}
impl Default for AppearanceConfig{fn default()->Self{Self{theme:"lucy".into(),animations:true,activity_verbosity:"normal".into()}}}
impl Default for SessionConfig{fn default()->Self{Self{file:None,resume:true,max_history:100}}}
impl Default for LucyConfig{fn default()->Self{Self{general:Default::default(),models:Default::default(),planner:Default::default(),voice:Default::default(),hyprfast:Default::default(),appearance:Default::default(),sessions:Default::default()}}}
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
 pub fn validate(&self)->Result<()>{if self.planner.max_subtasks==0||self.planner.max_subtasks>256{bail!("planner.max_subtasks must be between 1 and 256");}if self.planner.max_depth==0||self.planner.max_depth>64{bail!("planner.max_depth must be between 1 and 64");}if self.hyprfast.max_candidates==0||self.hyprfast.max_candidates>68{bail!("hyprfast.max_candidates must be between 1 and 68");}if self.sessions.max_history==0{bail!("sessions.max_history must be greater than 0");}Ok(())}
 fn apply_env(&mut self)->Result<()>{
        if let Some(v)=env::var_os("OPENAI_MODEL"){self.models.main=v.to_string_lossy().into_owned();}
        if let Some(v)=env::var_os("LUCY_PLANNER_MODEL"){self.models.hyprfast_command=v.to_string_lossy().into_owned();}
        if let Some(v)=env::var_os("LUCY_HYPRFAST_COMMAND_MODEL"){self.models.hyprfast_command=v.to_string_lossy().into_owned();}
        if let Some(v)=env::var_os("GEMINI_MODEL"){self.models.hyprfast_command=v.to_string_lossy().into_owned();}
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
        if self.models.cheap_base_url.is_none(){
            if let Ok(v)=env::var("GEMINI_BASE_URL").or_else(|_| env::var("LUCY_CHEAP_BASE_URL")).or_else(|_| env::var("CHEAP_BASE_URL")).or_else(|_| env::var("LUCY_HYPRFAST_BASE_URL")){
                if !v.trim().is_empty(){ self.models.cheap_base_url=Some(v); }
            }
        }
        // Generic LLM API key: accept any brand via endpoint + key. Priority: config file > env
        if self.models.api_key.is_none(){
            for key in ["LUCY_API_KEY","OPENAI_API_KEY","ANTHROPIC_API_KEY","GEMINI_API_KEY","LLM_API_KEY","MISTRAL_API_KEY","OPENROUTER_API_KEY"]{
                if let Ok(v)=env::var(key){ if !v.trim().is_empty(){ self.models.api_key=Some(v); break; } }
            }
        }
        // Per-model API keys: OpenChat for main, Gemini for cheap
        if self.models.main_api_key.is_none(){
            for key in ["LUCY_MAIN_API_KEY","OPENCHAT_API_KEY","OPENAI_API_KEY","MAIN_API_KEY"]{
                if let Ok(v)=env::var(key){ if !v.trim().is_empty(){ self.models.main_api_key=Some(v); break; } }
            }
        }
        if self.models.cheap_api_key.is_none(){
            for key in ["LUCY_CHEAP_API_KEY","GEMINI_API_KEY","CHEAP_API_KEY","LUCY_HYPRFAST_API_KEY","HYPRFAST_API_KEY"]{
                if let Ok(v)=env::var(key){ if !v.trim().is_empty(){ self.models.cheap_api_key=Some(v); break; } }
            }
        }
        // Fallback: if per-model not set but generic is, clone generic
        if self.models.main_api_key.is_none(){ if let Some(v)=self.models.api_key.clone(){ self.models.main_api_key=Some(v); } }
        if self.models.cheap_api_key.is_none(){ if let Some(v)=self.models.api_key.clone(){ self.models.cheap_api_key=Some(v); } }
        if self.models.main_base_url.is_none(){ if let Some(v)=self.models.base_url.clone(){ self.models.main_base_url=Some(v); } }
        if self.models.cheap_base_url.is_none(){ if let Some(v)=self.models.base_url.clone(){ self.models.cheap_base_url=Some(v); } }

        if self.voice.api_key.is_none(){
            for key in ["GROQ_API_KEY","LUCY_STT_API_KEY","STT_API_KEY"]{
                if let Ok(v)=env::var(key){ if !v.trim().is_empty(){ self.voice.api_key=Some(v); break; } }
            }
        }
        if let Some(v)=env::var_os("LUCY_STT_MODEL"){self.voice.model=v.to_string_lossy().into_owned();}
        if let Some(v)=env::var_os("LUCY_STT_LANGUAGE"){self.voice.language=Some(v.to_string_lossy().into_owned());}
        if let Some(v)=env::var_os("LUCY_HYPRFAST_MAX_CANDIDATES"){self.hyprfast.max_candidates=v.to_string_lossy().parse()?;}
        Ok(())
    }
    /// Returns the effective LLM API key from config or env (any brand) — generic fallback
    pub fn llm_api_key(&self)->Option<String>{
        self.main_api_key().or_else(|| self.cheap_api_key()).or_else(||{
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
    /// Per-model helpers — main (OpenChat) and cheap (Gemini)
    pub fn main_api_key(&self)->Option<String>{
        self.models.main_api_key.clone().filter(|v|!v.trim().is_empty()).or_else(||{
            for key in ["LUCY_MAIN_API_KEY","OPENCHAT_API_KEY","OPENAI_API_KEY"]{
                if let Ok(v)=env::var(key){ if !v.trim().is_empty(){ return Some(v); } }
            }
            self.models.api_key.clone().filter(|v|!v.trim().is_empty())
        })
    }
    pub fn cheap_api_key(&self)->Option<String>{
        self.models.cheap_api_key.clone().filter(|v|!v.trim().is_empty()).or_else(||{
            for key in ["LUCY_CHEAP_API_KEY","GEMINI_API_KEY","CHEAP_API_KEY"]{
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
    pub fn cheap_base_url(&self)->Option<String>{
        self.models.cheap_base_url.clone().filter(|v|!v.trim().is_empty()).or_else(||{
            for key in ["GEMINI_BASE_URL","LUCY_CHEAP_BASE_URL","CHEAP_BASE_URL","LUCY_HYPRFAST_BASE_URL"]{
                if let Ok(v)=env::var(key){ if !v.trim().is_empty(){ return Some(v); } }
            }
            self.models.base_url.clone().filter(|v|!v.trim().is_empty()).or_else(|| self.llm_base_url()).or_else(|| Some("https://generativelanguage.googleapis.com/v1beta/openai/".into()))
        })
    }
    pub fn api_key_for(&self, model:&str)->Option<String>{
        if model == self.models.main { self.main_api_key() } else if model == self.models.hyprfast_command { self.cheap_api_key() } else { self.llm_api_key() }
    }
    pub fn base_url_for(&self, model:&str)->Option<String>{
        if model == self.models.main { self.main_base_url() } else if model == self.models.hyprfast_command { self.cheap_base_url() } else { self.llm_base_url() }
    }
    pub fn stt_api_key(&self)->Option<String>{
        self.voice.api_key.clone().filter(|v|!v.trim().is_empty()).or_else(||{
            for key in ["GROQ_API_KEY","LUCY_STT_API_KEY","STT_API_KEY"]{
                if let Ok(v)=env::var(key){ if !v.trim().is_empty(){ return Some(v); } }
            }
            None
        })
    }
}
fn parse_value(raw:&str,old:&toml::Value)->Result<toml::Value>{if matches!(old,toml::Value::String(_)){return Ok(toml::Value::String(raw.to_owned()));}raw.parse::<toml::Value>().map_err(|e| anyhow::anyhow!("invalid value: {e}"))}
pub fn doctor()->Vec<(&'static str,bool,String)>{
    let config_status = match LucyConfig::load(){
        Ok(c)=>{
            let hypr_ok = command_exists(&c.hyprfast.command);
            let main_key = c.main_api_key();
            let cheap_key = c.cheap_api_key();
            let main_ok = main_key.is_some();
            let cheap_ok = cheap_key.is_some();
            let stt_key = c.stt_api_key();
            let stt_ok = stt_key.is_some();
            let main_endpoint = c.main_base_url().unwrap_or_else(||"https://api.openchat.ai/v1".into());
            let cheap_endpoint = c.cheap_base_url().unwrap_or_else(||"https://generativelanguage.googleapis.com/v1beta/openai/".into());
            let mask = |k:String| if k.len()>8 { format!("{}...{} ({} chars)", &k[..4], &k[k.len()-4..], k.len()) } else { "***".into() };
            let main_detail = if main_ok { format!("set {} — endpoint {}", mask(main_key.unwrap()), main_endpoint) } else { "missing — Settings > Main API Key (OpenChat) or env OPENCHAT_API_KEY / LUCY_MAIN_API_KEY".into() };
            let cheap_detail = if cheap_ok { format!("set {} — endpoint {}", mask(cheap_key.unwrap()), cheap_endpoint) } else { "missing — Settings > Cheap API Key (Gemini) or env GEMINI_API_KEY / LUCY_CHEAP_API_KEY".into() };
            vec![
                ("Config",true,LucyConfig::path().map(|p|p.display().to_string()).unwrap_or_default()),
                ("Main model",true,format!("{} (OpenChat)", c.models.main.clone())),
                ("Main Endpoint",true,main_endpoint),
                ("Main API Key",main_ok,main_detail),
                ("Cheap model",true,format!("{} (Gemini 3.5 Flash-Lite)", c.models.hyprfast_command.clone())),
                ("Cheap Endpoint",true,cheap_endpoint),
                ("Cheap API Key",cheap_ok,cheap_detail),
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
