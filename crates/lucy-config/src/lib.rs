use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::{env, fs, path::{Path, PathBuf}};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LucyConfig { pub general: GeneralConfig, pub models: ModelConfig, pub planner: PlannerConfig, pub voice: VoiceConfig, pub hyprfast: HyprFastConfig, pub appearance: AppearanceConfig, pub sessions: SessionConfig }
#[derive(Debug, Clone, Serialize, Deserialize)] #[serde(default)] pub struct GeneralConfig { pub startup_screen:String, pub compact_after_command:bool }
#[derive(Debug, Clone, Serialize, Deserialize)] #[serde(default)] pub struct ModelConfig { pub main:String, #[serde(alias="planner")] pub hyprfast_command:String, pub base_url:Option<String> }
#[derive(Debug, Clone, Serialize, Deserialize)] #[serde(default)] pub struct PlannerConfig { pub max_subtasks:usize, pub max_depth:usize, pub verify_state:bool, pub parallel:bool, pub replan_on_failure:bool }
#[derive(Debug, Clone, Serialize, Deserialize)] #[serde(default)] pub struct VoiceConfig { pub provider:String, pub model:String, pub language:Option<String>, pub push_to_talk:String }
#[derive(Debug, Clone, Serialize, Deserialize)] #[serde(default)] pub struct HyprFastConfig { pub command:String, pub args:Vec<String>, pub max_candidates:usize, pub batching:bool, pub parallel:bool, pub verify_actions:bool }
#[derive(Debug, Clone, Serialize, Deserialize)] #[serde(default)] pub struct AppearanceConfig { pub theme:String, pub animations:bool, pub activity_verbosity:String }
#[derive(Debug, Clone, Serialize, Deserialize)] #[serde(default)] pub struct SessionConfig { pub file:Option<PathBuf>, pub resume:bool, pub max_history:usize }
impl Default for GeneralConfig{fn default()->Self{Self{startup_screen:"mascot".into(),compact_after_command:true}}}
impl Default for ModelConfig{fn default()->Self{Self{main:"gpt-4o".into(),hyprfast_command:"gpt-4o-mini".into(),base_url:None}}}
impl Default for PlannerConfig{fn default()->Self{Self{max_subtasks:32,max_depth:8,verify_state:true,parallel:true,replan_on_failure:true}}}
impl Default for VoiceConfig{fn default()->Self{Self{provider:"groq".into(),model:"whisper-large-v3-turbo".into(),language:None,push_to_talk:"super+c".into()}}}
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
 pub fn set(&mut self,key:&str,raw:&str)->Result<()>{let mut value=toml::Value::try_from(&*self)?;let parts:Vec<_>=key.split('.').filter(|p|!p.is_empty()).collect();if parts.is_empty(){bail!("empty config key");}let mut cur=&mut value;for part in &parts[..parts.len()-1]{cur=cur.get_mut(*part).ok_or_else(||anyhow::anyhow!("unknown config key: {key}"))?;}let last=parts[parts.len()-1];let old=cur.get(last).ok_or_else(||anyhow::anyhow!("unknown config key: {key}"))?.clone();cur[last]=parse_value(raw,&old)?;*self=value.try_into()?;self.validate()}
 pub fn reset(&mut self){*self=Self::default();}
 pub fn validate(&self)->Result<()>{if self.planner.max_subtasks==0||self.planner.max_subtasks>256{bail!("planner.max_subtasks must be between 1 and 256");}if self.planner.max_depth==0||self.planner.max_depth>64{bail!("planner.max_depth must be between 1 and 64");}if self.hyprfast.max_candidates==0||self.hyprfast.max_candidates>68{bail!("hyprfast.max_candidates must be between 1 and 68");}if self.sessions.max_history==0{bail!("sessions.max_history must be greater than 0");}Ok(())}
 fn apply_env(&mut self)->Result<()>{if let Some(v)=env::var_os("OPENAI_MODEL"){self.models.main=v.to_string_lossy().into_owned();}if let Some(v)=env::var_os("LUCY_PLANNER_MODEL"){self.models.hyprfast_command=v.to_string_lossy().into_owned();}if let Some(v)=env::var_os("LUCY_HYPRFAST_COMMAND_MODEL"){self.models.hyprfast_command=v.to_string_lossy().into_owned();}if let Some(v)=env::var_os("OPENAI_BASE_URL"){self.models.base_url=Some(v.to_string_lossy().into_owned());}if let Some(v)=env::var_os("LUCY_STT_MODEL"){self.voice.model=v.to_string_lossy().into_owned();}if let Some(v)=env::var_os("LUCY_STT_LANGUAGE"){self.voice.language=Some(v.to_string_lossy().into_owned());}if let Some(v)=env::var_os("LUCY_HYPRFAST_MAX_CANDIDATES"){self.hyprfast.max_candidates=v.to_string_lossy().parse()?;}Ok(())}
}
fn parse_value(raw:&str,old:&toml::Value)->Result<toml::Value>{if matches!(old,toml::Value::String(_)){return Ok(toml::Value::String(raw.to_owned()));}raw.parse::<toml::Value>().map_err(|e| anyhow::anyhow!("invalid value: {e}"))}
pub fn doctor()->Vec<(&'static str,bool,String)>{match LucyConfig::load(){Ok(c)=>vec![("Config",true,LucyConfig::path().map(|p|p.display().to_string()).unwrap_or_default()),("Main model",true,c.models.main),("HyprFast command model",true,c.models.hyprfast_command),("HyprFast",command_exists(&c.hyprfast.command),c.hyprfast.command)],Err(e)=>vec![("Config",false,e.to_string())]}}
fn command_exists(command:&str)->bool{if command.is_empty(){return false;}std::process::Command::new("sh").args(["-c","command -v -- \"$1\" >/dev/null 2>&1","lucy",command]).status().map(|s|s.success()).unwrap_or(false)}
pub fn config_path()->Result<PathBuf>{LucyConfig::path()}
pub fn config_exists()->Result<bool>{Ok(Path::new(&LucyConfig::path()?).exists())}
