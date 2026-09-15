//! Native ADK-Rust integration for Lucy.
//!
//! Lucy keeps its latency-sensitive agent loop, HyprFast command compiler,
//! approvals, and MCP routing as the authoritative execution path. This crate
//! brings in the complementary ADK-Rust capabilities that Lucy did not already
//! implement, behind an intentionally small facade.

mod execution;
mod session;
pub use execution::{to_adk_event, LucyExecution, LUCY_APP_NAME};
pub use session::{LucySessionService, LUCY_SESSION_APP};

use std::{env, path::{Path, PathBuf}, sync::Arc};
use adk_core::{Content, Part};
use adk_memory::{MemoryEntry, MemoryService, SearchRequest, SqliteMemoryService};
use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};

pub use adk_audio;
pub use adk_core;
pub use adk_memory;
pub use adk_rust;
pub use adk_telemetry;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdkCapability { PersistentSemanticMemory, SessionServiceBackends, WorkflowAgents, GraphWorkflows, Artifacts, Guardrails, Skills, Plugins, CodeExecution, SandboxedExecution, ComputerUse, RealtimeAgents, AudioPipelines, Evaluation, Telemetry, A2aServer, Authentication, ExpandedModelProviders, McpSamplingAndTransports }
impl AdkCapability { pub const fn as_str(self)->&'static str { match self { Self::PersistentSemanticMemory=>"persistent-semantic-memory", Self::SessionServiceBackends=>"session-service-backends", Self::WorkflowAgents=>"workflow-agents", Self::GraphWorkflows=>"graph-workflows", Self::Artifacts=>"artifacts", Self::Guardrails=>"guardrails", Self::Skills=>"skills", Self::Plugins=>"plugins", Self::CodeExecution=>"code-execution", Self::SandboxedExecution=>"sandboxed-execution", Self::ComputerUse=>"computer-use", Self::RealtimeAgents=>"realtime-agents", Self::AudioPipelines=>"audio-pipelines", Self::Evaluation=>"evaluation", Self::Telemetry=>"telemetry", Self::A2aServer=>"a2a-server", Self::Authentication=>"authentication", Self::ExpandedModelProviders=>"expanded-model-providers", Self::McpSamplingAndTransports=>"mcp-sampling-and-transports" } } }
pub const ADK_CAPABILITIES:&[AdkCapability]=&[AdkCapability::PersistentSemanticMemory,AdkCapability::SessionServiceBackends,AdkCapability::WorkflowAgents,AdkCapability::GraphWorkflows,AdkCapability::Artifacts,AdkCapability::Guardrails,AdkCapability::Skills,AdkCapability::Plugins,AdkCapability::CodeExecution,AdkCapability::SandboxedExecution,AdkCapability::ComputerUse,AdkCapability::RealtimeAgents,AdkCapability::AudioPipelines,AdkCapability::Evaluation,AdkCapability::Telemetry,AdkCapability::A2aServer,AdkCapability::Authentication,AdkCapability::ExpandedModelProviders,AdkCapability::McpSamplingAndTransports];

const DEFAULT_MEMORY_RESULTS:usize=6;
const MAX_MEMORY_CONTEXT_CHARS:usize=6_000;
const MAX_MEMORY_FACT_CHARS:usize=1_200;
const MAX_MEMORY_CANDIDATES:usize=4;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryCandidate { pub kind:String, pub text:String }

pub struct LucyAdk { memory:Option<Arc<SqliteMemoryService>>, memory_path:PathBuf }
impl std::fmt::Debug for LucyAdk { fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result { f.debug_struct("LucyAdk").field("memory_enabled",&self.memory.is_some()).field("memory_path",&self.memory_path).finish() } }
impl LucyAdk {
    pub async fn open(state_dir:impl AsRef<Path>)->Self {
        let default_path=state_dir.as_ref().join("adk-memory.db");
        let memory_path=env::var_os("LUCY_ADK_MEMORY_DB").map(PathBuf::from).unwrap_or(default_path);
        if let Some(parent)=memory_path.parent(){if let Err(error)=tokio::fs::create_dir_all(parent).await{tracing::warn!(error=%error,path=%parent.display(),"ADK memory directory unavailable");return Self{memory:None,memory_path};}}
        let sqlite_url=format!("sqlite://{}",memory_path.display());
        match SqliteMemoryService::new(&sqlite_url).await{Ok(service)=>{if let Err(error)=service.migrate().await{tracing::warn!(error=%error,"ADK SQLite memory migration failed; disabling persistent memory");return Self{memory:None,memory_path};}tracing::info!(path=%memory_path.display(),"ADK persistent memory enabled");Self{memory:Some(Arc::new(service)),memory_path}},Err(error)=>{tracing::warn!(error=%error,path=%memory_path.display(),"ADK SQLite memory unavailable; continuing without persistent memory");Self{memory:None,memory_path}}}
    }
    pub fn capabilities(&self)->&'static [AdkCapability]{ADK_CAPABILITIES}
    pub fn memory_enabled(&self)->bool{self.memory.is_some()}
    pub fn memory_path(&self)->&Path{&self.memory_path}

    pub async fn memory_context(&self,query:&str,limit:usize)->Result<String>{
        if query.trim().is_empty()||self.memory.is_none(){return Ok(String::new());}
        let memories=self.search_memory(query,limit.max(1)).await?;
        let superseded:Vec<String>=memories.iter().flat_map(|e|superseded_texts(&content_text(&e.content))).collect();
        let mut out=String::new();let mut index=0usize;
        for entry in memories.iter(){let text=content_text(&entry.content);if text.trim().is_empty()||is_tombstone(&text){continue;}if superseded.iter().any(|s|normalize_memory_text(s)==normalize_memory_text(&text)){continue;}index+=1;let block=format!("\n- Memory {} [{}]: {}\n",index,entry.timestamp.to_rfc3339(),text.trim());if out.len()+block.len()>MAX_MEMORY_CONTEXT_CHARS{break;}out.push_str(&block);}
        Ok(out)
    }

    pub async fn remember_interaction(&self,_session_id:&str,prompt:&str,response:&str)->Result<()>{let candidates=curate_interaction(prompt,response);if candidates.is_empty(){return Ok(());}self.remember_candidates(&candidates).await}

    pub async fn remember_candidates(&self,candidates:&[MemoryCandidate])->Result<()>{
        let Some(memory)=&self.memory else{return Ok(());};let user_id=local_user_id();let mut entries=Vec::new();
        for candidate in candidates.iter().take(MAX_MEMORY_CANDIDATES){let text=normalize_memory_text(&candidate.text);if !safe_durable_text(&text)||text.chars().count()>MAX_MEMORY_FACT_CHARS{continue;}let kind=normalize_kind(&candidate.kind);if kind.is_empty(){continue;}let stored=format!("[{kind}] {text}");if self.memory_contains_exact(memory,&user_id,&stored).await?{continue;}entries.push(MemoryEntry{content:Content::new("memory").with_text(stored),author:"lucy-memory".to_owned(),timestamp:Utc::now()});}
        if entries.is_empty(){return Ok(());}memory.add_session("lucy",&user_id,"curated-memory",entries).await.context("storing curated ADK memory")
    }

    pub async fn remember_fact(&self,fact:&str)->Result<()>{let fact=normalize_memory_text(fact);if !safe_durable_text(&fact){return Ok(());}self.remember_candidates(&[MemoryCandidate{kind:"fact".to_owned(),text:fact}]).await}

    pub async fn forget_memory(&self,query:&str)->Result<()>{
        let Some(memory)=&self.memory else{return Ok(());};let query=query.trim();if query.is_empty(){return Ok(());}let user_id=local_user_id();let found=self.search_memory(query,DEFAULT_MEMORY_RESULTS).await?;let mut entries=Vec::new();
        for entry in found{let text=content_text(&entry.content);if text.trim().is_empty()||is_tombstone(&text){continue;}entries.push(MemoryEntry{content:Content::new("memory").with_text(format!("[tombstone] {}",text.trim())),author:"lucy-memory".to_owned(),timestamp:Utc::now()});}
        if entries.is_empty(){return Ok(());}memory.add_session("lucy",&user_id,"memory-tombstones",entries).await.context("storing memory tombstone")
    }

    pub async fn search_memory(&self,query:&str,limit:usize)->Result<Vec<MemoryEntry>>{let Some(memory)=&self.memory else{return Ok(Vec::new());};let response=memory.search(SearchRequest{query:query.to_owned(),user_id:local_user_id(),app_name:"lucy".to_owned(),limit:Some(limit.max(1)),min_score:None,project_id:None}).await.context("searching ADK memory")?;Ok(response.memories)}
    async fn memory_contains_exact(&self,memory:&SqliteMemoryService,user_id:&str,stored:&str)->Result<bool>{let response=memory.search(SearchRequest{query:stored.to_owned(),user_id:user_id.to_owned(),app_name:"lucy".to_owned(),limit:Some(DEFAULT_MEMORY_RESULTS),min_score:None,project_id:None}).await.context("checking ADK memory for duplicate")?;Ok(response.memories.iter().any(|entry|content_text(&entry.content).trim().eq_ignore_ascii_case(stored)))}
}

fn content_text(content:&Content)->String{content.parts.iter().filter_map(Part::text).collect::<Vec<_>>().join(" ")}
fn normalize_kind(kind:&str)->String{match kind.trim().to_ascii_lowercase().as_str(){"fact"=>"fact".to_owned(),"preference"=>"preference".to_owned(),"decision"=>"decision".to_owned(),"project"=>"project".to_owned(),"instruction"=>"instruction".to_owned(),_=>String::new()}}
fn normalize_memory_text(text:&str)->String{text.split_whitespace().collect::<Vec<_>>().join(" ")}

fn safe_durable_text(text:&str)->bool{
    if text.trim().is_empty(){return false;}
    let lower=text.to_ascii_lowercase();
    ["api key","apikey","password","secret","private key","access token","bearer token","refresh token","seed phrase"]
        .iter()
        .all(|needle| !lower.contains(needle))
}

fn is_tombstone(text:&str)->bool{text.trim_start().to_ascii_lowercase().starts_with("[tombstone]")}
fn superseded_texts(text:&str)->Vec<String>{let Some(start)=text.find("[supersedes:") else{return Vec::new();};let rest=&text[start+12..];let Some(end)=rest.rfind(']') else{return Vec::new();};rest[..end].split(" | ").map(normalize_memory_text).filter(|s|!s.is_empty()).collect()}

fn curate_interaction(prompt:&str,_response:&str)->Vec<MemoryCandidate>{
    let text=normalize_memory_text(prompt);
    if text.is_empty()||text.chars().count()>MAX_MEMORY_FACT_CHARS{return Vec::new();}
    let lower=text.to_ascii_lowercase();
    if is_transient(&lower){return Vec::new();}
    let(kind,explicit)=if contains_any(&lower,&["remember that","remember this","don't forget","do not forget","from now on","always "]){("instruction",true)}else if contains_any(&lower,&["i prefer ","i'd prefer ","i would prefer ","i like ","i love ","i dislike ","i hate ","i don't like ","my preference "]){("preference",true)}else if contains_any(&lower,&["i decided ","we decided ","let's use ","we'll use ","i chose ","i choose ","the plan is "]){("decision",true)}else if contains_any(&lower,&["i'm working on ","i am working on ","my project ","i use ","i'm using ","i am using ","my name is ","call me ","i live in ","i work in ","i study "]){("fact",true)}else{("fact",false)};
    if !explicit{return Vec::new();}
    vec![MemoryCandidate{kind:kind.to_owned(),text}]
}

fn is_transient(lower:&str)->bool{
    let trimmed=lower.trim_start();
    contains_any(lower,&["what is ","what's ","who is ","who's ","how do i ","how can i ","can you ","could you ","please open ","open ","close ","search for ","look up ","show me ","tell me ","what time ","weather","thanks","thank you","hello","hi ","hey "])
        || trimmed.starts_with("run ")
}
fn contains_any(text:&str,needles:&[&str])->bool{needles.iter().any(|needle|text.contains(needle))}
fn local_user_id()->String{env::var("LUCY_USER_ID").or_else(|_|env::var("USER")).unwrap_or_else(|_|"local".to_owned())}

#[cfg(test)]
mod tests{use super::*;
#[test]fn capability_manifest_is_stable(){assert!(ADK_CAPABILITIES.contains(&AdkCapability::PersistentSemanticMemory));assert!(ADK_CAPABILITIES.contains(&AdkCapability::GraphWorkflows));assert!(ADK_CAPABILITIES.contains(&AdkCapability::SandboxedExecution));assert!(ADK_CAPABILITIES.contains(&AdkCapability::Evaluation));assert_eq!(AdkCapability::Telemetry.as_str(),"telemetry");assert!(!ADK_CAPABILITIES.iter().any(|cap|cap.as_str()=="browser-automation"));}
#[test]fn local_user_has_a_safe_fallback(){assert!(!local_user_id().trim().is_empty());}
#[test]fn memory_context_budget_is_reasonable(){assert!(MAX_MEMORY_CONTEXT_CHARS>=1_000);}
#[test]fn transient_requests_are_not_saved(){assert!(curate_interaction("open the browser and search for Rust docs","Done").is_empty());assert!(curate_interaction("what is the capital of France?","Paris").is_empty());}
#[test]fn durable_preferences_and_facts_are_saved(){let preference=curate_interaction("I prefer concise answers in the terminal","Got it");assert_eq!(preference[0].kind,"preference");let fact=curate_interaction("I use Rust for my main projects","Great");assert_eq!(fact[0].kind,"fact");}
#[test]fn explicit_instructions_are_saved(){let memory=curate_interaction("Remember that I always want tests run before merging","Understood");assert_eq!(memory[0].kind,"instruction");}
#[test]fn duplicate_memory_is_stable(){let candidate=MemoryCandidate{kind:"fact".into(),text:"I use Rust".into()};assert_eq!(normalize_kind(&candidate.kind),"fact");assert!(normalize_kind("unknown").is_empty());}
#[test]fn supersession_parser_hides_old_text(){let text="[preference] I prefer concise answers [supersedes: [preference] I prefer detailed answers]";let values=superseded_texts(text);assert_eq!(values.len(),1);assert_eq!(values[0],"[preference] I prefer detailed answers");}
#[test]fn secret_filter_blocks_credentials(){assert!(!safe_durable_text("my password is abc"));assert!(!safe_durable_text("my API key is abc"));assert!(safe_durable_text("I prefer Rust"));}
#[test]fn tombstones_are_recognized(){assert!(is_tombstone("[tombstone] [fact] old"));}
}