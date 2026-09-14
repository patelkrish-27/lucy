//! Hierarchical planner: main-model triage, cheap-model command compiler,
//! main-model closed-loop execution with verification and recovery.

use std::collections::{HashMap, VecDeque};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use lucy_agent::OpenAIProvider;
use lucy_core::{InterruptSignal, TurnMessage};
use lucy_hyprfast::{HyprFastCatalog, Route};

use super::MAX_CONTEXT_CHARS;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct SubTask {
    pub(crate) id: String,
    pub(crate) goal: String,
    #[serde(default)]
    pub(crate) category: String,
    #[serde(default)]
    pub(crate) depends_on: Vec<String>,
    #[serde(default)]
    pub(crate) success_condition: Option<String>,
    #[serde(default)]
    pub(crate) required_observation: Option<String>,
    #[serde(default)]
    pub(crate) constraints: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Triage {
    #[serde(default)] pub(crate) mode: String,
    #[serde(default)] pub(crate) reply: Option<String>,
    #[serde(default)] pub(crate) subtasks: Vec<SubTask>,
}
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct PlannedCommand {
    pub(crate) tool: String,
    pub(crate) arguments: Value,
    #[serde(default)] pub(crate) verify: Option<String>,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DecisionKind { Continue, Replan, Complete }
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct MainDecision {
    pub(crate) decision: DecisionKind,
    #[serde(default)] pub(crate) subtask: Option<SubTask>,
    #[serde(default)] pub(crate) reason: String,
}

async fn parse_with_retry<T>(provider:&OpenAIProvider,model:&str,system:&str,user:&str,interrupt:&InterruptSignal)->Result<T>
where T: for<'de> Deserialize<'de> {
    let v=provider.complete_json(model,system,user,interrupt.clone()).await?;
    match serde_json::from_value::<T>(v){Ok(t)=>Ok(t),Err(e)=>{
        let retry=format!("{user}\n\nYour previous output was invalid JSON: {e}. Return ONLY valid JSON matching the requested schema.");
        Ok(serde_json::from_value(provider.complete_json(model,system,&retry,interrupt.clone()).await?)?)
    }}
}

pub(crate) async fn triage_request(provider:&OpenAIProvider,model:&str,prompt:&str,catalog:&HyprFastCatalog,route:&Option<Route>,history:&[TurnMessage],interrupt:&InterruptSignal)->Result<Triage>{
    let route_context=route.as_ref().map(|r|catalog.context_for(r)).unwrap_or_default();
    let recent_history=history.iter().rev().take(8).map(|m|format!("{:?}",m)).collect::<Vec<_>>().join("\n");
    let user=format!("New request:\n{prompt}\n\nRecent session context:\n{recent_history}\n\nInitial tool context:\n{route_context}");
    validate_triage(parse_with_retry(provider,model,super::prompts::TRIAGE,&user,interrupt).await?)
}

fn validate_triage(t:Triage)->Result<Triage>{
    let mode=t.mode.to_ascii_lowercase();
    if mode!="chat"&&mode!="act"{return Err(anyhow!("main model returned unknown triage mode: {}",t.mode));}
    if mode=="act"&&t.subtasks.is_empty(){return Err(anyhow!("main model chose act but supplied no subtasks"));}
    for s in &t.subtasks { validate_subtask(s)?; }
    Ok(Triage{mode,reply:t.reply,subtasks:t.subtasks})
}

fn validate_subtask(s:&SubTask)->Result<()> {
    if s.id.trim().is_empty(){return Err(anyhow!("subtask id must not be empty"));}
    if s.goal.trim().is_empty(){return Err(anyhow!("subtask {} has an empty goal",s.id));}
    if s.category.trim().is_empty(){return Err(anyhow!("subtask {} has an empty category",s.id));}
    if s.success_condition.as_deref().map(str::trim).unwrap_or("").is_empty(){return Err(anyhow!("subtask {} must define success_condition",s.id));}
    if s.required_observation.as_deref().map(str::trim).unwrap_or("").is_empty(){return Err(anyhow!("subtask {} must define required_observation",s.id));}
    Ok(())
}

pub(crate) async fn decide_next(provider:&OpenAIProvider,model:&str,prompt:&str,remaining:&VecDeque<SubTask>,completed:&HashMap<String,Value>,last_subtask:&SubTask,last_result:&Value,catalog:&HyprFastCatalog,interrupt:&InterruptSignal)->Result<MainDecision>{
    let user=format!("Task:\n{prompt}\n\nLast subtask:\n{}\nLast result/state:\n{last_result}\n\nRemaining planned subtasks:\n{}\n\nAll completed results/state:\n{}\n\nHyprFast capability summary:\n{}",serde_json::to_string(last_subtask)?,serde_json::to_string(remaining)?,serde_json::to_string(completed)?,serde_json::to_string(&catalog.summary())?);
    let mut d:MainDecision=parse_with_retry(provider,model,super::prompts::CONTROLLER,&user,interrupt).await?;
    if let Some(s)=&d.subtask{validate_subtask(s)?;}
    if matches!(d.decision,DecisionKind::Continue)&&d.subtask.is_some(){return Err(anyhow!("continue decision must not include a subtask"));}
    if matches!(d.decision,DecisionKind::Replan)&&d.subtask.is_none(){return Err(anyhow!("replan decision must include a subtask"));}
    if d.reason.trim().is_empty(){d.reason="State evaluated.".into();}
    Ok(d)
}

pub(crate) async fn plan_command(provider:&OpenAIProvider,model:&str,prompt:&str,subtask:&SubTask,schemas:&[Value],context:&str,interrupt:&InterruptSignal)->Result<PlannedCommand>{
    let tools=serde_json::to_string(schemas)?;
    let user=format!("Original task (context only):\n{prompt}\n\nSubtask:\n{}\n\nCurrent context:\n{context}\n\nAllowed tools and schemas:\n{tools}",serde_json::to_string(subtask)?);
    parse_with_retry(provider,model,super::prompts::HYPRFAST,&user,interrupt).await
}

pub(crate) fn build_context(catalog:&HyprFastCatalog,route:&Route,completed:&HashMap<String,Value>,depends_on:&[String])->String{
    let mut out=catalog.context_for(route);for id in depends_on{if let Some(v)=completed.get(id){out.push_str(&format!("\nDependency {id} result: {v}"));}}
    if out.len()>MAX_CONTEXT_CHARS{out.truncate(MAX_CONTEXT_CHARS);out.push_str("\n[context truncated]");}out
}
pub(crate) fn truncate_json(v:Value)->Value{let s=v.to_string();if s.len()<=MAX_CONTEXT_CHARS{return v;}let end=s.char_indices().nth(MAX_CONTEXT_CHARS).map(|(i,_)|i).unwrap_or(s.len());serde_json::json!({"truncated":true,"preview":&s[..end]})}
pub(crate) fn action_requires_verification(catalog:&HyprFastCatalog,tool:&str,_category:&str,explicit_verify:bool)->bool{if explicit_verify{return true;}catalog.capability_for_mcp_name(tool).map(|c|!c.read_only).unwrap_or(true)}
fn verification_category(catalog:&HyprFastCatalog,tool:&str,fallback:&str)->String{catalog.capability_for_mcp_name(tool).map(|c|match c.domain{lucy_hyprfast::Domain::Browser|lucy_hyprfast::Domain::Stagehand|lucy_hyprfast::Domain::Hints=>"browser",lucy_hyprfast::Domain::Excalidraw=>"excalidraw",lucy_hyprfast::Domain::Vision=>"vision",_=>"desktop"}).unwrap_or(fallback).to_owned()}
pub(crate) fn verification_subtask(catalog:&HyprFastCatalog,tool:&str,original_goal:&str,fallback_category:&str,id:usize)->SubTask{SubTask{id:format!("verify-{id}"),goal:format!("Observe and verify the current state after: {original_goal}. Confirm whether the intended change actually happened; do not make another change."),category:verification_category(catalog,tool,fallback_category),depends_on:Vec::new(),success_condition:Some("The observed state proves whether the intended change happened.".into()),required_observation:Some("Observe the current state without changing it.".into()),constraints:vec!["read-only; do not modify state".into()]}}

#[cfg(test)]
mod tests{use super::*;#[test]fn subtask_requires_structured_success_and_observation(){let s=SubTask{id:"1".into(),goal:"Open browser".into(),category:"browser".into(),depends_on:vec![],success_condition:Some("Browser is open".into()),required_observation:Some("Observe browser state".into()),constraints:vec![]};assert!(validate_subtask(&s).is_ok());}#[test]fn legacy_subtask_is_rejected(){let s=SubTask{id:"1".into(),goal:"Open browser".into(),category:"browser".into(),depends_on:vec![],success_condition:None,required_observation:None,constraints:vec![]};assert!(validate_subtask(&s).is_err());}}
