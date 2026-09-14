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
    #[serde(default)] pub(crate) category: String,
    #[serde(default)] pub(crate) depends_on: Vec<String>,
    #[serde(default)] pub(crate) success_condition: Option<String>,
    #[serde(default)] pub(crate) required_observation: Option<String>,
    #[serde(default)] pub(crate) constraints: Vec<String>,
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

#[derive(Debug, Default)]
struct ContextBuilder { sections: Vec<(String, String)> }
impl ContextBuilder {
    fn section(mut self, name: impl Into<String>, value: impl Into<String>) -> Self { let value=value.into(); if !value.trim().is_empty(){self.sections.push((name.into(),value));} self }
    fn finish(self)->String{let mut out=String::new();for(name,value)in self.sections{let block=format!("\n## {name}\n{value}\n");if out.len()+block.len()>MAX_CONTEXT_CHARS{let remaining=MAX_CONTEXT_CHARS.saturating_sub(out.len());if remaining>32{out.push_str(&block[..remaining.min(block.len())]);}out.push_str("\n[context truncated]");break;}out.push_str(&block);}out}
}
fn recent_history(history:&[TurnMessage],limit:usize)->String{history.iter().rev().take(limit).rev().map(|m|format!("{:?}",m)).collect::<Vec<_>>().join("\n")}
fn full_hyprfast_index(catalog:&HyprFastCatalog)->String{let mut tools:Vec<_>=catalog.tools.values().collect();tools.sort_by(|a,b|a.name.cmp(&b.name));let mut out=String::new();for tool in tools{let caps=tool.capabilities.iter().map(|c|format!("{:?}",c)).collect::<Vec<_>>().join(",");out.push_str(&format!("{} | domain={:?} | op={:?} | caps=[{}] | read_only={} | destructive={} | batchable={} | {}\n",tool.name,tool.domain,tool.operation,caps,tool.read_only,tool.destructive,tool.batchable,tool.description));}out}

async fn parse_with_retry<T>(provider:&OpenAIProvider,model:&str,system:&str,user:&str,interrupt:&InterruptSignal)->Result<T> where T:for<'de>Deserialize<'de>{let v=provider.complete_json(model,system,user,interrupt.clone()).await?;match serde_json::from_value::<T>(v){Ok(t)=>Ok(t),Err(e)=>{let retry=format!("{user}\n\nYour previous output was invalid JSON: {e}. Return ONLY valid JSON matching the requested schema.");Ok(serde_json::from_value(provider.complete_json(model,system,&retry,interrupt.clone()).await?)?)}}}

pub(crate) async fn triage_request(provider:&OpenAIProvider,model:&str,prompt:&str,catalog:&HyprFastCatalog,route:&Option<Route>,history:&[TurnMessage],interrupt:&InterruptSignal)->Result<Triage>{let route_context=route.as_ref().map(|r|catalog.context_for(r)).unwrap_or_default();let user=ContextBuilder::default().section("CURRENT USER REQUEST",prompt).section("RECENT SESSION HISTORY (context only)",recent_history(history,8)).section("CURRENT CAPABILITY ROUTE",route_context).section("PLANNING RULE","Plan from the desired outcome. Current observations and tool evidence outrank assumptions and stale history.").finish();validate_triage(parse_with_retry(provider,model,super::prompts::TRIAGE,&user,interrupt).await?)}

fn validate_triage(t:Triage)->Result<Triage>{let mode=t.mode.to_ascii_lowercase();if mode!="chat"&&mode!="act"{return Err(anyhow!("main model returned unknown triage mode: {}",t.mode));}if mode=="act"&&t.subtasks.is_empty(){return Err(anyhow!("main model chose act but supplied no subtasks"));}if mode=="act"{validate_plan(&t.subtasks)?;}Ok(Triage{mode,reply:t.reply,subtasks:t.subtasks})}
fn validate_subtask(s:&SubTask)->Result<()> {if s.id.trim().is_empty(){return Err(anyhow!("subtask id must not be empty"));}if s.goal.trim().is_empty(){return Err(anyhow!("subtask {} has an empty goal",s.id));}if s.category.trim().is_empty(){return Err(anyhow!("subtask {} has an empty category",s.id));}if s.success_condition.as_deref().map(str::trim).unwrap_or("").is_empty(){return Err(anyhow!("subtask {} must define success_condition",s.id));}if s.required_observation.as_deref().map(str::trim).unwrap_or("").is_empty(){return Err(anyhow!("subtask {} must define required_observation",s.id));}Ok(())}

/// Validate the complete plan before any tool is allowed to run.
/// Dependencies must point to known tasks, IDs must be unique, and the graph
/// must be acyclic. Execution can therefore safely consume tasks in any order.
pub(crate) fn validate_plan(tasks:&[SubTask])->Result<()> {let mut ids=std::collections::HashSet::new();for task in tasks{validate_subtask(task)?;if !ids.insert(task.id.as_str()){return Err(anyhow!("duplicate subtask id: {}",task.id));}}for task in tasks{for dep in &task.depends_on{if !ids.contains(dep.as_str()){return Err(anyhow!("subtask {} depends on unknown task {}",task.id,dep));}if dep==&task.id{return Err(anyhow!("subtask {} depends on itself",task.id));}}}if super::has_dependency_cycle(tasks){return Err(anyhow!("subtask plan contains a dependency cycle"));}Ok(())}

pub(crate) async fn decide_next(provider:&OpenAIProvider,model:&str,prompt:&str,remaining:&VecDeque<SubTask>,completed:&HashMap<String,Value>,last_subtask:&SubTask,last_result:&Value,catalog:&HyprFastCatalog,interrupt:&InterruptSignal)->Result<MainDecision>{let completed_json=serde_json::to_string(completed)?;let remaining_json=serde_json::to_string(remaining)?;let last_json=serde_json::to_string(last_subtask)?;let user=ContextBuilder::default().section("ORIGINAL USER GOAL",prompt).section("LAST SUBTASK",last_json).section("LAST EXECUTION RESULT / OBSERVATION (highest-priority live evidence)",last_result.to_string()).section("REMAINING PLAN",remaining_json).section("COMPLETED RESULTS / OBSERVATIONS",completed_json).section("AVAILABLE CAPABILITIES",serde_json::to_string(&catalog.summary())?).section("CONTROLLER RULE","Decide complete only when the user outcome is supported by evidence. Continue only if the existing plan remains valid. Replan when state, information, or assumptions differ.").finish();let mut d:MainDecision=parse_with_retry(provider,model,super::prompts::CONTROLLER,&user,interrupt).await?;if let Some(s)=&d.subtask{validate_subtask(s)?;}if matches!(d.decision,DecisionKind::Continue)&&d.subtask.is_some(){return Err(anyhow!("continue decision must not include a subtask"));}if matches!(d.decision,DecisionKind::Replan)&&d.subtask.is_none(){return Err(anyhow!("replan decision must include a subtask"));}if d.reason.trim().is_empty(){d.reason="State evaluated.".into();}Ok(d)}

pub(crate) async fn plan_command(provider:&OpenAIProvider,model:&str,prompt:&str,subtask:&SubTask,schemas:&[Value],context:&str,interrupt:&InterruptSignal)->Result<PlannedCommand>{let tools=serde_json::to_string(schemas)?;let user=ContextBuilder::default().section("ORIGINAL USER TASK (context only; do not reinterpret)",prompt).section("EXACT SUBTASK TO COMPILE",serde_json::to_string(subtask)?).section("CURRENT EXECUTION CONTEXT / OBSERVATIONS",context).section("ALLOWED TOOLS AND EXACT SCHEMAS",tools).section("COMPILER RULE","Use only supplied evidence. If required state is missing and an allowed observation tool exists, choose observation rather than guessing.").finish();parse_with_retry(provider,model,super::prompts::HYPRFAST,&user,interrupt).await}

pub(crate) fn build_context(catalog:&HyprFastCatalog,route:&Route,completed:&HashMap<String,Value>,depends_on:&[String])->String{let mut builder=ContextBuilder::default().section("FULL HYPRFAST CAPABILITY INDEX (all discovered commands; metadata only)",full_hyprfast_index(catalog)).section("CURRENT ROUTED CAPABILITIES WITH EXECUTABLE SCHEMAS",catalog.context_for(route));for id in depends_on{if let Some(v)=completed.get(id){builder=builder.section(format!("DEPENDENCY RESULT: {id}"),truncate_json(v.clone()).to_string());}}builder.finish()}
pub(crate) fn truncate_json(v:Value)->Value{let s=v.to_string();if s.len()<=MAX_CONTEXT_CHARS{return v;}let end=s.char_indices().nth(MAX_CONTEXT_CHARS).map(|(i,_)|i).unwrap_or(s.len());serde_json::json!({"truncated":true,"preview":&s[..end]})}
pub(crate) fn action_requires_verification(catalog:&HyprFastCatalog,tool:&str,_category:&str,explicit_verify:bool)->bool{if explicit_verify{return true;}catalog.capability_for_mcp_name(tool).map(|c|!c.read_only).unwrap_or(true)}
fn verification_category(catalog:&HyprFastCatalog,tool:&str,fallback:&str)->String{catalog.capability_for_mcp_name(tool).map(|c|match c.domain{lucy_hyprfast::Domain::Browser|lucy_hyprfast::Domain::Stagehand|lucy_hyprfast::Domain::Hints=>"browser",lucy_hyprfast::Domain::Excalidraw=>"excalidraw",lucy_hyprfast::Domain::Vision=>"vision",_=>"desktop"}).unwrap_or(fallback).to_owned()}
pub(crate) fn verification_subtask(catalog:&HyprFastCatalog,tool:&str,original_goal:&str,fallback_category:&str,id:usize)->SubTask{SubTask{id:format!("verify-{id}"),goal:format!("Observe and verify the current state after: {original_goal}. Confirm whether the intended change actually happened; do not make another change."),category:verification_category(catalog,tool,fallback_category),depends_on:Vec::new(),success_condition:Some("The observed state proves whether the intended change happened.".into()),required_observation:Some("Observe the current state without changing it.".into()),constraints:vec!["read-only; do not modify state".into()]}}

#[cfg(test)]
mod tests{use super::*;fn task(id:&str,deps:&[&str])->SubTask{SubTask{id:id.into(),goal:id.into(),category:"browser".into(),depends_on:deps.iter().map(|s|(*s).into()).collect(),success_condition:Some("succeeded".into()),required_observation:Some("observe".into()),constraints:vec![]}}#[test]fn subtask_requires_structured_success_and_observation(){let s=task("1",&[]);assert!(validate_subtask(&s).is_ok());}#[test]fn legacy_subtask_is_rejected(){let mut s=task("1",&[]);s.success_condition=None;s.required_observation=None;assert!(validate_subtask(&s).is_err());}#[test]fn rejects_unknown_dependency(){let s=task("a",&["missing"]);assert!(validate_plan(&[s]).is_err());}#[test]fn rejects_cycles(){let a=task("a",&["b"]);let b=task("b",&["a"]);assert!(validate_plan(&[a,b]).is_err());}#[test]fn accepts_out_of_order_acyclic_plan(){let a=task("a",&["b"]);let b=task("b",&[]);assert!(validate_plan(&[a,b]).is_ok());}}
