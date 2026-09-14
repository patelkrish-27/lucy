//! Hierarchical planner: main-model triage, cheap-model command compiler,
//! main-model closed-loop execution with verification and recovery.
//!
//! Every user command flows through here (see `LucyRuntime::submit`):
//!  1. the main model triages `chat` (answer directly) vs `act` (subtasks),
//!  2. the cheap model compiles each subtask to exactly one command,
//!  3. the main model executes, verifies, recovers and finally summarizes.

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
}
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Triage {
    #[serde(default)]
    pub(crate) mode: String,
    #[serde(default)]
    pub(crate) reply: Option<String>,
    #[serde(default)]
    pub(crate) subtasks: Vec<SubTask>,
}
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct PlannedCommand {
    pub(crate) tool: String,
    pub(crate) arguments: Value,
    #[serde(default)]
    pub(crate) verify: Option<String>,
}
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DecisionKind {
    Continue,
    Replan,
    Complete,
}
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct MainDecision {
    pub(crate) decision: DecisionKind,
    #[serde(default)]
    pub(crate) subtask: Option<SubTask>,
    #[serde(default)]
    pub(crate) reason: String,
}

async fn parse_with_retry<T>(
    provider: &OpenAIProvider,
    model: &str,
    system: &str,
    user: &str,
    interrupt: &InterruptSignal,
) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    let v = provider
        .complete_json(model, system, user, interrupt.clone())
        .await?;
    match serde_json::from_value::<T>(v) {
        Ok(t) => Ok(t),
        Err(e) => {
            let retry_user = format!("{user}\n\nYour previous output was invalid JSON: {e}. Return ONLY valid JSON matching the requested schema.");
            let v2 = provider
                .complete_json(model, system, &retry_user, interrupt.clone())
                .await?;
            Ok(serde_json::from_value::<T>(v2)?)
        }
    }
}

/// Step 1 — main-model triage over system prompt + history + new request:
/// answer directly (`chat`) or break the task down (`act`).
pub(crate) async fn triage_request(
    provider: &OpenAIProvider,
    model: &str,
    prompt: &str,
    catalog: &HyprFastCatalog,
    route: &Option<Route>,
    history: &[TurnMessage],
    interrupt: &InterruptSignal,
) -> Result<Triage> {
    let route_context = route
        .as_ref()
        .map(|r| catalog.context_for(r))
        .unwrap_or_default();
    let recent_history = history
        .iter()
        .rev()
        .take(8)
        .map(|m| format!("{:?}", m))
        .collect::<Vec<_>>()
        .join("\n");
    let system = super::prompts::TRIAGE;
    let user=format!("New request:\n{}\n\nRecent session context:\n{}\n\nInitial tool context:\n{}",prompt,recent_history,route_context);
    let t: Triage = parse_with_retry(provider, model, system, &user, interrupt).await?;
    validate_triage(t)
}

fn validate_triage(t: Triage) -> Result<Triage> {
    let mode = t.mode.to_ascii_lowercase();
    if mode != "chat" && mode != "act" {
        return Err(anyhow!(
            "main model returned unknown triage mode: {}",
            t.mode
        ));
    }
    if mode == "act" && t.subtasks.is_empty() {
        return Err(anyhow!("main model chose act but supplied no subtasks"));
    }
    Ok(Triage {
        mode,
        reply: t.reply,
        subtasks: t.subtasks,
    })
}

/// Step 3 — main-model closed-loop control after each command: confirm the
/// work, continue the plan, or replan with one recovery subtask.
pub(crate) async fn decide_next(
    provider: &OpenAIProvider,
    model: &str,
    prompt: &str,
    remaining: &VecDeque<SubTask>,
    completed: &HashMap<String, Value>,
    last_subtask: &SubTask,
    last_result: &Value,
    catalog: &HyprFastCatalog,
    interrupt: &InterruptSignal,
) -> Result<MainDecision> {
    let remaining_json=serde_json::to_string(remaining)?;let completed_json=serde_json::to_string(completed)?;let system = super::prompts::CONTROLLER;
    let user=format!("Task:\n{}\n\nLast subtask:\n{}\nLast result/state:\n{}\n\nRemaining planned subtasks:\n{}\n\nAll completed results/state:\n{}\n\nHyprFast capability summary:\n{}",prompt,serde_json::to_string(last_subtask)?,last_result,remaining_json,completed_json,serde_json::to_string(&catalog.summary())?);
    parse_with_retry(provider, model, system, &user, interrupt).await
}

/// Step 2 — cheap-model command compiler: one planned subtask becomes
/// exactly one tool call within the routed capability set.
pub(crate) async fn plan_command(
    provider: &OpenAIProvider,
    model: &str,
    prompt: &str,
    subtask: &SubTask,
    schemas: &[Value],
    context: &str,
    interrupt: &InterruptSignal,
) -> Result<PlannedCommand> {
    let system = super::prompts::HYPRFAST;
    let tools=serde_json::to_string(schemas)?;let user=format!("Original task (context only):\n{}\n\nSubtask category: {}\nSubtask: {}\nDependencies: {:?}\n\nCurrent HyprFast context:\n{}\n\nAllowed tools and schemas:\n{}",prompt,subtask.category,subtask.goal,subtask.depends_on,context,tools);parse_with_retry(provider,model,system,&user,interrupt).await
}

pub(crate) fn build_context(
    catalog: &HyprFastCatalog,
    route: &Route,
    completed: &HashMap<String, Value>,
    depends_on: &[String],
) -> String {
    let mut out = catalog.context_for(route);
    for id in depends_on {
        if let Some(v) = completed.get(id) {
            out.push_str(&format!("\nDependency {} result: {}", id, v));
        }
    }
    if out.len() > MAX_CONTEXT_CHARS {
        out.truncate(MAX_CONTEXT_CHARS);
        out.push_str("\n[context truncated]");
    }
    out
}

pub(crate) fn truncate_json(v: Value) -> Value {
    let s = v.to_string();
    if s.len() <= MAX_CONTEXT_CHARS {
        return v;
    }
    let end = s
        .char_indices()
        .nth(MAX_CONTEXT_CHARS)
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    serde_json::json!({"truncated":true,"preview":&s[..end]})
}

pub(crate) fn action_requires_verification(
    catalog: &HyprFastCatalog,
    tool: &str,
    category: &str,
    explicit_verify: bool,
) -> bool {
    if explicit_verify {
        return true;
    }
    let Some(capability) = catalog.capability_for_mcp_name(tool) else {
        return true;
    };
    if capability.read_only {
        return false;
    }
    let _ = category;
    true
}

fn verification_category(catalog: &HyprFastCatalog, tool: &str, fallback: &str) -> String {
    catalog
        .capability_for_mcp_name(tool)
        .map(|capability| match capability.domain {
            lucy_hyprfast::Domain::Browser
            | lucy_hyprfast::Domain::Stagehand
            | lucy_hyprfast::Domain::Hints => "browser".to_owned(),
            lucy_hyprfast::Domain::Excalidraw => "excalidraw".to_owned(),
            lucy_hyprfast::Domain::Vision => "vision".to_owned(),
            lucy_hyprfast::Domain::Desktop
            | lucy_hyprfast::Domain::Tasks
            | lucy_hyprfast::Domain::Clipboard
            | lucy_hyprfast::Domain::System
            | lucy_hyprfast::Domain::Unknown => "desktop".to_owned(),
        })
        .unwrap_or_else(|| fallback.to_owned())
}

pub(crate) fn verification_subtask(
    catalog: &HyprFastCatalog,
    tool: &str,
    original_goal: &str,
    fallback_category: &str,
    id: usize,
) -> SubTask {
    SubTask{id:format!("verify-{id}"),goal:format!("Observe and verify the current state after: {original_goal}. Confirm whether the intended change actually happened; do not make another change."),category:verification_category(catalog,tool,fallback_category),depends_on:Vec::new()}
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucy_mcp::McpToolDefinition;
    fn tool(name: &str, description: &str) -> McpToolDefinition {
        McpToolDefinition {
            name: name.into(),
            description: Some(description.into()),
            input_schema: serde_json::json!({"type":"object"}),
        }
    }
    #[test]
    fn state_changing_tool_requires_verification() {
        let catalog = HyprFastCatalog::from_tools(vec![
            tool("browser_click", "click browser element"),
            tool("screenshot", "capture desktop"),
        ]);
        assert!(action_requires_verification(
            &catalog,
            "mcp_hyprfast_browser_click",
            "browser",
            false
        ));
        assert!(!action_requires_verification(
            &catalog,
            "mcp_hyprfast_screenshot",
            "vision",
            false
        ));
        assert!(action_requires_verification(
            &catalog,
            "unknown_tool",
            "browser",
            false
        ));
    }
    #[test]
    fn verification_uses_browser_domain_for_browser_actions() {
        let catalog = HyprFastCatalog::from_tools(vec![tool(
            "browser_click",
            "click browser element",
        )]);
        let subtask = verification_subtask(
            &catalog,
            "mcp_hyprfast_browser_click",
            "click the button",
            "desktop",
            7,
        );
        assert_eq!(subtask.id, "verify-7");
        assert_eq!(subtask.category, "browser");
        assert!(subtask.goal.contains("do not make another change"));
    }
    #[test]
    fn triage_accepts_chat_and_act_modes() {
        let chat: Triage = serde_json::from_value(
            serde_json::json!({"mode":"chat","reply":"Hello! How can I help?"}),
        )
        .expect("chat triage parses");
        let chat = validate_triage(chat).expect("chat validates");
        assert_eq!(chat.mode, "chat");
        assert!(chat.subtasks.is_empty());
        let act: Triage = serde_json::from_value(
            serde_json::json!({"mode":"act","subtasks":[{"id":"1","goal":"Open the browser","category":"browser"}]}),
        )
        .expect("act triage parses");
        let act = validate_triage(act).expect("act validates");
        assert_eq!(act.mode, "act");
        assert_eq!(act.subtasks.len(), 1);
        assert_eq!(act.subtasks[0].category, "browser");
    }
    #[test]
    fn triage_rejects_unknown_mode_and_empty_act() {
        let unknown: Triage =
            serde_json::from_value(serde_json::json!({"mode":"dance"})).expect("parses");
        assert!(validate_triage(unknown).is_err());
        let empty: Triage =
            serde_json::from_value(serde_json::json!({"mode":"act","subtasks":[]}))
                .expect("parses");
        assert!(validate_triage(empty).is_err());
    }
    #[test]
    fn files_shell_subtasks_use_local_tools() {
        use lucy_tools::default_registry;
        let registry = default_registry();
        let local = registry.local_tool_names();
        assert!(local.contains("shell"));
        assert!(local.contains("read_file"));
        assert!(!local.iter().any(|n| n.starts_with("mcp_")));
        let schemas = registry.definitions_for_names(&local);
        assert_eq!(schemas.len(), local.len());
        assert!(!schemas.is_empty());
    }
}
