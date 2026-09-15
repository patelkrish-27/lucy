//! Unified execution primitives shared by Lucy and ADK-Rust.
//!
//! This is the first step of the deep merge: Lucy remains responsible for
//! planning and capability policy, while ADK becomes the canonical identity
//! and event representation at the runtime boundary.

use adk_core::{AdkIdentity, AppName, Content, Event, ExecutionIdentity, InvocationId, Part, UserId};
use lucy_core::{AgentEvent, SessionId, TurnMessage};

pub const LUCY_APP_NAME: &str = "lucy";

#[derive(Debug, Clone)]
pub struct LucyExecution {
    pub identity: ExecutionIdentity,
}

impl LucyExecution {
    pub fn new(session_id: &SessionId, user_id: impl Into<String>) -> Self {
        let user = user_id.into();
        let session = session_id.0.to_string();
        let adk = AdkIdentity::new(
            AppName::try_from(LUCY_APP_NAME).expect("static Lucy app name is valid"),
            UserId::try_from(user).expect("Lucy user id must be non-empty and valid"),
            adk_core::SessionId::try_from(session).expect("Lucy session id is valid"),
        );
        Self {
            identity: ExecutionIdentity {
                adk,
                invocation_id: InvocationId::generate(),
                branch: String::new(),
                agent_name: "lucy".to_owned(),
            },
        }
    }

    pub fn session_id(&self) -> &str { self.identity.adk.session_id.as_ref() }
    pub fn invocation_id(&self) -> &str { self.identity.invocation_id.as_ref() }
}

/// Translate Lucy's existing event stream into canonical ADK events.
pub fn to_adk_event(execution: &LucyExecution, event: &AgentEvent) -> Option<Event> {
    match event {
        AgentEvent::History { message } => history_to_event(execution, message),
        AgentEvent::TextDelta { text } => {
            let mut adk = Event::with_id(format!("lucy-stream-{}", execution.invocation_id()), execution.invocation_id());
            adk.author = "lucy".to_owned();
            adk.llm_response.partial = true;
            adk.set_content(Content::new("model").with_text(text));
            Some(adk)
        }
        AgentEvent::ToolStarted { id, name, input } => {
            let mut adk = Event::new(execution.invocation_id());
            adk.author = "lucy".to_owned();
            adk.set_content(Content {
                role: "model".to_owned(),
                parts: vec![Part::FunctionCall {
                    name: name.clone(), args: input.clone(), id: Some(id.clone()), thought_signature: None,
                }],
            });
            Some(adk)
        }
        AgentEvent::ToolFinished { id, name, output, is_error } => {
            let mut adk = Event::new(execution.invocation_id());
            adk.author = "lucy".to_owned();
            let response = if *is_error { serde_json::json!({ "error": output }) } else { output.clone() };
            adk.set_content(Content {
                role: "function".to_owned(),
                parts: vec![Part::FunctionResponse {
                    function_response: adk_core::FunctionResponseData::new(name.clone(), response),
                    id: Some(id.clone()),
                    annotations: None,
                }],
            });
            Some(adk)
        }
        AgentEvent::Thinking { text } => {
            let mut adk = Event::new(execution.invocation_id());
            adk.author = "lucy".to_owned();
            adk.set_content(Content {
                role: "model".to_owned(),
                parts: vec![Part::Thinking { thinking: text.clone(), signature: None }],
            });
            Some(adk)
        }
        AgentEvent::ApprovalRequest { id, name, input } => {
            let mut adk = Event::new(execution.invocation_id());
            adk.author = "lucy".to_owned();
            adk.actions.tool_confirmation = Some(adk_core::ToolConfirmationRequest {
                tool_name: name.clone(), args: input.clone(), function_call_id: Some(id.clone()),
            });
            Some(adk)
        }
        AgentEvent::Error { message } => {
            let mut adk = Event::new(execution.invocation_id());
            adk.author = "lucy".to_owned();
            adk.set_content(Content::new("tool").with_text(message));
            Some(adk)
        }
        AgentEvent::Status { .. } | AgentEvent::Progress { .. } | AgentEvent::Done => None,
    }
}

fn history_to_event(execution: &LucyExecution, message: &TurnMessage) -> Option<Event> {
    let mut event = Event::new(execution.invocation_id());
    match message {
        TurnMessage::User(text) => {
            event.author = "user".to_owned();
            event.set_content(Content::new("user").with_text(text));
        }
        TurnMessage::Assistant(turn) => {
            event.author = "lucy".to_owned();
            let mut content = Content::new("model");
            if let Some(text) = &turn.text { content = content.with_text(text); }
            for call in &turn.tool_calls {
                content.parts.push(Part::FunctionCall {
                    name: call.name.clone(), args: call.input.clone(), id: Some(call.id.clone()), thought_signature: None,
                });
            }
            if content.parts.is_empty() { return None; }
            event.set_content(content);
        }
        TurnMessage::Tool(result) => {
            event.author = "tool".to_owned();
            event.set_content(Content {
                role: "function".to_owned(),
                parts: vec![Part::FunctionResponse {
                    function_response: adk_core::FunctionResponseData::new(result.name.clone(), result.output.clone()),
                    id: Some(result.call_id.clone()),
                    annotations: None,
                }],
            });
        }
    }
    Some(event)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn creates_stable_adk_identity_per_execution() {
        let execution = LucyExecution::new(&SessionId::default(), "local");
        assert_eq!(execution.identity.adk.app_name.as_ref(), LUCY_APP_NAME);
        assert_eq!(execution.identity.agent_name, "lucy");
        assert!(!execution.session_id().is_empty());
        assert!(!execution.invocation_id().is_empty());
    }

    #[test]
    fn maps_user_history_to_adk_event() {
        let execution = LucyExecution::new(&SessionId::default(), "local");
        let event = to_adk_event(&execution, &AgentEvent::History { message: TurnMessage::User("hello".into()) }).unwrap();
        assert_eq!(event.author, "user");
        assert_eq!(event.content().unwrap().parts[0].text(), Some("hello"));
        assert_eq!(event.invocation_id, execution.invocation_id());
    }

    #[test]
    fn maps_tool_results_to_function_response() {
        let execution = LucyExecution::new(&SessionId::default(), "local");
        let event = to_adk_event(&execution, &AgentEvent::ToolFinished { id: "call-1".into(), name: "demo".into(), output: Value::String("ok".into()), is_error: false }).unwrap();
        assert_eq!(event.author, "lucy");
        assert!(matches!(event.content().unwrap().parts[0], Part::FunctionResponse { .. }));
    }
}