//! Wire protocol between the Lucy mobile app and the desktop gateway.
//!
//! One JSON object per WebSocket text frame, tagged by `type`. The protocol is
//! deliberately narrow: the phone can start one task, watch what the agent
//! does, answer approval prompts, stop a run, and browse sessions. Everything
//! else (planning, tool choice, verification) stays on the desktop.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use lucy_core::{SessionData, SessionMeta};

/// Bump when a breaking change lands. The app reads it from `hello` and can
/// show "update the app" instead of failing in a confusing way.
pub const PROTOCOL_VERSION: u32 = 1;

/// Client → server.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    /// Prove possession of a device token, or redeem a one-time pairing token.
    Auth {
        token: String,
        /// Human-readable device name (e.g. "Krish's Pixel").
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// Start a task. `id` is client-chosen and echoed back; when omitted the
    /// server generates one.
    Task {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        prompt: String,
    },
    /// Answer an `approval_request`.
    Approval {
        id: String,
        decision: ApprovalDecision,
    },
    /// Kill switch: interrupt the running task and deny anything still pending.
    Stop,
    Ping,
    #[serde(rename = "sessions_list")]
    SessionsList,
    #[serde(rename = "sessions_get")]
    SessionsGet { id: String },
    #[serde(rename = "sessions_new")]
    SessionsNew {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },
    #[serde(rename = "sessions_switch")]
    SessionsSwitch { id: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalDecision {
    AllowOnce,
    AllowAlways,
    Deny,
}

impl From<ApprovalDecision> for lucy_core::ApprovalDecision {
    fn from(d: ApprovalDecision) -> Self {
        match d {
            ApprovalDecision::AllowOnce => lucy_core::ApprovalDecision::AllowOnce,
            ApprovalDecision::AllowAlways => lucy_core::ApprovalDecision::AllowAlways,
            ApprovalDecision::Deny => lucy_core::ApprovalDecision::Deny,
        }
    }
}

/// Server → client.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    /// First frame on every connection, before any auth has happened. Carries
    /// just enough for the app to decide whether this is the Lucy it paired
    /// with and whether it needs to pair again.
    Hello {
        protocol: u32,
        server: String,
        version: String,
        /// True when at least one device has been paired with this gateway.
        paired: bool,
        /// True when the heavy runtime (MCP children, browser plumbing) is
        /// currently loaded. The app can show "computer idle" vs "loaded".
        runtime_loaded: bool,
    },
    AuthOk {
        device: String,
        /// Present only when a one-time pairing token was redeemed: the
        /// long-lived token the app must now store. Never sent again.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        device_token: Option<String>,
    },
    Error {
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        code: Option<String>,
    },
    Pong,
    TaskStarted {
        task_id: String,
        prompt: String,
    },
    Status {
        message: String,
    },
    Progress {
        message: String,
    },
    Thinking {
        text: String,
    },
    ToolStarted {
        id: String,
        name: String,
        input: Value,
    },
    ToolFinished {
        id: String,
        name: String,
        output: Value,
        is_error: bool,
    },
    ApprovalRequest {
        id: String,
        name: String,
        input: Value,
    },
    TextDelta {
        text: String,
    },
    /// Terminal frame for one task. `complete` is the verifier's/harness's own
    /// verdict, not merely "the run stopped".
    TaskDone {
        task_id: String,
        summary: String,
        complete: bool,
    },
    Sessions {
        items: Vec<SessionMeta>,
        /// The session new tasks will run in.
        current: Option<String>,
    },
    Session {
        session: Box<SessionData>,
    },
}

impl ServerMessage {
    pub fn to_json(&self) -> String {
        // Every variant is plain data; serialization cannot fail.
        serde_json::to_string(self).unwrap_or_else(|e| {
            format!(
                r#"{{"type":"error","message":"failed to encode server message: {e}"}}"#
            )
        })
    }
}

/// One-shot pairing payload encoded in the QR code (and printed as text).
///
/// The app stores `token` only until it redeems it; what it keeps afterwards is
/// the `device_token` returned by `AuthOk`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairingPayload {
    /// Payload format version, so a future QR is not misread.
    pub lucy: u32,
    /// Desktop hostname, for display.
    pub name: String,
    /// WebSocket endpoint, e.g. `ws://192.168.1.20:9847/ws`.
    pub url: String,
    /// REST base, e.g. `http://192.168.1.20:9847`.
    pub http: String,
    /// One-time pairing token.
    pub token: String,
    /// Stable id of this desktop gateway (survives restarts).
    pub server_id: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn client_messages_round_trip_by_type_tag() {
        let cases = [
            (
                json!({"type":"task","prompt":"open the browser"}),
                ClientMessage::Task {
                    id: None,
                    prompt: "open the browser".into(),
                },
            ),
            (
                json!({"type":"task","id":"t1","prompt":"x"}),
                ClientMessage::Task {
                    id: Some("t1".into()),
                    prompt: "x".into(),
                },
            ),
            (
                json!({"type":"approval","id":"a1","decision":"allow_once"}),
                ClientMessage::Approval {
                    id: "a1".into(),
                    decision: ApprovalDecision::AllowOnce,
                },
            ),
            (json!({"type":"stop"}), ClientMessage::Stop),
            (json!({"type":"ping"}), ClientMessage::Ping),
            (json!({"type":"sessions_list"}), ClientMessage::SessionsList),
            (
                json!({"type":"sessions_get","id":"s1"}),
                ClientMessage::SessionsGet { id: "s1".into() },
            ),
            (
                json!({"type":"sessions_new"}),
                ClientMessage::SessionsNew { title: None },
            ),
            (
                json!({"type":"sessions_switch","id":"s1"}),
                ClientMessage::SessionsSwitch { id: "s1".into() },
            ),
            (
                json!({"type":"auth","token":"abc"}),
                ClientMessage::Auth {
                    token: "abc".into(),
                    name: None,
                },
            ),
        ];
        for (value, expected) in cases {
            let parsed: ClientMessage = serde_json::from_value(value.clone())
                .unwrap_or_else(|e| panic!("{value} should parse: {e}"));
            assert_eq!(parsed, expected, "parsed {value}");
            // And the enum serializes back to the same tag.
            let back = serde_json::to_value(&parsed).unwrap();
            assert_eq!(back["type"], value["type"]);
        }
    }

    #[test]
    fn all_three_approval_decisions_cross_the_wire() {
        for (wire, decision) in [
            ("allow_once", ApprovalDecision::AllowOnce),
            ("allow_always", ApprovalDecision::AllowAlways),
            ("deny", ApprovalDecision::Deny),
        ] {
            let text = format!(r#"{{"type":"approval","id":"a","decision":"{wire}"}}"#);
            let msg: ClientMessage = serde_json::from_str(&text).unwrap();
            assert_eq!(
                msg,
                ClientMessage::Approval {
                    id: "a".into(),
                    decision
                }
            );
            // The wire name is the documented one.
            let as_core: lucy_core::ApprovalDecision = decision.into();
            assert_eq!(
                as_core,
                match decision {
                    ApprovalDecision::AllowOnce => lucy_core::ApprovalDecision::AllowOnce,
                    ApprovalDecision::AllowAlways => lucy_core::ApprovalDecision::AllowAlways,
                    ApprovalDecision::Deny => lucy_core::ApprovalDecision::Deny,
                }
            );
        }
    }

    #[test]
    fn server_messages_are_tagged_and_hello_carries_pairing_state() {
        let hello = ServerMessage::Hello {
            protocol: PROTOCOL_VERSION,
            server: "desktop".into(),
            version: "0.1.0".into(),
            paired: false,
            runtime_loaded: false,
        };
        let value: Value = serde_json::from_str(&hello.to_json()).unwrap();
        assert_eq!(value["type"], "hello");
        assert_eq!(value["protocol"], 1);
        assert_eq!(value["paired"], false);
        assert_eq!(value["runtime_loaded"], false);
    }

    #[test]
    fn task_done_distinguishes_stopped_from_verified() {
        let done = ServerMessage::TaskDone {
            task_id: "t".into(),
            summary: "did the thing".into(),
            complete: false,
        };
        let value: Value = serde_json::from_str(&done.to_json()).unwrap();
        assert_eq!(value["complete"], false);
    }

    #[test]
    fn pairing_payload_is_stable_json() {
        let payload = PairingPayload {
            lucy: 1,
            name: "desktop".into(),
            url: "ws://10.0.0.2:9847/ws".into(),
            http: "http://10.0.0.2:9847".into(),
            token: "lucy_pair_deadbeef".into(),
            server_id: "srv-1".into(),
        };
        let text = serde_json::to_string(&payload).unwrap();
        let back: PairingPayload = serde_json::from_str(&text).unwrap();
        assert_eq!(back, payload);
    }
}
