use anyhow::{anyhow, Context, Result};
use lucy_core::*;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub struct OpenAIProvider {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl OpenAIProvider {
    pub fn new(api_key: String, model: String, base_url: Option<String>) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .context("failed to build HTTP client")?;
        let base_url = base_url.unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        Ok(Self { client, api_key, model, base_url })
    }

    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY").context("OPENAI_API_KEY is not set")?;
        let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());
        let base_url = std::env::var("OPENAI_BASE_URL").ok();
        Self::new(api_key, model, base_url)
    }
}

impl ModelProvider for OpenAIProvider {
    fn run_turn<'life0, 'life1, 'async_trait>(
        &'life0 self,
        request: ModelRequest,
        events: mpsc::UnboundedSender<AgentEvent>,
        interrupt: InterruptSignal,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<ModelTurn>> + Send + 'async_trait>>
    where
        'life0: 'async_trait,
        'life1: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move {
            if interrupt.is_set() {
                return Err(LucyError::Cancelled.into());
            }

            let mut messages = Vec::new();
            messages.push(json!({
                "role": "system",
                "content": "You are Lucy, a Rust-native AI assistant contextually helping users safely and accurately."
            }));

            for msg in request.history {
                match msg {
                    TurnMessage::User(text) => messages.push(json!({"role": "user", "content": text})),
                    TurnMessage::Assistant(text) => messages.push(json!({"role": "assistant", "content": text})),
                    TurnMessage::Tool(res) => messages.push(json!({
                        "role": "tool",
                        "tool_call_id": res.call_id,
                        "name": res.name,
                        "content": res.output.to_string()
                    })),
                }
            }

            let mut payload = json!({
                "model": self.model,
                "messages": messages,
            });

            if !request.tools.is_empty() {
                payload["tools"] = json!(request.tools.iter().map(|t| {
                    json!({
                        "type": "function",
                        "function": t
                    })
                }).collect::<Vec<_>>());
            }

            let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
            let res = self.client.post(&url)
                .bearer_auth(&self.api_key)
                .json(&payload)
                .send()
                .await
                .context("OpenAI API request failed")?;

            let status = res.status();
            let body = res.text().await.context("failed to read response body")?;
            if !status.is_success() {
                return Err(anyhow!("OpenAI API returned {}: {}", status, body));
            }

            let resp_json: Value = serde_json::from_str(&body).context("invalid JSON response from OpenAI")?;
            let choice = resp_json["choices"][0].clone();
            let message = &choice["message"];
            
            let text = message["content"].as_str().map(|s| s.to_string());
            let mut tool_calls = Vec::new();

            if let Some(calls) = message["tool_calls"].as_array() {
                for call in calls {
                    let id = call["id"].as_str().unwrap_or_default().to_string();
                    let name = call["function"]["name"].as_str().unwrap_or_default().to_string();
                    let args_str = call["function"]["arguments"].as_str().unwrap_or("{}");
                    let input: Value = serde_json::from_str(args_str).unwrap_or(Value::Object(Default::default()));
                    tool_calls.push(ToolCall { id, name, input });
                }
            }

            let stop = tool_calls.is_empty();
            Ok(ModelTurn { text, tool_calls, stop })
        })
    }
