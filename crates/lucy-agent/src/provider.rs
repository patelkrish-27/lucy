use anyhow::{anyhow, Context, Result};
use lucy_core::*;
use reqwest::Client;
use serde_json::{json, Value};
use std::time::Duration;
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub struct OpenAIProvider { client: Client, api_key: String, model: String, base_url: String }
impl OpenAIProvider {
    pub fn new(api_key:String,model:String,base_url:Option<String>)->Result<Self>{let client=Client::builder().timeout(Duration::from_secs(120)).build().context("failed to build HTTP client")?;Ok(Self{client,api_key,model,base_url:base_url.unwrap_or_else(||"https://api.openai.com/v1".to_string())})}
    pub fn from_env()->Result<Self>{let api_key=std::env::var("OPENAI_API_KEY").context("OPENAI_API_KEY is not set")?;let model=std::env::var("OPENAI_MODEL").unwrap_or_else(|_|"gpt-4o".to_string());Self::new(api_key,model,std::env::var("OPENAI_BASE_URL").ok())}
}
#[async_trait::async_trait]
impl ModelProvider for OpenAIProvider {
    async fn run_turn(&self,request:ModelRequest,_events:mpsc::UnboundedSender<AgentEvent>,interrupt:InterruptSignal)->anyhow::Result<ModelTurn>{
        if interrupt.is_set(){return Err(LucyError::Cancelled.into());}
        let mut messages=Vec::new();
        messages.push(json!({"role":"system","content":"You are Lucy, a fast and capable AI computer-operation assistant. Prefer precise tool use, minimize unnecessary steps, and verify important actions."}));
        for msg in request.history {
            match msg {
                TurnMessage::User(text)=>messages.push(json!({"role":"user","content":text})),
                TurnMessage::Assistant(turn)=>{
                    let mut m=json!({"role":"assistant","content":turn.text});
                    if !turn.tool_calls.is_empty(){m["tool_calls"]=Value::Array(turn.tool_calls.iter().map(|c|json!({"id":c.id,"type":"function","function":{"name":c.name,"arguments":serde_json::to_string(&c.input).unwrap_or_else(|_|"{}".into())}})).collect());}
                    messages.push(m);
                }
                TurnMessage::Tool(res)=>messages.push(json!({"role":"tool","tool_call_id":res.call_id,"name":res.name,"content":res.output.to_string()})),
            }
        }
        let mut payload=json!({"model":self.model,"messages":messages});
        if !request.tools.is_empty(){payload["tools"]=json!(request.tools.iter().map(|t|json!({"type":"function","function":{"name":t["name"],"description":t["description"],"parameters":t["input_schema"]}})).collect::<Vec<_>>());}
        let url=format!("{}/chat/completions",self.base_url.trim_end_matches('/'));
        let res=self.client.post(&url).bearer_auth(&self.api_key).json(&payload).send().await.context("OpenAI API request failed")?;
        let status=res.status();let body=res.text().await.context("failed to read response body")?;
        if !status.is_success(){return Err(anyhow!("OpenAI API returned {}: {}",status,body));}
        let resp_json:Value=serde_json::from_str(&body).context("invalid JSON response from OpenAI")?;let message=&resp_json["choices"][0]["message"];let text=message["content"].as_str().map(str::to_owned);let mut tool_calls=Vec::new();
        if let Some(calls)=message["tool_calls"].as_array(){for call in calls{let id=call["id"].as_str().unwrap_or_default().to_owned();let name=call["function"]["name"].as_str().unwrap_or_default().to_owned();let args=call["function"]["arguments"].as_str().unwrap_or("{}");let input:Value=serde_json::from_str(args).unwrap_or_else(|_|json!({}));tool_calls.push(ToolCall{id,name,input});}}
        Ok(ModelTurn{text,stop:tool_calls.is_empty(),tool_calls})
    }
}
