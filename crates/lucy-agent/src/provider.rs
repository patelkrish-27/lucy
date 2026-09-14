use anyhow::{anyhow, Context, Result};
use lucy_config::LucyConfig;
use lucy_core::*;
use reqwest::Client;
use serde_json::{json, Value};
use std::time::Duration;
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub struct OpenAIProvider { client: Client, config: LucyConfig, model: String }
impl OpenAIProvider {
    pub fn new(api_key:String,model:String,base_url:Option<String>)->Result<Self>{
        let client=Client::builder().timeout(Duration::from_secs(120)).build().context("failed to build HTTP client")?;
        // Create minimal config for generic new() — stores api_key/base_url as legacy fallback
        let mut cfg = LucyConfig::default();
        cfg.models.main = model.clone();
        cfg.models.api_key = Some(api_key);
        cfg.models.base_url = base_url.clone();
        // Also populate per-model so cheap inherits same if not overridden
        cfg.models.main_api_key = cfg.models.api_key.clone();
        cfg.models.main_base_url = base_url.clone();
        cfg.models.cheap_api_key = cfg.models.api_key.clone();
        cfg.models.cheap_base_url = base_url.clone();
        Ok(Self{client, config: cfg, model})
    }
    pub fn from_config(cfg:&LucyConfig)->Result<Self>{
        // Validate at least one per-model key is present; main is required
        let main_key = cfg.main_api_key().or_else(|| cfg.llm_api_key()).context(
            "Main LLM API key is not set — open Settings (Ctrl+,) set 'Main API Key (OpenChat)' or export OPENCHAT_API_KEY / LUCY_MAIN_API_KEY / OPENAI_API_KEY"
        )?;
        // Cheap key optional at init — if missing, will error later when cheap model is used
        let _ = main_key;
        let client=Client::builder().timeout(Duration::from_secs(120)).build().context("failed to build HTTP client")?;
        Ok(Self{client, config: cfg.clone(), model: cfg.models.main.clone()})
    }
    pub fn from_env()->Result<Self>{
        // Try per-model env first, fallback to generic
        let cfg = LucyConfig::load().unwrap_or_default();
        let api_key = cfg.main_api_key().or_else(|| cfg.llm_api_key()).or_else(||{
            std::env::var("LUCY_API_KEY")
                .or_else(|_| std::env::var("OPENAI_API_KEY"))
                .or_else(|_| std::env::var("OPENCHAT_API_KEY"))
                .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
                .or_else(|_| std::env::var("GEMINI_API_KEY"))
                .or_else(|_| std::env::var("LLM_API_KEY")).ok()
        }).context("LLM API key is not set — set LUCY_MAIN_API_KEY / OPENCHAT_API_KEY / OPENAI_API_KEY")?;
        let model=std::env::var("OPENAI_MODEL").or_else(|_| std::env::var("LUCY_MODEL")).unwrap_or_else(|_| cfg.models.main.clone());
        let mut cfg2 = cfg;
        cfg2.models.main = model.clone();
        cfg2.models.main_api_key = Some(api_key);
        let client=Client::builder().timeout(Duration::from_secs(120)).build().context("failed to build HTTP client")?;
        Ok(Self{client, config: cfg2, model})
    }
    fn api_key_for(&self, model:&str)->Result<String>{
        self.config.api_key_for(model).with_context(|| format!("API key for model '{model}' is not set — set via Settings (Main/Cheap API Key) or env OPENCHAT_API_KEY / GEMINI_API_KEY"))
    }
    fn base_url_for(&self, model:&str)->String{
        self.config.base_url_for(model).unwrap_or_else(|| "https://api.openai.com/v1".to_string())
    }
    pub async fn complete_json(&self, model:&str, system:&str, user:&str, interrupt:InterruptSignal)->Result<Value>{
        if interrupt.is_set(){return Err(LucyError::Cancelled.into());}
        let api_key = self.api_key_for(model)?;
        let base_url = self.base_url_for(model);
        let payload=json!({"model":model,"messages":[{"role":"system","content":system},{"role":"user","content":user}],"temperature":0,"response_format":{"type":"json_object"}});
        let url=format!("{}/chat/completions",base_url.trim_end_matches('/'));
        let res=self.client.post(&url).bearer_auth(&api_key).json(&payload).send().await.context("JSON model API request failed")?;
        let status=res.status();let body=res.text().await.context("failed to read JSON model response")?;
        if !status.is_success(){return Err(anyhow!("JSON model API returned {}: {}",status,body));}
        let resp:Value=serde_json::from_str(&body).context("invalid JSON model response")?;
        let content=resp["choices"][0]["message"]["content"].as_str().ok_or_else(||anyhow!("JSON model returned no content"))?;
        serde_json::from_str(content).or_else(|_|{let cleaned=content.trim().trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```").trim();serde_json::from_str(cleaned).context("JSON model returned invalid JSON")})
    }
}
#[async_trait::async_trait]
impl ModelProvider for OpenAIProvider {
    async fn run_turn(&self,request:ModelRequest,_events:mpsc::UnboundedSender<AgentEvent>,interrupt:InterruptSignal)->anyhow::Result<ModelTurn>{
        if interrupt.is_set(){return Err(LucyError::Cancelled.into());}
        let mut messages=Vec::new();
        messages.push(json!({"role":"system","content":"You are Lucy, a fast and capable AI computer-operation assistant. Prefer precise tool use, minimize unnecessary steps, and verify important actions."}));
        for msg in request.history {match msg{TurnMessage::User(text)=>messages.push(json!({"role":"user","content":text})),TurnMessage::Assistant(turn)=>{let mut m=json!({"role":"assistant","content":turn.text});if !turn.tool_calls.is_empty(){m["tool_calls"]=Value::Array(turn.tool_calls.iter().map(|c|json!({"id":c.id,"type":"function","function":{"name":c.name,"arguments":serde_json::to_string(&c.input).unwrap_or_else(|_|"{}".into())}})).collect());}messages.push(m)},TurnMessage::Tool(res)=>messages.push(json!({"role":"tool","tool_call_id":res.call_id,"name":res.name,"content":res.output.to_string()}))}}
        let mut payload=json!({"model":self.model,"messages":messages});
        if !request.tools.is_empty(){payload["tools"]=json!(request.tools.iter().map(|t|json!({"type":"function","function":{"name":t["name"],"description":t["description"],"parameters":t["input_schema"]}})).collect::<Vec<_>>());}
        let api_key = self.api_key_for(&self.model)?;
        let base_url = self.base_url_for(&self.model);
        let url=format!("{}/chat/completions",base_url.trim_end_matches('/'));
        let res=self.client.post(&url).bearer_auth(&api_key).json(&payload).send().await.context("OpenAI API request failed")?;let status=res.status();let body=res.text().await.context("failed to read response body")?;if !status.is_success(){return Err(anyhow!("OpenAI API returned {}: {}",status,body));}
        let resp_json:Value=serde_json::from_str(&body).context("invalid JSON response from OpenAI")?;let message=&resp_json["choices"][0]["message"];let text=message["content"].as_str().map(str::to_owned);let mut tool_calls=Vec::new();if let Some(calls)=message["tool_calls"].as_array(){for call in calls{let id=call["id"].as_str().unwrap_or_default().to_owned();let name=call["function"]["name"].as_str().unwrap_or_default().to_owned();let args=call["function"]["arguments"].as_str().unwrap_or("{}");let input:Value=serde_json::from_str(args).unwrap_or_else(|_|json!({}));tool_calls.push(ToolCall{id,name,input});}}
        Ok(ModelTurn{text,stop:tool_calls.is_empty(),tool_calls})
    }
}
