use anyhow::{anyhow, Context, Result};
use lucy_config::LucyConfig;
use lucy_core::*;
use reqwest::Client;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub struct OpenAIProvider { client: Client, config: LucyConfig, model: Arc<std::sync::RwLock<String>>, usage: Arc<Mutex<TokenUsage>> }
impl OpenAIProvider {
    pub fn new(api_key:String,model:String,base_url:Option<String>)->Result<Self>{
        let client=Client::builder().timeout(Duration::from_secs(120)).build().context("failed to build HTTP client")?;
        // Create minimal config for generic new() — stores api_key/base_url as legacy fallback
        let mut cfg = LucyConfig::default();
        cfg.models.main = model.clone();
        cfg.models.api_key = Some(api_key);
        cfg.models.base_url = base_url.clone();
        cfg.models.main_api_key = cfg.models.api_key.clone();
        cfg.models.main_base_url = base_url.clone();
        Ok(Self{client, config: cfg, model: Arc::new(std::sync::RwLock::new(model)), usage: Arc::new(Mutex::new(TokenUsage::default()))})
    }
    pub fn set_model(&self,model:String){if let Ok(mut g)=self.model.write(){*g=model;}}
    pub fn model(&self)->String{self.model.read().map(|g|g.clone()).unwrap_or_default()}
    pub fn from_config(cfg:&LucyConfig)->Result<Self>{
        // Validate at least one per-model key is present; main is required
        let main_key = cfg.main_api_key().or_else(|| cfg.llm_api_key()).context(
            "Main LLM API key is not set — open Settings (Ctrl+,) set 'Main API Key (OpenChat)' or export OPENCHAT_API_KEY / LUCY_MAIN_API_KEY / OPENAI_API_KEY"
        )?;
        let _ = main_key;
        let client=Client::builder().timeout(Duration::from_secs(120)).build().context("failed to build HTTP client")?;
        Ok(Self{client, config: cfg.clone(), model: Arc::new(std::sync::RwLock::new(cfg.models.main.clone())), usage: Arc::new(Mutex::new(TokenUsage::default()))})
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
        Ok(Self{client, config: cfg2, model: Arc::new(std::sync::RwLock::new(model)), usage: Arc::new(Mutex::new(TokenUsage::default()))})
    }
    fn api_key(&self)->Result<String>{
        self.config.main_api_key().or_else(|| self.config.llm_api_key()).with_context(|| "Main LLM API key is not set — set it in Lucy Settings or via OPENCHAT_API_KEY / LUCY_MAIN_API_KEY / OPENAI_API_KEY")
    }
    fn base_url(&self)->String{
        self.config.main_base_url().or_else(|| self.config.llm_base_url()).unwrap_or_else(|| "https://api.openai.com/v1".to_string())
    }
    pub fn usage(&self)->TokenUsage{self.usage.lock().map(|g|g.clone()).unwrap_or_default()}
    pub fn reset_usage(&self){if let Ok(mut g)=self.usage.lock(){*g=TokenUsage::default();}}
    fn parse_usage(resp:&Value)->Option<TokenUsage>{
        let u=resp.get("usage")?;
        let prompt_tokens=u.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0);
        let completion_tokens=u.get("completion_tokens").and_then(Value::as_u64).unwrap_or(0);
        let total_tokens=u.get("total_tokens").and_then(Value::as_u64).unwrap_or_else(||prompt_tokens+completion_tokens);
        Some(TokenUsage{prompt_tokens,completion_tokens,total_tokens})
    }
    fn truncate_body(body:&str)->String{body.chars().take(500).collect()}
    async fn post_json(&self,url:&str,api_key:&str,payload:&Value,interrupt:&InterruptSignal)->Result<reqwest::Response>{
        if interrupt.is_set(){return Err(LucyError::Cancelled.into());}
        for attempt in 1..=3u32{
            if interrupt.is_set(){return Err(LucyError::Cancelled.into());}
            let builder=self.client.post(url).bearer_auth(api_key).json(payload);
            // Cancel-safe send: dropping the reqwest future on cancel is fine.
            let send_outcome=tokio::select!{
                res=builder.send()=>Some(res),
                _=interrupt.notified()=>None,
            };
            let send_res=match send_outcome{
                None=>return Err(LucyError::Cancelled.into()),
                Some(r)=>r,
            };
            match send_res{
                Err(e)=>{
                    if attempt>=3{return Err(e).context("OpenAI API request failed");}
                    if interrupt.is_set(){return Err(LucyError::Cancelled.into());}
                    let backoff=if attempt==1{1}else{2};
                    tokio::select!{
                        _=tokio::time::sleep(Duration::from_secs(backoff))=>{},
                        _=interrupt.notified()=>return Err(LucyError::Cancelled.into()),
                    }
                    if interrupt.is_set(){return Err(LucyError::Cancelled.into());}
                    continue;
                }
                Ok(resp)=>{
                    let status=resp.status();
                    if status.is_success(){return Ok(resp);}
                    let retryable=status.as_u16()==429||status.is_server_error();
                    if retryable{
                        if attempt>=3{
                            let body=resp.text().await.unwrap_or_default();
                            return Err(anyhow!("OpenAI API returned {}: {}",status,Self::truncate_body(&body)));
                        }
                        drop(resp);
                        if interrupt.is_set(){return Err(LucyError::Cancelled.into());}
                        let backoff=if attempt==1{1}else{2};
                        tokio::select!{
                            _=tokio::time::sleep(Duration::from_secs(backoff))=>{},
                            _=interrupt.notified()=>return Err(LucyError::Cancelled.into()),
                        }
                        if interrupt.is_set(){return Err(LucyError::Cancelled.into());}
                        continue;
                    }else{
                        let body=resp.text().await.context("failed to read response body").map(|b|Self::truncate_body(&b)).unwrap_or_default();
                        return Err(anyhow!("OpenAI API returned {}: {}",status,body));
                    }
                }
            }
        }
        Err(anyhow!("OpenAI API request failed after retries"))
    }
    pub async fn complete_json(&self, model:&str, system:&str, user:&str, interrupt:InterruptSignal)->Result<Value>{
        if interrupt.is_set(){return Err(LucyError::Cancelled.into());}
        let api_key = self.api_key()?;
        let base_url = self.base_url();
        let payload=json!({"model":model,"messages":[{"role":"system","content":system},{"role":"user","content":user}],"temperature":0,"response_format":{"type":"json_object"}});
        let url=format!("{}/chat/completions",base_url.trim_end_matches('/'));
        if interrupt.is_set(){return Err(LucyError::Cancelled.into());}
        let res=self.post_json(&url,&api_key,&payload,&interrupt).await?;
        let status=res.status();let body=res.text().await.context("failed to read JSON model response")?;
        if !status.is_success(){return Err(anyhow!("OpenAI API returned {}: {}",status,Self::truncate_body(&body)));}
        let resp:Value=serde_json::from_str(&body).context("invalid JSON model response")?;
        if let Some(u)=Self::parse_usage(&resp){if let Ok(mut total)=self.usage.lock(){total.add(&u);}}
        let content=resp["choices"][0]["message"]["content"].as_str().ok_or_else(||anyhow!("JSON model returned no content"))?;
        Self::extract_json(content)
    }
    /// Parse model output robustly: strict JSON first, then fenced code blocks,
    /// then the largest balanced `{...}` object embedded in prose. Some
    /// backends (agentic proxies, non-instruct models) return commentary
    /// around the JSON or ignore `response_format` entirely; without this
    /// the whole hierarchical plan fails on one malformed reply.
    pub fn extract_json(content:&str)->Result<Value>{
        if let Ok(v)=serde_json::from_str::<Value>(content){return Ok(v);}
        let trimmed=content.trim();
        for fence in ["```json","```JSON","```"]{
            if let Some(rest)=trimmed.strip_prefix(fence){
                if let Some(end)=rest.rfind("```"){
                    if let Ok(v)=serde_json::from_str::<Value>(rest[..end].trim()){return Ok(v);}
                } else if let Ok(v)=serde_json::from_str::<Value>(rest.trim()){return Ok(v);}
            }
        }
        if let Some(obj)=largest_balanced_object(trimmed){
            if let Ok(v)=serde_json::from_str::<Value>(&obj){return Ok(v);}
        }
        Err(anyhow!("JSON model returned invalid JSON: {}",Self::truncate_body(content)))
    }
    /// Raw text completion used when the caller needs graceful degradation
    /// (e.g. triage falling back to chat instead of erroring). Tracks token
    /// usage like the JSON path.
    pub async fn complete_content(&self, model:&str, system:&str, user:&str, interrupt:InterruptSignal)->Result<String>{
        if interrupt.is_set(){return Err(LucyError::Cancelled.into());}
        let api_key = self.api_key()?;
        let base_url = self.base_url();
        let payload=json!({"model":model,"messages":[{"role":"system","content":system},{"role":"user","content":user}],"temperature":0,"response_format":{"type":"json_object"}});
        let url=format!("{}/chat/completions",base_url.trim_end_matches('/'));
        if interrupt.is_set(){return Err(LucyError::Cancelled.into());}
        let res=self.post_json(&url,&api_key,&payload,&interrupt).await?;
        let status=res.status();let body=res.text().await.context("failed to read JSON model response")?;
        if !status.is_success(){return Err(anyhow!("OpenAI API returned {}: {}",status,Self::truncate_body(&body)));}
        let resp:Value=serde_json::from_str(&body).context("invalid JSON model response")?;
        if let Some(u)=Self::parse_usage(&resp){if let Ok(mut total)=self.usage.lock(){total.add(&u);}}
        resp["choices"][0]["message"]["content"].as_str().map(str::to_owned).ok_or_else(||anyhow!("JSON model returned no content"))
    }
    /// §10.1 — native function/tool calling for the v2 harness.
    ///
    /// Unlike `complete_json`/`complete_content` (which paste tool schemas
    /// as text and parse JSON-in-prose), this passes a real `tools` array
    /// and reads the provider's guaranteed `tool_calls` array. The returned
    /// `ModelTurn.text` carries the planner's goal/success-condition JSON
    /// (or chat reply); `ModelTurn.tool_calls` IS the plan.
    pub async fn plan_with_tools(
        &self,
        system:&str,
        history:&[TurnMessage],
        user_text:&str,
        tools:&[Value],
        interrupt:InterruptSignal,
    )->Result<ModelTurn>{
        if interrupt.is_set(){return Err(LucyError::Cancelled.into());}
        let mut messages=Vec::new();
        messages.push(json!({"role":"system","content":system}));
        for msg in history {
            match msg {
                TurnMessage::User(text)=>messages.push(json!({"role":"user","content":text})),
                TurnMessage::Assistant(turn)=>{
                    let mut m=json!({"role":"assistant","content":turn.text});
                    if !turn.tool_calls.is_empty(){
                        m["tool_calls"]=Value::Array(turn.tool_calls.iter().map(|c|json!({"id":c.id,"type":"function","function":{"name":c.name,"arguments":serde_json::to_string(&c.input).unwrap_or_else(|_|"{}".into())}})).collect());
                    }
                    messages.push(m)
                }
                TurnMessage::Tool(res)=>messages.push(json!({"role":"tool","tool_call_id":res.call_id,"name":res.name,"content":res.output.to_string()})),
            }
        }
        messages.push(json!({"role":"user","content":user_text}));
        let model=self.model();
        let mut payload=json!({"model":model,"messages":messages,"temperature":0});
        if !tools.is_empty(){
            payload["tools"]=json!(tools.iter().map(|t|{
                // Accept both `{"name","description","input_schema"}` (our
                // registry shape) and pre-built `{"type":"function",...}`.
                if t.get("type").and_then(Value::as_str)==Some("function"){ t.clone() }
                else { json!({"type":"function","function":{"name":t["name"],"description":t["description"],"parameters":t["input_schema"]}}) }
            }).collect::<Vec<_>>());
            payload["tool_choice"]=serde_json::json!("auto");
        }
        let api_key = self.api_key()?;
        let base_url = self.base_url();
        let url=format!("{}/chat/completions",base_url.trim_end_matches('/'));
        if interrupt.is_set(){return Err(LucyError::Cancelled.into());}
        let res=self.post_json(&url,&api_key,&payload,&interrupt).await?;
        let status=res.status();let body=res.text().await.context("failed to read response body")?;
        if !status.is_success(){return Err(anyhow!("OpenAI API returned {}: {}",status,Self::truncate_body(&body)));}
        let resp_json:Value=serde_json::from_str(&body).context("invalid JSON response from OpenAI")?;
        let message=&resp_json["choices"][0]["message"];
        let text=message["content"].as_str().map(str::to_owned);
        let mut tool_calls=Vec::new();
        if let Some(calls)=message["tool_calls"].as_array(){
            for call in calls{
                let id=call["id"].as_str().unwrap_or_default().to_owned();
                let id=if id.is_empty(){ format!("plan-{}", uuid::Uuid::new_v4()) } else { id };
                let name=call["function"]["name"].as_str().unwrap_or_default().to_owned();
                let args=call["function"]["arguments"].as_str().unwrap_or("{}");
                let input:Value=serde_json::from_str(args).unwrap_or_else(|_|json!({}));
                if name.is_empty(){ continue; }
                if !input.is_object(){ continue; }
                tool_calls.push(ToolCall{id,name,input});
            }
        }
        let usage=Self::parse_usage(&resp_json);
        if let Some(ref u)=usage{if let Ok(mut total)=self.usage.lock(){total.add(u);}}
        Ok(ModelTurn{text,stop:tool_calls.is_empty(),tool_calls,usage})
    }
}
/// Return the largest balanced `{...}` substring, respecting strings/escapes.
fn largest_balanced_object(text:&str)->Option<String>{
    let bytes=text.as_bytes();let mut best:Option<(usize,usize)>=None;
    let mut i=0;
    while i<bytes.len(){
        if bytes[i]!=b'{'{i+=1;continue;}
        let mut depth=0i32;let mut in_str=false;let mut esc=false;let mut j=i;
        while j<bytes.len(){
            let b=bytes[j];
            if in_str{if esc{esc=false;}else if b==b'\\'{esc=true;}else if b==b'"'{in_str=false;}}
            else{if b==b'"'{in_str=true;}else if b==b'{'{depth+=1;}else if b==b'}'{depth-=1;if depth==0{match best{Some((_,len))if j+1-i<=len=>{},_=>best=Some((i,j+1-i))}}break;}}
            j+=1;
        }
        i+=1;
    }
    best.map(|(s,l)|text[s..s+l].to_owned())
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn extracts_strict_json(){assert_eq!(OpenAIProvider::extract_json(r#"{"mode":"chat","reply":"hi"}"#).unwrap()["mode"],"chat");}
    #[test]fn extracts_fenced_json(){assert_eq!(OpenAIProvider::extract_json("```json\n{\"mode\":\"chat\",\"reply\":\"hi\"}\n```").unwrap()["mode"],"chat");}
    #[test]fn extracts_prose_embedded_json(){
        let v=OpenAIProvider::extract_json("Sure! Here you go: {\"mode\":\"chat\",\"reply\":\"hi\"} hope that helps.").unwrap();
        assert_eq!(v["mode"],"chat");
    }
    #[test]fn rejects_plain_text(){assert!(OpenAIProvider::extract_json("I am playing the video now.").is_err());}
    #[test]fn balanced_object_respects_strings(){assert_eq!(largest_balanced_object(r#"a {"k":"} not end"} b"#).unwrap(),r#"{"k":"} not end"}"#);}
}
#[async_trait::async_trait]
impl ModelProvider for OpenAIProvider {
    async fn run_turn(&self,request:ModelRequest,_events:mpsc::UnboundedSender<AgentEvent>,interrupt:InterruptSignal)->anyhow::Result<ModelTurn>{
        if interrupt.is_set(){return Err(LucyError::Cancelled.into());}
        let mut messages=Vec::new();
        messages.push(json!({"role":"system","content":"You are Lucy, a warm and friendly AI assistant that operates the user's computer. You are concise: give short answers and brief summaries of what you did, and expand only when asked. You act through tools: prefer precise tool use, minimize unnecessary steps, and verify important actions before reporting success. Never claim an error occurred unless a tool call actually returned an error. Never claim you completed a computer action unless a tool call just performed it. Never ask the user for permission to turn on, open or enable apps: just call the appropriate tool, and any approval needed is handled by the app automatically. When asked to play music or video, act immediately (open a search or stream URL with a browser, launch or shell tool) instead of describing what you would do. Reply in the user's language. Format every reply for a plain-text terminal: short paragraphs separated by blank lines, one list item per line starting with '- ', no **bold** markers and no backticks."}));
        for msg in request.history {match msg{TurnMessage::User(text)=>messages.push(json!({"role":"user","content":text})),TurnMessage::Assistant(turn)=>{let mut m=json!({"role":"assistant","content":turn.text});if !turn.tool_calls.is_empty(){m["tool_calls"]=Value::Array(turn.tool_calls.iter().map(|c|json!({"id":c.id,"type":"function","function":{"name":c.name,"arguments":serde_json::to_string(&c.input).unwrap_or_else(|_|"{}".into())}})).collect());}messages.push(m)},TurnMessage::Tool(res)=>messages.push(json!({"role":"tool","tool_call_id":res.call_id,"name":res.name,"content":res.output.to_string()}))}}
        let model=self.model();let mut payload=json!({"model":model,"messages":messages});
        if !request.tools.is_empty(){payload["tools"]=json!(request.tools.iter().map(|t|json!({"type":"function","function":{"name":t["name"],"description":t["description"],"parameters":t["input_schema"]}})).collect::<Vec<_>>());}
        let api_key = self.api_key()?;
        let base_url = self.base_url();
        let url=format!("{}/chat/completions",base_url.trim_end_matches('/'));
        if interrupt.is_set(){return Err(LucyError::Cancelled.into());}
        let res=self.post_json(&url,&api_key,&payload,&interrupt).await?;let status=res.status();let body=res.text().await.context("failed to read response body")?;if !status.is_success(){return Err(anyhow!("OpenAI API returned {}: {}",status,Self::truncate_body(&body)));}
        let resp_json:Value=serde_json::from_str(&body).context("invalid JSON response from OpenAI")?;let message=&resp_json["choices"][0]["message"];let text=message["content"].as_str().map(str::to_owned);let mut tool_calls=Vec::new();if let Some(calls)=message["tool_calls"].as_array(){for call in calls{let id=call["id"].as_str().unwrap_or_default().to_owned();let name=call["function"]["name"].as_str().unwrap_or_default().to_owned();let args=call["function"]["arguments"].as_str().unwrap_or("{}");let input:Value=serde_json::from_str(args).unwrap_or_else(|_|json!({}));tool_calls.push(ToolCall{id,name,input});}}
        let usage=Self::parse_usage(&resp_json);
        if let Some(ref u)=usage{if let Ok(mut total)=self.usage.lock(){total.add(u);}}
        Ok(ModelTurn{text,stop:tool_calls.is_empty(),tool_calls,usage})
    }
}
