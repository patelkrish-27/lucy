use anyhow::{Context, Result, anyhow};
use lucy_config::{
    DEFAULT_TEXT_BASE_URL, LucyConfig, OPENROUTER_REFERER, OPENROUTER_TITLE, ReasoningLevel,
    is_openrouter, model_for_endpoint,
};
use lucy_core::*;
use reqwest::{Client, RequestBuilder};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Output-token ceiling sent with every request.
///
/// Omitting `max_tokens` is not "unbounded" at OpenRouter: the gateway sizes
/// the request against the model's *full* context window and reserves the
/// difference as possible spend. On a 65k-context model that is a ~65k-token
/// reservation, which a free-tier key with no credits can never afford — the
/// gateway rejects the turn with `402 Payment Required: You requested up to
/// 65535 tokens, but can only afford 54814`. It reads like a billing problem
/// and is really a missing field, so Lucy states the ceiling explicitly.
///
/// 8192 is comfortably above every structured reply it asks for (a plan, a
/// routing verdict, a chat answer) and small enough to be affordable on a
/// zero-credit account.
pub const MAX_OUTPUT_TOKENS: u32 = 8192;

/// The `max_tokens` this call sends: the caller's ceiling when it gave one,
/// otherwise [`MAX_OUTPUT_TOKENS`].
///
/// A stage that answers in a fixed shape caps its own completion instead of
/// asking the model to be brief. Measured on the planner: the same
/// `agent_plan` prompt answered in 913, 1040, 1271 and 2233 completion tokens,
/// and [`crate`] parses at most the first six objectives — so the tail was
/// generated at ~30 tok/s, billed, and discarded. A ceiling turns that into a
/// bounded 1200-token answer; [`OpenAIProvider::extract_json`] recovers the
/// complete prefix when the cut lands mid-object.
fn output_ceiling(max_tokens: Option<u32>) -> u32 {
    max_tokens
        .filter(|n| *n > 0)
        .unwrap_or(MAX_OUTPUT_TOKENS)
        .min(MAX_OUTPUT_TOKENS)
}

/// Attach whatever headers a specific endpoint expects.
///
/// OpenRouter asks every client to identify itself with `HTTP-Referer` and
/// `X-Title`; requests without them are accepted but rank anonymously. Every
/// other OpenAI-compatible endpoint is left untouched, so a gateway that
/// rejects unknown headers is unaffected.
pub fn apply_endpoint_headers(builder: RequestBuilder, url: &str) -> RequestBuilder {
    if is_openrouter(url) {
        builder
            .header("HTTP-Referer", OPENROUTER_REFERER)
            .header("X-Title", OPENROUTER_TITLE)
    } else {
        builder
    }
}

/// A concrete (endpoint, key, model) triple to call.
///
/// Lets one [`OpenAIProvider`] serve every reasoning tier across every
/// connected provider: the runtime resolves `provider_id/model` from the config
/// into one of these and hands it to the completion methods.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelTarget {
    pub base_url: String,
    /// `None` for a local endpoint that needs no key.
    pub api_key: Option<String>,
    pub model: String,
}

impl ModelTarget {
    pub fn new(
        base_url: impl Into<String>,
        api_key: Option<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            api_key: api_key.filter(|k| !k.trim().is_empty()),
            model: model.into(),
        }
    }

    /// Resolve a stored `provider_id/model` key against the config. Falls back
    /// to the legacy `main_*` endpoint when no provider owns the key.
    pub fn from_config(config: &LucyConfig, key: &str) -> Result<Self> {
        let (base_url, api_key, model) = config.resolve_endpoint(key).context(
            "no connected provider serves this model — run Test Connection in /settings",
        )?;
        Ok(Self::new(base_url, api_key, model))
    }

    /// `provider_id/model` style label for UI display.
    pub fn label(&self, key: &str) -> String {
        let k = key.trim();
        if !k.is_empty() {
            return k.to_owned();
        }
        self.model.clone()
    }

    /// The target corrected for its endpoint: OpenRouter only knows
    /// `vendor/model` ids, so a bare configured name (a leftover
    /// `gemini-web`, a hand-typed `gpt-4`) is replaced with a valid one
    /// instead of 404ing on every turn.
    pub fn for_endpoint(&self) -> Self {
        Self {
            base_url: self.base_url.clone(),
            api_key: self.api_key.clone(),
            model: model_for_endpoint(&self.base_url, &self.model),
        }
    }
}

/// True when `base_url` points at a loopback interface. Local
/// OpenAI-compatible servers (llama.cpp, Ollama, the OpenChat proxy on
/// 127.0.0.1:11435, …) take no API key, so the startup key gate must not
/// block them — otherwise even `hello` never reaches the pipeline.
pub fn endpoint_needs_key(base_url: &str) -> bool {
    let hostport = base_url
        .split("://")
        .nth(1)
        .unwrap_or(base_url)
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default()
        .rsplit('@')
        .next()
        .unwrap_or_default();
    // Strip an optional port without mistaking it for the host: only a
    // single colon means `host:port`; bracketed IPv6 keeps `::` inside.
    let host = if let Some(stripped) = hostport.strip_prefix('[') {
        stripped.split(']').next().unwrap_or(stripped)
    } else if hostport.matches(':').count() == 1 {
        hostport.split(':').next().unwrap_or(hostport)
    } else {
        hostport
    };
    let h = host.trim().to_ascii_lowercase();
    if h == "localhost" || h == "::1" {
        return false;
    }
    if let Ok(ip) = h.parse::<std::net::IpAddr>() {
        return !ip.is_loopback();
    }
    // `127.*` without parsing as a full IP (e.g. trailing dot edge cases).
    if h.starts_with("127.") {
        return false;
    }
    true
}

/// Everything the runtime needs from a text model, and nothing else.
///
/// The seam exists so the two-speed agent loop can be tested against a
/// [`testing::StubProvider`](crate::testing::StubProvider) and assert its LLM
/// call budget: the fast lane is measured by *how few* slow calls a healthy
/// task spends, which is only observable if the slow lane is substitutable.
/// Every method is a thin mirror of the matching [`OpenAIProvider`] method.
#[async_trait::async_trait]
pub trait ModelProvider: Send + Sync {
    /// The model this provider calls when no explicit target is given.
    fn model(&self) -> String;
    /// Point the provider at a different model.
    fn set_model(&self, model: String);
    /// Cumulative token usage across every call this provider served.
    fn usage(&self) -> TokenUsage;

    async fn complete_json(
        &self,
        model: &str,
        purpose: &str,
        system: &str,
        user: &str,
        interrupt: InterruptSignal,
    ) -> Result<Value>;

    async fn complete_text(
        &self,
        model: &str,
        purpose: &str,
        system: &str,
        user: &str,
        interrupt: InterruptSignal,
    ) -> Result<String>;

    /// [`Self::complete_json`] against an explicit endpoint/key/model.
    ///
    /// `max_tokens` caps the completion for this call only; `None` uses
    /// [`MAX_OUTPUT_TOKENS`]. A stage that wants a short, fixed-shape answer
    /// (a plan, a verdict) should pass a ceiling rather than asking politely in
    /// the prompt — a 1200-token plan that arrives in 40 seconds is a latency
    /// bug, and only `max_tokens` bounds it.
    async fn complete_json_on(
        &self,
        target: &ModelTarget,
        purpose: &str,
        system: &str,
        user: &str,
        interrupt: InterruptSignal,
        max_tokens: Option<u32>,
    ) -> Result<Value>;

    /// [`Self::complete_text`] against an explicit endpoint/key/model.
    async fn complete_text_on(
        &self,
        target: &ModelTarget,
        purpose: &str,
        system: &str,
        user: &str,
        interrupt: InterruptSignal,
        max_tokens: Option<u32>,
    ) -> Result<String>;

    /// One chat-completion turn against an explicit endpoint/key/model, with the
    /// provider's own tool-calling wire format, returning whatever the model
    /// chose to do: prose, tool calls, or both.
    ///
    /// `tools` is the live catalog in Lucy's tool-definition shape (see
    /// [`tool_chat_payload`] for the translation to the wire). Passing an empty
    /// slice asks the same question with no tools offered, which is how a caller
    /// gets a plain answer from the same prompt.
    #[allow(clippy::too_many_arguments)]
    async fn complete_with_tools_on(
        &self,
        target: &ModelTarget,
        purpose: &str,
        system: &str,
        user: &str,
        tools: &[Value],
        interrupt: InterruptSignal,
        max_tokens: Option<u32>,
    ) -> Result<AssistantTurn>;
}

#[async_trait::async_trait]
impl ModelProvider for OpenAIProvider {
    fn model(&self) -> String {
        OpenAIProvider::model(self)
    }
    fn set_model(&self, model: String) {
        OpenAIProvider::set_model(self, model);
    }
    fn usage(&self) -> TokenUsage {
        OpenAIProvider::usage(self)
    }
    async fn complete_json(
        &self,
        model: &str,
        purpose: &str,
        system: &str,
        user: &str,
        interrupt: InterruptSignal,
    ) -> Result<Value> {
        OpenAIProvider::complete_json(self, model, purpose, system, user, interrupt).await
    }
    async fn complete_text(
        &self,
        model: &str,
        purpose: &str,
        system: &str,
        user: &str,
        interrupt: InterruptSignal,
    ) -> Result<String> {
        OpenAIProvider::complete_text(self, model, purpose, system, user, interrupt).await
    }
    async fn complete_json_on(
        &self,
        target: &ModelTarget,
        purpose: &str,
        system: &str,
        user: &str,
        interrupt: InterruptSignal,
        max_tokens: Option<u32>,
    ) -> Result<Value> {
        OpenAIProvider::complete_json_on(self, target, purpose, system, user, interrupt, max_tokens)
            .await
    }
    async fn complete_text_on(
        &self,
        target: &ModelTarget,
        purpose: &str,
        system: &str,
        user: &str,
        interrupt: InterruptSignal,
        max_tokens: Option<u32>,
    ) -> Result<String> {
        OpenAIProvider::complete_text_on(self, target, purpose, system, user, interrupt, max_tokens)
            .await
    }
    #[allow(clippy::too_many_arguments)]
    async fn complete_with_tools_on(
        &self,
        target: &ModelTarget,
        purpose: &str,
        system: &str,
        user: &str,
        tools: &[Value],
        interrupt: InterruptSignal,
        max_tokens: Option<u32>,
    ) -> Result<AssistantTurn> {
        OpenAIProvider::complete_with_tools_on(
            self,
            target,
            purpose,
            system,
            user,
            tools,
            interrupt,
            max_tokens,
        )
        .await
    }
}

#[derive(Debug, Clone)]
pub struct OpenAIProvider {
    client: Client,
    config: LucyConfig,
    model: Arc<std::sync::RwLock<String>>,
    usage: Arc<Mutex<TokenUsage>>,
}
impl OpenAIProvider {
    pub fn new(api_key: String, model: String, base_url: Option<String>) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .context("failed to build HTTP client")?;
        // Create minimal config for generic new() — stores api_key/base_url as legacy fallback
        let mut cfg = LucyConfig::default();
        cfg.set_anchor_model(&model);
        cfg.models.api_key = Some(api_key);
        cfg.models.base_url = base_url.clone();
        cfg.models.text_api_key = cfg.models.api_key.clone();
        cfg.models.text_base_url = base_url.clone();
        Ok(Self {
            client,
            config: cfg,
            model: Arc::new(std::sync::RwLock::new(model)),
            usage: Arc::new(Mutex::new(TokenUsage::default())),
        })
    }
    pub fn set_model(&self, model: String) {
        if let Ok(mut g) = self.model.write() {
            *g = model;
        }
    }
    pub fn model(&self) -> String {
        self.model.read().map(|g| g.clone()).unwrap_or_default()
    }
    pub fn from_config(cfg: &LucyConfig) -> Result<Self> {
        // A key is required unless the endpoint is a local server that takes
        // none (loopback). Without this, a keyless `http://127.0.0.1:11435`
        // setup fails here and no command — not even `hello` — ever reaches
        // the classify → level → model → respond pipeline.
        let base = cfg
            .text_base_url()
            .or_else(|| cfg.llm_base_url())
            .unwrap_or_else(|| DEFAULT_TEXT_BASE_URL.to_string());
        let needs_key = endpoint_needs_key(&base);
        let main_key = cfg.text_api_key().or_else(|| cfg.llm_api_key());
        if needs_key && main_key.is_none() {
            return Err(anyhow!(
                "No LLM API key is set — connect a provider in Settings (Ctrl+,) or export OPENCHAT_API_KEY / LUCY_MAIN_API_KEY / OPENAI_API_KEY"
            ));
        }
        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .context("failed to build HTTP client")?;
        Ok(Self {
            client,
            config: cfg.clone(),
            // Level 3 is the anchor model, so it is the sensible starting point
            // for a caller that has not picked a tier yet.
            model: Arc::new(std::sync::RwLock::new(
                cfg.resolve_level_model(ReasoningLevel::L3),
            )),
            usage: Arc::new(Mutex::new(TokenUsage::default())),
        })
    }
    pub fn from_env() -> Result<Self> {
        // Try per-model env first, fallback to generic. A key is only
        // required for non-local endpoints (see `from_config`).
        let cfg = LucyConfig::load().unwrap_or_default();
        let api_key = cfg
            .text_api_key()
            .or_else(|| cfg.llm_api_key())
            .or_else(|| {
                std::env::var("LUCY_API_KEY")
                    .or_else(|_| std::env::var("OPENAI_API_KEY"))
                    .or_else(|_| std::env::var("OPENCHAT_API_KEY"))
                    .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
                    .or_else(|_| std::env::var("GEMINI_API_KEY"))
                    .or_else(|_| std::env::var("LLM_API_KEY"))
                    .ok()
            });
        let base = cfg
            .text_base_url()
            .or_else(|| cfg.llm_base_url())
            .unwrap_or_else(|| DEFAULT_TEXT_BASE_URL.to_string());
        if endpoint_needs_key(&base) && api_key.is_none() {
            return Err(anyhow!(
                "LLM API key is not set — set LUCY_MAIN_API_KEY / OPENCHAT_API_KEY / OPENAI_API_KEY"
            ));
        }
        let model = std::env::var("OPENAI_MODEL")
            .or_else(|_| std::env::var("LUCY_MODEL"))
            .unwrap_or_else(|_| cfg.resolve_level_model(ReasoningLevel::L3));
        let mut cfg2 = cfg;
        cfg2.set_anchor_model(&model);
        cfg2.models.text_api_key = api_key;
        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .context("failed to build HTTP client")?;
        Ok(Self {
            client,
            config: cfg2,
            model: Arc::new(std::sync::RwLock::new(model)),
            usage: Arc::new(Mutex::new(TokenUsage::default())),
        })
    }
    fn api_key(&self) -> Option<String> {
        self.config
            .text_api_key()
            .or_else(|| self.config.llm_api_key())
    }
    fn base_url(&self) -> String {
        self.config
            .text_base_url()
            .or_else(|| self.config.llm_base_url())
            .unwrap_or_else(|| DEFAULT_TEXT_BASE_URL.to_string())
    }
    pub fn usage(&self) -> TokenUsage {
        self.usage.lock().map(|g| g.clone()).unwrap_or_default()
    }
    pub fn reset_usage(&self) {
        if let Ok(mut g) = self.usage.lock() {
            *g = TokenUsage::default();
        }
    }
    fn parse_usage(resp: &Value) -> Option<TokenUsage> {
        let u = resp.get("usage")?;
        let prompt_tokens = u.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0);
        let completion_tokens = u
            .get("completion_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let total_tokens = u
            .get("total_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_else(|| prompt_tokens + completion_tokens);
        Some(TokenUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens,
        })
    }
    fn truncate_body(body: &str) -> String {
        body.chars().take(500).collect()
    }

    /// What an error response contributes to a message.
    ///
    /// A provider's error body is JSON in almost every case, and it says one
    /// useful thing — `"message"` — inside a wrapper that says nothing. Quoting
    /// the wrapper is how a raw `{"error":{"message":…}}` used to end up as the
    /// error text; quoting the message keeps the classifiable sentence and lets
    /// the status line beside it carry the classification. A non-JSON body (an
    /// HTML error page, a proxy banner) is truncated as before, because there is
    /// nothing to extract from it.
    fn body_excerpt(body: &str) -> String {
        match serde_json::from_str::<Value>(body) {
            Ok(value) => lucy_core::payload_message(&value)
                .map(|m| Self::truncate_body(&m))
                .unwrap_or_else(|| Self::truncate_body(body)),
            Err(_) => Self::truncate_body(body),
        }
    }
    /// The assistant text of a chat completion, tolerating the shape a
    /// reasoning model returns: `content` is `null` (or empty) when the whole
    /// turn went into `reasoning`/`reasoning_content`, which is where
    /// OpenRouter's reasoning models put a JSON answer that overruns the
    /// visible budget.
    fn response_text(resp: &Value) -> String {
        let msg = &resp["choices"][0]["message"];
        for key in ["content", "reasoning_content", "reasoning"] {
            if let Some(s) = msg.get(key).and_then(Value::as_str)
                && !s.trim().is_empty()
            {
                return s.trim().to_owned();
            }
        }
        String::new()
    }
    async fn post_json(
        &self,
        url: &str,
        api_key: Option<&str>,
        payload: &Value,
        interrupt: &InterruptSignal,
    ) -> Result<reqwest::Response> {
        if interrupt.is_set() {
            return Err(LucyError::Cancelled.into());
        }
        for attempt in 1..=3u32 {
            if interrupt.is_set() {
                return Err(LucyError::Cancelled.into());
            }
            let mut builder = self.client.post(url);
            // Keyless local servers get no `Authorization` header at all;
            // some reject even an empty `Bearer` value.
            if let Some(key) = api_key.filter(|k| !k.trim().is_empty()) {
                builder = builder.bearer_auth(key);
            }
            let builder = apply_endpoint_headers(builder, url).json(payload);
            // Cancel-safe send: dropping the reqwest future on cancel is fine.
            let send_outcome = tokio::select! {
                res=builder.send()=>Some(res),
                _=interrupt.notified()=>None,
            };
            let send_res = match send_outcome {
                None => return Err(LucyError::Cancelled.into()),
                Some(r) => r,
            };
            match send_res {
                Err(e) => {
                    if attempt >= 3 {
                        return Err(e).context("OpenAI API request failed");
                    }
                    if interrupt.is_set() {
                        return Err(LucyError::Cancelled.into());
                    }
                    let backoff = if attempt == 1 { 1 } else { 2 };
                    tokio::select! {
                        _=tokio::time::sleep(Duration::from_secs(backoff))=>{},
                        _=interrupt.notified()=>return Err(LucyError::Cancelled.into()),
                    }
                    if interrupt.is_set() {
                        return Err(LucyError::Cancelled.into());
                    }
                    continue;
                }
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        return Ok(resp);
                    }
                    let retryable = status.as_u16() == 429 || status.is_server_error();
                    if retryable {
                        if attempt >= 3 {
                            let body = resp.text().await.unwrap_or_default();
                            return Err(anyhow!(
                                "OpenAI API returned {}: {}",
                                status,
                                Self::body_excerpt(&body)
                            ));
                        }
                        drop(resp);
                        if interrupt.is_set() {
                            return Err(LucyError::Cancelled.into());
                        }
                        let backoff = if attempt == 1 { 1 } else { 2 };
                        tokio::select! {
                            _=tokio::time::sleep(Duration::from_secs(backoff))=>{},
                            _=interrupt.notified()=>return Err(LucyError::Cancelled.into()),
                        }
                        if interrupt.is_set() {
                            return Err(LucyError::Cancelled.into());
                        }
                        continue;
                    } else {
                        let body = resp
                            .text()
                            .await
                            .context("failed to read response body")
                            .map(|b| Self::truncate_body(&b))
                            .unwrap_or_default();
                        let code = status.as_u16();
                        if code == 401 || code == 403 {
                            // No inline advice here: `lucy_core::friendly` already
                            // maps 401/403 to the Connect-providers remedy, and a
                            // second copy in the error text is duplicated on every
                            // surface that classifies it.
                            return Err(anyhow!("OpenAI API returned {}: {body}", status));
                        }
                        return Err(anyhow!("OpenAI API returned {}: {}", status, body));
                    }
                }
            }
        }
        Err(anyhow!("OpenAI API request failed after retries"))
    }
    /// The endpoint/key/model triple for a model *name*.
    ///
    /// `model` is whatever the caller holds: sometimes a bare name, sometimes a
    /// stored `provider_id/model` key. Resolving it through the config means a
    /// connected provider always wins over the legacy `text_base_url`, so a
    /// `openrouter/…` key is sent to OpenRouter rather than to the local
    /// OpenChat proxy on 127.0.0.1:11435 — which would reject it as an unknown
    /// model name. A bare name still falls back to the legacy endpoint, which
    /// is what a keyless local server setup relies on.
    fn target_for(&self, model: &str) -> ModelTarget {
        let trimmed = model.trim();
        let resolved = self
            .config
            .provider_owns_model_key(trimmed)
            .then(|| self.config.resolve_endpoint(trimmed))
            .flatten();
        match resolved {
            Some((base_url, api_key, model)) => ModelTarget::new(base_url, api_key, model),
            None => ModelTarget::new(self.base_url(), self.api_key(), model),
        }
    }

    pub async fn complete_json(
        &self,
        model: &str,
        purpose: &str,
        system: &str,
        user: &str,
        interrupt: InterruptSignal,
    ) -> Result<Value> {
        let target = self.target_for(model);
        self.complete_json_on(&target, purpose, system, user, interrupt, None)
            .await
    }
    pub async fn complete_text(
        &self,
        model: &str,
        purpose: &str,
        system: &str,
        user: &str,
        interrupt: InterruptSignal,
    ) -> Result<String> {
        let target = self.target_for(model);
        self.complete_text_on(&target, purpose, system, user, interrupt, None)
            .await
    }

    /// [`Self::complete_text`] against an explicit endpoint/key/model.
    ///
    /// `purpose` names the pipeline stage (`answer_turn`, `plan_commands`,
    /// `route_verify`, …) and is recorded in the model-call log. `max_tokens`
    /// caps this call's completion; `None` uses [`MAX_OUTPUT_TOKENS`].
    pub async fn complete_text_on(
        &self,
        target: &ModelTarget,
        purpose: &str,
        system: &str,
        user: &str,
        interrupt: InterruptSignal,
        max_tokens: Option<u32>,
    ) -> Result<String> {
        let started = std::time::Instant::now();
        let prompt = format!("{system}\n\n{user}");
        if interrupt.is_set() {
            let err: Result<String> = Err(LucyError::Cancelled.into());
            log_llm_call(
                "complete_text",
                purpose,
                target,
                started,
                None,
                &prompt,
                "",
                Some("cancelled".to_owned()),
            );
            return err;
        }
        if target.model.trim().is_empty() {
            log_llm_call(
                "complete_text",
                purpose,
                target,
                started,
                None,
                &prompt,
                "",
                Some("no model selected".to_owned()),
            );
            return Err(anyhow!("no model selected"));
        }
        let corrected = target.for_endpoint();
        let target = &corrected;
        let payload = json!({"model":target.model,"messages":[{"role":"system","content":system},{"role":"user","content":user}],"temperature":0.3,"max_tokens":output_ceiling(max_tokens)});
        let url = format!("{}/chat/completions", target.base_url);
        if interrupt.is_set() {
            log_llm_call(
                "complete_text",
                purpose,
                target,
                started,
                None,
                &prompt,
                "",
                Some("cancelled".to_owned()),
            );
            return Err(LucyError::Cancelled.into());
        }
        let res = match self
            .post(&url, target.api_key.as_deref(), &payload, &interrupt)
            .await
        {
            Ok(res) => res,
            Err(e) => {
                log_llm_call(
                    "complete_text",
                    purpose,
                    target,
                    started,
                    None,
                    &prompt,
                    "",
                    Some(format!("{e:#}")),
                );
                return Err(e);
            }
        };
        let status = res.status();
        let body = match res.text().await.context("failed to read model response") {
            Ok(body) => body,
            Err(e) => {
                log_llm_call(
                    "complete_text",
                    purpose,
                    target,
                    started,
                    None,
                    &prompt,
                    "",
                    Some(format!("{e:#}")),
                );
                return Err(e);
            }
        };
        if !status.is_success() {
            let msg = format!(
                "{} returned {}: {}",
                target.model,
                status,
                Self::body_excerpt(&body)
            );
            log_llm_call(
                "complete_text",
                purpose,
                target,
                started,
                None,
                &prompt,
                "",
                Some(msg.clone()),
            );
            return Err(anyhow!("{msg}"));
        }
        let resp: Value = match serde_json::from_str(&body).context("invalid model response") {
            Ok(resp) => resp,
            Err(e) => {
                log_llm_call(
                    "complete_text",
                    purpose,
                    target,
                    started,
                    None,
                    &prompt,
                    "",
                    Some(format!("{e:#}")),
                );
                return Err(e);
            }
        };
        let usage = Self::parse_usage(&resp);
        if let (Some(u), Ok(mut total)) = (usage.as_ref(), self.usage.lock()) {
            total.add(u);
        }
        let content = Self::response_text(&resp);
        log_llm_call(
            "complete_text",
            purpose,
            target,
            started,
            usage.as_ref(),
            &prompt,
            &content,
            None,
        );
        if content.is_empty() {
            return Err(anyhow!("model '{}' returned an empty reply", target.model));
        }
        Ok(content)
    }

    /// [`Self::complete_json`] against an explicit endpoint/key/model.
    ///
    /// `purpose` names the pipeline stage and is recorded in the model-call log.
    /// `max_tokens` caps this call's completion; `None` uses
    /// [`MAX_OUTPUT_TOKENS`]. A capped reply that the ceiling cut mid-object is
    /// recovered by [`Self::extract_json`] rather than discarded, so capping a
    /// long list costs the tail and not the whole answer.
    pub async fn complete_json_on(
        &self,
        target: &ModelTarget,
        purpose: &str,
        system: &str,
        user: &str,
        interrupt: InterruptSignal,
        max_tokens: Option<u32>,
    ) -> Result<Value> {
        let started = std::time::Instant::now();
        let prompt = format!("{system}\n\n{user}");
        if interrupt.is_set() {
            log_llm_call(
                "complete_json",
                purpose,
                target,
                started,
                None,
                &prompt,
                "",
                Some("cancelled".to_owned()),
            );
            return Err(LucyError::Cancelled.into());
        }
        if target.model.trim().is_empty() {
            log_llm_call(
                "complete_json",
                purpose,
                target,
                started,
                None,
                &prompt,
                "",
                Some("no model selected".to_owned()),
            );
            return Err(anyhow!("no model selected"));
        }
        let corrected = target.for_endpoint();
        let target = &corrected;
        let payload = json!({"model":target.model,"messages":[{"role":"system","content":system},{"role":"user","content":user}],"temperature":0,"max_tokens":output_ceiling(max_tokens),"response_format":{"type":"json_object"}});
        let url = format!("{}/chat/completions", target.base_url);
        if interrupt.is_set() {
            log_llm_call(
                "complete_json",
                purpose,
                target,
                started,
                None,
                &prompt,
                "",
                Some("cancelled".to_owned()),
            );
            return Err(LucyError::Cancelled.into());
        }
        let res = match self
            .post(&url, target.api_key.as_deref(), &payload, &interrupt)
            .await
        {
            Ok(res) => res,
            Err(e) => {
                log_llm_call(
                    "complete_json",
                    purpose,
                    target,
                    started,
                    None,
                    &prompt,
                    "",
                    Some(format!("{e:#}")),
                );
                return Err(e);
            }
        };
        let status = res.status();
        let body = match res
            .text()
            .await
            .context("failed to read JSON model response")
        {
            Ok(body) => body,
            Err(e) => {
                log_llm_call(
                    "complete_json",
                    purpose,
                    target,
                    started,
                    None,
                    &prompt,
                    "",
                    Some(format!("{e:#}")),
                );
                return Err(e);
            }
        };
        if !status.is_success() {
            let msg = format!(
                "{} returned {}: {}",
                target.model,
                status,
                Self::body_excerpt(&body)
            );
            log_llm_call(
                "complete_json",
                purpose,
                target,
                started,
                None,
                &prompt,
                "",
                Some(msg.clone()),
            );
            return Err(anyhow!("{msg}"));
        }
        let resp: Value = match serde_json::from_str(&body).context("invalid JSON model response") {
            Ok(resp) => resp,
            Err(e) => {
                log_llm_call(
                    "complete_json",
                    purpose,
                    target,
                    started,
                    None,
                    &prompt,
                    "",
                    Some(format!("{e:#}")),
                );
                return Err(e);
            }
        };
        let usage = Self::parse_usage(&resp);
        if let (Some(u), Ok(mut total)) = (usage.as_ref(), self.usage.lock()) {
            total.add(u);
        }
        let content = Self::response_text(&resp);
        if content.is_empty() {
            log_llm_call(
                "complete_json",
                purpose,
                target,
                started,
                usage.as_ref(),
                &prompt,
                "",
                Some("JSON model returned no content".to_owned()),
            );
            return Err(anyhow!("JSON model returned no content"));
        }
        match Self::extract_json(&content) {
            Ok(value) => {
                log_llm_call(
                    "complete_json",
                    purpose,
                    target,
                    started,
                    usage.as_ref(),
                    &prompt,
                    &content,
                    None,
                );
                Ok(value)
            }
            Err(e) => {
                log_llm_call(
                    "complete_json",
                    purpose,
                    target,
                    started,
                    usage.as_ref(),
                    &prompt,
                    &content,
                    Some(format!("{e:#}")),
                );
                Err(e)
            }
        }
    }

    /// `post_json` with an optional key: a `None` key sends no `Authorization`
    /// header at all, which local OpenAI-compatible servers expect.
    async fn post(
        &self,
        url: &str,
        api_key: Option<&str>,
        payload: &Value,
        interrupt: &InterruptSignal,
    ) -> Result<reqwest::Response> {
        match api_key.filter(|k| !k.trim().is_empty()) {
            Some(key) => self.post_json(url, Some(key), payload, interrupt).await,
            None => self.post_json(url, None, payload, interrupt).await,
        }
    }

    /// POST a chat completion, read the body, and parse it, adding any reported
    /// usage to the provider's cumulative total.
    ///
    /// Usage is counted before the caller inspects the content, because the
    /// tokens were spent whether or not the turn turns out to be usable.
    async fn chat_completion(
        &self,
        url: &str,
        api_key: Option<&str>,
        model: &str,
        payload: &Value,
        interrupt: &InterruptSignal,
    ) -> Result<Value> {
        let res = self.post(url, api_key, payload, interrupt).await?;
        let status = res.status();
        let body = res.text().await.context("failed to read model response")?;
        if !status.is_success() {
            return Err(anyhow!(
                "{} returned {}: {}",
                model,
                status,
                Self::body_excerpt(&body)
            ));
        }
        let resp: Value = serde_json::from_str(&body).context("invalid model response")?;
        if let (Some(u), Ok(mut total)) = (Self::parse_usage(&resp), self.usage.lock()) {
            total.add(&u);
        }
        Ok(resp)
    }

    /// One chat-completion turn that may call tools, against an explicit
    /// endpoint/key/model.
    ///
    /// Unlike [`Self::complete_json_on`], the answer is not forced into a JSON
    /// document: the model either emits `tool_calls` or speaks, and this returns
    /// both parts so the caller can run the calls and report the prose without a
    /// second round trip. A turn with neither is an error — a model that says
    /// nothing has not answered, and silently returning an empty turn would read
    /// as "nothing to do".
    #[allow(clippy::too_many_arguments)]
    pub async fn complete_with_tools_on(
        &self,
        target: &ModelTarget,
        purpose: &str,
        system: &str,
        user: &str,
        tools: &[Value],
        interrupt: InterruptSignal,
        max_tokens: Option<u32>,
    ) -> Result<AssistantTurn> {
        const OP: &str = "complete_with_tools";
        let started = std::time::Instant::now();
        let prompt = format!("{system}\n\n{user}");
        if interrupt.is_set() {
            log_llm_call(
                OP,
                purpose,
                target,
                started,
                None,
                &prompt,
                "",
                Some("cancelled".to_owned()),
            );
            return Err(LucyError::Cancelled.into());
        }
        if target.model.trim().is_empty() {
            log_llm_call(
                OP,
                purpose,
                target,
                started,
                None,
                &prompt,
                "",
                Some("no model selected".to_owned()),
            );
            return Err(anyhow!("no model selected"));
        }
        let corrected = target.for_endpoint();
        let target = &corrected;
        let payload = tool_chat_payload(&target.model, system, user, tools, max_tokens);
        let url = format!("{}/chat/completions", target.base_url);
        let resp = match self
            .chat_completion(
                &url,
                target.api_key.as_deref(),
                &target.model,
                &payload,
                &interrupt,
            )
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                log_llm_call(
                    OP,
                    purpose,
                    target,
                    started,
                    None,
                    &prompt,
                    "",
                    Some(format!("{e:#}")),
                );
                return Err(e);
            }
        };
        match Self::parse_assistant_turn(&resp) {
            Ok(turn) => {
                log_llm_call(
                    OP,
                    purpose,
                    target,
                    started,
                    Self::parse_usage(&resp).as_ref(),
                    &prompt,
                    &Self::render_turn(&turn),
                    None,
                );
                Ok(turn)
            }
            Err(e) => {
                log_llm_call(
                    OP,
                    purpose,
                    target,
                    started,
                    Self::parse_usage(&resp).as_ref(),
                    &prompt,
                    "",
                    Some(format!("{e:#}")),
                );
                Err(e)
            }
        }
    }

    /// The assistant half of a chat completion: its prose and its tool calls.
    ///
    /// A tool-calling turn carries `content: null` (or empty) and a speaking turn
    /// carries no `tool_calls`, so both keys are optional and either may be
    /// absent. `content` falls back through the reasoning fields, because a
    /// reasoning model that spent its visible budget on reasoning has still
    /// answered and its text should not be dropped.
    pub fn parse_assistant_turn(resp: &Value) -> Result<AssistantTurn> {
        let text = Self::response_text(resp);
        let tool_calls = Self::parse_tool_calls(&resp["choices"][0]["message"])?;
        if tool_calls.is_empty() && text.is_empty() {
            return Err(anyhow!("model returned neither text nor tool calls"));
        }
        Ok(AssistantTurn {
            text: (!text.is_empty()).then_some(text),
            tool_calls,
        })
    }

    /// `message.tool_calls` as [`ToolCall`]s, with each `function.arguments`
    /// JSON string parsed into the `input` the executor wants.
    ///
    /// The three tolerances here are the places a generic OpenAI-compatible
    /// server varies, and all of them keep the turn usable instead of dropping
    /// it: a missing `id` is synthesised from the call's position (the id is
    /// what a later `tool` message must echo, and the position is unique within
    /// the turn), and empty arguments become an empty object rather than a
    /// parse failure. Arguments that are *present but not* JSON are an error:
    /// that call would execute against garbage input, and the caller has to hear
    /// about it.
    fn parse_tool_calls(message: &Value) -> Result<Vec<ToolCall>> {
        let Some(raw) = message.get("tool_calls").and_then(Value::as_array) else {
            return Ok(Vec::new());
        };
        let mut calls = Vec::with_capacity(raw.len());
        for (index, call) in raw.iter().enumerate() {
            let function = call.get("function");
            let name = function
                .and_then(|f| f.get("name"))
                .or_else(|| call.get("name"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|n| !n.is_empty())
                .ok_or_else(|| {
                    anyhow!(
                        "tool call {index} carries no function name: {}",
                        Self::truncate_body(&call.to_string())
                    )
                })?;
            let id = call
                .get("id")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .unwrap_or_else(|| format!("call_{index}"));
            let arguments = function
                .and_then(|f| f.get("arguments"))
                .or_else(|| call.get("arguments"));
            let input = match arguments {
                None | Some(Value::Null) => json!({}),
                Some(Value::String(s)) if s.trim().is_empty() => json!({}),
                Some(Value::String(s)) => serde_json::from_str::<Value>(s).map_err(|e| {
                    anyhow!(
                        "tool call '{name}' carried arguments that are not JSON: {e} — {}",
                        Self::truncate_body(s)
                    )
                })?,
                Some(other) => other.clone(),
            };
            calls.push(ToolCall {
                id,
                name: name.to_owned(),
                input,
            });
        }
        Ok(calls)
    }

    /// One-line-per-call rendering of a turn for the model-call log, so a
    /// tool-calling round trip is legible in the trace even when its `content`
    /// was `null`.
    fn render_turn(turn: &AssistantTurn) -> String {
        let mut out = Vec::new();
        if let Some(text) = &turn.text {
            out.push(text.clone());
        }
        for call in &turn.tool_calls {
            out.push(format!(
                "[tool_call {} {} {}]",
                call.id,
                call.name,
                Self::truncate_body(&call.input.to_string())
            ));
        }
        out.join("\n")
    }

    /// Parse model output robustly: strict JSON first, then fenced code blocks,
    /// then — for a reply the `max_tokens` ceiling cut short — the complete
    /// prefix of the document with its open containers closed, then the largest
    /// balanced `{...}` object embedded in prose.
    ///
    /// The last step is what makes capping safe. Without it a ceiling that
    /// lands inside the final array element costs the whole answer, because
    /// `largest_balanced_object` then finds one finished element and hands back
    /// a bare object the caller's schema no longer matches. With it the same
    /// cut costs only the element being written.
    pub fn extract_json(content: &str) -> Result<Value> {
        if let Ok(v) = serde_json::from_str::<Value>(content) {
            return Ok(v);
        }
        let trimmed = content.trim();
        for fence in ["```json", "```JSON", "```"] {
            if let Some(rest) = trimmed.strip_prefix(fence) {
                if let Some(end) = rest.rfind("```") {
                    if let Ok(v) = serde_json::from_str::<Value>(rest[..end].trim()) {
                        return Ok(v);
                    }
                } else if let Ok(v) = serde_json::from_str::<Value>(rest.trim()) {
                    return Ok(v);
                }
            }
        }
        // Order matters, and it is the opposite of what reads naturally. The
        // truncation repair goes FIRST because it keeps the document's shape:
        // a cut `{"objectives":[{…},{…},{…` still carries its `objectives` key,
        // whereas `largest_balanced_object` on the same text returns one bare
        // finished element. Tried second, it would silently replace a two-element
        // plan with a shape the caller's schema does not match. It returns
        // `None` on any document that ended cleanly, so a complete reply
        // embedded in prose still falls through to the scan below.
        if let Some(v) = truncated_json_prefix(trimmed) {
            return Ok(v);
        }
        if let Some(obj) = largest_balanced_object(trimmed) {
            if let Ok(v) = serde_json::from_str::<Value>(&obj) {
                return Ok(v);
            }
        }
        Err(anyhow!(
            "JSON model returned invalid JSON: {}",
            Self::truncate_body(content)
        ))
    }
}
/// The OpenAI chat-completions payload for one tool-capable turn.
///
/// Three deliberate choices, each of which some gateway rejects or some model
/// answers badly to:
///
/// - `tools` and `tool_choice` appear together or not at all. An empty `tools`
///   array is an error on some gateways and a silent no-op on others, and
///   `tool_choice` without `tools` is invalid outright — so a caller offering no
///   tools gets a plain completion request and a prose answer.
/// - `response_format: json_object` is never sent. It asks for JSON *text*, the
///   other occupant of the same reply channel as `tool_calls`; asking for both
///   yields either a JSON blob that merely narrates an intent to call a tool or
///   a 400 from a gateway that treats the two as exclusive. A model that wants
///   no tool leaves `tool_calls` empty and speaks in `content`, so both answers
///   still arrive in one round trip.
/// - `temperature` is 0: which tool to call is the one decision in this loop
///   that should not vary between two attempts at the same state.
pub fn tool_chat_payload(
    model: &str,
    system: &str,
    user: &str,
    tools: &[Value],
    max_tokens: Option<u32>,
) -> Value {
    let mut payload = json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": output_ceiling(max_tokens),
    });
    if let Some(functions) = openai_functions(tools) {
        payload["tools"] = Value::Array(functions);
        payload["tool_choice"] = json!("auto");
    }
    payload
}

/// The wire `tools` array, or `None` when there is nothing to offer.
///
/// A definition with no name is dropped rather than sent: an unnamed function
/// cannot be called back, so including it would only spend the model's
/// attention on an option it cannot take.
fn openai_functions(tools: &[Value]) -> Option<Vec<Value>> {
    let functions: Vec<Value> = tools.iter().filter_map(openai_function).collect();
    (!functions.is_empty()).then_some(functions)
}

/// One Lucy tool definition in the OpenAI `function` shape.
///
/// Lucy describes a tool as `{name, description, input_schema}` — one shape for
/// the local registry and the MCP bridge alike — while the API nests those same
/// fields under `function` and calls the schema `parameters`. Translating here
/// keeps a tool described in one place, where its executor already implements
/// it; a definition that is already in the wire shape is read through unchanged.
fn openai_function(definition: &Value) -> Option<Value> {
    let nested = definition.get("function").filter(|f| f.is_object());
    let name = definition
        .get("name")
        .or_else(|| nested.and_then(|f| f.get("name")))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|n| !n.is_empty())?;
    let mut function = serde_json::Map::new();
    function.insert("name".to_owned(), json!(name));
    if let Some(description) = definition
        .get("description")
        .or_else(|| nested.and_then(|f| f.get("description")))
        .and_then(Value::as_str)
    {
        function.insert("description".to_owned(), json!(description));
    }
    // `parameters` is required by the API even for a tool that takes no
    // arguments, and a bare empty object is what that tool looks like there.
    let parameters = definition
        .get("input_schema")
        .or_else(|| nested.and_then(|f| f.get("parameters")))
        .filter(|p| p.is_object())
        .cloned()
        .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
    function.insert("parameters".to_owned(), parameters);
    Some(json!({"type": "function", "function": function}))
}

/// Append one LLM call to the unified model-call log. Best-effort: never
/// fails, never carries credentials (the [`ModelTarget`] endpoint holds no key
/// material in the record — only the base URL and model name are stored).
#[allow(clippy::too_many_arguments)]
fn log_llm_call(
    operation: &'static str,
    purpose: &str,
    target: &ModelTarget,
    started: std::time::Instant,
    usage: Option<&TokenUsage>,
    prompt: &str,
    response: &str,
    err: Option<String>,
) {
    let mut rec = ModelCallRecord::new(
        ModelCallKind::Llm,
        operation,
        target.base_url.clone(),
        target.model.clone(),
        started.elapsed().as_millis() as u64,
    )
    .with_purpose(purpose)
    .with_excerpts(prompt, response);
    if let Some(u) = usage {
        rec = rec.with_usage(u.prompt_tokens, u.completion_tokens, u.total_tokens);
    }
    if let Some(e) = err {
        rec = rec.failed(e);
    }
    log_model_call(&rec);
}
/// Return the largest balanced `{...}` substring, respecting strings/escapes.
fn largest_balanced_object(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut best: Option<(usize, usize)> = None;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'{' {
            i += 1;
            continue;
        }
        let mut depth = 0i32;
        let mut in_str = false;
        let mut esc = false;
        let mut j = i;
        while j < bytes.len() {
            let b = bytes[j];
            if in_str {
                if esc {
                    esc = false;
                } else if b == b'\\' {
                    esc = true;
                } else if b == b'"' {
                    in_str = false;
                }
            } else {
                if b == b'"' {
                    in_str = true;
                } else if b == b'{' {
                    depth += 1;
                } else if b == b'}' {
                    depth -= 1;
                    if depth == 0 {
                        match best {
                            Some((_, len)) if j + 1 - i <= len => {}
                            _ => best = Some((i, j + 1 - i)),
                        }
                    }
                    break;
                }
            }
            j += 1;
        }
        i += 1;
    }
    best.map(|(s, l)| text[s..s + l].to_owned())
}

/// One open JSON container and the offset a repair may truncate it at.
struct TruncFrame {
    /// `{` or `[`.
    open: u8,
    /// Byte offset at which this container's last *finished* child ends, or
    /// `None` when it has none.
    ///
    /// Only separators record it, never the value itself: a value is finished
    /// exactly when a `,` or a closing bracket follows it, so cutting at that
    /// separator always lands on a boundary. Recording where a scalar *began*
    /// would cut mid-token instead.
    last_complete: Option<usize>,
}

/// The bracket that closes `open`.
fn closer_for(open: u8) -> u8 {
    if open == b'{' { b'}' } else { b']' }
}

/// Recover the longest parseable prefix of a document an output ceiling cut
/// short: drop the element being written, then close every open container.
///
/// `{"a":[{"x":1},{"x":2},{"x":3` becomes `{"a":[{"x":1},{"x":2}]}` — two of
/// three elements survive, which is the whole point of capping: the caller
/// loses the tail, not the answer.
///
/// Returns `None` whenever the cut is not cleanly recoverable — a document that
/// ended on its own, a reply cut inside a string, or a cut before the first
/// element closed — so the caller still gets its normal parse error rather than
/// a hollow value that matches no schema.
fn truncated_json_prefix(text: &str) -> Option<Value> {
    let bytes = text.as_bytes();
    // Start at the first container opener, so a reply that opened with prose is
    // repaired from the JSON rather than from its preamble.
    let start = bytes.iter().position(|b| *b == b'{' || *b == b'[')?;
    let mut stack: Vec<TruncFrame> = Vec::new();
    let mut in_str = false;
    let mut esc = false;

    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_str {
            if esc {
                esc = false;
            } else if b == b'\\' {
                esc = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' | b'[' => stack.push(TruncFrame {
                open: b,
                last_complete: None,
            }),
            b'}' | b']' => {
                let frame = stack.pop()?;
                // A closer that does not match the opener it closes means this was
                // never truncated JSON of a shape this can repair — a `}` inside
                // an array, say. Comparing `frame.open` to `b` directly is the
                // classic slip here, because an object opens with `{` and closes
                // with `}`, so that check rejects every well-formed document.
                if closer_for(frame.open) != b {
                    return None;
                }
                // The container that just closed is a finished child of its
                // parent, which may therefore be truncated just past it.
                if let Some(parent) = stack.last_mut() {
                    parent.last_complete = Some(i + 1);
                }
            }
            // Everything before a comma is a finished element.
            b',' => {
                if let Some(frame) = stack.last_mut() {
                    frame.last_complete = Some(i);
                }
            }
            // `:`, whitespace and scalar bodies carry no structure worth
            // recording: their values are picked up by the separator rules.
            _ => {}
        }
    }
    // A document that ended on its own needs no repair.
    if stack.is_empty() {
        return None;
    }
    // An unterminated string does NOT rule out a repair: the cut usually lands
    // inside the *next* key or value, and the offset recorded at the previous
    // separator is still a clean boundary. Refusing here would throw away the
    // two finished objectives of a plan whose third was cut mid-word. The guard
    // that does matter is below — no recorded boundary means no recovery.

    // Close from the innermost container outward. Once a frame yields a cut
    // offset, every frame still open above it closes at that same offset, which
    // is what keeps the outer shape intact when the element being written was
    // dropped. A frame with no finished child of its own is dropped instead —
    // but only while no offset has been found yet, so the first frame with one
    // is always the innermost thing that can be kept.
    let mut end: Option<usize> = None;
    let mut closers: Vec<u8> = Vec::new();
    while let Some(frame) = stack.pop() {
        if end.is_none() {
            end = frame.last_complete;
        }
        if end.is_some() {
            closers.push(if frame.open == b'{' { b'}' } else { b']' });
        }
    }
    let end = end?;
    let mut repaired = String::with_capacity(end - start + closers.len());
    repaired.push_str(&text[start..end]);
    // Already innermost-first, which is the order they must be written in.
    for c in &closers {
        repaired.push(*c as char);
    }
    serde_json::from_str::<Value>(&repaired).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    /// The planner's reply, cut mid-objective by the output ceiling.
    ///
    /// This is the shape that makes capping safe. Without recovery the same cut
    /// loses the whole plan: `largest_balanced_object` finds the two finished
    /// objectives and hands back a bare array element, which no longer matches
    /// the `{"objectives":[…]}` shape the parser looks for.
    #[test]
    fn a_ceiling_that_cuts_the_last_objective_keeps_the_ones_before_it() {
        let cut = r#"{"objectives":[
            {"description":"open the site","success_check":"the page is loaded","success_probe":"location.pathname.length>1","suggested_action":"click the search field","action":"type","text":"a"},
            {"description":"enter the query","success_check":"results are listed","success_probe":"document.querySelectorAll('li').length>0","suggested_action":"click the field labelled Search","action":"type","text":"b"},
            {"description":"open the matching resu"#;
        let v = OpenAIProvider::extract_json(cut).expect("a cut reply is recoverable");
        let arr = v["objectives"].as_array().expect("wrapper key survives");
        assert_eq!(
            arr.len(),
            2,
            "the element being written is dropped: {arr:?}"
        );
        assert_eq!(arr[1]["text"], "b", "the last complete element is intact");
    }

    /// The same recovery has to work for a bare top-level array, which is a
    /// shape [`OpenAIProvider::extract_json`] accepts.
    #[test]
    fn a_cut_top_level_array_keeps_its_finished_elements() {
        let cut = r#"[{"description":"one"},{"description":"two"},{"descri"#;
        let v = OpenAIProvider::extract_json(cut).expect("a cut array is recoverable");
        let arr = v.as_array().expect("still an array");
        assert_eq!(arr.len(), 2, "{arr:?}");
    }

    /// Recovery must not invent structure where there was nothing complete to
    /// keep: a reply cut inside its first element has no salvageable prefix, and
    /// the caller needs its normal parse error more than a hollow object.
    #[test]
    fn a_cut_before_the_first_element_is_not_recovered() {
        assert!(OpenAIProvider::extract_json(r#"{"objectives":[{"descri"#).is_err());
        assert!(OpenAIProvider::extract_json(r#"{"needs_actions": tr"#).is_err());
    }

    /// A complete reply must never be rewritten by the recovery path, including
    /// one that would parse differently if a trailing comma were tolerated.
    #[test]
    fn a_complete_reply_is_left_alone() {
        let v = OpenAIProvider::extract_json(r#"{"objectives":[{"description":"one"},{"two":2}]}"#)
            .unwrap();
        assert_eq!(v["objectives"].as_array().unwrap().len(), 2);
        assert!(truncated_json_prefix(r#"{"a":[1,2]}"#).is_none());
        assert!(truncated_json_prefix("[1, 2, 3]").is_none());
    }

    /// A scalar is a finished value, so a cut right after one is recoverable
    /// even though no element ever closed.
    #[test]
    fn a_cut_after_a_scalar_keeps_it() {
        let v = OpenAIProvider::extract_json(r#"{"needs_actions": true, "note"#).unwrap();
        assert_eq!(v["needs_actions"], true);
    }

    /// A Lucy tool definition in the wire `function` shape, with the two
    /// payload keys that must accompany it.
    fn tool_catalog() -> Value {
        json!({
            "name": "read_file",
            "description": "Read a UTF-8 text file.",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        })
    }

    fn chat_response(message: Value) -> Value {
        json!({
            "choices": [{"message": message}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        })
    }

    /// The two keys travel together: `tools` without `tool_choice` is invalid,
    /// and `response_format: json_object` contradicts `tool_calls` because both
    /// claim the same reply channel.
    #[test]
    fn an_offered_catalog_sends_the_tool_keys_and_no_response_format() {
        let p = tool_chat_payload(
            "m",
            "sys",
            "usr",
            &[tool_catalog()],
            Some(512),
        );
        let tools = p["tools"].as_array().expect("tools is an array");
        assert_eq!(tools.len(), 1);
        let f = &tools[0]["function"];
        assert_eq!(f["name"], "read_file", "the name must survive nesting");
        assert_eq!(f["description"], "Read a UTF-8 text file.");
        // Lucy's `input_schema` is the API's `parameters`, unchanged.
        assert_eq!(f["parameters"], tool_catalog()["input_schema"]);
        assert_eq!(p["tool_choice"], "auto");
        assert!(
            p.get("response_format").is_none(),
            "forcing JSON text would compete with tool_calls: {p}"
        );
        assert_eq!(p["max_tokens"], 512);
    }

    /// A definition already in the wire shape is read through rather than
    /// double-nested, so a caller holding an MCP/catalog payload can pass it
    /// unchanged.
    #[test]
    fn a_definition_already_in_wire_shape_passes_through() {
        let wire = json!({"type": "function", "function": {
            "name": "list_dir",
            "description": "List files.",
            "parameters": {"type": "object"},
        }});
        let p = tool_chat_payload("m", "s", "u", &[wire], None);
        assert_eq!(p["tools"][0]["function"]["name"], "list_dir");
        assert_eq!(p["tools"][0]["function"]["parameters"], json!({"type": "object"}));
    }

    /// An absent catalog is a plain completion request, not an empty `tools`
    /// array plus a `tool_choice` with nothing to choose from.
    #[test]
    fn no_offered_tools_omits_both_keys() {
        for catalog in [
            Vec::new(),
            // A definition the model could never call back is not a tool.
            vec![json!({"description": "nameless", "input_schema": {}})],
        ] {
            let p = tool_chat_payload("m", "s", "u", &catalog, None);
            assert!(p.get("tools").is_none(), "{p}");
            assert!(p.get("tool_choice").is_none(), "{p}");
            assert!(p.get("response_format").is_none(), "{p}");
            // A call with no tools still has to be answerable.
            assert_eq!(p["model"], "m");
            assert_eq!(p["messages"][0]["role"], "system");
            assert_eq!(p["messages"][1]["content"], "u");
        }
    }

    #[test]
    fn tool_calls_become_named_inputs_with_parsed_arguments() {
        let turn = OpenAIProvider::parse_assistant_turn(&chat_response(json!({
            "content": null,
            "tool_calls": [
                {"id": "call_a", "type": "function",
                 "function": {"name": "read_file", "arguments": r#"{"path":"a.txt"}"#}},
                {"id": "call_b", "type": "function",
                 "function": {"name": "git", "arguments": r#"{"args":["status","--short"]}"#}},
            ],
        })))
        .expect("a tool-calling turn parses");
        assert_eq!(turn.text, None, "no prose was sent");
        assert_eq!(turn.tool_calls.len(), 2);
        assert_eq!(turn.tool_calls[0].id, "call_a");
        assert_eq!(turn.tool_calls[0].name, "read_file");
        assert_eq!(turn.tool_calls[0].input, json!({"path": "a.txt"}));
        assert_eq!(
            turn.tool_calls[1].input,
            json!({"args": ["status", "--short"]}),
            "arguments are parsed, not passed as a string"
        );
    }

    /// A turn may carry prose *and* calls — the model can say what it is about
    /// to do — so neither key may be read as "the other one is empty".
    #[test]
    fn a_turn_can_carry_prose_and_calls_together() {
        let turn = OpenAIProvider::parse_assistant_turn(&chat_response(json!({
            "content": "reading it now",
            "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}],
        })))
        .expect("parses");
        assert_eq!(turn.text.as_deref(), Some("reading it now"));
        assert_eq!(turn.tool_calls.len(), 1);
    }

    /// Reasoning models move the whole turn into `reasoning_content` and leave
    /// `content` null; a tool-less turn answered entirely there has still
    /// answered, and dropping it would read as silence.
    #[test]
    fn a_reply_held_in_the_reasoning_field_still_counts_as_text() {
        let turn = OpenAIProvider::parse_assistant_turn(&chat_response(json!({
            "content": null,
            "reasoning_content": "there is nothing to do",
        })))
        .expect("parses");
        assert_eq!(turn.text.as_deref(), Some("there is nothing to do"));
        assert!(turn.tool_calls.is_empty());
    }

    /// The id is what the `tool` result message must echo, and arguments are
    /// optional: a no-argument call is `{}`, not a parse failure, and a server
    /// that omits the id still yields a usable turn because position is unique
    /// within it.
    #[test]
    fn a_call_without_an_id_or_arguments_still_parses() {
        let turn = OpenAIProvider::parse_assistant_turn(&chat_response(json!({
            "tool_calls": [
                {"function": {"name": "list_dir", "arguments": "  "}},
                {"function": {"name": "task_next", "arguments": null}},
            ],
        })))
        .expect("parses");
        assert_eq!(turn.tool_calls[0].id, "call_0");
        assert_eq!(turn.tool_calls[1].id, "call_1");
        assert_eq!(turn.tool_calls[0].input, json!({}));
        assert_eq!(turn.tool_calls[1].input, json!({}));
    }

    /// Arguments that are present but unparseable must fail the turn: the call
    /// would execute against garbage input, and the caller has to hear about it.
    #[test]
    fn arguments_that_are_not_json_fail_the_turn() {
        let err = OpenAIProvider::parse_assistant_turn(&chat_response(json!({
            "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": "path=a.txt"}}],
        })))
        .expect_err("unparseable arguments");
        assert!(err.to_string().contains("read_file"), "{err:#}");
    }

    /// Neither half present is not a completed turn. Returning it as empty would
    /// read downstream as "the model decided nothing needs doing".
    #[test]
    fn a_reply_with_neither_text_nor_calls_is_an_error() {
        assert!(OpenAIProvider::parse_assistant_turn(&chat_response(json!({"content": null}))).is_err());
        assert!(OpenAIProvider::parse_assistant_turn(&chat_response(json!({"content": "  "}))).is_err());
    }

    #[test]
    fn extracts_strict_json() {
        assert_eq!(
            OpenAIProvider::extract_json(r#"{"mode":"chat","reply":"hi"}"#).unwrap()["mode"],
            "chat"
        );
    }
    #[test]
    fn extracts_fenced_json() {
        assert_eq!(
            OpenAIProvider::extract_json("```json\n{\"mode\":\"chat\",\"reply\":\"hi\"}\n```")
                .unwrap()["mode"],
            "chat"
        );
    }
    #[test]
    fn extracts_prose_embedded_json() {
        let v = OpenAIProvider::extract_json(
            "Sure! Here you go: {\"mode\":\"chat\",\"reply\":\"hi\"} hope that helps.",
        )
        .unwrap();
        assert_eq!(v["mode"], "chat");
    }
    #[test]
    fn rejects_plain_text() {
        assert!(OpenAIProvider::extract_json("I am playing the video now.").is_err());
    }
    #[test]
    fn model_target_trims_and_drops_blank_keys() {
        let t = ModelTarget::new("http://x.test/v1/", Some("  ".into()), "flash");
        assert_eq!(t.base_url, "http://x.test/v1");
        assert_eq!(t.api_key, None);
        assert_eq!(t.model, "flash");
    }

    #[test]
    fn openrouter_requests_carry_the_attribution_headers() {
        let client = Client::new();
        let url = "https://openrouter.ai/api/v1/chat/completions";
        let req = apply_endpoint_headers(client.post(url), url)
            .build()
            .expect("request builds");
        assert_eq!(
            req.headers()
                .get("http-referer")
                .and_then(|v| v.to_str().ok()),
            Some(lucy_config::OPENROUTER_REFERER)
        );
        assert_eq!(
            req.headers().get("x-title").and_then(|v| v.to_str().ok()),
            Some(lucy_config::OPENROUTER_TITLE)
        );
    }

    #[test]
    fn other_endpoints_get_no_extra_headers() {
        let client = Client::new();
        for url in [
            "https://api.groq.com/openai/v1/chat/completions",
            "http://127.0.0.1:11435/v1/chat/completions",
            "https://openrouter.ai.evil.test/v1/chat/completions",
        ] {
            let req = apply_endpoint_headers(client.post(url), url)
                .build()
                .expect("request builds");
            assert!(req.headers().get("http-referer").is_none(), "{url}");
            assert!(req.headers().get("x-title").is_none(), "{url}");
        }
    }

    #[test]
    fn an_openrouter_target_rewrites_an_unqualified_model() {
        let t = ModelTarget::new(
            "https://openrouter.ai/api/v1",
            Some("sk-or-x".into()),
            "gemini-web",
        );
        let fixed = t.for_endpoint();
        assert_eq!(fixed.model, lucy_config::OPENROUTER_DEFAULT_MODEL);
        assert_eq!(fixed.base_url, "https://openrouter.ai/api/v1");
        assert_eq!(fixed.api_key.as_deref(), Some("sk-or-x"));

        // A `vendor/model` id is already valid and is passed through.
        let ok = ModelTarget::new(
            "https://openrouter.ai/api/v1",
            Some("sk-or-x".into()),
            "z-ai/glm-5.3",
        );
        assert_eq!(ok.for_endpoint().model, "z-ai/glm-5.3");

        // Other endpoints are untouched.
        let local = ModelTarget::new("http://127.0.0.1:11435/v1", None, "gemini-web");
        assert_eq!(local.for_endpoint().model, "gemini-web");
    }

    #[test]
    fn model_target_resolves_provider_key_from_config() {
        let mut cfg = LucyConfig::default();
        cfg.providers = vec![lucy_config::ProviderConfig {
            id: "groq".into(),
            name: "Groq".into(),
            api_url: "https://api.groq.test".into(),
            api_key: "sk-x".into(),
            provider_type: lucy_config::ProviderType::Text,
            available_models: vec!["flash".into()],
            deprecated_models: Vec::new(),
        }];
        let t = ModelTarget::from_config(&cfg, "groq/flash").unwrap();
        assert_eq!(t.base_url, "https://api.groq.test/v1");
        assert_eq!(t.api_key.as_deref(), Some("sk-x"));
        assert_eq!(t.model, "flash");
    }

    #[test]
    fn balanced_object_respects_strings() {
        assert_eq!(
            largest_balanced_object(r#"a {"k":"} not end"} b"#).unwrap(),
            r#"{"k":"} not end"}"#
        );
    }

    #[test]
    fn loopback_endpoints_need_no_key() {
        for url in [
            "http://127.0.0.1:11435/v1",
            "http://localhost:11435/v1",
            "http://localhost:8001",
            "http://127.0.0.1:8080",
            "http://[::1]:11435/v1",
            "http://127.1.2.3/v1",
        ] {
            assert!(!endpoint_needs_key(url), "{url}");
        }
        for url in [
            "https://api.openai.com/v1",
            "https://api.groq.com/openai/v1",
            "https://example.com/v1",
            "",
        ] {
            assert!(endpoint_needs_key(url), "{url}");
        }
    }

    #[test]
    fn from_config_allows_keyless_loopback() {
        // Mirrors the user's setup: local OpenChat proxy, no key anywhere.
        let mut cfg = LucyConfig::default();
        cfg.models.default_text = "gemini-web".into();
        cfg.models.text_base_url = Some("http://127.0.0.1:11435/v1".into());
        cfg.models.text_api_key = None;
        cfg.models.api_key = None;
        cfg.models.base_url = None;
        assert!(OpenAIProvider::from_config(&cfg).is_ok());
    }

    #[test]
    fn from_config_still_requires_key_for_remote() {
        // Hermetic: point HOME at an empty dir so `~/.openchat/api_key`
        // (which exists on dev machines) can't rescue the lookup.
        static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = ENV_LOCK.lock().unwrap();
        let prev = std::env::var_os("HOME");
        let tmp = std::env::temp_dir().join(format!("lucy-no-home-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        unsafe { std::env::set_var("HOME", &tmp) };
        let mut cfg = LucyConfig::default();
        cfg.models.default_text = "gpt-5".into();
        cfg.models.text_base_url = Some("https://api.openai.com/v1".into());
        cfg.models.text_api_key = None;
        cfg.models.api_key = None;
        cfg.models.base_url = None;
        let res = OpenAIProvider::from_config(&cfg).is_err();
        if let Some(h) = prev {
            unsafe { std::env::set_var("HOME", h) };
        } else {
            unsafe { std::env::remove_var("HOME") };
        }
        let _ = std::fs::remove_dir_all(&tmp);
        assert!(res, "remote endpoint without any key must fail");
    }
}

#[cfg(test)]
mod endpoint_routing_tests {
    use super::*;

    fn provider_with(cfg: LucyConfig) -> OpenAIProvider {
        OpenAIProvider::from_config(&cfg).expect("provider builds")
    }

    /// Regression: a `provider_id/model` key must reach the provider that owns
    /// it. Sending `openrouter/…` to the legacy `text_base_url` (the local
    /// OpenChat proxy on 127.0.0.1:11435) fails with "Unknown model name".
    #[test]
    fn a_connected_provider_key_routes_to_that_provider_not_the_legacy_endpoint() {
        let mut cfg = LucyConfig::default();
        cfg.models.default_text = "openrouter/google/gemini-2.5-flash-lite".into();
        // The legacy endpoint is a DIFFERENT host that must not be used for a
        // key that names a connected provider.
        cfg.models.text_base_url = Some("http://127.0.0.1:11435/v1".into());
        cfg.models.text_api_key = Some("legacy-local-key".into());
        cfg.providers = vec![lucy_config::ProviderConfig {
            id: "openrouter".into(),
            name: "OpenRouter".into(),
            api_url: lucy_config::OPENROUTER_BASE_URL.into(),
            api_key: "sk-or-test".into(),
            provider_type: lucy_config::ProviderType::Text,
            available_models: vec!["google/gemini-2.5-flash-lite".into()],
            deprecated_models: Vec::new(),
        }];
        let p = provider_with(cfg);

        let t = p.target_for("openrouter/google/gemini-2.5-flash-lite");
        assert_eq!(t.base_url, lucy_config::OPENROUTER_BASE_URL);
        assert_eq!(t.api_key.as_deref(), Some("sk-or-test"));
        // The provider prefix is consumed, leaving the wire model id.
        assert_eq!(t.model, "google/gemini-2.5-flash-lite");
    }

    /// A bare model name has no provider to route to, so it must keep using the
    /// legacy endpoint — that is what a keyless local server setup relies on.
    #[test]
    fn a_bare_model_name_still_uses_the_legacy_endpoint() {
        let mut cfg = LucyConfig::default();
        cfg.models.default_text = "gemini-web".into();
        cfg.models.text_base_url = Some("http://127.0.0.1:11435/v1".into());
        cfg.providers = vec![lucy_config::ProviderConfig {
            id: "openrouter".into(),
            name: "OpenRouter".into(),
            api_url: lucy_config::OPENROUTER_BASE_URL.into(),
            api_key: "sk-or-test".into(),
            provider_type: lucy_config::ProviderType::Text,
            available_models: vec!["google/gemini-2.5-flash-lite".into()],
            deprecated_models: Vec::new(),
        }];
        let p = provider_with(cfg);

        let t = p.target_for("gemini-web");
        assert_eq!(t.base_url, "http://127.0.0.1:11435/v1");
        assert_eq!(t.model, "gemini-web");
    }

    /// An unknown prefix is not a provider reference: it must fall back rather
    /// than be treated as a provider id that happens to be missing.
    #[test]
    fn an_unknown_provider_prefix_falls_back_to_the_legacy_endpoint() {
        let mut cfg = LucyConfig::default();
        cfg.models.text_base_url = Some("http://127.0.0.1:11435/v1".into());
        let p = provider_with(cfg);
        let t = p.target_for("some-unknown-provider/flash");
        assert_eq!(t.base_url, "http://127.0.0.1:11435/v1");
    }

    /// Regression: OpenRouter sizes an un-bounded request against the model's
    /// full context and rejects it on a zero-credit key with 402. Every request
    /// Lucy sends must state an output ceiling.
    #[test]
    fn the_output_ceiling_is_finite_and_affordable() {
        assert!(MAX_OUTPUT_TOKENS > 0, "a zero ceiling returns no content");
        assert!(
            MAX_OUTPUT_TOKENS <= 16384,
            "a free-tier key cannot afford more than this per request: {MAX_OUTPUT_TOKENS}"
        );
    }
}
