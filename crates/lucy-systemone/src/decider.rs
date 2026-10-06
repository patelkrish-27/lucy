//! `decider-serve` (Mapika/decider-2b-vision) REST client.
//!
//! The classification server answers two endpoints and the settings screen
//! only ever asks for a base URL — **no API key**:
//!
//! - `GET {base}/health` → [`DeciderHealth`]
//! - `POST {base}/predict` → [`PredictOutcome`] (also accepted at `/decide`,
//!   `/v1/decide`, `/v1/systemone` for older builds)
//!
//! Both `lucy-systemone` and `lucy-runtime` share the types in this module so
//! the wire schema is defined exactly once. [`DeciderClient::classify_turn`]
//! wraps the two-question turn routing into a **single forward pass** and
//! returns a [`TurnClassification`].

use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use lucy_config::{ClassificationConfig, LucyConfig, ReasoningLevel};
use lucy_core::{ModelCallKind, ModelCallRecord, log_model_call};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use tracing::{debug, warn};

/// Endpoints tried in order. The first one that answers wins for the rest of
/// the client's life, so a probe costs one request, not four.
const PREDICT_PATHS: [&str; 4] = ["/predict", "/decide", "/v1/decide", "/v1/systemone"];

/// `GET {base}/health`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct DeciderHealth {
    /// Expected `"ok"`.
    #[serde(default)]
    pub status: String,
    /// Expected `"Mapika/decider-2b-vision"`.
    #[serde(default)]
    pub model: String,
    /// Torch device string, e.g. `"cuda:0"`.
    #[serde(default)]
    pub device: String,
    /// e.g. `"bfloat16"`.
    #[serde(default)]
    pub dtype: String,
    /// Free-form VRAM report (`{"total_gb": …, "used_gb": …, …}`).
    #[serde(default)]
    pub vram: Value,
    /// Whether the checkpoint is resident.
    #[serde(default)]
    pub loaded: bool,
    /// Anything else the server reported.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl DeciderHealth {
    /// True when the server answered and reports itself healthy.
    pub fn is_ok(&self) -> bool {
        let status_ok = self.status.is_empty()
            || self.status.eq_ignore_ascii_case("ok")
            || self.status.eq_ignore_ascii_case("healthy");
        status_ok && self.loaded
    }

    /// One-line status for the settings screen.
    pub fn summary(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if !self.model.is_empty() {
            parts.push(self.model.clone());
        }
        if !self.device.is_empty() {
            parts.push(format!("device {}", self.device));
        }
        if !self.dtype.is_empty() {
            parts.push(format!("dtype {}", self.dtype));
        }
        if let Some(vram) = self.vram_line() {
            parts.push(vram);
        }
        parts.push(if self.loaded { "loaded" } else { "not loaded" }.into());
        parts.join(" · ")
    }

    /// Compact VRAM line, tolerating either a string or a numeric object.
    pub fn vram_line(&self) -> Option<String> {
        match &self.vram {
            Value::String(s) if !s.trim().is_empty() => Some(format!("vram {s}")),
            Value::Object(map) => {
                let mut used: Vec<String> = Vec::new();
                for key in ["used_gb", "used", "allocated_gb"] {
                    if let Some(v) = map.get(key).and_then(value_as_gb) {
                        used.push(format!("{key} {v}"));
                    }
                }
                for key in ["total_gb", "total", "capacity_gb"] {
                    if let Some(v) = map.get(key).and_then(value_as_gb) {
                        used.push(format!("{key} {v}"));
                    }
                }
                for key in ["free_gb", "free"] {
                    if let Some(v) = map.get(key).and_then(value_as_gb) {
                        used.push(format!("{key} {v}"));
                    }
                }
                if used.is_empty() {
                    let mut fallback: Vec<String> = map
                        .iter()
                        .filter_map(|(k, v)| v.as_f64().map(|f| format!("{k} {f}")))
                        .take(3)
                        .collect();
                    fallback.sort();
                    if fallback.is_empty() {
                        None
                    } else {
                        Some(format!("vram {}", fallback.join(", ")))
                    }
                } else {
                    Some(format!("vram {}", used.join(", ")))
                }
            }
            Value::Number(n) => Some(format!("vram {n}")),
            _ => None,
        }
    }
}

fn value_as_gb(v: &Value) -> Option<String> {
    match v {
        Value::Number(n) => Some(format!("{} GB", n)),
        Value::String(s) => Some(s.clone()),
        _ => None,
    }
}

/// One question in the array-style `/predict` payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeciderQuestion {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub question: String,
    pub options: Vec<String>,
}

impl DeciderQuestion {
    pub fn new<I, S>(question: impl Into<String>, options: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            id: None,
            question: question.into(),
            options: options.into_iter().map(Into::into).collect(),
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }
}

/// `/predict` request body.
///
/// `questions` serializes as an **array** of `{question, options}` entries. The
/// Laya-style dict schema (`{"key": {"type": …, "instructions": …}}`) that
/// `decider-serve` also supports is reachable through
/// [`DeciderRequest::from_typed_questions`], which serializes `questions` as an
/// object instead — both are natively supported server-side.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeciderRequest {
    /// User command plus a brief note of the active state.
    pub context: String,
    /// Array-of-questions form (default).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub questions: Vec<DeciderQuestion>,
    /// Dict-of-typed-questions form. Mutually exclusive with `questions`.
    #[serde(
        default,
        rename = "typed_questions",
        skip_serializing_if = "HashMap::is_empty"
    )]
    pub typed_questions: HashMap<String, Value>,
    /// Optional screenshot / vision context, base64 or a URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    /// Alias accepted by servers that expect a URL rather than base64.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
}

impl DeciderRequest {
    pub const INTENT_QUESTION: &'static str = "what kind of request is this";
    pub const INTENT_OPT_CHAT: &'static str =
        "chat = the user wants a conversational reply, information, or explanation";
    pub const INTENT_OPT_ACT: &'static str =
        "act = the user wants an action performed (open app, play media, control device, etc.)";

    pub const LEVEL_QUESTION: &'static str =
        "what reasoning complexity level is required for this command";
    pub const LEVEL_OPT_1: &'static str = "1 = simple, direct command";
    pub const LEVEL_OPT_2: &'static str = "2 = moderate multi-step workflow";
    pub const LEVEL_OPT_3: &'static str = "3 = complex reasoning or planning";

    /// Third head: which knowledge topic this request will need.
    ///
    /// It is asked in the *same* forward pass as intent and level, so it costs
    /// nothing extra, and its answer only reorders a deterministic FTS result
    /// set — it never decides whether to search. That constraint is the whole
    /// design: the ablation work on small local models found an adaptive router
    /// losing to fixed hybrid retrieval, and a classifier asked to find
    /// knowledge unaided would hallucinate topics that do not exist.
    pub const KNOWLEDGE_QUESTION: &'static str =
        "which of these known topics does this request most likely involve";
    pub const KNOWLEDGE_OPT_NONE: &'static str = "none = none of these topics are relevant";
    /// Question id for the knowledge head, distinct from `q1`/`q2`.
    pub const KNOWLEDGE_QID: &'static str = "q3";

    /// The turn-routing request: one forward pass, two question heads.
    pub fn turn_routing(context: impl Into<String>) -> Self {
        Self {
            context: context.into(),
            questions: vec![
                DeciderQuestion::new(
                    Self::INTENT_QUESTION,
                    [Self::INTENT_OPT_CHAT, Self::INTENT_OPT_ACT],
                )
                .with_id("q1"),
                DeciderQuestion::new(
                    Self::LEVEL_QUESTION,
                    [Self::LEVEL_OPT_1, Self::LEVEL_OPT_2, Self::LEVEL_OPT_3],
                )
                .with_id("q2"),
            ],
            ..Self::default()
        }
    }

    /// The turn-routing request plus a knowledge question whose options are
    /// drawn from the topics that actually exist on disk.
    ///
    /// The options come from the live knowledge index rather than from any list
    /// in the code, so a new topic needs no code change and the model can never
    /// be offered a topic Lucy does not have.
    pub fn turn_routing_with_knowledge(context: impl Into<String>, topics: Vec<String>) -> Self {
        let mut request = Self::turn_routing(context);
        if topics.is_empty() {
            return request;
        }
        let mut options = topics;
        options.push(Self::KNOWLEDGE_OPT_NONE.to_owned());
        request.questions.push(
            DeciderQuestion::new(Self::KNOWLEDGE_QUESTION, options).with_id(Self::KNOWLEDGE_QID),
        );
        request
    }

    /// Attach a screenshot (raw base64) for vision-aware classification.
    pub fn with_image_base64(mut self, b64: impl Into<String>) -> Self {
        self.image = Some(b64.into());
        self
    }

    /// Attach an image URL for vision-aware classification.
    pub fn with_image_url(mut self, url: impl Into<String>) -> Self {
        self.image_url = Some(url.into());
        self
    }

    /// The wire JSON. `questions` is an array normally and an object when
    /// typed questions were supplied, matching the two schemas
    /// `decider-serve` accepts.
    pub fn to_json(&self) -> Value {
        let mut root = Map::new();
        root.insert("context".into(), Value::String(self.context.clone()));
        root.insert("state".into(), Value::String(self.context.clone()));
        if !self.typed_questions.is_empty() {
            let map: Map<String, Value> = self
                .typed_questions
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            root.insert("questions".into(), Value::Object(map));
        } else {
            root.insert(
                "questions".into(),
                serde_json::to_value(&self.questions).unwrap_or_else(|_| json!([])),
            );
        }
        if let Some(img) = &self.image {
            root.insert("image".into(), Value::String(img.clone()));
        }
        if let Some(url) = &self.image_url {
            root.insert("image_url".into(), Value::String(url.clone()));
        }
        Value::Object(root)
    }
}

/// Laya-style dict schema: `{"key": {"type": …, "instructions": …, "criteria": {…}}}`.
pub fn typed_question(kind: &str, instructions: &str, criteria: Option<Value>) -> Value {
    let mut q = Map::new();
    q.insert("type".into(), Value::String(kind.to_owned()));
    q.insert(
        "instructions".into(),
        Value::String(instructions.to_owned()),
    );
    if let Some(c) = criteria {
        q.insert("criteria".into(), c);
    }
    Value::Object(q)
}

/// Which of the two turn branches the classifier chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TurnBranch {
    /// `"yes"` — answer with text only.
    RequiresOnlyResponse,
    /// `"no"` — the goal needs system/browser actions.
    RequiresActions,
}

impl TurnBranch {
    pub fn from_wire(raw: &str) -> Option<Self> {
        let s = raw.trim().to_ascii_lowercase();
        if s.starts_with("act") || matches!(s.as_str(), "no" | "false" | "0" | "action" | "actions")
        {
            Some(Self::RequiresActions)
        } else if s.starts_with("chat") || matches!(s.as_str(), "yes" | "true" | "1" | "text") {
            Some(Self::RequiresOnlyResponse)
        } else {
            None
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RequiresOnlyResponse => "chat",
            Self::RequiresActions => "act",
        }
    }
    /// True when the goal needs actions rather than a text reply.
    pub fn needs_actions(&self) -> bool {
        matches!(self, Self::RequiresActions)
    }
}

/// One head's answer as it came off the wire.
#[derive(Debug, Clone, PartialEq)]
pub struct AnswerReading {
    /// The winning option label (`"yes"`, `"2"`, …).
    pub choice: String,
    /// Model-reported confidence, 0.0..=1.0.
    pub confidence: f64,
    /// Per-option probabilities, when the server reports them.
    pub probabilities: HashMap<String, f64>,
}

impl AnswerReading {
    fn from_value(v: &Value) -> Option<Self> {
        // Choice form: {"choice": "yes", "confidence": …, "probabilities": {…}}
        if let Some(choice) = v.get("choice").and_then(Value::as_str) {
            return Some(Self {
                choice: choice.trim().to_string(),
                confidence: v
                    .get("confidence")
                    .and_then(Value::as_f64)
                    .or_else(|| v.as_f64())
                    .unwrap_or(1.0),
                probabilities: probability_map(v.get("probabilities")),
            });
        }
        // Bare-label form: {"requires_only_response": "yes"}
        if let Some(s) = v.as_str() {
            return Some(Self {
                choice: s.trim().to_string(),
                confidence: 1.0,
                probabilities: HashMap::new(),
            });
        }
        // Bare-number form: {"reasoning_level": 3}, {"requires_only_response": 1}.
        if let Some(n) = v.as_f64() {
            return Some(Self {
                choice: (n.round() as i64).to_string(),
                confidence: 1.0,
                probabilities: HashMap::new(),
            });
        }
        // Numeric noul form: {"noul": 0.82}
        if let Some(n) = v.get("noul").and_then(Value::as_f64) {
            return Some(Self {
                choice: if n > 0.5 { "yes".into() } else { "no".into() },
                confidence: n,
                probabilities: HashMap::new(),
            });
        }
        // Score form: {"score": 2}
        if let Some(s) = v.get("score").and_then(Value::as_f64) {
            return Some(Self {
                choice: s.round().clamp(1.0, 3.0).to_string(),
                confidence: v.get("confidence").and_then(Value::as_f64).unwrap_or(1.0),
                probabilities: probability_map(v.get("probabilities")),
            });
        }
        // `{"1": 0.2, "2": 0.5, "3": 0.3}` — argmax over the object.
        if let Some(map) = v.as_object() {
            let probs: HashMap<String, f64> = map
                .iter()
                .filter_map(|(k, val)| val.as_f64().map(|f| (k.clone(), f)))
                .collect();
            if !probs.is_empty() {
                let (choice, conf) = probs
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(k, p)| (k.clone(), *p))
                    .unwrap_or_default();
                return Some(Self {
                    choice,
                    confidence: conf,
                    probabilities: probs,
                });
            }
        }
        None
    }
}

fn probability_map(v: Option<&Value>) -> HashMap<String, f64> {
    v.and_then(Value::as_object)
        .map(|m| {
            m.iter()
                .filter_map(|(k, val)| val.as_f64().map(|f| (k.clone(), f)))
                .collect()
        })
        .unwrap_or_default()
}

/// The parsed `answers` map from `/predict`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PredictOutcome {
    pub raw: Map<String, Value>,
}

impl PredictOutcome {
    /// Look up one answer by the key the question was asked under, falling
    /// back to a positional match when the server numbered the answers.
    pub fn answer(&self, key: &str) -> Option<AnswerReading> {
        if let Some(r) = self.raw.get(key).and_then(AnswerReading::from_value) {
            return Some(r);
        }
        if let Ok(idx) = key.parse::<usize>() {
            if let Some(r) = self
                .raw
                .get(&idx.to_string())
                .and_then(AnswerReading::from_value)
            {
                return Some(r);
            }
            if let Some(r) = self
                .raw
                .get(&format!("q{}", idx + 1))
                .and_then(AnswerReading::from_value)
            {
                return Some(r);
            }
        }
        None
    }
}

/// The result of the single forward pass that routes a turn.
#[derive(Debug, Clone, PartialEq)]
pub struct TurnClassification {
    /// `"yes"` (text-only) or `"no"` (needs actions).
    pub branch: TurnBranch,
    /// `1` / `2` / `3` reasoning complexity.
    pub reasoning_level: ReasoningLevel,
    /// Confidence of the branch head.
    pub confidence: f64,
    /// Per-option probabilities of the branch head, when reported.
    pub probabilities: HashMap<String, f64>,
    /// Confidence of the reasoning-level head.
    pub level_confidence: f64,
    /// Server-reported latency, in milliseconds.
    pub latency_ms: Option<f64>,
    /// Set by the caller when the verdict came from a degraded path, so the
    /// TUI can warn instead of silently pretending the classifier answered.
    pub summary_note: Option<String>,
    /// Knowledge topic the classifier picked, when the knowledge head ran and
    /// answered with something other than the "none" option.
    ///
    /// Advisory only: callers use it to *order* a deterministic recall result
    /// set, never to decide whether to search. Empty means "no signal", and an
    /// empty signal leaves recall exactly as it would have been.
    pub knowledge_topic: Option<String>,
}

impl TurnClassification {
    /// Terse line for the TUI chat log.
    pub fn summary(&self) -> String {
        format!(
            "intent: {} ({}%) · reasoning level {} ({})",
            self.branch.as_str(),
            (self.confidence * 100.0).round() as i64,
            self.reasoning_level.as_number(),
            self.reasoning_level.short()
        )
    }

    /// Read the verdict out of an `answers` map. Missing heads fall back to
    /// the safe defaults: answer in text at Level 2.
    pub fn from_outcome(outcome: &PredictOutcome, latency_ms: Option<f64>) -> Self {
        let branch_reading = outcome
            .answer("q1")
            .or_else(|| outcome.answer(DeciderRequest::INTENT_QUESTION))
            .or_else(|| outcome.answer("requires_only_response"))
            .or_else(|| outcome.answer("mode"))
            .or_else(|| outcome.answer("0"));
        let branch = branch_reading
            .as_ref()
            .and_then(|r| TurnBranch::from_wire(&r.choice))
            .unwrap_or(TurnBranch::RequiresOnlyResponse);
        let level_reading = outcome
            .answer("q2")
            .or_else(|| outcome.answer(DeciderRequest::LEVEL_QUESTION))
            .or_else(|| outcome.answer("reasoning_level"))
            .or_else(|| outcome.answer("1"))
            .or_else(|| outcome.answer("level"));
        let reasoning_level = level_reading
            .as_ref()
            .and_then(|r| ReasoningLevel::from_wire(&r.choice))
            .unwrap_or_default();
        Self {
            branch,
            reasoning_level,
            confidence: branch_reading.as_ref().map_or(0.0, |r| r.confidence),
            probabilities: branch_reading
                .as_ref()
                .map(|r| r.probabilities.clone())
                .unwrap_or_default(),
            level_confidence: level_reading.as_ref().map_or(0.0, |r| r.confidence),
            latency_ms,
            summary_note: None,
            knowledge_topic: outcome
                .answer(DeciderRequest::KNOWLEDGE_QID)
                .or_else(|| outcome.answer(DeciderRequest::KNOWLEDGE_QUESTION))
                .map(|r| normalize_knowledge_choice(&r.choice))
                .filter(|choice| choice.is_some())
                .flatten(),
        }
    }
}

/// Turn a knowledge-head answer into a topic slug, or `None` for "no signal".
///
/// The `"none = …"` option is part of the question, so the model has an honest
/// way to say nothing applies; a build that ignores it would report a topic on
/// every single turn.
fn normalize_knowledge_choice(choice: &str) -> Option<String> {
    let raw = choice.trim();
    if raw.is_empty() {
        return None;
    }
    let value = raw
        .split_once('=')
        .map(|(_, rest)| rest.trim())
        .unwrap_or(raw)
        .trim();
    if value.is_empty() {
        return None;
    }
    let lower = value.to_ascii_lowercase();
    if lower.starts_with("none") {
        return None;
    }
    Some(lucy_knowledge::store::slugify(value))
}

/// Client for the `decider-serve` REST API.
#[derive(Debug, Clone)]
pub struct DeciderClient {
    http: Client,
    config: ClassificationConfig,
    /// Index into [`PREDICT_PATHS`] that answered last time.
    predict_path: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl DeciderClient {
    pub fn new(config: ClassificationConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(Duration::from_millis(config.timeout_ms.max(1000)))
            .build()
            .context("failed to build HTTP client for the classification model")?;
        Ok(Self {
            http,
            config,
            // 0 = "/predict", the documented endpoint; the fallback sweep only
            // starts after a 404.
            predict_path: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        })
    }

    /// Build from the persisted config. No API key: the classification server
    /// is unauthenticated.
    pub fn from_lucy_config(cfg: &LucyConfig) -> Result<Self> {
        Self::new(cfg.classification.clone())
    }

    pub fn base_url(&self) -> &str {
        self.config.base_url()
    }

    pub fn is_enabled(&self) -> bool {
        self.config.enabled && !self.config.base_url().is_empty()
    }

    pub fn confidence_threshold(&self) -> f64 {
        self.config.confidence_threshold as f64
    }

    /// `GET {base}/health`.
    pub async fn health(&self) -> Result<DeciderHealth> {
        let url = format!("{}/health", self.base_url());
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .with_context(|| format!("could not reach the classification model at {url}"))?;
        let status = resp.status();
        let body = resp
            .text()
            .await
            .context("failed to read the /health response")?;
        if !status.is_success() {
            bail!(
                "{url} returned {status}: {}",
                body.chars().take(200).collect::<String>()
            );
        }
        serde_json::from_str(&body).with_context(|| {
            format!(
                "{url} returned a non-JSON body: {}",
                body.chars().take(120).collect::<String>()
            )
        })
    }

    /// `GET {base}/health` plus a minimal `POST /predict` probe, so `[Test]`
    /// reports both reachability and that inference actually works.
    pub async fn probe(&self) -> Result<DeciderProbe> {
        let health = self.health().await?;
        if !health.is_ok() {
            warn!(
                url = %self.base_url(),
                loaded = health.loaded,
                status = %health.status,
                "classification server is up but not reporting a loaded checkpoint"
            );
        }
        let mut request = DeciderRequest::turn_routing("health probe");
        request.context = "health probe".into();
        request.questions.truncate(1);
        let predict = self.predict_with_purpose(&request, "probe").await;
        let predict_note = match &predict {
            Ok(outcome) => match outcome.answer("requires_only_response") {
                Some(a) => format!("predict ok ({})", a.choice),
                None => "predict ok (no answers)".to_string(),
            },
            Err(e) => format!("predict failed: {e:#}"),
        };
        Ok(DeciderProbe {
            health,
            predict_ok: predict.is_ok(),
            predict_note,
        })
    }

    /// `POST {base}/predict` with a fallback sweep across the alternate paths.
    ///
    /// Every call is appended to the unified model-call log (`kind =
    /// classification`), including failures.
    pub async fn predict(&self, request: &DeciderRequest) -> Result<PredictOutcome> {
        self.predict_with_purpose(request, "predict").await
    }

    /// [`Self::predict`] with an explicit pipeline purpose for the log
    /// (`probe`, `predict`, …). [`Self::classify_turn`] logs its own verdict
    /// line instead, so it uses [`Self::predict_inner`] directly.
    pub async fn predict_with_purpose(
        &self,
        request: &DeciderRequest,
        purpose: &str,
    ) -> Result<PredictOutcome> {
        let started = std::time::Instant::now();
        let result = self.predict_inner(request).await;
        let latency_ms = started.elapsed().as_millis() as u64;
        let mut rec = ModelCallRecord::new(
            ModelCallKind::Classification,
            "predict",
            self.base_url().to_owned(),
            self.model_name(),
            latency_ms,
        )
        .with_purpose(purpose)
        .with_excerpts(&request.context, "");
        match &result {
            Ok(outcome) => {
                rec = rec.with_detail(format!("answers={}", outcome.raw.len()));
            }
            Err(e) => {
                rec = rec.failed(format!("{e:#}"));
            }
        }
        log_model_call(&rec);
        result
    }

    async fn predict_inner(&self, request: &DeciderRequest) -> Result<PredictOutcome> {
        use std::sync::atomic::Ordering;
        let body = request.to_json();
        let start = std::sync::atomic::AtomicUsize::load(&self.predict_path, Ordering::SeqCst);
        let mut last_err: Option<anyhow::Error> = None;
        for step in 0..PREDICT_PATHS.len() {
            let idx = (start + step) % PREDICT_PATHS.len();
            let path = PREDICT_PATHS[idx];
            let url = format!("{}{path}", self.base_url());
            match self
                .http
                .post(&url)
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .json(&body)
                .send()
                .await
            {
                Ok(resp) => {
                    let status = resp.status();
                    if status == reqwest::StatusCode::NOT_FOUND && step + 1 < PREDICT_PATHS.len() {
                        debug!(%url, "classification endpoint not found, trying next path");
                        last_err = Some(anyhow!("{url} returned 404"));
                        continue;
                    }
                    let text = resp
                        .text()
                        .await
                        .context("failed to read the /predict response")?;
                    if !status.is_success() {
                        bail!(
                            "{url} returned {status}: {}",
                            text.chars().take(200).collect::<String>()
                        );
                    }
                    // Pin the path that worked so later turns cost one request.
                    self.predict_path
                        .store(idx, std::sync::atomic::Ordering::SeqCst);
                    return Self::parse_predict(&text);
                }
                Err(e) => {
                    last_err = Some(anyhow!("{url}: {e:#}"));
                }
            }
        }
        Err(last_err.unwrap_or_else(|| {
            anyhow!("no classification endpoint answered at {}", self.base_url())
        }))
    }

    /// Accept `{"answers": {…}}`, a bare `{"…": {…}}` map, or
    /// `{"result": {"answers": {…}}}`.
    fn parse_predict(text: &str) -> Result<PredictOutcome> {
        let value: Value = serde_json::from_str(text).with_context(|| {
            format!(
                "non-JSON /predict body: {}",
                text.chars().take(160).collect::<String>()
            )
        })?;
        let answers = value
            .get("answers")
            .or_else(|| value.pointer("/result/answers"))
            .and_then(Value::as_object)
            .cloned()
            .or_else(|| {
                value
                    .as_object()
                    .filter(|m| !m.contains_key("latency_ms") && !m.contains_key("model"))
                    .cloned()
            })
            .ok_or_else(|| anyhow!("/predict response had no 'answers' object: {text}"))?;
        Ok(PredictOutcome { raw: answers })
    }

    /// The single forward pass that routes a turn: intent + reasoning level in
    /// one `POST /predict`. The verdict (branch, level, confidence) is
    /// appended to the unified model-call log.
    pub async fn classify_turn(&self, context: &str) -> Result<TurnClassification> {
        self.classify_turn_with_knowledge(context, &[]).await
    }

    /// [`Self::classify_turn`] plus an optional third head naming the knowledge
    /// topic the request looks like it involves.
    ///
    /// `topics` are slugs read from the live knowledge index. An empty slice is
    /// exactly [`Self::classify_turn`], so a caller with no knowledge base pays
    /// nothing and a server that only knows how to answer two heads is
    /// unaffected — the extra head is dropped when there is nothing to ask
    /// about.
    pub async fn classify_turn_with_knowledge(
        &self,
        context: &str,
        topics: &[String],
    ) -> Result<TurnClassification> {
        let started = std::time::Instant::now();
        let request = DeciderRequest::turn_routing_with_knowledge(context, topics.to_vec());
        let outcome = match self.predict_inner(&request).await {
            Ok(outcome) => outcome,
            Err(e) => {
                log_model_call(
                    &ModelCallRecord::new(
                        ModelCallKind::Classification,
                        "classify_turn",
                        self.base_url().to_owned(),
                        self.model_name(),
                        started.elapsed().as_millis() as u64,
                    )
                    .with_purpose("turn_routing")
                    .with_excerpts(context, "")
                    .failed(format!("{e:#}")),
                );
                return Err(e);
            }
        };
        let mut classification = TurnClassification::from_outcome(
            &outcome,
            Some(started.elapsed().as_secs_f64() * 1000.0),
        );
        if classification.confidence == 0.0 {
            // No confidence reported: treat the verdict as certain rather than
            // discarding a perfectly clear answer.
            classification.confidence = 1.0;
        }
        log_model_call(
            &ModelCallRecord::new(
                ModelCallKind::Classification,
                "classify_turn",
                self.base_url().to_owned(),
                self.model_name(),
                started.elapsed().as_millis() as u64,
            )
            .with_purpose("turn_routing")
            .with_excerpts(context, "")
            .with_detail(format!(
                "branch={} level={} conf={:.2} level_conf={:.2} knowledge={}",
                classification.branch.as_str(),
                classification.reasoning_level.as_number(),
                classification.confidence,
                classification.level_confidence,
                classification.knowledge_topic.as_deref().unwrap_or("-"),
            )),
        );
        Ok(classification)
    }

    /// Model label for the log. The server owns the checkpoint
    /// (`ClassificationConfig::model` is informational and often empty).
    fn model_name(&self) -> String {
        let m = self.config.model.trim();
        if m.is_empty() {
            "decider-2b-vision".to_owned()
        } else {
            m.to_owned()
        }
    }
}

/// What the settings `[Test]` button reports back.
#[derive(Debug, Clone, PartialEq)]
pub struct DeciderProbe {
    pub health: DeciderHealth,
    pub predict_ok: bool,
    pub predict_note: String,
}

impl DeciderProbe {
    /// One-line summary: model / device / VRAM status.
    pub fn summary(&self) -> String {
        let mut out = format!("ok — {}", self.health.summary());
        if !self.predict_ok {
            out.push_str(&format!(" · {}", self.predict_note));
        }
        out
    }
}

/// Heuristic fallback used when `decider-serve` is unreachable.
///
/// Keeps the app usable with no classifier at all: obvious UI verbs route to
/// the action branch, everything else answers in text at Level 2.
pub fn heuristic_turn_classification(prompt: &str) -> TurnClassification {
    let lower = prompt.to_ascii_lowercase();
    let action_verbs = [
        "open ",
        "launch",
        "click",
        "type ",
        "press ",
        "run ",
        "search",
        "play ",
        "install",
        "close ",
        "switch ",
        "focus ",
        "go to",
        "navigate",
        "screenshot",
        "download",
        "scroll",
        "move ",
        "resize",
        "minimize",
        "maximize",
        "copy ",
        "paste",
        "delete",
        "create ",
        "set ",
        "start ",
        "stop ",
        "restart",
        "mute",
        "volume",
        "brightness",
        "workspace",
        "tab ",
    ];
    let question_words = [
        "what", "why", "how", "when", "where", "who", "which", "explain",
    ];
    let first = lower.split_whitespace().next().unwrap_or_default();
    let mentions_action = action_verbs.iter().any(|v| lower.contains(v));
    let starts_with_action = action_verbs.iter().any(|v| lower.starts_with(v));
    let needs_actions = matches!(
        first,
        "open" | "launch" | "run" | "play" | "install" | "search" | "click" | "type" | "press"
    ) || (starts_with_action && !lower.contains('?'))
        || (lower.contains(" && ") || lower.starts_with('&'))
        || (mentions_action
            && !lower.contains('?')
            && !question_words.iter().any(|q| lower.starts_with(q)));

    let words = lower.split_whitespace().count();
    let level = if needs_actions {
        ReasoningLevel::L3
    } else if words > 60 || lower.contains("step by step") || lower.contains("compare") {
        ReasoningLevel::L2
    } else {
        ReasoningLevel::L1
    };
    TurnClassification {
        branch: if needs_actions {
            TurnBranch::RequiresActions
        } else {
            TurnBranch::RequiresOnlyResponse
        },
        reasoning_level: level,
        confidence: 0.0,
        probabilities: HashMap::new(),
        level_confidence: 0.0,
        latency_ms: None,
        summary_note: None,
        // No classifier ran, so there is no knowledge signal. Recall proceeds
        // on its deterministic ranking, which is the point of the fallback.
        knowledge_topic: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Question;

    fn client(base: &str) -> DeciderClient {
        DeciderClient::new(ClassificationConfig {
            enabled: true,
            classification_api_url: base.into(),
            timeout_ms: 1000,
            model: String::new(),
            confidence_threshold: 0.1,
        })
        .unwrap()
    }

    #[test]
    fn health_parses_documented_payload() {
        let raw = r#"{"status":"ok","model":"Mapika/decider-2b-vision","device":"cuda:0",
            "dtype":"bfloat16","vram":{"total_gb":24.0,"used_gb":7.5},"loaded":true}"#;
        let h: DeciderHealth = serde_json::from_str(raw).unwrap();
        assert!(h.is_ok());
        assert_eq!(h.model, "Mapika/decider-2b-vision");
        let s = h.summary();
        assert!(s.contains("Mapika/decider-2b-vision"), "{s}");
        assert!(s.contains("device cuda:0"), "{s}");
        assert!(s.contains("dtype bfloat16"), "{s}");
        assert!(s.contains("used_gb"), "{s}");
        assert!(s.ends_with("loaded"), "{s}");
    }

    #[test]
    fn health_not_ok_when_checkpoint_missing() {
        let h: DeciderHealth = serde_json::from_str(r#"{"status":"ok","loaded":false}"#).unwrap();
        assert!(!h.is_ok());
        assert!(h.summary().contains("not loaded"));
    }

    #[test]
    fn health_vram_accepts_string_form() {
        let h: DeciderHealth =
            serde_json::from_str(r#"{"vram":"24.0/7.5 GB","loaded":true,"status":"ok"}"#).unwrap();
        assert!(h.summary().contains("vram 24.0/7.5 GB"), "{}", h.summary());
    }

    #[test]
    fn turn_request_matches_spec_payload() {
        let json = DeciderRequest::turn_routing("open yt & play this song").to_json();
        assert_eq!(json["context"], "open yt & play this song");
        assert_eq!(json["state"], "open yt & play this song");
        let qs = json["questions"].as_array().unwrap();
        assert_eq!(qs.len(), 2);
        assert_eq!(qs[0]["id"], "q1");
        assert_eq!(qs[0]["question"], DeciderRequest::INTENT_QUESTION);
        assert!(
            qs[0]["options"]
                .as_array()
                .unwrap()
                .iter()
                .any(|o| o.as_str().unwrap().starts_with("act"))
        );
        assert_eq!(qs[1]["id"], "q2");
        assert_eq!(qs[1]["question"], DeciderRequest::LEVEL_QUESTION);
    }

    #[test]
    fn typed_questions_serialize_as_dict() {
        let mut request = DeciderRequest {
            context: "ctx".into(),
            ..Default::default()
        };
        request.typed_questions.insert(
            "requires_only_response".into(),
            typed_question("noul", "text only or actions?", None),
        );
        request.typed_questions.insert(
            "reasoning_level".into(),
            typed_question(
                "choice",
                "how complex?",
                Some(json!({"1": "", "2": "", "3": ""})),
            ),
        );
        let json = request.to_json();
        let qs = json["questions"].as_object().unwrap();
        assert_eq!(qs["requires_only_response"]["type"], "noul");
        assert_eq!(qs["reasoning_level"]["type"], "choice");
        assert_eq!(qs["reasoning_level"]["criteria"]["3"], "");
    }

    #[test]
    fn image_context_is_optional_and_carried() {
        let plain = DeciderRequest::turn_routing("hi").to_json();
        assert!(plain.get("image").is_none());
        let with = DeciderRequest::turn_routing("hi")
            .with_image_base64("iVBORw0K")
            .with_image_url("file:///tmp/s.png")
            .to_json();
        assert_eq!(with["image"], "iVBORw0K");
        assert_eq!(with["image_url"], "file:///tmp/s.png");
    }

    #[test]
    fn parses_nested_result_answers() {
        let outcome = DeciderClient::parse_predict(
            r#"{"result":{"answers":{"requires_only_response":"no","reasoning_level":3}}}"#,
        )
        .unwrap();
        let c = TurnClassification::from_outcome(&outcome, Some(12.0));
        assert_eq!(c.branch, TurnBranch::RequiresActions);
        assert_eq!(c.reasoning_level, ReasoningLevel::L3);
        assert_eq!(c.latency_ms, Some(12.0));
    }

    #[test]
    fn parses_choice_answers_with_probabilities() {
        let outcome = DeciderClient::parse_predict(
            r#"{"answers":{
                 "requires_only_response":{"choice":"yes","confidence":0.91,"probabilities":{"yes":0.91,"no":0.09}},
                 "reasoning_level":{"choice":"1","confidence":0.8}}}"#,
        )
        .unwrap();
        let c = TurnClassification::from_outcome(&outcome, None);
        assert_eq!(c.branch, TurnBranch::RequiresOnlyResponse);
        assert!(!c.branch.needs_actions());
        assert_eq!(c.reasoning_level, ReasoningLevel::L1);
        assert!((c.confidence - 0.91).abs() < 1e-6);
        assert!((c.probabilities["no"] - 0.09).abs() < 1e-6);
        assert!(c.summary().contains("intent: chat") || c.summary().contains("intent: yes"));
        assert!(c.summary().contains("reasoning level 1"));
    }

    #[test]
    fn parses_new_choice_answers_from_decider_serve() {
        let outcome = DeciderClient::parse_predict(
            r#"{"answers":{
                 "q1":{"type":"choice","choice":"act = the user wants an action performed (open app, play media, control device, etc.)","confidence":0.985},
                 "q2":{"type":"choice","choice":"1 = simple, direct command","confidence":0.58}}}"#,
        )
        .unwrap();
        let c = TurnClassification::from_outcome(&outcome, None);
        assert_eq!(c.branch, TurnBranch::RequiresActions);
        assert!(c.branch.needs_actions());
        assert_eq!(c.reasoning_level, ReasoningLevel::L1);
        assert_eq!(c.confidence, 0.985);
        assert!(c.summary().contains("intent: act"));
    }

    #[test]
    fn positional_answers_are_understood() {
        let outcome =
            DeciderClient::parse_predict(r#"{"answers":{"0":"no","1":{"score":2.0}}}"#).unwrap();
        let c = TurnClassification::from_outcome(&outcome, None);
        assert_eq!(c.branch, TurnBranch::RequiresActions);
        assert_eq!(c.reasoning_level, ReasoningLevel::L2);
    }

    #[test]
    fn argmax_object_answers_are_understood() {
        let outcome = DeciderClient::parse_predict(
            r#"{"answers":{"reasoning_level":{"1":0.1,"2":0.7,"3":0.2}}}"#,
        )
        .unwrap();
        let c = TurnClassification::from_outcome(&outcome, None);
        assert_eq!(c.reasoning_level, ReasoningLevel::L2);
        assert!((c.level_confidence - 0.7).abs() < 1e-6);
    }

    #[test]
    fn missing_heads_fall_back_to_safe_defaults() {
        let outcome = DeciderClient::parse_predict(r#"{"answers":{}}"#).unwrap();
        let c = TurnClassification::from_outcome(&outcome, None);
        assert_eq!(c.branch, TurnBranch::RequiresOnlyResponse);
        assert_eq!(c.reasoning_level, ReasoningLevel::L2);
    }

    #[test]
    fn rejects_payloads_without_answers() {
        assert!(DeciderClient::parse_predict(r#"{"latency_ms":5}"#).is_err());
        assert!(DeciderClient::parse_predict("not json").is_err());
    }

    #[tokio::test]
    async fn health_against_dead_port_errors_clearly() {
        let c = client("http://127.0.0.1:9");
        let err = c.health().await.unwrap_err();
        assert!(
            err.to_string()
                .contains("could not reach the classification model"),
            "{err:#}"
        );
    }

    #[test]
    fn is_enabled_requires_url_and_flag() {
        let mut c = client("http://localhost:8001");
        assert!(c.is_enabled());
        c.config.enabled = false;
        assert!(!c.is_enabled());
        c.config.enabled = true;
        c.config.classification_api_url = String::new();
        assert!(!c.is_enabled());
    }

    #[test]
    fn heuristic_routes_action_commands_to_level_three() {
        let c = heuristic_turn_classification("open yt & play this song");
        assert_eq!(c.branch, TurnBranch::RequiresActions);
        assert_eq!(c.reasoning_level, ReasoningLevel::L3);
        assert_eq!(c.confidence, 0.0, "heuristics claim no confidence");
    }

    #[test]
    fn heuristic_answers_questions_in_text() {
        for q in [
            "what is rust?",
            "explain how borrow checking works",
            "why is the sky blue?",
        ] {
            assert_eq!(
                heuristic_turn_classification(q).branch,
                TurnBranch::RequiresOnlyResponse,
                "{q}"
            );
        }
    }

    #[test]
    fn heuristic_escalates_long_questions() {
        let long = format!(
            "{} words",
            (0..80).map(|_| "detail").collect::<Vec<_>>().join(" ")
        );
        assert_eq!(
            heuristic_turn_classification(&long).reasoning_level,
            ReasoningLevel::L2
        );
    }

    #[test]
    fn typed_question_matches_lucy_systemone_schema() {
        // The dict schema must stay wire-compatible with `crate::types::Question`.
        let q = Question::choice_simple("pick", &[("1", ""), ("2", ""), ("3", "")]);
        let expected = typed_question("choice", "pick", Some(json!({"1": "", "2": "", "3": ""})));
        assert_eq!(serde_json::to_value(&q).unwrap(), expected);
    }
}
