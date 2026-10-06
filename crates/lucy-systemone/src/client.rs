use anyhow::{Context, Result, anyhow};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tracing::debug;

use crate::bridge::LayaDaemonBridge;
use crate::types::{Answer, PredictRequest, PredictResponse, Question};
use lucy_config::LucyConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionMode {
    Chat,
    Act,
}

/// Unified System-1 Decision Client for Lucy.
///
/// Backend is `decider-serve` (`Mapika/decider-2b-vision`, default
/// `http://localhost:8001`), which natively accepts the laya-style dict
/// schema (`state` + `{id: {type, instructions, criteria}}`) at
/// `POST {base}/predict`. The user starts the server manually
/// (`python app.py --port 8001` in `~/Projects/decider-serve`); lucy never
/// starts it, never spawns processes, and never blocks startup waiting for
/// it — a missing server fails fast with [`SystemOneClient::api_down_message`].
/// The legacy direct CUDA Python daemon (`"laya_direct"`) is only constructed
/// on explicit opt-in.
#[derive(Clone)]
pub struct SystemOneClient {
    http: Client,
    config: lucy_config::SystemOneConfig,
    resolved_python: String,
    direct_bridge: Option<Arc<LayaDaemonBridge>>,
    /// P0 honesty counter: every `predict()` call, all providers.
    predict_calls: Arc<AtomicU64>,
}

impl SystemOneClient {
    pub fn new(config: lucy_config::SystemOneConfig, resolved_python: String) -> Self {
        let http = Client::builder()
            .timeout(Duration::from_millis(config.timeout_ms.max(1000)))
            .build()
            .unwrap_or_default();
        // Only pay for the daemon handle when the provider can actually use
        // it. The HTTP paths (`laya_api`, legacy `laya` alias) never spawn python.
        let direct_bridge = match config.provider.as_str() {
            "laya_direct" => Some(Arc::new(LayaDaemonBridge::new(&resolved_python))),
            _ => None,
        };
        Self {
            http,
            config,
            resolved_python,
            direct_bridge,
            predict_calls: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn from_lucy_config(cfg: &LucyConfig) -> Self {
        let resolved_python = cfg.resolve_system_one_python();
        Self::new(cfg.system_one.clone(), resolved_python)
    }

    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    pub fn provider(&self) -> &str {
        &self.config.provider
    }

    pub fn model(&self) -> &str {
        &self.config.model
    }

    pub fn python_path(&self) -> &str {
        &self.resolved_python
    }

    /// P0 honesty counter: total `predict()` calls since construction.
    pub fn predict_calls(&self) -> u64 {
        self.predict_calls.load(Ordering::SeqCst)
    }

    /// Primary inference entrypoint: sends state and typed questions to the System-1 engine.
    pub async fn predict(
        &self,
        state: &Value,
        questions: HashMap<String, Question>,
    ) -> Result<PredictResponse> {
        self.predict_calls.fetch_add(1, Ordering::SeqCst);
        let req = PredictRequest {
            state: state.clone(),
            questions,
            model: Some(self.config.model.clone()),
        };

        match self.config.provider.as_str() {
            "laya_direct" => {
                debug!("Predicting via direct Laya Python daemon");
                self.predict_direct(&req).await
            }
            // `decider` is the canonical name; `laya_api`/`laya` are legacy
            // aliases kept so old configs keep working. All three speak the
            // same `POST {base}/predict` schema (decider-serve accepts the
            // laya-style dict natively).
            "decider" | "decider_serve" | "laya_api" | "laya" => {
                debug!(
                    "Predicting via decider-serve HTTP at {}",
                    self.config.base_url
                );
                // Fail fast when the manually started server is missing —
                // lucy never starts it itself.
                if !self.check_api_alive().await {
                    return Err(anyhow!("{}", self.api_down_message()));
                }
                self.predict_http(&req).await
            }
            "typesafe" | "jev" => {
                debug!(
                    "Predicting via TypeSafe JEV API at {}",
                    self.config.base_url
                );
                self.predict_typesafe(&req).await
            }
            other => Err(anyhow!(
                "unknown System-1 provider '{other}' — use 'decider' (decider-serve at system_one.base_url), 'laya_direct' (local daemon), 'typesafe' or 'jev'"
            )),
        }
    }

    /// Actionable message when the manually started decider-serve is unreachable.
    /// Lucy never starts the server itself.
    pub fn api_down_message(&self) -> String {
        format!(
            "decider-serve not reachable at {} — start it manually first (`python app.py --port 8001` in ~/Projects/decider-serve), then retry",
            self.config.base_url
        )
    }

    async fn predict_direct(&self, req: &PredictRequest) -> Result<PredictResponse> {
        match &self.direct_bridge {
            Some(bridge) => bridge.predict(req).await,
            None => Err(anyhow!(
                "direct Laya daemon not initialised for provider '{}' — set provider to 'laya_direct' for local inference, or '{}' ",
                self.config.provider,
                self.api_down_message()
            )),
        }
    }

    /// No-op kept for API compatibility. Lucy never starts the server and
    /// never blocks startup on it: the user starts `decider-serve` manually
    /// and each `predict()` fails fast when it is missing.
    pub async fn warmup(&self) -> Result<()> {
        Ok(())
    }

    /// No-op kept for API compatibility (see [`Self::warmup`]). Always
    /// returns `Ok` immediately, ignoring the deadline.
    pub async fn warmup_with_deadline(&self, _deadline: Duration) -> Result<()> {
        Ok(())
    }

    /// Quick health check on the manually started decider-serve endpoint.
    pub async fn check_api_alive(&self) -> bool {
        let url = format!("{}/health", self.config.base_url.trim_end_matches('/'));
        match self
            .http
            .get(&url)
            .timeout(Duration::from_millis(400))
            .send()
            .await
        {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    async fn predict_http(&self, req: &PredictRequest) -> Result<PredictResponse> {
        let url = format!("{}/predict", self.config.base_url.trim_end_matches('/'));
        // Retries on transport failure: the first request after a server
        // restart can race the model load and time out client-side while the
        // server keeps loading — a retry then hits a warm resident model.
        // Predicts are idempotent, so this is safe.
        // Backoff grows per attempt to span a full cold load across retries.
        let backoffs = [Duration::from_millis(300), Duration::from_secs(2)];
        let mut attempts = 0;
        let resp = loop {
            attempts += 1;
            let mut builder = self.http.post(&url).json(req);
            if let Some(key) = &self.config.api_key {
                builder = builder.bearer_auth(key);
            }
            match builder.send().await {
                Ok(resp) => break resp,
                Err(e) if (e.is_timeout() || e.is_connect()) && attempts <= backoffs.len() => {
                    let wait = backoffs[attempts - 1];
                    debug!(
                        "decider-serve request failed ({e}); retrying in {wait:?} (attempt {attempts})"
                    );
                    tokio::time::sleep(wait).await;
                    continue;
                }
                Err(e) => return Err(anyhow!("{}: {e:#}", self.api_down_message())),
            }
        };
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("decider-serve returned HTTP {}: {}", status, body));
        }

        let result: PredictResponse = resp
            .json()
            .await
            .context("failed to parse decider-serve response")?;
        Ok(result)
    }

    async fn predict_typesafe(&self, req: &PredictRequest) -> Result<PredictResponse> {
        let base = if self.config.base_url.starts_with("http")
            && !self.config.base_url.contains("localhost")
        {
            self.config.base_url.trim_end_matches('/').to_string()
        } else {
            "https://api.typesafe.ai/v1".to_string()
        };
        let url = format!("{base}/systemone");

        let key = self.config.api_key.as_deref().ok_or_else(|| {
            anyhow!("TypeSafe JEV requires an API key — set TYPESAFE_API_KEY or lucy config system_one.api_key")
        })?;

        let resp = self
            .http
            .post(&url)
            .bearer_auth(key)
            .json(req)
            .send()
            .await
            .context("failed to send request to TypeSafe JEV")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("TypeSafe JEV returned HTTP {}: {}", status, body));
        }

        let result: PredictResponse = resp
            .json()
            .await
            .context("failed to parse TypeSafe JEV response")?;
        Ok(result)
    }

    /// Fast decision: Classify user input as `chat` vs `act` in sub-35ms.
    pub async fn decide_mode(
        &self,
        prompt: &str,
        recent_history: &str,
    ) -> Result<(DecisionMode, f64)> {
        let state = json!({
            "user_request": prompt,
            "recent_history": recent_history,
        });

        let mut criteria = HashMap::new();
        criteria.insert("chat".to_string(), json!("Pure conversation, answering questions, writing text or code explanation without computer actions"));
        criteria.insert("act".to_string(), json!("Action requiring computer use, opening apps, clicking UI elements, typing into fields, navigating, or running commands"));

        let q = Question::choice(
            "Classify whether this user request is pure chat or requires taking action on the computer.",
            criteria,
        );

        let mut questions = HashMap::new();
        questions.insert("mode".to_string(), q);

        let resp = self.predict(&state, questions).await?;
        let ans = resp
            .answers
            .get("mode")
            .ok_or_else(|| anyhow!("missing 'mode' answer from System-1"))?;
        let choice = ans.as_choice().unwrap_or("act");
        let conf = ans.confidence();

        let mode = match choice {
            "chat" => DecisionMode::Chat,
            _ => DecisionMode::Act,
        };
        Ok((mode, conf))
    }

    /// Fast decision: Select the best domain for an `act` request.
    pub async fn decide_domain(&self, goal: &str) -> Result<(String, f64)> {
        let state = json!({
            "goal": goal,
        });

        let mut criteria = HashMap::new();
        criteria.insert("desktop".to_string(), json!("Desktop apps, windows, workspaces, native UI navigation, clicking and typing in GUI apps"));
        criteria.insert(
            "browser".to_string(),
            json!("Web browsing, opening websites, Google Flights, online search, web automation"),
        );
        criteria.insert(
            "shell".to_string(),
            json!("Terminal commands, bash scripts, compiling code, system administration"),
        );
        criteria.insert(
            "files".to_string(),
            json!("Local files, editing code or text files on disk"),
        );

        let q = Question::choice(
            "Which execution domain is best suited for fulfilling this goal?",
            criteria,
        );

        let mut questions = HashMap::new();
        questions.insert("domain".to_string(), q);

        let resp = self.predict(&state, questions).await?;
        let ans = resp
            .answers
            .get("domain")
            .ok_or_else(|| anyhow!("missing 'domain' answer from System-1"))?;
        let choice = ans.as_choice().unwrap_or("desktop").to_string();
        let conf = ans.confidence();
        Ok((choice, conf))
    }

    /// Calibrated safety decision: Is this tool call or command dangerous?
    pub async fn is_destructive(&self, description: &str) -> Result<bool> {
        let state = json!({ "action": description });
        let q = Question::noul("Is this command or action destructive, harmful, or irreversible?");
        let mut questions = HashMap::new();
        questions.insert("destructive".to_string(), q);

        let resp = self.predict(&state, questions).await?;
        let ans = resp
            .answers
            .get("destructive")
            .ok_or_else(|| anyhow!("missing 'destructive' answer"))?;
        let prob = ans.as_noul().unwrap_or(0.0);
        Ok(prob > 0.5)
    }

    /// Speculative fan-out: one forward pass with multiple Question heads.
    ///
    /// Sends `state` + N heads (`operation`, `click_target`, `type_target`,
    /// `select_target`, `kind`, ...) in a **single** `predict()` call. The
    /// caller dispatches on the winning `operation` head and ignores the
    /// operation-irrelevant target heads — fan-out, not cascade. Single
    /// `predict` cost regardless of how many speculative targets are scored.
    pub async fn predict_speculative(
        &self,
        state: &Value,
        questions: HashMap<String, Question>,
    ) -> Result<PredictResponse> {
        self.predict(state, questions).await
    }

    /// Helper to evaluate a batch of choices respecting Laya's head_max_len=192 constraint.
    async fn evaluate_choice_batch(
        &self,
        state: &Value,
        question_prompt: &str,
        key: &str,
        batch: &[(String, String)],
    ) -> Result<Vec<(String, f64)>> {
        let mut criteria = HashMap::new();
        for (name, desc) in batch {
            let mut d = desc.trim();
            if d.is_empty() {
                d = name.as_str();
            }
            let truncated: String = d.chars().take(28).collect();
            criteria.insert(name.clone(), json!(truncated));
        }

        let q = Question::choice(question_prompt, criteria);
        let mut questions = HashMap::new();
        questions.insert(key.to_string(), q);

        let resp = self.predict(state, questions).await?;
        let ans = resp
            .answers
            .get(key)
            .ok_or_else(|| anyhow!("missing '{key}' answer from System-1"))?;

        let mut scored = Vec::new();
        if let Answer::Choice(c) = ans {
            for (name, prob) in &c.probabilities {
                scored.push((name.clone(), *prob));
            }
            if scored.is_empty() {
                scored.push((c.choice.clone(), c.confidence));
            }
        } else if let Some(choice) = ans.as_choice() {
            scored.push((choice.to_string(), ans.confidence()));
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored)
    }

    /// Use Laya to decide which tool(s) to use for a given goal.
    /// Returns ranked list of (tool_name, probability).
    pub async fn decide_tools_multi(
        &self,
        goal: &str,
        candidate_tools: &[(String, String)],
        max_tools: usize,
    ) -> Result<Vec<(String, f64)>> {
        if candidate_tools.is_empty() {
            return Ok(Vec::new());
        }
        if candidate_tools.len() == 1 {
            return Ok(vec![(candidate_tools[0].0.clone(), 1.0)]);
        }

        let state = json!({ "task_goal": goal });
        let prompt = format!("Which tool is best suited to achieve the task: '{goal}'?");

        let chunk_size = 5;
        if candidate_tools.len() <= chunk_size {
            let scored = self
                .evaluate_choice_batch(&state, &prompt, "tool", candidate_tools)
                .await?;
            return Ok(scored.into_iter().take(max_tools.max(1)).collect());
        }

        let mut top_candidates = Vec::new();
        for chunk in candidate_tools.chunks(chunk_size) {
            if let Ok(scored) = self
                .evaluate_choice_batch(&state, &prompt, "tool", chunk)
                .await
            {
                if let Some(winner) = scored.first() {
                    let desc = candidate_tools
                        .iter()
                        .find(|(n, _)| n == &winner.0)
                        .map(|(_, d)| d.clone())
                        .unwrap_or_default();
                    top_candidates.push((winner.0.clone(), desc));
                }
            }
        }

        if top_candidates.is_empty() {
            return Ok(candidate_tools
                .iter()
                .take(max_tools)
                .map(|(n, _)| (n.clone(), 0.5))
                .collect());
        }

        if top_candidates.len() > chunk_size {
            top_candidates.truncate(chunk_size);
        }

        let final_scored = self
            .evaluate_choice_batch(&state, &prompt, "tool", &top_candidates)
            .await?;
        Ok(final_scored.into_iter().take(max_tools.max(1)).collect())
    }

    /// Use Laya to decide which UI element or button to click.
    pub async fn decide_click_target(
        &self,
        goal: &str,
        candidates: &HashMap<String, String>,
    ) -> Result<(String, f64)> {
        if candidates.is_empty() {
            return Err(anyhow!("No clickable elements available to choose from"));
        }
        if candidates.len() == 1 {
            let (k, _) = candidates.iter().next().unwrap();
            return Ok((k.clone(), 1.0));
        }

        let state = json!({ "goal": goal });
        let prompt =
            format!("Which element should be clicked to advance the user's goal: '{goal}'?");
        let pairs: Vec<(String, String)> = candidates
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let chunk_size = 5;
        if pairs.len() <= chunk_size {
            let scored = self
                .evaluate_choice_batch(&state, &prompt, "click_target", &pairs)
                .await?;
            let top = scored
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("no click target selected"))?;
            return Ok(top);
        }

        let mut winners = Vec::new();
        for chunk in pairs.chunks(chunk_size) {
            if let Ok(scored) = self
                .evaluate_choice_batch(&state, &prompt, "click_target", chunk)
                .await
            {
                if let Some(w) = scored.first() {
                    let desc = candidates.get(&w.0).cloned().unwrap_or_default();
                    winners.push((w.0.clone(), desc));
                }
            }
        }

        if winners.len() > chunk_size {
            winners.truncate(chunk_size);
        }

        let final_scored = self
            .evaluate_choice_batch(&state, &prompt, "click_target", &winners)
            .await?;
        let top = final_scored
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("no click target selected"))?;
        Ok(top)
    }

    /// Use Laya to decide which editable field to type into.
    pub async fn decide_type_target(
        &self,
        goal: &str,
        candidates: &HashMap<String, String>,
    ) -> Result<(String, f64)> {
        if candidates.is_empty() {
            return Err(anyhow!("No editable fields available to choose from"));
        }
        if candidates.len() == 1 {
            let (k, _) = candidates.iter().next().unwrap();
            return Ok((k.clone(), 1.0));
        }

        let state = json!({ "goal": goal });
        let prompt = format!("Which text input field should be filled for the goal: '{goal}'?");
        let pairs: Vec<(String, String)> = candidates
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let chunk_size = 5;
        if pairs.len() <= chunk_size {
            let scored = self
                .evaluate_choice_batch(&state, &prompt, "type_target", &pairs)
                .await?;
            let top = scored
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("no type target selected"))?;
            return Ok(top);
        }

        let mut winners = Vec::new();
        for chunk in pairs.chunks(chunk_size) {
            if let Ok(scored) = self
                .evaluate_choice_batch(&state, &prompt, "type_target", chunk)
                .await
            {
                if let Some(w) = scored.first() {
                    let desc = candidates.get(&w.0).cloned().unwrap_or_default();
                    winners.push((w.0.clone(), desc));
                }
            }
        }

        if winners.len() > chunk_size {
            winners.truncate(chunk_size);
        }

        let final_scored = self
            .evaluate_choice_batch(&state, &prompt, "type_target", &winners)
            .await?;
        let top = final_scored
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("no type target selected"))?;
        Ok(top)
    }

    /// Use Laya to decide if the user's goal is visibly satisfied.
    pub async fn is_goal_satisfied(
        &self,
        goal: &str,
        current_state: &str,
        recent_actions: &[String],
    ) -> Result<(bool, f64)> {
        let state = json!({
            "goal": goal,
            "current_screen": current_state,
            "recent_actions": recent_actions,
        });

        let q = Question::noul(format!(
            "Based on the current screen and actions taken, is the user's goal '{goal}' already visibly and completely satisfied?"
        ));

        let mut questions = HashMap::new();
        questions.insert("satisfied".to_string(), q);

        let resp = self.predict(&state, questions).await?;
        let ans = resp
            .answers
            .get("satisfied")
            .ok_or_else(|| anyhow!("missing 'satisfied' answer from System-1"))?;

        let prob = ans.as_noul().unwrap_or(0.0);
        let conf = ans.confidence();
        Ok((prob > 0.75, conf))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_client(base_url: &str) -> SystemOneClient {
        SystemOneClient::new(
            lucy_config::SystemOneConfig {
                enabled: true,
                provider: "decider".into(),
                base_url: base_url.into(),
                api_key: None,
                model: "Mapika/decider-2b-vision".into(),
                direct_python: None,
                confidence_threshold: 0.1,
                timeout_ms: 1000,
                auto_start: false,
                server_command: None,
                server_model: None,
            },
            "python3".into(),
        )
    }

    #[tokio::test]
    async fn warmup_never_blocks_startup() {
        // Warmup is a no-op: it must return Ok immediately even when nothing
        // listens, so `LucyRuntime::new()` never waits on the decision backend.
        let client = test_client("http://127.0.0.1:9");
        let start = std::time::Instant::now();
        client
            .warmup_with_deadline(Duration::from_secs(30))
            .await
            .expect("warmup must never fail");
        client.warmup().await.expect("warmup must never fail");
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "warmup blocked startup for {:?}",
            start.elapsed()
        );
    }

    #[tokio::test]
    async fn predict_fails_fast_with_actionable_message_when_server_missing() {
        // Nothing listens on 9 (discard port): predict must fail fast with
        // the manual-start message, never hang or spawn anything.
        let client = test_client("http://127.0.0.1:9");
        let mut questions = HashMap::new();
        questions.insert("mode".to_string(), Question::noul("is this chat?"));
        let err = client
            .predict(&json!({"user_request": "hi"}), questions)
            .await
            .expect_err("predict against a dead port must fail");
        assert!(
            err.to_string().contains("decider-serve not reachable"),
            "unexpected error: {err:#}"
        );
    }

    #[tokio::test]
    async fn legacy_laya_provider_alias_still_routes_to_http() {
        // Old configs with `provider = "laya_api"` keep working: they hit
        // the same `POST {base}/predict` path as `"decider"`.
        let mut client = test_client("http://127.0.0.1:9");
        client.config.provider = "laya_api".into();
        let mut questions = HashMap::new();
        questions.insert("mode".to_string(), Question::noul("is this chat?"));
        let err = client
            .predict(&json!({"user_request": "hi"}), questions)
            .await
            .expect_err("predict against a dead port must fail");
        assert!(
            err.to_string().contains("decider-serve not reachable"),
            "unexpected error: {err:#}"
        );
    }

    #[tokio::test]
    async fn live_decider_mode_decision() {
        // Integration: requires the manually started `decider-serve` on :8001.
        // Skipped (not failed) when the server is absent so plain `cargo
        // test` stays green on machines without it. Generous timeout: a
        // VRAM-pressured card can take seconds per forward pass.
        let mut live = test_client("http://localhost:8001");
        live.config.timeout_ms = 60_000;
        let live = SystemOneClient::new(live.config.clone(), "python3".into());
        if !live.check_api_alive().await {
            eprintln!("skipping live decider test: decider-serve not running");
            return;
        }
        let (mode, conf) = live
            .decide_mode("open youtube and play despacito", "")
            .await
            .expect("decide_mode against live decider-serve must succeed");
        assert_eq!(mode, DecisionMode::Act);
        assert!(conf > 0.5, "confidence too low: {conf}");
    }
}
