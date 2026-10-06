//! Connected-provider management: Test Connection + Save.
//!
//! Backs the **Connect providers** section of `/settings`. Both operations are
//! deliberately free of any global state — they take a [`ProviderConfig`]
//! (possibly unsaved, straight from the settings inputs) and return a value, so
//! the TUI can run them without persisting anything until `[Save]` is pressed.

use std::time::Duration;

use anyhow::{Context, Result, bail};
use lucy_config::{DEFAULT_TEXT_MODEL, LucyConfig, ProviderConfig, ProviderType};
use reqwest::Client;
use serde_json::Value;

use crate::provider::apply_endpoint_headers;

/// Outcome of a successful `Test Connection`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderHealth {
    /// Provider id, filled in when the caller left it blank.
    pub id: String,
    pub name: String,
    pub provider_type: ProviderType,
    pub api_url: String,
    /// Endpoint that answered (`…/v1/models`).
    pub models_endpoint: String,
    /// Model ids discovered from the response, sorted and deduped.
    pub available_models: Vec<String>,
    /// Ids the endpoint flagged as deprecated (a subset of `available_models`).
    pub deprecated_models: Vec<String>,
    /// Round-trip time of the probe, for the settings status line.
    pub latency_ms: u64,
}

impl ProviderHealth {
    /// One-line summary for the settings screen / chat log.
    pub fn summary(&self) -> String {
        format!(
            "{} — ok · {} model(s) · {} ms",
            self.name,
            self.available_models.len(),
            self.latency_ms
        )
    }
}

fn probe_client(timeout: Duration) -> Result<Client> {
    Client::builder()
        .timeout(timeout)
        .build()
        .context("failed to build HTTP client")
}

/// Normalize a base URL the way the settings screen stores it: trimmed, no
/// trailing slash.
pub fn normalize_api_url(raw: &str) -> String {
    raw.trim().trim_end_matches('/').to_owned()
}

/// One entry from an OpenAI-compatible `/models` payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelEntry {
    pub id: String,
    pub deprecated: bool,
}

/// Pull the model entries (id + deprecation flag) out of a `/models` payload.
///
/// Accepts the documented `{"data":[{"id":"…"}]}` shape and tolerates a bare
/// array, plus a few vendor-specific spellings (`name`, `model`,
/// `model_name`) so a non-conforming endpoint still yields something usable.
/// An entry listed twice is deprecated only when every occurrence is.
pub fn parse_model_entries(body: &Value) -> Vec<ModelEntry> {
    let entries = body
        .get("data")
        .and_then(Value::as_array)
        .or_else(|| body.get("models").and_then(Value::as_array))
        .or_else(|| body.as_array());
    let Some(entries) = entries else {
        return Vec::new();
    };
    let mut out: Vec<ModelEntry> = Vec::new();
    for e in entries {
        let Some(id) = e
            .get("id")
            .or_else(|| e.get("name"))
            .or_else(|| e.get("model"))
            .or_else(|| e.get("model_name"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
        else {
            continue;
        };
        // OpenAI-style `deprecated: true`; anything else (missing, a
        // deprecated string form) means live.
        let deprecated = e
            .get("deprecated")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        match out.iter_mut().find(|o| o.id == id) {
            Some(existing) => existing.deprecated &= deprecated,
            None => out.push(ModelEntry { id, deprecated }),
        }
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

/// Pull the model ids out of an OpenAI-compatible `/models` payload.
///
/// Accepts the documented `{"data":[{"id":"…"}]}` shape and tolerates a bare
/// array, plus a few vendor-specific spellings (`name`, `model`,
/// `model_name`) so a non-conforming endpoint still yields something usable.
pub fn parse_model_ids(body: &Value) -> Vec<String> {
    let mut ids: Vec<String> = parse_model_entries(body)
        .into_iter()
        .map(|e| e.id)
        .collect();
    ids.dedup();
    ids
}

/// `Test Connection`: verify `api_url` + `api_key` and discover the models.
///
/// Probes `GET {api_url}/models`, adding the conventional `/v1` suffix when the
/// URL does not already end in a version segment. An empty model list is a
/// failure: a 200 that advertises nothing cannot populate the model dropdowns.
pub async fn test_connection(provider: &ProviderConfig) -> Result<ProviderHealth> {
    if !provider.is_usable() {
        bail!("API URL and API Key are both required before testing a connection");
    }
    let url = provider.models_endpoint();
    let key = provider.api_key.trim();
    let (entries, latency_ms) = probe_models(&url, key, Duration::from_secs(15)).await?;
    let available_models: Vec<String> = entries.iter().map(|e| e.id.clone()).collect();
    let deprecated_models: Vec<String> = entries
        .iter()
        .filter(|e| e.deprecated)
        .map(|e| e.id.clone())
        .collect();
    // A blank id is assigned by `save_provider` against the config being
    // edited; the probe itself is id-agnostic, so echo the draft's id.
    let id = provider.id.trim().to_owned();
    let name = if provider.name.trim().is_empty() {
        provider.label()
    } else {
        provider.name.trim().to_owned()
    };
    Ok(ProviderHealth {
        id,
        name,
        provider_type: provider.provider_type,
        api_url: normalize_api_url(&provider.api_url),
        models_endpoint: url,
        available_models,
        deprecated_models,
        latency_ms,
    })
}

/// Shared `/models` probe: `GET {url}`, JSON shape check, entry extraction.
///
/// `api_key` may be empty for a local endpoint that takes none — no
/// `Authorization` header is sent then. An empty model list is a failure: a
/// 200 that advertises nothing cannot populate the model dropdowns.
async fn probe_models(
    url: &str,
    api_key: &str,
    timeout: Duration,
) -> Result<(Vec<ModelEntry>, u64)> {
    let started = std::time::Instant::now();
    let client = probe_client(timeout)?;
    let mut req = client.get(url);
    let key = api_key.trim();
    if !key.is_empty() {
        // Some OpenAI-compatible gateways (Groq, LM Studio, llama.cpp) want the
        // key on a second header too; harmless when ignored.
        req = req.bearer_auth(key).header("x-api-key", key);
    }
    let req = apply_endpoint_headers(req, url);
    let resp = req
        .send()
        .await
        .with_context(|| format!("could not reach {url}"))?;
    let status = resp.status();
    let body = resp
        .text()
        .await
        .context("failed to read the /models response")?;
    if !status.is_success() {
        let snippet: String = body.chars().take(200).collect();
        let hint = match status.as_u16() {
            401 | 403 => " — the API key was rejected",
            404 => " — the URL has no /models endpoint (check the base URL)",
            429 => " — rate limited, try again shortly",
            _ => "",
        };
        bail!("{url} returned {status}{hint}: {snippet}");
    }
    let value: Value = serde_json::from_str(&body).with_context(|| {
        format!(
            "{url} returned a non-JSON body: {}",
            body.chars().take(120).collect::<String>()
        )
    })?;
    let mut entries = parse_model_entries(&value);
    if entries.is_empty() {
        bail!("{url} answered but listed no models — nothing to save");
    }
    entries.sort_by(|a, b| a.id.cmp(&b.id));
    Ok((entries, started.elapsed().as_millis() as u64))
}

/// `Test Connection` + `Save` in one step: probe, write the provider (with the
/// discovered model list) into `config`, and persist to disk.
pub async fn save_provider(
    config: &mut LucyConfig,
    provider: &ProviderConfig,
) -> Result<ProviderHealth> {
    let health = test_connection(provider).await?;
    let stored = config.save_provider(ProviderConfig {
        id: health.id.clone(),
        name: health.name.clone(),
        api_url: health.api_url.clone(),
        api_key: provider.api_key.trim().to_owned(),
        provider_type: health.provider_type,
        available_models: health.available_models.clone(),
        deprecated_models: health.deprecated_models.clone(),
    })?;
    config.save()?;
    Ok(ProviderHealth {
        id: stored.id,
        ..health
    })
}

/// Refresh the legacy text endpoint's provider entry from its live `/models`.
///
/// `[models] text_base_url` names an endpoint but nothing in the provider list
/// is tied to it until this runs, so its models never reach the dropdowns and
/// the last-resort default is chosen blind. Probe it, register it as a text
/// provider (keeping the user's id, url, key), and reconcile the compiled-in
/// default — when `[models] default_text` is the compiled-in
/// `DEFAULT_TEXT_MODEL` and the endpoint no longer serves a live id by that
/// name, rebind it to the endpoint's first live model. Nothing is written to
/// disk when nothing changed; a probe failure leaves the old config in place.
pub async fn sync_text_endpoint(config: &mut LucyConfig) -> Result<Option<ProviderHealth>> {
    let raw_url = config.text_base_url().unwrap_or_default();
    if raw_url.trim().is_empty() {
        return Ok(None);
    }
    let url = normalize_api_url(&raw_url);
    let key = config.text_api_key().unwrap_or_default();
    let draft = ProviderConfig {
        api_url: url.clone(),
        ..Default::default()
    };
    let probe_url = draft.models_endpoint();
    let want = draft.openai_base_url();
    let (entries, latency_ms) = probe_models(&probe_url, &key, Duration::from_secs(4)).await?;
    let mut models: Vec<String> = entries.iter().map(|e| e.id.clone()).collect();
    models.sort();
    models.dedup();
    let mut deprecated: Vec<String> = entries
        .iter()
        .filter(|e| e.deprecated)
        .map(|e| e.id.clone())
        .collect();
    deprecated.sort();
    deprecated.dedup();

    let mut changed = false;
    match config.providers.iter_mut().find(|p| {
        p.provider_type == ProviderType::Text
            && p.openai_base_url()
                .trim_end_matches('/')
                .eq_ignore_ascii_case(want.trim_end_matches('/'))
    }) {
        Some(p) => {
            if p.available_models != models
                || p.deprecated_models != deprecated
                || p.api_key.trim() != key.trim()
            {
                p.available_models = models.clone();
                p.deprecated_models = deprecated.clone();
                p.api_key = key.trim().to_owned();
                changed = true;
            }
        }
        None => {
            let taken: Vec<String> = config.providers.iter().map(|p| p.id.clone()).collect();
            let mut p = ProviderConfig {
                id: String::new(),
                name: String::new(),
                api_url: url.clone(),
                api_key: key.trim().to_owned(),
                provider_type: ProviderType::Text,
                available_models: models.clone(),
                deprecated_models: deprecated.clone(),
            };
            p.ensure_id(&taken);
            config.providers.push(p);
            changed = true;
        }
    }

    let selected = config.models.default_text.trim().to_owned();
    if selected == DEFAULT_TEXT_MODEL {
        let stale =
            !models.iter().any(|m| m == &selected) || deprecated.iter().any(|m| m == &selected);
        if stale {
            if let Some(first) = models.iter().find(|m| !deprecated.contains(m)) {
                config.models.default_text = first.clone();
                changed = true;
            }
        }
    }

    if changed {
        config.save()?;
    }

    let stored = config.providers.iter().find(|p| {
        p.provider_type == ProviderType::Text
            && p.openai_base_url()
                .trim_end_matches('/')
                .eq_ignore_ascii_case(want.trim_end_matches('/'))
    });
    Ok(Some(ProviderHealth {
        id: stored.map(|p| p.id.clone()).unwrap_or_default(),
        name: stored.map(|p| p.label()).unwrap_or_default(),
        provider_type: ProviderType::Text,
        api_url: url,
        models_endpoint: probe_url,
        available_models: models,
        deprecated_models: deprecated,
        latency_ms,
    }))
}

/// Persist a provider that has already been tested (or whose model list was
/// entered by hand) without re-probing the endpoint.
pub fn persist_provider(
    config: &mut LucyConfig,
    provider: ProviderConfig,
) -> Result<ProviderConfig> {
    let stored = config.save_provider(provider)?;
    config.save()?;
    Ok(stored)
}

/// Persist the `decider-serve` API URL. No API key is involved — the local
/// classification server is unauthenticated.
pub fn persist_classification_url(config: &mut LucyConfig, url: &str) -> Result<String> {
    let saved = config.save_classification_url(url)?;
    config.save()?;
    Ok(saved)
}

/// Persist the voice model plus the rest of the chat settings touched by the
/// dropdowns. The three reasoning tiers have their own bindings — see
/// [`persist_level_model`] — and there is no main model to keep in sync.
pub fn persist_chat_settings(
    config: &mut LucyConfig,
    chat_mode: lucy_config::ChatMode,
    voice_model: &str,
    auto_compact: bool,
) -> Result<()> {
    config.set_chat_mode(chat_mode);
    config.set_voice_model(voice_model);
    config.set_auto_compact(auto_compact);
    config.save()
}

/// Persist one reasoning tier's model binding.
pub fn persist_level_model(
    config: &mut LucyConfig,
    level: lucy_config::ReasoningLevel,
    model: &str,
) -> Result<()> {
    config
        .chat
        .reasoning_levels
        .set(level, model.trim().to_owned());
    config.save()
}

/// Guard used before writing: refuse to persist a selection that points at a
/// provider which is no longer connected.
pub fn ensure_model_selectable(config: &LucyConfig, key: &str) -> Result<()> {
    let key = key.trim();
    if key.is_empty() {
        return Ok(());
    }
    // Only an explicit `<known-provider-id>/<model>` reference is checked. A
    // bare model name (or one whose prefix is not a connected provider, e.g.
    // `meta-llama/Llama-3-8b`) is resolved against the single text provider by
    // the provider layer, so it passes through.
    let Some((provider_id, model)) = key.split_once('/') else {
        return Ok(());
    };
    let Some(provider) = config.provider(provider_id) else {
        return Ok(());
    };
    if !provider
        .available_models
        .iter()
        .any(|m| m.trim() == model.trim())
    {
        bail!(
            "{key} is not available on provider {} (run Test Connection to refresh its model list)",
            provider.label()
        );
    }
    Ok(())
}

/// Validate a provider draft before `[Test]`/`[Save]` so obvious mistakes are
/// reported without a network round-trip.
pub fn validate_provider_draft(provider: &ProviderConfig) -> Result<()> {
    if provider.api_url.trim().is_empty() {
        bail!("API URL is required");
    }
    let url = provider.api_url.trim();
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        bail!("API URL must start with http:// or https://");
    }
    if url
        .split("://")
        .nth(1)
        .is_none_or(|rest| rest.trim().is_empty())
    {
        bail!("API URL has no host");
    }
    if provider.api_key.trim().is_empty() {
        bail!("API Key is required");
    }
    Ok(())
}

/// Reader-facing message for a failed probe, already phrased for the TUI.
///
/// Two layers, in this order, because they know different things:
/// [`lucy_core::explain`] classifies protocol facts (a JSON body, an HTTP class,
/// a transport failure) and this function adds the advice that only makes sense
/// on the Connect-providers screen — where the `/v1` suffix and the key field
/// live. Neither keeps a payload: the raw body is still on the error and in the
/// model-call log.
pub fn friendly_probe_error(err: &anyhow::Error) -> String {
    let raw = format!("{err:#}");
    if raw.trim().is_empty() {
        return "unknown error".to_owned();
    }
    let mut explained = lucy_core::explain(&raw);
    let low = raw.to_ascii_lowercase();
    // These two override the generic hint rather than appending to it: on this
    // screen a 404 almost always means the same thing — the base URL lost its
    // `/v1` — and two hints reading "check the model name" and "add /v1" is worse
    // advice than either alone.
    //
    // The 404 has to be *recognised as a status*, not found as three digits: a
    // stub server on port 404 — or any endpoint whose URL carries those digits —
    // otherwise turned a 401 "check your API key" into "add /v1", which is the
    // same substring matching AGENTS.md warns about, applied to a port number.
    let is_404 = low.contains("no /models endpoint") || explained.is_status(404);
    if is_404 {
        explained.hint = Some("the base URL is probably missing its /v1 suffix".to_owned());
    } else if low.contains("dns")
        || low.contains("could not reach")
        || low.contains("connection refused")
        || low.contains("connect")
    {
        explained.hint = Some("is the endpoint running and reachable?".to_owned());
    }
    explained.one_line()
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;

    fn draft(url: &str, key: &str) -> ProviderConfig {
        ProviderConfig {
            id: "groq".into(),
            name: "Groq".into(),
            api_url: url.into(),
            api_key: key.into(),
            provider_type: ProviderType::Text,
            available_models: Vec::new(),
            deprecated_models: Vec::new(),
        }
    }

    #[test]
    fn parses_openai_data_shape() {
        let v: Value = serde_json::from_str(
            r#"{"object":"list","data":[{"id":"flash"},{"id":"alpha"},{"id":"flash"}]}"#,
        )
        .unwrap();
        assert_eq!(parse_model_ids(&v), vec!["alpha", "flash"]);
    }

    #[test]
    fn parses_deprecation_flags_from_the_models_payload() {
        let v: Value = serde_json::from_str(
            r#"{"data":[{"id":"a","deprecated":true},{"id":"b","deprecated":false},{"id":"a"}]}"#,
        )
        .unwrap();
        let entries = parse_model_entries(&v);
        // "a" is listed twice; once without a flag means not every occurrence
        // was deprecated, so it counts as live.
        assert_eq!(
            entries,
            vec![
                ModelEntry {
                    id: "a".into(),
                    deprecated: false
                },
                ModelEntry {
                    id: "b".into(),
                    deprecated: false
                },
            ]
        );
        let v: Value =
            serde_json::from_str(r#"{"data":[{"id":"a","deprecated":true},{"id":"b"}]}"#).unwrap();
        assert_eq!(
            parse_model_entries(&v),
            vec![
                ModelEntry {
                    id: "a".into(),
                    deprecated: true
                },
                ModelEntry {
                    id: "b".into(),
                    deprecated: false
                },
            ]
        );
    }

    #[test]
    fn parses_bare_array_and_alt_keys() {
        let bare: Value = serde_json::from_str(r#"[{"id":"a"},{"id":"b"}]"#).unwrap();
        assert_eq!(parse_model_ids(&bare), vec!["a", "b"]);
        let alt: Value =
            serde_json::from_str(r#"{"models":[{"name":"m1"},{"model":"m2"}]}"#).unwrap();
        assert_eq!(parse_model_ids(&alt), vec!["m1", "m2"]);
        let empty: Value = serde_json::from_str(r#"{"data":[]}"#).unwrap();
        assert!(parse_model_ids(&empty).is_empty());
        let junk: Value = serde_json::from_str("{}").unwrap();
        assert!(parse_model_ids(&junk).is_empty());
    }

    #[test]
    fn draft_validation_rejects_bad_input() {
        assert!(validate_provider_draft(&draft("https://a.test/v1", "k")).is_ok());
        assert!(validate_provider_draft(&draft("", "k")).is_err());
        assert!(validate_provider_draft(&draft("ftp://a.test", "k")).is_err());
        assert!(validate_provider_draft(&draft("https://", "k")).is_err());
        assert!(validate_provider_draft(&draft("https://a.test", "  ")).is_err());
    }

    #[tokio::test]
    async fn test_connection_requires_url_and_key() {
        let err = test_connection(&draft("https://a.test", " "))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("API URL and API Key"));
    }

    #[test]
    fn selectable_check_rejects_unknown_model() {
        let cfg = LucyConfig {
            providers: vec![ProviderConfig {
                available_models: vec!["flash".into()],
                ..draft("https://a.test/v1", "k")
            }],
            ..Default::default()
        };
        assert!(ensure_model_selectable(&cfg, "groq/flash").is_ok());
        assert!(ensure_model_selectable(&cfg, "groq/nope").is_err());
        // Empty and bare-name selections are always allowed.
        assert!(ensure_model_selectable(&cfg, "").is_ok());
        assert!(ensure_model_selectable(&cfg, "some-hand-typed-model").is_ok());
    }

    #[test]
    fn probe_error_hints_are_actionable() {
        let e401 =
            anyhow!("http://x/v1/models returned 401 Unauthorized — the API key was rejected");
        assert!(friendly_probe_error(&e401).contains("check the API Key"));
        let e404 = anyhow!("returned 404 — the URL has no /models endpoint");
        assert!(friendly_probe_error(&e404).contains("/v1 suffix"));
        let edns = anyhow!("could not reach http://localhost:9/v1/models: connection refused");
        assert!(friendly_probe_error(&edns).contains("reachable"));
    }

    /// The screen's job is to be readable, so the general rule is asserted here
    /// too: whatever the failure class, no payload reaches the status line.
    #[test]
    fn a_probe_error_never_shows_the_response_body() {
        let cases = [
            "returned HTTP 429: {\"error\":{\"message\":\"Rate limit reached\",\"type\":\"rate_limit_error\"}}",
            "returned HTTP 500: {\"error\":{\"message\":\"upstream exploded\"}}",
            "<!DOCTYPE html><html>502 Bad Gateway</html>",
        ];
        for raw in cases {
            let err = anyhow!("{raw}");
            let msg = friendly_probe_error(&err);
            assert!(!msg.contains('{'), "{msg}");
            assert!(!msg.contains("<"), "{msg}");
            assert!(!msg.is_empty());
        }
    }

    #[test]
    fn a_probe_error_keeps_the_endpoint_specific_advice() {
        // Each class of probe failure has one next step, and two of them are the
        // screen's own knowledge rather than anything in the error text.
        for (raw, hint) in [
            ("returned HTTP 401 — unauthorized", "API Key"),
            ("returned 404 — the URL has no /models endpoint", "/v1 suffix"),
            ("could not reach http://localhost:9: connection refused", "reachable"),
            ("returned HTTP 429: rate limit reached", "wait a few seconds"),
        ] {
            let err = anyhow!("{raw}");
            assert!(
                friendly_probe_error(&err).contains(hint),
                "{raw} lost its hint"
            );
        }
    }
}
