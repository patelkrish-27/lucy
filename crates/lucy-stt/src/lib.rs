use anyhow::{anyhow, Context, Result};
use futures::{stream, StreamExt};
use lucy_adk::adk_audio::{
    encode, AudioError, AudioFormat, AudioFrame, AudioCapture, CaptureConfig, SpeechSegment,
    SttOptions, SttProvider, Transcript, VadProcessor,
};
use lucy_config::LucyConfig;
use reqwest::{multipart, Client, StatusCode};
use serde::Deserialize;
use std::{
    path::Path,
    sync::{Arc, Mutex},
    time::Duration,
};

const GROQ_ENDPOINT: &str = "https://api.groq.com/openai/v1/audio/transcriptions";
const DEFAULT_MODEL: &str = "whisper-large-v3-turbo";
const DEFAULT_LANGUAGE: &str = "en";
const DEFAULT_PROMPT: &str = "User voice commands for computer assistant";
const MAX_UPLOAD_BYTES: usize = 25 * 1024 * 1024;

// Preserve the previous Lucy voice behavior while moving microphone/VAD/frame/
// codec handling into ADK Audio.
const START_THRESHOLD: f32 = 0.012;
const END_THRESHOLD: f32 = 0.008;
const START_WINDOW: Duration = Duration::from_millis(120);
const SILENCE_AFTER_SPEECH: Duration = Duration::from_millis(750);
const MAX_UTTERANCE: Duration = Duration::from_secs(20);
const HOLD_MIN_AUDIO: Duration = Duration::from_millis(350);

#[derive(Debug, Clone)]
pub struct GroqSttConfig {
    pub api_key: String,
    pub model: String,
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub temperature: f32,
    pub timeout: Duration,
}

impl GroqSttConfig {
    pub fn from_config(config: &LucyConfig) -> Result<Self> {
        let api_key = config
            .stt
            .api_key
            .clone()
            .ok_or_else(|| anyhow!("Groq STT API key is not configured"))?;
        Ok(Self {
            api_key,
            model: config.stt.model.clone().unwrap_or_else(|| DEFAULT_MODEL.into()),
            language: config.stt.language.clone(),
            prompt: config.stt.prompt.clone(),
            temperature: config.stt.temperature,
            timeout: Duration::from_secs(60),
        })
    }
}

#[derive(Debug, Deserialize)]
struct GroqResponse {
    text: String,
}

pub struct GroqSttProvider {
    config: GroqSttConfig,
    client: Client,
}

impl GroqSttProvider {
    pub fn new(config: GroqSttConfig) -> Result<Self> {
        let client = Client::builder().timeout(config.timeout).build()?;
        Ok(Self { config, client })
    }
}

#[async_trait::async_trait]
impl SttProvider for GroqSttProvider {
    async fn transcribe(&self, wav: bytes::Bytes, opts: SttOptions) -> std::result::Result<Transcript, AudioError> {
        let model = opts
            .model_hint
            .as_deref()
            .unwrap_or(&self.config.model)
            .to_owned();
        let language = opts
            .language
            .as_deref()
            .or(self.config.language.as_deref())
            .unwrap_or(DEFAULT_LANGUAGE)
            .to_owned();
        let prompt = self
            .config
            .prompt
            .as_deref()
            .unwrap_or(DEFAULT_PROMPT)
            .to_owned();

        // ADK Audio and Lucy's workspace reqwest currently resolve different
        // `bytes` crate versions. Convert to an owned Vec so the multipart
        // request remains independent of that crate-version detail.
        let part = multipart::Part::bytes(wav.to_vec())
            .file_name("lucy-live.wav")
            .mime_str("audio/wav")
            .map_err(|e| AudioError::Stt {
                provider: "groq".into(),
                message: format!("invalid audio MIME type: {e}"),
            })?;
        let form = multipart::Form::new()
            .part("file", part)
            .text("model", model)
            .text("response_format", "json")
            .text("temperature", self.config.temperature.to_string())
            .text("language", language)
            .text("prompt", prompt);

        let response = self
            .client
            .post(GROQ_ENDPOINT)
            .bearer_auth(&self.config.api_key)
            .multipart(form)
            .send()
            .await
            .map_err(|e| AudioError::Stt {
                provider: "groq".into(),
                message: format!("request failed: {e}"),
            })?;

        let status = response.status();
        let body = response.text().await.map_err(|e| AudioError::Stt {
            provider: "groq".into(),
            message: format!("failed to read response: {e}"),
        })?;
        if !status.is_success() {
            return Err(AudioError::Stt {
                provider: "groq".into(),
                message: format!("Groq returned {status}: {body}"),
            });
        }
        let parsed: GroqResponse = serde_json::from_str(&body).map_err(|e| AudioError::Stt {
            provider: "groq".into(),
            message: format!("invalid Groq response: {e}"),
        })?;
        Ok(Transcript { text: parsed.text, language: None })
    }
}

// Remaining capture/VAD helpers are intentionally implemented behind ADK's
// audio interfaces; the provider above is the only cloud-specific adapter.
