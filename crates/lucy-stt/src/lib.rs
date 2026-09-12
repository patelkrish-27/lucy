use anyhow::{anyhow, Context, Result};
use reqwest::{multipart, Client, StatusCode};
use serde::Deserialize;
use std::{path::Path, time::Duration};

const GROQ_ENDPOINT: &str = "https://api.groq.com/openai/v1/audio/transcriptions";
const DEFAULT_MODEL: &str = "whisper-large-v3-turbo";

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
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("GROQ_API_KEY")
            .context("GROQ_API_KEY is not set")?;

        Ok(Self {
            api_key,
            model: std::env::var("LUCY_STT_MODEL")
                .unwrap_or_else(|_| DEFAULT_MODEL.to_owned()),
            language: std::env::var("LUCY_STT_LANGUAGE").ok(),
            prompt: std::env::var("LUCY_STT_PROMPT").ok(),
            temperature: std::env::var("LUCY_STT_TEMPERATURE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
            timeout: Duration::from_secs(60),
        })
    }
}

#[derive(Debug, Clone)]
pub struct GroqStt {
    client: Client,
    config: GroqSttConfig,
}

#[derive(Debug, Deserialize)]
struct GroqErrorResponse {
    error: Option<GroqError>,
}

#[derive(Debug, Deserialize)]
struct GroqError {
    message: Option<String>,
}

impl GroqStt {
    pub fn new(config: GroqSttConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .context("failed to create HTTP client")?;
        Ok(Self { client, config })
    }

    pub fn from_env() -> Result<Self> {
        Self::new(GroqSttConfig::from_env()?)
    }

    pub async fn transcribe_file<P: AsRef<Path>>(&self, path: P) -> Result<String> {
        let path = path.as_ref();
        let audio = tokio::fs::read(path)
            .await
            .with_context(|| format!("failed to read audio file: {}", path.display()))?;
        let filename = path
            .file_name()
            .and_then(|v| v.to_str())
            .unwrap_or("audio.wav");

        let mime = mime_for_filename(filename);
        self.transcribe_bytes(audio, filename, mime).await
    }

    pub async fn transcribe_bytes(
        &self,
        audio: Vec<u8>,
        filename: &str,
        mime: &str,
    ) -> Result<String> {
        if audio.is_empty() {
            return Err(anyhow!("audio input is empty"));
        }
        if audio.len() > 25 * 1024 * 1024 {
            return Err(anyhow!("audio file exceeds Groq's 25 MB upload limit"));
        }

        let part = multipart::Part::bytes(audio)
            .file_name(filename.to_owned())
            .mime_str(mime)
            .context("invalid audio MIME type")?;

        let mut form = multipart::Form::new()
            .part("file", part)
            .text("model", self.config.model.clone())
            .text("response_format", "json")
            .text("temperature", self.config.temperature.to_string());

        if let Some(language) = &self.config.language {
            form = form.text("language", language.clone());
        }
        if let Some(prompt) = &self.config.prompt {
            form = form.text("prompt", prompt.clone());
        }

        let response = self.client
            .post(GROQ_ENDPOINT)
            .bearer_auth(&self.config.api_key)
            .multipart(form)
            .send()
            .await
            .context("Groq STT request failed")?;

        let status = response.status();
        let body = response.text().await.context("failed to read Groq response")?;

        if status != StatusCode::OK {
            let message = serde_json::from_str::<GroqErrorResponse>(&body)
                .ok()
                .and_then(|e| e.error)
                .and_then(|e| e.message)
                .unwrap_or(body);
            return Err(anyhow!("Groq STT returned {}: {}", status, message));
        }

        let result: TranscriptionResponse = serde_json::from_str(&body)
            .context("Groq returned an invalid transcription response")?;

        Ok(result.text.trim().to_owned())
    }
}

#[derive(Debug, Deserialize)]
struct TranscriptionResponse {
    text: String,
}

fn mime_for_filename(filename: &str) -> &'static str {
    match filename.rsplit('.').next().unwrap_or_default().to_ascii_lowercase().as_str() {
        "mp3" => "audio/mpeg",
        "mp4" => "audio/mp4",
        "m4a" => "audio/mp4",
        "ogg" => "audio/ogg",
        "oga" => "audio/ogg",
        "webm" => "audio/webm",
        "flac" => "audio/flac",
        "wav" => "audio/wav",
        _ => "application/octet-stream",
    }
}
