use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use reqwest::{multipart, Client, StatusCode};
use serde::Deserialize;
use std::{path::Path, sync::{Arc, Mutex}, time::{Duration, Instant}};

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
        let api_key = std::env::var("GROQ_API_KEY").context("GROQ_API_KEY is not set")?;
        Ok(Self {
            api_key,
            model: std::env::var("LUCY_STT_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_owned()),
            language: std::env::var("LUCY_STT_LANGUAGE").ok(),
            prompt: std::env::var("LUCY_STT_PROMPT").ok(),
            temperature: std::env::var("LUCY_STT_TEMPERATURE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0),
            timeout: Duration::from_secs(60),
        })
    }
}

#[derive(Debug, Clone)]
pub struct GroqStt { client: Client, config: GroqSttConfig }

#[derive(Debug, Deserialize)]
struct GroqErrorResponse { error: Option<GroqError> }
#[derive(Debug, Deserialize)]
struct GroqError { message: Option<String> }
#[derive(Debug, Deserialize)]
struct TranscriptionResponse { text: String }

impl GroqStt {
    pub fn new(config: GroqSttConfig) -> Result<Self> {
        let client = Client::builder().timeout(config.timeout).build().context("failed to create HTTP client")?;
        Ok(Self { client, config })
    }
    pub fn from_env() -> Result<Self> { Self::new(GroqSttConfig::from_env()?) }

    pub async fn transcribe_file<P: AsRef<Path>>(&self, path: P) -> Result<String> {
        let path = path.as_ref();
        let audio = tokio::fs::read(path).await.with_context(|| format!("failed to read audio file: {}", path.display()))?;
        let filename = path.file_name().and_then(|v| v.to_str()).unwrap_or("audio.wav");
        self.transcribe_bytes(audio, filename, mime_for_filename(filename)).await
    }

    pub async fn transcribe_bytes(&self, audio: Vec<u8>, filename: &str, mime: &str) -> Result<String> {
        if audio.is_empty() { return Err(anyhow!("audio input is empty")); }
        if audio.len() > 25 * 1024 * 1024 { return Err(anyhow!("audio exceeds Groq's 25 MB upload limit")); }
        let part = multipart::Part::bytes(audio).file_name(filename.to_owned()).mime_str(mime).context("invalid audio MIME type")?;
        let mut form = multipart::Form::new().part("file", part).text("model", self.config.model.clone()).text("response_format", "json").text("temperature", self.config.temperature.to_string());
        if let Some(language) = &self.config.language { form = form.text("language", language.clone()); }
        if let Some(prompt) = &self.config.prompt { form = form.text("prompt", prompt.clone()); }
        let response = self.client.post(GROQ_ENDPOINT).bearer_auth(&self.config.api_key).multipart(form).send().await.context("Groq STT request failed")?;
        let status = response.status();
        let body = response.text().await.context("failed to read Groq response")?;
        if status != StatusCode::OK {
            let message = serde_json::from_str::<GroqErrorResponse>(&body).ok().and_then(|e| e.error).and_then(|e| e.message).unwrap_or(body);
            return Err(anyhow!("Groq STT returned {}: {}", status, message));
        }
        Ok(serde_json::from_str::<TranscriptionResponse>(&body).context("Groq returned an invalid transcription response")?.text.trim().to_owned())
    }

    /// Captures one natural utterance from the default microphone.
    /// Recording starts after speech is detected and ends after a short silence.
    pub async fn listen_once(&self) -> Result<String> {
        let (audio, sample_rate, channels) = tokio::task::spawn_blocking(capture_utterance).await.context("microphone task failed")??;
        let wav = encode_wav(&audio, sample_rate, channels)?;
        self.transcribe_bytes(wav, "lucy-live.wav", "audio/wav").await
    }

    /// Continuously listens for utterances and invokes `on_text` for each transcription.
    pub async fn listen<F, Fut>(&self, mut on_text: F) -> Result<()>
    where F: FnMut(String) -> Fut, Fut: std::future::Future<Output = Result<()>> {
        loop {
            let text = self.listen_once().await?;
            if !text.is_empty() { on_text(text).await?; }
        }
    }
}

fn capture_utterance() -> Result<(Vec<i16>, u32, u16)> {
    let host = cpal::default_host();
    let device = host.default_input_device().ok_or_else(|| anyhow!("no default microphone found"))?;
    let supported = device.default_input_config().context("failed to get microphone configuration")?;
    let sample_rate = supported.sample_rate().0;
    let channels = supported.channels();
    let config: cpal::StreamConfig = supported.clone().into();
    let samples: Arc<Mutex<Vec<i16>>> = Arc::new(Mutex::new(Vec::new()));
    let samples_cb = samples.clone();
    let err_fn = |err| eprintln!("Lucy microphone error: {err}");
    let stream = match supported.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(&config, move |data: &[f32], _| push_f32(data, &samples_cb), err_fn, None),
        cpal::SampleFormat::I16 => device.build_input_stream(&config, move |data: &[i16], _| push_i16(data, &samples_cb), err_fn, None),
        cpal::SampleFormat::U16 => device.build_input_stream(&config, move |data: &[u16], _| push_u16(data, &samples_cb), err_fn, None),
        format => return Err(anyhow!("unsupported microphone sample format: {format:?}")),
    }.context("failed to build microphone stream")?;
    stream.play().context("failed to start microphone")?;

    const START_THRESHOLD: f32 = 0.012;
    const END_THRESHOLD: f32 = 0.008;
    const START_WINDOW: Duration = Duration::from_millis(120);
    const SILENCE_AFTER_SPEECH: Duration = Duration::from_millis(750);
    const MAX_UTTERANCE: Duration = Duration::from_secs(20);
    const POLL: Duration = Duration::from_millis(40);

    let started = Instant::now();
    let mut speech_started = false;
    let mut last_voice = Instant::now();
    let mut seen_samples = 0usize;
    loop {
        std::thread::sleep(POLL);
        let current = samples.lock().map_err(|_| anyhow!("microphone buffer poisoned"))?.clone();
        let recent = &current[seen_samples.min(current.len())..];
        if !recent.is_empty() {
            let rms = rms_i16(recent);
            if !speech_started && rms >= START_THRESHOLD && started.elapsed() >= START_WINDOW {
                speech_started = true;
                last_voice = Instant::now();
            } else if speech_started && rms >= END_THRESHOLD {
                last_voice = Instant::now();
            }
            seen_samples = current.len();
        }
        if speech_started && last_voice.elapsed() >= SILENCE_AFTER_SPEECH { break; }
        if started.elapsed() >= MAX_UTTERANCE { break; }
    }
    drop(stream);
    let result = samples.lock().map_err(|_| anyhow!("microphone buffer poisoned"))?.clone();
    if !speech_started || result.is_empty() { return Err(anyhow!("no speech detected")); }
    Ok((result, sample_rate, channels))
}

fn rms_i16(samples: &[i16]) -> f32 {
    if samples.is_empty() { return 0.0; }
    let sum = samples.iter().map(|&s| { let x = s as f32 / 32768.0; x * x }).sum::<f32>();
    (sum / samples.len() as f32).sqrt()
}
fn push_f32(data: &[f32], dst: &Arc<Mutex<Vec<i16>>>) { if let Ok(mut out) = dst.lock() { out.extend(data.iter().map(|&x| (x.clamp(-1.0, 1.0) * 32767.0) as i16)); } }
fn push_i16(data: &[i16], dst: &Arc<Mutex<Vec<i16>>>) { if let Ok(mut out) = dst.lock() { out.extend_from_slice(data); } }
fn push_u16(data: &[u16], dst: &Arc<Mutex<Vec<i16>>>) { if let Ok(mut out) = dst.lock() { out.extend(data.iter().map(|&x| (x as i32 - 32768) as i16)); } }

fn encode_wav(samples: &[i16], sample_rate: u32, channels: u16) -> Result<Vec<u8>> {
    let mut cursor = std::io::Cursor::new(Vec::new());
    let spec = hound::WavSpec { channels, sample_rate, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut writer = hound::WavWriter::new(&mut cursor, spec).context("failed to create WAV")?;
    for &sample in samples { writer.write_sample(sample).context("failed to encode WAV")?; }
    writer.finalize().context("failed to finalize WAV")?;
    Ok(cursor.into_inner())
}

fn mime_for_filename(filename: &str) -> &'static str {
    match filename.rsplit('.').next().unwrap_or_default().to_ascii_lowercase().as_str() {
        "mp3" => "audio/mpeg", "mp4" => "audio/mp4", "m4a" => "audio/mp4", "ogg" | "oga" => "audio/ogg", "webm" => "audio/webm", "flac" => "audio/flac", "wav" => "audio/wav", _ => "application/octet-stream",
    }
}
