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
    pub fn from_config(cfg: &LucyConfig) -> Result<Self> {
        let api_key = cfg.stt_api_key().context(
            "STT API key is not set — set Voice API Key in Settings or export GROQ_API_KEY / LUCY_STT_API_KEY",
        )?;
        Ok(Self {
            api_key,
            model: cfg.voice.model.clone(),
            language: cfg.voice.language.clone(),
            prompt: std::env::var("LUCY_STT_PROMPT").ok(),
            temperature: std::env::var("LUCY_STT_TEMPERATURE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
            timeout: Duration::from_secs(60),
        })
    }

    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("GROQ_API_KEY")
            .or_else(|_| std::env::var("LUCY_STT_API_KEY"))
            .or_else(|_| std::env::var("STT_API_KEY"))
            .context("STT API key is not set")?;
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

#[derive(Debug, Deserialize)]
struct GroqErrorResponse { error: Option<GroqError> }
#[derive(Debug, Deserialize)]
struct GroqError { message: Option<String> }
#[derive(Debug, Deserialize)]
struct TranscriptionResponse { text: String }

#[derive(Debug, Clone)]
struct AdkGroqStt { client: Client, config: GroqSttConfig }

impl AdkGroqStt {
    fn new(config: GroqSttConfig) -> Result<Self> {
        let client = Client::builder().timeout(config.timeout).build()
            .context("failed to create HTTP client")?;
        Ok(Self { client, config })
    }

    async fn transcribe_frame(&self, audio: &AudioFrame, opts: &SttOptions) -> std::result::Result<Transcript, AudioError> {
        let wav = encode(audio, AudioFormat::Wav).map_err(|e| AudioError::Codec(e.to_string()))?;
        if wav.len() > MAX_UPLOAD_BYTES {
            return Err(AudioError::Stt { provider: "groq".into(), message: "audio exceeds Groq's 25 MB upload limit".into() });
        }
        let model = opts.model_hint.as_deref().unwrap_or(&self.config.model).to_owned();
        let language = opts.language.as_deref().or(self.config.language.as_deref()).unwrap_or(DEFAULT_LANGUAGE).to_owned();
        let prompt = self.config.prompt.as_deref().unwrap_or(DEFAULT_PROMPT).to_owned();
        let part = multipart::Part::bytes(wav.to_vec()).file_name("lucy-live.wav").mime_str("audio/wav")
            .map_err(|e| AudioError::Stt { provider: "groq".into(), message: format!("invalid audio MIME type: {e}") })?;
        let form = multipart::Form::new().part("file", part).text("model", model)
            .text("response_format", "json").text("temperature", self.config.temperature.to_string())
            .text("language", language).text("prompt", prompt);
        let response = self.client.post(GROQ_ENDPOINT).bearer_auth(&self.config.api_key).multipart(form).send().await
            .map_err(|e| AudioError::Stt { provider: "groq".into(), message: format!("Groq STT request failed: {e}") })?;
        let status = response.status();
        let body = response.text().await.map_err(|e| AudioError::Stt { provider: "groq".into(), message: format!("failed to read Groq response: {e}") })?;
        if status != StatusCode::OK {
            let message = serde_json::from_str::<GroqErrorResponse>(&body).ok().and_then(|e| e.error).and_then(|e| e.message).unwrap_or(body);
            return Err(AudioError::Stt { provider: "groq".into(), message: format!("Groq STT returned {status}: {message}") });
        }
        let text = serde_json::from_str::<TranscriptionResponse>(&body).map_err(|e| AudioError::Stt { provider: "groq".into(), message: format!("Groq returned an invalid transcription response: {e}") })?.text.trim().to_owned();
        Ok(Transcript { text, language_detected: self.config.language.clone(), ..Default::default() })
    }
}

#[async_trait::async_trait]
impl SttProvider for AdkGroqStt {
    async fn transcribe(&self, audio: &AudioFrame, opts: &SttOptions) -> std::result::Result<Transcript, AudioError> { self.transcribe_frame(audio, opts).await }
    async fn transcribe_stream(&self, mut audio: std::pin::Pin<Box<dyn futures::Stream<Item = AudioFrame> + Send>>, opts: &SttOptions) -> std::result::Result<std::pin::Pin<Box<dyn futures::Stream<Item = std::result::Result<Transcript, AudioError>> + Send>>, AudioError> {
        let mut frames = Vec::new();
        while let Some(frame) = audio.next().await { frames.push(frame); }
        if frames.is_empty() { return Err(AudioError::Stt { provider: "groq".into(), message: "audio stream is empty".into() }); }
        let merged = lucy_adk::adk_audio::merge_frames(&frames);
        let result = self.transcribe_frame(&merged, opts).await;
        Ok(Box::pin(stream::once(async move { result })))
    }
}

#[derive(Debug, Clone)]
pub struct GroqStt { config: GroqSttConfig, provider: Arc<AdkGroqStt> }

impl GroqStt {
    pub fn new(config: GroqSttConfig) -> Result<Self> { let provider = Arc::new(AdkGroqStt::new(config.clone())?); Ok(Self { config, provider }) }
    pub fn from_config(cfg: &LucyConfig) -> Result<Self> { Self::new(GroqSttConfig::from_config(cfg)?) }
    pub fn from_env() -> Result<Self> { Self::new(GroqSttConfig::from_env()?) }
    pub async fn transcribe_file<P: AsRef<Path>>(&self, path: P) -> Result<String> {
        let path = path.as_ref();
        let audio = tokio::fs::read(path).await.with_context(|| format!("failed to read audio file: {}", path.display()))?;
        let filename = path.file_name().and_then(|v| v.to_str()).unwrap_or("audio.wav");
        self.transcribe_bytes(audio, filename, mime_for_filename(filename)).await
    }
    pub async fn transcribe_bytes(&self, audio: Vec<u8>, filename: &str, mime: &str) -> Result<String> {
        if audio.is_empty() { return Err(anyhow!("audio input is empty")); }
        if audio.len() > MAX_UPLOAD_BYTES { return Err(anyhow!("audio exceeds Groq's 25 MB upload limit")); }
        let part = multipart::Part::bytes(audio).file_name(filename.to_owned()).mime_str(mime).context("invalid audio MIME type")?;
        let form = multipart::Form::new().part("file", part).text("model", self.config.model.clone()).text("response_format", "json").text("temperature", self.config.temperature.to_string())
            .text("language", self.config.language.as_deref().unwrap_or(DEFAULT_LANGUAGE).to_owned()).text("prompt", self.config.prompt.as_deref().unwrap_or(DEFAULT_PROMPT).to_owned());
        let response = self.provider.client.post(GROQ_ENDPOINT).bearer_auth(&self.config.api_key).multipart(form).send().await.context("Groq STT request failed")?;
        let status = response.status(); let body = response.text().await.context("failed to read Groq response")?;
        if status != StatusCode::OK { let message = serde_json::from_str::<GroqErrorResponse>(&body).ok().and_then(|e| e.error).and_then(|e| e.message).unwrap_or(body); return Err(anyhow!("Groq STT returned {status}: {message}")); }
        Ok(serde_json::from_str::<TranscriptionResponse>(&body).context("Groq returned an invalid transcription response")?.text.trim().to_owned())
    }
    pub async fn listen_once(&self) -> Result<String> {
        let mut capture = AudioCapture::new(); let device_id = input_device_id()?; let config = adk_capture_config();
        let mut stream = capture.start_capture(&device_id, &config).map_err(|e| anyhow!(e.to_string()))?;
        let started = std::time::Instant::now(); let mut speech_started = false; let mut last_voice = started; let mut frames = Vec::new();
        while started.elapsed() < MAX_UTTERANCE {
            let Some(frame) = stream.recv().await else { break };
            let speaking = rms_i16(frame.samples()) >= if speech_started { END_THRESHOLD } else { START_THRESHOLD };
            frames.push(frame);
            if !speech_started { if speaking && started.elapsed() >= START_WINDOW { speech_started = true; last_voice = std::time::Instant::now(); } } else if speaking { last_voice = std::time::Instant::now(); }
            if speech_started && last_voice.elapsed() >= SILENCE_AFTER_SPEECH { break; }
        }
        capture.stop_capture(); if !speech_started { return Err(anyhow!("no speech detected")); }
        let frame = collected_frame(&frames)?;
        Ok(self.provider.transcribe_frame(&frame, &self.stt_options()).await.map_err(|e| anyhow!(e.to_string()))?.text)
    }
    pub async fn listen<F, Fut>(&self, mut on_text: F) -> Result<()> where F: FnMut(String) -> Fut, Fut: std::future::Future<Output = Result<()>> { loop { let text = self.listen_once().await?; if !text.is_empty() { on_text(text).await?; } } }
    pub fn start_hold(&self) -> Result<HoldCapture> { HoldCapture::start() }
    pub async fn finish_hold(&self, cap: HoldCapture) -> Result<String> {
        let frames = cap.finish(); let frame = collected_frame(&frames)?;
        if frame.duration_ms < HOLD_MIN_AUDIO.as_millis() as u32 { return Err(anyhow!("too short — hold the hotkey while speaking, then release")); }
        if !has_speech(frame.samples(), frame.sample_rate, frame.channels as u16) { return Err(anyhow!("no speech detected")); }
        Ok(self.provider.transcribe_frame(&frame, &self.stt_options()).await.map_err(|e| anyhow!(e.to_string()))?.text)
    }
    fn stt_options(&self) -> SttOptions { SttOptions { language: self.config.language.clone(), model_hint: Some(self.config.model.clone()), ..Default::default() } }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LucyEnergyVad;
impl VadProcessor for LucyEnergyVad {
    fn is_speech(&self, frame: &AudioFrame) -> bool { rms_i16(frame.samples()) >= END_THRESHOLD }
    fn segment(&self, frame: &AudioFrame) -> Vec<SpeechSegment> { if self.is_speech(frame) { vec![SpeechSegment { start_ms: 0, end_ms: frame.duration_ms }] } else { Vec::new() } }
}

pub struct HoldCapture { capture: Option<AudioCapture>, task: Option<tokio::task::JoinHandle<()>>, frames: Arc<Mutex<Vec<AudioFrame>>> }
impl HoldCapture {
    fn start() -> Result<Self> {
        let mut capture = AudioCapture::new(); let device_id = input_device_id()?; let config = adk_capture_config();
        let mut stream = capture.start_capture(&device_id, &config).map_err(|e| anyhow!(e.to_string()))?;
        let frames = Arc::new(Mutex::new(Vec::new())); let sink = Arc::clone(&frames);
        let task = tokio::spawn(async move { while let Some(frame) = stream.recv().await { if let Ok(mut frames) = sink.lock() { frames.push(frame); } } });
        Ok(Self { capture: Some(capture), task: Some(task), frames })
    }
    fn finish(mut self) -> Vec<AudioFrame> { if let Some(mut capture) = self.capture.take() { capture.stop_capture(); } if let Some(task) = self.task.take() { task.abort(); } self.frames.lock().map(|frames| frames.clone()).unwrap_or_default() }
}

fn input_device_id() -> Result<String> {
    if let Ok(id) = std::env::var("LUCY_AUDIO_INPUT_DEVICE") { if !id.trim().is_empty() { return Ok(id); } }
    let devices = AudioCapture::list_input_devices().map_err(|e| anyhow!(e.to_string()))?;
    devices.first().map(|d| d.id().to_owned()).ok_or_else(|| anyhow!("no default microphone found"))
}
fn adk_capture_config() -> CaptureConfig { CaptureConfig::default() }
fn collected_frame(frames: &[AudioFrame]) -> Result<AudioFrame> { if frames.is_empty() { return Err(anyhow!("no audio captured")); } Ok(lucy_adk::adk_audio::merge_frames(frames)) }
fn rms_i16(samples: &[i16]) -> f32 { if samples.is_empty() { return 0.0; } let sum = samples.iter().map(|&s| { let x = s as f32 / 32768.0; x * x }).sum::<f32>(); (sum / samples.len() as f32).sqrt() }
pub fn has_speech(samples: &[i16], sample_rate: u32, channels: u16) -> bool { if samples.is_empty() { return false; } let ch = channels.max(1) as usize; let window_size = ((sample_rate as usize) * ch / 10).max(1); let mut speech_frames = 0; let mut max_rms = 0.0f32; for chunk in samples.chunks(window_size) { let r = rms_i16(chunk); if r > max_rms { max_rms = r; } if r >= END_THRESHOLD { speech_frames += 1; } } speech_frames >= 2 || max_rms >= 0.015 }
fn mime_for_filename(filename: &str) -> &'static str { match filename.rsplit('.').next().unwrap_or_default().to_ascii_lowercase().as_str() { "mp3" => "audio/mpeg", "mp4" => "audio/mp4", "m4a" => "audio/mp4", "ogg" | "oga" => "audio/ogg", "webm" => "audio/webm", "flac" => "audio/flac", "wav" => "audio/wav", _ => "application/octet-stream" } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn detects_silence_correctly() { let silence = vec![0i16; 16000]; assert!(!has_speech(&silence, 16000, 1)); let noise: Vec<i16> = (0..16000).map(|i| if i % 2 == 0 { 40 } else { -40 }).collect(); assert!(!has_speech(&noise, 16000, 1)); }
    #[test] fn detects_speech_signal() { let mut audio = vec![0i16; 16000]; for i in 4000..10400 { audio[i] = ((i as f32 * 0.1).sin() * 2000.0) as i16; } assert!(has_speech(&audio, 16000, 1)); }
    #[test] fn adk_vad_preserves_lucy_threshold() { let frame = AudioFrame::new(vec![2000i16].into_iter().flat_map(i16::to_le_bytes).collect::<Vec<_>>(), 16000, 1); assert!(LucyEnergyVad::default().is_speech(&frame)); }
}
