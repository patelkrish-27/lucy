use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{
    env, fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct LucyConfig {
    pub general: GeneralConfig,
    pub models: ModelConfig,
    /// OpenAI-compatible endpoints the user connected in `/settings`.
    /// Each entry carries its own API key and the model list discovered by
    /// `Test Connection` (see `lucy_agent::ProviderClient`).
    pub providers: Vec<ProviderConfig>,
    /// `decider-serve` intent/reasoning classifier. API URL only — the local
    /// server takes no API key.
    pub classification: ClassificationConfig,
    /// Chat mode, selected models, reasoning tiers, auto-compaction.
    pub chat: ChatSettingsConfig,
    pub planner: PlannerConfig,
    pub voice: VoiceConfig,
    pub hyprfast: HyprFastConfig,
    pub appearance: AppearanceConfig,
    pub sessions: SessionConfig,
    pub approvals: ApprovalConfig,
    pub browser: BrowserConfig,
    pub harness: HarnessConfig,
    pub system_one: SystemOneConfig,
    /// Lucy's durable knowledge base: what she remembers, and how much of it
    /// she is allowed to put in front of a model on any one turn.
    pub knowledge: KnowledgeConfig,
    /// The phone bridge: a local WebSocket/REST server the Lucy mobile app
    /// connects to. Off by default — this is the one part of Lucy that opens a
    /// network port, so it is opt-in and switchable at runtime.
    pub gateway: GatewayConfig,
    /// The on-screen companion: whether live screen/window summaries may be
    /// attached to a turn, and how much of a summary a single turn may carry.
    pub companion: CompanionConfig,
}

/// Which model family a connected provider serves. Drives which settings
/// dropdowns (the three reasoning tiers, `voice_model`) it populates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    #[default]
    Text,
    Voice,
}

impl ProviderType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Text => "Text",
            Self::Voice => "Voice",
        }
    }
    /// Case-insensitive parse accepting `text`/`voice` (and the UI labels).
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "text" | "chat" | "llm" => Some(Self::Text),
            "voice" | "audio" | "stt" | "tts" => Some(Self::Voice),
            _ => None,
        }
    }
}

/// One connected OpenAI-compatible endpoint.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ProviderConfig {
    /// Stable identifier — model selections reference this, never the name.
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub name: String,
    /// Base URL. The trailing `/v1` is optional; it is added when missing.
    #[serde(default)]
    pub api_url: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub api_key: String,
    #[serde(default)]
    pub provider_type: ProviderType,
    /// Populated by `Test Connection` (`GET {api_url}/models`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub available_models: Vec<String>,
    /// Model ids this endpoint flagged as deprecated. Aliases can stay listed
    /// indefinitely, but a deprecated id is never picked as a fresh default.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deprecated_models: Vec<String>,
}

impl ProviderConfig {
    /// Base URL with any trailing slash removed — safe to `format!` a path onto.
    pub fn base_url(&self) -> &str {
        self.api_url.trim().trim_end_matches('/')
    }

    /// Base URL guaranteed to end in `/v1` (or `/vN`), which is what an
    /// OpenAI-compatible `/models` + `/chat/completions` pair expects.
    pub fn openai_base_url(&self) -> String {
        let base = self.base_url();
        if base.is_empty() {
            return String::new();
        }
        let last = base.rsplit('/').next().unwrap_or_default();
        if last.len() > 1 && last.starts_with('v') && last[1..].chars().all(|c| c.is_ascii_digit())
        {
            base.to_owned()
        } else {
            format!("{base}/v1")
        }
    }

    /// `GET {openai_base_url}/models`.
    pub fn models_endpoint(&self) -> String {
        format!("{}/models", self.openai_base_url())
    }

    /// Slug derived from the name/URL, used as the id when the user leaves it
    /// blank. Reads no global state, so it is safe for a draft that has not
    /// been persisted yet.
    pub fn id_slug(&self) -> String {
        let seed = if !self.name.trim().is_empty() {
            self.name.clone()
        } else {
            self.api_url.clone()
        };
        // Collapse runs of separators so `https://a.test/v1` becomes
        // `https-a-test-v1` rather than `https---a-test-v1`.
        let mut slug = String::with_capacity(seed.len());
        for c in seed.to_ascii_lowercase().chars() {
            let c = if c.is_ascii_alphanumeric() { c } else { '-' };
            if c == '-' && slug.ends_with('-') {
                continue;
            }
            slug.push(c);
        }
        let slug = slug.trim_matches('-').to_owned();
        if slug.is_empty() {
            "provider".to_owned()
        } else {
            slug
        }
    }

    /// Assign an id when the user left it blank, uniquified against the ids
    /// already in use.
    pub fn ensure_id(&mut self, taken: &[String]) {
        if !self.id.trim().is_empty() {
            return;
        }
        let base = self.id_slug();
        let mut candidate = base.clone();
        let mut n = 2;
        while taken.iter().any(|t| t.eq_ignore_ascii_case(&candidate)) {
            candidate = format!("{base}-{n}");
            n += 1;
        }
        self.id = candidate;
    }

    /// Display label used by the settings dropdowns.
    pub fn label(&self) -> String {
        if !self.name.trim().is_empty() {
            self.name.trim().to_owned()
        } else if !self.api_url.trim().is_empty() {
            self.api_url.trim().to_owned()
        } else {
            "(unnamed provider)".into()
        }
    }

    /// True when `url` and `key` are both present enough to attempt a probe.
    pub fn is_usable(&self) -> bool {
        !self.api_url.trim().is_empty() && !self.api_key.trim().is_empty()
    }

    /// True when this endpoint marked `model` as deprecated.
    pub fn is_deprecated_model(&self, model: &str) -> bool {
        let m = model.trim();
        self.deprecated_models.iter().any(|d| d.trim() == m)
    }

    /// Advertised models that are not deprecated, in stored order.
    pub fn live_models(&self) -> impl Iterator<Item = &String> {
        self.available_models
            .iter()
            .filter(|m| !self.is_deprecated_model(m))
    }
}

/// OpenRouter's public OpenAI-compatible base URL.
pub const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";
/// Attribution OpenRouter asks every client to send, so an app can be
/// recognised on openrouter.ai rankings. Purely informational to the API.
pub const OPENROUTER_REFERER: &str = "https://github.com/patelkrish-27/lucy";
pub const OPENROUTER_TITLE: &str = "Lucy";
/// Model used when the configured name cannot exist on OpenRouter, whose ids
/// are always `vendor/model`.
pub const OPENROUTER_DEFAULT_MODEL: &str = "z-ai/glm-5.3";

/// The text model served by the bundled local OpenChat proxy, used when no
/// reasoning tier is bound at all. Lucy has exactly four selectable models —
/// L1, L2, L3 and the voice model — so this is the last-resort floor of the
/// tier fallback chain, never a fifth choice the user can pick.
pub const DEFAULT_TEXT_MODEL: &str = "gemini-web";

/// The endpoint that [`DEFAULT_TEXT_MODEL`] is served by.
pub const DEFAULT_TEXT_BASE_URL: &str = "http://127.0.0.1:11435/v1";

/// A ready-made endpoint the `/settings` provider form can fill in, so the
/// user picks a vendor instead of remembering a base URL and a key env var.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderPreset {
    /// Type the URL by hand. Never rewrites the API URL field.
    Custom,
    OpenRouter,
    OpenAi,
    Groq,
    Ollama,
    OpenChat,
}

impl ProviderPreset {
    /// Preset order, as the settings dropdown cycles it.
    pub const ALL: [ProviderPreset; 6] = [
        ProviderPreset::Custom,
        ProviderPreset::OpenRouter,
        ProviderPreset::OpenAi,
        ProviderPreset::Groq,
        ProviderPreset::Ollama,
        ProviderPreset::OpenChat,
    ];

    /// Display name in the settings dropdown.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Custom => "Custom",
            Self::OpenRouter => "OpenRouter",
            Self::OpenAi => "OpenAI",
            Self::Groq => "Groq",
            Self::Ollama => "Ollama",
            Self::OpenChat => "OpenChat (local)",
        }
    }

    /// Base URL to fill into the API URL field. `None` for [`Self::Custom`],
    /// which must never overwrite what the user typed.
    pub fn api_url(&self) -> Option<&'static str> {
        match self {
            Self::Custom => None,
            Self::OpenRouter => Some(OPENROUTER_BASE_URL),
            Self::OpenAi => Some("https://api.openai.com/v1"),
            Self::Groq => Some("https://api.groq.com/openai/v1"),
            Self::Ollama => Some("http://127.0.0.1:11434/v1"),
            Self::OpenChat => Some("http://127.0.0.1:11435/v1"),
        }
    }

    /// Env var holding this vendor's key, used to pre-fill the API Key field
    /// so `export OPENROUTER_API_KEY=…` is all a terminal user needs.
    pub fn api_key_env(&self) -> Option<&'static str> {
        match self {
            Self::Custom => None,
            Self::OpenRouter => Some("OPENROUTER_API_KEY"),
            Self::OpenAi => Some("OPENAI_API_KEY"),
            Self::Groq => Some("GROQ_API_KEY"),
            Self::Ollama | Self::OpenChat => None,
        }
    }

    /// Case-insensitive parse accepting the preset name (and the URL host, so
    /// a provider already connected to OpenRouter is recognised when the
    /// screen reopens).
    pub fn parse(raw: &str) -> Option<Self> {
        let raw = raw.trim();
        if raw.is_empty() {
            return None;
        }
        Self::ALL
            .into_iter()
            .find(|p| p.name().eq_ignore_ascii_case(raw))
            .or_else(|| match host_of(raw) {
                Some(host) => {
                    let host = host.to_ascii_lowercase();
                    if host.ends_with("openrouter.ai") {
                        Some(Self::OpenRouter)
                    } else if host.ends_with("api.openai.com") {
                        Some(Self::OpenAi)
                    } else if host.ends_with("api.groq.com") {
                        Some(Self::Groq)
                    } else {
                        None
                    }
                }
                None => None,
            })
    }

    /// The preset that owns `url`, if any.
    pub fn for_url(url: &str) -> Option<Self> {
        let base = url.trim().trim_end_matches('/');
        Self::ALL
            .into_iter()
            .find(|p| p.api_url().is_some_and(|u| u.trim_end_matches('/') == base))
            .or_else(|| Self::parse(url))
    }
}

/// Host (no scheme, no port, no path) of a URL-ish string.
fn host_of(url: &str) -> Option<&str> {
    let rest = url
        .split("://")
        .nth(1)
        .unwrap_or(url)
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default()
        .rsplit('@')
        .next()
        .unwrap_or_default();
    if rest.is_empty() {
        return None;
    }
    if let Some(stripped) = rest.strip_prefix('[') {
        Some(stripped.split(']').next().unwrap_or(stripped))
    } else if rest.matches(':').count() == 1 {
        Some(rest.split(':').next().unwrap_or(rest))
    } else {
        Some(rest)
    }
}

fn is_versioned_base(base: &str) -> bool {
    let last = base.rsplit('/').next().unwrap_or_default();
    last.len() > 1 && last.starts_with('v') && last[1..].chars().all(|c| c.is_ascii_digit())
}

/// True when `base_url` is an OpenRouter endpoint, which needs `vendor/model`
/// ids and the attribution headers.
pub fn is_openrouter(base_url: &str) -> bool {
    host_of(base_url).is_some_and(|h| {
        let h = h.to_ascii_lowercase();
        h == "openrouter.ai" || h.ends_with(".openrouter.ai")
    })
}

/// The model id to actually send to `base_url`.
///
/// Every other OpenAI-compatible endpoint takes the model name as configured.
/// OpenRouter only knows `vendor/model` ids, so a bare name (`gemini-web` from
/// the local OpenChat proxy, a hand-typed `gpt-4`) can never resolve there and
/// would 404 with an opaque "No endpoints found" body. Substitute the default
/// instead, so switching a config to OpenRouter needs no model edit.
pub fn model_for_endpoint(base_url: &str, model: &str) -> String {
    let model = model.trim();
    if model.is_empty() {
        return model.to_owned();
    }
    if is_openrouter(base_url) && !model.contains('/') {
        return OPENROUTER_DEFAULT_MODEL.to_owned();
    }
    model.to_owned()
}

/// `decider-serve` (Mapika/decider-2b-vision) connection settings.
///
/// Only `classification_api_url` is user-facing: the local server is
/// unauthenticated, so the settings screen never shows an API key here.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ClassificationConfig {
    pub enabled: bool,
    /// Base URL of the `decider-serve` REST API, e.g. `http://localhost:8001`.
    pub classification_api_url: String,
    /// Per-request budget. A cold model load is slow, so this is generous.
    pub timeout_ms: u64,
    /// Informational only — the server owns the loaded checkpoint.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub model: String,
    /// Below this, the classification verdict is treated as unusable and the
    /// turn falls back to the reasoning tiers / heuristic routing.
    pub confidence_threshold: f32,
}

impl Default for ClassificationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            classification_api_url: "http://localhost:8001".into(),
            timeout_ms: 20000,
            model: String::new(),
            confidence_threshold: 0.1,
        }
    }
}

impl ClassificationConfig {
    pub fn base_url(&self) -> &str {
        self.classification_api_url.trim().trim_end_matches('/')
    }
}

/// `Auto`: the classifier picks the reasoning tier per turn. `Manual`: every
/// turn goes to the Level 3 model, skipping classification entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ChatMode {
    #[default]
    Auto,
    Manual,
}

impl ChatMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::Manual => "Manual",
        }
    }
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" | "automatic" => Some(Self::Auto),
            "manual" => Some(Self::Manual),
            _ => None,
        }
    }
}

/// The three reasoning tiers. Any text model from any connected provider can
/// be bound to a tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningLevel {
    /// Fast / lightweight (Flash-lite).
    #[serde(rename = "1")]
    L1,
    /// Balanced (Flash).
    #[serde(rename = "2")]
    #[default]
    L2,
    /// Deep reasoning / action planning (Pro).
    #[serde(rename = "3")]
    L3,
}

impl ReasoningLevel {
    pub const ALL: [Self; 3] = [Self::L1, Self::L2, Self::L3];

    pub fn as_number(&self) -> u8 {
        match self {
            Self::L1 => 1,
            Self::L2 => 2,
            Self::L3 => 3,
        }
    }

    /// The wire value `decider-serve` answers with: `"1"`, `"2"`, `"3"`.
    pub fn as_wire(&self) -> &'static str {
        match self {
            Self::L1 => "1",
            Self::L2 => "2",
            Self::L3 => "3",
        }
    }

    /// Human label used in the settings dropdowns and the chat log header.
    pub fn label(&self) -> String {
        match self {
            Self::L1 => "Level 1 (fast)".into(),
            Self::L2 => "Level 2 (balanced)".into(),
            Self::L3 => "Level 3 (deep reasoning)".into(),
        }
    }

    /// Short tier name for the active-model line in the TUI header.
    pub fn short(&self) -> &'static str {
        match self {
            Self::L1 => "L1",
            Self::L2 => "L2",
            Self::L3 => "L3",
        }
    }

    /// Tiers to try, in order, when `level` has no model bound to it: `level`
    /// first, then each cheaper tier. An unbound tier therefore degrades
    /// *downwards*, so a half-configured setup still answers and a cheap
    /// fast-lane turn is never silently sent to a deeper model. When the whole
    /// chain is unbound, [`crate::LucyConfig::resolve_level_model`] falls back
    /// to the compiled-in default text model.
    ///
    /// L3's chain is `[L3, L2, L1]`, so the deep-reasoning tier — the one that
    /// plans actions — ends up on the deepest model the user actually bound.
    pub fn fallback_chain(&self) -> [Self; 3] {
        match self {
            Self::L1 => [Self::L1, Self::L1, Self::L1],
            Self::L2 => [Self::L2, Self::L1, Self::L1],
            Self::L3 => [Self::L3, Self::L2, Self::L1],
        }
    }

    pub fn from_wire(raw: &str) -> Option<Self> {
        let s = raw.trim();
        if s.starts_with('1') || s.eq_ignore_ascii_case("l1") {
            Some(Self::L1)
        } else if s.starts_with('2') || s.eq_ignore_ascii_case("l2") {
            Some(Self::L2)
        } else if s.starts_with('3') || s.eq_ignore_ascii_case("l3") {
            Some(Self::L3)
        } else {
            None
        }
    }
}

/// One selectable model: a concrete model name hosted by a specific provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelOption {
    pub provider_id: String,
    pub provider_name: String,
    pub model: String,
}

impl ModelOption {
    /// Stable `provider_id/model` key stored in the config.
    pub fn key(&self) -> String {
        format!("{}/{}", self.provider_id, self.model)
    }

    /// Dropdown label: `Provider · model`.
    pub fn label(&self) -> String {
        format!("{} · {}", self.provider_name, self.model)
    }
}

/// Chat mode, the three reasoning tiers, the voice model and auto-compaction.
///
/// There is no separate "main model": Level 3 is the anchor. Turns that need
/// actions always plan with L3, and `Manual` chat mode pins every turn to it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatSettingsConfig {
    pub chat_mode: ChatMode,
    /// Selected voice model (`provider_id/model`).
    pub voice_model: String,
    /// Summarize/compact session history automatically between turns.
    pub auto_compact: bool,
    pub reasoning_levels: ReasoningLevelsConfig,
    /// The pre-tier "main model" selection, read only so an old config file can
    /// migrate: it seeds `level3` on load and is cleared immediately after.
    /// Never serialized, so the key does not reappear under its old name the
    /// next time Lucy saves.
    #[serde(rename = "main_model", skip_serializing_if = "String::is_empty")]
    pub legacy_main_model: String,
}

/// Model bound to each reasoning tier (`provider_id/model`, or empty when the
/// user has not picked one — see [`ReasoningLevel::fallback_chain`]).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ReasoningLevelsConfig {
    pub level1: String,
    pub level2: String,
    pub level3: String,
}

impl ReasoningLevelsConfig {
    pub fn get(&self, level: ReasoningLevel) -> &str {
        match level {
            ReasoningLevel::L1 => &self.level1,
            ReasoningLevel::L2 => &self.level2,
            ReasoningLevel::L3 => &self.level3,
        }
    }
    pub fn set(&mut self, level: ReasoningLevel, model: String) {
        match level {
            ReasoningLevel::L1 => self.level1 = model,
            ReasoningLevel::L2 => self.level2 = model,
            ReasoningLevel::L3 => self.level3 = model,
        }
    }
}

impl Default for ChatSettingsConfig {
    fn default() -> Self {
        Self {
            chat_mode: ChatMode::default(),
            voice_model: String::new(),
            auto_compact: true,
            reasoning_levels: ReasoningLevelsConfig::default(),
            legacy_main_model: String::new(),
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GeneralConfig {
    pub startup_screen: String,
    pub compact_after_command: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    /// The text model used when no reasoning tier is bound. Reads the legacy
    /// `main` key; Lucy has no user-facing "main model" any more.
    #[serde(rename = "default_text", alias = "main")]
    pub default_text: String,
    // Legacy generic (fallback for both)
    pub base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Text-model endpoint overrides. These carry the reasoning tiers too: any
    /// tier bound to a bare (unqualified) model name is sent here, which is
    /// what the keyless local servers depend on.
    #[serde(rename = "text_api_key", alias = "main_api_key")]
    pub text_api_key: Option<String>,
    #[serde(rename = "text_base_url", alias = "main_base_url")]
    pub text_base_url: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PlannerConfig {
    pub max_subtasks: usize,
    pub max_depth: usize,
    pub verify_state: bool,
    pub parallel: bool,
    pub replan_on_failure: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VoiceConfig {
    pub provider: String,
    pub model: String,
    pub language: Option<String>,
    pub push_to_talk: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HyprFastConfig {
    pub command: String,
    pub args: Vec<String>,
    pub max_candidates: usize,
    pub batching: bool,
    pub parallel: bool,
    pub verify_actions: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppearanceConfig {
    pub theme: String,
    pub animations: bool,
    pub activity_verbosity: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SessionConfig {
    pub file: Option<PathBuf>,
    pub dir: Option<PathBuf>,
    pub resume: bool,
    pub max_history: usize,
}
/// The three approval modes, in increasing order of prompting.
pub const APPROVAL_MODES: [&str; 3] = ["never", "write", "always"];

/// How often Lucy stops to ask before running a tool.
///
/// - `never` — automode: never ask. The safety net is the kill switch
///   (`/stop`, `Ctrl+C`, `Esc`) rather than a prompt.
/// - `write` — ask only for tools the registry marks as needing approval.
/// - `always` — ask before every tool, including read-only ones.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ApprovalConfig {
    pub mode: String,
    /// Tools the user chose "always allow" on. Persisted so the choice
    /// survives a restart; previously this lived only in the in-memory
    /// `ApprovalGate`, so every session re-prompted for the same tool.
    pub always_allow: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BrowserConfig {
    /// CDP port the harness polls for readiness. Also handed to hyprfast as
    /// [`HYPRFAST_CDP_PORT_ENV`], because hyprfast is what launches the
    /// browser and therefore the side that must be told which port to open.
    pub cdp_port: u16,
    /// Seconds to poll the CDP port after launch before reporting FAILED.
    pub launch_timeout_secs: u64,
    /// Flags appended to every browser launch, forwarded to hyprfast as
    /// [`HYPRFAST_BROWSER_ARGS_ENV`]. Defaults to `["--disable-gpu"]`: the
    /// browser's "use graphics acceleration when available" default is on, and
    /// GPU compositing makes CDP geometry disagree with what the compositor
    /// hit-tests, which is exactly what the click guards reason about.
    ///
    /// This is the only browser-*process* knob, because hyprfast owns the
    /// launch — there is no second list anywhere.
    pub launch_args: Vec<String>,
    /// Profile directory (`--user-data-dir`) for the CDP browser, forwarded to
    /// hyprfast as [`HYPRFAST_USER_DATA_DIR_ENV`]. Empty means "let hyprfast
    /// decide", which is `~/.local/share/hyprfast/browser-profile`.
    ///
    /// This is where logins live, so it must not be a temporary directory: a
    /// tmpfs is RAM, and every sign-in would be gone after a reboot. It must
    /// also not be the browser's own default data directory — Chromium 136+
    /// silently ignores `--remote-debugging-port` there, so the port would
    /// never open and every browser call would fail with only a timeout to
    /// show for it.
    pub user_data_dir: String,
}
impl BrowserConfig {
    /// [`Self::launch_args`] as one whitespace-separated value, or `None` when
    /// there are none (so callers skip setting a variable rather than set an
    /// empty one).
    pub fn launch_args_value(&self) -> Option<String> {
        join_flags(&self.launch_args)
    }

    /// [`Self::user_data_dir`] trimmed, or `None` when unset/blank — an
    /// exported-but-empty value would otherwise reach hyprfast as an empty
    /// `--user-data-dir`.
    pub fn user_data_dir_value(&self) -> Option<String> {
        let dir = self.user_data_dir.trim();
        (!dir.is_empty()).then(|| dir.to_string())
    }
}

/// Join argv-style flags into one whitespace-separated value for an env var.
pub fn join_flags(flags: &[String]) -> Option<String> {
    let joined = flags.join(" ");
    if joined.trim().is_empty() {
        None
    } else {
        Some(joined)
    }
}

/// Env var hyprfast reads for extra browser-launch flags.
///
/// hyprfast is the only thing that spawns a browser, so this is how
/// [`BrowserConfig::launch_args`] reaches the process at all. Without it,
/// `[browser] launch_args` would be a setting that changes nothing.
pub const HYPRFAST_BROWSER_ARGS_ENV: &str = "HYPRFAST_BROWSER_ARGS";

/// Env var hyprfast reads for the CDP host to attach to and launch against.
pub const HYPRFAST_CDP_HOST_ENV: &str = "HYPRFAST_CDP_HOST";

/// Env var hyprfast reads for the CDP port to attach to and open.
///
/// lucy's `[browser] cdp_port` and hyprfast's port must agree or the two
/// attach to different browsers, so lucy passes its own value explicitly
/// instead of relying on both sides defaulting to 9222.
pub const HYPRFAST_CDP_PORT_ENV: &str = "HYPRFAST_CDP_PORT";

/// Env var hyprfast reads for the CDP browser's `--user-data-dir`.
///
/// Hyprfast is the only launcher, so this is how
/// [`BrowserConfig::user_data_dir`] reaches the process at all — without it
/// `[browser] user_data_dir` would be a setting that changes nothing. Lucy's
/// own profile lookup reads the same variable, because the preference it
/// writes before launch has to land in the directory hyprfast will open.
pub const HYPRFAST_USER_DATA_DIR_ENV: &str = "HYPRFAST_USER_DATA_DIR";
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HarnessConfig {
    /// §3 budget: alert when a single-domain, single-app task exceeds this
    /// many LLM calls. Routine tasks must stay within 4.
    pub max_llm_calls_single_task: usize,
    /// Max scoped-recovery LLM calls per run (genuine deviations only).
    pub max_recoveries: usize,
    /// Max deterministic retries per step before escalating to recovery.
    pub max_step_retries: usize,
    /// Kept for compatibility; it no longer selects a loop.
    ///
    /// Every act entry now runs `LucyRuntime::execute_goal_outcome`, which runs
    /// the ReAct loop, so there is nothing left for this flag to choose. It is
    /// still read, still round-trips through `/settings`, and still lands in an
    /// existing `config.toml` — removing a config key would need a migration to
    /// keep that file loadable. The blind command plan it used to switch away
    /// from is unreachable from every entry point; see
    /// `LucyRuntime::execute_goal`.
    pub agent_loop_enabled: bool,
    /// Topic words that mean "this goal is a web task", so the fast lane opens
    /// a browser before its first CDP call.
    ///
    /// This is the one place a *topic* word changes behaviour, so it lives in
    /// config rather than in code: a hardcoded list is a closed world that needs
    /// a release for every new noun, and AGENTS.md forbids that. The structural
    /// signals (any URL, any dotted domain, browser/web/search wording) are
    /// built into `fast_perception::goal_needs_browser` and need no entry here
    /// — only domain- and topic-specific vocabulary belongs in this list.
    ///
    /// Defaults carry the words Lucy's own runs actually used, so removing the
    /// hardcoding changes no behaviour until the user edits this. Add whatever
    /// your tasks say: "recipe", "reservation", "invoice", "flight".
    pub browser_goal_keywords: Vec<String>,
}

/// Lucy's durable knowledge base.
///
/// Every field here is a *budget*, not a behaviour switch, and the reasoning is
/// the same one that drives the rest of the harness: Lucy's models are cheap
/// and their context is worth more than the tokens. Knowledge is therefore
/// offered in three escalating layers — a bounded table of contents on every
/// turn, deterministic recall on a turn that looks like it needs it, and a tool
/// the model opens books with — rather than injected wholesale.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct KnowledgeConfig {
    /// Master switch. Off means no capture, no recall, no tools; the store is
    /// left untouched rather than deleted.
    pub enabled: bool,
    /// Knowledge base location. `None` means `~/.config/lucy/knowledge`, which is
    /// a plain directory of Markdown files — the point of the design is that a
    /// user can open it.
    pub dir: Option<PathBuf>,
    /// Characters of the generated topic index injected per turn. It is a table
    /// of contents: enough for the model to know what exists, never enough to
    /// crowd out the task. Sized against a Level 3 planner prompt whose skill
    /// body alone is capped at 16k.
    pub digest_budget_chars: usize,
    /// Characters of recalled knowledge injected per turn. Zero disables recall
    /// while leaving the index and the tools, which is the cheapest useful
    /// configuration on a very small context.
    pub recall_budget_chars: usize,
    /// Claims a single turn may capture. A turn producing more is an extractor
    /// summarising the conversation rather than remembering it.
    pub max_captures_per_turn: usize,
    /// Recalls a captured claim must serve before promotion makes it injectable
    /// on every turn. Promotion is closed to untrusted origins regardless, so
    /// this only ever delays *owner* knowledge — it cannot be bypassed.
    pub min_recalls_for_promotion: u32,
    /// Write the turn's observations to the store after answering. Off by
    /// default: capture costs one model call per turn, and the first thing to
    /// check is whether recall earns its keep before paying for writes.
    pub capture_enabled: bool,
    /// Give the classifier a third question head naming the knowledge topics a
    /// request looks like it will need, so recall can be ordered to match.
    ///
    /// It reorders a deterministic result set and never decides *whether* to
    /// search, because the ablation work on small local models found adaptive
    /// routing losing to fixed hybrid retrieval.
    pub classifier_routing_enabled: bool,
}

impl Default for KnowledgeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            dir: None,
            digest_budget_chars: 1_200,
            recall_budget_chars: 6_000,
            max_captures_per_turn: 4,
            min_recalls_for_promotion: 2,
            capture_enabled: false,
            classifier_routing_enabled: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SystemOneConfig {
    pub enabled: bool,
    /// System-1 decision backend. `"decider"` (default: `decider-serve` at
    /// `base_url`, started manually by the user — lucy never starts it).
    /// `"laya_api"` / `"laya"` are legacy aliases of `"decider"`.
    /// `"laya_direct"` (opt-in: spawn the slow local daemon),
    /// `"typesafe"`, `"jev"`.
    pub provider: String,
    pub base_url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// HuggingFace repo served by `decider-serve`.
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub direct_python: Option<String>,
    pub confidence_threshold: f32,
    pub timeout_ms: u64,
    /// Legacy: lucy used to auto-start `layaApi serve` when no server
    /// answered `base_url`. No longer used — the user starts `decider-serve`
    /// manually and lucy only ever uses `base_url`. Kept so old config files
    /// still parse; always treated as false.
    #[serde(default)]
    pub auto_start: bool,
    /// Legacy: explicit server executable for auto-start. Unused; kept so
    /// old config files still parse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_command: Option<String>,
    /// Legacy: model flag for auto-start. Unused; kept so old config files
    /// still parse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_model: Option<String>,
}

/// The Lucy mobile gateway: a local WebSocket + REST server the phone pairs
/// with so the app can submit tasks, watch live agent events, and answer
/// approval prompts. Opt-in by default.
///
/// This is the one subsystem that opens a listening socket, spawns no work of
/// its own, and can hold a full [`LucyRuntime`] (session store, ADK memory,
/// MCP child processes) alive. So every field here is about *not* paying for it
/// unless the user asked: `enabled` is the master switch, and the idle timeout
/// releases the runtime after a period with no connected client so an idle
/// bridge costs nothing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GatewayConfig {
    /// Master switch. Off means `lucy serve` refuses to bind a port and no
    /// gateway is started from the TUI. When a user runs `lucy serve` without
    /// this set, the command prints the one-line enable hint instead of
    /// silently starting a server.
    pub enabled: bool,
    /// Port the gateway listens on. The phone app discovers this from the
    /// pairing QR / payload rather than assuming it, but the default matches
    /// the documented port so a manually-typed address works.
    pub port: u16,
    /// Bind address. `127.0.0.1` keeps the gateway loopback-only until the user
    /// deliberately chooses `0.0.0.0` (LAN/Tailscale) — the pairing flow is the
    /// second layer, not the first.
    pub bind: String,
    /// Seconds with zero connected clients before the runtime is dropped to
    /// free its memory and MCP children. `0` keeps it alive for the process
    /// lifetime (useful when the user watches `lucy serve` interactively).
    pub idle_unload_secs: u64,
    /// Show the full pairing payload (including the raw secret) in the server
    /// log. Off by default: the secret belongs in the QR code and the phone's
    /// secure storage, not in a log file that may be shared.
    pub log_pairing_secret: bool,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            port: 9847,
            bind: "127.0.0.1".into(),
            // Five minutes: long enough that a phone that briefly sleeps does
            // not make the next task pay a cold start, short enough that a
            // bridge used once does not pin MCP children all day.
            idle_unload_secs: 300,
            log_pairing_secret: false,
        }
    }
}

impl GatewayConfig {
    /// Validate the values a listening socket depends on. Called from
    /// [`LucyConfig::validate`] so a typo fails at load, not at bind time.
    pub fn validate(&self) -> Result<()> {
        if self.port == 0 {
            bail!("gateway.port must be non-zero");
        }
        let bind = self.bind.trim();
        if bind.is_empty() {
            bail!("gateway.bind must not be empty");
        }
        Ok(())
    }
}

/// The on-screen companion: a small overlay that can answer questions about
/// what is currently visible, using a caller-supplied screen summary.
///
/// Every field here is a *budget*, not a behaviour switch: `enabled` is the
/// master switch, `include_screen_context` decides whether a turn carries the
/// latest summary at all, and `max_screen_chars` caps how much of it one turn
/// may carry, so a verbose summary cannot crowd out the request itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CompanionConfig {
    pub enabled: bool,
    pub include_screen_context: bool,
    pub max_screen_chars: usize,
}

impl Default for CompanionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            include_screen_context: true,
            max_screen_chars: 4_000,
        }
    }
}

impl CompanionConfig {
    /// True when a turn may carry the latest screen summary.
    pub fn screens_allowed(&self) -> bool {
        self.enabled && self.include_screen_context
    }

    /// Truncate `summary` to [`Self::max_screen_chars`] on a char boundary.
    /// A zero budget keeps nothing; the request itself is never truncated.
    pub fn truncate_summary(&self, summary: &str) -> String {
        let budget = self.max_screen_chars;
        if summary.chars().count() <= budget {
            return summary.to_owned();
        }
        summary.chars().take(budget).collect()
    }
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            startup_screen: "mascot".into(),
            compact_after_command: true,
        }
    }
}
impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            default_text: DEFAULT_TEXT_MODEL.into(),
            base_url: None,
            api_key: None,
            text_api_key: None,
            text_base_url: Some(DEFAULT_TEXT_BASE_URL.into()),
        }
    }
}
impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            max_subtasks: 32,
            max_depth: 8,
            verify_state: true,
            parallel: true,
            replan_on_failure: true,
        }
    }
}
impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            provider: "groq".into(),
            model: "whisper-large-v3-turbo".into(),
            language: None,
            push_to_talk: "f2".into(),
            api_key: None,
        }
    }
}
impl Default for HyprFastConfig {
    fn default() -> Self {
        Self {
            command: "hyprfast".into(),
            args: vec!["mcp".into()],
            max_candidates: 8,
            batching: true,
            parallel: true,
            verify_actions: true,
        }
    }
}
impl Default for AppearanceConfig {
    fn default() -> Self {
        Self {
            theme: "lucy".into(),
            animations: true,
            activity_verbosity: "normal".into(),
        }
    }
}
impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            file: None,
            dir: None,
            resume: true,
            max_history: 100,
        }
    }
}
impl Default for ApprovalConfig {
    fn default() -> Self {
        Self {
            mode: "write".into(),
            always_allow: Vec::new(),
        }
    }
}
impl ApprovalConfig {
    /// The configured mode, falling back to `write` for anything unparseable
    /// so a hand-edited typo degrades to "ask before risky tools" rather than
    /// silently disabling every prompt.
    pub fn mode_or_default(&self) -> &str {
        let m = self.mode.trim();
        if APPROVAL_MODES.contains(&m) {
            m
        } else {
            "write"
        }
    }

    /// True when automode is on: Lucy runs without stopping to ask.
    pub fn automode(&self) -> bool {
        self.mode_or_default() == "never"
    }
}
impl Default for BrowserConfig {
    fn default() -> Self {
        Self {
            cdp_port: 9222,
            launch_timeout_secs: 8,
            // "Use graphics acceleration when available" is on by default in
            // Brave/Chrome. Off here, not as an opt-in: a GPU-composited page
            // reports rects through CDP that the compositor can disagree with,
            // and every hit-test guard in the browser loop is reasoning about
            // those rects.
            launch_args: vec!["--disable-gpu".into()],
            // Empty = hyprfast resolves it under the user's data directory,
            // where it survives a reboot. Set this only to move the profile.
            user_data_dir: String::new(),
        }
    }
}
impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            max_llm_calls_single_task: 4,
            max_recoveries: 2,
            max_step_retries: 1,
            agent_loop_enabled: true,
            // Topic vocabulary only — see the field doc. Every entry here is a
            // word that used to be hardcoded in `goal_needs_browser`.
            browser_goal_keywords: [
                "youtube",
                "google",
                "wikipedia",
                "reddit",
                "amazon",
                "netflix",
                "twitch",
                "github",
                "song",
                "music",
                "video",
                "playlist",
                "podcast",
                "stream",
                "flight",
                "booking",
                "hotel",
                "news",
                "article",
                "shop",
                "cart",
                "checkout",
                "login",
                "form",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
        }
    }
}
impl Default for SystemOneConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            provider: "decider".into(),
            base_url: "http://localhost:8001".into(),
            api_key: None,
            model: "Mapika/decider-2b-vision".into(),
            direct_python: None,
            confidence_threshold: 0.1,
            // decider-serve answers warm text predicts in ~300ms; the first
            // request after (re)start can take ~20s while weights load.
            timeout_ms: 30000,
            auto_start: false,
            server_command: None,
            server_model: None,
        }
    }
}
impl LucyConfig {
    pub fn path() -> Result<PathBuf> {
        if let Ok(p) = env::var("LUCY_CONFIG") {
            return Ok(PathBuf::from(p));
        }
        let home = env::var_os("HOME").context("HOME is not set")?;
        Ok(PathBuf::from(home).join(".config/lucy/config.toml"))
    }
    pub fn load() -> Result<Self> {
        let path = Self::path()?;
        let mut cfg = if path.exists() {
            let text =
                fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
            toml::from_str::<Self>(&text).with_context(|| format!("parsing {}", path.display()))?
        } else {
            Self::default()
        };
        cfg.apply_env()?;
        cfg.validate()?;
        Ok(cfg)
    }
    pub fn save(&self) -> Result<()> {
        self.validate()?;
        let path = Self::path()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, toml::to_string_pretty(self)?)?;
        Ok(())
    }
    pub fn init_if_missing(&self) -> Result<()> {
        if !Self::path()?.exists() {
            self.save()?;
        }
        Ok(())
    }
    pub fn get(&self, key: &str) -> Result<String> {
        let value = toml::Value::try_from(self)?;
        let mut cur = &value;
        for part in key.split('.') {
            cur = cur
                .get(part)
                .ok_or_else(|| anyhow::anyhow!("unknown config key: {key}"))?;
        }
        Ok(cur.to_string().trim_matches('"').to_string())
    }
    pub fn set(&mut self, key: &str, raw: &str) -> Result<()> {
        let mut value = toml::Value::try_from(&*self)?;
        let parts: Vec<_> = key.split('.').filter(|p| !p.is_empty()).collect();
        if parts.is_empty() {
            bail!("empty config key");
        }
        let mut cur = &mut value;
        for part in &parts[..parts.len() - 1] {
            cur = cur
                .get_mut(*part)
                .ok_or_else(|| anyhow::anyhow!("unknown config key: {key}"))?;
        }
        let last = parts[parts.len() - 1];
        // Handle optional fields that may be None and thus missing from Value — create as String if missing
        let new_val = if let Some(old) = cur.get(last).cloned() {
            parse_value(raw, &old)?
        } else {
            // Missing key (e.g. Option<String> that was None) — treat as string
            toml::Value::String(raw.to_owned())
        };
        if let Some(table) = cur.as_table_mut() {
            table.insert(last.to_string(), new_val);
        } else {
            bail!("config key parent is not a table: {key}");
        }
        *self = value.try_into()?;
        self.validate()
    }
    pub fn reset(&mut self) {
        *self = Self::default();
    }
    pub fn validate(&self) -> Result<()> {
        if self.planner.max_subtasks == 0 || self.planner.max_subtasks > 256 {
            bail!("planner.max_subtasks must be between 1 and 256");
        }
        if self.planner.max_depth == 0 || self.planner.max_depth > 64 {
            bail!("planner.max_depth must be between 1 and 64");
        }
        if self.hyprfast.max_candidates == 0 || self.hyprfast.max_candidates > 68 {
            bail!("hyprfast.max_candidates must be between 1 and 68");
        }
        if self.sessions.max_history == 0 {
            bail!("sessions.max_history must be greater than 0");
        }
        if self.browser.cdp_port == 0 {
            bail!("browser.cdp_port must be non-zero");
        }
        if self.harness.max_llm_calls_single_task == 0
            || self.harness.max_llm_calls_single_task > 32
        {
            bail!("harness.max_llm_calls_single_task must be between 1 and 32");
        }
        if self.system_one.confidence_threshold < 0.0 || self.system_one.confidence_threshold > 1.0
        {
            bail!("system_one.confidence_threshold must be between 0.0 and 1.0");
        }
        match self.approvals.mode.as_str() {
            "never" | "write" | "always" => {}
            _ => bail!("approvals.mode must be never|write|always"),
        }
        if self.classification.confidence_threshold < 0.0
            || self.classification.confidence_threshold > 1.0
        {
            bail!("classification.confidence_threshold must be between 0.0 and 1.0");
        }
        for p in &self.providers {
            if p.api_url.trim().is_empty() {
                bail!("provider '{}' has an empty API URL", p.label());
            }
        }
        // These are context budgets on a model that is already cheap and
        // already context-bound, so an unbounded one spends the whole window on
        // knowledge and leaves none for the task.
        if self.knowledge.digest_budget_chars > 8_000 {
            bail!("knowledge.digest_budget_chars must be at most 8000");
        }
        if self.knowledge.recall_budget_chars > 24_000 {
            bail!("knowledge.recall_budget_chars must be at most 24000");
        }
        if self.knowledge.max_captures_per_turn == 0 || self.knowledge.max_captures_per_turn > 16 {
            bail!("knowledge.max_captures_per_turn must be between 1 and 16");
        }
        self.gateway.validate()?;
        if self.companion.max_screen_chars > 32_000 {
            bail!("companion.max_screen_chars must be at most 32000");
        }
        Ok(())
    }
    fn apply_env(&mut self) -> Result<()> {
        if let Some(v) = env::var_os("OPENAI_MODEL") {
            self.models.default_text = v.to_string_lossy().into_owned();
        }
        // Legacy generic base_url
        if let Some(v) = env::var_os("OPENAI_BASE_URL") {
            self.models.base_url = Some(v.to_string_lossy().into_owned());
        }
        if self.models.base_url.is_none() {
            if let Some(v) = env::var_os("ANTHROPIC_BASE_URL")
                .or_else(|| env::var_os("LLM_BASE_URL"))
                .or_else(|| env::var_os("LUCY_BASE_URL"))
            {
                self.models.base_url = Some(v.to_string_lossy().into_owned());
            }
        }
        // Per-model base_url overrides
        if self.models.text_base_url.is_none() {
            if let Ok(v) = env::var("OPENCHAT_BASE_URL")
                .or_else(|_| env::var("LUCY_MAIN_BASE_URL"))
                .or_else(|_| env::var("MAIN_BASE_URL"))
                .or_else(|_| env::var("OPENROUTER_BASE_URL"))
            {
                if !v.trim().is_empty() {
                    self.models.text_base_url = Some(v);
                }
            }
        }
        // Generic LLM API key: accept any brand via endpoint + key. Priority: config file > env
        if self.models.api_key.is_none() {
            for key in [
                "LUCY_API_KEY",
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "GEMINI_API_KEY",
                "LLM_API_KEY",
                "MISTRAL_API_KEY",
                "OPENROUTER_API_KEY",
            ] {
                if let Ok(v) = env::var(key) {
                    if !v.trim().is_empty() {
                        self.models.api_key = Some(v);
                        break;
                    }
                }
            }
        }
        // Text-model API key.
        if self.models.text_api_key.is_none() {
            for key in [
                "LUCY_MAIN_API_KEY",
                "OPENCHAT_API_KEY",
                "OPENAI_API_KEY",
                "MAIN_API_KEY",
            ] {
                if let Ok(v) = env::var(key) {
                    if !v.trim().is_empty() {
                        self.models.text_api_key = Some(v);
                        break;
                    }
                }
            }
        }
        // Fallback: if per-model not set but generic is, clone generic
        if self.models.text_api_key.is_none() {
            if let Some(v) = self.models.api_key.clone() {
                self.models.text_api_key = Some(v);
            }
        }
        if self.models.text_base_url.is_none() {
            if let Some(v) = self.models.base_url.clone() {
                self.models.text_base_url = Some(v);
            }
        }

        if self.voice.api_key.is_none() {
            for key in ["GROQ_API_KEY", "LUCY_STT_API_KEY", "STT_API_KEY"] {
                if let Ok(v) = env::var(key) {
                    if !v.trim().is_empty() {
                        self.voice.api_key = Some(v);
                        break;
                    }
                }
            }
        }
        if let Some(v) = env::var_os("LUCY_STT_MODEL") {
            self.voice.model = v.to_string_lossy().into_owned();
        }
        if let Some(v) = env::var_os("LUCY_STT_LANGUAGE") {
            self.voice.language = Some(v.to_string_lossy().into_owned());
        }
        if let Some(v) = env::var_os("LUCY_HYPRFAST_MAX_CANDIDATES") {
            self.hyprfast.max_candidates = v.to_string_lossy().parse()?;
        }
        if let Ok(v) = env::var("LUCY_BROWSER_CDP_PORT") {
            if !v.trim().is_empty() {
                self.browser.cdp_port = v.parse()?;
            }
        }

        // Mobile gateway overrides. Env is a convenient way to flip the bridge
        // on for one run without editing the config file, which is what a
        // service unit or a scripted test wants.
        if let Ok(v) = env::var("LUCY_GATEWAY_ENABLED") {
            if let Ok(b) = v.trim().parse::<bool>() {
                self.gateway.enabled = b;
            }
        }
        if let Ok(v) = env::var("LUCY_GATEWAY_PORT") {
            if !v.trim().is_empty() {
                self.gateway.port = v.parse()?;
            }
        }
        if let Ok(v) = env::var("LUCY_GATEWAY_BIND") {
            if !v.trim().is_empty() {
                self.gateway.bind = v.trim().to_string();
            }
        }

        // System One / JEV / Laya environment overrides
        if let Ok(v) = env::var("LUCY_SYSTEMONE_ENABLED").or_else(|_| env::var("LUCY_JEV_ENABLED"))
        {
            if let Ok(b) = v.parse::<bool>() {
                self.system_one.enabled = b;
            }
        }
        if let Ok(v) = env::var("LUCY_SYSTEMONE_PROVIDER").or_else(|_| env::var("LAYA_PROVIDER")) {
            if !v.trim().is_empty() {
                self.system_one.provider = v.trim().to_lowercase();
            }
        }
        if let Ok(v) = env::var("LUCY_SYSTEMONE_BASE_URL").or_else(|_| env::var("LAYA_API_URL")) {
            if !v.trim().is_empty() {
                self.system_one.base_url = v.trim().to_string();
            }
        }
        if self.system_one.api_key.is_none() {
            for key in [
                "LUCY_SYSTEMONE_API_KEY",
                "TYPESAFE_API_KEY",
                "JEV_API_KEY",
                "LAYA_API_KEY",
            ] {
                if let Ok(v) = env::var(key) {
                    if !v.trim().is_empty() {
                        self.system_one.api_key = Some(v);
                        break;
                    }
                }
            }
        }
        if let Ok(v) = env::var("LUCY_SYSTEMONE_MODEL")
            .or_else(|_| env::var("LAYA_MODEL"))
            .or_else(|_| env::var("TYPESAFE_MODEL"))
        {
            if !v.trim().is_empty() {
                self.system_one.model = v.trim().to_string();
            }
        }
        if let Ok(v) = env::var("LUCY_SYSTEMONE_PYTHON") {
            if !v.trim().is_empty() {
                self.system_one.direct_python = Some(v.trim().to_string());
            }
        }

        // decider-serve classification endpoint (API URL only, no key).
        if let Ok(v) = env::var("LUCY_CLASSIFICATION_URL")
            .or_else(|_| env::var("LUCY_DECIDER_URL"))
            .or_else(|_| env::var("DECIDER_API_URL"))
        {
            if !v.trim().is_empty() {
                self.classification.classification_api_url =
                    v.trim().trim_end_matches('/').to_string();
            }
        }
        if let Ok(v) = env::var("LUCY_CLASSIFICATION_ENABLED") {
            if let Ok(b) = v.parse::<bool>() {
                self.classification.enabled = b;
            }
        }
        if let Ok(v) = env::var("LUCY_CHAT_MODE") {
            if let Some(mode) = ChatMode::parse(&v) {
                self.chat.chat_mode = mode;
            }
        }
        if let Ok(v) = env::var("LUCY_AUTO_COMPACT") {
            if let Ok(b) = v.parse::<bool>() {
                self.chat.auto_compact = b;
                self.general.compact_after_command = b;
            }
        }
        if let Ok(v) = env::var("LUCY_AGENT_LOOP") {
            if let Ok(b) = v.parse::<bool>() {
                self.harness.agent_loop_enabled = b;
            }
        }
        // Migrate a pre-tier config file: `chat.main_model` was the single
        // default model back when Lucy had no reasoning tiers. Level 3 is its
        // successor (it plans actions and anchors `Manual` mode), so seed L3
        // rather than leaving the user with no model at all. An explicit L3
        // binding always wins, and the legacy field is dropped so the value is
        // never written back under the old name.
        let legacy = self.chat.legacy_main_model.trim().to_owned();
        if !legacy.is_empty() {
            if self.chat.reasoning_levels.level3.trim().is_empty() {
                self.chat.reasoning_levels.level3 = legacy;
            }
            self.chat.legacy_main_model.clear();
        }
        Ok(())
    }
    /// Returns the effective LLM API key from config or env (any brand) — generic fallback
    pub fn llm_api_key(&self) -> Option<String> {
        self.text_api_key().or_else(|| {
            self.models
                .api_key
                .clone()
                .filter(|v| !v.trim().is_empty())
                .or_else(|| {
                    for key in [
                        "LUCY_API_KEY",
                        "OPENAI_API_KEY",
                        "ANTHROPIC_API_KEY",
                        "GEMINI_API_KEY",
                        "LLM_API_KEY",
                        "MISTRAL_API_KEY",
                        "OPENROUTER_API_KEY",
                    ] {
                        if let Ok(v) = env::var(key) {
                            if !v.trim().is_empty() {
                                return Some(v);
                            }
                        }
                    }
                    None
                })
        })
    }
    pub fn llm_base_url(&self) -> Option<String> {
        self.models.base_url.clone().or_else(|| {
            env::var("OPENAI_BASE_URL")
                .ok()
                .filter(|v| !v.is_empty())
                .or_else(|| env::var("LLM_BASE_URL").ok())
                .or_else(|| env::var("LUCY_BASE_URL").ok())
        })
    }
    /// Credential for the text endpoint the reasoning tiers fall back to.
    pub fn text_api_key(&self) -> Option<String> {
        self.models
            .text_api_key
            .clone()
            .filter(|v| !v.trim().is_empty())
            .or_else(|| {
                for key in [
                    "LUCY_MAIN_API_KEY",
                    "OPENCHAT_API_KEY",
                    "OPENAI_API_KEY",
                    "OPENROUTER_API_KEY",
                ] {
                    if let Ok(v) = env::var(key) {
                        if !v.trim().is_empty() {
                            return Some(v);
                        }
                    }
                }
                self.models.api_key.clone().filter(|v| !v.trim().is_empty())
            })
            // Last resort: the local OpenChat proxy's own credential file.
            // The proxy's 401 names this exact path, so reading it makes a
            // default checkout work with zero setup. File is mode 600.
            .or_else(read_openchat_key_file)
    }
    pub fn text_base_url(&self) -> Option<String> {
        self.models
            .text_base_url
            .clone()
            .filter(|v| !v.trim().is_empty())
            .or_else(|| {
                for key in [
                    "OPENCHAT_BASE_URL",
                    "LUCY_MAIN_BASE_URL",
                    "MAIN_BASE_URL",
                    "OPENROUTER_BASE_URL",
                ] {
                    if let Ok(v) = env::var(key) {
                        if !v.trim().is_empty() {
                            return Some(v);
                        }
                    }
                }
                self.models
                    .base_url
                    .clone()
                    .filter(|v| !v.trim().is_empty())
                    .or_else(|| self.llm_base_url())
            })
    }
    pub fn stt_api_key(&self) -> Option<String> {
        self.voice
            .api_key
            .clone()
            .filter(|v| !v.trim().is_empty())
            .or_else(|| {
                for key in ["GROQ_API_KEY", "LUCY_STT_API_KEY", "STT_API_KEY"] {
                    if let Ok(v) = env::var(key) {
                        if !v.trim().is_empty() {
                            return Some(v);
                        }
                    }
                }
                None
            })
    }
    pub fn system_one_api_key(&self) -> Option<String> {
        self.system_one
            .api_key
            .clone()
            .filter(|v| !v.trim().is_empty())
            .or_else(|| {
                for key in [
                    "LUCY_SYSTEMONE_API_KEY",
                    "TYPESAFE_API_KEY",
                    "JEV_API_KEY",
                    "LAYA_API_KEY",
                ] {
                    if let Ok(v) = env::var(key) {
                        if !v.trim().is_empty() {
                            return Some(v);
                        }
                    }
                }
                None
            })
    }
    pub fn resolve_system_one_python(&self) -> String {
        if let Some(p) = self
            .system_one
            .direct_python
            .clone()
            .filter(|v| !v.trim().is_empty())
        {
            return p;
        }
        if let Ok(p) = env::var("LUCY_SYSTEMONE_PYTHON") {
            if !p.trim().is_empty() {
                return p;
            }
        }
        let mise_p = "/home/krish/.local/share/mise/installs/python/3.14.7/bin/python3";
        if std::path::Path::new(mise_p).exists() {
            return mise_p.to_string();
        }
        "python3".to_string()
    }
    pub fn cdp_port(&self) -> u16 {
        self.browser.cdp_port
    }

    // ---- providers, models, chat mode -------------------------------

    /// The compiled-in default text model. This is the floor of the tier
    /// fallback chain, not a fifth selectable model: `/settings` only ever
    /// offers L1, L2, L3 and the voice model.
    pub fn default_text_model(&self) -> String {
        let sel = self.models.default_text.trim();
        if !sel.is_empty() {
            return sel.to_owned();
        }
        // A live model from the endpoint Lucy is actually configured to call
        // beats the compiled-in name: when the server renames its models, the
        // last-probed list is the only source of truth that cannot go stale.
        if let Some(p) = self.endpoint_provider() {
            if let Some(model) = p.live_models().next() {
                return model.clone();
            }
        }
        DEFAULT_TEXT_MODEL.to_owned()
    }

    /// The connected text provider serving the legacy `[models] text_base_url`
    /// endpoint, matched on the normalized base URL (`/v1` present or not).
    pub fn endpoint_provider(&self) -> Option<&ProviderConfig> {
        let base = self.text_base_url()?;
        let base = base.trim().trim_end_matches('/');
        if base.is_empty() {
            return None;
        }
        let want = if is_versioned_base(base) {
            base.to_owned()
        } else {
            format!("{base}/v1")
        };
        self.providers.iter().find(|p| {
            p.provider_type == ProviderType::Text
                && p.openai_base_url()
                    .trim_end_matches('/')
                    .eq_ignore_ascii_case(&want)
        })
    }

    /// Persist the Level 3 binding. L3 is the anchor model — it plans actions
    /// and `Manual` chat mode pins every turn to it — so this is what the
    /// `Chat mode` section of `/settings` and the `Manual` shortcut write.
    pub fn set_anchor_model(&mut self, model: &str) {
        let model = model.trim().to_owned();
        self.chat.reasoning_levels.set(ReasoningLevel::L3, model);
    }

    /// The model used for the selected voice model, or the legacy
    /// `voice.model` when the user has not connected a voice provider yet.
    pub fn voice_model(&self) -> String {
        let sel = self.chat.voice_model.trim();
        if sel.is_empty() {
            self.voice.model.clone()
        } else {
            sel.to_owned()
        }
    }

    pub fn set_voice_model(&mut self, model: &str) {
        let model = model.trim().to_owned();
        self.chat.voice_model = model.clone();
        if !model.is_empty() {
            self.voice.model = model;
        }
    }

    pub fn chat_mode(&self) -> ChatMode {
        self.chat.chat_mode
    }

    pub fn set_chat_mode(&mut self, mode: ChatMode) {
        self.chat.chat_mode = mode;
    }

    /// The active approval mode, defaulting to `write` when unset/invalid.
    pub fn approval_mode(&self) -> &str {
        self.approvals.mode_or_default()
    }

    /// True when automode is on (Lucy never stops to ask).
    pub fn automode(&self) -> bool {
        self.approvals.automode()
    }

    /// Set the approval mode. Panics on an unknown value: every caller passes a
    /// compile-time `APPROVAL_MODES` entry, and silently storing a typo would
    /// make `validate()` fail on the next load with a confusing message.
    pub fn set_approval_mode(&mut self, mode: &str) {
        assert!(
            APPROVAL_MODES.contains(&mode),
            "unknown approval mode {mode:?}"
        );
        self.approvals.mode = mode.to_owned();
    }

    /// Tools the user has permanently allowed.
    pub fn always_allow(&self) -> &[String] {
        &self.approvals.always_allow
    }

    /// Persist the always-allow set, sorted and de-duplicated so the config
    /// file stays stable across writes.
    pub fn set_always_allow(&mut self, tools: Vec<String>) {
        let mut v: Vec<String> = tools.into_iter().filter(|t| !t.trim().is_empty()).collect();
        v.sort();
        v.dedup();
        self.approvals.always_allow = v;
    }

    /// Record one tool as permanently allowed.
    pub fn allow_tool_forever(&mut self, tool: &str) {
        let mut v = self.approvals.always_allow.clone();
        v.push(tool.to_owned());
        self.set_always_allow(v);
    }

    /// Human-readable mode summary for the TUI and `/auto status`.
    pub fn approval_mode_label(&self) -> &'static str {
        match self.approval_mode() {
            "never" => "auto — never ask",
            "always" => "always ask",
            _ => "ask before risky tools",
        }
    }

    pub fn auto_compact(&self) -> bool {
        self.chat.auto_compact
    }

    pub fn set_auto_compact(&mut self, on: bool) {
        self.chat.auto_compact = on;
        self.general.compact_after_command = on;
    }

    /// Whether the compatibility flag [`Self::agent_loop_enabled`] is set. It no
    /// longer changes which loop runs; see that field.
    pub fn agent_loop_enabled(&self) -> bool {
        self.harness.agent_loop_enabled
    }

    /// True when the phone bridge is switched on.
    pub fn gateway_enabled(&self) -> bool {
        self.gateway.enabled
    }

    /// Flip the bridge and persist it. The TUI `/serve on|off` command calls
    /// this; `lucy serve --enable/--disable` writes it without starting a
    /// server.
    pub fn set_gateway_enabled(&mut self, on: bool) {
        self.gateway.enabled = on;
    }

    /// Where the knowledge base lives: the configured directory, or
    /// `~/.config/lucy/knowledge`. Plain Markdown files, so this is a path a
    /// user can open in an editor rather than a database to inspect with a tool.
    pub fn knowledge_dir(&self) -> PathBuf {
        self.knowledge.dir.clone().unwrap_or_else(|| {
            PathBuf::from(env::var("HOME").unwrap_or_else(|_| ".".into()))
                .join(".config/lucy/knowledge")
        })
    }

    pub fn knowledge_enabled(&self) -> bool {
        self.knowledge.enabled
    }

    pub fn classification_api_url(&self) -> &str {
        self.classification.base_url()
    }

    /// Model bound to `level`, or an empty string when unset.
    pub fn level_model(&self, level: ReasoningLevel) -> String {
        self.chat.reasoning_levels.get(level).trim().to_owned()
    }

    /// The model a turn should actually call for `level`.
    ///
    /// A bound tier wins. An unbound one degrades to the next cheaper tier (see
    /// [`ReasoningLevel::fallback_chain`]), and only when the whole chain is
    /// unbound does it fall back to the compiled-in default text model. So
    /// there is never a call with no model: the tiers are self-contained.
    pub fn resolve_level_model(&self, level: ReasoningLevel) -> String {
        for candidate in level.fallback_chain() {
            let bound = self.level_model(candidate);
            if !bound.is_empty() {
                return bound;
            }
        }
        self.default_text_model()
    }

    /// True when `level` has no model of its own and [`Self::resolve_level_model`]
    /// will hand the turn to a different tier.
    pub fn level_is_unbound(&self, level: ReasoningLevel) -> bool {
        self.level_model(level).is_empty()
    }

    /// The chat-log note explaining which model an unbound `level` resolved to,
    /// or `None` when the tier is bound. The router appends it to the turn's
    /// routing line so a half-configured setup is visible rather than silent.
    ///
    /// The three outcomes are worded apart on purpose. An unbound tier whose
    /// chain contains a bound tier names that tier. An unbound L1 has no
    /// cheaper tier to skip to, so it lands on the compiled-in default — which
    /// must be reported as *that* rather than as "no reasoning tier bound",
    /// since L2 and L3 are perfectly well bound and the user needs to know the
    /// turn left their configured providers entirely.
    pub fn level_fallback_note(&self, level: ReasoningLevel) -> Option<String> {
        if !self.level_is_unbound(level) {
            return None;
        }
        let default_model = self.default_text_model();
        if let Some(landed) = level
            .fallback_chain()
            .into_iter()
            .find(|c| !self.level_model(*c).is_empty())
        {
            return Some(format!(
                "no model bound to {} — using {}",
                level.short(),
                landed.short()
            ));
        }
        let any_bound = ReasoningLevel::ALL
            .iter()
            .any(|l| !self.level_model(*l).is_empty());
        Some(if any_bound {
            format!(
                "no model bound to {} and no cheaper tier is bound — using the default {default_model} model",
                level.short(),
            )
        } else {
            format!("no reasoning tier bound — using the default {default_model} model")
        })
    }

    pub fn provider(&self, id: &str) -> Option<&ProviderConfig> {
        self.providers
            .iter()
            .find(|p| p.id.eq_ignore_ascii_case(id.trim()))
    }

    pub fn providers_of(&self, kind: ProviderType) -> Vec<&ProviderConfig> {
        self.providers
            .iter()
            .filter(|p| p.provider_type == kind)
            .collect()
    }

    /// Every model exposed by the connected providers of `kind`, as
    /// `provider_id/model` options for the settings dropdowns.
    pub fn model_options(&self, kind: ProviderType) -> Vec<ModelOption> {
        let mut out: Vec<ModelOption> = Vec::new();
        for p in self.providers_of(kind) {
            if p.id.trim().is_empty() {
                continue;
            }
            for m in &p.available_models {
                let model = m.trim();
                if model.is_empty() {
                    continue;
                }
                let opt = ModelOption {
                    provider_id: p.id.clone(),
                    provider_name: p.label(),
                    model: model.to_owned(),
                };
                if !out.contains(&opt) {
                    out.push(opt);
                }
            }
        }
        // Stable dropdown order regardless of discovery order.
        out.sort_by(|a, b| {
            a.provider_name
                .cmp(&b.provider_name)
                .then_with(|| a.model.cmp(&b.model))
        });
        out
    }

    pub fn text_model_options(&self) -> Vec<ModelOption> {
        self.model_options(ProviderType::Text)
    }

    pub fn voice_model_options(&self) -> Vec<ModelOption> {
        self.model_options(ProviderType::Voice)
    }

    /// Split a stored `provider_id/model` key back into its parts. Falls back
    /// to the provider whose discovered list contains the bare model name,
    /// then to the legacy endpoint's provider, when only the bare model name
    /// was stored (a leftover in `models.default_text`, or an older config).
    pub fn split_model_key(&self, key: &str) -> Option<(ProviderConfig, String)> {
        let key = key.trim();
        if key.is_empty() {
            return None;
        }
        if let Some((id, model)) = key
            .split_once('/')
            .filter(|(i, m)| !i.is_empty() && !m.is_empty() && self.provider(i).is_some())
        {
            let p = self.provider(id)?.clone();
            return Some((p, model.to_owned()));
        }
        let text_providers = self.providers_of(ProviderType::Text);
        // A bare name belongs to the provider whose discovered list actually
        // contains it — never merely to whichever provider sorts first.
        let listed = text_providers.iter().copied().find(|p| {
            p.available_models
                .iter()
                .any(|m| m.trim().eq_ignore_ascii_case(key))
        });
        let Some(p) = listed
            .or(self.endpoint_provider())
            .or(text_providers.first().copied())
        else {
            return None;
        };
        Some((p.clone(), key.to_owned()))
    }

    /// True when `key` is a `provider_id/model` reference to a *connected*
    /// provider, so [`Self::resolve_endpoint`] would route it there rather than
    /// to the legacy `text_base_url` endpoint.
    ///
    /// A bare model name returns false on purpose: `resolve_endpoint` falls back
    /// to `text_base_url` for those, which is what a keyless local server
    /// (llama.cpp, the OpenChat proxy) depends on.
    pub fn provider_owns_model_key(&self, key: &str) -> bool {
        let key = key.trim();
        let Some((id, model)) = key.split_once('/') else {
            return false;
        };
        if id.is_empty() || model.is_empty() {
            return false;
        }
        self.provider(id)
            .is_some_and(|p| p.provider_type == ProviderType::Text)
    }

    /// Endpoint credentials for a `provider_id/model` key. Falls back to the
    /// legacy `text_base_url`/`text_api_key` pair so a config with providers
    /// but no tier binding still routes somewhere.
    pub fn resolve_endpoint(&self, key: &str) -> Option<(String, Option<String>, String)> {
        if let Some((provider, model)) = self.split_model_key(key) {
            return Some((
                provider.openai_base_url(),
                Some(provider.api_key).filter(|k| !k.trim().is_empty()),
                model,
            ));
        }
        let base = self.text_base_url()?;
        Some((base, self.text_api_key(), self.default_text_model()))
    }

    /// Provider API key for the `voice` model, falling back to `stt_api_key`.
    pub fn voice_api_key_for(&self, key: &str) -> Option<String> {
        if let Some((provider, _)) = self.split_model_key(key)
            && !provider.api_key.trim().is_empty()
        {
            return Some(provider.api_key);
        }
        self.stt_api_key()
    }

    /// Insert or replace a provider by `id` (assigning an id when blank) and
    /// persist. Returns the stored provider, including the model list the
    /// caller discovered via `Test Connection`.
    pub fn save_provider(&mut self, mut provider: ProviderConfig) -> Result<ProviderConfig> {
        provider.id = provider.id.trim().to_owned();
        provider.name = provider.name.trim().to_owned();
        provider.api_url = provider.api_url.trim().trim_end_matches('/').to_owned();
        provider.api_key = provider.api_key.trim().to_owned();
        if provider.api_url.is_empty() {
            bail!("provider API URL must not be empty");
        }
        provider.available_models.retain(|m| !m.trim().is_empty());
        provider.available_models.sort();
        provider.available_models.dedup();
        if provider.id.is_empty() {
            // Uniquify against the in-memory provider list, never the file on
            // disk: a draft being edited must not depend on what is persisted.
            let taken: Vec<String> = self.providers.iter().map(|p| p.id.clone()).collect();
            provider.ensure_id(&taken);
        }
        match self.providers.iter().position(|p| p.id == provider.id) {
            Some(idx) => self.providers[idx] = provider.clone(),
            None => self.providers.push(provider.clone()),
        }
        Ok(provider)
    }

    /// Persist a `decider-serve` base URL (API URL only — never a key).
    pub fn save_classification_url(&mut self, url: &str) -> Result<String> {
        let url = url.trim().trim_end_matches('/').to_owned();
        if url.is_empty() {
            bail!("classification API URL must not be empty");
        }
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            bail!("classification API URL must start with http:// or https://");
        }
        self.classification.classification_api_url = url.clone();
        Ok(url)
    }

    /// Drop a provider and any model selection that referenced it, so no
    /// dropdown can point at a provider that no longer exists.
    pub fn remove_provider(&mut self, id: &str) -> bool {
        let before = self.providers.len();
        self.providers
            .retain(|p| !p.id.eq_ignore_ascii_case(id.trim()));
        if self.providers.len() == before {
            return false;
        }
        let stale = |key: &str| -> bool {
            !key.trim().is_empty()
                && key
                    .split_once('/')
                    .is_some_and(|(p, _)| p.eq_ignore_ascii_case(id))
        };
        if stale(&self.chat.voice_model) {
            self.chat.voice_model = String::new();
            self.voice.model = String::new();
        }
        for level in ReasoningLevel::ALL {
            let key = self.chat.reasoning_levels.get(level).to_owned();
            if stale(&key) {
                self.chat.reasoning_levels.set(level, String::new());
            }
        }
        true
    }
}
/// Read the local OpenChat proxy credential (`~/.openchat/api_key`).
/// Returns `None` when the file is missing, unreadable, or blank — never fails.
fn read_openchat_key_file() -> Option<String> {
    let home = env::var_os("HOME")?;
    let path = Path::new(&home).join(".openchat/api_key");
    let raw = fs::read_to_string(&path).ok()?;
    let key = raw.trim().to_owned();
    if key.is_empty() { None } else { Some(key) }
}
fn parse_value(raw: &str, old: &toml::Value) -> Result<toml::Value> {
    if matches!(old, toml::Value::String(_)) {
        return Ok(toml::Value::String(raw.to_owned()));
    }
    raw.parse::<toml::Value>()
        .map_err(|e| anyhow::anyhow!("invalid value: {e}"))
}
pub fn doctor() -> Vec<(&'static str, bool, String)> {
    let config_status = match LucyConfig::load() {
        Ok(c) => {
            let hypr_ok = command_exists(&c.hyprfast.command);
            let main_key = c.text_api_key();
            let main_ok = main_key.is_some();
            let stt_key = c.stt_api_key();
            let stt_ok = stt_key.is_some();
            let main_endpoint = c
                .text_base_url()
                .unwrap_or_else(|| DEFAULT_TEXT_BASE_URL.into());
            let mask = |k: String| {
                if k.len() > 8 {
                    format!("{}...{} ({} chars)", &k[..4], &k[k.len() - 4..], k.len())
                } else {
                    "***".into()
                }
            };
            let reachability = |endpoint: &str| -> String {
                let trunc = |e: String| {
                    let t = e.chars().take(80).collect::<String>();
                    format!(" — unreachable ({t})")
                };
                let Some((scheme, rest)) = endpoint.split_once("://") else {
                    return trunc("invalid url".into());
                };
                let default_port = match scheme {
                    "https" => 443,
                    "http" => 80,
                    _ => return trunc("unsupported scheme".into()),
                };
                let hostport = rest.split('/').next().unwrap_or("");
                let hostport = hostport.rsplit('@').next().unwrap_or(hostport);
                let (host, port) = if let Some(br) = hostport.strip_prefix('[') {
                    match br.split_once(']') {
                        Some((h, r)) => {
                            let p = r
                                .strip_prefix(':')
                                .unwrap_or("")
                                .parse::<u16>()
                                .unwrap_or(default_port);
                            (
                                h.to_string(),
                                if r.is_empty()
                                    || r.starts_with(':') && r[1..].parse::<u16>().is_ok()
                                {
                                    p
                                } else {
                                    default_port
                                },
                            )
                        }
                        None => return trunc("invalid host".into()),
                    }
                } else if hostport.matches(':').count() > 1 {
                    (hostport.to_string(), default_port)
                } else {
                    match hostport.rsplit_once(':') {
                        Some((h, p)) if !h.is_empty() && !p.is_empty() => match p.parse::<u16>() {
                            Ok(n) => (h.to_string(), n),
                            Err(_) => (hostport.to_string(), default_port),
                        },
                        _ => (hostport.to_string(), default_port),
                    }
                };
                if host.is_empty() {
                    return trunc("invalid host".into());
                }
                use std::net::{TcpStream, ToSocketAddrs};
                use std::time::Duration;
                let addrs: Vec<_> = match format!("{host}:{port}").to_socket_addrs() {
                    Ok(a) => a.collect(),
                    Err(e) => return trunc(e.to_string()),
                };
                if addrs.is_empty() {
                    return trunc("no addresses".into());
                }
                let mut last_err = "connection failed".to_string();
                for a in addrs {
                    match TcpStream::connect_timeout(&a, Duration::from_secs(3)) {
                        Ok(s) => {
                            drop(s);
                            return " — reachable".into();
                        }
                        Err(e) => {
                            last_err = e.to_string();
                        }
                    }
                }
                trunc(last_err)
            };
            let main_reach = reachability(&main_endpoint);
            let main_detail = if main_ok {
                format!(
                    "set {} — endpoint {}",
                    mask(main_key.unwrap()),
                    main_endpoint
                )
            } else {
                "missing — Settings > Connect providers, or env OPENCHAT_API_KEY / LUCY_MAIN_API_KEY".into()
            };
            let text_providers = c.providers_of(ProviderType::Text);
            let voice_providers = c.providers_of(ProviderType::Voice);
            let providers_detail = if c.providers.is_empty() {
                "none — add one in /settings > Connect providers".to_string()
            } else {
                let mut parts: Vec<String> = c
                    .providers
                    .iter()
                    .map(|p| {
                        format!(
                            "{} [{}] {} models",
                            p.label(),
                            p.provider_type.as_str().to_ascii_lowercase(),
                            p.available_models.len()
                        )
                    })
                    .collect();
                parts.retain(|s| !s.ends_with(" 0 models") || s.contains("none"));
                parts.join(", ")
            };
            let classification_endpoint = c.classification_api_url().to_string();
            let classification_reach = reachability(&classification_endpoint);
            let classification_detail = if !c.classification.enabled {
                "disabled — reasoning tiers / heuristic routing decide".to_string()
            } else {
                format!("decider-serve @ {classification_endpoint}{classification_reach}")
            };
            let system_one_detail = if !c.system_one.enabled {
                "disabled".to_string()
            } else {
                format!(
                    "{} ({}) @ {}{} — start decider-serve first (`python app.py --port 8001` in ~/Projects/decider-serve); lucy never starts it",
                    c.system_one.provider,
                    c.system_one.model,
                    c.system_one.base_url,
                    reachability(&c.system_one.base_url)
                )
            };
            vec![
                (
                    "Config",
                    true,
                    LucyConfig::path()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default(),
                ),
                (
                    "Default text model",
                    true,
                    format!(
                        "{} (used only when no reasoning tier is bound)",
                        c.default_text_model()
                    ),
                ),
                (
                    "Connected providers",
                    !c.providers.is_empty(),
                    format!(
                        "{providers_detail} — {} text / {} voice",
                        text_providers.len(),
                        voice_providers.len()
                    ),
                ),
                (
                    "Classification model",
                    c.classification.enabled,
                    classification_detail,
                ),
                (
                    "Chat mode",
                    true,
                    format!(
                        "{} · L1 {} · L2 {} · L3 {} · voice {} · auto-compact {}",
                        c.chat_mode().as_str(),
                        or_unset(&c.level_model(ReasoningLevel::L1)),
                        or_unset(&c.level_model(ReasoningLevel::L2)),
                        or_unset(&c.level_model(ReasoningLevel::L3)),
                        or_unset(&c.voice_model()),
                        if c.auto_compact() { "on" } else { "off" }
                    ),
                ),
                (
                    "Text Endpoint",
                    true,
                    format!("{main_endpoint}{main_reach}"),
                ),
                ("Text API Key", main_ok, main_detail),
                ("System One / Jev", c.system_one.enabled, system_one_detail),
                (
                    "HyprFast",
                    hypr_ok,
                    if hypr_ok {
                        c.hyprfast.command.clone()
                    } else {
                        format!("{} (not found - install hyprfast)", c.hyprfast.command)
                    },
                ),
                {
                    let (ok, detail) = browser_doctor_detail(&c);
                    ("CDP Browser", ok, detail)
                },
                (
                    "STT API Key",
                    stt_ok,
                    if stt_ok {
                        "set (voice enabled)".into()
                    } else {
                        "not set — voice disabled (set Voice API Key or GROQ_API_KEY)".into()
                    },
                ),
            ]
        }
        Err(e) => vec![("Config", false, e.to_string())],
    };
    config_status
}
/// State of the CDP browser, for the `doctor` row.
///
/// Three distinct outcomes, because they need three different fixes and the
/// symptom is otherwise identical every time — a browser task that "does
/// nothing" with no error anywhere:
///
/// - **reachable** — a debugging endpoint answered. The profile may still be
///   empty (first run); that is not a fault.
/// - **no browser running** — nothing to attach to. Lucy launches one itself
///   on the first browser task, so this is normal, not a warning.
/// - **a browser is running but not debuggable** — the real problem. A
///   browser on the default profile cannot have been started with
///   `--remote-debugging-port` (Chromium is single-instance per profile, so a
///   launch is forwarded to the running window and ignored), so the port will
///   never open and every CDP call times out against it. This row exists to
///   name that, because nothing else in the run does.
///
/// No `ok` verdict is claimed for the second case: it is reported as
/// unhealthy with the fix, since a user reading `doctor` has a browser open
/// and is asking why lucy cannot drive it.
fn browser_doctor_detail(config: &LucyConfig) -> (bool, String) {
    let port = config.browser.cdp_port;
    let profile = config
        .browser
        .user_data_dir_value()
        .unwrap_or_else(|| default_profile_dir().display().to_string());
    let ready = cdp_endpoint_answers(port);
    let running = browser_process_without_debug_port();
    match (ready, running) {
        (true, _) => (
            true,
            format!("CDP ready on 127.0.0.1:{port} · profile {profile}"),
        ),
        // Nothing running is the pre-first-task state, and hyprfast launches
        // on demand — reporting this as broken would cry wolf on a fresh
        // install and on every machine between browser tasks.
        (false, false) => (
            true,
            format!(
                "not running — hyprfast launches on 127.0.0.1:{port} on first use · profile {profile}"
            ),
        ),
        (false, true) => (
            false,
            format!(
                "a browser is running WITHOUT --remote-debugging-port={port}, so lucy cannot attach to it. \
                 Either close it and let lucy launch its own, or start yours with the flag and a dedicated profile: \
                 brave --remote-debugging-port={port} --user-data-dir={profile}"
            ),
        ),
    }
}

/// The profile directory lucy uses when `[browser] user_data_dir` is unset.
///
/// Mirrors `lucy-systemone`'s resolver and hyprfast's, because all three
/// have to name the same directory: lucy writes a preference into it before
/// launch, and hyprfast decides it at launch.
fn default_profile_dir() -> PathBuf {
    let base = match env::var("XDG_DATA_HOME") {
        Ok(dir) if !dir.trim().is_empty() => PathBuf::from(dir.trim()),
        _ => match env::var("HOME") {
            Ok(home) if !home.is_empty() => PathBuf::from(home).join(".local/share"),
            _ => PathBuf::from(".local/share"),
        },
    };
    base.join("hyprfast").join("browser-profile")
}

/// True when something answers `GET /json/version` on the CDP port.
///
/// Bounded by construction: this runs inside `doctor`, which must not hang
/// because a half-open port never answers. `HTTP/1.1` for the same reason
/// `fast_perception` uses it — Chromium's DevTools HTTP server closes an
/// HTTP/1.0 connection without replying.
fn cdp_endpoint_answers(port: u16) -> bool {
    use std::io::{Read, Write};
    use std::net::{TcpStream, ToSocketAddrs};
    use std::time::Duration;
    let addr = format!("127.0.0.1:{port}")
        .to_socket_addrs()
        .ok()
        .and_then(|mut it| it.next());
    let Some(addr) = addr else { return false };
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_millis(400)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(600)));
    let request = format!(
        "GET /json/version HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUser-Agent: lucy-doctor\r\nConnection: close\r\n\r\n"
    );
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut buf = [0u8; 64];
    matches!(stream.read(&mut buf), Ok(n) if n > 0)
}

/// True when a Chromium-family browser process is alive with no
/// `--remote-debugging-port` in its command line.
///
/// The command line, not just the name: a browser Lucy (or the user) started
/// *with* the flag is exactly what we want, and it also matches the name, so
/// name-only matching would report the healthy case as the broken one.
fn browser_process_without_debug_port() -> bool {
    let out = std::process::Command::new("sh")
        .args([
            "-c",
            // `ps` rather than `pgrep -f` on the full argv: pgrep's pattern
            // match runs over the whole line and its own `-f` handling differs
            // between procps versions. First field is the pid, so a browser's
            // renderer/zygote children — which never carry the flag and would
            // otherwise match — are excluded by requiring an executable-looking
            // first token.
            "ps -eo args= 2>/dev/null | grep -E '(^|/)(brave|brave-browser|chromium|chromium-browser|google-chrome|chrome)( |$)' | grep -v -- '--remote-debugging-port' || true",
        ])
        .output();
    match out {
        Ok(o) => !String::from_utf8_lossy(&o.stdout).trim().is_empty(),
        Err(_) => false,
    }
}

/// Placeholder for an unset model selection in `doctor` output.
fn or_unset(model: &str) -> String {
    let m = model.trim();
    if m.is_empty() {
        "(unset)".to_string()
    } else {
        m.to_string()
    }
}

fn command_exists(command: &str) -> bool {
    if command.is_empty() {
        return false;
    }
    std::process::Command::new("sh")
        .args([
            "-c",
            "command -v -- \"$1\" >/dev/null 2>&1",
            "lucy",
            command,
        ])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}
pub fn config_path() -> Result<PathBuf> {
    LucyConfig::path()
}
pub fn config_exists() -> Result<bool> {
    Ok(Path::new(&LucyConfig::path()?).exists())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_provider(name: &str, models: &[&str]) -> ProviderConfig {
        ProviderConfig {
            id: name.to_ascii_lowercase(),
            name: name.into(),
            api_url: format!("https://api.{name}.test"),
            api_key: "sk-test".into(),
            provider_type: ProviderType::Text,
            available_models: models.iter().map(|m| (*m).to_owned()).collect(),
            deprecated_models: Vec::new(),
        }
    }

    #[test]
    fn openrouter_preset_points_at_the_public_openai_compatible_url() {
        let p = ProviderPreset::OpenRouter;
        assert_eq!(p.api_url(), Some("https://openrouter.ai/api/v1"));
        assert_eq!(p.api_key_env(), Some("OPENROUTER_API_KEY"));
        // The version segment is already there, so /models is not doubled up.
        let cfg = ProviderConfig {
            api_url: p.api_url().unwrap().into(),
            ..Default::default()
        };
        assert_eq!(cfg.models_endpoint(), "https://openrouter.ai/api/v1/models");
    }

    #[test]
    fn presets_parse_by_name_and_by_url() {
        assert_eq!(
            ProviderPreset::parse("openrouter"),
            Some(ProviderPreset::OpenRouter)
        );
        assert_eq!(
            ProviderPreset::parse("OpenRouter"),
            Some(ProviderPreset::OpenRouter)
        );
        assert_eq!(
            ProviderPreset::parse("https://openrouter.ai/api/v1/"),
            Some(ProviderPreset::OpenRouter)
        );
        assert_eq!(ProviderPreset::parse("nope"), None);
        assert_eq!(ProviderPreset::parse(""), None);
        // A look-alike host is not OpenRouter.
        assert_eq!(
            ProviderPreset::parse("https://openrouter.ai.evil.test/v1"),
            None
        );
        assert_eq!(
            ProviderPreset::for_url("https://openrouter.ai/api/v1"),
            Some(ProviderPreset::OpenRouter)
        );
        assert_eq!(
            ProviderPreset::for_url("https://api.groq.com/openai/v1"),
            Some(ProviderPreset::Groq)
        );
        // `Custom` never claims a URL.
        assert_eq!(ProviderPreset::Custom.api_url(), None);
    }

    #[test]
    fn openrouter_endpoints_are_recognised_by_host() {
        assert!(is_openrouter("https://openrouter.ai/api/v1"));
        assert!(is_openrouter(
            "https://openrouter.ai/api/v1/chat/completions"
        ));
        assert!(is_openrouter("https://eu.openrouter.ai/api/v1"));
        assert!(!is_openrouter("https://api.groq.com/openai/v1"));
        assert!(!is_openrouter("http://127.0.0.1:11435/v1"));
        assert!(!is_openrouter("https://openrouter.ai.evil.test/v1"));
    }

    #[test]
    fn bare_model_names_are_corrected_for_openrouter_only() {
        // OpenRouter only knows `vendor/model` ids.
        assert_eq!(
            model_for_endpoint("https://openrouter.ai/api/v1", "gemini-web"),
            OPENROUTER_DEFAULT_MODEL
        );
        assert_eq!(
            model_for_endpoint("https://openrouter.ai/api/v1", "anthropic/claude-opus-5.5"),
            "anthropic/claude-opus-5.5"
        );
        // Every other endpoint takes the name as configured.
        assert_eq!(
            model_for_endpoint("http://127.0.0.1:11435/v1", "gemini-web"),
            "gemini-web"
        );
        assert_eq!(
            model_for_endpoint("https://api.groq.com/openai/v1", "gpt-4"),
            "gpt-4"
        );
        assert_eq!(model_for_endpoint("https://openrouter.ai/api/v1", "  "), "");
    }

    #[test]
    fn a_provider_selection_resolves_to_the_openrouter_endpoint() {
        let cfg = LucyConfig {
            providers: vec![ProviderConfig {
                id: "openrouter".into(),
                name: "OpenRouter".into(),
                api_url: OPENROUTER_BASE_URL.into(),
                api_key: "sk-or-test".into(),
                provider_type: ProviderType::Text,
                available_models: vec!["z-ai/glm-5.3".into()],
                ..Default::default()
            }],
            ..Default::default()
        };
        // `openrouter/z-ai/glm-5.3` has two slashes: the split must take the
        // provider id, not the vendor.
        let (url, key, model) = cfg.resolve_endpoint("openrouter/z-ai/glm-5.3").unwrap();
        assert_eq!(url, "https://openrouter.ai/api/v1");
        assert_eq!(key.as_deref(), Some("sk-or-test"));
        assert_eq!(model, "z-ai/glm-5.3");
        assert_eq!(cfg.text_model_options()[0].key(), "openrouter/z-ai/glm-5.3");

        // Saving the preset's draft (which carries no id) slugs the vendor
        // name into the id every model selection references.
        let mut fresh = LucyConfig::default();
        let stored = fresh
            .save_provider(ProviderConfig {
                name: "OpenRouter".into(),
                api_url: OPENROUTER_BASE_URL.into(),
                api_key: "sk-or-test".into(),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(stored.id, "openrouter");
    }

    #[test]
    fn openai_base_url_appends_v1_only_when_missing() {
        let mut p = text_provider("groq", &[]);
        assert_eq!(p.openai_base_url(), "https://api.groq.test/v1");
        assert_eq!(p.models_endpoint(), "https://api.groq.test/v1/models");
        p.api_url = "https://api.groq.test/v1/".into();
        assert_eq!(p.openai_base_url(), "https://api.groq.test/v1");
        p.api_url = "https://gw.test/openai/v2".into();
        assert_eq!(p.openai_base_url(), "https://gw.test/openai/v2");
    }

    #[test]
    fn provider_type_parses_ui_labels() {
        assert_eq!(ProviderType::parse("Text"), Some(ProviderType::Text));
        assert_eq!(ProviderType::parse("VOICE"), Some(ProviderType::Voice));
        assert_eq!(ProviderType::parse(" audio "), Some(ProviderType::Voice));
        assert_eq!(ProviderType::parse("nope"), None);
        assert_eq!(ProviderType::default(), ProviderType::Text);
    }

    #[test]
    fn model_options_split_by_provider_type_and_dedupe() {
        let cfg = LucyConfig {
            providers: vec![
                text_provider("groq", &["llama-3.3-70b", "flash"]),
                ProviderConfig {
                    provider_type: ProviderType::Voice,
                    available_models: vec!["whisper-large-v3-turbo".into(), "  ".into()],
                    ..text_provider("eleven", &[])
                },
            ],
            ..Default::default()
        };
        let text = cfg.text_model_options();
        assert_eq!(text.len(), 2);
        assert!(text.iter().all(|o| o.provider_id == "groq"));
        assert_eq!(text[0].key(), "groq/flash");
        assert_eq!(text[0].label(), "groq · flash");
        let voice = cfg.voice_model_options();
        assert_eq!(voice.len(), 1);
        assert_eq!(voice[0].key(), "eleven/whisper-large-v3-turbo");
    }

    #[test]
    fn save_provider_replaces_by_id_and_sorts_models() {
        let mut cfg = LucyConfig::default();
        let draft = ProviderConfig {
            id: String::new(),
            ..text_provider("groq", &["b", "a", "a"])
        };
        let saved = cfg.save_provider(draft).unwrap();
        assert!(!saved.id.is_empty());
        assert_eq!(cfg.providers.len(), 1);
        assert_eq!(cfg.providers[0].available_models, vec!["a", "b"]);

        // Second save with the generated id replaces in place.
        let saved2 = cfg
            .save_provider(ProviderConfig {
                id: saved.id.clone(),
                api_url: "https://api.groq.test/".into(),
                available_models: vec!["c".into()],
                ..text_provider("groq", &[])
            })
            .unwrap();
        assert_eq!(saved2.id, saved.id);
        assert_eq!(cfg.providers.len(), 1);
        assert_eq!(cfg.providers[0].api_url, "https://api.groq.test");
    }

    #[test]
    fn the_default_model_comes_from_the_live_text_endpoint() {
        // A synced provider serves the legacy `text_base_url` endpoint; when
        // nothing is explicitly selected, its first live (non-deprecated)
        // model beats the compiled-in name.
        let mut cfg = LucyConfig::default();
        cfg.models.default_text.clear();
        cfg.providers = vec![ProviderConfig {
            id: "openchat".into(),
            name: "OpenChat (local)".into(),
            api_url: DEFAULT_TEXT_BASE_URL.into(),
            api_key: "k".into(),
            provider_type: ProviderType::Text,
            available_models: vec!["gemini-flash".into(), "gemini-web".into()],
            deprecated_models: vec!["gemini-web".into()],
        }];
        assert_eq!(cfg.default_text_model(), "gemini-flash");

        // An explicit selection always wins, and the `/v1` suffix / a missing
        // one on either side matches the same provider.
        cfg.models.default_text = "gemini-pro".into();
        assert_eq!(cfg.default_text_model(), "gemini-pro");
        cfg.models.default_text.clear();
        cfg.models.text_base_url = Some("http://127.0.0.1:11435".into());
        cfg.providers[0].api_url = "http://127.0.0.1:11435/v1".into();
        assert_eq!(cfg.default_text_model(), "gemini-flash");
    }

    #[test]
    fn the_default_model_stays_compiled_in_when_nothing_is_synced() {
        let cfg = LucyConfig::default();
        assert_eq!(cfg.default_text_model(), DEFAULT_TEXT_MODEL);
        // A synced provider whose models are all deprecated is no better
        // than nothing to pick from.
        let mut cfg = LucyConfig::default();
        cfg.models.default_text.clear();
        cfg.providers = vec![ProviderConfig {
            id: "openchat".into(),
            name: "OpenChat (local)".into(),
            api_url: DEFAULT_TEXT_BASE_URL.into(),
            api_key: "k".into(),
            provider_type: ProviderType::Text,
            available_models: vec!["gemini-web".into()],
            deprecated_models: vec!["gemini-web".into()],
        }];
        assert_eq!(cfg.default_text_model(), DEFAULT_TEXT_MODEL);
    }

    #[test]
    fn a_bare_model_name_routes_to_the_provider_that_lists_it() {
        let mut cfg = LucyConfig::default();
        cfg.providers = vec![
            ProviderConfig {
                id: "openrouter".into(),
                name: "OpenRouter".into(),
                api_url: OPENROUTER_BASE_URL.into(),
                api_key: "sk-or".into(),
                provider_type: ProviderType::Text,
                available_models: vec!["google/gemini-2.5-flash-lite".into()],
                deprecated_models: Vec::new(),
            },
            ProviderConfig {
                id: "openchat".into(),
                name: "OpenChat (local)".into(),
                api_url: DEFAULT_TEXT_BASE_URL.into(),
                api_key: "k".into(),
                provider_type: ProviderType::Text,
                available_models: vec!["gemini-flash".into()],
                deprecated_models: Vec::new(),
            },
        ];
        let (p, model) = cfg.split_model_key("gemini-flash").unwrap();
        assert_eq!(p.id, "openchat");
        assert_eq!(model, "gemini-flash");
        let (p, _) = cfg.split_model_key("google/gemini-2.5-flash-lite").unwrap();
        assert_eq!(p.id, "openrouter");
        // A name no provider lists falls back to the legacy endpoint's
        // provider, not the first text provider.
        let (p, _) = cfg.split_model_key("gemini-web").unwrap();
        assert_eq!(p.id, "openchat");
    }

    #[test]
    fn remove_provider_clears_selections_that_pointed_at_it() {
        let mut cfg = LucyConfig {
            providers: vec![text_provider("groq", &["flash"])],
            ..Default::default()
        };
        cfg.set_anchor_model("groq/flash");
        cfg.set_voice_model("groq/flash");
        for level in ReasoningLevel::ALL {
            cfg.chat.reasoning_levels.set(level, "groq/flash".into());
        }
        assert!(cfg.remove_provider("GROQ"));
        assert!(!cfg.remove_provider("groq"));
        assert!(
            !cfg.resolve_level_model(ReasoningLevel::L3)
                .contains("groq/")
        );
        assert!(!cfg.voice_model().contains("groq/"));
        for level in ReasoningLevel::ALL {
            assert!(cfg.level_model(level).is_empty());
        }
    }

    #[test]
    fn anchor_model_is_the_level3_binding() {
        let mut cfg = LucyConfig::default();
        cfg.set_anchor_model("groq/flash");
        assert_eq!(
            cfg.level_model(ReasoningLevel::L3),
            "groq/flash",
            "the anchor model is Level 3 — there is no separate main model"
        );
        assert_eq!(cfg.resolve_level_model(ReasoningLevel::L3), "groq/flash");
        cfg.set_voice_model("eleven/tts");
        assert_eq!(cfg.voice.model, "eleven/tts");
    }

    #[test]
    fn an_unbound_tier_degrades_to_the_next_cheaper_tier() {
        let mut cfg = LucyConfig::default();
        // Nothing bound: every tier lands on the compiled-in default.
        assert_eq!(
            cfg.resolve_level_model(ReasoningLevel::L3),
            DEFAULT_TEXT_MODEL
        );
        assert_eq!(
            cfg.resolve_level_model(ReasoningLevel::L1),
            DEFAULT_TEXT_MODEL
        );

        // L1 bound only: L2 and L3 both fall *down* to it, never up.
        cfg.chat
            .reasoning_levels
            .set(ReasoningLevel::L1, "groq/fast".into());
        assert_eq!(cfg.resolve_level_model(ReasoningLevel::L1), "groq/fast");
        assert_eq!(cfg.resolve_level_model(ReasoningLevel::L2), "groq/fast");
        assert_eq!(cfg.resolve_level_model(ReasoningLevel::L3), "groq/fast");

        // L2 joins: L3 skips down to L2, L1 keeps its own cheaper model.
        cfg.chat
            .reasoning_levels
            .set(ReasoningLevel::L2, "openai/gpt-5".into());
        assert_eq!(cfg.resolve_level_model(ReasoningLevel::L1), "groq/fast");
        assert_eq!(cfg.resolve_level_model(ReasoningLevel::L2), "openai/gpt-5");
        assert_eq!(cfg.resolve_level_model(ReasoningLevel::L3), "openai/gpt-5");

        // L3 bound: every tier is its own model again.
        cfg.set_anchor_model("anthropic/claude");
        assert_eq!(cfg.resolve_level_model(ReasoningLevel::L1), "groq/fast");
        assert_eq!(cfg.resolve_level_model(ReasoningLevel::L2), "openai/gpt-5");
        assert_eq!(
            cfg.resolve_level_model(ReasoningLevel::L3),
            "anthropic/claude"
        );

        // Unbinding L1 last, with L2/L3 still bound, leaves it with no cheaper
        // tier to skip to — so it falls to the compiled-in default rather than
        // climbing back up to L3.
        cfg.chat
            .reasoning_levels
            .set(ReasoningLevel::L1, String::new());
        assert_eq!(
            cfg.resolve_level_model(ReasoningLevel::L1),
            DEFAULT_TEXT_MODEL
        );
    }

    #[test]
    fn the_fallback_note_names_the_tier_that_will_answer() {
        let mut cfg = LucyConfig::default();
        cfg.chat
            .reasoning_levels
            .set(ReasoningLevel::L2, "openai/gpt-5".into());

        // Unbound L3 skips down to the bound L2 and says so.
        assert_eq!(
            cfg.level_fallback_note(ReasoningLevel::L3).as_deref(),
            Some("no model bound to L3 — using L2")
        );
        // A bound tier has nothing to explain.
        assert!(cfg.level_fallback_note(ReasoningLevel::L2).is_none());
    }

    #[test]
    fn an_unbound_l1_names_the_default_not_a_missing_tier() {
        // L1 is the cheapest tier, so an unbound L1 has nothing to skip down to
        // and lands on the compiled-in default. L2/L3 are bound here, so the
        // note must say *that* — "no reasoning tier bound" would be a lie and
        // would hide the fact that the turn left the configured providers.
        let mut cfg = LucyConfig::default();
        cfg.chat
            .reasoning_levels
            .set(ReasoningLevel::L2, "openai/gpt-5".into());
        cfg.chat
            .reasoning_levels
            .set(ReasoningLevel::L3, "anthropic/claude".into());

        assert_eq!(
            cfg.resolve_level_model(ReasoningLevel::L1),
            DEFAULT_TEXT_MODEL
        );
        assert_eq!(
            cfg.level_fallback_note(ReasoningLevel::L1).as_deref(),
            Some(
                format!(
                    "no model bound to L1 and no cheaper tier is bound — using the default {DEFAULT_TEXT_MODEL} model"
                )
                .as_str()
            )
        );
    }

    #[test]
    fn with_no_tier_bound_at_all_the_note_says_so_once() {
        let cfg = LucyConfig::default();
        assert_eq!(
            cfg.level_fallback_note(ReasoningLevel::L3).as_deref(),
            Some(
                format!("no reasoning tier bound — using the default {DEFAULT_TEXT_MODEL} model")
                    .as_str()
            )
        );
    }

    #[test]
    fn a_legacy_main_model_seeds_level3_on_load() {
        // What an old config.toml held before the tiers existed.
        let raw = r#"
[chat]
main_model = "groq/flash"

[models]
main = "gemini-web"
main_base_url = "http://127.0.0.1:11435/v1"
main_api_key = "sk-legacy"
"#;
        let mut cfg: LucyConfig = toml::from_str(raw).unwrap();
        assert_eq!(cfg.chat.legacy_main_model, "groq/flash");
        cfg.apply_env().unwrap();
        assert_eq!(cfg.level_model(ReasoningLevel::L3), "groq/flash");
        assert!(
            cfg.chat.legacy_main_model.is_empty(),
            "the legacy key must not be written back"
        );
        // The renamed endpoint fields still read the old keys.
        assert_eq!(cfg.text_api_key().as_deref(), Some("sk-legacy"));
        assert_eq!(
            cfg.text_base_url().as_deref(),
            Some("http://127.0.0.1:11435/v1")
        );
    }

    #[test]
    fn an_explicit_level3_beats_the_legacy_main_model() {
        let raw = r#"
[chat]
main_model = "groq/flash"

[chat.reasoning_levels]
level3 = "openai/gpt-5"
"#;
        let mut cfg: LucyConfig = toml::from_str(raw).unwrap();
        cfg.apply_env().unwrap();
        assert_eq!(cfg.level_model(ReasoningLevel::L3), "openai/gpt-5");
        assert!(cfg.chat.legacy_main_model.is_empty());
    }

    #[test]
    fn a_migrated_config_never_writes_the_legacy_key_back() {
        let raw = r#"
[chat]
main_model = "groq/flash"
"#;
        let mut cfg: LucyConfig = toml::from_str(raw).unwrap();
        cfg.apply_env().unwrap();
        let text = toml::to_string_pretty(&cfg).unwrap();
        assert!(
            !text.contains("main_model"),
            "the migrated value must not reappear under its old name:\n{text}"
        );
        assert!(text.contains("level3 = \"groq/flash\""), "{text}");
    }

    #[test]
    fn resolve_endpoint_uses_provider_credentials() {
        let cfg = LucyConfig {
            providers: vec![text_provider("groq", &["flash"])],
            ..Default::default()
        };
        let (url, key, model) = cfg.resolve_endpoint("groq/flash").unwrap();
        assert_eq!(url, "https://api.groq.test/v1");
        assert_eq!(key.as_deref(), Some("sk-test"));
        assert_eq!(model, "flash");
    }

    #[test]
    fn blank_provider_ids_are_slugged_and_uniquified_in_memory() {
        let mut p = text_provider("Groq", &[]);
        p.id = String::new();
        assert_eq!(p.id_slug(), "groq");
        p.api_url = "https://api.example.test/v1".into();
        p.name = String::new();
        assert_eq!(p.id_slug(), "https-api-example-test-v1");

        // Uniquified against the ids handed in, without touching the disk.
        p.name = "Groq".into();
        p.ensure_id(&["groq".into(), "groq-2".into()]);
        assert_eq!(p.id, "groq-3");
        // An explicit id is never overwritten.
        p.id = "custom".into();
        p.ensure_id(&[]);
        assert_eq!(p.id, "custom");
    }

    #[test]
    fn classification_url_validation() {
        let mut cfg = LucyConfig::default();
        assert_eq!(cfg.classification_api_url(), "http://localhost:8001");
        assert!(cfg.save_classification_url("").is_err());
        assert!(cfg.save_classification_url("localhost:8001").is_err());
        let saved = cfg
            .save_classification_url("http://127.0.0.1:8001/")
            .unwrap();
        assert_eq!(saved, "http://127.0.0.1:8001");
        assert_eq!(cfg.classification_api_url(), "http://127.0.0.1:8001");
    }

    #[test]
    fn reasoning_level_wire_roundtrip() {
        for level in ReasoningLevel::ALL {
            assert_eq!(ReasoningLevel::from_wire(level.as_wire()), Some(level));
        }
        assert_eq!(ReasoningLevel::from_wire("9"), None);
        assert_eq!(ReasoningLevel::L3.as_number(), 3);
        assert_eq!(ReasoningLevel::L1.short(), "L1");
    }

    #[test]
    fn config_toml_roundtrips_new_sections() {
        let mut cfg = LucyConfig {
            providers: vec![text_provider("groq", &["flash"])],
            ..Default::default()
        };
        cfg.set_anchor_model("groq/flash");
        cfg.set_chat_mode(ChatMode::Manual);
        cfg.set_auto_compact(false);
        cfg.chat
            .reasoning_levels
            .set(ReasoningLevel::L1, "groq/flash".into());
        let text = toml::to_string_pretty(&cfg).unwrap();
        let back: LucyConfig = toml::from_str(&text).unwrap();
        assert_eq!(back.providers.len(), 1);
        assert_eq!(back.providers[0].available_models, vec!["flash"]);
        assert_eq!(back.chat_mode(), ChatMode::Manual);
        assert!(!back.auto_compact());
        assert_eq!(back.level_model(ReasoningLevel::L1), "groq/flash");
        assert_eq!(back.text_model_options().len(), 1);
    }

    #[test]
    fn approval_mode_round_trips_and_automode_is_persisted() {
        let mut cfg = LucyConfig::default();
        assert_eq!(cfg.approval_mode(), "write");
        assert!(!cfg.automode());

        cfg.set_approval_mode("never");
        assert!(cfg.automode());
        assert_eq!(cfg.approval_mode_label(), "auto — never ask");

        // The whole point of the fix: `/auto on` must survive a restart.
        let text = toml::to_string_pretty(&cfg).unwrap();
        let back: LucyConfig = toml::from_str(&text).unwrap();
        assert!(back.automode(), "automode was lost across a reload");
    }

    #[test]
    fn launch_flags_join_into_one_env_value_or_none() {
        // Rule under test: a flag list is one space-joined env value, and an
        // empty list yields None so callers don't set an empty variable.
        assert_eq!(
            join_flags(&[
                "--disable-gpu".to_string(),
                "--disable-gpu-compositing".to_string()
            ]),
            Some("--disable-gpu --disable-gpu-compositing".to_string())
        );
        assert_eq!(join_flags(&[]), None);
        assert_eq!(join_flags(&["  ".to_string()]), None);
    }

    #[test]
    fn browser_launch_args_expose_their_joined_value() {
        // The value is what lucy forwards to hyprfast, so an empty list must
        // produce None (hyprfast then uses browser defaults) and a populated
        // one the exact flag string.
        let mut empty = BrowserConfig::default();
        empty.launch_args = vec![];
        assert_eq!(empty.launch_args_value(), None);

        let mut browser = BrowserConfig::default();
        browser.launch_args = vec!["--disable-gpu".into(), "--no-sandbox".into()];
        assert_eq!(
            browser.launch_args_value().as_deref(),
            Some("--disable-gpu --no-sandbox")
        );
    }

    #[test]
    fn a_fresh_config_turns_graphics_acceleration_off() {
        // The browser's own default is acceleration ON. Lucy's hit-test
        // guards compare CDP rects against what the compositor hit-tests, so
        // the default has to be off rather than opt-in — otherwise every
        // install silently ships GPU compositing.
        assert!(
            BrowserConfig::default()
                .launch_args
                .iter()
                .any(|f| f == "--disable-gpu"),
            "default launch_args must disable GPU: {:?}",
            BrowserConfig::default().launch_args
        );
    }

    #[test]
    fn browser_launch_args_survive_a_config_reload() {
        let mut cfg = LucyConfig::default();
        cfg.browser.launch_args = vec!["--disable-gpu".into(), "--no-sandbox".into()];
        let text = toml::to_string_pretty(&cfg).unwrap();
        let back: LucyConfig = toml::from_str(&text).unwrap();
        assert_eq!(
            back.browser.launch_args,
            vec!["--disable-gpu", "--no-sandbox"]
        );
    }

    #[test]
    fn an_unset_user_data_dir_defers_to_hyprfast() {
        // Empty means "let hyprfast resolve it", so the default must not pin a
        // path here — a second default would be a second place to change.
        assert_eq!(BrowserConfig::default().user_data_dir, "");
        assert_eq!(BrowserConfig::default().user_data_dir_value(), None);
    }

    #[test]
    fn a_blank_user_data_dir_is_treated_as_unset() {
        // An exported-but-blank value would otherwise reach hyprfast as an
        // empty --user-data-dir.
        for blank in ["", "   ", "\t"] {
            let mut browser = BrowserConfig::default();
            browser.user_data_dir = blank.into();
            assert_eq!(
                browser.user_data_dir_value(),
                None,
                "{blank:?} should defer"
            );
        }
    }

    #[test]
    fn a_user_data_dir_is_trimmed_before_it_is_forwarded() {
        let mut browser = BrowserConfig::default();
        browser.user_data_dir = "  /srv/lucy/profile  ".into();
        assert_eq!(
            browser.user_data_dir_value().as_deref(),
            Some("/srv/lucy/profile")
        );
    }

    #[test]
    fn the_user_data_dir_survives_a_config_reload() {
        let mut cfg = LucyConfig::default();
        cfg.browser.user_data_dir = "/srv/lucy/profile".into();
        let text = toml::to_string_pretty(&cfg).unwrap();
        let back: LucyConfig = toml::from_str(&text).unwrap();
        assert_eq!(back.browser.user_data_dir, "/srv/lucy/profile");
    }

    /// A config written before this key existed must still load, with the key
    /// defaulting — `#[serde(default)]` is what makes that true.
    #[test]
    fn a_config_without_the_user_data_dir_key_still_loads() {
        let back: LucyConfig =
            toml::from_str("cdp_port = 9222\nlaunch_timeout_secs = 8\n").unwrap();
        assert_eq!(back.browser.cdp_port, 9222);
        assert_eq!(back.browser.user_data_dir_value(), None);
    }

    /// The default profile has to be somewhere a reboot cannot take. A tmpfs
    /// is RAM, which is why this used to lose every login on reboot.
    #[test]
    fn the_default_profile_directory_is_not_temporary() {
        let dir = default_profile_dir();
        let tmp = std::env::temp_dir();
        assert!(
            !dir.starts_with(&tmp),
            "profile must not live under {}: {}",
            tmp.display(),
            dir.display()
        );
        assert!(dir.ends_with("hyprfast/browser-profile"), "{dir:?}");
    }

    #[test]
    fn a_bogus_approval_mode_degrades_to_write() {
        let mut cfg = LucyConfig::default();
        cfg.approvals.mode = "yolo".into();
        // Not silently "never": a typo must never disable prompts.
        assert_eq!(cfg.approval_mode(), "write");
        assert!(!cfg.automode());
    }

    #[test]
    fn always_allow_persists_sorted_and_deduped() {
        let mut cfg = LucyConfig::default();
        cfg.set_always_allow(vec![
            "write_file".into(),
            "  ".into(),
            "edit_file".into(),
            "write_file".into(),
        ]);
        assert_eq!(cfg.always_allow(), ["edit_file", "write_file"]);
        let text = toml::to_string_pretty(&cfg).unwrap();
        let back: LucyConfig = toml::from_str(&text).unwrap();
        assert_eq!(back.always_allow(), ["edit_file", "write_file"]);
    }

    #[test]
    fn approval_config_defaults_fill_in_for_old_config_files() {
        // An `mcp`-era config.toml with only `mode = "write"` and no
        // `always_allow` key must still deserialize.
        let back: LucyConfig = toml::from_str("[approvals]\nmode = \"never\"\n").unwrap();
        assert_eq!(back.approval_mode(), "never");
        assert!(back.always_allow().is_empty());
    }

    #[test]
    fn chat_mode_parses_ui_labels() {
        assert_eq!(ChatMode::parse("Auto"), Some(ChatMode::Auto));
        assert_eq!(ChatMode::parse("MANUAL"), Some(ChatMode::Manual));
        assert_eq!(ChatMode::parse("x"), None);
        assert_eq!(ChatMode::default(), ChatMode::Auto);
    }

    #[test]
    fn the_agent_loop_is_the_default_path_and_round_trips() {
        let mut cfg = LucyConfig::default();
        assert!(
            cfg.agent_loop_enabled(),
            "the agent loop is the live act path"
        );
        assert!(cfg.validate().is_ok());
        cfg.set("harness.agent_loop_enabled", "false").unwrap();
        assert!(!cfg.agent_loop_enabled());
        assert_eq!(cfg.get("harness.agent_loop_enabled").unwrap(), "false");
        let text = toml::to_string_pretty(&cfg).unwrap();
        let back: LucyConfig = toml::from_str(&text).unwrap();
        assert!(!back.agent_loop_enabled());
        // A config file written before the flag existed takes the default.
        let legacy: LucyConfig = toml::from_str("[harness]\nmax_recoveries = 2\n").unwrap();
        assert!(legacy.agent_loop_enabled());
    }

    /// The topic list is the one piece of task vocabulary Lucy keeps, and it is
    /// configuration rather than code so a new noun needs no release — see
    /// AGENTS.md.
    #[test]
    fn browser_goal_keywords_are_configurable_and_round_trip() {
        let mut cfg = LucyConfig::default();
        // The shipped default carries the words the old hardcoded list had, so
        // moving them here changed no behaviour.
        assert!(
            cfg.harness
                .browser_goal_keywords
                .iter()
                .any(|k| k == "flight")
        );
        assert!(cfg.validate().is_ok());

        cfg.set(
            "harness.browser_goal_keywords",
            "[\"recipe\", \"reservation\"]",
        )
        .unwrap();
        assert_eq!(cfg.harness.browser_goal_keywords, ["recipe", "reservation"]);

        let text = toml::to_string_pretty(&cfg).unwrap();
        let back: LucyConfig = toml::from_str(&text).unwrap();
        assert_eq!(
            back.harness.browser_goal_keywords,
            ["recipe", "reservation"],
            "the list survives a save/load cycle"
        );

        // An empty list is valid: the structural signals still work, so the
        // user can opt out of topic matching entirely.
        cfg.set("harness.browser_goal_keywords", "[]").unwrap();
        assert!(cfg.harness.browser_goal_keywords.is_empty());
        assert!(cfg.validate().is_ok());

        // A config file written before this key existed takes the default rather
        // than silently launching no browser for a web task.
        let legacy: LucyConfig = toml::from_str("[harness]\nmax_recoveries = 2\n").unwrap();
        assert!(!legacy.harness.browser_goal_keywords.is_empty());
    }

    #[test]
    fn the_gateway_is_off_by_default() {
        // The bridge opens a listening socket and can hold an MCP-spawning
        // runtime; a fresh install must not pay for either until asked.
        let cfg = LucyConfig::default();
        assert!(!cfg.gateway_enabled());
        assert!(!cfg.gateway.enabled);
        assert_eq!(cfg.gateway.port, 9847);
        assert_eq!(cfg.gateway.bind, "127.0.0.1");
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn a_config_file_without_a_gateway_section_stays_off() {
        // Every install written before the gateway existed must keep working
        // and must not start serving: absence means disabled, not "inherit a
        // default that happens to be on".
        let legacy: LucyConfig = toml::from_str("[general]\nstartup_screen = \"mascot\"\n")
            .expect("legacy config parses");
        assert!(!legacy.gateway.enabled);
    }

    #[test]
    fn gateway_round_trips_and_validates() {
        let mut cfg = LucyConfig::default();
        cfg.set("gateway.enabled", "true").unwrap();
        cfg.set("gateway.port", "9911").unwrap();
        cfg.set("gateway.bind", "0.0.0.0").unwrap();
        assert!(cfg.gateway_enabled());
        assert_eq!(cfg.gateway.port, 9911);

        let text = toml::to_string_pretty(&cfg).unwrap();
        let back: LucyConfig = toml::from_str(&text).unwrap();
        assert!(back.gateway.enabled);
        assert_eq!(back.gateway.bind, "0.0.0.0");

        // A zero port is a bind failure waiting to happen; catch it at load.
        let mut bad = LucyConfig::default();
        bad.gateway.port = 0;
        assert!(bad.validate().is_err());
        let mut blank = LucyConfig::default();
        blank.gateway.bind = "  ".into();
        assert!(blank.validate().is_err());
    }
}
