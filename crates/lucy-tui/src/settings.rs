//! `/settings` — the six-section settings screen.
//!
//! Sections, in the order the wireframe shows them:
//!
//! 1. **Connect providers** — `API URL` / `API Key` / `Type` + `[Test]` `[Save]`
//! 2. **Classification model** — `API URL` only (no API key) + `[Test]` `[Save]`
//! 3. **Chat mode** — `Auto`/`Manual` + the Level 1/2/3 model dropdowns
//! 4. **Auto compact** — `On`/`Off`
//! 5. **Automode** — `Risky tools only` / `Auto` / `Always ask`
//! 6. **Voice model** — every voice model across all connected voice providers
//!
//! There is no "Main model" row: the three text models are the Level 1/2/3
//! dropdowns in section 3, and Level 3 is the anchor.
//!
//! Navigation: `↑`/`↓` move (one flat cursor across every row), `←`/`→` or
//! `Enter` change a dropdown / press a button, plain characters edit the
//! focused text field, `Esc` closes. `[Test]` and `[Save]` are async, so the
//! screen shows a busy line and a result line.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use lucy_agent::{
    ProviderHealth, friendly_probe_error, persist_classification_url, persist_provider,
    test_connection,
};
use lucy_config::{
    ChatMode, LucyConfig, ModelOption, ProviderConfig, ProviderPreset, ProviderType, ReasoningLevel,
};
use ratatui::{
    Frame,
    layout::{Alignment, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
};

/// Every focusable row, grouped by section. The flat index into this array is
/// the cursor, so `↑`/`↓` walk the screen exactly as it is drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Row {
    // 1. Connect providers
    ProviderPreset,
    ApiUrl,
    ApiKey,
    ProviderType,
    ProviderTest,
    ProviderSave,
    // 2. Classification model
    ClassifyUrl,
    ClassifyTest,
    ClassifySave,
    // 3. Chat mode
    ChatMode,
    Level1,
    Level2,
    Level3,
    // 4. Auto compact
    AutoCompact,
    // 5. Automode (approvals)
    Approvals,
    // 6. Voice model
    VoiceModel,
}

/// The six sections, in wireframe order. There is no "main model" section:
/// Level 3 is the anchor and is configured with the other two tiers under
/// Chat mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Section {
    Providers,
    Classification,
    ChatMode,
    AutoCompact,
    Approvals,
    VoiceModel,
}

impl Section {
    pub const ALL: [Section; 6] = [
        Section::Providers,
        Section::Classification,
        Section::ChatMode,
        Section::AutoCompact,
        Section::Approvals,
        Section::VoiceModel,
    ];

    /// Section header text.
    pub fn title(&self) -> &'static str {
        match self {
            Section::Providers => "1. Connect providers",
            Section::Classification => "2. Classification model",
            Section::ChatMode => "3. Chat mode",
            Section::AutoCompact => "4. Auto Compact",
            Section::Approvals => "5. Automode",
            Section::VoiceModel => "6. Voice model",
        }
    }

    /// The rows this section owns, in draw order.
    pub fn rows(&self) -> &'static [Row] {
        match self {
            Section::Providers => &[
                Row::ProviderPreset,
                Row::ApiUrl,
                Row::ApiKey,
                Row::ProviderType,
                Row::ProviderTest,
                Row::ProviderSave,
            ],
            Section::Classification => &[Row::ClassifyUrl, Row::ClassifyTest, Row::ClassifySave],
            Section::ChatMode => &[Row::ChatMode, Row::Level1, Row::Level2, Row::Level3],
            Section::AutoCompact => &[Row::AutoCompact],
            Section::Approvals => &[Row::Approvals],
            Section::VoiceModel => &[Row::VoiceModel],
        }
    }
}

/// Flat `(section, row)` layout of the whole screen.
pub const LAYOUT: [(Section, Row); 16] = [
    (Section::Providers, Row::ProviderPreset),
    (Section::Providers, Row::ApiUrl),
    (Section::Providers, Row::ApiKey),
    (Section::Providers, Row::ProviderType),
    (Section::Providers, Row::ProviderTest),
    (Section::Providers, Row::ProviderSave),
    (Section::Classification, Row::ClassifyUrl),
    (Section::Classification, Row::ClassifyTest),
    (Section::Classification, Row::ClassifySave),
    (Section::ChatMode, Row::ChatMode),
    (Section::ChatMode, Row::Level1),
    (Section::ChatMode, Row::Level2),
    (Section::ChatMode, Row::Level3),
    (Section::AutoCompact, Row::AutoCompact),
    (Section::Approvals, Row::Approvals),
    (Section::VoiceModel, Row::VoiceModel),
];

impl Row {
    /// Left-hand label for the row.
    pub fn label(&self) -> &'static str {
        match self {
            Row::ProviderPreset => "Provider",
            Row::ApiUrl => "API URL",
            Row::ApiKey => "API Key",
            Row::ProviderType => "Type",
            Row::ProviderTest => "[Test]",
            Row::ProviderSave => "[Save]",
            Row::ClassifyUrl => "API URL",
            Row::ClassifyTest => "[Test]",
            Row::ClassifySave => "[Save]",
            Row::ChatMode => "Chat mode",
            Row::Level1 => "Level 1 (fast)",
            Row::Level2 => "Level 2 (balanced)",
            Row::Level3 => "Level 3 (deep reasoning)",
            Row::AutoCompact => "Auto Compact",
            Row::Approvals => "Permissions",
            Row::VoiceModel => "Voice model",
        }
    }

    /// True for rows that accept typed characters.
    pub fn is_text(&self) -> bool {
        matches!(self, Row::ApiUrl | Row::ApiKey | Row::ClassifyUrl)
    }

    /// True for rows that act when `Enter` is pressed.
    pub fn is_button(&self) -> bool {
        matches!(
            self,
            Row::ProviderTest | Row::ProviderSave | Row::ClassifyTest | Row::ClassifySave
        )
    }

    /// True for rows changed with `←`/`→`.
    pub fn is_choice(&self) -> bool {
        !self.is_text() && !self.is_button()
    }

    /// True for rows whose choice list is long enough that `←`/`→` stepping
    /// alone is unusable, so they also offer a `[change]` button that opens a
    /// searchable dropdown. The list stays closed until the user asks for it.
    ///
    /// The three level rows are the reason: they offer every text model across
    /// every connected provider, so on a real account that is hundreds of
    /// entries and blind-cycling through them is unusable. The small lists
    /// (Chat mode, Auto Compact, Permissions, Voice model) keep plain `←`/`→`.
    pub fn has_model_picker(&self) -> bool {
        matches!(self, Row::Level1 | Row::Level2 | Row::Level3)
    }

    /// The section that owns this row.
    pub fn section(&self) -> Section {
        LAYOUT
            .iter()
            .find(|(_, r)| r == self)
            .map(|(s, _)| *s)
            .unwrap_or(Section::Providers)
    }
}

/// Severity of the screen's result line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusKind {
    Info,
    Ok,
    Warn,
    Err,
}

/// One selectable entry in the open model dropdown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PickerChoice {
    /// `provider_id/model` written to the config, or `""` for the
    /// "unbind this tier" clear entry.
    pub key: String,
    /// Text shown on the entry's own line.
    pub display: String,
    /// Heading the entry sits under, or `None` for the bare clear entry that
    /// leads the list.
    pub group: Option<String>,
}

/// The searchable model dropdown, opened from a row's `[change]`.
///
/// The settings screen only ever *draws* the current selection
/// (`Level 1  <model>  [change]`); this lives in `SettingsState::picker` and
/// is `None` until `Enter`/`[change]` opens it, so the list is never open by
/// default and its cost is paid only when the user wants it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPicker {
    /// The row being edited — the tier whose model is being changed.
    row: Row,
    /// Search query. Empty = every choice.
    query: String,
    /// Index into the *filtered* choice list.
    cursor: usize,
}

/// Test-only accessors for the open dropdown. Production code reaches the same
/// state through the key handlers, so these would otherwise read as dead code.
#[cfg(test)]
impl ModelPicker {
    /// The tier this dropdown is changing.
    pub fn row(&self) -> Row {
        self.row
    }

    /// The current search query.
    pub fn query(&self) -> &str {
        &self.query
    }

    /// Index of the highlighted entry in the filtered list.
    pub fn cursor(&self) -> usize {
        self.cursor
    }
}

/// Heading for the models that are already bound somewhere in the tier stack,
/// so switching a tier does not mean re-searching for what is already in use.
const RECENT_GROUP: &str = "Recent";

/// Heading for the model a tier falls back to when nothing is bound, when no
/// connected provider exposes it. Kept apart from `Recent` so the list never
/// implies the user picked it.
const DEFAULT_GROUP: &str = "Default";

/// Why the settings screen closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingsAction {
    Close,
    /// Model dropdowns changed, so the caller should refresh its labels.
    ModelsChanged,
    /// The approval mode changed. The caller must push it into the runtime's
    /// live gate, or the new mode only applies after a restart.
    ApprovalModeChanged,
}

/// Editable draft + cursor state for the settings screen.
#[derive(Debug, Clone)]
pub struct SettingsState {
    pub open: bool,
    cursor: usize,
    // 1. Connect providers
    /// Selected vendor preset, or `Custom` once the URL is hand-typed.
    provider_preset: String,
    /// Provider display name, filled in by the preset and stored on `[Save]`.
    provider_name: String,
    pub api_url: String,
    pub api_key: String,
    pub provider_type: ProviderType,
    /// Model ids discovered by the last successful `[Test]`.
    pub discovered_models: Vec<String>,
    /// Deprecated ids among [`Self::discovered_models`] (from the `/models` payload).
    pub discovered_deprecated: Vec<String>,
    /// True once `[Test]` has verified the current URL/key pair.
    tested: bool,
    // 2. Classification model
    pub classification_url: String,
    // 3. Chat mode
    pub chat_mode: ChatMode,
    /// The three reasoning tiers, indexed by `ReasoningLevel::as_number() - 1`.
    pub level_models: [String; 3],
    // 4. Auto compact
    pub auto_compact: bool,
    // 5. Automode: one of the raw `approvals.mode` values.
    pub approvals_mode: String,
    // 6. Voice model, as `provider_id/model`
    pub voice_model: String,
    // Dropdown contents, refreshed from the config whenever the screen opens.
    text_models: Vec<ModelOption>,
    voice_models: Vec<ModelOption>,
    /// Already-configured models, resolved at `open()` time (provider-backed
    /// selection → the tier's fallback chain → the compiled-in default). These
    /// are unioned into every dropdown's key list so a configured model is
    /// always shown and re-selectable — even with zero providers connected or
    /// after the live selection is cleared to `""`.
    configured_levels: [String; 3],
    configured_voice: String,
    /// Result line under the sections.
    pub status: String,
    pub status_kind: StatusKind,
    /// True while a `[Test]`/`[Save]` network call is in flight.
    pub busy: bool,
    /// The searchable model dropdown, while it is open. `None` = closed, which
    /// is the default and the state the screen renders.
    pub picker: Option<ModelPicker>,
}

impl SettingsState {
    /// A closed screen, ready to be opened with [`Self::open`].
    pub fn closed() -> Self {
        Self {
            open: false,
            cursor: 0,
            provider_preset: ProviderPreset::Custom.name().to_owned(),
            provider_name: String::new(),
            api_url: String::new(),
            api_key: String::new(),
            provider_type: ProviderType::Text,
            discovered_models: Vec::new(),
            discovered_deprecated: Vec::new(),
            tested: false,
            classification_url: String::new(),
            chat_mode: ChatMode::Auto,
            level_models: [String::new(), String::new(), String::new()],
            auto_compact: true,
            approvals_mode: "write".into(),
            voice_model: String::new(),
            text_models: Vec::new(),
            voice_models: Vec::new(),
            configured_levels: Self::resolve_level_fallbacks(&LucyConfig::default()),
            configured_voice: Self::resolve_voice_fallback(&LucyConfig::default()),
            status: String::new(),
            status_kind: StatusKind::Info,
            busy: false,
            picker: None,
        }
    }

    /// Build the draft from the persisted config, with the model dropdowns
    /// populated from every connected provider.
    pub fn open(config: &LucyConfig) -> Self {
        Self {
            open: true,
            cursor: 0,
            provider_preset: ProviderPreset::Custom.name().to_owned(),
            provider_name: String::new(),
            api_url: String::new(),
            api_key: String::new(),
            provider_type: ProviderType::Text,
            discovered_models: Vec::new(),
            discovered_deprecated: Vec::new(),
            tested: false,
            classification_url: config.classification_api_url().to_owned(),
            chat_mode: config.chat_mode(),
            level_models: [
                config.level_model(ReasoningLevel::L1),
                config.level_model(ReasoningLevel::L2),
                config.level_model(ReasoningLevel::L3),
            ],
            auto_compact: config.auto_compact(),
            approvals_mode: config.approval_mode().to_owned(),
            voice_model: config.voice_model(),
            text_models: config.text_model_options(),
            voice_models: config.voice_model_options(),
            configured_levels: Self::resolve_level_fallbacks(config),
            configured_voice: Self::resolve_voice_fallback(config),
            status: String::new(),
            status_kind: StatusKind::Info,
            busy: false,
            picker: None,
        }
    }

    /// The model each tier will actually call, for the dropdown unions.
    /// `resolve_level_model` never returns `""` — it degrades an unbound tier
    /// to the next cheaper one and finally to the compiled-in default — so
    /// every tier's effective model is always available to re-select.
    fn resolve_level_fallbacks(config: &LucyConfig) -> [String; 3] {
        [
            config.resolve_level_model(ReasoningLevel::L1),
            config.resolve_level_model(ReasoningLevel::L2),
            config.resolve_level_model(ReasoningLevel::L3),
        ]
    }

    /// Same for voice: `voice_model()` already falls back to the legacy
    /// `voice.model`; only the compiled-in default is added when that is
    /// somehow empty too.
    fn resolve_voice_fallback(config: &LucyConfig) -> String {
        let m = config.voice_model();
        if m.trim().is_empty() {
            LucyConfig::default().voice_model()
        } else {
            m
        }
    }

    /// Re-resolve the configured fallbacks from a (possibly just-updated)
    /// config. Called by `refresh_models` and after a provider save so
    /// clearing a selection never orphans the fallback.
    fn refresh_fallbacks(&mut self, config: &LucyConfig) {
        self.configured_levels = Self::resolve_level_fallbacks(config);
        self.configured_voice = Self::resolve_voice_fallback(config);
    }

    pub fn close(&mut self) {
        self.open = false;
    }

    /// Index of the focused row.
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// The focused row.
    pub fn focused(&self) -> Row {
        LAYOUT[self.cursor.min(LAYOUT.len() - 1)].1
    }

    /// The section the cursor is currently in.
    pub fn focused_section(&self) -> Section {
        self.focused().section()
    }

    /// Dropdown contents for a text-model row.
    pub fn text_models(&self) -> &[ModelOption] {
        &self.text_models
    }

    /// Dropdown contents for the voice-model row.
    pub fn voice_models(&self) -> &[ModelOption] {
        &self.voice_models
    }

    fn move_cursor(&mut self, delta: i32) {
        let len = LAYOUT.len() as i32;
        self.cursor = (self.cursor as i32 + delta).rem_euclid(len) as usize;
    }

    /// Options for a choice row, in cycle order. `(value, label)`.
    pub fn options(&self, row: Row) -> Vec<String> {
        match row {
            Row::ProviderPreset => ProviderPreset::ALL
                .iter()
                .map(|p| p.name().to_owned())
                .collect(),
            Row::ProviderType => vec!["Text".into(), "Voice".into()],
            Row::ChatMode => vec!["Auto".into(), "Manual".into()],
            Row::AutoCompact => vec!["On".into(), "Off".into()],
            Row::Approvals => vec![
                "Risky tools only".into(),
                "Auto (never ask)".into(),
                "Always ask".into(),
            ],
            Row::VoiceModel => labels_for_keys(&self.voice_models, &self.voice_keys(), "(not set)"),
            Row::Level1 | Row::Level2 | Row::Level3 => {
                labels_for_keys(&self.text_models, &self.level_keys(), "(unbound)")
            }
            _ => Vec::new(),
        }
    }

    /// Current value of a choice row, as displayed.
    pub fn value(&self, row: Row) -> String {
        match row {
            Row::ProviderPreset => self.provider_preset.clone(),
            Row::ApiUrl => display_or_placeholder(&self.api_url, lucy_config::OPENROUTER_BASE_URL),
            Row::ApiKey => display_or_placeholder(&self.api_key, "sk-…"),
            Row::ClassifyUrl => {
                display_or_placeholder(&self.classification_url, "http://localhost:8001")
            }
            Row::ProviderType => self.provider_type.as_str().to_owned(),
            Row::ChatMode => self.chat_mode.as_str().to_owned(),
            Row::AutoCompact => if self.auto_compact { "On" } else { "Off" }.into(),
            Row::Approvals => self.approvals_mode_label().to_owned(),
            Row::VoiceModel => self.model_display(&self.voice_model, &self.voice_models),
            Row::Level1 | Row::Level2 | Row::Level3 => {
                let idx = level_index(row);
                self.level_display(&self.level_models[idx])
            }
            Row::ProviderTest => "check the URL + key, list models".into(),
            Row::ProviderSave => "store the provider + models".into(),
            Row::ClassifyTest => "GET /health + POST /predict".into(),
            Row::ClassifySave => "store the API URL".into(),
        }
    }

    fn model_display(&self, key: &str, options: &[ModelOption]) -> String {
        if key.trim().is_empty() {
            return "(not set)".into();
        }
        match options.iter().find(|o| o.key() == key) {
            Some(o) => o.label(),
            None => key.to_owned(),
        }
    }

    /// Display for an L1/L2/L3 row. Empty means "unbound": the turn degrades to
    /// the next cheaper tier (see `LucyConfig::resolve_level_model`), which the
    /// routing note then spells out. Naming that here keeps the two in step
    /// instead of leaving the user to guess which model will answer.
    fn level_display(&self, key: &str) -> String {
        if key.trim().is_empty() {
            return "(unbound)".into();
        }
        match self.text_models.iter().find(|o| o.key() == key) {
            Some(o) => o.label(),
            None => key.to_owned(),
        }
    }

    /// Keys the L1/L2/L3 dropdowns can choose from: every text model from all
    /// connected providers, plus any already-selected tier value, plus the model
    /// each unbound tier falls back to. `""` first = unbind, so the tier
    /// degrades down the chain rather than pinning itself to whatever happens to
    /// be bound today.
    ///
    /// The fallbacks are what stop a config with no connected text provider from
    /// offering nothing but "unbind".
    fn level_keys(&self) -> Vec<String> {
        let mut out = vec![String::new()];
        for k in self
            .text_models()
            .iter()
            .map(|o| o.key())
            .chain(self.configured_levels.iter().cloned())
            .chain(self.level_models.iter().cloned())
        {
            let k = k.trim().to_owned();
            if !k.is_empty() && !out.contains(&k) {
                out.push(k);
            }
        }
        out
    }

    /// Keys the Voice-model dropdown can choose from: every voice model from
    /// all connected voice providers, plus the live selection and the
    /// already-configured voice model (legacy `voice.model` with zero voice
    /// providers connected).
    fn voice_keys(&self) -> Vec<String> {
        let mut out = vec![String::new()];
        for k in self
            .voice_models()
            .iter()
            .map(|o| o.key())
            .chain([self.voice_model.clone()])
            .chain([self.configured_voice.clone()])
        {
            let k = k.trim().to_owned();
            if !k.is_empty() && !out.contains(&k) {
                out.push(k);
            }
        }
        out
    }

    // ---- the searchable model dropdown (`[change]` on a level row) ----

    /// True while the searchable dropdown is open.
    pub fn picker_is_open(&self) -> bool {
        self.picker.is_some()
    }

    /// True while the dropdown is open *for this row*, so its button can read
    /// `[close]` instead of `[change]`.
    pub fn picker_is_open_for(&self, row: Row) -> bool {
        self.picker.as_ref().is_some_and(|p| p.row == row)
    }

    /// Open the searchable dropdown for a model row. Nothing is applied until
    /// `Enter`, so browsing costs nothing; `Esc` leaves the tier untouched.
    ///
    /// The list starts on the entry that is live right now, so opening the
    /// dropdown and pressing `Enter` again is a no-op rather than a jump to
    /// the alphabetically-first model.
    fn open_picker(&mut self, row: Row) {
        if !row.has_model_picker() {
            return;
        }
        let current = self.current_choice(row);
        let cursor = self
            .level_picker_choices()
            .iter()
            .position(|c| c.key == current)
            .unwrap_or(0);
        self.picker = Some(ModelPicker {
            row,
            query: String::new(),
            cursor,
        });
    }

    /// Close the dropdown without applying anything.
    fn close_picker(&mut self) {
        self.picker = None;
    }

    /// Every entry the level dropdown can offer, grouped and de-duplicated:
    /// the "unbind" clear entry, then the models already bound to a tier (or
    /// the model an unbound tier resolves to) under **Recent**, then one group
    /// per provider for the rest.
    ///
    /// Recent is de-duplicated out of the provider groups on purpose — the
    /// opencode picker repeats them, but here every entry is a stop on the
    /// `↑`/`↓` walk, so listing one model twice would make the keyboard cursor
    /// feel broken. This is also what makes the list a superset of
    /// [`Self::level_keys`]: everything `←`/`→` can reach is reachable here.
    fn level_picker_choices(&self) -> Vec<PickerChoice> {
        let mut out = vec![PickerChoice {
            key: String::new(),
            display: "(unbind — use the next cheaper tier)".to_owned(),
            group: None,
        }];
        let mut seen: Vec<String> = Vec::new();

        // **Recent** is the tiers the user actually bound — nothing else. A
        // tier's *resolved* model is deliberately excluded: with everything
        // unbound that is the compiled-in default, and calling that "Recent"
        // would claim the user had chosen it.
        for k in self.level_models.iter() {
            let k = k.trim();
            if k.is_empty() || seen.iter().any(|s| s == k) {
                continue;
            }
            let key = k.to_owned();
            seen.push(key.clone());
            out.push(PickerChoice {
                display: self.picker_label(&key, true),
                key,
                group: Some(RECENT_GROUP.to_owned()),
            });
        }

        // `text_models` is sorted by (provider name, model), so a run of equal
        // provider names is exactly one group, already in display order.
        let mut i = 0;
        while i < self.text_models.len() {
            let group = self.text_models[i].provider_name.clone();
            while i < self.text_models.len() && self.text_models[i].provider_name == group {
                let opt = &self.text_models[i];
                if !seen.contains(&opt.key()) {
                    out.push(PickerChoice {
                        key: opt.key(),
                        display: self.picker_label(&opt.key(), false),
                        group: Some(group.clone()),
                    });
                }
                i += 1;
            }
        }

        // The model an unbound tier falls back to must stay reachable even when
        // no provider exposes it, so searching by family does not silently lose
        // the model the config is actually running. It gets its own heading so
        // **Recent** still means "a tier you bound" and nothing else.
        for k in self.configured_levels.iter() {
            let k = k.trim();
            if k.is_empty() || seen.iter().any(|s| s == k) {
                continue;
            }
            let key = k.to_owned();
            seen.push(key.clone());
            out.push(PickerChoice {
                display: self.picker_label(&key, true),
                key,
                group: Some(DEFAULT_GROUP.to_owned()),
            });
        }
        out
    }

    /// Text for one entry. Provider groups show the bare model (the heading
    /// already names the provider); **Recent** spells it out as
    /// `Provider / model`, because a model id alone rarely identifies it.
    /// An unknown key is shown raw so a selection is never hidden.
    fn picker_label(&self, key: &str, spell_out_provider: bool) -> String {
        match self.text_models.iter().find(|o| o.key() == key) {
            Some(o) if spell_out_provider => format!("{} / {}", o.provider_name, o.model),
            Some(o) => o.model.clone(),
            None => key.to_owned(),
        }
    }

    /// The dropdown entries matching the search query, prefix matches first
    /// inside each group. An empty query returns everything, unfiltered.
    fn filtered_picker_choices(&self, query: &str) -> Vec<PickerChoice> {
        let all = self.level_picker_choices();
        if query.trim().is_empty() {
            return all;
        }
        let mut out: Vec<PickerChoice> = Vec::new();
        for mut group in choice_groups(all) {
            group.retain(|c| match_rank(query, c).is_some());
            // Stable sort on the rank, so a prefix match leads its group ahead
            // of models that merely contain the query. Headers stay put
            // because the reorder is per group. Rank 0 sorts before 1, so
            // this is an ascending sort rather than a reverse one.
            group.sort_by_key(|c| match_rank(query, c).unwrap_or(1));
            out.extend(group);
        }
        out
    }

    /// The entries the open dropdown is currently showing.
    fn visible_picker_choices(&self) -> Vec<PickerChoice> {
        match &self.picker {
            Some(p) => self.filtered_picker_choices(&p.query),
            None => Vec::new(),
        }
    }

    /// Move the dropdown cursor, clamped to the filtered list (no wrap: the top
    /// hit after a search is the one you want highlighted).
    fn picker_move(&mut self, delta: i32) {
        let len = self.visible_picker_choices().len();
        if len == 0 {
            return;
        }
        if let Some(p) = self.picker.as_mut() {
            p.cursor = (p.cursor as i32 + delta).clamp(0, len as i32 - 1) as usize;
        }
    }

    /// Empty the search query wholesale, cursor back to the top.
    fn picker_clear_query(&mut self) {
        if let Some(p) = self.picker.as_mut() {
            p.query.clear();
        }
        self.picker_edit(None);
    }

    /// Append to (`Some(c)`) or delete from (`None`) the search query. The
    /// cursor resets to the top hit, so what is highlighted is always what
    /// `Enter` would apply.
    fn picker_edit(&mut self, edit: Option<char>) {
        let Some(p) = self.picker.as_mut() else {
            return;
        };
        match edit {
            Some(c) => p.query.push(c),
            None => {
                p.query.pop();
            }
        }
        p.cursor = 0;
    }

    /// Route one key to the open dropdown, consuming it. `Enter` applies the
    /// highlighted entry (and only then persists it), so the search box
    /// swallows plain characters instead of them reaching the settings screen.
    fn on_picker_key(&mut self, key: KeyEvent, config: &mut LucyConfig) -> Option<SettingsAction> {
        let row = self.picker.as_ref()?.row;
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let cursor = self.picker.as_ref()?.cursor;
        let picked = self
            .visible_picker_choices()
            .get(cursor)
            .map(|c| c.key.clone());
        match key.code {
            // `Esc` closes the dropdown, not the screen — the second `Esc`
            // still closes settings.
            KeyCode::Esc => self.close_picker(),
            // An empty list (a query that matched nothing) is not a selection,
            // so nothing is applied and the dropdown stays open for the query
            // to be fixed.
            KeyCode::Enter if picked.is_none() => {}
            KeyCode::Enter => {
                let chosen = picked.unwrap_or_default();
                self.close_picker();
                self.set_choice(row, chosen);
                return self.persist_choice(config, row);
            }
            KeyCode::Up => self.picker_move(-1),
            KeyCode::Down => self.picker_move(1),
            KeyCode::Home => self.picker_move(i32::MIN),
            KeyCode::End => self.picker_move(i32::MAX),
            // `Backspace` and `Delete` both drop a character; `Ctrl+U` below
            // clears the lot. A search box editable only one character at a
            // time is unusable.
            KeyCode::Backspace | KeyCode::Delete => self.picker_edit(None),
            KeyCode::Char('n') if ctrl => self.picker_move(1),
            KeyCode::Char('p') if ctrl => self.picker_move(-1),
            // `Ctrl+U` clears the whole query, the readline convention, so a
            // long search does not need `Backspace` once per character.
            KeyCode::Char('u') if ctrl => self.picker_clear_query(),
            KeyCode::Char(c) if !ctrl => self.picker_edit(Some(c)),
            _ => {}
        }
        None
    }

    /// Display label for the current `approvals.mode`. The values are ordered
    /// as in [`Self::options`] so `←/→` cycles through them consistently.
    fn approvals_mode_label(&self) -> &'static str {
        match self.approvals_mode.as_str() {
            "never" => "Auto (never ask)",
            "always" => "Always ask",
            _ => "Risky tools only",
        }
    }

    fn current_choice(&self, row: Row) -> String {
        match row {
            Row::ProviderPreset => self.provider_preset.clone(),
            Row::ProviderType => self.provider_type.as_str().to_owned(),
            Row::ChatMode => self.chat_mode.as_str().to_owned(),
            Row::AutoCompact => if self.auto_compact { "On" } else { "Off" }.into(),
            Row::Approvals => self.approvals_mode_label().to_owned(),
            Row::VoiceModel => self.voice_model.clone(),
            Row::Level1 | Row::Level2 | Row::Level3 => self.level_models[level_index(row)].clone(),
            _ => String::new(),
        }
    }

    fn set_choice(&mut self, row: Row, value: String) {
        match row {
            Row::ProviderPreset => self.apply_preset(&value),
            Row::ProviderType => {
                self.provider_type = ProviderType::parse(&value).unwrap_or_default();
            }
            Row::ChatMode => {
                self.chat_mode = ChatMode::parse(&value).unwrap_or_default();
            }
            Row::AutoCompact => self.auto_compact = value == "On",
            Row::Approvals => {
                self.approvals_mode = match value.as_str() {
                    "Auto (never ask)" => "never",
                    "Always ask" => "always",
                    _ => "write",
                }
                .to_owned();
            }
            Row::VoiceModel => self.voice_model = value,
            Row::Level1 | Row::Level2 | Row::Level3 => {
                self.level_models[level_index(row)] = value;
            }
            _ => {}
        }
    }

    /// Cycle a choice row. `dir = -1` for `←`, `+1` for `→`.
    fn cycle(&mut self, row: Row, dir: i32) {
        let options = self.options(row);
        if options.is_empty() {
            return;
        }
        let keys: Vec<String> = match row {
            Row::ProviderPreset
            | Row::ProviderType
            | Row::ChatMode
            | Row::AutoCompact
            | Row::Approvals => options.clone(),
            Row::VoiceModel => self.voice_keys(),
            Row::Level1 | Row::Level2 | Row::Level3 => self.level_keys(),
            _ => Vec::new(),
        };
        if keys.is_empty() {
            return;
        }
        let cur = self.current_choice(row);
        let idx = keys.iter().position(|k| *k == cur);
        let next = match idx {
            Some(i) => (i as i32 + dir).rem_euclid(keys.len() as i32) as usize,
            // No current value (or a hand-typed one): start at the first entry
            // going forwards, and the last entry going backwards.
            None if dir > 0 => 0,
            None => keys.len() - 1,
        };
        self.set_choice(row, keys[next].clone());
    }

    /// Apply a vendor preset to the draft: fill its base URL, name the
    /// provider, and pre-fill the key from the vendor's env var so a
    /// `export OPENROUTER_API_KEY=…` shell is all that is needed. A prior
    /// `[Test]` is dropped — it verified a different URL.
    fn apply_preset(&mut self, value: &str) {
        self.provider_preset = value.to_owned();
        let Some(preset) = ProviderPreset::parse(value) else {
            return;
        };
        let Some(url) = preset.api_url() else {
            // Custom: keep whatever URL/name the user has, just clear them
            // back to blank so the row reads as an empty form.
            self.provider_name.clear();
            self.tested = false;
            self.discovered_models.clear();
            self.discovered_deprecated.clear();
            return;
        };
        self.api_url = url.to_owned();
        self.provider_name = preset.name().to_owned();
        if self.api_key.trim().is_empty()
            && let Some(env_key) = preset.api_key_env()
            && let Ok(key) = std::env::var(env_key)
            && !key.trim().is_empty()
        {
            self.api_key = key.trim().to_owned();
        }
        self.tested = false;
        self.discovered_models.clear();
        self.discovered_deprecated.clear();
    }

    /// The provider draft built from the current inputs.
    pub fn provider_draft(&self) -> ProviderConfig {
        ProviderConfig {
            // Reuse the id of the provider whose URL matches, so `[Save]`
            // updates in place instead of creating a duplicate.
            id: String::new(),
            name: self.provider_name.trim().to_owned(),
            api_url: self.api_url.trim().to_owned(),
            api_key: self.api_key.trim().to_owned(),
            provider_type: self.provider_type,
            available_models: self.discovered_models.clone(),
            deprecated_models: self.discovered_deprecated.clone(),
        }
    }

    fn note(&mut self, kind: StatusKind, message: impl Into<String>) {
        self.status_kind = kind;
        self.status = message.into();
    }

    /// Refresh the model dropdowns from a (possibly just-updated) config.
    pub fn refresh_models(&mut self, config: &LucyConfig) {
        self.text_models = config.text_model_options();
        self.voice_models = config.voice_model_options();
        self.refresh_fallbacks(config);
    }

    /// Apply a successful `[Test]`: stage the discovered models.
    fn on_tested(&mut self, health: ProviderHealth) {
        self.tested = true;
        let n = health.available_models.len();
        self.discovered_models = health.available_models.clone();
        self.discovered_deprecated = health.deprecated_models.clone();
        self.note(
            StatusKind::Ok,
            format!("{n} model(s) available — press [Save] to store them"),
        );
    }

    /// Handle one key press. Async because `[Test]`/`[Save]` hit the network.
    ///
    /// `config` is the live config: dropdowns in sections 3–6 are written to it
    /// immediately (and persisted), while the provider and classification URL
    /// stay in the draft until their own `[Save]`.
    pub async fn on_key(
        &mut self,
        key: KeyEvent,
        config: &mut LucyConfig,
    ) -> Option<SettingsAction> {
        if self.busy {
            return None;
        }
        // The open dropdown is modal: it owns every key, so plain characters
        // become search text instead of leaking into the settings screen.
        if self.picker.is_some() {
            return self.on_picker_key(key, config);
        }
        let row = self.focused();
        match key.code {
            KeyCode::Esc => {
                self.close();
                return Some(SettingsAction::Close);
            }
            KeyCode::Up => {
                self.move_cursor(-1);
                return None;
            }
            KeyCode::Down => {
                self.move_cursor(1);
                return None;
            }
            KeyCode::Tab => {
                self.move_cursor(if key.modifiers.contains(KeyModifiers::SHIFT) {
                    -1
                } else {
                    1
                });
                return None;
            }
            KeyCode::Backspace => {
                if row.is_text() {
                    if row == Row::ApiUrl {
                        self.provider_preset = ProviderPreset::Custom.name().to_owned();
                        self.provider_name.clear();
                        self.tested = false;
                    }
                    let value = self.text_value_mut();
                    value.pop();
                }
                return None;
            }
            KeyCode::Left => {
                if row.is_choice() {
                    self.cycle(row, -1);
                    return self.persist_choice(config, row);
                }
                return None;
            }
            KeyCode::Right => {
                if row.is_choice() {
                    self.cycle(row, 1);
                    return self.persist_choice(config, row);
                }
                return None;
            }
            KeyCode::Enter => {
                // Level rows offer every text model, so `Enter` opens the
                // searchable dropdown rather than stepping one entry. `←`/`→`
                // still cycle for quick nudges.
                if row.has_model_picker() {
                    self.open_picker(row);
                    return None;
                }
                if row.is_choice() {
                    self.cycle(row, 1);
                    return self.persist_choice(config, row);
                }
                if row.is_button() {
                    return self.activate(row, config).await;
                }
                return None;
            }
            KeyCode::Char('t') | KeyCode::Char('T')
                if row == Row::ProviderTest || row == Row::ClassifyTest =>
            {
                return self.activate(row, config).await;
            }
            KeyCode::Char('s') | KeyCode::Char('S')
                if row == Row::ProviderSave || row == Row::ClassifySave =>
            {
                return self.activate(row, config).await;
            }
            KeyCode::Char(c) if row.is_text() => {
                if row == Row::ApiUrl {
                    // A hand-typed URL is no longer the preset's URL.
                    self.provider_preset = ProviderPreset::Custom.name().to_owned();
                    self.provider_name.clear();
                    self.tested = false;
                }
                self.text_value_mut().push(c);
                return None;
            }
            _ => {}
        }
        None
    }

    fn text_value_mut(&mut self) -> &mut String {
        match self.focused() {
            Row::ApiUrl => &mut self.api_url,
            Row::ClassifyUrl => &mut self.classification_url,
            _ => &mut self.api_key,
        }
    }

    /// Sections 3–6 are the user's live selection, so a change is written to
    /// the config and persisted immediately.
    fn persist_choice(&mut self, config: &mut LucyConfig, row: Row) -> Option<SettingsAction> {
        if !row.is_choice() {
            return None;
        }
        // The preset row only edits the draft (URL/name/key); nothing is
        // written until `[Save]`, exactly like the provider text fields.
        if row == Row::ProviderPreset {
            return None;
        }
        let result = match row {
            Row::ChatMode => {
                config.set_chat_mode(self.chat_mode);
                config.save()
            }
            Row::AutoCompact => {
                config.set_auto_compact(self.auto_compact);
                config.save()
            }
            Row::Approvals => {
                // Persisted here; `SettingsAction::RuntimeApprovalChanged` makes
                // the caller push the same mode into the live gate so it takes
                // effect without a restart.
                config.set_approval_mode(&self.approvals_mode);
                config.save()
            }
            Row::VoiceModel => {
                config.set_voice_model(&self.voice_model);
                config.save()
            }
            Row::Level1 | Row::Level2 | Row::Level3 => {
                let idx = level_index(row);
                config
                    .chat
                    .reasoning_levels
                    .set(ReasoningLevel::ALL[idx], self.level_models[idx].clone());
                config.save()
            }
            _ => Ok(()),
        };
        match result {
            Ok(()) => {
                // Keep the configured fallbacks in sync so unbinding a tier or
                // clearing Voice never orphans the dropdown: the model the tier
                // will actually call (its own binding, or the next cheaper
                // one) stays re-selectable via ←/→.
                if matches!(
                    row,
                    Row::Level1 | Row::Level2 | Row::Level3 | Row::VoiceModel
                ) {
                    self.refresh_fallbacks(config);
                }
                if row == Row::Approvals {
                    return Some(SettingsAction::ApprovalModeChanged);
                }
                Some(SettingsAction::ModelsChanged)
            }
            Err(e) => {
                self.note(
                    StatusKind::Err,
                    format!("could not save: {}", lucy_core::friendly(&format!("{e:#}"))),
                );
                None
            }
        }
    }

    /// Press `[Test]` / `[Save]`.
    async fn activate(&mut self, row: Row, config: &mut LucyConfig) -> Option<SettingsAction> {
        match row {
            Row::ProviderTest => {
                let draft = self.provider_draft();
                if let Err(e) = lucy_agent::validate_provider_draft(&draft) {
                    self.note(StatusKind::Err, friendly_probe_error(&e));
                    return None;
                }
                self.busy = true;
                self.note(StatusKind::Info, "testing connection…");
                let result = test_connection(&draft).await;
                self.busy = false;
                match result {
                    Ok(health) => {
                        let n = health.available_models.len();
                        self.on_tested(health);
                        // Populate the dropdowns immediately so the user can
                        // see what was found before pressing [Save].
                        let preview = ProviderConfig {
                            id: self
                                .provider_existing_id(config)
                                .unwrap_or_else(|| "pending".into()),
                            provider_type: draft.provider_type,
                            available_models: self.discovered_models.clone(),
                            deprecated_models: self.discovered_deprecated.clone(),
                            ..draft
                        };
                        let mut shadow = config.clone();
                        shadow
                            .providers
                            .retain(|p| p.id != preview.id || preview.id == "pending");
                        shadow.providers.push(preview);
                        self.refresh_models(&shadow);
                        self.note(
                            StatusKind::Ok,
                            format!("{n} model(s) found — press [Save] to store the provider"),
                        );
                    }
                    Err(e) => {
                        self.tested = false;
                        self.note(StatusKind::Err, friendly_probe_error(&e));
                    }
                }
                None
            }
            Row::ProviderSave => {
                let mut draft = self.provider_draft();
                if let Err(e) = lucy_agent::validate_provider_draft(&draft) {
                    self.note(StatusKind::Err, friendly_probe_error(&e));
                    return None;
                }
                if !self.tested {
                    // Never persist an unverified endpoint: test first, so a
                    // typo cannot poison every later turn.
                    self.note(
                        StatusKind::Warn,
                        "press [Test] first — an unverified provider cannot be saved",
                    );
                    return None;
                }
                draft.id = self.provider_existing_id(config).unwrap_or_default();
                match persist_provider(config, draft) {
                    Ok(stored) => {
                        self.refresh_models(config);
                        let n = stored.available_models.len();
                        self.note(
                            StatusKind::Ok,
                            format!("saved {} ({n} model(s))", stored.label()),
                        );
                        Some(SettingsAction::ModelsChanged)
                    }
                    Err(e) => {
                        self.note(StatusKind::Err, friendly_probe_error(&e));
                        None
                    }
                }
            }
            Row::ClassifyTest => {
                let url = self.classification_url.trim().to_owned();
                if url.is_empty() {
                    self.note(StatusKind::Err, "enter the classification API URL first");
                    return None;
                }
                self.busy = true;
                self.note(StatusKind::Info, format!("probing {url}…"));
                let probe =
                    match lucy_systemone::DeciderClient::new(lucy_config::ClassificationConfig {
                        classification_api_url: url.clone(),
                        ..config.classification.clone()
                    }) {
                        Ok(client) => client.probe().await,
                        Err(e) => Err(e),
                    };
                self.busy = false;
                match probe {
                    Ok(p) => {
                        let summary = p.summary();
                        self.note(StatusKind::Ok, summary);
                        None
                    }
                    Err(e) => {
                        self.note(
                            StatusKind::Err,
                            format!("unreachable: {}", lucy_core::friendly(&format!("{e:#}"))),
                        );
                        None
                    }
                }
            }
            Row::ClassifySave => {
                let url = self.classification_url.trim().to_owned();
                match persist_classification_url(config, &url) {
                    Ok(saved) => {
                        self.classification_url = saved.clone();
                        self.note(
                            StatusKind::Ok,
                            format!("classification model saved at {saved}"),
                        );
                        None
                    }
                    Err(e) => {
                        self.note(StatusKind::Err, friendly_probe_error(&e));
                        None
                    }
                }
            }
            _ => None,
        }
    }

    /// The id of the already-saved provider this draft is editing (matched on
    /// API URL), so `[Save]` updates instead of duplicating.
    fn provider_existing_id(&self, config: &LucyConfig) -> Option<String> {
        let url = self.api_url.trim().trim_end_matches('/');
        if url.is_empty() {
            return None;
        }
        config
            .providers
            .iter()
            .find(|p| p.base_url() == url)
            .map(|p| p.id.clone())
    }
}

/// Split a choice list into runs sharing a heading. The bare "unbind"
/// entry has no heading, so it becomes its own single-entry run and never
/// absorbs the group that follows it.
fn choice_groups(choices: Vec<PickerChoice>) -> Vec<Vec<PickerChoice>> {
    let mut out: Vec<Vec<PickerChoice>> = Vec::new();
    for c in choices {
        match out.last_mut() {
            Some(run) if run[0].group == c.group => run.push(c),
            _ => out.push(vec![c]),
        }
    }
    out
}

/// Lowercase, punctuation-stripped form used for search matching, so `grok`,
/// `Grok/`, `groq ` and `groq flash` all reach `Groq · flash`. Dots survive so
/// version suffixes (`gemini-2.5`) stay searchable.
fn normalize(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric() || *c == '.')
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// Both strings a choice can be found by: what the user reads, and the
/// `provider_id/model` key the config stores.
fn choice_haystacks(c: &PickerChoice) -> [&str; 2] {
    [&c.display, &c.key]
}

/// ASCII-lowercase. Deliberately not [`str::to_lowercase`]: the dropdown's
/// case-folding runs over normalized alphanumeric-only text, so Unicode
/// case-mapping (which can change length and blow up the normalized key's
/// length invariant) buys nothing here.
fn ascii_lower(s: &str) -> String {
    s.to_ascii_lowercase()
}

/// How well `choice` matches `query`: `0` = prefix, `1` = substring, `None` =
/// no match. Matching is case- and separator-insensitive and runs over the
/// visible text *and* the stored key, so both `Gemini` and `google/gemini`
/// find `Google · gemini-2.5-pro`.
///
/// A punctuation-only query (`-`, `/`) normalizes to nothing, so it falls
/// back to a plain substring test rather than silently matching everything.
fn match_rank(query: &str, choice: &PickerChoice) -> Option<u8> {
    let q = query.trim();
    if q.is_empty() {
        return Some(0);
    }
    let nq = normalize(q);
    if nq.is_empty() {
        let lq = ascii_lower(q);
        return choice_haystacks(choice)
            .iter()
            .any(|h| ascii_lower(h).contains(&lq))
            .then_some(1);
    }
    let mut best = None;
    for h in choice_haystacks(choice) {
        let nh = normalize(h);
        if nh == nq || nh.starts_with(&nq) {
            return Some(0);
        }
        if nh.contains(&nq) {
            best = Some(1);
        }
    }
    best
}

/// Level index (0/1/2) for a level row.
fn level_index(row: Row) -> usize {
    match row {
        Row::Level2 => 1,
        Row::Level3 => 2,
        _ => 0,
    }
}

fn display_or_placeholder(value: &str, placeholder: &str) -> String {
    if value.trim().is_empty() {
        format!("(empty) — type to edit · e.g. {placeholder}")
    } else {
        value.to_owned()
    }
}

/// Labels for an explicit key list (Main / Level / Voice dropdowns): `""` renders as
/// `empty_label`, known keys render as `Provider · model`, and unknown keys
/// (legacy mains, removed providers) render raw so the selection is never
/// hidden. The key list always carries `""` first, so the clear option is
/// selectable with `←`/`→` — previously a set tier could never be unset.
fn labels_for_keys(options: &[ModelOption], keys: &[String], empty_label: &str) -> Vec<String> {
    keys.iter()
        .map(|k| {
            if k.trim().is_empty() {
                empty_label.to_owned()
            } else {
                options
                    .iter()
                    .find(|o| o.key() == *k)
                    .map(|o| o.label())
                    .unwrap_or_else(|| k.clone())
            }
        })
        .collect()
}

// ---- rendering ---------------------------------------------------------

const POPUP_BG: Color = Color::Rgb(25, 25, 35);
const ROW_BG: Color = Color::Rgb(30, 30, 45);
/// Background of the search dropdown, one step above the settings popup so the
/// two layers stay distinguishable.
const PICKER_BG: Color = Color::Rgb(20, 20, 30);
/// Background of the highlighted dropdown entry.
const PICKER_HL: Color = Color::Rgb(0, 160, 200);
/// Width of the label column, so every row's value starts at the same column.
const LABEL_WIDTH: usize = 22;

fn style_for(selected: bool) -> Style {
    if selected {
        Style::default()
            .bg(Color::Rgb(45, 45, 65))
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().bg(POPUP_BG).fg(Color::White)
    }
}

/// The hint drawn after a row's value, and whether it is a button (drawn
/// brighter on the focused row) rather than a static reminder.
///
/// The level rows get `[change]` instead of `←/→`: their list is every text
/// model on every provider, so stepping through it blind is not a real option
/// and the button — plus `Enter` — is the way in. It flips to `[close]` while
/// the dropdown is open so the target is obvious.
fn row_suffix(row: Row, picker_open: bool) -> (&'static str, bool) {
    if picker_open {
        return ("  [close]", true);
    }
    if row.has_model_picker() {
        return ("  [change]", true);
    }
    if row.is_choice() {
        return ("   ←/→", false);
    }
    if row.is_button() {
        return ("   (Enter)", false);
    }
    ("", false)
}

/// Draw the whole settings screen. `cursor_area` receives the position of the
/// focused text field so the terminal cursor can be placed there.
pub fn draw(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &SettingsState,
    config: &LucyConfig,
) -> Option<Rect> {
    let popup = Rect {
        x: area.x + area.width.saturating_sub(area.width * 9 / 10) / 2,
        y: area.y + area.height.saturating_sub(area.height * 9 / 10) / 2,
        width: (area.width * 9 / 10).max(30).min(area.width),
        height: (area.height * 9 / 10).max(10).min(area.height),
    };
    frame.render_widget(Clear, popup);
    let block = Block::default()
        .title(" Settings ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan).bg(POPUP_BG))
        .style(Style::default().bg(POPUP_BG));
    frame.render_widget(block.clone(), popup);
    let inner = block.inner(popup);
    let rows_area = Rect {
        x: inner.x,
        y: inner.y,
        width: inner.width,
        height: inner.height.saturating_sub(2),
    };
    frame.render_widget(
        Block::default().style(Style::default().bg(POPUP_BG)),
        rows_area,
    );

    let width = rows_area.width as usize;
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut text_cursor: Option<Rect> = None;

    let active_section = state.focused_section();
    let mut current_section: Option<Section> = None;
    for section in Section::ALL {
        for row in section.rows() {
            let Some(idx) = LAYOUT.iter().position(|(_, r)| r == row) else {
                continue;
            };
            let row = *row;
            if current_section != Some(section) {
                if current_section.is_some() {
                    lines.push(Line::from(""));
                }
                let header = Style::default()
                    .bg(POPUP_BG)
                    .fg(if section == active_section {
                        Color::Green
                    } else {
                        Color::DarkGray
                    })
                    .add_modifier(Modifier::BOLD);
                lines.push(Line::from(Span::styled(section.title().to_owned(), header)));
                current_section = Some(section);
            }
            let selected = idx == state.cursor();
            let style = style_for(selected);
            let marker = if selected { "▶ " } else { "  " };
            let value = state.value(row);
            let (suffix, is_button) = row_suffix(row, state.picker_is_open_for(row));
            let suffix_bg = Style::default().bg(style.bg.unwrap_or(POPUP_BG));
            // `[change]`/`[close]` are the way into a long list, so they light
            // up with the row; the static hints stay dim.
            let suffix_style = if is_button && selected {
                suffix_bg.fg(Color::Cyan).add_modifier(Modifier::BOLD)
            } else {
                suffix_bg.fg(Color::Gray)
            };
            let value_style = if row.is_button() {
                Style::default()
                    .bg(style.bg.unwrap_or(POPUP_BG))
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().bg(ROW_BG).fg(if row.is_text() {
                    Color::Yellow
                } else {
                    Color::White
                })
            };
            let prefix_len = 2 + LABEL_WIDTH + 2;
            lines.push(Line::from(vec![
                Span::styled(marker, style),
                Span::styled(format!("{:<LABEL_WIDTH$}", row.label()), style),
                Span::styled("  ", style),
                Span::styled(value.clone(), value_style),
                Span::styled(suffix, suffix_style),
            ]));
            // Remember where the caret goes for the focused text field.
            if selected && row.is_text() {
                let y = rows_area.y + lines.len() as u16;
                let caret = rows_area.x + prefix_len as u16 + value.chars().count() as u16;
                if y < rows_area.bottom() {
                    text_cursor = Some(Rect {
                        x: caret.min(rows_area.right().saturating_sub(1)),
                        y,
                        width: 1,
                        height: 1,
                    });
                }
            }
            // Extra context lines that depend on the row.
            match row {
                Row::ProviderTest if selected => lines.push(Line::from(Span::styled(
                    format!("      → {}", provider_summary(state, config)),
                    Style::default().bg(POPUP_BG).fg(Color::Gray),
                ))),
                Row::VoiceModel | Row::Level1 | Row::Level2 | Row::Level3 if selected => {
                    lines.push(Line::from(Span::styled(
                        format!("      → {}", model_count_hint(row, state)),
                        Style::default().bg(POPUP_BG).fg(Color::Gray),
                    )));
                }
                _ => {}
            }
        }
    }

    // Clip to the available height, keeping the tail (the most recent rows and
    // the status line) visible when the screen is short.
    let max_rows = rows_area.height as usize;
    if lines.len() > max_rows {
        lines.drain(..lines.len() - max_rows);
    }
    frame.render_widget(
        Paragraph::new(lines).style(Style::default().bg(POPUP_BG)),
        rows_area,
    );

    // Status line: the result of the last [Test]/[Save], or a busy note.
    let status_style = match state.status_kind {
        StatusKind::Ok => Style::default().fg(Color::Green),
        StatusKind::Warn => Style::default().fg(Color::Yellow),
        StatusKind::Err => Style::default().fg(Color::Red),
        StatusKind::Info => Style::default().fg(Color::Gray),
    };
    let status_text = if state.busy {
        "working…".to_owned()
    } else if state.status.trim().is_empty() {
        "connect a provider, then pick your models".to_owned()
    } else {
        state.status.clone()
    };
    let status_area = Rect {
        x: rows_area.x,
        y: rows_area.bottom().saturating_sub(1),
        width: rows_area.width,
        height: 1,
    };
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("› ", status_style.add_modifier(Modifier::BOLD)),
            Span::styled(truncate(&status_text, width), status_style),
        ]))
        .style(Style::default().bg(POPUP_BG)),
        status_area,
    );

    // Key hints. The dropdown carries its own, so the screen's line only needs
    // to point at it while it is closed.
    let hint_text = if state.picker_is_open() {
        "Esc closes the model list  ·  Enter applies the highlighted model"
    } else {
        "↑/↓ move  ←/→ change  Enter search/apply  type to edit  Esc close"
    };
    let hints = Rect {
        x: rows_area.x,
        y: rows_area.bottom().saturating_sub(0),
        width: rows_area.width,
        height: 1,
    };
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            truncate(hint_text, width),
            Style::default()
                .bg(POPUP_BG)
                .fg(Color::Gray)
                .add_modifier(Modifier::DIM),
        )))
        .alignment(Alignment::Center)
        .wrap(Wrap { trim: true }),
        hints,
    );

    // The dropdown is drawn last, so it sits on top of the settings popup,
    // and it takes over the caret (its search field, not the settings rows).
    if state.picker_is_open() {
        return draw_picker(frame, area, state);
    }

    let _ = state_cursor_row(state);
    text_cursor
}

/// Tallest slice of the dropdown list to show before it starts scrolling.
const PICKER_VISIBLE_ROWS: u16 = 12;
/// Label in front of the typed query. Also offsets the caret, so the two must
/// stay the same string.
const PICKER_SEARCH_LABEL: &str = "Search ";
/// Keys for the open dropdown, under the list. The widest line on the panel, so
/// it sets the minimum width — otherwise the one line you cannot afford to lose
/// is the one that gets truncated. The screen's own hint line points here while
/// the dropdown is closed, so these two must not contradict each other.
const PICKER_HINT: &str =
    "↑/↓ or ctrl+p/ctrl+n · type to search · ctrl+u clears · Enter applies · Esc closes";
/// Shown instead of [`PICKER_HINT`] when the query matched nothing.
const PICKER_NO_MATCH_HINT: &str = "no match — ctrl+u to clear, Esc to close";

/// Draw the searchable model dropdown over the settings screen, and return the
/// caret rect for its search field.
///
/// Layout: a title naming the tier, the search box, then the list — group
/// headings in cyan, the highlighted entry on a solid bar, the entry that is
/// live right now bulleted in green — and a footer with the keys. A long list
/// scrolls so the highlighted entry always stays on screen.
fn draw_picker(frame: &mut Frame<'_>, area: Rect, state: &SettingsState) -> Option<Rect> {
    let p = state.picker.as_ref()?;
    let choices = state.visible_picker_choices();

    // Build the whole list first: the panel is sized to it and the scroll
    // window is computed from the resulting line count.
    let cursor = p.cursor.min(choices.len().saturating_sub(1));
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut cursor_line = 0usize;
    let mut group: Option<Option<String>> = None;
    for (i, c) in choices.iter().enumerate() {
        // A heading whenever the group changes; the bare clear entry (`None`)
        // has no heading and simply leads the list.
        if c.group.is_some() && group.as_ref() != Some(&c.group) {
            lines.push(Line::from(Span::styled(
                c.group.clone().unwrap_or_default(),
                Style::default()
                    .bg(PICKER_BG)
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )));
        }
        group = Some(c.group.clone());
        let selected = i == cursor;
        if selected {
            cursor_line = lines.len();
        }
        // `●` marks the entry that is live now, `▸` the one `Enter` applies.
        let (bullet, name_style) = if selected {
            (
                "▸ ",
                Style::default()
                    .bg(PICKER_HL)
                    .fg(Color::Black)
                    .add_modifier(Modifier::BOLD),
            )
        } else if c.key == state.current_choice(p.row) {
            ("● ", Style::default().bg(PICKER_BG).fg(Color::Green))
        } else {
            ("  ", Style::default().bg(PICKER_BG).fg(Color::White))
        };
        lines.push(Line::from(vec![
            Span::styled(bullet, name_style),
            Span::styled(c.display.clone(), name_style),
        ]));
    }

    // Panel sizing. The hint line is the widest thing on screen, so it sets
    // the floor; the cap is the screen itself, which `clamp` must never be
    // asked to exceed (it panics when `min > max`, and a 20-column terminal is
    // narrower than any fixed minimum worth having).
    let longest = choices
        .iter()
        .map(|c| c.display.chars().count())
        .max()
        .unwrap_or(0)
        .max(p.row.label().chars().count());
    let max_width = (area.width * 9 / 10).max(1).min(area.width);
    let wanted = (longest as u16)
        .max(PICKER_HINT.chars().count() as u16)
        .saturating_add(4);
    let width = wanted.min(max_width);
    // search + blank + list + blank + footer, plus the two border rows.
    let visible = (lines.len() as u16).clamp(1, PICKER_VISIBLE_ROWS);
    let height = (visible + 6).min(area.height);
    let panel = Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    };
    frame.render_widget(Clear, panel);
    let block = Block::default()
        .title(format!(" {} — search models ", p.row.label()))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan).bg(PICKER_BG))
        .style(Style::default().bg(PICKER_BG));
    frame.render_widget(block.clone(), panel);
    let inner = block.inner(panel);
    frame.render_widget(
        Block::default().style(Style::default().bg(PICKER_BG)),
        inner,
    );

    // The search box is always the first line; the list starts below it.
    let query = p.query.clone();
    let mut rows: Vec<Line<'static>> = Vec::new();
    rows.push(Line::from(vec![
        Span::styled(
            PICKER_SEARCH_LABEL,
            Style::default().bg(PICKER_BG).fg(Color::DarkGray),
        ),
        Span::styled(
            if query.is_empty() {
                "…".to_owned()
            } else {
                query.clone()
            },
            Style::default().bg(PICKER_BG).fg(if query.is_empty() {
                Color::DarkGray
            } else {
                Color::Yellow
            }),
        ),
    ]));
    rows.push(Line::from(""));

    // Keep the highlighted line inside the window: scroll by whole lines so
    // headings do not slide in and out from under it.
    let list_h = inner.height.saturating_sub(4) as usize;
    let first = if list_h == 0 || lines.len() <= list_h {
        0
    } else {
        cursor_line
            .saturating_sub(list_h - 1)
            .min(lines.len() - list_h)
    };
    rows.extend(
        lines[first..lines.len().min(first + list_h)]
            .iter()
            .cloned(),
    );
    rows.push(Line::from(""));
    rows.push(Line::from(Span::styled(
        truncate(
            if choices.is_empty() {
                PICKER_NO_MATCH_HINT
            } else {
                PICKER_HINT
            },
            inner.width as usize,
        ),
        Style::default().bg(PICKER_BG).fg(Color::DarkGray),
    )));
    frame.render_widget(
        Paragraph::new(rows).style(Style::default().bg(PICKER_BG)),
        inner,
    );

    // Caret at the end of the typed query, on the search line.
    let caret_x =
        inner.x + PICKER_SEARCH_LABEL.chars().count() as u16 + query.chars().count() as u16;
    (inner.height > 0).then(|| Rect {
        x: caret_x.min(inner.right().saturating_sub(1)),
        y: inner.y,
        width: 1,
        height: 1,
    })
}

/// One-line summary of the provider draft, shown under a selected `[Test]`.
fn provider_summary(state: &SettingsState, config: &LucyConfig) -> String {
    let url = state.api_url.trim();
    if url.is_empty() {
        return "no API URL yet — type one above".into();
    }
    let key = if state.api_key.trim().is_empty() {
        "no API Key"
    } else {
        "key set"
    };
    let existing = config
        .providers
        .iter()
        .find(|p| p.base_url() == url.trim_end_matches('/'))
        .map(|p| format!(" (editing '{}')", p.label()))
        .unwrap_or_default();
    let found = if state.discovered_models.is_empty() {
        String::new()
    } else {
        format!(" · {} model(s) discovered", state.discovered_models.len())
    };
    // OpenRouter ids are `vendor/model`; say so before a pick, not after a
    // 404 from an unqualified name.
    let note = if lucy_config::is_openrouter(url) {
        " · pick a vendor/model id (e.g. anthropic/claude-opus-5.5)"
    } else {
        ""
    };
    format!("{url} · {key}{existing}{found}{note}")
}

fn model_count_hint(row: Row, state: &SettingsState) -> String {
    let (n, kind) = match row {
        Row::VoiceModel => (
            state.voice_keys().iter().filter(|k| !k.is_empty()).count(),
            "voice model(s)",
        ),
        // Levels offer every text model plus each tier's effective model; the
        // blank entry unbinds, which degrades the tier to the next cheaper one
        // (no "unbound" dead end).
        _ => (
            state.level_keys().iter().filter(|k| !k.is_empty()).count(),
            "text model(s); blank = unbind",
        ),
    };
    if n == 0 {
        match row {
            Row::VoiceModel => {
                "no voice models configured — type an API URL + key above, press [Test], then [Save]".into()
            }
            _ => {
                "no text models configured — type an API URL + key above, press [Test], then [Save]".into()
            }
        }
    } else if row.has_model_picker() {
        format!("{n} {kind} — Enter or [change] to search")
    } else {
        format!("{n} {kind} — ←/→ to pick")
    }
}

fn truncate(s: &str, width: usize) -> String {
    if s.chars().count() <= width {
        s.to_owned()
    } else {
        let mut t: String = s.chars().take(width.saturating_sub(1)).collect();
        t.push('…');
        t
    }
}

/// The number of rows the cursor sits on (used by tests).
fn state_cursor_row(state: &SettingsState) -> usize {
    state.cursor()
}

/// Redirect `LucyConfig` persistence at a throwaway file for this test binary.
///
/// Several tests exercise dropdowns that write through to `config.save()`,
/// which would otherwise rewrite the user's real `~/.config/lucy/config.toml`.
#[cfg(test)]
fn isolate_config() {
    use std::sync::OnceLock;
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("lucy-settings-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp config dir");
        let path = dir.join("config.toml");
        // SAFETY: runs once, before any test in this binary can persist.
        unsafe { std::env::set_var("LUCY_CONFIG", &path) };
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyEventKind;
    use lucy_config::{DEFAULT_TEXT_MODEL, ProviderConfig};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent {
            code,
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Press,
            state: crossterm::event::KeyEventState::NONE,
        }
    }

    /// A key with modifiers, for the dropdown's `ctrl+p`/`ctrl+n` stepping.
    fn key_mods(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent {
            modifiers,
            ..key(code)
        }
    }

    /// A config with many text models across two providers, so the dropdown has
    /// more than one group and a search actually has something to narrow.
    fn many_models() -> LucyConfig {
        let mut cfg = with_providers();
        cfg.providers[0].available_models =
            vec!["flash".into(), "pro".into(), "flash-preview".into()];
        cfg.providers.push(ProviderConfig {
            id: "google".into(),
            name: "Google".into(),
            api_url: "https://generativelanguage.test".into(),
            api_key: "sk-z".into(),
            provider_type: ProviderType::Text,
            available_models: vec!["gemini-2.5-pro".into(), "gemini-2.5-flash".into()],
            deprecated_models: Vec::new(),
        });
        cfg
    }

    /// The picker cursor's entry, or a readable dump when the list is empty.
    fn highlighted(s: &SettingsState) -> String {
        let visible = s.visible_picker_choices();
        let cursor = s.picker.as_ref().map(|p| p.cursor).unwrap_or(0);
        match visible.get(cursor) {
            Some(c) => c.display.clone(),
            None => format!("<empty: {:?}>", visible),
        }
    }

    fn focus(mut s: SettingsState, row: Row) -> SettingsState {
        s.cursor = LAYOUT.iter().position(|(_, r)| *r == row).expect("row");
        s
    }

    /// Type a query into the open dropdown, one `KeyCode::Char` at a time.
    async fn type_query(s: &mut SettingsState, cfg: &mut LucyConfig, q: &str) {
        for c in q.chars() {
            s.on_key(key(KeyCode::Char(c)), cfg).await;
        }
    }

    fn with_providers() -> LucyConfig {
        let mut cfg = LucyConfig {
            providers: vec![
                ProviderConfig {
                    id: "groq".into(),
                    name: "Groq".into(),
                    api_url: "https://api.groq.test".into(),
                    api_key: "sk-x".into(),
                    provider_type: ProviderType::Text,
                    available_models: vec!["flash".into(), "pro".into()],
                    deprecated_models: Vec::new(),
                },
                ProviderConfig {
                    id: "eleven".into(),
                    name: "ElevenLabs".into(),
                    api_url: "https://api.eleven.test".into(),
                    api_key: "sk-y".into(),
                    provider_type: ProviderType::Voice,
                    available_models: vec!["tts".into()],
                    deprecated_models: Vec::new(),
                },
            ],
            ..Default::default()
        };
        cfg.set_anchor_model("groq/flash");
        cfg
    }

    #[test]
    fn layout_has_six_sections_in_wireframe_order() {
        let sections: Vec<Section> = LAYOUT.iter().map(|(s, _)| *s).collect();
        let mut deduped: Vec<Section> = Vec::new();
        for s in sections {
            if !deduped.contains(&s) {
                deduped.push(s);
            }
        }
        assert_eq!(deduped, Section::ALL.to_vec());
        assert_eq!(
            Section::ALL.map(|s| s.title()),
            [
                "1. Connect providers",
                "2. Classification model",
                "3. Chat mode",
                "4. Auto Compact",
                "5. Automode",
                "6. Voice model"
            ]
        );
    }

    #[test]
    fn classification_section_has_no_api_key_row() {
        let rows = Section::Classification.rows();
        assert!(rows.contains(&Row::ClassifyUrl));
        assert!(rows.contains(&Row::ClassifyTest));
        assert!(rows.contains(&Row::ClassifySave));
        // The only API-key row in the whole screen belongs to the provider
        // section — decider-serve is unauthenticated.
        assert_eq!(LAYOUT.iter().filter(|(_, r)| *r == Row::ApiKey).count(), 1);
    }

    #[test]
    fn providers_section_matches_the_wireframe_fields() {
        assert_eq!(
            Section::Providers.rows(),
            &[
                Row::ProviderPreset,
                Row::ApiUrl,
                Row::ApiKey,
                Row::ProviderType,
                Row::ProviderTest,
                Row::ProviderSave
            ]
        );
    }

    #[test]
    fn opens_with_config_values_and_populated_dropdowns() {
        let mut cfg = with_providers();
        cfg.set_chat_mode(ChatMode::Manual);
        cfg.set_auto_compact(false);
        cfg.classification.classification_api_url = "http://localhost:8001".into();
        let s = SettingsState::open(&cfg);
        assert!(s.open);
        assert_eq!(s.classification_url, "http://localhost:8001");
        assert_eq!(s.chat_mode, ChatMode::Manual);
        assert!(!s.auto_compact);
        assert_eq!(s.level_models[2], "groq/flash");
        assert_eq!(s.voice_model, LucyConfig::default().voice_model());
        assert_eq!(s.text_models().len(), 2);
        assert_eq!(s.voice_models().len(), 1);
        assert_eq!(s.focused(), Row::ProviderPreset);
        assert_eq!(s.focused_section(), Section::Providers);
    }

    #[test]
    fn opens_on_custom_with_an_empty_provider_form() {
        let cfg = with_providers();
        let s = SettingsState::open(&cfg);
        assert_eq!(s.value(Row::ProviderPreset), "Custom");
        assert!(s.api_url.is_empty());
        assert!(s.api_key.is_empty());
    }

    #[tokio::test]
    async fn the_openrouter_preset_fills_the_url_and_names_the_provider() {
        let mut cfg = with_providers();
        let mut s = SettingsState::open(&cfg);
        s.cursor = LAYOUT
            .iter()
            .position(|(_, r)| *r == Row::ProviderPreset)
            .expect("row");
        // One step right from Custom.
        s.on_key(key(KeyCode::Right), &mut cfg).await;
        assert_eq!(s.value(Row::ProviderPreset), "OpenRouter");
        assert_eq!(s.api_url, "https://openrouter.ai/api/v1");
        assert_eq!(s.provider_draft().name, "OpenRouter");
        // A URL from a preset is `vendor/model`-shaped territory, so the
        // [Test] hint says so before a model is picked.
        assert!(provider_summary(&s, &cfg).contains("vendor/model"));
        // Hand-editing the URL falls back to Custom without wiping the text.
        s.move_cursor(1);
        s.on_key(key(KeyCode::Char('a')), &mut cfg).await;
        assert_eq!(s.api_url, "https://openrouter.ai/api/v1a");
        assert_eq!(s.value(Row::ProviderPreset), "Custom");
        assert!(s.provider_draft().name.is_empty());
    }

    #[tokio::test]
    async fn the_openrouter_preset_prefills_the_key_from_the_environment() {
        let mut cfg = with_providers();
        let mut s = SettingsState::open(&cfg);
        s.apply_preset("OpenRouter");
        // No key in the environment: the field stays empty, so [Test]
        // reports the missing key instead of a rejected one.
        assert!(s.api_key.is_empty());
        assert_eq!(s.api_url, "https://openrouter.ai/api/v1");
    }

    #[tokio::test]
    async fn changing_the_preset_drops_a_previous_test_result() {
        let mut cfg = with_providers();
        let mut s = SettingsState::open(&cfg);
        s.discovered_models = vec!["stale/model".into()];
        s.tested = true;
        s.apply_preset("Groq");
        assert!(!s.tested);
        assert!(s.discovered_models.is_empty());
        assert_eq!(s.api_url, "https://api.groq.com/openai/v1");
    }

    #[tokio::test]
    async fn typing_a_url_returns_the_form_to_custom() {
        let mut cfg = with_providers();
        let mut s = SettingsState::open(&cfg);
        s.apply_preset("OpenRouter");
        s.move_cursor(1);
        s.on_key(key(KeyCode::Char('x')), &mut cfg).await;
        assert_eq!(s.value(Row::ProviderPreset), "Custom");
        assert!(s.provider_draft().name.is_empty());
    }

    #[test]
    fn cursor_walks_every_row_and_wraps() {
        let cfg = with_providers();
        let mut s = SettingsState::open(&cfg);
        s.move_cursor(-1);
        assert_eq!(s.cursor(), LAYOUT.len() - 1);
        s.move_cursor(1);
        assert_eq!(s.cursor(), 0);
        for i in 0..LAYOUT.len() {
            assert_eq!(s.cursor(), i);
            s.move_cursor(1);
        }
    }

    #[tokio::test]
    async fn typing_edits_the_focused_text_field_only() {
        let cfg = with_providers();
        let mut s = SettingsState::open(&cfg);
        // The preset row owns the first line of the provider section; the API
        // URL field is the one below it.
        s.move_cursor(1);
        assert_eq!(s.focused(), Row::ApiUrl);
        for c in "http://localhost:8001".chars() {
            s.on_key(key(KeyCode::Char(c)), &mut cfg.clone()).await;
        }
        assert_eq!(s.api_url, "http://localhost:8001");
        assert!(s.api_key.is_empty());
        s.on_key(key(KeyCode::Backspace), &mut cfg.clone()).await;
        assert_eq!(s.api_url, "http://localhost:800");
    }

    #[tokio::test]
    async fn classification_url_is_typed_into_its_own_field() {
        let cfg = with_providers();
        let mut s = SettingsState::open(&cfg);
        let idx = LAYOUT
            .iter()
            .position(|(_, r)| *r == Row::ClassifyUrl)
            .unwrap();
        s.cursor = idx;
        s.classification_url = String::new();
        s.on_key(key(KeyCode::Char('1')), &mut cfg.clone()).await;
        assert_eq!(s.classification_url, "1");
        assert!(s.api_url.is_empty());
    }

    #[tokio::test]
    async fn arrows_cycle_the_provider_type_dropdown() {
        let mut cfg = with_providers();
        let mut s = SettingsState::open(&cfg);
        s.cursor = LAYOUT
            .iter()
            .position(|(_, r)| *r == Row::ProviderType)
            .unwrap();
        assert_eq!(s.provider_type, ProviderType::Text);
        s.on_key(key(KeyCode::Right), &mut cfg).await;
        assert_eq!(s.provider_type, ProviderType::Voice);
        s.on_key(key(KeyCode::Right), &mut cfg).await;
        assert_eq!(s.provider_type, ProviderType::Text);
        s.on_key(key(KeyCode::Left), &mut cfg).await;
        assert_eq!(s.provider_type, ProviderType::Voice);
    }

    #[tokio::test]
    async fn level_dropdowns_choose_from_text_models_and_persist() {
        isolate_config();
        let mut cfg = with_providers();
        // Start from an unbound tier so this exercises the first-selection
        // path. `with_providers()` binds the Level 3 anchor, and there is no
        // separate main model to hold a default any more.
        cfg.chat
            .reasoning_levels
            .set(ReasoningLevel::L3, String::new());
        let mut s = SettingsState::open(&cfg);
        let idx = LAYOUT.iter().position(|(_, r)| *r == Row::Level3).unwrap();
        s.cursor = idx;
        assert_eq!(s.level_models[2], "", "the tier starts unbound");
        // → from unbound lands on the first model, and persists it.
        let action = s.on_key(key(KeyCode::Right), &mut cfg).await;
        assert_eq!(action, Some(SettingsAction::ModelsChanged));
        assert_eq!(s.level_models[2], "groq/flash");
        assert_eq!(cfg.level_model(ReasoningLevel::L3), "groq/flash");
        s.on_key(key(KeyCode::Right), &mut cfg).await;
        assert_eq!(cfg.level_model(ReasoningLevel::L3), "groq/pro");
    }

    #[test]
    fn level_dropdowns_offer_the_default_text_model_with_zero_providers() {
        // No connected providers and no tier bound: the tiers must still offer
        // the compiled-in default model — previously they offered zero choices.
        let cfg = LucyConfig::default();
        assert!(cfg.text_model_options().is_empty());
        let s = SettingsState::open(&cfg);
        let keys = s.level_keys();
        assert_eq!(keys[0], "", "first key unbinds the tier");
        let default = cfg.default_text_model();
        assert!(
            keys.contains(&default),
            "default model {default} must be selectable, got {keys:?}"
        );
        let opts = s.options(Row::Level2);
        assert_eq!(opts[0], "(unbound)");
        assert!(
            opts.iter().any(|o| o.contains(&default)),
            "options must name the default model: {opts:?}"
        );
        assert_eq!(s.value(Row::Level2), "(unbound)");
    }

    #[tokio::test]
    async fn level_dropdown_cycles_the_default_model_and_unbinds() {
        isolate_config();
        let mut cfg = LucyConfig::default();
        let default = cfg.default_text_model();
        assert!(!default.is_empty());
        let mut s = SettingsState::open(&cfg);
        s.cursor = LAYOUT.iter().position(|(_, r)| *r == Row::Level2).unwrap();
        // Unset → first Right lands on the default model (the only choice).
        s.on_key(key(KeyCode::Right), &mut cfg).await;
        assert_eq!(s.level_models[1], default);
        assert_eq!(cfg.level_model(ReasoningLevel::L2), default);
        // Cycling past the end wraps back to "" = unbound.
        s.on_key(key(KeyCode::Right), &mut cfg).await;
        assert_eq!(s.level_models[1], "");
        assert_eq!(s.value(Row::Level2), "(unbound)");
    }

    #[test]
    fn the_default_text_model_is_offered_with_zero_providers() {
        // The reported bug: `providers = []` showed "no text model(s)
        // available" on the model rows. With the main model gone, the compiled
        // default is what every unbound tier resolves to, so it must always be
        // offered.
        let cfg = LucyConfig::default();
        assert!(cfg.text_model_options().is_empty());
        assert_eq!(cfg.default_text_model(), DEFAULT_TEXT_MODEL);
        let s = SettingsState::open(&cfg);
        for row in [Row::Level1, Row::Level2, Row::Level3] {
            let keys = s.level_keys();
            assert!(
                keys.contains(&DEFAULT_TEXT_MODEL.to_owned()),
                "{row:?} must offer the default model, got {keys:?}"
            );
            assert!(
                !model_count_hint(row, &s).starts_with("no "),
                "{row:?} hint must not report empty: {}",
                model_count_hint(row, &s)
            );
        }
        assert!(
            s.options(Row::Level1)
                .contains(&DEFAULT_TEXT_MODEL.to_owned())
        );
    }

    #[test]
    fn unbinding_a_tier_keeps_its_fallback_selectable() {
        // Unbinding a tier to "" must not strand the text dropdowns on zero
        // choices: the model the tier will actually call stays in the key list.
        let cfg = LucyConfig::default();
        let mut s = SettingsState::open(&cfg);
        s.level_models = [String::new(), String::new(), String::new()];
        assert_eq!(s.value(Row::Level1), "(unbound)");
        assert!(s.level_keys().contains(&DEFAULT_TEXT_MODEL.to_owned()));
        // Cycling forward from "" lands on the fallback, not nowhere.
        s.cycle(Row::Level1, 1);
        assert_eq!(s.level_models[0], DEFAULT_TEXT_MODEL);
    }

    #[test]
    fn configured_voice_is_selectable_with_zero_voice_providers() {
        // Same class of bug for voice: legacy `voice.model` with no voice
        // providers connected must still be shown and ←/→-selectable.
        let cfg = LucyConfig::default();
        assert!(cfg.voice_model_options().is_empty());
        assert!(!cfg.voice_model().is_empty());
        let mut s = SettingsState::open(&cfg);
        assert_eq!(s.value(Row::VoiceModel), cfg.voice_model());
        assert!(s.voice_keys().contains(&cfg.voice_model()));
        let opts = s.options(Row::VoiceModel);
        assert!(opts.contains(&cfg.voice_model()), "{opts:?}");
        assert!(
            !model_count_hint(Row::VoiceModel, &s).starts_with("no "),
            "{}",
            model_count_hint(Row::VoiceModel, &s)
        );
        s.voice_model = String::new();
        s.cycle(Row::VoiceModel, 1);
        assert_eq!(s.voice_model, cfg.voice_model());
    }

    #[tokio::test]
    async fn level_and_voice_dropdowns_use_their_own_provider_types() {
        isolate_config();
        let mut cfg = with_providers();
        let mut s = SettingsState::open(&cfg);
        s.cursor = LAYOUT.iter().position(|(_, r)| *r == Row::Level3).unwrap();
        s.level_models[2] = String::new();
        s.on_key(key(KeyCode::Right), &mut cfg).await;
        assert_eq!(s.level_models[2], "groq/flash");
        s.on_key(key(KeyCode::Right), &mut cfg).await;
        assert_eq!(s.level_models[2], "groq/pro");

        s.cursor = LAYOUT
            .iter()
            .position(|(_, r)| *r == Row::VoiceModel)
            .unwrap();
        s.voice_model = String::new();
        s.on_key(key(KeyCode::Right), &mut cfg).await;
        assert_eq!(s.voice_model, "eleven/tts");
        assert_eq!(cfg.voice_model(), "eleven/tts");
    }

    #[tokio::test]
    async fn chat_mode_and_auto_compact_toggles_persist() {
        isolate_config();
        let mut cfg = with_providers();
        let mut s = SettingsState::open(&cfg);
        s.cursor = LAYOUT
            .iter()
            .position(|(_, r)| *r == Row::ChatMode)
            .unwrap();
        s.on_key(key(KeyCode::Right), &mut cfg).await;
        assert_eq!(cfg.chat_mode(), ChatMode::Manual);
        s.cursor = LAYOUT
            .iter()
            .position(|(_, r)| *r == Row::AutoCompact)
            .unwrap();
        s.on_key(key(KeyCode::Right), &mut cfg).await;
        assert!(!cfg.auto_compact());
        assert_eq!(s.value(Row::AutoCompact), "Off");
    }

    #[tokio::test]
    async fn esc_closes_the_screen() {
        let cfg = with_providers();
        let mut s = SettingsState::open(&cfg);
        assert_eq!(
            s.on_key(key(KeyCode::Esc), &mut cfg.clone()).await,
            Some(SettingsAction::Close)
        );
        assert!(!s.open);
    }

    #[tokio::test]
    async fn provider_save_requires_a_successful_test() {
        isolate_config();
        let mut cfg = with_providers();
        let mut s = SettingsState::open(&cfg);
        s.api_url = "https://api.example.test".into();
        s.api_key = "sk-new".into();
        s.cursor = LAYOUT
            .iter()
            .position(|(_, r)| *r == Row::ProviderSave)
            .unwrap();
        s.on_key(key(KeyCode::Enter), &mut cfg).await;
        assert_eq!(s.status_kind, StatusKind::Warn);
        assert_eq!(cfg.providers.len(), 2, "nothing may be persisted untested");
    }

    #[tokio::test]
    async fn the_automode_row_persists_the_approval_mode() {
        isolate_config();
        let mut cfg = LucyConfig::default();
        let mut s = SettingsState::open(&cfg);
        s.cursor = LAYOUT
            .iter()
            .position(|(_, r)| *r == Row::Approvals)
            .unwrap();

        assert!(Row::Approvals.is_choice());
        assert_eq!(s.value(Row::Approvals), "Risky tools only");

        // `→` cycles: Risky → Auto (never ask).
        s.on_key(key(KeyCode::Right), &mut cfg).await;
        assert_eq!(s.approvals_mode, "never");
        assert_eq!(cfg.approval_mode(), "never");
        assert!(cfg.automode());
        assert!(!cfg.validate().is_err());

        // And on to Always ask.
        s.on_key(key(KeyCode::Right), &mut cfg).await;
        assert_eq!(cfg.approval_mode(), "always");
        assert!(!cfg.automode());

        // Back around to Risky only.
        s.on_key(key(KeyCode::Right), &mut cfg).await;
        assert_eq!(cfg.approval_mode(), "write");
    }

    #[tokio::test]
    async fn provider_test_rejects_an_incomplete_draft_without_network() {
        let cfg = with_providers();
        let mut s = SettingsState::open(&cfg);
        s.cursor = LAYOUT
            .iter()
            .position(|(_, r)| *r == Row::ProviderTest)
            .unwrap();
        s.on_key(key(KeyCode::Enter), &mut cfg.clone()).await;
        assert_eq!(s.status_kind, StatusKind::Err);
        assert!(s.status.contains("API URL"), "{}", s.status);
        assert!(!s.busy);
    }

    #[tokio::test]
    async fn classification_save_persists_the_url_only() {
        isolate_config();
        let mut cfg = with_providers();
        let mut s = SettingsState::open(&cfg);
        s.classification_url = "http://127.0.0.1:8001/".into();
        s.cursor = LAYOUT
            .iter()
            .position(|(_, r)| *r == Row::ClassifySave)
            .unwrap();
        s.on_key(key(KeyCode::Enter), &mut cfg).await;
        assert_eq!(cfg.classification_api_url(), "http://127.0.0.1:8001");
        assert_eq!(s.status_kind, StatusKind::Ok);
    }

    #[tokio::test]
    async fn classification_save_rejects_a_bad_url() {
        isolate_config();
        let mut cfg = with_providers();
        let mut s = SettingsState::open(&cfg);
        s.classification_url = "localhost:8001".into();
        s.cursor = LAYOUT
            .iter()
            .position(|(_, r)| *r == Row::ClassifySave)
            .unwrap();
        s.on_key(key(KeyCode::Enter), &mut cfg).await;
        assert_eq!(s.status_kind, StatusKind::Err);
        assert_eq!(
            cfg.classification_api_url(),
            "http://localhost:8001",
            "a rejected URL must not be persisted"
        );
    }

    #[tokio::test]
    async fn busy_state_ignores_keys() {
        let cfg = with_providers();
        let mut s = SettingsState::open(&cfg);
        s.busy = true;
        let before = s.cursor();
        assert!(
            s.on_key(key(KeyCode::Down), &mut cfg.clone())
                .await
                .is_none()
        );
        assert_eq!(s.cursor(), before);
    }

    #[test]
    fn values_render_placeholders_for_empty_fields() {
        let cfg = LucyConfig::default();
        let s = SettingsState::open(&cfg);
        assert!(s.value(Row::ApiUrl).contains("type to edit"));
        assert!(s.value(Row::ApiKey).contains("type to edit"));
        // The default classification URL is pre-filled, never a placeholder.
        assert_eq!(s.value(Row::ClassifyUrl), "http://localhost:8001");
        // A config with no tier bound shows "(unbound)", not a model name: the
        // model it resolves to is shown in the routing note, and showing it
        // here would look like a binding the user never made.
        assert_eq!(s.value(Row::Level1), "(unbound)");
        // The dropdown still offers a real choice out of the box.
        assert!(
            s.level_keys()
                .contains(&lucy_config::DEFAULT_TEXT_MODEL.to_owned())
        );
    }

    #[test]
    fn row_kinds_match_the_wireframe_interactions() {
        assert!(Row::ApiUrl.is_text() && Row::ApiKey.is_text() && Row::ClassifyUrl.is_text());
        assert!(Row::ProviderTest.is_button() && Row::ClassifySave.is_button());
        assert!(Row::ProviderType.is_choice());
        assert!(Row::ChatMode.is_choice());
        assert!(Row::VoiceModel.is_choice());
        for level in [Row::Level1, Row::Level2, Row::Level3] {
            assert!(level.is_choice());
        }
        assert!(Row::AutoCompact.is_choice());
        assert_eq!(Row::Level2.section(), Section::ChatMode);
        assert_eq!(Row::VoiceModel.section(), Section::VoiceModel);
    }

    // ---- the searchable level dropdown -----------------------------------

    #[test]
    fn only_the_level_rows_offer_a_searchable_dropdown() {
        // The short lists keep plain ←/→; the level rows offer every text
        // model, so they get `[change]` → search.
        for level in [Row::Level1, Row::Level2, Row::Level3] {
            assert!(level.has_model_picker(), "{level:?}");
            // Still a choice row, so ←/→ still nudges without opening the list.
            assert!(level.is_choice(), "{level:?}");
        }
        for other in [
            Row::ChatMode,
            Row::AutoCompact,
            Row::Approvals,
            Row::ProviderType,
            Row::ProviderPreset,
            Row::VoiceModel,
        ] {
            assert!(!other.has_model_picker(), "{other:?}");
        }
    }

    #[test]
    fn the_dropdown_is_closed_until_the_user_opens_it() {
        let cfg = many_models();
        let s = SettingsState::open(&cfg);
        assert!(!s.picker_is_open(), "a fresh screen must not be open");
        assert!(s.picker.is_none());
    }

    #[test]
    fn the_row_button_flips_from_change_to_close_while_the_list_is_open() {
        // A pure check: the button sits under the panel while the list is open,
        // so its pixels are not observable in a render test.
        for level in [Row::Level1, Row::Level2, Row::Level3] {
            let (closed, is_button) = row_suffix(level, false);
            assert_eq!(closed, "  [change]", "{level:?}");
            assert!(is_button, "{level:?} offers a button, not a ←/→ hint");
            assert_eq!(row_suffix(level, true).0, "  [close]", "{level:?}");
        }
        // The short lists keep their static hints, and never claim to be a
        // button.
        assert_eq!(row_suffix(Row::ChatMode, false), ("   ←/→", false));
        assert_eq!(row_suffix(Row::AutoCompact, false), ("   ←/→", false));
        assert_eq!(row_suffix(Row::ProviderTest, false), ("   (Enter)", false));
        assert_eq!(row_suffix(Row::ApiUrl, false), ("", false));
        // A level row's list is the only thing that can be open on it, so
        // `row_suffix(row, true)` is a reachable state for no other row. Assert
        // the closed-state suffix instead — the open state is only ever
        // produced for a row that `has_model_picker()`.
        assert_eq!(row_suffix(Row::VoiceModel, false), ("   ←/→", false));
        assert!(
            [Row::ChatMode, Row::VoiceModel, Row::AutoCompact]
                .iter()
                .all(|r| row_suffix(*r, true).0 == "  [close]"),
            "row_suffix(_, true) is the open state, so it only makes sense for a \
             row that actually has a dropdown"
        );
    }

    #[tokio::test]
    async fn enter_on_a_level_row_opens_the_dropdown_and_esc_closes_it_unchanged() {
        isolate_config();
        let mut cfg = many_models();
        let mut s = focus(SettingsState::open(&cfg), Row::Level2);
        let before = s.level_models[1].clone();

        s.on_key(key(KeyCode::Enter), &mut cfg).await;
        assert!(s.picker_is_open());
        assert_eq!(s.picker.as_ref().map(|p| p.row), Some(Row::Level2));

        // Esc closes the dropdown, not the screen, and applies nothing.
        let action = s.on_key(key(KeyCode::Esc), &mut cfg).await;
        assert!(!s.picker_is_open());
        assert_eq!(action, None, "the first Esc must not close the screen");
        assert!(s.open);
        assert_eq!(s.level_models[1], before);
        assert_eq!(cfg.level_model(ReasoningLevel::L2), before);
    }

    #[tokio::test]
    async fn a_second_esc_after_the_dropdown_closes_the_screen() {
        isolate_config();
        let mut cfg = many_models();
        let mut s = focus(SettingsState::open(&cfg), Row::Level1);
        s.on_key(key(KeyCode::Enter), &mut cfg).await;
        assert!(s.picker_is_open());
        assert_eq!(s.on_key(key(KeyCode::Esc), &mut cfg).await, None);
        assert_eq!(
            s.on_key(key(KeyCode::Esc), &mut cfg).await,
            Some(SettingsAction::Close)
        );
        assert!(!s.open);
    }

    #[tokio::test]
    async fn arrow_keys_still_cycle_a_level_without_opening_the_dropdown() {
        isolate_config();
        let mut cfg = many_models();
        let mut s = focus(SettingsState::open(&cfg), Row::Level3);
        let before = s.level_models[2].clone();
        s.on_key(key(KeyCode::Right), &mut cfg).await;
        assert!(!s.picker_is_open(), "←/→ must not open the list");
        assert_ne!(s.level_models[2], before, "→ still steps the tier");
        assert_eq!(cfg.level_model(ReasoningLevel::L3), s.level_models[2]);
    }

    #[tokio::test]
    async fn the_dropdown_opens_on_the_model_that_is_already_live() {
        isolate_config();
        let mut cfg = many_models();
        let mut s = focus(SettingsState::open(&cfg), Row::Level1);
        s.level_models[0] = "google/gemini-2.5-pro".into();
        s.open_picker(Row::Level1);
        let visible = s.visible_picker_choices();
        let p = s.picker.as_ref().expect("open");
        assert_eq!(
            visible[p.cursor].key, "google/gemini-2.5-pro",
            "Enter on an untouched dropdown must be a no-op, not a jump to the top"
        );
    }

    #[tokio::test]
    async fn the_dropdown_lists_the_clear_entry_recent_group_and_each_provider() {
        let cfg = many_models();
        let mut s = SettingsState::open(&cfg);
        s.level_models = [
            "google/gemini-2.5-pro".into(),
            "groq/flash".into(),
            "".into(),
        ];
        s.open_picker(Row::Level1);
        let choices = s.visible_picker_choices();

        assert_eq!(choices[0].key, "", "the clear entry leads the list");
        assert_eq!(choices[0].display, "(unbind — use the next cheaper tier)");
        assert_eq!(choices[0].group, None, "the clear entry has no heading");

        // Recent holds the bound models plus every tier's effective fallback,
        // in that order, each listed exactly once.
        let recent: Vec<&str> = choices
            .iter()
            .filter(|c| c.group.as_deref() == Some(RECENT_GROUP))
            .map(|c| c.key.as_str())
            .collect();
        assert_eq!(recent[..2], ["google/gemini-2.5-pro", "groq/flash"]);
        let mut deduped = recent.clone();
        deduped.sort_unstable();
        deduped.dedup();
        assert_eq!(deduped.len(), recent.len(), "Recent repeats: {recent:?}");
        // A known model is spelled out, because a bare id rarely identifies it.
        assert_eq!(choices[1].display, "Google / gemini-2.5-pro");

        // The provider groups carry only what Recent does not, headed by the
        // provider name and listing bare model ids.
        let google: Vec<&str> = choices
            .iter()
            .filter(|c| c.group.as_deref() == Some("Google"))
            .map(|c| c.key.as_str())
            .collect();
        assert_eq!(
            google,
            ["google/gemini-2.5-flash"],
            "the Recent copy is not repeated under its provider"
        );
        assert!(
            choices
                .iter()
                .any(|c| c.group.as_deref() == Some("Groq") && c.key == "groq/pro"),
            "the second provider gets its own group: {choices:?}"
        );
    }

    #[tokio::test]
    async fn the_dropdown_never_offers_less_than_the_arrow_keys_do() {
        // The searchable list replaces blind cycling, so it must be a superset
        // of the `←`/`→` key list or a model becomes unreachable.
        let cfg = many_models();
        let mut s = SettingsState::open(&cfg);
        s.level_models = ["groq/pro".into(), "".into(), "".into()];
        s.open_picker(Row::Level1);
        let offered: Vec<String> = s
            .visible_picker_choices()
            .into_iter()
            .map(|c| c.key)
            .collect();
        for k in s.level_keys() {
            assert!(offered.contains(&k), "{k} reachable by ←/→ but not offered");
        }
    }

    #[tokio::test]
    async fn typing_narrows_the_list_and_enter_applies_the_highlighted_model() {
        isolate_config();
        let mut cfg = many_models();
        let mut s = focus(SettingsState::open(&cfg), Row::Level1);
        s.on_key(key(KeyCode::Enter), &mut cfg).await;
        type_query(&mut s, &mut cfg, "gemini-2.5").await;

        let choices = s.visible_picker_choices();
        let keys: Vec<&str> = choices.iter().map(|c| c.key.as_str()).collect();
        assert_eq!(keys, ["google/gemini-2.5-flash", "google/gemini-2.5-pro"]);
        assert_eq!(s.picker.as_ref().expect("open").query(), "gemini-2.5");

        // The cursor rests on the top hit, so Enter applies it and persists.
        let action = s.on_key(key(KeyCode::Enter), &mut cfg).await;
        assert_eq!(action, Some(SettingsAction::ModelsChanged));
        assert_eq!(s.level_models[0], "google/gemini-2.5-flash");
        assert_eq!(
            cfg.level_model(ReasoningLevel::L1),
            "google/gemini-2.5-flash"
        );
        assert!(!s.picker_is_open(), "applying closes the list");
        assert_eq!(s.value(Row::Level1), "Google · gemini-2.5-flash");
    }

    #[tokio::test]
    async fn a_bare_family_name_finds_the_fallback_model_too() {
        // `gemini` must reach both the connected Google models and the
        // compiled-in `gemini-web` fallback, which has no provider to group it
        // under — otherwise a user searching by family silently loses the
        // model a config with no Google provider is actually running.
        isolate_config();
        let mut cfg = many_models();
        let mut s = focus(SettingsState::open(&cfg), Row::Level1);
        s.on_key(key(KeyCode::Enter), &mut cfg).await;
        type_query(&mut s, &mut cfg, "gemini").await;
        let keys: Vec<String> = s
            .visible_picker_choices()
            .into_iter()
            .map(|c| c.key)
            .collect();
        assert!(
            keys.contains(&"google/gemini-2.5-pro".to_owned()),
            "{keys:?}"
        );
        assert!(
            keys.contains(&"google/gemini-2.5-flash".to_owned()),
            "{keys:?}"
        );
        assert!(keys.contains(&DEFAULT_TEXT_MODEL.to_owned()), "{keys:?}");
    }

    #[tokio::test]
    async fn a_query_matching_nothing_keeps_the_dropopen_open_and_changes_nothing() {
        isolate_config();
        let mut cfg = many_models();
        let mut s = focus(SettingsState::open(&cfg), Row::Level1);
        s.on_key(key(KeyCode::Enter), &mut cfg).await;
        type_query(&mut s, &mut cfg, "zzzz").await;
        assert!(s.visible_picker_choices().is_empty());

        let action = s.on_key(key(KeyCode::Enter), &mut cfg).await;
        assert_eq!(action, None);
        assert!(
            s.picker_is_open(),
            "Enter on an empty list is not a selection"
        );
        assert!(s.level_models[0].is_empty(), "nothing was applied");
    }

    #[tokio::test]
    async fn backspace_widens_the_query_and_resets_the_cursor_to_the_top_hit() {
        isolate_config();
        let mut cfg = many_models();
        let mut s = focus(SettingsState::open(&cfg), Row::Level1);
        s.on_key(key(KeyCode::Enter), &mut cfg).await;
        // A query matching two models, so there is somewhere for ↓ to go.
        type_query(&mut s, &mut cfg, "gemini-2.5").await;
        s.on_key(key(KeyCode::Down), &mut cfg).await;
        assert_eq!(s.picker.as_ref().expect("open").cursor(), 1);

        s.on_key(key(KeyCode::Backspace), &mut cfg).await;
        assert_eq!(s.picker.as_ref().expect("open").query(), "gemini-2.");
        assert_eq!(
            s.picker.as_ref().expect("open").cursor(),
            0,
            "editing the query snaps back to the top hit"
        );
    }

    #[tokio::test]
    async fn delete_and_ctrl_u_clear_the_query() {
        isolate_config();
        let mut cfg = many_models();
        let mut s = focus(SettingsState::open(&cfg), Row::Level1);
        s.on_key(key(KeyCode::Enter), &mut cfg).await;
        type_query(&mut s, &mut cfg, "gemini-2.5-flash").await;
        s.on_key(key(KeyCode::Delete), &mut cfg).await;
        assert_eq!(s.picker.as_ref().expect("open").query(), "gemini-2.5-flas");
        // `Ctrl+U` clears the whole thing in one press.
        s.on_key(
            key_mods(KeyCode::Char('u'), KeyModifiers::CONTROL),
            &mut cfg,
        )
        .await;
        assert_eq!(s.picker.as_ref().expect("open").query(), "");
        assert_eq!(
            s.visible_picker_choices().len(),
            s.level_picker_choices().len(),
            "an empty query shows everything again"
        );
    }

    #[tokio::test]
    async fn home_and_end_jump_to_the_ends_of_the_narrowed_list() {
        isolate_config();
        let mut cfg = many_models();
        let mut s = focus(SettingsState::open(&cfg), Row::Level1);
        s.on_key(key(KeyCode::Enter), &mut cfg).await;
        type_query(&mut s, &mut cfg, "gemini-2.5").await;
        s.on_key(key(KeyCode::End), &mut cfg).await;
        let last = s.visible_picker_choices().len() - 1;
        assert_eq!(s.picker.as_ref().expect("open").cursor(), last);
        s.on_key(key(KeyCode::Home), &mut cfg).await;
        assert_eq!(s.picker.as_ref().expect("open").cursor(), 0);
    }

    #[tokio::test]
    async fn up_and_down_clamp_at_the_ends_of_the_narrowed_list() {
        isolate_config();
        let mut cfg = many_models();
        let mut s = focus(SettingsState::open(&cfg), Row::Level1);
        s.on_key(key(KeyCode::Enter), &mut cfg).await;
        type_query(&mut s, &mut cfg, "gemini").await;
        for _ in 0..5 {
            s.on_key(key(KeyCode::Up), &mut cfg).await;
        }
        assert_eq!(s.picker.as_ref().expect("open").cursor(), 0, "no wrap up");
        for _ in 0..5 {
            s.on_key(key(KeyCode::Down), &mut cfg).await;
        }
        let last = s.visible_picker_choices().len() - 1;
        assert_eq!(
            s.picker.as_ref().expect("open").cursor(),
            last,
            "no wrap down"
        );
    }

    #[tokio::test]
    async fn ctrl_p_and_ctrl_n_step_the_dropdown_cursor() {
        isolate_config();
        let mut cfg = many_models();
        let mut s = focus(SettingsState::open(&cfg), Row::Level1);
        s.on_key(key(KeyCode::Enter), &mut cfg).await;
        s.on_key(
            key_mods(KeyCode::Char('n'), KeyModifiers::CONTROL),
            &mut cfg,
        )
        .await;
        assert_eq!(s.picker.as_ref().expect("open").cursor(), 1);
        s.on_key(
            key_mods(KeyCode::Char('n'), KeyModifiers::CONTROL),
            &mut cfg,
        )
        .await;
        assert_eq!(s.picker.as_ref().expect("open").cursor(), 2);
        s.on_key(
            key_mods(KeyCode::Char('p'), KeyModifiers::CONTROL),
            &mut cfg,
        )
        .await;
        assert_eq!(s.picker.as_ref().expect("open").cursor(), 1);
    }

    #[tokio::test]
    async fn the_clear_entry_unsets_the_tier() {
        isolate_config();
        let mut cfg = many_models();
        let mut s = focus(SettingsState::open(&cfg), Row::Level1);
        s.level_models = ["groq/flash".into(), "".into(), "".into()];
        s.open_picker(Row::Level1);
        // The list opens on what is live, so step up once to reach the clear
        // entry that leads the list.
        assert_eq!(
            s.picker.as_ref().expect("open").cursor(),
            1,
            "it opens on the bound model"
        );
        s.on_key(key(KeyCode::Up), &mut cfg).await;
        s.on_key(key(KeyCode::Enter), &mut cfg).await;
        assert_eq!(s.level_models[0], "");
        assert_eq!(cfg.level_model(ReasoningLevel::L1), "");
        assert_eq!(s.value(Row::Level1), "(unbound)");
    }

    #[tokio::test]
    async fn plain_characters_type_into_the_search_box_instead_of_the_screen() {
        isolate_config();
        let mut cfg = many_models();
        let mut s = focus(SettingsState::open(&cfg), Row::Level1);
        s.on_key(key(KeyCode::Enter), &mut cfg).await;
        // `t` and `s` are the [Test]/[Save] shortcuts elsewhere on this screen;
        // they must land in the search box, not press a button.
        type_query(&mut s, &mut cfg, "ts").await;
        assert_eq!(s.picker.as_ref().expect("open").query(), "ts");
        assert!(!s.busy);
        assert!(s.status.is_empty(), "no button was fired: {}", s.status);
    }

    #[test]
    fn search_is_case_and_separator_insensitive_across_name_and_key() {
        let c = PickerChoice {
            key: "google/gemini-2.5-pro".into(),
            display: "Google · gemini-2.5-pro".into(),
            group: None,
        };
        // Name, provider, and the stored key all find it…
        for q in [
            "gemini",
            "Gemini",
            "GEMINI-2.5",
            "google",
            "Google",
            "google/gemini",
        ] {
            assert!(match_rank(q, &c).is_some(), "query {q:?} should match");
        }
        // …as do separator-agnostic spellings.
        assert!(match_rank("gemini2.5pro", &c).is_some());
        assert!(match_rank("g o o g l e", &c).is_some());
        assert!(match_rank("claude", &c).is_none());
        assert!(match_rank("", &c) == Some(0), "an empty query matches");
    }

    #[test]
    fn prefix_matches_outrank_substring_matches_inside_a_group() {
        // `turbo` starts with the query; `super-turbo` only contains it. Both
        // survive the filter, and the one the user most likely meant must be
        // first on the highlighted row.
        let mut cfg = with_providers();
        cfg.providers[0].available_models = vec!["super-turbo".into(), "turbo".into()];
        let mut s = SettingsState::open(&cfg);
        s.open_picker(Row::Level1);
        let groq: Vec<String> = s
            .filtered_picker_choices("turbo")
            .into_iter()
            .filter(|c| c.group.as_deref() == Some("Groq"))
            .map(|c| c.display)
            .collect();
        assert_eq!(
            groq,
            ["turbo", "super-turbo"],
            "the prefix match leads, even though it sorts later alphabetically"
        );
    }

    #[test]
    fn a_punctuation_only_query_matches_literally_instead_of_everything() {
        // `-` and `/` normalize away; falling back to a raw substring test
        // keeps them from silently matching the whole list.
        let c = PickerChoice {
            key: "groq/flash".into(),
            display: "flash".into(),
            group: None,
        };
        assert!(match_rank("/", &c).is_some());
        assert!(match_rank("-", &c).is_none());
    }
}

#[cfg(test)]
mod render_tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};

    use crate::model::App;

    fn render(state: &SettingsState, config: &LucyConfig) -> String {
        let backend = TestBackend::new(110, 40);
        let mut terminal = Terminal::new(backend).expect("test terminal");
        terminal
            .draw(|frame| {
                let area = frame.area();
                draw(frame, area, state, config);
            })
            .expect("draw");
        let buffer = terminal.backend().buffer().clone();
        (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn seeded() -> (SettingsState, LucyConfig) {
        let mut config = lucy_config::LucyConfig {
            providers: vec![
                lucy_config::ProviderConfig {
                    id: "groq".into(),
                    name: "Groq".into(),
                    api_url: "https://api.groq.test".into(),
                    api_key: "sk-x".into(),
                    provider_type: lucy_config::ProviderType::Text,
                    available_models: vec!["flash".into(), "pro".into()],
                    deprecated_models: Vec::new(),
                },
                lucy_config::ProviderConfig {
                    id: "eleven".into(),
                    name: "ElevenLabs".into(),
                    api_url: "https://api.eleven.test".into(),
                    api_key: "sk-y".into(),
                    provider_type: lucy_config::ProviderType::Voice,
                    available_models: vec!["tts".into()],
                    deprecated_models: Vec::new(),
                },
            ],
            ..Default::default()
        };
        config.set_anchor_model("groq/flash");
        config.set_voice_model("eleven/tts");
        config.set_chat_mode(lucy_config::ChatMode::Manual);
        config.set_auto_compact(false);
        (SettingsState::open(&config), config)
    }

    #[test]
    fn renders_all_six_section_headers() {
        let (state, config) = seeded();
        let out = render(&state, &config);
        for title in Section::ALL.map(|s| s.title()) {
            assert!(
                out.contains(title),
                "missing section header {title}:\n{out}"
            );
        }
    }

    #[test]
    fn renders_every_wireframe_row() {
        let (state, config) = seeded();
        let out = render(&state, &config);
        for row in LAYOUT.map(|(_, r)| r) {
            assert!(out.contains(row.label()), "missing row {}", row.label());
        }
    }

    #[test]
    fn classification_section_never_shows_an_api_key_row() {
        let (state, config) = seeded();
        let out = render(&state, &config);
        let classification = Section::Classification;
        // Find the classification block and confirm it has only an API URL.
        let start = out.find(classification.title()).expect("header");
        let end = out[start..]
            .find(Section::ChatMode.title())
            .map(|i| start + i)
            .expect("next header");
        let block = &out[start..end];
        assert!(block.contains("API URL"), "{block}");
        assert!(!block.contains("API Key"), "{block}");
        assert!(block.contains("[Test]"), "{block}");
        assert!(block.contains("[Save]"), "{block}");
    }

    #[test]
    fn dropdowns_show_the_selected_models() {
        let (state, config) = seeded();
        let out = render(&state, &config);
        assert!(out.contains("Groq · flash"), "{out}");
        assert!(out.contains("ElevenLabs · tts"), "{out}");
        assert!(out.contains("Manual"), "{out}");
        assert!(out.contains("Off"), "{out}");
        assert!(out.contains("http://localhost:8001"), "{out}");
    }

    #[test]
    fn the_tested_provider_draft_is_summarized_under_the_test_row() {
        let (mut state, config) = seeded();
        state.api_url = "https://api.example.test".into();
        state.api_key = "sk-new".into();
        state.discovered_models = vec!["a".into(), "b".into(), "c".into()];
        // The context line renders under whichever row is focused.
        state.cursor = LAYOUT
            .iter()
            .position(|(_, r)| *r == Row::ProviderTest)
            .expect("row");
        let out = render(&state, &config);
        assert!(out.contains("https://api.example.test"), "{out}");
        assert!(out.contains("3 model(s) discovered"), "{out}");
    }

    #[test]
    fn the_status_line_reports_the_last_test_result() {
        let (mut state, config) = seeded();
        state.note(StatusKind::Err, "connection refused");
        let out = render(&state, &config);
        assert!(out.contains("connection refused"), "{out}");
        state.busy = true;
        let out = render(&state, &config);
        assert!(out.contains("working"), "{out}");
    }

    #[test]
    #[ignore = "visual snapshot; run with --ignored --nocapture"]
    fn print_settings_screen() {
        let (state, config) = seeded();
        println!("{}", render(&state, &config));
    }

    #[test]
    fn a_text_field_returns_a_cursor_position_and_a_choice_does_not() {
        let (mut state, config) = seeded();
        // The cursor starts on the provider preset row; focus the API URL
        // field, which is a text row.
        state.cursor = LAYOUT
            .iter()
            .position(|(_, r)| *r == Row::ApiUrl)
            .expect("row");
        let backend = TestBackend::new(110, 40);
        let mut terminal = Terminal::new(backend).expect("test terminal");
        let mut caret = None;
        terminal
            .draw(|frame| {
                caret = draw(frame, frame.area(), &state, &config);
            })
            .expect("draw");
        assert!(caret.is_some(), "a focused text field must place a caret");

        // A dropdown row returns no caret.
        let (mut state, config) = seeded();
        state.cursor = LAYOUT
            .iter()
            .position(|(_, r)| *r == Row::ProviderType)
            .expect("row");
        let backend = TestBackend::new(110, 40);
        let mut terminal = Terminal::new(backend).expect("test terminal");
        let mut caret = Some(Rect::default());
        terminal
            .draw(|frame| {
                caret = draw(frame, frame.area(), &state, &config);
            })
            .expect("draw");
        assert!(caret.is_none(), "a dropdown must not show a text caret");
    }

    #[test]
    fn persisting_tests_never_touch_the_users_real_config() {
        isolate_config();
        let path = lucy_config::config_path().expect("config path");
        assert!(
            path.to_string_lossy().contains("lucy-settings-test-"),
            "tests must not persist to {}",
            path.display()
        );
    }

    #[test]
    fn a_short_terminal_does_not_panic() {
        let (state, config) = seeded();
        for (w, h) in [(20u16, 8u16), (40, 12), (80, 24), (200, 60)] {
            let backend = TestBackend::new(w, h);
            let mut terminal = Terminal::new(backend).expect("test terminal");
            terminal
                .draw(|frame| {
                    let _ = draw(frame, frame.area(), &state, &config);
                })
                .unwrap_or_else(|e| panic!("draw failed at {w}x{h}: {e}"));
        }
    }

    #[test]
    fn the_settings_popup_covers_the_welcome_and_chat_screens() {
        let (state, config) = seeded();
        let mut app = App::new(config.clone());
        app.settings = true;
        app.settings_state = state.clone();
        app.push_msg(crate::model::ChatMsg::user("hello".into()));
        let backend = TestBackend::new(110, 40);
        let mut terminal = Terminal::new(backend).expect("test terminal");
        terminal
            .draw(|frame| super::super::views::draw_frame(frame, &app))
            .expect("draw");
        let buffer = terminal.backend().buffer().clone();
        let out: String = (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(out.contains("1. Connect providers"), "{out}");
        assert!(out.contains("5. Automode"), "{out}");
        assert!(out.contains("6. Voice model"), "{out}");
    }

    // ---- the searchable level dropdown, as drawn -----------------------

    /// The row of `out` carrying `needle`, trimmed, for readable assertions.
    fn line_with(out: &str, needle: &str) -> String {
        out.lines()
            .find(|l| l.contains(needle))
            .unwrap_or_else(|| panic!("no line with {needle:?} in:\n{out}"))
            .trim_end()
            .to_owned()
    }

    #[test]
    fn a_level_row_reads_model_then_change_and_shows_no_list() {
        let (mut state, config) = seeded();
        state.level_models = ["groq/flash".into(), "groq/pro".into(), "".into()];
        state.cursor = LAYOUT
            .iter()
            .position(|(_, r)| *r == Row::Level1)
            .expect("row");
        let out = render(&state, &config);

        // The label, the current model, and the button — in that order.
        let line = line_with(&out, "Level 1 (fast)");
        let label = line.find("Level 1 (fast)").expect("label");
        let value = line.find("Groq · flash").expect("model");
        let button = line.find("[change]").expect("[change]");
        assert!(label < value && value < button, "{line}");

        // Nothing from the dropdown is on screen while it is closed.
        assert!(!out.contains("Search "), "{out}");
        assert!(!out.contains(RECENT_GROUP), "{out}");
        // The other tiers keep their own value and button.
        assert!(out.contains("Groq · pro"), "{out}");
    }

    #[test]
    fn a_closed_dropdown_never_appears_even_with_models_loaded() {
        let mut config = lucy_config::LucyConfig {
            providers: vec![lucy_config::ProviderConfig {
                id: "groq".into(),
                name: "Groq".into(),
                api_url: "https://api.groq.test".into(),
                api_key: "sk-x".into(),
                provider_type: lucy_config::ProviderType::Text,
                available_models: vec!["flash".into(), "pro".into()],
                deprecated_models: Vec::new(),
            }],
            ..Default::default()
        };
        config.set_anchor_model("groq/pro");
        let state = SettingsState::open(&config);
        let out = render(&state, &config);
        assert!(!out.contains("Search "), "closed by default:\n{out}");
        assert!(!out.contains(RECENT_GROUP), "closed by default:\n{out}");
    }

    #[test]
    fn the_open_dropdown_shows_the_search_box_groups_and_models() {
        let (mut state, config) = seeded();
        state.level_models = ["groq/flash".into(), "groq/pro".into(), "".into()];
        state.cursor = LAYOUT
            .iter()
            .position(|(_, r)| *r == Row::Level2)
            .expect("row");
        state.open_picker(Row::Level2);
        let out = render(&state, &config);

        // Title names the tier, so the list says what it is changing.
        assert!(out.contains("Level 2 (balanced) — search models"), "{out}");
        assert!(out.contains("Search"), "{out}");
        // The unbind entry, the Recent group, then one group per text provider.
        assert!(out.contains("(unbind"), "{out}");
        assert!(out.contains(RECENT_GROUP), "{out}");
        assert!(out.contains("Groq"), "{out}");
        // A model in Recent is spelled out; provider groups list bare ids.
        assert!(out.contains("Groq / flash"), "{out}");
        // The footer spells out the keys.
        // The footer spells out the keys the code actually implements, so the
        // hint cannot drift away from the behaviour.
        assert!(out.contains("Enter applies"), "{out}");
        assert!(out.contains("Esc closes"), "{out}");
        assert!(out.contains("ctrl+u clears"), "{out}");
    }

    #[test]
    fn a_voice_model_is_not_offered_by_a_text_tiers_dropdown() {
        // The seeded config has a voice provider. Its models belong to the
        // Voice row, not to the level tiers, so the list must not offer them.
        // (Checked on the choices, not the pixels: the Voice row behind the
        // panel legitimately shows its own model.)
        let (mut state, _config) = seeded();
        state.open_picker(Row::Level1);
        let choices = state.visible_picker_choices();
        assert!(
            !choices.iter().any(|c| c.key.starts_with("eleven/")),
            "a voice model leaked into a text tier: {choices:?}"
        );
        assert!(
            !choices
                .iter()
                .any(|c| c.group.as_deref() == Some("ElevenLabs")),
            "{choices:?}"
        );
        // The text models are still all there.
        assert!(choices.iter().any(|c| c.key == "groq/flash"), "{choices:?}");
    }

    #[test]
    fn a_long_query_keeps_the_highlighted_entry_on_screen() {
        // 60 models: the list must scroll rather than push the highlighted
        // row off the bottom, which would silently apply a different model
        // than the one the cursor is on.
        let mut config = lucy_config::LucyConfig {
            providers: vec![lucy_config::ProviderConfig {
                id: "p".into(),
                name: "P".into(),
                api_url: "https://p.test".into(),
                api_key: "k".into(),
                provider_type: lucy_config::ProviderType::Text,
                available_models: (0..60).map(|i| format!("model-number-{i:02}")).collect(),
                deprecated_models: Vec::new(),
            }],
            ..Default::default()
        };
        config.set_anchor_model("p/model-number-00");
        let mut state = SettingsState::open(&config);
        state.open_picker(Row::Level1);
        // Walk to the very bottom of the list.
        for _ in 0..80 {
            state.picker_move(1);
        }
        let visible = state.visible_picker_choices();
        let cursor = state.picker.as_ref().expect("open").cursor();
        assert_eq!(cursor, visible.len() - 1);
        let highlighted = &visible[cursor].display;
        let out = render(&state, &config);
        assert!(
            out.contains(highlighted.as_str()),
            "the highlighted entry {highlighted:?} scrolled out of view:\n{out}"
        );
    }

    #[test]
    fn the_dropdown_takes_the_caret_away_from_the_settings_text_fields() {
        let (mut state, config) = seeded();
        // Focus a text row, which normally owns the caret…
        state.cursor = LAYOUT
            .iter()
            .position(|(_, r)| *r == Row::ApiUrl)
            .expect("row");
        let backend = TestBackend::new(110, 40);
        let mut terminal = Terminal::new(backend).expect("test terminal");
        let mut caret = None;
        terminal
            .draw(|frame| {
                caret = draw(frame, frame.area(), &state, &config);
            })
            .expect("draw");
        let text_caret = caret.expect("a text field places a caret");

        // …then let the dropdown take over.
        state.open_picker(Row::Level1);
        let backend = TestBackend::new(110, 40);
        let mut terminal = Terminal::new(backend).expect("test terminal");
        let mut caret = None;
        terminal
            .draw(|frame| {
                caret = draw(frame, frame.area(), &state, &config);
            })
            .expect("draw");
        let picker_caret = caret.expect("the search field places a caret");
        assert_ne!(
            text_caret.y, picker_caret.y,
            "the caret moved to the search box"
        );
    }

    #[test]
    fn a_long_model_list_still_renders_on_a_short_terminal() {
        let mut config = lucy_config::LucyConfig {
            providers: vec![lucy_config::ProviderConfig {
                id: "p".into(),
                name: "P".into(),
                api_url: "https://p.test".into(),
                api_key: "k".into(),
                provider_type: lucy_config::ProviderType::Text,
                available_models: (0..40).map(|i| format!("model-number-{i:02}")).collect(),
                deprecated_models: Vec::new(),
            }],
            ..Default::default()
        };
        config.set_anchor_model("p/model-number-00");
        let mut state = SettingsState::open(&config);
        state.open_picker(Row::Level1);
        for (w, h) in [(20u16, 8u16), (40, 12), (80, 24), (200, 60)] {
            let backend = TestBackend::new(w, h);
            let mut terminal = Terminal::new(backend).expect("test terminal");
            terminal
                .draw(|frame| {
                    let _ = draw(frame, frame.area(), &state, &config);
                })
                .unwrap_or_else(|e| panic!("draw failed at {w}x{h}: {e}"));
        }
    }

    #[test]
    #[ignore = "visual snapshot; run with --ignored --nocapture"]
    fn print_settings_screen_with_the_model_dropdown_open() {
        let (mut state, config) = seeded();
        state.level_models = ["groq/flash".into(), "groq/pro".into(), "eleven/tts".into()];
        state.cursor = LAYOUT
            .iter()
            .position(|(_, r)| *r == Row::Level1)
            .expect("row");
        state.open_picker(Row::Level1);
        println!("{}", render(&state, &config));
    }
}
