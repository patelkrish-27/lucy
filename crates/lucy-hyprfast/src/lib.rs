use anyhow::Result;
use lucy_mcp::{McpServerConfig, McpToolDefinition, StdioMcpClient};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    path::PathBuf,
    process::Command,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Domain {
    Browser,
    Desktop,
    Vision,
    Tasks,
    Clipboard,
    Excalidraw,
    Hints,
    System,
    Unknown,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Capability {
    Observe,
    Click,
    Type,
    Keyboard,
    Pointer,
    Window,
    Launch,
    Navigate,
    Extract,
    Act,
    Batch,
    Task,
    Clipboard,
    Draw,
    Wait,
    Bindings,
    Unknown,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Operation {
    Observe,
    Act,
    Query,
    Manage,
    Transform,
    Unknown,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCapability {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub domain: Domain,
    pub capabilities: Vec<Capability>,
    pub operation: Operation,
    pub read_only: bool,
    pub destructive: bool,
    pub batchable: bool,
    pub semantic: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Environment {
    pub os: String,
    pub desktop: String,
    pub display_server: String,
    pub wayland: bool,
    pub hyprland: bool,
    pub hyprctl: bool,
    pub accessibility: bool,
    pub clipboard: bool,
    pub screenshot: bool,
}
impl Default for Environment {
    fn default() -> Self {
        Self::detect()
    }
}
impl Environment {
    pub fn detect() -> Self {
        let os = if cfg!(target_os = "linux") {
            "linux"
        } else if cfg!(target_os = "windows") {
            "windows"
        } else if cfg!(target_os = "macos") {
            "macos"
        } else {
            "unknown"
        }
        .into();
        let desktop_name = std::env::var("XDG_CURRENT_DESKTOP")
            .unwrap_or_default()
            .to_ascii_lowercase();
        let session = std::env::var("XDG_SESSION_TYPE")
            .unwrap_or_default()
            .to_ascii_lowercase();
        let wayland = session == "wayland" || std::env::var_os("WAYLAND_DISPLAY").is_some();
        let hyprland = std::env::var_os("HYPRLAND_INSTANCE_SIGNATURE").is_some()
            || desktop_name.contains("hyprland")
            || std::env::var_os("HYPRLAND_CMD").is_some();
        let desktop = if hyprland {
            "hyprland"
        } else if !desktop_name.is_empty() {
            &desktop_name
        } else {
            "unknown"
        }
        .into();
        let display_server = if wayland {
            "wayland"
        } else if session == "x11" || std::env::var_os("DISPLAY").is_some() {
            "x11"
        } else {
            "unknown"
        }
        .into();
        let hyprctl = hyprland && command_exists("hyprctl");
        let accessibility = std::env::var_os("AT_SPI_BUS_ADDRESS").is_some()
            || std::env::var("LUCY_ACCESSIBILITY")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
        let clipboard =
            command_exists("wl-copy") || command_exists("xclip") || command_exists("xsel");
        let screenshot = command_exists("grim")
            || command_exists("gnome-screenshot")
            || command_exists("scrot")
            || command_exists("hyprshot");
        Self {
            os,
            desktop,
            display_server,
            wayland,
            hyprland,
            hyprctl,
            accessibility,
            clipboard,
            screenshot,
        }
    }
    pub fn context(&self) -> String {
        format!(
            "OS={} desktop={} display_server={} wayland={} hyprland={} hyprctl={} accessibility={} clipboard={} screenshot={}",
            self.os,
            self.desktop,
            self.display_server,
            self.wayland,
            self.hyprland,
            self.hyprctl,
            self.accessibility,
            self.clipboard,
            self.screenshot
        )
    }
}
fn command_exists(command: &str) -> bool {
    Command::new(command)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HyprFastCatalog {
    pub tools: HashMap<String, ToolCapability>,
    #[serde(default)]
    pub environment: Environment,
}
impl HyprFastCatalog {
    pub fn from_tools(tools: Vec<McpToolDefinition>) -> Self {
        let mut catalog = Self {
            tools: HashMap::new(),
            environment: Environment::detect(),
        };
        for tool in tools {
            let c = classify(&tool);
            catalog.tools.insert(c.name.clone(), c);
        }
        catalog
    }
    pub fn len(&self) -> usize {
        self.tools.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
    pub fn by_domain(&self, domain: Domain) -> Vec<&ToolCapability> {
        self.tools.values().filter(|t| t.domain == domain).collect()
    }
    pub fn summary(&self) -> BTreeMap<String, usize> {
        let mut out = BTreeMap::new();
        for tool in self.tools.values() {
            *out.entry(format!("{:?}", tool.domain)).or_insert(0) += 1;
        }
        out
    }
    pub fn capability_for_mcp_name(&self, name: &str) -> Option<&ToolCapability> {
        self.tools
            .values()
            .find(|tool| full_name(&tool.name) == name || tool.name == name)
    }
    pub async fn discover(config: McpServerConfig) -> Result<Self> {
        Ok(Self::discover_with_definitions(config).await?.0)
    }
    /// Discover the catalog AND return the raw tool definitions behind it.
    ///
    /// The `npx`-spawned Computer Use server costs ~1.4s per handshake, so its
    /// definitions are served from an on-disk cache while fresh (default TTL
    /// 24h, `LUCY_MCP_DEFS_TTL_SECS`, `0` = always refresh). The fast native
    /// `hyprfast` server (~8ms) is always fetched live. Both fetches run
    /// concurrently, and a stale cache is used as fallback when a live fetch
    /// fails, so startup stays in the millisecond range in the common case.
    pub async fn discover_with_definitions(
        config: McpServerConfig,
    ) -> Result<(Self, Vec<McpToolDefinition>, Vec<McpToolDefinition>)> {
        let stale_cache = load_cached_defs();
        let fresh_cache = stale_cache.clone().filter(|c| !defs_cache_stale(c));
        let hyprfast_client = StdioMcpClient::new(config);
        let (hyprfast_res, computer_res) = tokio::join!(
            hyprfast_client.list_tools(),
            resolve_computer_use_defs(fresh_cache.as_ref().map(|c| c.computer_use.clone())),
        );
        let hyprfast_defs = match hyprfast_res {
            Ok(defs) => defs,
            Err(error) => match stale_cache
                .as_ref()
                .map(|c| c.hyprfast.clone())
                .filter(|d| !d.is_empty())
            {
                Some(defs) => {
                    tracing::warn!(error=%error, "HyprFast MCP unavailable; using cached tool definitions");
                    defs
                }
                None => return Err(error.context("failed to discover HyprFast MCP tools")),
            },
        };
        let computer_live_ok = computer_res.is_ok();
        let computer_defs = match computer_res {
            Ok(defs) => defs,
            Err(error) => match stale_cache
                .as_ref()
                .map(|c| c.computer_use.clone())
                .filter(|d| !d.is_empty())
            {
                Some(defs) => {
                    tracing::warn!(error=%error, "ADK Computer Use unavailable; using cached tool definitions");
                    defs
                }
                None => {
                    tracing::warn!(error=%error, "ADK Computer Use unavailable; continuing with HyprFast only");
                    Vec::new()
                }
            },
        };
        // Toll the slow path only when we actually paid it: persist defs fetched
        // live so the next startup can skip the `npx` spawn entirely. A failed
        // live fetch must NOT refresh the timestamp, or a transient failure would
        // pin stale defs as "fresh" for a full TTL.
        if fresh_cache.is_none() && computer_use_enabled() && computer_live_ok {
            save_cached_defs(&hyprfast_defs, &computer_defs).await;
        }
        let mut tools = hyprfast_defs.clone();
        for def in &computer_defs {
            let mut d = def.clone();
            d.name = format!("computer_use_{}", d.name);
            tools.push(d);
        }
        if !computer_defs.is_empty() {
            tracing::info!(
                computer_use = computer_defs.len(),
                "ADK Computer Use MCP merged into capability catalog"
            );
        }
        Ok((Self::from_tools(tools), hyprfast_defs, computer_defs))
    }
    pub async fn discover_default() -> Result<Self> {
        Self::discover(default_config()).await
    }
    pub fn cache_path() -> PathBuf {
        std::env::var("LUCY_HYPRFAST_CACHE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
                    .join(".local/state/lucy/hyprfast-catalog.json")
            })
    }
    pub async fn save(&self) -> Result<()> {
        let path = Self::cache_path();
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(path, serde_json::to_vec_pretty(self)?).await?;
        Ok(())
    }
    pub async fn load() -> Result<Self> {
        Ok(serde_json::from_slice(
            &tokio::fs::read(Self::cache_path()).await?,
        )?)
    }
    /// §11.4.1 + §6 — v2 harness tool filtering.
    ///
    /// Batch-tool preference is enforced by *removing* tools from the offered
    /// schema, not by asking nicely in the prompt: when a batch tool covers a
    /// domain, redundant single-action tools for that domain are hidden, and a
    /// launch-type tool is hidden whenever the runtime already has a usable
    /// session (`browser_ready == true`).
    ///
    /// Hint-first variant, used by the Level 3 planner. Plans run blind, so the
    /// offered set is the one a blind plan can actually complete: navigation
    /// (the cheap way to skip an interaction step entirely), the self-resolving
    /// hint pipeline, and the natives. The ref-needing single-action CDP verbs
    /// (`browser_click` / `browser_type` / `browser_hover` /
    /// `browser_select_option`) are hidden whenever a self-resolving
    /// interaction tool exists, because a plan can never produce the `ref` they
    /// need.
    ///
    /// Degrades in two ways, both logged: with no self-resolving interaction
    /// tool in the catalog (a stale cache from an older hyprfast) the ref-needing
    /// verbs stay, and if filtering would empty the set entirely the unfiltered
    /// candidates come back.
    pub fn planner_tool_set(&self, candidates: &[String], browser_ready: bool) -> Vec<String> {
        let self_resolving = candidates.iter().any(|n| {
            self.capability_for_mcp_name(n)
                .map(is_self_resolving_interaction)
                .unwrap_or(false)
        });
        if !self_resolving {
            tracing::warn!(
                "no self-resolving interaction tool (hint_act/hint_batch) in the catalog — \
                 planner keeps the ref-needing browser verbs"
            );
        }
        let mut out: Vec<String> = Vec::new();
        let has_batch = candidates.iter().any(|n| {
            self.capability_for_mcp_name(n)
                .map(is_batch_tool)
                .unwrap_or(false)
        });
        for name in candidates {
            // Removed tools are never offered, even when the installed hyprfast
            // binary still advertises them: `ground` / `act_fast` / `act_batch`
            // need a Gemini key (`ground.env`) Lucy no longer uses, and
            // `stagehand_*` is gone. Filtering here (not prompting) is what
            // keeps the planner from emitting a step the catalog cannot run.
            if is_removed_tool(name) {
                continue;
            }
            let Some(cap) = self.capability_for_mcp_name(name) else {
                // Unknown names still pass through unless they are removed
                // spellings handled above.
                if is_removed_tool(name) {
                    continue;
                }
                out.push(name.clone());
                continue;
            };
            if is_removed_tool(&cap.name) {
                continue;
            }
            if is_launch_tool(cap) {
                // "Open X" never means "launch a second window" (§6). The launch
                // tool is only offered when no usable session exists.
                if matches!(cap.domain, Domain::Browser) && browser_ready {
                    continue;
                }
                if matches!(cap.domain, Domain::Desktop)
                    && browser_ready
                    && cap.name.to_ascii_lowercase().contains("browser")
                {
                    continue;
                }
            }
            if self_resolving && needs_element_address(cap) {
                // A blind plan has no snapshot ref and rarely a real selector,
                // so the self-resolving verb is the only one it can complete.
                continue;
            }
            if has_batch
                && !is_planner_core(cap)
                && matches!(cap.domain, Domain::Browser | Domain::Hints)
                && !cap.read_only
                && !is_batch_tool(cap)
                && !is_observation_tool(cap)
            {
                // Covered by the batch tool — hide the single-action alternative.
                continue;
            }
            out.push(name.clone());
        }
        // Never hide *every* observation path: if filtering removed all
        // read-only tools but candidates had some, restore them so the planner
        // can always end its sequence with an observation (§4.3).
        if !out.iter().any(|n| {
            self.capability_for_mcp_name(n)
                .map(|c| c.read_only || is_observation_tool(c))
                .unwrap_or(false)
        }) {
            for name in candidates {
                if self
                    .capability_for_mcp_name(name)
                    .map(|c| c.read_only || is_observation_tool(c))
                    .unwrap_or(false)
                    && !out.contains(name)
                {
                    out.push(name.clone());
                }
            }
        }
        if out.is_empty() && !candidates.is_empty() {
            tracing::warn!(
                "planner tool filtering removed everything — offering the unfiltered catalog"
            );
            return candidates.to_vec();
        }
        out
    }

    /// Batch tools available for a domain (used to decide filtering + prompts).
    pub fn batch_tools_for_domain(&self, domain: Domain) -> Vec<&ToolCapability> {
        self.by_domain(domain)
            .into_iter()
            .filter(|c| is_batch_tool(c))
            .collect()
    }
    /// Compact `domain(count)` summary for the Router prompt (§4.1 — counts
    /// only, not full schemas).
    pub fn domain_summary_line(&self) -> String {
        self.summary()
            .into_iter()
            .map(|(k, v)| format!("{k}({v})"))
            .collect::<Vec<_>>()
            .join(", ")
    }
}
pub fn full_name(name: &str) -> String {
    if name.starts_with("computer_use_") {
        format!(
            "mcp_computer_use_{}",
            sanitize(name.trim_start_matches("computer_use_"))
        )
    } else {
        format!("mcp_hyprfast_{}", sanitize(name))
    }
}
fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}
/// Tools hyprfast no longer ships (or Lucy no longer uses): vision grounding
/// via Gemini (`ground` / `act_fast` / `act_batch`, all requiring
/// `~/.config/hyprfast/ground.env`) and the old `stagehand_*` LLM tools.
/// The installed binary may still advertise them, so every planner-facing
/// path filters them by name — never by description — and no prompt may tell
/// the model to reach for them. A plan naming one is a plan for a tool the
/// catalog will not run.
pub fn is_removed_tool(name: &str) -> bool {
    let base = name
        .trim()
        .strip_prefix("mcp_hyprfast_")
        .or_else(|| name.trim().strip_prefix("mcp_computer_use_"))
        .or_else(|| name.trim().strip_prefix("mcp_"))
        .unwrap_or(name.trim())
        .to_ascii_lowercase();
    matches!(base.as_str(), "ground" | "act_fast" | "act_batch")
        || base.starts_with("stagehand")
}
/// A batch/plan tool accepts an ordered list of steps in one call
/// (`execute_plan`, `hint_batch`, ...). Detected via the Batch capability,
/// the catalog `batchable` flag, or well-known batch names.
pub fn is_batch_tool(cap: &ToolCapability) -> bool {
    if is_removed_tool(&cap.name) {
        return false;
    }
    if cap.batchable || cap.capabilities.contains(&Capability::Batch) {
        return true;
    }
    let n = cap.name.to_ascii_lowercase();
    n.contains("execute_plan") || n.contains("hint_batch") || n.contains("batch")
}
/// Launch-type tools create new sessions/windows. Hidden when a usable
/// session already exists (§6: "Open X" never means a second window).
pub fn is_launch_tool(cap: &ToolCapability) -> bool {
    if cap.capabilities.contains(&Capability::Launch) {
        return true;
    }
    let n = cap.name.to_ascii_lowercase();
    n.contains("browser_open")
        || n.contains("browser_launch")
        || n.contains("launch")
        || n.contains("open_application")
}
/// Read-only observation tools. The planner must end every sequence with
/// one of these (§4.3) so the Verifier never needs its own round-trip.
pub fn is_observation_tool(cap: &ToolCapability) -> bool {
    if cap.read_only || cap.capabilities.contains(&Capability::Observe) {
        return true;
    }
    let n = cap.name.to_ascii_lowercase();
    n.contains("snapshot")
        || n.contains("extract")
        || n.contains("observe")
        || n.contains("get_ui")
        || n.contains("ui_tree")
        || n.contains("find_element")
        || n.contains("screenshot")
}

/// Self-resolving interaction tools: one natural-language `instruction` in,
/// the element located at runtime (heuristic → Decider-2B). These are
/// the only interaction verbs a blind plan can complete, because nothing has
/// to be observed first.
pub fn is_self_resolving_interaction(cap: &ToolCapability) -> bool {
    if is_removed_tool(&cap.name) {
        return false;
    }
    let n = cap.name.to_ascii_lowercase();
    n.contains("hint_act")
        || n.contains("hint_batch")
        || n.contains("hint_resolve")
        || n.contains("find_and_click")
        || n.contains("find_and_type")
        || n.contains("execute_plan")
}

/// Single-action CDP verbs that need an element address (`ref` from a
/// snapshot, or a hand-written CSS `selector`).
pub fn needs_element_address(cap: &ToolCapability) -> bool {
    matches!(
        cap.name.to_ascii_lowercase().as_str(),
        "browser_click" | "browser_type" | "browser_hover" | "browser_select_option"
    )
}

/// Tools the planner keeps even when a batch tool covers their domain:
/// navigation (a full-URL step replaces an interaction step outright), the
/// hint pipeline, raw evaluation, verification/pacing, and the desktop
/// natives. Matched by name on purpose — `read_only` is inferred from the
/// description text and is already wrong often enough to be useless here
/// (`tar`+`get` in "target" reads as "get").
pub fn is_planner_core(cap: &ToolCapability) -> bool {
    let n = cap.name.to_ascii_lowercase();
    [
        // navigation + the browser context around it
        "browser_open",
        "browser_navigate",
        "browser_go_back",
        "browser_go_forward",
        "browser_tabs",
        "browser_console",
        "browser_evaluate",
        "browser_wait",
        "browser_screenshot",
        "browser_execute_plan",
        // hint pipeline
        "hint_act",
        "hint_batch",
        "hint_snapshot",
        "hint_click",
        "hint_type",
        "hint_clear",
        "hint_resolve",
        "hint_resolve_batch",
        "find_and_click",
        "find_and_type",
        // verification + pacing
        "verify",
        "verify_action",
        "verify_element",
        "wait_until",
        "wait_for",
        // desktop natives
        "desktop",
        "hypr",
        "launch",
        "ui",
        "click_ui",
        "pointer",
        "keyboard",
        "screenshot",
    ]
    .contains(&n.as_str())
        || n.starts_with("task_")
}
pub fn default_config() -> McpServerConfig {
    McpServerConfig {
        name: "hyprfast".into(),
        command: "hyprfast".into(),
        args: vec!["mcp".into()],
        env: Default::default(),
    }
}
pub fn computer_use_config() -> McpServerConfig {
    McpServerConfig {
        name: "computer_use".into(),
        command: std::env::var("LUCY_COMPUTER_USE_COMMAND").unwrap_or_else(|_| "npx".into()),
        args: std::env::var("LUCY_COMPUTER_USE_ARGS")
            .map(|v| v.split_whitespace().map(str::to_owned).collect())
            .unwrap_or_else(|_| vec!["-y".into(), "@zavora-ai/computer-use-mcp".into()]),
        env: Default::default(),
    }
}
pub fn computer_use_enabled() -> bool {
    std::env::var("LUCY_COMPUTER_USE_ENABLED")
        .map(|v| v != "0" && v.to_ascii_lowercase() != "false")
        .unwrap_or(true)
}

/// Raw MCP tool definitions cached on disk so startup can skip spawning the
/// slow `npx` Computer Use server on every run.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedToolDefs {
    saved_at_secs: u64,
    hyprfast: Vec<McpToolDefinition>,
    computer_use: Vec<McpToolDefinition>,
}
/// Default 24h. `LUCY_MCP_DEFS_TTL_SECS=0` forces a live refresh every run.
const DEFAULT_DEFS_TTL_SECS: u64 = 24 * 3600;
fn defs_cache_path() -> PathBuf {
    std::env::var("LUCY_MCP_DEFS_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
                .join(".local/state/lucy/mcp-tool-defs.json")
        })
}
fn defs_cache_ttl_secs() -> u64 {
    std::env::var("LUCY_MCP_DEFS_TTL_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_DEFS_TTL_SECS)
}
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
fn defs_cache_stale(cache: &CachedToolDefs) -> bool {
    let ttl = defs_cache_ttl_secs();
    if ttl == 0 {
        return true;
    }
    now_secs().saturating_sub(cache.saved_at_secs) > ttl
}
fn load_cached_defs() -> Option<CachedToolDefs> {
    let path = defs_cache_path();
    let data = std::fs::read(path).ok()?;
    serde_json::from_slice(&data).ok()
}
async fn save_cached_defs(hyprfast: &[McpToolDefinition], computer_use: &[McpToolDefinition]) {
    let cache = CachedToolDefs {
        saved_at_secs: now_secs(),
        hyprfast: hyprfast.to_vec(),
        computer_use: computer_use.to_vec(),
    };
    let path = defs_cache_path();
    if let Some(parent) = path.parent() {
        if tokio::fs::create_dir_all(parent).await.is_err() {
            return;
        }
    }
    if let Ok(data) = serde_json::to_vec(&cache) {
        if tokio::fs::write(path, data).await.is_ok() {
            tracing::info!("saved MCP tool-definition cache");
        }
    }
}
/// Resolve Computer Use defs: disabled -> empty, fresh cache -> cache hit (no
/// spawn), otherwise one live `npx` handshake.
async fn resolve_computer_use_defs(
    cached: Option<Vec<McpToolDefinition>>,
) -> Result<Vec<McpToolDefinition>> {
    if !computer_use_enabled() {
        return Ok(Vec::new());
    }
    if let Some(defs) = cached {
        if !defs.is_empty() {
            tracing::info!(
                tools = defs.len(),
                "using cached Computer Use tool definitions"
            );
            return Ok(defs);
        }
    }
    Ok(StdioMcpClient::new(computer_use_config())
        .list_tools()
        .await?)
}
fn classify(tool: &McpToolDefinition) -> ToolCapability {
    let name = tool.name.to_ascii_lowercase();
    let desc = tool
        .description
        .clone()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let text = format!("{} {}", name, desc);
    let domain = if has(&text, &["browser_", "browser "]) {
        Domain::Browser
    } else if has(&text, &["hint_"]) {
        Domain::Hints
    } else if has(&text, &["screenshot", "vision"]) {
        if name.starts_with("computer_use_") {
            Domain::Desktop
        } else {
            Domain::Vision
        }
    } else if has(&text, &["clipboard"]) {
        Domain::Clipboard
    } else if has(&text, &["excalidraw"]) {
        Domain::Excalidraw
    } else if has(&text, &["task_"]) {
        Domain::Tasks
    } else if has(
        &text,
        &[
            "computer_use_",
            "computer use",
            "hypr",
            "desktop",
            "window",
            "pointer",
            "keyboard",
            "click_ui",
            "ui_",
            "application",
            "menu",
            "form",
        ],
    ) {
        Domain::Desktop
    } else {
        Domain::System
    };
    let mut capabilities = Vec::new();
    for (terms, cap) in [
        (
            &["screenshot", "observe", "read", "inspect", "query"][..],
            Capability::Observe,
        ),
        (&["click", "press"][..], Capability::Click),
        (&["type", "fill"][..], Capability::Type),
        (&["keyboard", "key_"][..], Capability::Keyboard),
        (&["pointer", "mouse"][..], Capability::Pointer),
        (&["window", "workspace", "space"][..], Capability::Window),
        (
            &["launch", "open_application", "discover_application"][..],
            Capability::Launch,
        ),
        (&["navigate", "goto", "url"][..], Capability::Navigate),
        (
            &["extract", "text", "content", "element"][..],
            Capability::Extract,
        ),
        (&["act", "action"][..], Capability::Act),
        (&["batch"][..], Capability::Batch),
        (&["task_"][..], Capability::Task),
        (&["clipboard"][..], Capability::Clipboard),
        (&["excalidraw", "draw"][..], Capability::Draw),
        (&["wait"][..], Capability::Wait),
        (&["bind"][..], Capability::Bindings),
    ] {
        if has(&text, terms) {
            capabilities.push(cap)
        }
    }
    if capabilities.is_empty() {
        capabilities.push(Capability::Unknown)
    }
    let read_only = matches!(domain, Domain::Vision)
        || has(
            &text,
            &[
                "screenshot",
                "inspect",
                "get",
                "list",
                "find",
                "query",
                "focused",
                "frontmost",
            ],
        );
    let destructive = has(
        &text,
        &[
            "close", "kill", "delete", "remove", "destroy", "shutdown", "logout",
        ],
    );
    let batchable = has(&text, &["batch"]);
    let operation = if read_only {
        Operation::Observe
    } else if has(&text, &["list", "get", "find", "query", "screenshot"]) {
        Operation::Query
    } else if has(&text, &["draw", "clipboard", "task_"]) {
        Operation::Manage
    } else if has(&text, &["extract"]) {
        Operation::Transform
    } else {
        Operation::Act
    };
    let semantic = has(
        &text,
        &[
            "accessibility",
            "semantic",
            "element",
            "button",
            "form",
            "menu",
            "ui tree",
            "application",
            "window",
        ],
    );
    ToolCapability {
        name: tool.name.clone(),
        description: tool.description.clone().unwrap_or_default(),
        input_schema: tool.input_schema.clone(),
        domain,
        capabilities,
        operation,
        read_only,
        destructive,
        batchable,
        semantic,
    }
}
fn has(text: &str, terms: &[&str]) -> bool {
    terms.iter().any(|term| text.contains(term))
}
#[cfg(test)]
mod tests {
    use super::*;
    fn tool(name: &str, description: &str) -> McpToolDefinition {
        McpToolDefinition {
            name: name.into(),
            description: Some(description.into()),
            input_schema: serde_json::json!({"type":"object"}),
        }
    }
    #[test]
    fn categorizes_tools() {
        let c = HyprFastCatalog::from_tools(vec![
            tool("browser_click", "click browser element"),
            tool("screenshot", "capture desktop"),
            tool("hint_batch", "execute batch hint steps"),
            tool("task_init", "start task"),
            tool(
                "hint_act",
                "Vimium-primary hint_act snapshot+heuristic pick+click",
            ),
            tool(
                "browser_execute_plan",
                "Structured execution plan batch navigate click type",
            ),
        ]);
        assert_eq!(c.tools["browser_click"].domain, Domain::Browser);
        assert!(
            c.tools["browser_click"]
                .capabilities
                .contains(&Capability::Click)
        );
        assert_eq!(c.tools["screenshot"].domain, Domain::Vision);
        assert!(c.tools["hint_batch"].batchable);
        assert_eq!(c.tools["task_init"].domain, Domain::Tasks);
        // Hint pipeline must classify into Hints / Batch.
        assert_eq!(c.tools["hint_act"].domain, Domain::Hints);
        assert!(c.tools["browser_execute_plan"].batchable);
    }

    #[test]
    fn removed_vision_tools_are_never_offered() {
        let c = HyprFastCatalog::from_tools(vec![
            tool("browser_navigate", "CDP: navigate browser tab to URL"),
            tool("hint_act", "PRIMARY browser interaction"),
            tool("ground", "Visual grounding fallback"),
            tool("act_fast", "Fused ground+click"),
            tool("act_batch", "Batch fused steps"),
            tool("stagehand_act", "Old LLM browser tool"),
        ]);
        assert!(is_removed_tool("ground"));
        assert!(is_removed_tool("mcp_hyprfast_act_fast"));
        assert!(is_removed_tool("act_batch"));
        assert!(is_removed_tool("stagehand_act"));
        assert!(!is_removed_tool("hint_act"));
        let all = vec![
            "mcp_hyprfast_browser_navigate".to_string(),
            "mcp_hyprfast_hint_act".to_string(),
            "mcp_hyprfast_ground".to_string(),
            "mcp_hyprfast_act_fast".to_string(),
            "mcp_hyprfast_act_batch".to_string(),
            "mcp_hyprfast_stagehand_act".to_string(),
        ];
        let offered = c.planner_tool_set(&all, false);
        assert!(offered.iter().any(|x| x.contains("hint_act")));
        assert!(offered.iter().any(|x| x.contains("browser_navigate")));
        for banned in ["ground", "act_fast", "act_batch", "stagehand"] {
            assert!(
                !offered.iter().any(|x| x.contains(banned)),
                "{banned} must never be offered: {offered:?}"
            );
        }
    }
    #[test]
    fn planner_tool_set_hides_launch_when_session_ready() {
        let c = HyprFastCatalog::from_tools(vec![
            tool("browser_launch", "launch browser"),
            tool("browser_navigate", "navigate browser to url"),
            tool("browser_snapshot", "observe browser snapshot"),
        ]);
        let all = vec![
            "mcp_hyprfast_browser_launch".to_string(),
            "mcp_hyprfast_browser_navigate".to_string(),
            "mcp_hyprfast_browser_snapshot".to_string(),
        ];
        let ready = c.planner_tool_set(&all, true);
        assert!(
            !ready.iter().any(|x| x.contains("browser_launch")),
            "launch must be hidden when a session exists: {ready:?}"
        );
        assert!(ready.iter().any(|x| x.contains("browser_snapshot")));
        let cold = c.planner_tool_set(&all, false);
        assert!(
            cold.iter().any(|x| x.contains("browser_launch")),
            "launch must be offered when no session exists"
        );
    }
    #[test]
    fn planner_tool_set_prefers_batch_over_singles() {
        let c = HyprFastCatalog::from_tools(vec![
            tool(
                "browser_execute_plan",
                "execute batch plan of browser steps",
            ),
            tool("browser_navigate", "navigate browser to url"),
            tool("browser_click", "click browser element"),
            tool("browser_snapshot", "observe browser snapshot"),
        ]);
        let all = vec![
            "mcp_hyprfast_browser_execute_plan".to_string(),
            "mcp_hyprfast_browser_navigate".to_string(),
            "mcp_hyprfast_browser_click".to_string(),
            "mcp_hyprfast_browser_snapshot".to_string(),
        ];
        let offered = c.planner_tool_set(&all, true);
        assert!(
            offered.iter().any(|x| x.contains("execute_plan")),
            "batch tool must survive filtering"
        );
        // `browser_navigate` is planner-core: a full-URL navigation step removes
        // an interaction step outright, so hiding it behind the batch tool is a
        // regression (this assertion used to demand the opposite).
        assert!(
            offered.iter().any(|x| x.contains("browser_navigate")),
            "navigate must survive filtering: {offered:?}"
        );
        assert!(!offered.iter().any(|x| x.contains("browser_click")));
        assert!(
            offered.iter().any(|x| x.contains("browser_snapshot")),
            "observation must always survive filtering"
        );
    }
    /// The live hyprfast catalog shape for a browser goal, as `classify()`
    /// actually reads it: `hint_act`'s description names `browser_click`, so it
    /// classifies into Browser and only the name-based rules can save it.
    fn live_browser_catalog() -> HyprFastCatalog {
        HyprFastCatalog::from_tools(vec![
            tool(
                "browser_navigate",
                "CDP: navigate browser tab to URL (auto-discovers ws://9222, creates tab if needed)",
            ),
            tool(
                "browser_open",
                "Hypr+CDP: launch Brave with --remote-debugging-port=9222 on workspace and navigate",
            ),
            tool(
                "browser_click",
                "CDP: click element. Use ref from snapshot or CSS selector via element.",
            ),
            tool(
                "browser_type",
                "CDP: type text into editable element (ref from snapshot)",
            ),
            tool("browser_hover", "CDP: hover element"),
            tool("browser_select_option", "CDP: select option in dropdown"),
            tool(
                "browser_evaluate",
                "CDP: evaluate JavaScript in page (Runtime.evaluate)",
            ),
            tool("browser_wait", "CDP: wait N seconds (browser)"),
            tool("browser_tabs", "CDP: list browser tabs/targets (GET /json)"),
            tool("browser_go_back", "CDP: go back (history.back)"),
            tool(
                "browser_screenshot",
                "CDP: capture browser tab screenshot via Page.captureScreenshot",
            ),
            tool(
                "browser_snapshot",
                "CDP: capture accessibility snapshot (AX tree). Returns refs for click/type.",
            ),
            tool(
                "hint_act",
                "PRIMARY browser interaction. Give ONE natural-language instruction; it snapshots clickable elements, resolves the target itself (heuristic -> Decider-2B), then clicks or types. Use this instead of browser_click/browser_type when you have no snapshot ref.",
            ),
            tool(
                "hint_batch",
                "Batch hint actions: ONE snapshot, resolve all instructions, dispatch in parallel (max 12 steps).",
            ),
            tool(
                "browser_execute_plan",
                "Run a structured multi-step browser plan in one call: {steps:[{action: navigate|click|type|select|press|hover|wait|eval|extract, ...}]}.",
            ),
            tool(
                "hint_resolve",
                "Hint resolve: hint_snapshot -> Decider -> hint_click (single instruction)",
            ),
            tool(
                "find_and_click",
                "Composite: find + click (semantic find then hint/browser click)",
            ),
            tool(
                "find_and_type",
                "Composite: find + type (semantic find then hint/browser type)",
            ),
            tool(
                "verify",
                "Verify (DOM first, Decider visual only when necessary) -> success/failure/uncertain",
            ),
            tool(
                "wait_until",
                "Wait until predicate via DOM/AX polling + visual fallback",
            ),
            tool("desktop", "Instant desktop snapshot (no screenshot, <5ms)"),
            tool(
                "hypr",
                "Window/workspace ops: workspace/focus_window/move_window",
            ),
            tool(
                "pointer",
                "Mouse: move|click|drag|scroll at global logical coords",
            ),
            tool(
                "keyboard",
                "Keyboard: type (text) or key (combo like ctrl+t). window focuses first.",
            ),
            tool(
                "task_init",
                "Task state: init todo list for multi-step action",
            ),
        ])
    }
    fn all_candidates(catalog: &HyprFastCatalog) -> Vec<String> {
        let mut names: Vec<String> = catalog.tools.values().map(|t| full_name(&t.name)).collect();
        names.sort();
        names
    }
    #[test]
    fn planner_tool_set_is_hint_first_on_the_live_catalog() {
        let c = live_browser_catalog();
        let offered = c.planner_tool_set(&all_candidates(&c), false);
        let bare: Vec<&str> = offered
            .iter()
            .map(|n| n.trim_start_matches("mcp_hyprfast_"))
            .collect();
        for kept in [
            "hint_act",
            "hint_batch",
            "browser_navigate",
            "browser_open",
            "browser_evaluate",
            "browser_wait",
            "browser_tabs",
            "browser_go_back",
            "find_and_click",
            "find_and_type",
            "hint_resolve",
            "verify",
            "wait_until",
            "desktop",
            "hypr",
            "pointer",
            "keyboard",
            "task_init",
        ] {
            assert!(bare.contains(&kept), "planner lost {kept}: {bare:?}");
        }
        for hidden in [
            "browser_click",
            "browser_type",
            "browser_hover",
            "browser_select_option",
        ] {
            assert!(!bare.contains(&hidden), "{hidden} must be hidden: {bare:?}");
        }
    }
    #[test]
    fn planner_tool_set_keeps_ref_needing_verbs_without_a_self_resolving_tool() {
        // Stale cache from an older hyprfast: no hint pipeline at all. Hiding
        // the ref-needing verbs then leaves the planner with no way to interact.
        let c = HyprFastCatalog::from_tools(vec![
            tool("browser_navigate", "CDP: navigate browser tab to URL"),
            tool(
                "browser_click",
                "CDP: click element. Use ref from snapshot or CSS selector.",
            ),
            tool(
                "browser_type",
                "CDP: type text into editable element (ref from snapshot)",
            ),
            tool(
                "browser_snapshot",
                "CDP: capture accessibility snapshot (AX tree)",
            ),
        ]);
        let all = all_candidates(&c);
        let offered = c.planner_tool_set(&all, false);
        assert_eq!(
            offered.len(),
            all.len(),
            "nothing may be hidden: {offered:?}"
        );
    }
    #[test]
    fn planner_tool_set_never_returns_empty() {
        let c = HyprFastCatalog::from_tools(vec![
            tool("hint_batch", "Batch hint actions: dispatch in parallel"),
            tool(
                "hint_act",
                "PRIMARY browser interaction. resolves the target itself",
            ),
        ]);
        let all = all_candidates(&c);
        let offered = c.planner_tool_set(&all, false);
        assert!(!offered.is_empty());
    }
    #[test]
    fn domain_summary_line_counts() {
        let c = HyprFastCatalog::from_tools(vec![
            tool("browser_click", "click browser element"),
            tool("browser_navigate", "navigate browser"),
        ]);
        let line = c.domain_summary_line();
        assert!(line.contains("Browser(2)"), "got {line:?}");
    }
}
#[cfg(test)]
mod cache_tests {
    use super::*;
    #[test]
    fn computer_use_toggle_defaults_on() {
        unsafe { std::env::remove_var("LUCY_COMPUTER_USE_ENABLED") };
        assert!(computer_use_enabled());
        unsafe { std::env::set_var("LUCY_COMPUTER_USE_ENABLED", "0") };
        assert!(!computer_use_enabled());
        unsafe { std::env::set_var("LUCY_COMPUTER_USE_ENABLED", "false") };
        assert!(!computer_use_enabled());
        unsafe { std::env::remove_var("LUCY_COMPUTER_USE_ENABLED") };
    }
    #[test]
    fn defs_cache_freshness_follows_ttl() {
        unsafe { std::env::remove_var("LUCY_MCP_DEFS_TTL_SECS") };
        let fresh = CachedToolDefs {
            saved_at_secs: now_secs(),
            hyprfast: vec![],
            computer_use: vec![],
        };
        assert!(!defs_cache_stale(&fresh));
        let ancient = CachedToolDefs {
            saved_at_secs: 0,
            hyprfast: vec![],
            computer_use: vec![],
        };
        assert!(defs_cache_stale(&ancient));
        unsafe { std::env::set_var("LUCY_MCP_DEFS_TTL_SECS", "0") };
        assert!(defs_cache_stale(&fresh));
        unsafe { std::env::remove_var("LUCY_MCP_DEFS_TTL_SECS") };
    }
    #[test]
    fn defs_cache_roundtrip() {
        let dir = std::env::temp_dir().join(format!("lucy-defs-cache-{}", now_secs()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("defs.json");
        unsafe { std::env::set_var("LUCY_MCP_DEFS_CACHE", path.to_str().unwrap()) };
        assert!(load_cached_defs().is_none());
        let defs = vec![McpToolDefinition {
            name: "click".into(),
            description: None,
            input_schema: serde_json::json!({}),
        }];
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(save_cached_defs(&[], &defs));
        let loaded = load_cached_defs().expect("cache should exist after save");
        assert!(!defs_cache_stale(&loaded));
        assert_eq!(loaded.computer_use.len(), 1);
        assert_eq!(loaded.computer_use[0].name, "click");
        unsafe { std::env::remove_var("LUCY_MCP_DEFS_CACHE") };
        let _ = std::fs::remove_dir_all(dir);
    }
}
