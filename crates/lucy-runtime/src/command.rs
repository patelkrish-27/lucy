//! Whiteboard command pipeline: one `LucyCommand` in, one goal state out.
//!
//! This module is the code form of the architecture sketch:
//!
//! ```text
//! Lucy command (own STT → text; TTS beep acknowledges)
//!     │
//!     ▼
//! Identify: does the command need only a response,
//!           or actions to reach a goal state?
//!     │
//!     ├── requires only response ──► decide reasoning level 1 / 2 / 3
//!     │                              └── answer with L1 (Flash-lite) /
//!     │                                  L2 (Flash) / L3 (Pro) model
//!     │
//!     └── needs action ──► execute_goal_outcome ──► ReAct
//!                            │
//!                            ▼
//!                          user goal ──► perceive (screen, URL)
//!                                          │
//!                                          ▼
//!                                        decide ── 1 LLM call ──► tool_calls
//!                                          │
//!                                          ▼
//!                                        execute ── the named calls, in order,
//!                                        │          each gated, each observed
//!                                          │
//!                                          └──► observe ──► decide ──► …
//!                                                 │
//!                                                 ▼
//!                                          the page confirms the goal ──► done
//!                                                 └──► budget spent / claim refuted ──► honest outcome
//! ```
//!
//! One act entry, one loop. There is no second "agent mode" beside the plain
//! prompt: `/agent`, `lucy act`, `lucy agent`, a typed prompt that needs
//! actions and the gateway all reach `execute_goal_outcome`, so a goal cannot
//! be run one way in the TUI and another way in the CLI.
//!
//! The stages map onto the existing runtime pieces so this file stays the
//! single place that documents the order:
//! - identify → [`lucy_systemone::TurnBranch`] via `classify_turn`
//! - reasoning level → [`lucy_config::ReasoningLevel`] via `route_turn`
//! - respond → `answer_turn` on the tier's model
//! - act → `execute_goal_outcome`, the one entry into [`react`], which owns the
//!   perceive → decide → execute → observe cycle, the budgets and the screen
//!   oracle that grades a completion claim
//!
//! What is deliberately *not* here: a tool or a site list. The loop reads the
//! live tool catalog and the goal, so a new site or a new way of asking is a
//! model decision, never a code change. See `AGENTS.md`.

use lucy_config::ReasoningLevel;
use lucy_hyprfast::{HyprFastCatalog, full_name};
use lucy_systemone::TurnBranch;
use tracing::warn;

use super::router::filter_tool_brief;

/// Where a command's text came from. Both feed the same pipeline; voice just
/// plays the TTS-beep acknowledgment first (`lucy_stt::ack_beep`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandSource {
    Typed,
    Voice,
}

/// One user command flowing through the pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LucyCommand {
    pub text: String,
    pub source: CommandSource,
}

impl LucyCommand {
    pub fn typed(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            source: CommandSource::Typed,
        }
    }

    pub fn voice(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            source: CommandSource::Voice,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty()
    }
}

/// Whiteboard stage 1 verdict: response-only, or actions to a goal state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandKind {
    RequiresOnlyResponse,
    NeedsAction,
}

impl CommandKind {
    pub fn from_branch(branch: TurnBranch) -> Self {
        match branch {
            TurnBranch::RequiresOnlyResponse => Self::RequiresOnlyResponse,
            TurnBranch::RequiresActions => Self::NeedsAction,
        }
    }

    pub fn needs_action(&self) -> bool {
        matches!(self, Self::NeedsAction)
    }
}

/// Whiteboard stage 2: the tier's canonical model family.
///
/// These are family names (`flash-lite` / `flash` / `pro`); the actual
/// endpoint is whatever `provider_id/model` the user bound to the tier in
/// `/settings` (see `LucyConfig::resolve_level_model`, which degrades an
/// unbound tier to the next cheaper one).
pub fn tier_model_family(level: ReasoningLevel) -> &'static str {
    match level {
        ReasoningLevel::L1 => "flash-lite",
        ReasoningLevel::L2 => "flash",
        ReasoningLevel::L3 => "pro",
    }
}

/// How many chars of the hyprfast SKILL.md the L3 planner sees. The full file
/// (~34KB) would drown the goal. The cut used to land at 12k, mid-table and
/// past the load-bearing sections: "When Hints Fail → Fallback Chain" and
/// "Browser Automation (CDP)" (including `HYPRFAST_CDP_HOST`/`PORT`) both sat
/// at ~12.4k and ~12.8k, so the budget amputated exactly the tiebreakers the
/// surviving half of the file depends on. 16k keeps the CDP + fallback sections
/// and the Brave/Chromium gotchas, and [`truncate_skill`] now cuts on a
/// section boundary so a shorter budget degrades to whole sections instead of a
/// shredded table. The facts that must survive at *any* budget are in
/// [`super::turn::PLANNER_OPERATING_RULES`], which is never truncated.
pub const ACTION_SKILL_BUDGET: usize = 16_000;

/// Whiteboard stage 3 (act branch): hyprfast SKILL + user request → L3 prompt.
///
/// `skill_body` is the full `skills/lucy/SKILL.md`; `catalog_brief` is the
/// `{tool_catalog_brief}` with exact tool names + input schemas; `catalog`, when
/// present, narrows that brief to the hint-first planner tool set
/// ([`HyprFastCatalog::planner_tool_set`]) so the planner is never *offered* a
/// verb a blind plan cannot complete. `knowledge` is the already-budgeted
/// knowledge section from `super::knowledge`, empty when there is nothing worth
/// saying — it sits *before* the skill body so a truncated skill cannot push it
/// out, which is the same reason [`super::turn::PLANNER_OPERATING_RULES`] lives
/// outside the budget.
pub fn render_action_plan_prompt(
    skill_body: &str,
    catalog: Option<&HyprFastCatalog>,
    catalog_brief: &str,
    goal: &str,
    history: &str,
    knowledge: &str,
) -> String {
    let (brief, note) = planner_brief(catalog, catalog_brief);
    let skill = truncate_skill(skill_body.trim(), ACTION_SKILL_BUDGET);
    format!(
        "{instructions}\n{rules}\n{knowledge}## Hyprfast skill (recipes — consult before using its tools)\n{skill}\n\n## Available tools\n{brief}\n\n{note}\n## Recent history\n{history}\n\n## Request\n{goal}\n",
        instructions = super::turn::PLAN_INSTRUCTIONS,
        rules = super::turn::PLANNER_OPERATING_RULES,
        knowledge = knowledge,
        skill = if skill.is_empty() {
            "(skill unavailable — use the tool catalog below)".to_owned()
        } else {
            skill
        },
    )
}

/// Tools the planner is offered: `catalog_brief` minus everything
/// [`HyprFastCatalog::planner_tool_set`] hides. Only `mcp_hyprfast_*` lines are
/// candidates — local tools and any other MCP server keep their line, since
/// the hyprfast catalog says nothing about them.
///
/// Falls back to the unfiltered brief (logged) when there is no catalog, when
/// the catalog is empty, or when filtering would leave the planner with nothing.
///
/// Crate-visible because "which tools the model is offered" has to be one
/// answer, not one per loop: the blind plan ([`render_action_plan_prompt`]) and
/// the ReAct loop both read it from here, so a tool that is hidden from one is
/// hidden from both.
pub(crate) fn planner_brief(catalog: Option<&HyprFastCatalog>, catalog_brief: &str) -> (String, String) {
    let Some(catalog) = catalog.filter(|c| !c.is_empty()) else {
        warn!("no hyprfast catalog — the planner sees every discovered tool");
        return (catalog_brief.to_owned(), String::new());
    };
    let mut candidates: Vec<String> = catalog.tools.values().map(|t| full_name(&t.name)).collect();
    candidates.sort();
    // `browser_ready` is false on purpose: nothing here has probed CDP, and
    // hiding `browser_open` would strand a first run with no browser at all.
    let offered = catalog.planner_tool_set(&candidates, false);
    if offered.is_empty() {
        warn!("planner tool filtering produced an empty set — offering the full catalog");
        return (catalog_brief.to_owned(), String::new());
    }
    let offered: std::collections::HashSet<String> = offered.into_iter().collect();
    let offered_count = offered.len();
    let brief = filter_tool_brief(catalog_brief, |line| {
        let name = line.trim();
        !name.starts_with("mcp_hyprfast_") || offered.contains(name)
    });
    let note = format!(
        "Only {offered_count} of {} hyprfast tools above are usable from a blind plan: the single-action\nbrowser verbs that need a snapshot ref are hidden. Interact with `hint_act`.\n",
        candidates.len()
    );
    (brief, note)
}

/// Cap `s` at `max` chars, cutting on the last `## ` section boundary so the
/// tail is never a shredded table. Falls back to a hard cut for text with no
/// headings.
pub(crate) fn truncate_skill(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_owned();
    }
    let head: String = s.chars().take(max).collect();
    let boundary = [head.rfind("\n## "), head.rfind("\n# ")]
        .into_iter()
        .flatten()
        .max()
        .filter(|cut| *cut > max / 2);
    match boundary {
        Some(cut) => format!("{}\n… (skill body truncated)", head[..cut].trim_end()),
        None => truncate_chars(s, max),
    }
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_owned();
    }
    let mut out: String = s.chars().take(max).collect();
    out.push('…');
    out
}

/// One-line pipeline summary for the chat log / CLI, e.g.
/// `chat · L2 (balanced) · Groq · flash` or `act · L3 (pro model) → ReAct loop`.
///
/// The act branch reports no step count. The ReAct loop has no plan to count:
/// the model names the next call after seeing the last one, so a total is only
/// known once the run is over, and it is in the outcome's own stats. A step
/// count here would describe a plan that no longer exists.
pub fn pipeline_summary(
    kind: CommandKind,
    level: ReasoningLevel,
    model_label: &str,
    planned_steps: Option<usize>,
) -> String {
    match kind {
        CommandKind::RequiresOnlyResponse => {
            format!(
                "chat · {} ({}) · {}",
                level.short(),
                level.label(),
                model_label
            )
        }
        CommandKind::NeedsAction => {
            let _ = planned_steps;
            format!(
                "act · {} ({} model) → ReAct loop: decide, act, observe, repeat",
                level.short(),
                tier_model_family(level)
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucy_mcp::McpToolDefinition;

    fn def(name: &str, description: &str) -> McpToolDefinition {
        McpToolDefinition {
            name: name.into(),
            description: Some(description.into()),
            input_schema: serde_json::json!({"type": "object"}),
        }
    }

    #[test]
    fn branch_maps_to_command_kind() {
        assert_eq!(
            CommandKind::from_branch(TurnBranch::RequiresOnlyResponse),
            CommandKind::RequiresOnlyResponse
        );
        assert_eq!(
            CommandKind::from_branch(TurnBranch::RequiresActions),
            CommandKind::NeedsAction
        );
        assert!(CommandKind::NeedsAction.needs_action());
        assert!(!CommandKind::RequiresOnlyResponse.needs_action());
    }

    #[test]
    fn tiers_map_to_flash_lite_flash_pro() {
        assert_eq!(tier_model_family(ReasoningLevel::L1), "flash-lite");
        assert_eq!(tier_model_family(ReasoningLevel::L2), "flash");
        assert_eq!(tier_model_family(ReasoningLevel::L3), "pro");
    }

    #[test]
    fn action_prompt_carries_skill_catalog_and_goal() {
        let out = render_action_plan_prompt(
            "# lucy skill\nuse hint_act wisely",
            None,
            "browser_navigate — Navigate",
            "open yt & play this song",
            "(none)",
            "",
        );
        assert!(out.contains("lucy skill"));
        assert!(out.contains("browser_navigate"));
        assert!(out.contains("open yt & play this song"));
        assert!(out.contains(r#"{"commands""#));
        assert!(out.contains("strictly ordered"));
    }

    /// The rendered planner prompt is what a weak model actually reads: it
    /// must offer `hint_act` and ban the tools hyprfast removed.
    #[test]
    fn action_prompt_names_hint_act_and_no_removed_tool() {
        let out = render_action_plan_prompt(
            "# lucy skill",
            None,
            "mcp_hyprfast_hint_act — PRIMARY browser interaction\nmcp_hyprfast_browser_navigate — CDP",
            "play despacito song on yt",
            "(none)",
            "",
        );
        assert!(out.contains("`hint_act` is the PRIMARY"), "{out}");
        assert!(out.contains("mcp_hyprfast_hint_act"));
        let lower = out.to_lowercase();
        // Removed tools are banned, never offered: the prompt must carry the
        // ban and must not carry keys/config for them.
        assert!(out.contains("Never emit"), "{out}");
        assert!(!lower.contains("ground.env"), "{out}");
        assert!(!lower.contains("gemini_api"), "{out}");
    }

    /// The load-bearing facts must survive any skill budget: they live outside
    /// the truncated body.
    #[test]
    fn action_prompt_keeps_operating_rules_outside_the_budget() {
        let big = "x".repeat(ACTION_SKILL_BUDGET * 3);
        let out = render_action_plan_prompt(&big, None, "catalog", "goal", "(none)", "");
        for fact in [
            "HYPRFAST_CDP_HOST",
            "--remote-debugging-port=9222",
            "Hint + Decider",
            "browser_evaluate",
        ] {
            assert!(out.contains(fact), "operating rules lost {fact}");
        }
        // The body really is capped, and the cap is marked.
        assert!(out.contains('…'), "uncapped skill body");
        assert!(out.chars().count() < ACTION_SKILL_BUDGET * 2);
    }

    /// L3: enforcement by hiding. A catalog advertising the ref-needing single
    /// verbs must not have them offered to the planner.
    #[test]
    fn action_prompt_hides_redundant_browser_verbs() {
        let catalog = HyprFastCatalog::from_tools(vec![
            def("browser_navigate", "CDP: navigate browser tab to URL"),
            def("browser_open", "Hypr+CDP: launch Brave and navigate"),
            def(
                "browser_click",
                "CDP: click element. Use ref from snapshot.",
            ),
            def(
                "browser_type",
                "CDP: type text into editable element (ref from snapshot)",
            ),
            def(
                "browser_snapshot",
                "CDP: capture accessibility snapshot (AX tree)",
            ),
            def(
                "hint_act",
                "PRIMARY browser interaction. resolves the target itself (heuristic -> Decider-2B -> vision)",
            ),
        ]);
        let brief = "## Local tools\nshell — run a command\n\
                     ## HyprFast MCP tools (desktop/window/browser/hints/tasks)\n\
                     mcp_hyprfast_browser_click — CDP: click element\n\
                     mcp_hyprfast_browser_navigate — CDP: navigate\n\
                     mcp_hyprfast_browser_open — Hypr+CDP: launch Brave\n\
                     mcp_hyprfast_browser_snapshot — CDP: accessibility snapshot\n\
                     mcp_hyprfast_browser_type — CDP: type text\n\
                     mcp_hyprfast_hint_act — PRIMARY browser interaction";
        let out = render_action_plan_prompt("", Some(&catalog), brief, "play a song", "(none)", "");
        assert!(out.contains("mcp_hyprfast_hint_act"), "{out}");
        assert!(out.contains("mcp_hyprfast_browser_navigate"), "{out}");
        assert!(!out.contains("mcp_hyprfast_browser_click"), "{out}");
        assert!(!out.contains("mcp_hyprfast_browser_type"), "{out}");
        // Non-hyprfast lines survive: the filter only owns the hyprfast catalog.
        assert!(out.contains("shell — run a command"), "{out}");
    }

    /// A stale cache with no hint pipeline must not leave the planner with no
    /// interaction verb at all.
    #[test]
    fn action_prompt_degrades_when_the_catalog_has_no_hint_tool() {
        let catalog = HyprFastCatalog::from_tools(vec![
            def("browser_navigate", "CDP: navigate browser tab to URL"),
            def(
                "browser_click",
                "CDP: click element. Use ref from snapshot.",
            ),
        ]);
        let brief = "## HyprFast MCP tools\n\
                     mcp_hyprfast_browser_click — CDP: click element\n\
                     mcp_hyprfast_browser_navigate — CDP: navigate";
        let out = render_action_plan_prompt("", Some(&catalog), brief, "play a song", "(none)", "");
        assert!(out.contains("mcp_hyprfast_browser_click"), "{out}");
        assert!(out.contains("mcp_hyprfast_browser_navigate"), "{out}");
    }

    /// Truncation cuts on a section boundary so the planner never reads half a
    /// markdown table.
    #[test]
    fn skill_truncation_snaps_to_a_section_boundary() {
        let body = (0..8)
            .map(|n| {
                format!(
                    "# Section {n}\n| a | b |\n|---|---|\n{}\nfiller\n",
                    "-".repeat(300)
                )
            })
            .collect::<String>();
        let cut = truncate_skill(&body, 900);
        let head = cut
            .strip_suffix("\n… (skill body truncated)")
            .expect("cut must be marked");
        // A boundary cut is a true prefix of the body: no line survives in half.
        assert!(body.starts_with(head.trim_end()), "cut mid-line:\n{cut}");
        assert!(head.trim_end().ends_with("filler"), "{cut}");
        assert!(cut.chars().count() <= 940, "cut overshot the budget");
    }

    #[test]
    fn long_skill_is_capped() {
        let big = "x".repeat(ACTION_SKILL_BUDGET + 100);
        let out = render_action_plan_prompt(&big, None, "catalog", "goal", "(none)", "");
        // Skill body is truncated to the budget; the rest is fixed prompt
        // instructions plus tiny catalog/goal/history slots.
        let fixed_overhead = crate::turn::PLAN_INSTRUCTIONS.chars().count()
            + crate::turn::PLANNER_OPERATING_RULES.chars().count()
            + "cataloggoal(none)".len()
            + 500;
        assert!(out.chars().count() < big.chars().count() + fixed_overhead);
        assert!(out.contains('…'));
    }

    #[test]
    fn empty_skill_degrades_to_catalog() {
        let out = render_action_plan_prompt("", None, "catalog", "goal", "(none)", "");
        assert!(out.contains("skill unavailable"));
        assert!(out.contains("catalog"));
    }

    /// The exact regression: at 12k the cut landed mid-table and amputated
    /// "When Hints Fail" (~12.4k) and "Browser Automation (CDP)" (~12.8k), so the
    /// surviving half of the file stated the Hint+Decider priority and then lost
    /// the tiebreaker and the CDP facts.
    #[test]
    fn real_skill_keeps_the_fallback_chain_and_cdp_sections() {
        let cut = truncate_skill(crate::router::LUCY_SKILL.trim(), ACTION_SKILL_BUDGET);
        for section in [
            "## Lucy Agent — Primary Workflow (Hint + Decider Pipeline)",
            "### Priority Order (MUST follow)",
            "### When Hints Fail → Fallback Chain",
            "## Browser Automation (CDP)",
            "## No-vision fallback (no AX tree)",
            "## Brave/Chromium gotchas",
        ] {
            assert!(cut.contains(section), "budget amputated {section}");
        }
        for fact in [
            "HYPRFAST_CDP_HOST",
            "--remote-debugging-port=9222",
            "auto-adds CDP flags",
        ] {
            assert!(cut.contains(fact), "budget amputated {fact}");
        }
        assert!(cut.chars().count() <= ACTION_SKILL_BUDGET + 64);
    }

    #[test]
    fn summaries_read_like_the_whiteboard() {
        let chat = pipeline_summary(
            CommandKind::RequiresOnlyResponse,
            ReasoningLevel::L2,
            "Groq · flash",
            None,
        );
        assert!(chat.contains("chat") && chat.contains("L2"), "{chat}");
        let act = pipeline_summary(CommandKind::NeedsAction, ReasoningLevel::L3, "Groq · pro", None);
        assert!(act.contains("act") && act.contains("ReAct"), "{act}");
    }

    /// The act summary names the loop, never a step count: the ReAct loop has
    /// no plan, so a count taken from the caller would be describing a plan
    /// that does not exist. Holds for any `planned_steps` the caller holds.
    #[test]
    fn act_summary_carries_no_step_count() {
        for n in [None, Some(0), Some(1), Some(99)] {
            let act = pipeline_summary(CommandKind::NeedsAction, ReasoningLevel::L3, "p", n);
            assert!(!act.contains("step"), "step count leaked into {act}");
            assert!(!act.contains('4'), "a literal count leaked into {act}");
            assert!(act.contains("ReAct"), "{act}");
        }
    }

    #[test]
    fn voice_and_typed_share_the_pipeline() {
        let v = LucyCommand::voice("play despacito");
        let t = LucyCommand::typed("play despacito");
        assert_eq!(v.text, t.text);
        assert_ne!(v.source, t.source);
        assert!(!v.is_empty());
        assert!(LucyCommand::typed("   ").is_empty());
    }
}
