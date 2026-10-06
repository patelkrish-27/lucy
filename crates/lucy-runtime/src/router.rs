//! Capability discovery helpers: the tool catalog brief, the skills on disk,
//! and the subtask decomposition template used by the action-branch fallback.
//!
//! Intent classification itself lives in [`super::turn`] (`classify_turn` →
//! `decider-serve`); this module only supplies the static material those
//! calls render into their prompts.

use anyhow::Result;
use lucy_core::TurnMessage;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;

pub const SUBTASKS_TEMPLATE: &str = include_str!("../../../prompts/subtasks.md");

/// One-line `name — description` catalog for the `{tool_catalog_brief}` slot.
pub fn tool_catalog_brief() -> String {
    let registry = lucy_tools::default_registry();
    let mut defs = registry.definitions();
    defs.sort_by(|a, b| {
        a.get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .cmp(b.get("name").and_then(Value::as_str).unwrap_or_default())
    });
    defs.iter()
        .filter_map(|d| {
            let name = d.get("name")?.as_str()?;
            let desc = d.get("description")?.as_str().unwrap_or("");
            Some(format!("{name} — {desc}"))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Compact recent history for the `{history}` slot. `(none)` when empty.
pub fn format_history(history: &[TurnMessage], limit: usize) -> String {
    let turns: Vec<String> = history
        .iter()
        .rev()
        .take(limit.max(1))
        .rev()
        .map(|m| match m {
            TurnMessage::User(t) => format!("user: {t}"),
            TurnMessage::Assistant(turn) => {
                format!("assistant: {}", turn.text.clone().unwrap_or_default())
            }
            TurnMessage::Tool(res) => format!(
                "tool {} -> {}",
                res.name,
                res.output.to_string().chars().take(300).collect::<String>()
            ),
        })
        .collect();
    if turns.is_empty() {
        "(none)".to_owned()
    } else {
        turns.join("\n")
    }
}

/// One step of a decomposed `act` goal: a single concrete action plus the
/// observable fact proving it is done. Produced by the main LLM, executed
/// in order by the automation loop.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Subtask {
    pub index: usize,
    pub description: String,
    pub success_condition: String,
}

impl Subtask {
    /// Atomic fallback: the goal itself as a single subtask, used when the
    /// decomposition call fails or returns nothing usable.
    pub fn fallback(goal: &str) -> Self {
        Self {
            index: 1,
            description: goal.trim().to_owned(),
            success_condition: "The requested end state is visibly complete.".into(),
        }
    }
}

pub fn render_subtasks_prompt(user_request: &str) -> String {
    SUBTASKS_TEMPLATE.replace("{user_request}", user_request)
}

/// Parse the main-LLM decomposition (`{"subtasks":[...]}`) into a numbered,
/// cleaned list. Rejects empty descriptions; caps at 8; renumbers 1-based.
/// Returns an empty vec when nothing usable is present (caller falls back
/// to [`Subtask::fallback`]).
pub fn parse_subtasks(value: &Value) -> Vec<Subtask> {
    let arr = match value.get("subtasks").and_then(Value::as_array) {
        Some(a) => a,
        None => return Vec::new(),
    };
    arr.iter()
        .filter_map(|t| {
            let description = t
                .get("description")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())?;
            let success_condition = t
                .get("success_condition")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .unwrap_or("This step's result is visible on screen.");
            Some((description.to_owned(), success_condition.to_owned()))
        })
        .take(8)
        .enumerate()
        .map(|(i, (description, success_condition))| Subtask {
            index: i + 1,
            description,
            success_condition,
        })
        .collect()
}

/// Numbered human-readable checklist shared by CLI and TUI.
pub fn format_subtasks(subtasks: &[Subtask]) -> String {
    subtasks
        .iter()
        .map(|s| format!("  {}. {} ({})", s.index, s.description, s.success_condition))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Vendored lucy skill (`skills/lucy/SKILL.md` in this repo): owns the
/// local file/shell tool recipes plus the Hyprland/desktop/CDP-browser/
/// Stagehand/task tool recipes (previously the `hyprfast` skill, now
/// embedded here so there is exactly one skill to consult). Embedded at
/// compile time and seeded onto disk once (see [`seed_lucy_skill`]) so
/// auto-discovery finds it even when lucy runs outside the repo checkout.
pub const LUCY_SKILL: &str = include_str!("../../../skills/lucy/SKILL.md");

/// A discovered skill: `skills/<name>/SKILL.md` (or any dir scanned by
/// [`discover_skills`]). New skills are picked up automatically at startup —
/// no code change, no recompile.
#[derive(Debug, Clone)]
pub struct SkillInfo {
    pub name: String,
    pub description: String,
    pub path: PathBuf,
}

impl SkillInfo {
    pub fn body(&self) -> Result<String> {
        Ok(std::fs::read_to_string(&self.path)?)
    }
}

/// Skill search locations, first match wins on name collisions:
/// `$LUCY_SKILLS_DIR`, then `~/.config/lucy/skills` (primary home for
/// user-added skills), then `./skills` (repo checkout / dev fallback).
pub fn skill_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(p) = std::env::var("LUCY_SKILLS_DIR") {
        if !p.trim().is_empty() {
            dirs.push(PathBuf::from(p));
        }
    }
    dirs.push(
        PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
            .join(".config/lucy/skills"),
    );
    dirs.push(PathBuf::from("skills"));
    dirs
}

/// Scan `dirs` for `*/SKILL.md` files. `name:`/`description:` come from the
/// front-matter; missing `name:` falls back to the directory name. Entries
/// without a readable `SKILL.md` are ignored.
pub fn discover_skills(dirs: &[PathBuf]) -> Vec<SkillInfo> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for dir in dirs {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let mut candidates: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_dir())
            .collect();
        candidates.sort();
        for candidate in candidates {
            let file = candidate.join("SKILL.md");
            let text = match std::fs::read_to_string(&file) {
                Ok(t) => t,
                Err(_) => continue,
            };
            let name = front_matter_field(&text, "name")
                .filter(|n| !n.trim().is_empty())
                .or_else(|| {
                    candidate
                        .file_name()
                        .and_then(|n| n.to_str())
                        .map(str::to_owned)
                })
                .unwrap_or_default();
            if name.trim().is_empty() || !seen.insert(name.clone()) {
                continue;
            }
            let description = front_matter_field(&text, "description").unwrap_or_default();
            out.push(SkillInfo {
                name,
                description,
                path: file,
            });
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Brief lines for the Skills section of the catalog brief.
pub fn skill_brief_lines(skills: &[SkillInfo]) -> Vec<String> {
    skills
        .iter()
        .map(|s| {
            let desc = truncate_one_line(&s.description, 800);
            if desc.is_empty() {
                format!("{} — (see {})", s.name, s.path.display())
            } else {
                format!("{} — {desc}", s.name)
            }
        })
        .collect()
}

/// Ensure the vendored lucy skill exists on disk so auto-discovery finds
/// it. Writes `LUCY_SKILL` to the primary skills dir only when no
/// `lucy/SKILL.md` exists in any search location — never overwrites user
/// edits. A stale `hyprfast/SKILL.md` from before the rename is removed so
/// the router sees exactly one base skill.
pub fn seed_lucy_skill() -> Result<PathBuf> {
    for dir in skill_dirs() {
        let existing = dir.join("lucy").join("SKILL.md");
        if existing.exists() {
            remove_legacy_hyprfast_skill();
            return Ok(existing);
        }
    }
    let primary = std::env::var("LUCY_SKILLS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
                .join(".config/lucy/skills")
        });
    let dest = primary.join("lucy").join("SKILL.md");
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if !dest.exists() {
        std::fs::write(&dest, LUCY_SKILL)?;
    }
    remove_legacy_hyprfast_skill();
    Ok(dest)
}

/// Best-effort cleanup of the pre-rename `hyprfast` skill dir. Never fails
/// seeding: I/O errors are ignored on purpose.
fn remove_legacy_hyprfast_skill() {
    for dir in skill_dirs() {
        let legacy = dir.join("hyprfast").join("SKILL.md");
        if legacy.exists() {
            let _ = std::fs::remove_file(&legacy);
            // Remove the parent dir when it is now empty.
            if let Some(parent) = legacy.parent() {
                let _ = std::fs::remove_dir(parent);
            }
        }
    }
}

fn front_matter_field(text: &str, field: &str) -> Option<String> {
    let mut lines = text.lines();
    if lines.next()?.trim() != "---" {
        return None;
    }
    let prefix = format!("{field}:");
    for line in lines {
        let t = line.trim();
        if t == "---" {
            break;
        }
        if let Some(rest) = t.strip_prefix(&prefix) {
            return Some(rest.trim().to_owned());
        }
    }
    None
}

fn truncate_one_line(s: &str, max: usize) -> String {
    let one: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if one.chars().count() <= max {
        one
    } else {
        let mut t: String = one.chars().take(max).collect();
        t.push('…');
        t
    }
}

/// Compose the full `{tool_catalog_brief}` value: local tools, one section
/// per MCP server group, then skills. `mcp_sections` holds
/// `(section_title, [(tool_name, description)])` pairs.
pub fn build_tool_brief(
    local_lines: Vec<String>,
    mcp_sections: &[(String, Vec<(String, String)>)],
    skill_lines: Vec<String>,
) -> String {
    let mut out = String::new();
    out.push_str("## Local tools\n");
    for line in local_lines {
        out.push_str(&line);
        out.push('\n');
    }
    for (title, tools) in mcp_sections {
        out.push_str(&format!("## {title}\n"));
        for (name, desc) in tools {
            out.push_str(&format!("{name} — {}\n", truncate_one_line(desc, 160)));
        }
    }
    if !skill_lines.is_empty() {
        out.push_str("## Skills (consult the skill file for recipes before using its tools)\n");
        for line in skill_lines {
            out.push_str(&line);
            out.push('\n');
        }
    }
    out
}

/// The tool name a `build_tool_brief` line advertises (`name — description`,
/// or the bare line when the description is empty).
pub fn brief_line_tool(line: &str) -> &str {
    line.split(" — ").next().unwrap_or(line).trim()
}

/// [`build_tool_brief`] output with the lines `keep` rejects removed, used to
/// enforce a planner-facing tool set by hiding rather than by asking. Section
/// headings left with nothing under them are dropped so the planner never
/// sees an empty section.
pub fn filter_tool_brief(brief: &str, keep: impl Fn(&str) -> bool) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut head: Option<String> = None;
    let mut body: Vec<String> = Vec::new();
    for line in brief.lines() {
        if line.starts_with("## ") {
            flush_brief_section(&mut out, head.as_deref(), &body);
            head = Some(line.to_owned());
            body.clear();
            continue;
        }
        if keep(brief_line_tool(line)) {
            body.push(line.to_owned());
        }
    }
    flush_brief_section(&mut out, head.as_deref(), &body);
    out.join("\n")
}

fn flush_brief_section(out: &mut Vec<String>, head: Option<&str>, body: &[String]) {
    if body.is_empty() {
        return;
    }
    if let Some(h) = head {
        out.push(h.to_owned());
    }
    out.extend(body.iter().cloned());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn parses_a_numbered_subtask_list() {
        let v = serde_json::json!({
            "subtasks": [
                {"description": "Open the browser to youtube.com", "success_condition": "youtube.com is visible"},
                {"description": "Search YouTube for Despacito", "success_condition": "results are visible"},
                {"description": "Play the first result", "success_condition": "video is playing"},
                {"description": "Verify playback", "success_condition": "playback is progressing"}
            ]
        });
        let subs = parse_subtasks(&v);
        assert_eq!(subs.len(), 4);
        assert_eq!(subs[0].index, 1);
        assert_eq!(subs[0].description, "Open the browser to youtube.com");
        assert_eq!(subs[3].index, 4);
        assert_eq!(subs[3].success_condition, "playback is progressing");
    }

    #[test]
    fn subtasks_reject_empty_and_cap_at_eight() {
        let mut arr = vec![
            serde_json::json!({"description": "  ", "success_condition": "x"}),
            serde_json::json!({"description": "Real step"}),
        ];
        for i in 0..10 {
            arr.push(serde_json::json!({"description": format!("Step {i}")}));
        }
        let v = serde_json::json!({"subtasks": arr});
        let subs = parse_subtasks(&v);
        assert_eq!(subs.len(), 8);
        assert_eq!(subs[0].description, "Real step");
        // Missing success_condition gets the default, indices renumbered.
        assert!(!subs[0].success_condition.is_empty());
        assert_eq!(subs[7].index, 8);
        assert!(parse_subtasks(&serde_json::json!({})).is_empty());
        assert!(parse_subtasks(&serde_json::json!({"subtasks": []})).is_empty());
    }

    #[test]
    fn subtasks_prompt_renders_request() {
        let out = render_subtasks_prompt("play Despacito song on YouTube");
        assert!(out.contains("play Despacito song on YouTube"));
        assert!(!out.contains("{user_request}"));
        assert!(out.contains("Despacito"));
    }

    #[test]
    fn empty_history_renders_none() {
        assert_eq!(format_history(&[], 8), "(none)");
    }

    #[test]
    fn brief_composes_sections_in_order() {
        let out = build_tool_brief(
            vec!["shell — run a command".into()],
            &[(
                "HyprFast MCP tools".into(),
                vec![("mcp_hyprfast_desktop".into(), "snapshot".into())],
            )],
            vec!["hyprfast — skill".into()],
        );
        assert!(out.contains("## Local tools\nshell — run a command"));
        assert!(out.contains("## HyprFast MCP tools\nmcp_hyprfast_desktop — snapshot"));
        assert!(out.contains("## Skills"));
        assert!(out.contains("hyprfast — skill"));
    }

    #[test]
    fn brief_without_skills_omits_section() {
        let out = build_tool_brief(vec![], &[], vec![]);
        assert!(out.contains("## Local tools"));
        assert!(!out.contains("## Skills"));
    }

    fn write_skill(dir: &Path, name: &str, front_matter: &str, body: &str) -> PathBuf {
        let skill_dir = dir.join(name);
        std::fs::create_dir_all(&skill_dir).unwrap();
        let path = skill_dir.join("SKILL.md");
        std::fs::write(&path, format!("{front_matter}\n{body}")).unwrap();
        path
    }

    fn temp_skills_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("lucy-skills-test-{}-{}", std::process::id(), tag));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn discovers_skills_from_disk() {
        let dir = temp_skills_dir("basic");
        write_skill(
            &dir,
            "hyprfast",
            "---\nname: hyprfast\ndescription: Fast desktop automation.\n---",
            "# hyprfast body",
        );
        write_skill(
            &dir,
            "browser",
            "---\nname: browser\ndescription: Browser recipes.\n---",
            "# browser body",
        );
        let skills = discover_skills(&[dir.clone()]);
        assert_eq!(skills.len(), 2);
        assert_eq!(skills[0].name, "browser");
        assert_eq!(skills[1].name, "hyprfast");
        assert_eq!(skills[1].description, "Fast desktop automation.");
        assert_eq!(
            skills[1].body().unwrap(),
            "---\nname: hyprfast\ndescription: Fast desktop automation.\n---\n# hyprfast body"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn skill_name_falls_back_to_dir_name() {
        let dir = temp_skills_dir("fallback");
        write_skill(&dir, "plain", "no front matter here", "# body");
        let skills = discover_skills(&[dir.clone()]);
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "plain");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn first_dir_wins_on_name_collision() {
        let first = temp_skills_dir("collision-a");
        let second = temp_skills_dir("collision-b");
        write_skill(
            &first,
            "dup",
            "---\nname: dup\ndescription: First.\n---",
            "",
        );
        write_skill(
            &second,
            "dup",
            "---\nname: dup\ndescription: Second.\n---",
            "",
        );
        let skills = discover_skills(&[first.clone(), second.clone()]);
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].description, "First.");
        let _ = std::fs::remove_dir_all(&first);
        let _ = std::fs::remove_dir_all(&second);
    }

    #[test]
    fn missing_dirs_are_ignored() {
        let skills = discover_skills(&[PathBuf::from("/definitely-not-a-real-skills-dir-xyz")]);
        assert!(skills.is_empty());
    }

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn seed_writes_lucy_when_absent() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = temp_skills_dir("seed");
        unsafe { std::env::set_var("LUCY_SKILLS_DIR", dir.to_str().unwrap()) };
        // Point HOME away so the real config dir is not scanned/seeded.
        let tmp_home = temp_skills_dir("seed-home");
        unsafe { std::env::set_var("HOME", tmp_home.to_str().unwrap()) };
        let dest = seed_lucy_skill().unwrap();
        assert_eq!(dest, dir.join("lucy").join("SKILL.md"));
        let skills = discover_skills(&skill_dirs());
        assert!(skills.iter().any(|s| s.name == "lucy"));
        assert!(
            !skills.iter().any(|s| s.name == "hyprfast"),
            "legacy hyprfast skill must be removed"
        );
        unsafe { std::env::remove_var("LUCY_SKILLS_DIR") };
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&tmp_home);
    }

    #[test]
    fn seed_removes_legacy_hyprfast_skill() {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = temp_skills_dir("seed-legacy");
        unsafe { std::env::set_var("LUCY_SKILLS_DIR", dir.to_str().unwrap()) };
        let tmp_home = temp_skills_dir("seed-legacy-home");
        unsafe { std::env::set_var("HOME", tmp_home.to_str().unwrap()) };
        // Simulate a pre-rename install: only hyprfast exists on disk.
        write_skill(
            &dir,
            "hyprfast",
            "---\nname: hyprfast\ndescription: Legacy.\n---",
            "# legacy",
        );
        let dest = seed_lucy_skill().unwrap();
        assert_eq!(dest, dir.join("lucy").join("SKILL.md"));
        assert!(!dir.join("hyprfast").join("SKILL.md").exists());
        unsafe { std::env::remove_var("LUCY_SKILLS_DIR") };
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&tmp_home);
    }
}
