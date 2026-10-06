//! Self-improving skills system for Lucy.
//!
//! Records skill usage outcomes, auto-creates new skills when task patterns
//! repeat, and improves existing skills based on success/failure rates.
//!
//! Skills are SKILL.md files with YAML front-matter, discovered from disk.
//! The model decides which skill to use — this crate only tracks outcomes
//! and manages the skill files.

pub mod creator;
pub mod improver;
pub mod templates;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::sync::RwLock;
use tracing::{debug, info};

/// A reusable skill discovered from disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillInfo {
    pub name: String,
    pub description: String,
    pub body: String,
    pub success_rate: f64,
    pub usage_count: u32,
}

/// A detected task pattern that may become a skill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskPattern {
    pub description: String,
    pub steps: Vec<String>,
    pub success_count: u32,
    pub failure_count: u32,
}

/// A proposed change to an existing skill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillImprovement {
    pub skill_name: String,
    pub change: SkillChange,
    pub reason: String,
}

/// The kind of change to apply to a skill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SkillChange {
    UpdatedDescription(String),
    AddedStep(String),
    RemovedStep(String),
    ReorderedSteps(Vec<String>),
}

pub use templates::SkillTemplate;

/// Internal record for tracking outcomes per skill.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct OutcomeRecord {
    success_count: u32,
    failure_count: u32,
    contexts: Vec<String>,
}

/// Internal record for tracking task patterns.
#[derive(Debug, Clone, Deserialize, Serialize)]
struct PatternRecord {
    description: String,
    steps: Vec<String>,
    success_count: u32,
    failure_count: u32,
    occurrences: u32,
}

/// The main skill store — manages skill files, outcomes, and patterns.
pub struct SkillStore {
    db_path: PathBuf,
    skills_dir: PathBuf,
    outcomes: RwLock<HashMap<String, OutcomeRecord>>,
    patterns: RwLock<HashMap<String, PatternRecord>>,
}

impl SkillStore {
    /// Open (or create) a SkillStore rooted at the given directory.
    ///
    /// Skills live in `<path>/skills` and outcome/pattern tracking data in
    /// `<path>/lucy-skills.db` (plus sibling JSON files).
    pub async fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let root = path.into();
        let skills_dir = root.join("skills");
        let db_path = root.join("lucy-skills.db");

        // Ensure the skills directory exists
        tokio::fs::create_dir_all(&skills_dir)
            .await
            .with_context(|| format!("Failed to create skills directory: {}", skills_dir.display()))?;

        if let Some(parent) = db_path.parent() {
            tokio::fs::create_dir_all(parent).await.ok();
        }

        let store = Self {
            db_path,
            skills_dir,
            outcomes: RwLock::new(HashMap::new()),
            patterns: RwLock::new(HashMap::new()),
        };

        // Load existing data from disk if present
        store.load().await?;

        Ok(store)
    }

    /// Record a success or failure outcome for a skill.
    pub async fn record_outcome(&self, skill: &str, success: bool, context: &str) -> Result<()> {
        let mut outcomes = self.outcomes.write().await;
        let record = outcomes.entry(skill.to_string()).or_default();

        if success {
            record.success_count += 1;
        } else {
            record.failure_count += 1;
        }

        // Keep last 100 contexts for pattern analysis
        if record.contexts.len() >= 100 {
            record.contexts.remove(0);
        }
        record.contexts.push(context.to_string());

        debug!(
            skill = %skill,
            success,
            total_success = record.success_count,
            total_failure = record.failure_count,
            "Recorded outcome"
        );

        self.save_outcomes(&outcomes).await?;
        Ok(())
    }

    /// Check if a task pattern qualifies for skill creation.
    ///
    /// Returns `Some(SkillInfo)` if the pattern has 3+ occurrences and
    /// can be expressed as a reusable skill.
    pub async fn maybe_create_skill(&self, pattern: &TaskPattern) -> Result<Option<SkillInfo>> {
        let total = pattern.success_count + pattern.failure_count;

        // Need at least 3 occurrences to consider creating a skill
        if total < 3 {
            debug!(
                pattern = %pattern.description,
                occurrences = total,
                "Pattern does not have enough occurrences for skill creation"
            );
            return Ok(None);
        }

        // Check if a skill with a similar name already exists
        let existing = self.find_similar_skill(&pattern.description).await;
        if existing.is_some() {
            debug!(
                pattern = %pattern.description,
                "Similar skill already exists, skipping creation"
            );
            return Ok(None);
        }

        // Generate a skill name from the pattern description
        let skill_name = Self::generate_skill_name(&pattern.description);

        // Check if skill file already exists on disk
        let skill_path = self.skills_dir.join(format!("{}.md", skill_name));
        if skill_path.exists() {
            debug!(
                skill = %skill_name,
                "Skill file already exists on disk"
            );
            return Ok(None);
        }

        // Create the skill
        let skill_info = self.create_skill_from_pattern(&skill_name, pattern).await?;

        info!(
            skill = %skill_name,
            occurrences = total,
            "Created new skill from pattern"
        );

        Ok(Some(skill_info))
    }

    /// Analyze all skills and propose improvements based on success rates.
    ///
    /// Returns a list of proposed improvements.
    pub async fn improve_skills(&self) -> Result<Vec<SkillImprovement>> {
        let mut improvements = Vec::new();
        let skills = self.list_skills().await;

        for skill in &skills {
            let total = skill.usage_count;
            if total == 0 {
                continue;
            }

            let rate = skill.success_rate;

            // Low success rate: flag for improvement
            if total >= 5 && rate < 0.5 {
                let improvement = self
                    .generate_improvement(skill, rate, total, "low_success")
                    .await?;
                if let Some(imp) = improvement {
                    improvements.push(imp);
                }
            }

            // High success rate: mark as stable (no change needed, but log it)
            if total >= 10 && rate > 0.9 {
                debug!(
                    skill = %skill.name,
                    rate,
                    "Skill is stable with high success rate"
                );
            }
        }

        Ok(improvements)
    }

    /// List all skills discovered from disk.
    pub async fn list_skills(&self) -> Vec<SkillInfo> {
        let mut skills = Vec::new();

        let mut entries = match tokio::fs::read_dir(&self.skills_dir).await {
            Ok(entries) => entries,
            Err(_) => return skills,
        };

        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("md") {
                continue;
            }

            if let Ok(content) = tokio::fs::read_to_string(&path).await {
                if let Some(skill) = Self::parse_skill_file(&content, &path) {
                    // Merge with outcome data
                    let outcomes = self.outcomes.read().await;
                    let mut skill = skill;
                    if let Some(record) = outcomes.get(&skill.name) {
                        let total = record.success_count + record.failure_count;
                        skill.usage_count = total;
                        skill.success_rate = if total > 0 {
                            record.success_count as f64 / total as f64
                        } else {
                            0.0
                        };
                    }
                    skills.push(skill);
                }
            }
        }

        skills
    }

    /// Get a specific skill by name.
    pub async fn get_skill(&self, name: &str) -> Option<SkillInfo> {
        let skills = self.list_skills().await;
        skills.into_iter().find(|s| s.name == name)
    }

    /// Get the skills directory path.
    pub fn skills_dir(&self) -> &Path {
        &self.skills_dir
    }

    /// Get the database path.
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    // --- Private helpers ---

    async fn load(&self) -> Result<()> {
        // Load outcomes from a JSON file alongside the db
        let outcomes_path = self.db_path.with_extension("outcomes.json");
        if outcomes_path.exists() {
            let data = tokio::fs::read_to_string(&outcomes_path).await?;
            let outcomes: HashMap<String, OutcomeRecord> = serde_json::from_str(&data)?;
            *self.outcomes.write().await = outcomes;
        }

        // Load patterns
        let patterns_path = self.db_path.with_extension("patterns.json");
        if patterns_path.exists() {
            let data = tokio::fs::read_to_string(&patterns_path).await?;
            let patterns: HashMap<String, PatternRecord> = serde_json::from_str(&data)?;
            *self.patterns.write().await = patterns;
        }

        Ok(())
    }

    async fn save_outcomes(&self, outcomes: &HashMap<String, OutcomeRecord>) -> Result<()> {
        let outcomes_path = self.db_path.with_extension("outcomes.json");
        let json = serde_json::to_string_pretty(outcomes)?;
        tokio::fs::write(&outcomes_path, json).await?;
        Ok(())
    }

    async fn save_patterns(&self, patterns: &HashMap<String, PatternRecord>) -> Result<()> {
        let patterns_path = self.db_path.with_extension("patterns.json");
        let json = serde_json::to_string_pretty(patterns)?;
        tokio::fs::write(&patterns_path, json).await?;
        Ok(())
    }

    async fn find_similar_skill(&self, description: &str) -> Option<SkillInfo> {
        let skills = self.list_skills().await;
        let desc_lower = description.to_lowercase();

        // Simple similarity: check if any skill name or description shares
        // significant words with the pattern description
        let desc_words: std::collections::HashSet<_> = desc_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();

        for skill in skills {
            let skill_name_lower = skill.name.to_lowercase();
            let skill_desc_lower = skill.description.to_lowercase();
            let skill_words: std::collections::HashSet<_> = skill_name_lower
                .split_whitespace()
                .chain(skill_desc_lower.split_whitespace())
                .filter(|w| w.len() > 3)
                .collect();

            let intersection: std::collections::HashSet<_> =
                desc_words.intersection(&skill_words).collect();

            // If more than 50% of words match, consider it similar
            if !desc_words.is_empty() && intersection.len() * 2 >= desc_words.len() {
                return Some(skill);
            }
        }

        None
    }

    fn generate_skill_name(description: &str) -> String {
        // Convert description to a kebab-case skill name
        let mut name = String::new();
        let mut prev_was_sep = true;

        for c in description.chars() {
            if c.is_alphanumeric() {
                name.push(c.to_ascii_lowercase());
                prev_was_sep = false;
            } else if !prev_was_sep && !name.is_empty() {
                name.push('-');
                prev_was_sep = true;
            }
        }

        // Trim trailing separator
        while name.ends_with('-') {
            name.pop();
        }

        // Limit length
        if name.len() > 50 {
            name = name.chars().take(50).collect();
            while name.ends_with('-') {
                name.pop();
            }
        }

        if name.is_empty() {
            name = format!("skill-{}", uuid::Uuid::new_v4().to_string()[..8].to_string());
        }

        name
    }

    async fn create_skill_from_pattern(
        &self,
        skill_name: &str,
        pattern: &TaskPattern,
    ) -> Result<SkillInfo> {
        let body = self.generate_skill_body(pattern);
        let description = format!(
            "Auto-created skill for: {}. Based on {} successful executions out of {} total.",
            pattern.description,
            pattern.success_count,
            pattern.success_count + pattern.failure_count
        );

        let skill_info = SkillInfo {
            name: skill_name.to_string(),
            description: description.clone(),
            body: body.clone(),
            success_rate: if pattern.success_count + pattern.failure_count > 0 {
                pattern.success_count as f64 / (pattern.success_count + pattern.failure_count) as f64
            } else {
                0.0
            },
            usage_count: pattern.success_count + pattern.failure_count,
        };

        // Write the SKILL.md file
        let skill_path = self.skills_dir.join(format!("{}.md", skill_name));
        let content = Self::format_skill_file(&skill_info);
        tokio::fs::write(&skill_path, content)
            .await
            .with_context(|| format!("Failed to write skill file: {}", skill_path.display()))?;

        // Record the pattern
        let mut patterns = self.patterns.write().await;
        patterns.insert(
            skill_name.to_string(),
            PatternRecord {
                description: pattern.description.clone(),
                steps: pattern.steps.clone(),
                success_count: pattern.success_count,
                failure_count: pattern.failure_count,
                occurrences: pattern.success_count + pattern.failure_count,
            },
        );
        self.save_patterns(&patterns).await?;

        Ok(skill_info)
    }

    fn generate_skill_body(&self, pattern: &TaskPattern) -> String {
        let mut body = String::new();

        body.push_str(&format!("# {}\n\n", pattern.description));
        body.push_str("## Steps\n\n");

        for (i, step) in pattern.steps.iter().enumerate() {
            body.push_str(&format!("{}. {}\n", i + 1, step));
        }

        body.push_str("\n## Notes\n\n");
        body.push_str(&format!(
            "- This skill was auto-created from {} observed executions.\n",
            pattern.success_count + pattern.failure_count
        ));
        body.push_str(&format!(
            "- Success rate: {:.1}%\n",
            if pattern.success_count + pattern.failure_count > 0 {
                pattern.success_count as f64 / (pattern.success_count + pattern.failure_count) as f64 * 100.0
            } else {
                0.0
            }
        ));

        body
    }

    fn format_skill_file(skill: &SkillInfo) -> String {
        format!(
            "---\nname: {}\ndescription: {}\n---\n\n{}\n",
            skill.name, skill.description, skill.body
        )
    }

    fn parse_skill_file(content: &str, path: &Path) -> Option<SkillInfo> {
        let name = path.file_stem()?.to_str()?.to_string();

        // Parse YAML front-matter
        let (front_matter, body) = Self::split_front_matter(content)?;

        let description = Self::extract_yaml_field(&front_matter, "description")
            .unwrap_or_default();

        Some(SkillInfo {
            name,
            description,
            body: body.trim().to_string(),
            success_rate: 0.0,
            usage_count: 0,
        })
    }

    fn split_front_matter(content: &str) -> Option<(String, String)> {
        let mut lines = content.lines();

        // First line must be ---
        if lines.next()?.trim() != "---" {
            return None;
        }

        let mut front_matter = String::new();
        let mut found_end = false;

        for line in lines.by_ref() {
            if line.trim() == "---" {
                found_end = true;
                break;
            }
            front_matter.push_str(line);
            front_matter.push('\n');
        }

        if !found_end {
            return None;
        }

        let body: Vec<&str> = lines.collect();
        Some((front_matter, body.join("\n")))
    }

    fn extract_yaml_field(front_matter: &str, field: &str) -> Option<String> {
        for line in front_matter.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix(&format!("{}:", field)) {
                return Some(rest.trim().to_string());
            }
        }
        None
    }

    async fn generate_improvement(
        &self,
        skill: &SkillInfo,
        rate: f64,
        total: u32,
        reason: &str,
    ) -> Result<Option<SkillImprovement>> {
        match reason {
            "low_success" => {
                // Propose adding a troubleshooting step
                let new_step = format!(
                    "If the above steps fail, verify prerequisites and try an alternative approach. (Success rate: {:.1}% over {} uses)",
                    rate * 100.0,
                    total
                );

                Ok(Some(SkillImprovement {
                    skill_name: skill.name.clone(),
                    change: SkillChange::AddedStep(new_step),
                    reason: format!(
                        "Low success rate ({:.1}%) over {} uses — adding troubleshooting guidance",
                        rate * 100.0,
                        total
                    ),
                }))
            }
            _ => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_skill_name_converts_to_kebab_case() {
        assert_eq!(
            SkillStore::generate_skill_name("Play a song on YouTube"),
            "play-a-song-on-youtube"
        );
    }

    #[test]
    fn generate_skill_name_handles_empty_description() {
        let name = SkillStore::generate_skill_name("");
        assert!(!name.is_empty());
        assert!(name.starts_with("skill-"));
    }

    #[test]
    fn generate_skill_name_truncates_long_names() {
        let long_desc = "a".repeat(100);
        let name = SkillStore::generate_skill_name(&long_desc);
        assert!(name.len() <= 50);
    }

    #[test]
    fn parse_skill_file_extracts_name_and_description() {
        let content = r#"---
name: test-skill
description: A test skill
---

# Test Body

Some content here.
"#;
        let path = Path::new("/tmp/test-skill.md");
        let skill = SkillStore::parse_skill_file(content, path).unwrap();

        assert_eq!(skill.name, "test-skill");
        assert_eq!(skill.description, "A test skill");
        assert!(skill.body.contains("Test Body"));
    }

    #[test]
    fn parse_skill_file_returns_none_without_front_matter() {
        let content = "# Just a heading\n\nSome content";
        let path = Path::new("/tmp/no-front-matter.md");
        assert!(SkillStore::parse_skill_file(content, path).is_none());
    }

    #[test]
    fn format_skill_file_produces_valid_yaml_front_matter() {
        let skill = SkillInfo {
            name: "my-skill".to_string(),
            description: "Does things".to_string(),
            body: "# Body\n\nSteps here.".to_string(),
            success_rate: 0.95,
            usage_count: 20,
        };

        let content = SkillStore::format_skill_file(&skill);
        assert!(content.starts_with("---\n"));
        assert!(content.contains("name: my-skill"));
        assert!(content.contains("description: Does things"));
        assert!(content.contains("# Body"));
    }
}
