//! Skill improvement based on outcome tracking.
//!
//! Analyzes success rates and proposes changes to skill definitions.
//! Skills with low success rates get troubleshooting steps added.
//! Skills with high success rates are marked as stable.

use anyhow::Result;
use tracing::{info, warn};

use crate::{SkillChange, SkillImprovement, SkillInfo, SkillStore};

/// Minimum uses before considering a skill for improvement.
pub const MIN_USES_FOR_IMPROVEMENT: u32 = 5;

/// Success rate below which a skill is flagged for improvement.
pub const LOW_SUCCESS_THRESHOLD: f64 = 0.5;

/// Success rate above which a skill is considered stable.
pub const HIGH_SUCCESS_THRESHOLD: f64 = 0.9;

/// Minimum uses before considering a skill stable.
pub const MIN_USES_FOR_STABILITY: u32 = 10;

/// Analyzes skills and generates improvement suggestions.
pub struct SkillImprover;

impl SkillImprover {
    /// Create a new SkillImprover.
    pub fn new() -> Self {
        Self
    }

    /// Analyze all skills and return proposed improvements.
    pub async fn analyze(&self, store: &SkillStore) -> Result<Vec<SkillImprovement>> {
        let skills = store.list_skills().await;
        let mut improvements = Vec::new();

        for skill in &skills {
            if let Some(improvement) = self.analyze_skill(skill).await {
                improvements.push(improvement);
            }
        }

        Ok(improvements)
    }

    /// Analyze a single skill and return an improvement if needed.
    async fn analyze_skill(&self, skill: &SkillInfo) -> Option<SkillImprovement> {
        let total = skill.usage_count;

        if total < MIN_USES_FOR_IMPROVEMENT {
            return None;
        }

        let rate = skill.success_rate;

        if rate < LOW_SUCCESS_THRESHOLD {
            self.generate_low_success_improvement(skill, rate, total)
        } else if rate > HIGH_SUCCESS_THRESHOLD && total >= MIN_USES_FOR_STABILITY {
            self.mark_stable(skill, rate, total)
        } else {
            None
        }
    }

    fn generate_low_success_improvement(
        &self,
        skill: &SkillInfo,
        rate: f64,
        total: u32,
    ) -> Option<SkillImprovement> {
        warn!(
            skill = %skill.name,
            rate,
            total,
            "Skill has low success rate"
        );

        let new_step = format!(
            "If the above steps fail, verify prerequisites and try an alternative approach. (Auto-added due to {:.1}% success rate over {} uses)",
            rate * 100.0,
            total
        );

        Some(SkillImprovement {
            skill_name: skill.name.clone(),
            change: SkillChange::AddedStep(new_step),
            reason: format!(
                "Low success rate ({:.1}%) over {} uses — adding troubleshooting guidance",
                rate * 100.0,
                total
            ),
        })
    }

    fn mark_stable(&self, skill: &SkillInfo, rate: f64, total: u32) -> Option<SkillImprovement> {
        info!(
            skill = %skill.name,
            rate,
            total,
            "Skill is stable"
        );

        // Stable skills don't need changes, but we return an improvement
        // that updates the description to note stability
        let new_description = format!(
            "{} [Stable: {:.1}% success over {} uses]",
            skill.description, rate * 100.0, total
        );

        Some(SkillImprovement {
            skill_name: skill.name.clone(),
            change: SkillChange::UpdatedDescription(new_description),
            reason: format!(
                "High success rate ({:.1}%) over {} uses — marking as stable",
                rate * 100.0,
                total
            ),
        })
    }

    /// Apply an improvement to a skill file on disk.
    pub async fn apply_improvement(
        &self,
        store: &SkillStore,
        improvement: &SkillImprovement,
    ) -> Result<()> {
        let skill = store
            .get_skill(&improvement.skill_name)
            .await
            .ok_or_else(|| anyhow::anyhow!("Skill not found: {}", improvement.skill_name))?;

        let skill_path = store.skills_dir().join(format!("{}.md", skill.name));
        let content = tokio::fs::read_to_string(&skill_path).await?;

        let new_content = match &improvement.change {
            SkillChange::UpdatedDescription(new_desc) => {
                Self::update_description_in_file(&content, new_desc)?
            }
            SkillChange::AddedStep(step) => Self::add_step_to_body(&content, step)?,
            SkillChange::RemovedStep(step) => Self::remove_step_from_body(&content, step)?,
            SkillChange::ReorderedSteps(steps) => Self::reorder_steps_in_body(&content, steps)?,
        };

        tokio::fs::write(&skill_path, new_content).await?;

        info!(
            skill = %improvement.skill_name,
            "Applied improvement"
        );

        Ok(())
    }

    fn update_description_in_file(content: &str, new_desc: &str) -> Result<String> {
        let mut lines: Vec<String> = content.lines().map(String::from).collect();
        let mut in_front_matter = false;
        let mut found_end = false;
        let mut fm_start = 0;
        let mut fm_end = 0;

        for (i, line) in lines.iter().enumerate() {
            if i == 0 && line.trim() == "---" {
                in_front_matter = true;
                fm_start = i;
                continue;
            }
            if in_front_matter && line.trim() == "---" {
                fm_end = i;
                found_end = true;
                break;
            }
        }

        if !found_end {
            return Err(anyhow::anyhow!("Invalid front-matter in skill file"));
        }

        // Find and replace the description line
        let mut desc_found = false;
        for i in (fm_start + 1)..fm_end {
            if lines[i].trim_start().starts_with("description:") {
                lines[i] = format!("description: {}", new_desc);
                desc_found = true;
                break;
            }
        }

        if !desc_found {
            // Add description after name
            let mut name_idx = None;
            for i in (fm_start + 1)..fm_end {
                if lines[i].trim_start().starts_with("name:") {
                    name_idx = Some(i);
                    break;
                }
            }
            if let Some(idx) = name_idx {
                lines.insert(idx + 1, format!("description: {}", new_desc));
            }
        }

        Ok(lines.join("\n") + "\n")
    }

    fn add_step_to_body(content: &str, step: &str) -> Result<String> {
        // Find the last numbered step and add after it
        let mut lines: Vec<String> = content.lines().map(String::from).collect();
        let mut last_step_line = None;
        let mut step_num = 0;

        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim_start();
            if let Some(rest) = trimmed.strip_prefix(|c: char| c.is_ascii_digit()) {
                if rest.starts_with(". ") || rest.starts_with(") ") {
                    if let Ok(n) = trimmed[..1].parse::<u32>() {
                        if n > step_num {
                            step_num = n;
                            last_step_line = Some(i);
                        }
                    }
                }
            }
        }

        let new_line = format!("{}. {}", step_num + 1, step);

        if let Some(idx) = last_step_line {
            lines.insert(idx + 1, new_line);
        } else {
            // No existing steps — add a Steps section
            lines.push(String::new());
            lines.push("## Additional Steps".to_string());
            lines.push(new_line);
        }

        Ok(lines.join("\n") + "\n")
    }

    fn remove_step_from_body(content: &str, step: &str) -> Result<String> {
        let mut lines: Vec<String> = content.lines().map(String::from).collect();
        let step_lower = step.to_lowercase();

        lines.retain(|line| {
            let trimmed = line.trim_start();
            // Keep lines that don't contain the step text
            !trimmed.to_lowercase().contains(&step_lower)
        });

        Ok(lines.join("\n") + "\n")
    }

    fn reorder_steps_in_body(content: &str, steps: &[String]) -> Result<String> {
        let mut lines: Vec<String> = content.lines().map(String::from).collect();

        // Find the steps section and replace it
        let mut in_steps = false;
        let mut steps_start = None;
        let mut steps_end = None;

        for (i, line) in lines.iter().enumerate() {
            if line.trim() == "## Steps" || line.trim() == "## Additional Steps" {
                in_steps = true;
                steps_start = Some(i + 1);
                continue;
            }
            if in_steps {
                if line.starts_with("## ") || (line.trim().is_empty() && steps_end.is_none() && steps_start.map_or(false, |s| i > s)) {
                    if line.starts_with("## ") {
                        steps_end = Some(i);
                        break;
                    }
                }
                // Check if we hit a non-step line after steps
                let trimmed = line.trim_start();
                let is_step = trimmed.chars().next().map_or(false, |c| c.is_ascii_digit())
                    && (trimmed.contains(". ") || trimmed.contains(") "));
                if !is_step && !trimmed.is_empty() {
                    steps_end = Some(i);
                    break;
                }
            }
        }

        if let (Some(start), Some(end)) = (steps_start, steps_end) {
            let mut new_section = Vec::new();
            for (i, step) in steps.iter().enumerate() {
                new_section.push(format!("{}. {}", i + 1, step));
            }
            lines.splice(start..end, new_section);
        }

        Ok(lines.join("\n") + "\n")
    }
}

impl Default for SkillImprover {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn low_success_rate_triggers_improvement() {
        let improver = SkillImprover::new();
        let skill = SkillInfo {
            name: "test".to_string(),
            description: "test".to_string(),
            body: "test".to_string(),
            success_rate: 0.3,
            usage_count: 10,
        };

        let result = improver.analyze_skill(&skill).await;
        assert!(result.is_some());
        let imp = result.unwrap();
        assert!(matches!(imp.change, SkillChange::AddedStep(_)));
    }

    #[tokio::test]
    async fn high_success_rate_marks_stable() {
        let improver = SkillImprover::new();
        let skill = SkillInfo {
            name: "test".to_string(),
            description: "test".to_string(),
            body: "test".to_string(),
            success_rate: 0.95,
            usage_count: 15,
        };

        let result = improver.analyze_skill(&skill).await;
        assert!(result.is_some());
        let imp = result.unwrap();
        assert!(matches!(imp.change, SkillChange::UpdatedDescription(_)));
    }

    #[tokio::test]
    async fn insufficient_usage_returns_none() {
        let improver = SkillImprover::new();
        let skill = SkillInfo {
            name: "test".to_string(),
            description: "test".to_string(),
            body: "test".to_string(),
            success_rate: 0.3,
            usage_count: 2,
        };

        let result = improver.analyze_skill(&skill).await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn moderate_success_rate_returns_none() {
        let improver = SkillImprover::new();
        let skill = SkillInfo {
            name: "test".to_string(),
            description: "test".to_string(),
            body: "test".to_string(),
            success_rate: 0.7,
            usage_count: 10,
        };

        let result = improver.analyze_skill(&skill).await;
        assert!(result.is_none());
    }

    #[test]
    fn update_description_replaces_existing() {
        let content = r#"---
name: test
description: old description
---

# Body
"#;
        let result = SkillImprover::update_description_in_file(content, "new description").unwrap();
        assert!(result.contains("description: new description"));
        assert!(!result.contains("old description"));
    }

    #[test]
    fn add_step_appends_numbered_step() {
        let content = r#"---
name: test
description: test
---

# Body

## Steps

1. First step
2. Second step
"#;
        let result = SkillImprover::add_step_to_body(content, "Third step").unwrap();
        assert!(result.contains("3. Third step"));
    }

    #[test]
    fn remove_step_deletes_matching_line() {
        let content = r#"---
name: test
description: test
---

# Body

## Steps

1. First step
2. Second step
3. Third step
"#;
        let result = SkillImprover::remove_step_from_body(content, "Second step").unwrap();
        assert!(!result.contains("Second step"));
        assert!(result.contains("First step"));
        assert!(result.contains("Third step"));
    }
}
