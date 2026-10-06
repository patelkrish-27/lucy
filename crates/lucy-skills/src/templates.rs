//! Skill template definitions.
//!
//! Templates provide a consistent skeleton for new skills. They are
//! intentionally generic: the model decides what a skill is for, templates
//! only shape how it is written down.

use serde::{Deserialize, Serialize};

use crate::TaskPattern;

/// A template for generating new skills from patterns.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkillTemplate {
    pub name: String,
    pub description: String,
    pub body_template: String,
}

impl SkillTemplate {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        body_template: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            body_template: body_template.into(),
        }
    }

    /// Render the template body for a pattern.
    ///
    /// Placeholders: `{description}` and `{steps}` (a numbered list).
    pub fn render(&self, pattern: &TaskPattern) -> String {
        let mut steps = String::new();
        for (i, step) in pattern.steps.iter().enumerate() {
            steps.push_str(&format!("{}. {}\n", i + 1, step));
        }
        self.body_template
            .replace("{description}", &pattern.description)
            .replace("{steps}", steps.trim_end())
    }
}

/// The standard template every auto-created skill starts from.
pub fn standard() -> SkillTemplate {
    SkillTemplate::new(
        "standard",
        "Default skeleton for an auto-created skill",
        "# {description}\n\n## Steps\n\n{steps}\n\n## Notes\n\n- Auto-created from repeated task patterns.\n",
    )
}

/// A checklist-oriented variant for multi-step tasks.
pub fn checklist() -> SkillTemplate {
    SkillTemplate::new(
        "checklist",
        "Checklist skeleton for multi-step tasks",
        "# {description}\n\n## Checklist\n\n{steps}\n\n## Verification\n\nConfirm each step completed before moving on.\n",
    )
}

/// All built-in templates.
pub fn default_templates() -> Vec<SkillTemplate> {
    vec![standard(), checklist()]
}

/// Look up a template by name.
pub fn find_template(name: &str) -> Option<SkillTemplate> {
    default_templates().into_iter().find(|t| t.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_pattern() -> TaskPattern {
        TaskPattern {
            description: "An example task".to_string(),
            steps: vec!["first".to_string(), "second".to_string()],
            success_count: 3,
            failure_count: 0,
        }
    }

    #[test]
    fn render_substitutes_description_and_steps() {
        let rendered = standard().render(&sample_pattern());
        assert!(rendered.contains("# An example task"));
        assert!(rendered.contains("1. first"));
        assert!(rendered.contains("2. second"));
        assert!(!rendered.contains("{description}"));
        assert!(!rendered.contains("{steps}"));
    }

    #[test]
    fn default_templates_have_unique_names() {
        let templates = default_templates();
        let mut names: Vec<&str> = templates.iter().map(|t| t.name.as_str()).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), templates.len());
    }

    #[test]
    fn find_template_returns_known_template() {
        assert_eq!(find_template("standard").unwrap().name, "standard");
        assert!(find_template("does-not-exist").is_none());
    }
}
