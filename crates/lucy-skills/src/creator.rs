//! Pattern detection and skill creation.
//!
//! Detects repeated task patterns and creates new skills when a pattern
//! appears 3+ times with a clear success/failure signal.

use anyhow::Result;
use tracing::debug;

use crate::{SkillInfo, SkillStore, TaskPattern};

/// Minimum occurrences before a pattern qualifies for skill creation.
pub const MIN_PATTERN_OCCURRENCES: u32 = 3;

/// Detects patterns from a sequence of task descriptions and their outcomes.
///
/// This is a similarity-based clustering: tasks with similar descriptions
/// are grouped together, and clusters with 3+ occurrences become patterns.
pub struct PatternDetector {
    /// Similarity threshold (0.0–1.0). Higher = stricter matching.
    threshold: f64,
}

impl PatternDetector {
    /// Create a new detector with the given similarity threshold.
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    /// Detect patterns from a list of (description, success) pairs.
    ///
    /// Returns clusters of similar tasks that meet the minimum occurrence count.
    pub fn detect_patterns(&self, tasks: &[(String, bool)]) -> Vec<TaskPattern> {
        let mut clusters: Vec<Vec<(String, bool)>> = Vec::new();

        for (desc, success) in tasks {
            let mut found_cluster = false;

            for cluster in &mut clusters {
                let rep_desc = &cluster[0].0;
                if self.similarity(desc, rep_desc) >= self.threshold {
                    cluster.push((desc.clone(), *success));
                    found_cluster = true;
                    break;
                }
            }

            if !found_cluster {
                clusters.push(vec![(desc.clone(), *success)]);
            }
        }

        // Filter to clusters with enough occurrences and convert to patterns
        clusters
            .into_iter()
            .filter(|c| c.len() as u32 >= MIN_PATTERN_OCCURRENCES)
            .map(|cluster| {
                let success_count = cluster.iter().filter(|(_, s)| *s).count() as u32;
                let failure_count = cluster.iter().filter(|(_, s)| !*s).count() as u32;

                // Use the most common description as the representative
                let description = cluster[0].0.clone();

                // Generate steps from the pattern
                let steps = Self::infer_steps(&description);

                TaskPattern {
                    description,
                    steps,
                    success_count,
                    failure_count,
                }
            })
            .collect()
    }

    /// Compute similarity between two strings using Jaccard similarity
    /// on word sets.
    fn similarity(&self, a: &str, b: &str) -> f64 {
        let a_lower = a.to_lowercase();
        let set_a: std::collections::HashSet<_> = a_lower
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .collect();

        let b_lower = b.to_lowercase();
        let set_b: std::collections::HashSet<_> = b_lower
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .collect();

        if set_a.is_empty() && set_b.is_empty() {
            return 1.0;
        }
        if set_a.is_empty() || set_b.is_empty() {
            return 0.0;
        }

        let intersection: std::collections::HashSet<_> =
            set_a.intersection(&set_b).collect();
        let union: std::collections::HashSet<_> = set_a.union(&set_b).collect();

        intersection.len() as f64 / union.len() as f64
    }

    /// Infer steps from a task description.
    ///
    /// This is intentionally generic — it splits on common action verbs
    /// and produces a numbered step list. The model is the router; this
    /// just provides a starting skeleton.
    fn infer_steps(description: &str) -> Vec<String> {
        let mut steps = Vec::new();

        // Split on common step indicators
        let parts: Vec<&str> = description
            .split(|c| c == ',' || c == ';' || c == '.' || c == '\n')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        if parts.len() > 1 {
            for part in parts {
                steps.push(part.to_string());
            }
        } else {
            // Single action — break it into generic phases
            steps.push(format!("Prepare: {}", description));
            steps.push(format!("Execute: {}", description));
            steps.push("Verify the outcome".to_string());
        }

        steps
    }
}

/// Attempt to create skills for all detected patterns.
///
/// Returns the skills that were actually created.
pub async fn create_skills_from_patterns(
    store: &SkillStore,
    patterns: &[TaskPattern],
) -> Result<Vec<SkillInfo>> {
    let mut created = Vec::new();

    for pattern in patterns {
        if let Some(skill) = store.maybe_create_skill(pattern).await? {
            debug!(
                skill = %skill.name,
                "Created skill from pattern"
            );
            created.push(skill);
        }
    }

    Ok(created)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similarity_identical_strings_is_one() {
        let detector = PatternDetector::new(0.5);
        assert_eq!(detector.similarity("hello world", "hello world"), 1.0);
    }

    #[test]
    fn similarity_completely_different_is_zero() {
        let detector = PatternDetector::new(0.5);
        assert_eq!(detector.similarity("abc def", "xyz uvw"), 0.0);
    }

    #[test]
    fn similarity_partial_overlap() {
        let detector = PatternDetector::new(0.5);
        let sim = detector.similarity("play song music", "play video music");
        assert!(sim > 0.0 && sim < 1.0);
    }

    #[test]
    fn detect_patterns_groups_similar_tasks() {
        let detector = PatternDetector::new(0.3);
        let tasks = vec![
            ("play a song on youtube".to_string(), true),
            ("play music on youtube".to_string(), true),
            ("play video on youtube".to_string(), false),
            ("send email to boss".to_string(), true),
        ];

        let patterns = detector.detect_patterns(&tasks);
        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0].success_count, 2);
        assert_eq!(patterns[0].failure_count, 1);
    }

    #[test]
    fn detect_patterns_requires_minimum_occurrences() {
        let detector = PatternDetector::new(0.3);
        let tasks = vec![
            ("unique task one".to_string(), true),
            ("unique task two".to_string(), false),
        ];

        let patterns = detector.detect_patterns(&tasks);
        assert!(patterns.is_empty());
    }

    #[test]
    fn infer_steps_splits_on_delimiters() {
        let steps = PatternDetector::infer_steps("open app, click button, verify result");
        assert_eq!(steps.len(), 3);
        assert!(steps[0].contains("open app"));
        assert!(steps[1].contains("click button"));
        assert!(steps[2].contains("verify result"));
    }

    #[test]
    fn infer_steps_provides_generic_phrases_for_single_action() {
        let steps = PatternDetector::infer_steps("do something complex");
        assert_eq!(steps.len(), 3);
        assert!(steps[0].starts_with("Prepare:"));
        assert!(steps[1].starts_with("Execute:"));
        assert!(steps[2].contains("Verify"));
    }
}
