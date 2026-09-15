use std::collections::{HashMap, HashSet, VecDeque};

use super::planner::SubTask;

/// A dependency-safe execution wave. Tasks in the same wave have no dependency
/// relationship with each other and are safe to consider for concurrent
/// HyprFast execution, subject to their resource/conflict keys.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionWave {
    pub tasks: Vec<SubTask>,
}

/// Phase 4 scheduler foundation: turn a linear task queue into dependency-safe
/// waves without asking the model to reason about concurrency.
///
/// The scheduler is deliberately conservative. Tasks touching the same
/// computer domain are kept in separate waves unless the category is known to
/// be read-only. This prevents two actions from racing on the same UI state.
pub fn build_waves(tasks: Vec<SubTask>) -> Vec<ExecutionWave> {
    let mut pending: VecDeque<SubTask> = tasks.into_iter().collect();
    let mut completed = HashSet::<String>::new();
    let mut waves = Vec::new();

    while !pending.is_empty() {
        let mut wave = Vec::new();
        let mut wave_keys = HashSet::new();
        let mut deferred = VecDeque::new();

        while let Some(task) = pending.pop_front() {
            let deps_ready = task.depends_on.iter().all(|id| completed.contains(id));
            let key = conflict_key(&task);
            if deps_ready && !wave_keys.contains(&key) {
                wave_keys.insert(key);
                wave.push(task);
            } else {
                deferred.push_back(task);
            }
        }

        if wave.is_empty() {
            // Dependency cycle or malformed graph. Preserve the remaining order
            // instead of spinning forever; the caller can reject the graph.
            waves.push(ExecutionWave {
                tasks: deferred.into_iter().collect(),
            });
            break;
        }

        for task in &wave {
            completed.insert(task.id.clone());
        }
        pending = deferred;
        waves.push(ExecutionWave { tasks: wave });
    }

    waves
}

/// Returns true when a task can be safely considered for parallel execution.
/// Observations are read-only; mutating operations remain serialized by domain.
pub fn parallel_candidate(task: &SubTask) -> bool {
    matches!(
        task.category.as_str(),
        "browser"
            | "vision"
            | "desktop"
            | "excalidraw"
            | "clipboard"
            | "tasks"
            | "stagehand"
            | "hints"
    )
}

/// Detect a dependency cycle before execution. An empty graph is valid.
pub fn has_dependency_cycle(tasks: &[SubTask]) -> bool {
    let ids: HashSet<&str> = tasks.iter().map(|t| t.id.as_str()).collect();
    let mut indegree: HashMap<&str, usize> = tasks.iter().map(|t| (t.id.as_str(), 0)).collect();
    let mut edges: HashMap<&str, Vec<&str>> = HashMap::new();

    for task in tasks {
        for dep in &task.depends_on {
            if !ids.contains(dep.as_str()) {
                continue;
            }
            *indegree.entry(task.id.as_str()).or_default() += 1;
            edges
                .entry(dep.as_str())
                .or_default()
                .push(task.id.as_str());
        }
    }

    let mut ready: VecDeque<&str> = indegree
        .iter()
        .filter_map(|(id, degree)| (*degree == 0).then_some(*id))
        .collect();
    let mut visited = 0usize;

    while let Some(id) = ready.pop_front() {
        visited += 1;
        if let Some(children) = edges.get(id) {
            for child in children {
                let degree = indegree.get_mut(child).expect("child exists");
                *degree -= 1;
                if *degree == 0 {
                    ready.push_back(child);
                }
            }
        }
    }

    visited != ids.len()
}

fn conflict_key(task: &SubTask) -> String {
    // A category is the safest resource boundary available before the cheap
    // compiler resolves the exact HyprFast command. More granular resource
    // keys can be added later from capability metadata/state facts.
    match task.category.as_str() {
        "browser" | "stagehand" | "hints" => "browser".to_owned(),
        "excalidraw" => "excalidraw".to_owned(),
        "vision" => "vision".to_owned(),
        "clipboard" => "clipboard".to_owned(),
        "desktop" | "tasks" => "desktop".to_owned(),
        other if other.is_empty() => "unknown".to_owned(),
        other => other.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(id: &str, category: &str, deps: &[&str]) -> SubTask {
        SubTask {
            id: id.into(),
            goal: id.into(),
            category: category.into(),
            depends_on: deps.iter().map(|s| (*s).into()).collect(),
        }
    }

    #[test]
    fn independent_domains_share_a_wave() {
        let waves = build_waves(vec![
            task("browser-observe", "browser", &[]),
            task("desktop-observe", "desktop", &[]),
        ]);
        assert_eq!(waves.len(), 1);
        assert_eq!(waves[0].tasks.len(), 2);
    }

    #[test]
    fn same_domain_is_serialized() {
        let waves = build_waves(vec![
            task("browser-a", "browser", &[]),
            task("browser-b", "browser", &[]),
        ]);
        assert_eq!(waves.len(), 2);
    }

    #[test]
    fn dependencies_create_ordered_waves() {
        let waves = build_waves(vec![
            task("observe", "browser", &[]),
            task("click", "browser", &["observe"]),
            task("report", "desktop", &["click"]),
        ]);
        assert_eq!(
            waves.iter().map(|w| w.tasks.len()).collect::<Vec<_>>(),
            vec![1, 1, 1]
        );
        assert_eq!(waves[1].tasks[0].id, "click");
    }

    #[test]
    fn detects_cycles() {
        let tasks = vec![task("a", "browser", &["b"]), task("b", "browser", &["a"])];
        assert!(has_dependency_cycle(&tasks));
    }

    #[test]
    fn ignores_missing_external_dependencies_for_cycle_detection() {
        let tasks = vec![task("a", "browser", &["external"])];
        assert!(!has_dependency_cycle(&tasks));
    }
}
