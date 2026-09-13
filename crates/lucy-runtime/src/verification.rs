use lucy_hyprfast::{Domain, HyprFastCatalog};

/// Returns true when a HyprFast command changed state and should normally be
/// followed by an observation before Lucy considers the operation complete.
pub fn action_requires_verification(
    catalog: &HyprFastCatalog,
    tool: &str,
    category: &str,
    explicit_verify: bool,
) -> bool {
    if explicit_verify {
        return true;
    }

    let Some(capability) = catalog.capability_for_mcp_name(tool) else {
        // Unknown tools are treated conservatively. The caller can still opt
        // out through the planner configuration.
        return !matches!(category, "vision" | "browser" | "excalidraw");
    };

    !capability.read_only
}

/// Pick an observation domain that is likely to expose the state affected by
/// the previous action. The cheap model still chooses the exact HyprFast tool.
pub fn verification_category(catalog: &HyprFastCatalog, tool: &str, fallback: &str) -> String {
    catalog
        .capability_for_mcp_name(tool)
        .map(|capability| match capability.domain {
            Domain::Browser | Domain::Stagehand | Domain::Hints => "browser",
            Domain::Excalidraw => "excalidraw",
            Domain::Vision => "vision",
            Domain::Desktop | Domain::Tasks | Domain::Clipboard | Domain::System | Domain::Unknown => "desktop",
        })
        .unwrap_or_else(|| fallback.to_owned())
}

pub fn verification_subtask(
    catalog: &HyprFastCatalog,
    tool: &str,
    original_goal: &str,
    fallback_category: &str,
    id: usize,
) -> super::SubTask {
    super::SubTask {
        id: format!("verify-{id}"),
        goal: format!("Observe and verify the current UI state after: {original_goal}. Confirm whether the intended change actually happened; do not make another change."),
        category: verification_category(catalog, tool, fallback_category),
        depends_on: Vec::new(),
    }
}
