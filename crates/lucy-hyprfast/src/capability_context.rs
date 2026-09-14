use super::{HyprFastCatalog, ToolCapability};

/// Builds a compact model-facing index of the entire discovered HyprFast catalog.
/// Exact JSON schemas stay in the routed tool set; this index teaches the command
/// compiler what capabilities exist without duplicating large schemas.
pub fn full_catalog_context(catalog: &HyprFastCatalog) -> String {
    let mut tools: Vec<&ToolCapability> = catalog.tools.values().collect();
    tools.sort_by(|a, b| a.name.cmp(&b.name));

    let mut out = format!("HyprFast has {} discovered commands. Treat this index as capability metadata; only supplied schemas are executable.\n", tools.len());
    out.push_str("All available commands:\n");
    for tool in tools {
        let capabilities = tool.capabilities.iter().map(|c| format!("{:?}", c)).collect::<Vec<_>>().join(",");
        out.push_str(&format!("- {} | domain={:?} | operation={:?} | capabilities=[{}] | read_only={} | destructive={} | batchable={} | {}\n", tool.name, tool.domain, tool.operation, capabilities, tool.read_only, tool.destructive, tool.batchable, tool.description));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucy_mcp::McpToolDefinition;

    fn tool(name: &str, description: &str) -> McpToolDefinition {
        McpToolDefinition { name: name.into(), description: Some(description.into()), input_schema: serde_json::json!({"type":"object"}) }
    }

    #[test]
    fn exposes_every_discovered_command() {
        let catalog = HyprFastCatalog::from_tools(vec![tool("browser_click", "click"), tool("browser_navigate", "navigate")]);
        let context = full_catalog_context(&catalog);
        assert!(context.contains("browser_click"));
        assert!(context.contains("browser_navigate"));
        assert!(context.contains("HyprFast has 2 discovered commands"));
    }
}
