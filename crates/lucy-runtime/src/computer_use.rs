use std::collections::BTreeMap;

use anyhow::{Context, Result};
use lucy_mcp::{McpServerConfig, McpToolDefinition, StdioMcpClient};

#[derive(Debug, Clone)]
pub struct ComputerTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub read_only: bool,
    pub destructive: bool,
    pub semantic: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ComputerUseCatalog {
    pub tools: BTreeMap<String, ComputerTool>,
}

#[derive(Debug, Clone, Default)]
pub struct ComputerRoute {
    pub candidates: Vec<String>,
    pub strategy: String,
}

impl ComputerUseCatalog {
    pub async fn discover(config: McpServerConfig) -> Result<Self> {
        let client = StdioMcpClient::new(config);
        let definitions = client
            .list_tools()
            .await
            .context("failed to discover ADK Computer Use MCP tools")?;
        Ok(Self::from_tools(definitions))
    }

    pub fn from_tools(definitions: Vec<McpToolDefinition>) -> Self {
        let mut tools = BTreeMap::new();
        for definition in definitions {
            let name = format!("mcp_computer_use_{}", sanitize(&definition.name));
            let lower = format!("{} {}", definition.name, definition.description.clone().unwrap_or_default()).to_ascii_lowercase();
            let read_only = any(&lower, &["screenshot", "get_", "list_", "find_", "inspect", "read_clipboard", "focused", "frontmost", "capabilities", "doctor"]);
            let destructive = any(&lower, &["close", "kill", "delete", "shutdown", "remove", "terminate"]);
            let semantic = any(&lower, &["element", "ui_tree", "button", "menu", "form", "accessibility", "focused"]);
            tools.insert(name.clone(), ComputerTool {
                name,
                description: definition.description.unwrap_or_default(),
                input_schema: definition.input_schema,
                read_only,
                destructive,
                semantic,
            });
        }
        Self { tools }
    }

    pub fn len(&self) -> usize { self.tools.len() }
    pub fn is_empty(&self) -> bool { self.tools.is_empty() }

    pub fn route(&self, goal: &str) -> ComputerRoute {
        let text = goal.to_ascii_lowercase();
        let explicit_app = any(&text, &["open app", "launch app", "start app", "application", "program"]);
        let explicit_window = any(&text, &["window", "workspace", "focus", "switch to"]);
        let explicit_menu = any(&text, &["menu", "settings", "preferences"]);
        let explicit_form = any(&text, &["form", "fill", "field", "input"]);
        let explicit_clipboard = any(&text, &["clipboard", "copy", "paste"]);
        let explicit_visual = any(&text, &["screenshot", "screen", "look", "see", "visual"]);
        let mut scored: Vec<(i32, &ComputerTool)> = self.tools.values().map(|tool| {
            let n = tool.name.to_ascii_lowercase();
            let mut score = 0;
            if explicit_visual && any(&n, &["screenshot", "observe", "screen"]) { score += 100; }
            if explicit_app && any(&n, &["application", "app_"]) { score += 90; }
            if explicit_window && any(&n, &["window", "space", "focus", "display"]) { score += 90; }
            if explicit_menu && any(&n, &["menu"]) { score += 100; }
            if explicit_form && any(&n, &["form", "field", "element", "type"]) { score += 90; }
            if explicit_clipboard && any(&n, &["clipboard", "copy", "paste"]) { score += 90; }
            if any(&text, &["click", "press", "button", "select"]) && any(&n, &["click", "button", "element", "press"]) { score += 80; }
            if any(&text, &["type", "write", "enter", "fill"]) && any(&n, &["type", "form", "key", "element"]) { score += 80; }
            if any(&text, &["scroll", "drag", "move mouse", "mouse"]) && any(&n, &["scroll", "drag", "mouse", "pointer"]) { score += 80; }
            if tool.semantic && any(&text, &["button", "field", "control", "element", "menu"]) { score += 25; }
            if tool.read_only && any(&text, &["find", "inspect", "what", "check", "see"]) { score += 20; }
            if tool.destructive && !any(&text, &["close", "delete", "kill", "shutdown", "remove"]) { score -= 100; }
            (score, tool)
        }).collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.name.cmp(&b.1.name)));
        let candidates = scored.into_iter().filter(|(score, _)| *score > 0).take(10).map(|(_, t)| t.name.clone()).collect::<Vec<_>>();
        let strategy = if explicit_visual { "visual-observe" }
            else if explicit_menu { "semantic-menu" }
            else if explicit_form { "semantic-form" }
            else if explicit_app || explicit_window { "app-window-semantic" }
            else { "accessibility-first" };
        ComputerRoute { candidates, strategy: strategy.into() }
    }

    pub fn context_for(&self, route: &ComputerRoute) -> String {
        let mut out = format!("ADK Computer Use catalog: {} tools; strategy={}\n", self.len(), route.strategy);
        out.push_str("Candidates:\n");
        for name in &route.candidates {
            if let Some(tool) = self.tools.get(name) {
                out.push_str(&format!("  {} | read_only={} | destructive={} | semantic={} | {}\n", tool.name, tool.read_only, tool.destructive, tool.semantic, tool.description));
            }
        }
        out
    }

    pub fn definitions_for(&self, names: &std::collections::HashSet<String>) -> Vec<lucy_mcp::McpToolDefinition> {
        // The registry is authoritative for executable schemas. This helper only exists
        // for tests and routing metadata; runtime obtains the exact schemas from it.
        names.iter().filter_map(|name| self.tools.get(name)).map(|tool| McpToolDefinition {
            name: tool.name.strip_prefix("mcp_computer_use_").unwrap_or(&tool.name).to_string(),
            description: Some(tool.description.clone()),
            input_schema: tool.input_schema.clone(),
        }).collect()
    }
}

fn sanitize(value: &str) -> String { value.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect() }
fn any(text: &str, terms: &[&str]) -> bool { terms.iter().any(|term| text.contains(term)) }

#[cfg(test)]
mod tests {
    use super::*;
    fn tool(name: &str, description: &str) -> McpToolDefinition { McpToolDefinition { name: name.into(), description: Some(description.into()), input_schema: serde_json::json!({"type":"object"}) } }

    #[test]
    fn routes_semantic_button_before_pointer() {
        let c = ComputerUseCatalog::from_tools(vec![tool("find_element", "find accessibility element"), tool("press_button", "press a button"), tool("left_click", "click coordinates")]);
        let r = c.route("click the Save button");
        assert!(r.candidates.iter().any(|x| x.contains("press_button")));
        assert_eq!(r.strategy, "accessibility-first");
    }

    #[test]
    fn routes_menu_operations() {
        let c = ComputerUseCatalog::from_tools(vec![tool("list_menu_bar", "list menu bar"), tool("select_menu_item", "select menu item"), tool("left_click", "click coordinates")]);
        let r = c.route("open the File menu and select Save");
        assert!(r.candidates.iter().any(|x| x.contains("menu")));
        assert_eq!(r.strategy, "semantic-menu");
    }
}
