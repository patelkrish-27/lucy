use lucy_hyprfast::{Domain, HyprFastCatalog};

pub fn action_requires_verification(catalog:&HyprFastCatalog,tool:&str,category:&str,explicit_verify:bool)->bool{
    if explicit_verify{return true;}
    let Some(capability)=catalog.capability_for_mcp_name(tool) else{return !matches!(category,"vision"|"browser"|"excalidraw")};
    !capability.read_only
}

pub fn verification_category(catalog:&HyprFastCatalog,tool:&str,fallback:&str)->String{
    catalog.capability_for_mcp_name(tool).map(|capability|match capability.domain{
        Domain::Browser|Domain::Stagehand|Domain::Hints=>"browser",
        Domain::Excalidraw=>"excalidraw",
        Domain::Vision=>"vision",
        Domain::Desktop|Domain::Tasks|Domain::Clipboard|Domain::System|Domain::Unknown=>"desktop",
    }).unwrap_or_else(||fallback.to_owned())
}

pub fn verification_subtask(catalog:&HyprFastCatalog,tool:&str,original_goal:&str,fallback_category:&str,id:usize)->super::SubTask{
    super::SubTask{id:format!("verify-{id}"),goal:format!("Observe and verify the current UI state after: {original_goal}. Confirm whether the intended change actually happened; do not make another change."),category:verification_category(catalog,tool,fallback_category),depends_on:Vec::new()}
}

#[cfg(test)]
mod tests{
    use super::*;
    use lucy_mcp::McpToolDefinition;
    fn tool(name:&str,description:&str)->McpToolDefinition{McpToolDefinition{name:name.into(),description:Some(description.into()),input_schema:serde_json::json!({"type":"object"})}}
    #[test]
    fn state_changing_tool_requires_verification(){let catalog=HyprFastCatalog::from_tools(vec![tool("browser_click","click browser element"),tool("screenshot","capture desktop")]);assert!(action_requires_verification(&catalog,"mcp_hyprfast_browser_click","browser",false));assert!(!action_requires_verification(&catalog,"mcp_hyprfast_screenshot","vision",false));}
    #[test]
    fn verification_uses_browser_domain_for_browser_actions(){let catalog=HyprFastCatalog::from_tools(vec![tool("browser_click","click browser element")]);let subtask=verification_subtask(&catalog,"mcp_hyprfast_browser_click","click the button","desktop",7);assert_eq!(subtask.id,"verify-7");assert_eq!(subtask.category,"browser");assert!(subtask.goal.contains("do not make another change"));}
}
