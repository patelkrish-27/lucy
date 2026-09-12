use std::{collections::HashMap, sync::Arc};
use anyhow::{anyhow, Result};
use lucy_core::*;
use serde_json::Value;

pub struct ToolRegistry { tools: HashMap<String, Arc<dyn Tool>> }
impl ToolRegistry {
    pub fn new() -> Self { Self { tools: HashMap::new() } }
    pub fn register<T: Tool + 'static>(&mut self, tool: T) { self.tools.insert(tool.name().to_string(), Arc::new(tool)); }
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> { self.tools.get(name).cloned() }
    pub fn definitions(&self) -> Vec<Value> {
        self.tools.values().map(|t| serde_json::json!({"name":t.name(),"description":t.description(),"input_schema":t.parameters_schema()})).collect()
    }
    pub async fn execute(&self, name: &str, input: Value, ctx: ToolContext) -> Result<Value> {
        let tool = self.get(name).ok_or_else(|| anyhow!(LucyError::ToolNotFound(name.to_string())))?;
        tool.execute(input, ctx).await
    }
}
impl Default for ToolRegistry { fn default() -> Self { Self::new() } }

pub struct ShellTool;
#[async_trait::async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &str { "shell" }
    fn description(&self) -> &str { "Run a shell command in Lucy's working directory." }
    fn parameters_schema(&self) -> Value { serde_json::json!({"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}) }
    async fn execute(&self, input: Value, ctx: ToolContext) -> Result<Value> {
        if ctx.interrupt.is_set() { return Err(LucyError::Cancelled.into()); }
        let command = input.get("command").and_then(Value::as_str).ok_or_else(|| anyhow!(LucyError::InvalidInput("command is required".into())))?;
        let output = tokio::process::Command::new("sh").arg("-lc").arg(command).current_dir(ctx.working_dir.unwrap_or_else(|| std::env::current_dir().unwrap_or_default())).output().await?;
        Ok(serde_json::json!({"status":output.status.code(),"stdout":String::from_utf8_lossy(&output.stdout),"stderr":String::from_utf8_lossy(&output.stderr)}))
    }
}
