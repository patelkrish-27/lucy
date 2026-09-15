//! Native ADK-Rust integration for Lucy.
//!
//! Lucy keeps its latency-sensitive agent loop, HyprFast command compiler,
//! approvals, and MCP routing as the authoritative execution path. This crate
//! brings in the complementary ADK-Rust capabilities that Lucy did not already
//! implement, behind an intentionally small facade.

use std::{env, path::{Path, PathBuf}, sync::Arc};

use adk_core::Content;
use adk_memory::{MemoryEntry, MemoryService, SearchRequest, SqliteMemoryService};
use anyhow::{Context, Result};
use chrono::Utc;

pub use adk_audio;
pub use adk_core;
pub use adk_memory;
pub use adk_rust;
pub use adk_telemetry;

/// Capabilities contributed by the ADK-Rust union that are not Lucy's
/// latency-sensitive core loop.
///
/// Browser automation is intentionally excluded: HyprFast is Lucy's single
/// authoritative browser backend. Exposing ADK Browser alongside HyprFast
/// would create duplicate tools and make agent tool selection ambiguous.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdkCapability {
    PersistentSemanticMemory,
    SessionServiceBackends,
    WorkflowAgents,
    GraphWorkflows,
    Artifacts,
    Guardrails,
    Skills,
    Plugins,
    CodeExecution,
    SandboxedExecution,
    ComputerUse,
    RealtimeAgents,
    AudioPipelines,
    Evaluation,
    Telemetry,
    A2aServer,
    Authentication,
    ExpandedModelProviders,
    McpSamplingAndTransports,
}

impl AdkCapability {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PersistentSemanticMemory => "persistent-semantic-memory",
            Self::SessionServiceBackends => "session-service-backends",
            Self::WorkflowAgents => "workflow-agents",
            Self::GraphWorkflows => "graph-workflows",
            Self::Artifacts => "artifacts",
            Self::Guardrails => "guardrails",
            Self::Skills => "skills",
            Self::Plugins => "plugins",
            Self::CodeExecution => "code-execution",
            Self::SandboxedExecution => "sandboxed-execution",
            Self::ComputerUse => "computer-use",
            Self::RealtimeAgents => "realtime-agents",
            Self::AudioPipelines => "audio-pipelines",
            Self::Evaluation => "evaluation",
            Self::Telemetry => "telemetry",
            Self::A2aServer => "a2a-server",
            Self::Authentication => "authentication",
            Self::ExpandedModelProviders => "expanded-model-providers",
            Self::McpSamplingAndTransports => "mcp-sampling-and-transports",
        }
    }
}

pub const ADK_CAPABILITIES: &[AdkCapability] = &[
    AdkCapability::PersistentSemanticMemory,
    AdkCapability::SessionServiceBackends,
    AdkCapability::WorkflowAgents,
    AdkCapability::GraphWorkflows,
    AdkCapability::Artifacts,
    AdkCapability::Guardrails,
    AdkCapability::Skills,
    AdkCapability::Plugins,
    AdkCapability::CodeExecution,
    AdkCapability::SandboxedExecution,
    AdkCapability::ComputerUse,
    AdkCapability::RealtimeAgents,
    AdkCapability::AudioPipelines,
    AdkCapability::Evaluation,
    AdkCapability::Telemetry,
    AdkCapability::A2aServer,
    AdkCapability::Authentication,
    AdkCapability::ExpandedModelProviders,
    AdkCapability::McpSamplingAndTransports,
];

const DEFAULT_MEMORY_RESULTS: usize = 6;
const MAX_MEMORY_CONTEXT_CHARS: usize = 6_000;

/// Optional ADK services owned by Lucy.
///
/// The default SQLite memory location lives under Lucy's state directory. Set
/// `LUCY_ADK_MEMORY_DB` to override it. Memory initialization is intentionally
/// optional: failure to initialize this extension must never prevent the core
/// Lucy agent from starting.
pub struct LucyAdk {
    memory: Option<Arc<SqliteMemoryService>>,
    memory_path: PathBuf,
}

impl std::fmt::Debug for LucyAdk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LucyAdk")
            .field("memory_enabled", &self.memory.is_some())
            .field("memory_path", &self.memory_path)
            .finish()
    }
}

impl LucyAdk {
    pub async fn open(state_dir: impl AsRef<Path>) -> Self {
        let default_path = state_dir.as_ref().join("adk-memory.db");
        let memory_path = env::var_os("LUCY_ADK_MEMORY_DB")
            .map(PathBuf::from)
            .unwrap_or(default_path);

        if let Some(parent) = memory_path.parent() {
            if let Err(error) = tokio::fs::create_dir_all(parent).await {
                tracing::warn!(error = %error, path = %parent.display(), "ADK memory directory unavailable");
                return Self { memory: None, memory_path };
            }
        }

        let sqlite_url = format!("sqlite://{}", memory_path.display());
        match SqliteMemoryService::new(&sqlite_url).await {
            Ok(service) => {
                if let Err(error) = service.migrate().await {
                    tracing::warn!(error = %error, "ADK SQLite memory migration failed; disabling persistent memory");
                    return Self { memory: None, memory_path };
                }
                tracing::info!(path = %memory_path.display(), "ADK persistent memory enabled");
                Self { memory: Some(Arc::new(service)), memory_path }
            }
            Err(error) => {
                tracing::warn!(error = %error, path = %memory_path.display(), "ADK SQLite memory unavailable; continuing without persistent memory");
                Self { memory: None, memory_path }
            }
        }
    }

    pub fn capabilities(&self) -> &'static [AdkCapability] { ADK_CAPABILITIES }
    pub fn memory_enabled(&self) -> bool { self.memory.is_some() }
    pub fn memory_path(&self) -> &Path { &self.memory_path }

    /// Return bounded, model-ready semantic memory context for a request.
    /// Memory is advisory: live observations and the current request always
    /// outrank it. Search failures are returned to the caller so Lucy can
    /// degrade gracefully without losing the main interaction.
    pub async fn memory_context(&self, query: &str, limit: usize) -> Result<String> {
        if query.trim().is_empty() || self.memory.is_none() { return Ok(String::new()); }
        let memories = self.search_memory(query, limit.max(1)).await?;
        let mut out = String::new();
        for (index, entry) in memories.iter().enumerate() {
            let block = format!("\n- Memory {}: {:?}\n", index + 1, entry.content);
            if out.len() + block.len() > MAX_MEMORY_CONTEXT_CHARS { break; }
            out.push_str(&block);
        }
        Ok(out)
    }

    /// Persist the completed interaction without adding an extra LLM call.
    /// Lucy's existing session history remains the source of truth for turns.
    pub async fn remember_interaction(&self, session_id: &str, prompt: &str, response: &str) -> Result<()> {
        let Some(memory) = &self.memory else { return Ok(()); };
        if prompt.trim().is_empty() && response.trim().is_empty() { return Ok(()); }

        let user_id = local_user_id();
        let mut entries = Vec::with_capacity(2);
        if !prompt.trim().is_empty() {
            entries.push(MemoryEntry {
                content: Content::new("user").with_text(prompt.to_owned()),
                author: "user".to_owned(),
                timestamp: Utc::now(),
            });
        }
        if !response.trim().is_empty() {
            entries.push(MemoryEntry {
                content: Content::new("model").with_text(response.to_owned()),
                author: "lucy".to_owned(),
                timestamp: Utc::now(),
            });
        }
        memory.add_session("lucy", &user_id, session_id, entries).await.context("storing interaction in ADK memory")
    }

    pub async fn search_memory(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>> {
        let Some(memory) = &self.memory else { return Ok(Vec::new()); };
        let response = memory.search(SearchRequest {
            query: query.to_owned(),
            user_id: local_user_id(),
            app_name: "lucy".to_owned(),
            limit: Some(limit.max(1)),
            min_score: None,
            project_id: None,
        }).await.context("searching ADK memory")?;
        Ok(response.memories)
    }

    /// Store an explicit durable fact without requiring a separate session.
    pub async fn remember_fact(&self, fact: &str) -> Result<()> {
        let fact = fact.trim();
        if fact.is_empty() { return Ok(()); }
        let Some(memory) = &self.memory else { return Ok(()); };
        let entry = MemoryEntry {
            content: Content::new("memory").with_text(fact.to_owned()),
            author: "lucy-memory".to_owned(),
            timestamp: Utc::now(),
        };
        memory.add_session("lucy", &local_user_id(), "explicit-memory", vec![entry]).await.context("storing explicit memory")
    }
}

fn local_user_id() -> String {
    env::var("LUCY_USER_ID").or_else(|_| env::var("USER")).unwrap_or_else(|_| "local".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_manifest_is_stable() {
        assert!(ADK_CAPABILITIES.contains(&AdkCapability::PersistentSemanticMemory));
        assert!(ADK_CAPABILITIES.contains(&AdkCapability::GraphWorkflows));
        assert!(ADK_CAPABILITIES.contains(&AdkCapability::SandboxedExecution));
        assert!(ADK_CAPABILITIES.contains(&AdkCapability::Evaluation));
        assert_eq!(AdkCapability::Telemetry.as_str(), "telemetry");
        assert!(!ADK_CAPABILITIES.iter().any(|cap| cap.as_str() == "browser-automation"));
    }

    #[test]
    fn local_user_has_a_safe_fallback() { assert!(!local_user_id().trim().is_empty()); }

    #[test]
    fn memory_context_budget_is_reasonable() { assert!(MAX_MEMORY_CONTEXT_CHARS >= 1_000); }
}
