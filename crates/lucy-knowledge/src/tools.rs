//! The two model-facing tools, which is where a knowledge base stops being a
//! library and becomes something an agent can use.
//!
//! Anthropic's framing for this shape is "just in time" retrieval: the model
//! holds lightweight identifiers and pulls content in when it decides it needs
//! it, rather than having everything pre-loaded. That is the only approach that
//! survives a cheap model, because the alternative is paying for every topic on
//! every turn.
//!
//! The split is deliberate and asymmetric:
//!
//! * `kb_search` — the model finds something. It may look at *quarantined*
//!   material, because reading a page Lucy found is legitimate; obeying it is not.
//! * `kb_get` — the model opens one topic by name.
//!
//! Both are read-only and need no approval: they cannot change the machine.

use crate::{KnowledgeStore, MemoryHub, store::SEARCH_LIMIT};
use async_trait::async_trait;
use lucy_core::{Tool, ToolContext};
use serde_json::{Value, json};
use std::sync::Arc;

pub struct KbSearchTool {
    store: Arc<KnowledgeStore>,
}

impl KbSearchTool {
    pub fn new(store: Arc<KnowledgeStore>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Tool for KbSearchTool {
    fn name(&self) -> &str {
        "kb_search"
    }

    fn description(&self) -> &str {
        "Search Lucy's knowledge base for a topic, a preference, or a durable fact she \
         already knows. Returns matching topics with their text. Read-only: it changes \
         nothing. Use it when the knowledge index lists a topic that looks relevant but \
         you do not have its content yet."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, in plain words."},
                "limit": {"type": "integer", "description": "Max results (1-8, default 4)."}
            },
            "required": ["query"]
        })
    }

    /// Reading knowledge is not an action on the machine, so it never raises an
    /// approval prompt. Default `requires_approval` is `true`; this overrides it.
    fn requires_approval(&self) -> bool {
        false
    }

    async fn execute(&self, input: Value, ctx: ToolContext) -> anyhow::Result<Value> {
        if ctx.interrupt.is_set() {
            return Err(lucy_core::LucyError::Cancelled.into());
        }
        let query = input
            .get("query")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|q| !q.is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!(lucy_core::LucyError::InvalidInput(
                    "query is required".into()
                ))
            })?;
        let limit = input
            .get("limit")
            .and_then(Value::as_u64)
            .unwrap_or(4)
            .clamp(1, SEARCH_LIMIT as u64) as usize;
        let hits = self.store.search(query, limit, false).await;
        if hits.is_empty() {
            return Ok(json!({
                "matches": [],
                "note": "nothing in the knowledge base matched; carry on without it"
            }));
        }
        Ok(json!({
            "matches": hits
                .iter()
                .map(|h| json!({
                    "slug": h.slug,
                    "heading": h.heading,
                    "text": h.body,
                    "origin": h.origin.as_str(),
                    "promoted": h.promoted
                }))
                .collect::<Vec<_>>()
        }))
    }
}

pub struct KbGetTool {
    store: Arc<KnowledgeStore>,
}

impl KbGetTool {
    pub fn new(store: Arc<KnowledgeStore>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Tool for KbGetTool {
    fn name(&self) -> &str {
        "kb_get"
    }

    fn description(&self) -> &str {
        "Read one knowledge topic in full, by the slug shown in Lucy's knowledge index \
         (for example `kb_get {\"slug\": \"preference-use-zsh\"}`). Read-only: it changes \
         nothing. Use it when a listed topic looks relevant and you need its content."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "slug": {"type": "string", "description": "Topic slug from the knowledge index."}
            },
            "required": ["slug"]
        })
    }

    fn requires_approval(&self) -> bool {
        false
    }

    async fn execute(&self, input: Value, ctx: ToolContext) -> anyhow::Result<Value> {
        if ctx.interrupt.is_set() {
            return Err(lucy_core::LucyError::Cancelled.into());
        }
        let slug = input
            .get("slug")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!(lucy_core::LucyError::InvalidInput(
                    "slug is required".into()
                ))
            })?;
        match self.store.topic(slug).await {
            Some(text) => Ok(json!({"slug": slug, "text": text})),
            None => Err(crate::store::unknown_topic(slug)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        provenance::SourceKind,
        store::{Candidate, KnowledgeStore},
    };

    fn context() -> ToolContext {
        let (events, _rx) = tokio::sync::mpsc::unbounded_channel();
        ToolContext {
            session_id: lucy_core::SessionId::default(),
            tool_call_id: "kb-1".into(),
            working_dir: None,
            execution_mode: lucy_core::ExecutionMode::Agent,
            events,
            interrupt: lucy_core::InterruptSignal::new(),
        }
    }

    async fn seeded(tag: &str) -> (std::path::PathBuf, Arc<KnowledgeStore>) {
        let dir = std::env::temp_dir().join(format!(
            "lucy-kb-tools-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = tokio::fs::remove_dir_all(&dir).await;
        let store = Arc::new(KnowledgeStore::open(&dir).await);
        let id = store
            .store(Candidate {
                slug: "shell".into(),
                heading: "interactive shell".into(),
                body: "the interactive shell is fish".into(),
                source: SourceKind::OwnerUtterance,
                supersedes: None,
            })
            .await
            .expect("storing owner memory");
        store.promote(id).await.expect("promoting owner memory");
        (dir, store)
    }

    #[tokio::test]
    async fn both_knowledge_tools_are_read_only() {
        let (dir, store) = seeded("approval").await;
        assert!(
            !KbSearchTool::new(store.clone()).requires_approval(),
            "reading Lucy's own notes cannot touch the machine"
        );
        assert!(!KbGetTool::new(store.clone()).requires_approval());
        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn search_finds_a_topic_and_says_so_when_it_finds_nothing() {
        let (dir, store) = seeded("search").await;
        let tool = KbSearchTool::new(store.clone());
        let hit = tool
            .execute(json!({"query": "interactive shell"}), context())
            .await
            .expect("search succeeds");
        let matches = hit["matches"].as_array().expect("an array of matches");
        assert_eq!(matches[0]["slug"], "shell");

        let miss = tool
            .execute(json!({"query": "quantum chromodynamics"}), context())
            .await
            .expect("an empty result is not an error");
        assert_eq!(miss["matches"].as_array().map(Vec::len), Some(0));
        assert!(
            miss["note"]
                .as_str()
                .is_some_and(|n| n.contains("carry on")),
            "an empty result must tell the model to proceed, not to retry"
        );
        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn get_returns_one_topic_and_names_the_miss() {
        let (dir, store) = seeded("get").await;
        let tool = KbGetTool::new(store.clone());
        let ok = tool
            .execute(json!({"slug": "shell"}), context())
            .await
            .expect("found");
        assert!(ok["text"].as_str().unwrap_or_default().contains("fish"));

        let miss = tool
            .execute(json!({"slug": "nonexistent"}), context())
            .await
            .expect_err("an unknown slug is an error the model can act on");
        assert!(
            miss.to_string().contains("kb_search"),
            "the error must point at the way out: {miss}"
        );
        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn a_missing_argument_is_reported_rather_than_guessed() {
        let (dir, store) = seeded("args").await;
        assert!(
            KbSearchTool::new(store.clone())
                .execute(json!({}), context())
                .await
                .is_err()
        );
        assert!(
            KbGetTool::new(store.clone())
                .execute(json!({"slug": "  "}), context())
                .await
                .is_err()
        );
        let _ = tokio::fs::remove_dir_all(&dir).await;
    }
}


/// Search the layered Memory Hub. This is separate from kb_search because
/// memory assets include episodic/persona/scenario records as well as durable KB.
pub struct MemorySearchTool { hub: Arc<MemoryHub> }
impl MemorySearchTool {
    pub fn new(hub: Arc<MemoryHub>) -> Self { Self { hub } }
}
#[async_trait]
impl Tool for MemorySearchTool {
    fn name(&self) -> &str { "memory_search" }
    fn description(&self) -> &str {
        "Search Lucy's layered memory on demand. Use this for exact facts, project context,
         conversation recall, scenarios, and persona. It does not inject the whole memory store."
    }
    fn parameters_schema(&self) -> Value {
        json!({"type":"object","properties":{
            "query":{"type":"string"},
            "layer":{"type":"string","enum":["conversation","atom","scenario","persona"]},
            "bootstrap":{"type":"boolean","description":"Prefer scenario/persona context before precise facts."},
            "limit":{"type":"integer","minimum":1,"maximum":16}
        },"required":["query"]})
    }
    fn requires_approval(&self) -> bool { false }
    async fn execute(&self, input: Value, ctx: ToolContext) -> anyhow::Result<Value> {
        if ctx.interrupt.is_set() { return Err(lucy_core::LucyError::Cancelled.into()); }
        let query = input.get("query").and_then(Value::as_str).map(str::trim)
            .filter(|q| !q.is_empty()).ok_or_else(|| anyhow::anyhow!("query is required"))?;
        let limit = input.get("limit").and_then(Value::as_u64).unwrap_or(6).clamp(1,16) as usize;
        let layer = match input.get("layer").and_then(Value::as_str) {
            Some("conversation") => Some(crate::MemoryLayer::Conversation),
            Some("atom") => Some(crate::MemoryLayer::Atom),
            Some("scenario") => Some(crate::MemoryLayer::Scenario),
            Some("persona") => Some(crate::MemoryLayer::Persona),
            Some(other) => return Err(anyhow::anyhow!("unknown memory layer: {other}")),
            None => None,
        };
        let hits = if input.get("bootstrap").and_then(Value::as_bool).unwrap_or(false) && layer.is_none() {
            self.hub.bootstrap(query, limit).await
        } else {
            self.hub.search_layers(query, layer, limit).await
        };
        Ok(json!({"matches": hits, "count": hits.len()}))
    }
}

/// Lightweight maintenance/status surfaces modelled after self-hosted memory hubs.
/// Status is safe; slim requires approval because it mutates the local memory index.
pub struct MemoryStatusTool { hub: Arc<MemoryHub> }
impl MemoryStatusTool { pub fn new(hub: Arc<MemoryHub>) -> Self { Self { hub } } }
#[async_trait]
impl Tool for MemoryStatusTool {
    fn name(&self) -> &str { "memory_status" }
    fn description(&self) -> &str { "Show counts for Lucy's conversation, atom, scenario, persona and reusable asset memory." }
    fn parameters_schema(&self) -> Value { json!({"type":"object","properties":{}}) }
    fn requires_approval(&self) -> bool { false }
    async fn execute(&self, _input: Value, ctx: ToolContext) -> anyhow::Result<Value> {
        if ctx.interrupt.is_set() { return Err(lucy_core::LucyError::Cancelled.into()); }
        Ok(serde_json::to_value(self.hub.stats().await)?)
    }
}

pub struct MemoryAssetsTool { hub: Arc<MemoryHub> }
impl MemoryAssetsTool { pub fn new(hub: Arc<MemoryHub>) -> Self { Self { hub } } }
#[async_trait]
impl Tool for MemoryAssetsTool {
    fn name(&self) -> &str { "memory_assets" }
    fn description(&self) -> &str { "List reusable Chat Memory, Skill, Wiki and CodeGraph assets known to Lucy." }
    fn parameters_schema(&self) -> Value {
        json!({"type":"object","properties":{"kind":{"type":"string","enum":["chat_memory","skill","wiki","code_graph"]}}})
    }
    fn requires_approval(&self) -> bool { false }
    async fn execute(&self, input: Value, ctx: ToolContext) -> anyhow::Result<Value> {
        if ctx.interrupt.is_set() { return Err(lucy_core::LucyError::Cancelled.into()); }
        let kind = match input.get("kind").and_then(Value::as_str) {
            Some("chat_memory") => Some(crate::AssetKind::ChatMemory),
            Some("skill") => Some(crate::AssetKind::Skill),
            Some("wiki") => Some(crate::AssetKind::Wiki),
            Some("code_graph") => Some(crate::AssetKind::CodeGraph),
            Some(other) => return Err(anyhow::anyhow!("unknown asset kind: {other}")),
            None => None,
        };
        Ok(json!({"assets": self.hub.assets(kind).await}))
    }
}

pub struct MemorySlimTool { hub: Arc<MemoryHub> }
impl MemorySlimTool { pub fn new(hub: Arc<MemoryHub>) -> Self { Self { hub } } }
#[async_trait]
impl Tool for MemorySlimTool {
    fn name(&self) -> &str { "memory_slim" }
    fn description(&self) -> &str {
        "Conservatively maintain Lucy memory by removing exact duplicate L1 atoms only. \
         It never deletes authored Markdown knowledge or scenario/persona records."
    }
    fn parameters_schema(&self) -> Value { json!({"type":"object","properties":{}}) }
    fn requires_approval(&self) -> bool { true }
    async fn execute(&self, _input: Value, ctx: ToolContext) -> anyhow::Result<Value> {
        if ctx.interrupt.is_set() { return Err(lucy_core::LucyError::Cancelled.into()); }
        let (before, after, removed) = self.hub.slim().await?;
        Ok(json!({"removed_duplicates":removed,"before":before,"after":after}))
    }
}
