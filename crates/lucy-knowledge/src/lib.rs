//! Lucy's knowledge base: a library of plain-Markdown topics over a rebuildable
//! SQLite index, read at three levels so a cheap model with a small context
//! window still gets the right context and nothing else.
//!
//! # Why it is shaped this way
//!
//! The design is not invented here. It is the shape every production memory
//! system converged on, and the reasons are load-bearing:
//!
//! * **Markdown files are the source of truth, the index is derived.** Nothing
//!   Lucy knows is unreachable from a text editor. Grok Build, OpenClaw and
//!   MUSE all store plain files; OpenClaw states the principle as "no hidden
//!   state", and makes the index rebuildable rather than authoritative.
//! * **Three read levels, not one big injection.** The generated digest is a
//!   table of contents (~1.2k chars). Recall is keyword-deterministic (~6k).
//!   `kb_get` pulls a book on demand. This is Anthropic's "just in time"
//!   retrieval, and it is the only shape that fits a model whose context is
//!   worth more than the tokens.
//! * **Provenance is structural, not textual.** Origin lives in a column the
//!   model cannot write through prose, and `untrusted` content can be *read* but
//!   never *injected*. This matters far more for Lucy than for a chatbot: the
//!   agent loop feeds live page text into planner prompts, so a page that can
//!   write instructions has already won.
//! * **Writing is the hard part.** Capture runs off the reply path, behind
//!   deterministic score and secret gates, and lands as *unpromoted*. Promotion
//!   is recall-weighted and closed to untrusted origins, so no model output can
//!   put itself into a prompt.
//! * **Retrieval is deterministic first.** FTS5 with a broad OR query is the
//!   recall floor. The ablation work on local 7B models found adaptive routing
//!   *losing* to fixed hybrid retrieval and a LLM-compiled wiki costing ~21x the
//!   tokens per query with no break-even point, so nothing here depends on a
//!   small model choosing correctly to return a result.

pub mod capture;
pub mod provenance;
pub mod store;
pub mod tools;
pub mod hub;

pub use provenance::{Origin, SourceKind};
pub use store::{Candidate, Chunk, KnowledgeStats, KnowledgeStore, TopicLine};
pub use tools::{KbGetTool, KbSearchTool, MemoryAssetsTool, MemorySearchTool, MemorySlimTool, MemoryStatusTool};
pub use hub::{AssetKind, MemoryAsset, MemoryHub, MemoryItem, MemoryLayer};

use std::path::{Path, PathBuf};

/// Where the knowledge base lives, unless configuration says otherwise.
///
/// Under `~/.config/lucy/knowledge`, beside the skills directory it shares a
/// mental model with: drop a Markdown file in, and it is knowledge with no code
/// change and no release.
pub fn default_root() -> PathBuf {
    if let Some(dir) = std::env::var_os("LUCY_KNOWLEDGE_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
        .join(".config/lucy/knowledge")
}

/// Open the store at `root`, creating it when absent.
pub async fn open(root: impl AsRef<Path>) -> KnowledgeStore {
    KnowledgeStore::open(root).await
}

/// Open the store at [`default_root`].
pub async fn open_default() -> KnowledgeStore {
    open(default_root()).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_root_is_editable_by_hand() {
        // An env override exists so tests and sandboxes do not touch the real
        // knowledge base; the default must stay a plain directory a user can
        // open in their editor.
        let root = default_root();
        assert!(root.ends_with("knowledge"));
    }

    #[tokio::test]
    async fn a_fresh_store_opens_empty_rather_than_failing() {
        let dir = std::env::temp_dir().join(format!("lucy-kb-lib-{}", std::process::id()));
        let _ = tokio::fs::remove_dir_all(&dir).await;
        let store = open(&dir).await;
        assert!(store.all_chunks().await.is_empty());
        assert!(store.digest().await.is_empty());
        assert!(store.context_for("anything").await.is_empty());
        let _ = tokio::fs::remove_dir_all(&dir).await;
    }
}
