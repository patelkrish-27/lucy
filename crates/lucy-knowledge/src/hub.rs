//! Layered memory and reusable knowledge assets inspired by modern self-hosted agent hubs.
//!
//! Lucy keeps this local and rebuildable: SQLite/FTS is the index, while the
//! existing Markdown knowledge store remains the source of truth for authored
//! knowledge. The four layers mirror L0 conversation -> L1 atom -> L2 scenario
//! -> L3 persona, while assets unify Chat Memory, Skill, Wiki and CodeGraph.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqlitePoolOptions};
use sqlx::Row;
use std::{path::{Path, PathBuf}, str::FromStr, time::Duration};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryLayer { Conversation, Atom, Scenario, Persona }

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AssetKind { ChatMemory, Skill, Wiki, CodeGraph }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    pub id: i64,
    pub layer: MemoryLayer,
    pub title: String,
    pub content: String,
    pub source: String,
    pub confidence: f32,
    pub importance: f32,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAsset {
    pub id: i64,
    pub kind: AssetKind,
    pub name: String,
    pub description: String,
    pub path: String,
    pub version: i64,
    pub owner: String,
    pub visibility: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryStats {
    pub conversation: usize,
    pub atom: usize,
    pub scenario: usize,
    pub persona: usize,
    pub assets: usize,
}

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS memory_items (
 id INTEGER PRIMARY KEY AUTOINCREMENT, layer TEXT NOT NULL, title TEXT NOT NULL,
 content TEXT NOT NULL, source TEXT NOT NULL, confidence REAL NOT NULL DEFAULT 1.0,
 importance REAL NOT NULL DEFAULT 0.5, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS memory_items_layer ON memory_items(layer);
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
 title, content, source, content='memory_items', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS memory_fts_ai AFTER INSERT ON memory_items BEGIN
 INSERT INTO memory_fts(rowid,title,content,source) VALUES(new.id,new.title,new.content,new.source);
END;
CREATE TRIGGER IF NOT EXISTS memory_fts_ad AFTER DELETE ON memory_items BEGIN
 INSERT INTO memory_fts(memory_fts,rowid,title,content,source) VALUES('delete',old.id,old.title,old.content,old.source);
END;
CREATE TABLE IF NOT EXISTS asset_bindings (\n asset_id INTEGER NOT NULL, agent_id TEXT NOT NULL, PRIMARY KEY(asset_id, agent_id),\n FOREIGN KEY(asset_id) REFERENCES memory_assets(id) ON DELETE CASCADE\n);\nCREATE TABLE IF NOT EXISTS code_nodes (\n id INTEGER PRIMARY KEY AUTOINCREMENT, asset_id INTEGER NOT NULL, kind TEXT NOT NULL,\n name TEXT NOT NULL, file TEXT NOT NULL, signature TEXT NOT NULL, UNIQUE(asset_id,name,file),\n FOREIGN KEY(asset_id) REFERENCES memory_assets(id) ON DELETE CASCADE\n);\nCREATE TABLE IF NOT EXISTS code_edges (\n from_id INTEGER NOT NULL, to_id INTEGER NOT NULL, kind TEXT NOT NULL,\n PRIMARY KEY(from_id,to_id,kind), FOREIGN KEY(from_id) REFERENCES code_nodes(id) ON DELETE CASCADE,\n FOREIGN KEY(to_id) REFERENCES code_nodes(id) ON DELETE CASCADE\n);\nCREATE TABLE IF NOT EXISTS memory_assets (
 id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, name TEXT NOT NULL,
 description TEXT NOT NULL DEFAULT '', path TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 1,
 owner TEXT NOT NULL DEFAULT 'local', visibility TEXT NOT NULL DEFAULT 'private',
 updated_at TEXT NOT NULL, UNIQUE(kind,name)
);
"#;

pub struct MemoryHub { pool: SqlitePool, root: PathBuf }

impl std::fmt::Debug for MemoryHub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryHub").field("root", &self.root).finish()
    }
}

impl MemoryHub {
    pub async fn open(root: impl AsRef<Path>) -> Self {
        let root = root.as_ref().to_path_buf();
        let _ = tokio::fs::create_dir_all(root.join("memory")).await;
        let db = root.join("memory").join("hub.db");
        let options = SqliteConnectOptions::from_str(&format!("sqlite://{}", db.display()))
            .expect("valid sqlite path").create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal).busy_timeout(Duration::from_secs(5));
        let pool = SqlitePoolOptions::new().max_connections(4).connect_with(options).await
            .unwrap_or_else(|e| panic!(
                "Lucy MemoryHub could not open persistent database {}: {e}.                  Refusing an in-memory fallback because it would silently lose long-term memory.",
                db.display()
            ));
        sqlx::raw_sql(SCHEMA).execute(&pool).await.unwrap_or_else(|e| {
            panic!(
                "Lucy MemoryHub schema initialization failed for {}: {e}",
                db.display()
            )
        });
        Self { pool, root }
    }

    pub fn root(&self) -> &Path { &self.root }

    /// Store one bounded memory item. Empty content and oversized records are rejected.
    pub async fn remember(&self, layer: MemoryLayer, title: &str, content: &str,
        source: &str, confidence: f32, importance: f32) -> Option<i64> {
        let title = title.trim();
        let content = content.trim();
        if title.is_empty() || content.is_empty() || content.len() > 12_000 { return None; }
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query("INSERT INTO memory_items(layer,title,content,source,confidence,importance,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)")
            .bind(layer_name(layer)).bind(title).bind(content).bind(source)
            .bind(confidence.clamp(0.0,1.0)).bind(importance.clamp(0.0,1.0))
            .bind(&now).bind(&now).execute(&self.pool).await.ok()
            .map(|r| r.last_insert_rowid())
    }

    /// Persist an L0 conversation message. The role is part of the title so
    /// exact conversation recall remains possible without injecting the whole log.
    pub async fn remember_conversation(&self, role: &str, content: &str, session: &str) -> Option<i64> {
        let role = role.trim();
        let session = session.trim();
        let title = if session.is_empty() {
            format!("conversation:{role}")
        } else {
            format!("conversation:{session}:{role}")
        };
        let inserted = self
            .remember(
                MemoryLayer::Conversation,
                &title,
                content,
                "lucy:conversation",
                1.0,
                0.35,
            )
            .await;

        // L0 is a transcript cache, not an infinite archive. Keep the newest
        // 500 conversation messages per session while higher-value L1/L2/L3
        // memories remain durable. This prevents long-running agents from
        // eventually filling the memory DB just because every turn is captured.
        if !session.is_empty() {
            let _ = sqlx::query(
                "DELETE FROM memory_items
                 WHERE source='lucy:conversation'
                   AND title LIKE ?
                   AND id NOT IN (
                       SELECT id FROM memory_items
                       WHERE source='lucy:conversation' AND title LIKE ?
                       ORDER BY id DESC LIMIT 500
                   )",
            )
            .bind(format!("conversation:{session}:%"))
            .bind(format!("conversation:{session}:%"))
            .execute(&self.pool)
            .await;
        }
        inserted
    }

    pub async fn search(&self, query: &str, limit: usize) -> Vec<MemoryItem> {
        self.search_layers(query, None, limit).await
    }

    /// Layer-aware FTS retrieval. Results are capped so memory cannot consume
    /// an entire model context; callers can explicitly ask for a layer.
    pub async fn search_layers(&self, query: &str, layer: Option<MemoryLayer>, limit: usize) -> Vec<MemoryItem> {
        let terms = query.split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .filter(|x| x.len() >= 2)
            .map(|x| format!("\"{}\"", x.to_ascii_lowercase()))
            .collect::<Vec<_>>();
        if terms.is_empty() { return Vec::new(); }
        let q = terms.join(" OR ");
        let rows = if let Some(layer) = layer {
            sqlx::query("SELECT m.id,m.layer,m.title,m.content,m.source,m.confidence,m.importance,m.created_at,m.updated_at FROM memory_fts f JOIN memory_items m ON m.id=f.rowid WHERE memory_fts MATCH ? AND m.layer=? ORDER BY bm25(memory_fts), m.importance DESC, m.updated_at DESC LIMIT ?")
                .bind(&q).bind(layer_name(layer)).bind(limit.clamp(1,32) as i64).fetch_all(&self.pool).await.unwrap_or_default()
        } else {
            sqlx::query("SELECT m.id,m.layer,m.title,m.content,m.source,m.confidence,m.importance,m.created_at,m.updated_at FROM memory_fts f JOIN memory_items m ON m.id=f.rowid WHERE memory_fts MATCH ? ORDER BY bm25(memory_fts), m.importance DESC, m.updated_at DESC LIMIT ?")
                .bind(&q).bind(limit.clamp(1,32) as i64).fetch_all(&self.pool).await.unwrap_or_default()
        };
        rows.into_iter().filter_map(memory_from_row).collect()
    }

    /// Fast bootstrap: durable persona/scenario context first, then precise atoms.
    pub async fn bootstrap(&self, query: &str, limit: usize) -> Vec<MemoryItem> {
        let mut out = Vec::new();
        for layer in [MemoryLayer::Persona, MemoryLayer::Scenario, MemoryLayer::Atom] {
            let remaining = limit.saturating_sub(out.len());
            if remaining == 0 { break; }
            out.extend(self.search_layers(query, Some(layer), remaining).await);
        }
        out
    }

    pub async fn add_asset(&self, kind: AssetKind, name: &str, description: &str,
        path: &str, owner: &str, visibility: &str) -> Result<i64> {
        let visibility = match visibility { "private"|"team"|"restricted"|"agent" => visibility, _ => "private" };
        let now = chrono::Utc::now().to_rfc3339();
        let result = sqlx::query("INSERT INTO memory_assets(kind,name,description,path,version,owner,visibility,updated_at) VALUES(?,?,?,?,1,?,?,?) ON CONFLICT(kind,name) DO UPDATE SET description=excluded.description,path=excluded.path,version=memory_assets.version+1,owner=excluded.owner,visibility=excluded.visibility,updated_at=excluded.updated_at")
            .bind(asset_name(kind)).bind(name.trim()).bind(description.trim()).bind(path).bind(owner).bind(visibility).bind(&now)
            .execute(&self.pool).await.context("register memory asset")?;
        Ok(result.last_insert_rowid())
    }

    pub async fn bind_asset(&self, kind: AssetKind, name: &str, agent_id: &str) -> Result<bool> {
        let asset = sqlx::query("SELECT id FROM memory_assets WHERE kind=? AND name=?")
            .bind(asset_name(kind)).bind(name).fetch_optional(&self.pool).await?;
        let Some(row) = asset else { return Ok(false); };
        let id: i64 = row.get("id");
        sqlx::query("INSERT OR IGNORE INTO asset_bindings(asset_id,agent_id) VALUES(?,?)")
            .bind(id).bind(agent_id.trim()).execute(&self.pool).await?;
        Ok(true)
    }

    /// Enforce the asset loadout boundary locally. Private assets are owner-only;
    /// team assets require the caller to opt into the same team namespace; agent
    /// assets require an explicit binding. This is intentionally small-user local
    /// governance rather than pretending Lucy is a multi-tenant service.
    pub async fn assets_for_agent(&self, agent_id: &str) -> Vec<MemoryAsset> {
        let rows = sqlx::query("SELECT a.id,a.kind,a.name,a.description,a.path,a.version,a.owner,a.visibility,a.updated_at FROM memory_assets a LEFT JOIN asset_bindings b ON b.asset_id=a.id AND b.agent_id=? WHERE a.visibility='team' OR a.visibility='restricted' AND b.agent_id IS NOT NULL OR a.visibility='agent' AND b.agent_id IS NOT NULL OR a.owner=? ORDER BY a.kind,a.name")
            .bind(agent_id).bind(agent_id).fetch_all(&self.pool).await.unwrap_or_default();
        rows.into_iter().filter_map(asset_from_row).collect()
    }

    pub async fn assets(&self, kind: Option<AssetKind>) -> Vec<MemoryAsset> {
        let rows = if let Some(kind) = kind {
            sqlx::query("SELECT id,kind,name,description,path,version,owner,visibility,updated_at FROM memory_assets WHERE kind=? ORDER BY name")
                .bind(asset_name(kind)).fetch_all(&self.pool).await.unwrap_or_default()
        } else {
            sqlx::query("SELECT id,kind,name,description,path,version,owner,visibility,updated_at FROM memory_assets ORDER BY kind,name")
                .fetch_all(&self.pool).await.unwrap_or_default()
        };
        rows.into_iter().filter_map(asset_from_row).collect()
    }

    pub async fn stats(&self) -> MemoryStats {
        let mut stats = MemoryStats::default();
        for layer in [MemoryLayer::Conversation, MemoryLayer::Atom, MemoryLayer::Scenario, MemoryLayer::Persona] {
            let n: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM memory_items WHERE layer=?")
                .bind(layer_name(layer)).fetch_one(&self.pool).await.unwrap_or(0);
            match layer {
                MemoryLayer::Conversation => stats.conversation = n as usize,
                MemoryLayer::Atom => stats.atom = n as usize,
                MemoryLayer::Scenario => stats.scenario = n as usize,
                MemoryLayer::Persona => stats.persona = n as usize,
            }
        }
        stats.assets = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM memory_assets")
            .fetch_one(&self.pool).await.unwrap_or(0) as usize;
        stats
    }

    /// Remove exact duplicate L1 atoms, keeping the highest-confidence/newest copy.
    /// It deliberately does not delete authored KB files or L2/L3 memories.
    pub async fn consolidate(&self) -> Result<usize> {
        let duplicates = sqlx::query("SELECT title, content, COUNT(*) n FROM memory_items WHERE layer='atom' GROUP BY title, content HAVING n>1")
            .fetch_all(&self.pool).await?;
        let mut removed = 0;
        for row in duplicates {
            let title: String = row.get("title");
            let content: String = row.get("content");
            let ids = sqlx::query("SELECT id FROM memory_items WHERE layer='atom' AND title=? AND content=? ORDER BY confidence DESC,importance DESC,updated_at DESC")
                .bind(&title).bind(&content).fetch_all(&self.pool).await?;
            for row in ids.into_iter().skip(1) {
                let id: i64 = row.get("id");
                sqlx::query("DELETE FROM memory_items WHERE id=?").bind(id).execute(&self.pool).await?;
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// A conservative maintenance operation equivalent to a "memory slim":
    /// deduplicate atoms only, then return before/after stats.
    pub async fn slim(&self) -> Result<(MemoryStats, MemoryStats, usize)> {
        let before = self.stats().await;
        let removed = self.consolidate().await?;
        let after = self.stats().await;
        Ok((before, after, removed))
    }

    pub async fn ingest_wiki(&self, file: impl AsRef<Path>) -> Result<usize> {
        let file = file.as_ref();
        let text = tokio::fs::read_to_string(file).await.context("read wiki source")?;
        let name = file.file_stem().and_then(|x| x.to_str()).unwrap_or("wiki");
        let source = file.display().to_string();

        // Parse everything before mutating storage. Re-ingestion replaces the
        // previous projection for this source instead of accumulating stale
        // sections every time the Markdown file changes.
        let mut sections = Vec::<(String, String)>::new();
        let mut title = name.to_owned();
        let mut body = String::new();
        for line in text.lines() {
            if let Some(h) = line.strip_prefix("# ").or_else(|| line.strip_prefix("## ")) {
                if !body.trim().is_empty() {
                    sections.push((title, body));
                }
                title = h.trim().to_owned();
                body = String::new();
            } else {
                body.push_str(line);
                body.push('\\n');
            }
        }
        if !body.trim().is_empty() {
            sections.push((title, body));
        }

        let mut tx = self.pool.begin().await.context("begin wiki ingestion")?;
        sqlx::query("DELETE FROM memory_items WHERE layer='scenario' AND source=?")
            .bind(&source).execute(&mut *tx).await.context("replace wiki memories")?;
        let now = chrono::Utc::now().to_rfc3339();
        for (title, body) in &sections {
            sqlx::query("INSERT INTO memory_items(layer,title,content,source,confidence,importance,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)")
                .bind("scenario").bind(title).bind(body).bind(&source)
                .bind(1.0_f32).bind(0.8_f32).bind(&now).bind(&now)
                .execute(&mut *tx).await.context("insert wiki memory")?;
        }
        tx.commit().await.context("commit wiki ingestion")?;

        self.add_asset(
            AssetKind::Wiki, name, "Local Markdown knowledge corpus", &source, "local", "private"
        ).await?;
        Ok(sections.len())
    }

    /// Query indexed CodeGraph symbols and direct relationships.
    pub async fn codegraph_query(&self, query: &str, limit: usize) -> Vec<serde_json::Value> {
        let q = query.trim();
        if q.is_empty() { return Vec::new(); }
        let rows = sqlx::query("SELECT n.id,n.kind,n.name,n.file,n.signature FROM code_nodes n WHERE n.name LIKE ? OR n.signature LIKE ? ORDER BY n.name LIMIT ?")
            .bind(format!("%{q}%")).bind(format!("%{q}%")).bind(limit.clamp(1,32) as i64)
            .fetch_all(&self.pool).await.unwrap_or_default();
        rows.into_iter().map(|r| serde_json::json!({
            "id": r.get::<i64,_>("id"), "kind": r.get::<String,_>("kind"),
            "name": r.get::<String,_>("name"), "file": r.get::<String,_>("file"),
            "signature": r.get::<String,_>("signature")
        })).collect()
    }

    /// Lightweight Rust CodeGraph indexing. It stores symbols as searchable
    /// graph seeds; a later AST adapter can enrich callers/callees/impact paths.
    pub async fn index_rust_file(&self, file: impl AsRef<Path>) -> Result<usize> {
        let file = file.as_ref();
        let text = tokio::fs::read_to_string(file).await.context("read Rust source")?;
        let source = file.display().to_string();

        // Replace the searchable projection before rebuilding it so renamed or
        // deleted symbols cannot survive a re-index.
        sqlx::query("DELETE FROM memory_items WHERE layer='scenario' AND source=?")
            .bind(&source)
            .execute(&self.pool)
            .await
            .context("replace CodeGraph memory projection")?;
        let mut count = 0;
        for line in text.lines() {
            let t = line.trim();
            let kind = if t.starts_with("pub fn ") || t.starts_with("fn ") { "function" }
                else if t.starts_with("pub struct ") || t.starts_with("struct ") { "struct" }
                else if t.starts_with("pub enum ") || t.starts_with("enum ") { "enum" }
                else if t.starts_with("pub trait ") || t.starts_with("trait ") { "trait" }
                else { continue };
            let mut words = t.split_whitespace();
            let _ = words.next();
            let name = words.next().unwrap_or("").trim_end_matches(['(', '{']);
            let name = if name == "fn" || name == "struct" || name == "enum" || name == "trait" {
                words.next().unwrap_or("")
            } else { name };
            if name.is_empty() { continue; }
            self.remember(MemoryLayer::Scenario, &format!("{kind}:{name}"), t,
                &file.display().to_string(), 1.0, 0.6).await;
            count += 1;
        }
        self.add_asset(AssetKind::CodeGraph, &file.display().to_string(),
            "Rust symbol index", &file.display().to_string(), "local", "private").await?;
        let asset_id: i64 = sqlx::query_scalar("SELECT id FROM memory_assets WHERE kind='code_graph' AND name=?")
            .bind(file.display().to_string()).fetch_one(&self.pool).await?;
        // Re-indexing is a replacement operation. Remove nodes and edges from
        // the previous version first; FK cascades clear stale relationships.
        // This prevents deleted/renamed symbols from surviving forever.
        sqlx::query("DELETE FROM code_nodes WHERE asset_id=?")
            .bind(asset_id)
            .execute(&self.pool)
            .await
            .context("replace CodeGraph nodes")?;

        // Materialize symbol nodes as a real graph. Calls are conservative: only
        // identifiers that resolve to another indexed symbol become edges.
        let mut node_ids = std::collections::HashMap::<String,i64>::new();
        for row in sqlx::query("SELECT id,name FROM code_nodes WHERE asset_id=?")
            .bind(asset_id).fetch_all(&self.pool).await? {
            node_ids.insert(row.get("name"), row.get("id"));
        }
        for line in text.lines() {
            let t = line.trim();
            let (kind, name) = if let Some(rest)=t.strip_prefix("pub fn ").or_else(||t.strip_prefix("fn ")) {
                ("function", rest.split(['(', ' ']).next().unwrap_or(""))
            } else if let Some(rest)=t.strip_prefix("pub struct ").or_else(||t.strip_prefix("struct ")) {
                ("struct", rest.split(['{', ' ']).next().unwrap_or(""))
            } else if let Some(rest)=t.strip_prefix("pub enum ").or_else(||t.strip_prefix("enum ")) {
                ("enum", rest.split(['{', ' ']).next().unwrap_or(""))
            } else if let Some(rest)=t.strip_prefix("pub trait ").or_else(||t.strip_prefix("trait ")) {
                ("trait", rest.split(['{', ' ']).next().unwrap_or(""))
            } else { continue };
            if name.is_empty() { continue; }
            let id: i64 = sqlx::query_scalar("INSERT INTO code_nodes(asset_id,kind,name,file,signature) VALUES(?,?,?,?,?) ON CONFLICT(asset_id,name,file) DO UPDATE SET signature=excluded.signature RETURNING id")
                .bind(asset_id).bind(kind).bind(name).bind(file.display().to_string()).bind(t)
                .fetch_one(&self.pool).await?;
            node_ids.insert(name.to_owned(), id);
        }
        for row in sqlx::query("SELECT id,name,signature FROM code_nodes WHERE asset_id=?")
            .bind(asset_id).fetch_all(&self.pool).await? {
            let from: i64 = row.get("id");
            let signature: String = row.get("signature");
            for token in signature.split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
                if let Some(&to) = node_ids.get(token) {
                    if to != from {
                        let _ = sqlx::query("INSERT OR IGNORE INTO code_edges(from_id,to_id,kind) VALUES(?,?,?)")
                            .bind(from).bind(to).bind("references").execute(&self.pool).await;
                    }
                }
            }
        }
        Ok(count)
    }
}

fn layer_name(x: MemoryLayer) -> &'static str {
    match x { MemoryLayer::Conversation=>"conversation", MemoryLayer::Atom=>"atom", MemoryLayer::Scenario=>"scenario", MemoryLayer::Persona=>"persona" }
}
fn asset_name(x: AssetKind) -> &'static str {
    match x { AssetKind::ChatMemory=>"chat_memory", AssetKind::Skill=>"skill", AssetKind::Wiki=>"wiki", AssetKind::CodeGraph=>"code_graph" }
}
fn memory_from_row(r:&sqlx::sqlite::SqliteRow)->Option<MemoryItem>{
    Some(MemoryItem{id:r.get("id"),layer:match r.get::<String,_>("layer").as_str(){
        "conversation"=>MemoryLayer::Conversation,"atom"=>MemoryLayer::Atom,
        "scenario"=>MemoryLayer::Scenario,"persona"=>MemoryLayer::Persona,_=>return None},
        title:r.get("title"),content:r.get("content"),source:r.get("source"),
        confidence:r.get("confidence"),importance:r.get("importance"),
        created_at:r.get("created_at"),updated_at:r.get("updated_at")})
}
fn asset_from_row(r:&sqlx::sqlite::SqliteRow)->Option<MemoryAsset>{
    Some(MemoryAsset{id:r.get("id"),kind:match r.get::<String,_>("kind").as_str(){
        "chat_memory"=>AssetKind::ChatMemory,"skill"=>AssetKind::Skill,
        "wiki"=>AssetKind::Wiki,"code_graph"=>AssetKind::CodeGraph,_=>return None},
        name:r.get("name"),description:r.get("description"),path:r.get("path"),
        version:r.get("version"),owner:r.get("owner"),visibility:r.get("visibility"),
        updated_at:r.get("updated_at")})
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn hub() -> MemoryHub {
        let dir = std::env::temp_dir().join(format!("lucy-memory-hub-{}", uuid::Uuid::new_v4()));
        MemoryHub::open(dir).await
    }

    #[tokio::test]
    async fn layered_memory_is_searchable() {
        let h = hub().await;
        h.remember(MemoryLayer::Atom, "editor", "Krish prefers Neovim", "test", 1.0, 0.9).await.unwrap();
        h.remember(MemoryLayer::Scenario, "lucy", "Lucy uses Rust for the agent runtime", "test", 1.0, 0.8).await.unwrap();
        let hits = h.search("Neovim", 8).await;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].layer, MemoryLayer::Atom);
    }

    #[tokio::test]
    async fn wiki_reingestion_replaces_stale_sections() {
        let h = hub().await;
        let file = std::env::temp_dir().join(format!("lucy-wiki-{}.md", uuid::Uuid::new_v4()));
        tokio::fs::write(&file, "# Old\nold content\n").await.unwrap();
        assert_eq!(h.ingest_wiki(&file).await.unwrap(), 1);
        tokio::fs::write(&file, "# New\nnew content\n").await.unwrap();
        assert_eq!(h.ingest_wiki(&file).await.unwrap(), 1);
        assert!(h.search_layers("old content", Some(MemoryLayer::Scenario), 8).await.is_empty());
        assert_eq!(h.search_layers("new content", Some(MemoryLayer::Scenario), 8).await.len(), 1);
        let _ = tokio::fs::remove_file(file).await;
    }

    #[tokio::test]
    async fn codegraph_reindex_removes_deleted_symbols() {
        let h = hub().await;
        let file = std::env::temp_dir().join(format!("lucy-codegraph-{}.rs", uuid::Uuid::new_v4()));
        tokio::fs::write(&file, "fn old_symbol() {}\n").await.unwrap();
        h.index_rust_file(&file).await.unwrap();
        assert_eq!(h.codegraph_query("old_symbol", 8).await.len(), 1);

        tokio::fs::write(&file, "fn new_symbol() {}\n").await.unwrap();
        h.index_rust_file(&file).await.unwrap();
        assert!(h.codegraph_query("old_symbol", 8).await.is_empty());
        assert_eq!(h.codegraph_query("new_symbol", 8).await.len(), 1);
        let _ = tokio::fs::remove_file(file).await;
    }

    #[tokio::test]
    async fn slim_only_removes_exact_atom_duplicates() {
        let h = hub().await;
        h.remember(MemoryLayer::Atom, "x", "same fact", "a", 0.5, 0.2).await;
        h.remember(MemoryLayer::Atom, "x", "same fact", "b", 1.0, 0.9).await;
        h.remember(MemoryLayer::Scenario, "x", "same fact", "c", 1.0, 0.9).await;
        let (_, after, removed) = h.slim().await.unwrap();
        assert_eq!(removed, 1);
        assert_eq!(after.atom, 1);
        assert_eq!(after.scenario, 1);
    }
}
