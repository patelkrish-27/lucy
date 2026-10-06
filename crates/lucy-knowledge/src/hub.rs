//! Memory Hub primitives for layered chat memory and reusable knowledge assets.

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
CREATE TABLE IF NOT EXISTS memory_assets (
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
            .unwrap_or_else(|_| SqlitePoolOptions::new().max_connections(1)
                .connect_lazy("sqlite::memory:").expect("sqlite"));
        let _ = sqlx::raw_sql(SCHEMA).execute(&pool).await;
        Self { pool, root }
    }

    pub fn root(&self) -> &Path { &self.root }

    pub async fn remember(&self, layer: MemoryLayer, title: &str, content: &str,
        source: &str, confidence: f32, importance: f32) -> Option<i64> {
        let content = content.trim();
        if content.is_empty() || content.len() > 12_000 { return None; }
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query("INSERT INTO memory_items(layer,title,content,source,confidence,importance,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)")
            .bind(layer_name(layer)).bind(title.trim()).bind(content).bind(source)
            .bind(confidence.clamp(0.0,1.0)).bind(importance.clamp(0.0,1.0))
            .bind(&now).bind(&now).execute(&self.pool).await.ok()
            .map(|r| r.last_insert_rowid())
    }

    pub async fn search(&self, query: &str, limit: usize) -> Vec<MemoryItem> {
        let terms = query.split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .filter(|x| x.len() >= 2).map(|x| format!(""{}"", x.to_ascii_lowercase()))
            .collect::<Vec<_>>();
        if terms.is_empty() { return Vec::new(); }
        let q = terms.join(" OR ");
        let rows = sqlx::query("SELECT m.id,m.layer,m.title,m.content,m.source,m.confidence,m.importance,m.created_at,m.updated_at FROM memory_fts f JOIN memory_items m ON m.id=f.rowid WHERE memory_fts MATCH ? ORDER BY bm25(memory_fts) LIMIT ?")
            .bind(q).bind(limit.clamp(1,32) as i64).fetch_all(&self.pool).await.unwrap_or_default();
        rows.into_iter().filter_map(memory_from_row).collect()
    }

    pub async fn add_asset(&self, kind: AssetKind, name: &str, description: &str,
        path: &str, owner: &str, visibility: &str) -> Result<i64> {
        let now = chrono::Utc::now().to_rfc3339();
        let result = sqlx::query("INSERT INTO memory_assets(kind,name,description,path,version,owner,visibility,updated_at) VALUES(?,?,?,?,1,?,?,?) ON CONFLICT(kind,name) DO UPDATE SET description=excluded.description,path=excluded.path,version=memory_assets.version+1,owner=excluded.owner,visibility=excluded.visibility,updated_at=excluded.updated_at")
            .bind(asset_name(kind)).bind(name).bind(description).bind(path).bind(owner).bind(visibility).bind(&now)
            .execute(&self.pool).await.context("register memory asset")?;
        Ok(result.last_insert_rowid())
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

    pub async fn ingest_wiki(&self, file: impl AsRef<Path>) -> Result<usize> {
        let file = file.as_ref();
        let text = tokio::fs::read_to_string(file).await.context("read wiki source")?;
        let name = file.file_stem().and_then(|x| x.to_str()).unwrap_or("wiki");
        self.add_asset(AssetKind::Wiki, name, "Local Markdown knowledge corpus",
            &file.display().to_string(), "local", "private").await?;
        let mut count = 0usize;
        let mut title = name.to_owned();
        let mut body = String::new();
        for line in text.lines() {
            if let Some(h) = line.strip_prefix("# ").or_else(|| line.strip_prefix("## ")) {
                if !body.trim().is_empty() {
                    self.remember(MemoryLayer::Scenario, &title, &body,
                        &file.display().to_string(), 1.0, 0.8).await;
                    count += 1;
                }
                title = h.trim().to_owned();
                body.clear();
            } else { body.push_str(line); body.push('
'); }
        }
        if !body.trim().is_empty() {
            self.remember(MemoryLayer::Scenario, &title, &body,
                &file.display().to_string(), 1.0, 0.8).await;
            count += 1;
        }
        Ok(count)
    }

    /// Lightweight Rust CodeGraph indexing. It intentionally stores symbols as
    /// searchable graph seeds; language-specific AST adapters can enrich this later.
    pub async fn index_rust_file(&self, file: impl AsRef<Path>) -> Result<usize> {
        let file = file.as_ref();
        let text = tokio::fs::read_to_string(file).await.context("read Rust source")?;
        let mut count = 0;
        for line in text.lines() {
            let t = line.trim();
            let kind = if t.starts_with("pub fn ") || t.starts_with("fn ") { "function" }
                else if t.starts_with("pub struct ") || t.starts_with("struct ") { "struct" }
                else if t.starts_with("pub enum ") || t.starts_with("enum ") { "enum" }
                else if t.starts_with("pub trait ") || t.starts_with("trait ") { "trait" }
                else { continue };
            let name = t.split_whitespace().nth(2).unwrap_or("").trim_end_matches(['(', '{']);
            if name.is_empty() { continue; }
            self.remember(MemoryLayer::Scenario, name, t, &file.display().to_string(), 1.0, 0.6).await;
            count += 1;
        }
        self.add_asset(AssetKind::CodeGraph, &file.display().to_string(),
            "Rust symbol index", &file.display().to_string(), "local", "private").await?;
        Ok(count)
    }

    pub async fn consolidate(&self) -> Result<usize> {
        let duplicates = sqlx::query("SELECT title, COUNT(*) n FROM memory_items WHERE layer='atom' GROUP BY title HAVING n>1")
            .fetch_all(&self.pool).await?;
        let mut removed = 0;
        for row in duplicates {
            let title: String = row.get("title");
            let ids = sqlx::query("SELECT id FROM memory_items WHERE layer='atom' AND title=? ORDER BY confidence DESC,updated_at DESC")
                .bind(&title).fetch_all(&self.pool).await?;
            for row in ids.into_iter().skip(1) {
                let id: i64 = row.get("id");
                sqlx::query("DELETE FROM memory_items WHERE id=?").bind(id).execute(&self.pool).await?;
                removed += 1;
            }
        }
        Ok(removed)
    }
}

fn layer_name(x: MemoryLayer) -> &'static str {
    match x { MemoryLayer::Conversation=>"conversation", MemoryLayer::Atom=>"atom",
        MemoryLayer::Scenario=>"scenario", MemoryLayer::Persona=>"persona" }
}
fn asset_name(x: AssetKind) -> &'static str {
    match x { AssetKind::ChatMemory=>"chat_memory", AssetKind::Skill=>"skill",
        AssetKind::Wiki=>"wiki", AssetKind::CodeGraph=>"code_graph" }
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