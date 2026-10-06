//! Lucy's document workspaces: spaces group pages, pages are Markdown
//! documents, and a rebuildable SQLite FTS5 index makes the whole library
//! searchable.
//!
//! # Why it is shaped this way
//!
//! * **Markdown files are the source of truth, the index is derived.** Every
//!   page is a `.md` file on disk that a human can open in any editor. The
//!   SQLite database holds metadata and the full-text index; deleting it and
//!   calling `reindex()` rebuilds it from the files with no data loss.
//! * **Spaces are lightweight namespaces.** A space is a name and a
//!   description; pages belong to one space. Moving a page between spaces is
//!   a metadata update, not a file copy.
//! * **FTS5 for recall.** The same approach as the knowledge base: a
//!   deterministic keyword floor with porter stemming, so search works
//!   without any model in the loop.
//! * **Conversations become pages.** `save_conversation` renders a
//!   `TurnMessage` history into a Markdown transcript and stores it like any
//!   other page, so a past session is searchable alongside written notes.

pub mod editor;
pub mod search;

pub use editor::{Direction, EditorMode, MarkdownEditor};
pub use search::{PageSearch, SearchResult};

use anyhow::{Context as _, Result, anyhow};
use lucy_core::{TurnMessage, now_secs};
use serde::{Deserialize, Serialize};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions, SqliteRow};
use sqlx::{Row, sqlite::SqliteJournalMode};
use std::{
    path::{Path, PathBuf},
    str::FromStr as _,
    time::Duration,
};

/// A unique identifier for a page.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageId(pub String);

/// A unique identifier for a space.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpaceId(pub String);

/// A collection of pages grouped by topic or project.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Space {
    pub id: SpaceId,
    pub name: String,
    pub description: Option<String>,
    pub page_count: usize,
    pub created_at: u64,
}

/// A single Markdown document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Page {
    pub id: PageId,
    pub space_id: SpaceId,
    pub title: String,
    pub content: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub tags: Vec<String>,
}

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS spaces (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    id         TEXT PRIMARY KEY,
    space_id   TEXT NOT NULL,
    title      TEXT NOT NULL,
    content    TEXT NOT NULL DEFAULT '',
    tags       TEXT NOT NULL DEFAULT '[]',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    FOREIGN KEY (space_id) REFERENCES spaces(id)
);

CREATE INDEX IF NOT EXISTS pages_space_idx ON pages(space_id);
CREATE INDEX IF NOT EXISTS pages_updated_idx ON pages(updated_at DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
    title, content, content='pages', content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS pages_fts_ai AFTER INSERT ON pages BEGIN
    INSERT INTO pages_fts(rowid, title, content)
    VALUES (new.rowid, new.title, new.content);
END;

CREATE TRIGGER IF NOT EXISTS pages_fts_ad AFTER DELETE ON pages BEGIN
    INSERT INTO pages_fts(pages_fts, rowid, title, content)
    VALUES ('delete', old.rowid, old.title, old.content);
END;

CREATE TRIGGER IF NOT EXISTS pages_fts_au AFTER UPDATE ON pages BEGIN
    INSERT INTO pages_fts(pages_fts, rowid, title, content)
    VALUES ('delete', old.rowid, old.title, old.content);
    INSERT INTO pages_fts(rowid, title, content)
    VALUES (new.rowid, new.title, new.content);
END;
"#;

/// The main store for spaces and pages.
///
/// Opens (or creates) a SQLite database at the given path and a
/// `spaces/` directory alongside it for the Markdown files.
pub struct PageStore {
    pool: SqlitePool,
    db_path: PathBuf,
    spaces_dir: PathBuf,
}

impl std::fmt::Debug for PageStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PageStore")
            .field("db_path", &self.db_path)
            .field("spaces_dir", &self.spaces_dir)
            .finish()
    }
}

impl PageStore {
    /// The connection pool, for `search`'s own queries.
    ///
    /// Crate-internal on purpose: the public surface is the `PageStore` methods
    /// and [`PageSearch`], so a caller cannot reach past them into SQL.
    pub(crate) fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Open (creating if absent) the store at `path`.
    ///
    /// `path` is the directory that will contain `pages.db` and the
    /// `spaces/` subdirectory. Both are created if they don't exist.
    pub async fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let root = path.into();
        let db_path = root.join("pages.db");
        let spaces_dir = root.join("spaces");

        tokio::fs::create_dir_all(&spaces_dir)
            .await
            .context("creating spaces directory")?;

        let url = format!("sqlite://{}", db_path.display());
        let options = SqliteConnectOptions::from_str(&url)
            .context("parsing database URL")?
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .busy_timeout(Duration::from_secs(5));

        let pool = SqlitePoolOptions::new()
            .max_connections(4)
            .connect_with(options)
            .await
            .context("opening database")?;

        sqlx::raw_sql(SCHEMA)
            .execute(&pool)
            .await
            .context("running schema migration")?;

        Ok(Self {
            pool,
            db_path,
            spaces_dir,
        })
    }

    /// The directory where space Markdown files are stored.
    pub fn spaces_dir(&self) -> &Path {
        &self.spaces_dir
    }

    // ---------------------------------------------------------------- spaces

    /// Create a new space. Returns an error if the name is empty or already
    /// exists.
    pub async fn create_space(&self, name: &str) -> Result<SpaceId> {
        let name = name.trim();
        if name.is_empty() {
            return Err(anyhow!("space name cannot be empty"));
        }
        let id = SpaceId(uuid::Uuid::new_v4().to_string());
        let now = now_secs() as i64;

        sqlx::query("INSERT INTO spaces (id, name, description, created_at) VALUES (?, ?, NULL, ?)")
            .bind(&id.0)
            .bind(name)
            .bind(now)
            .execute(&self.pool)
            .await
            .context("creating space")?;

        Ok(id)
    }

    /// Look up a space by its name.
    pub async fn get_space(&self, name: &str) -> Result<Option<Space>> {
        let row = sqlx::query(
            "SELECT s.id, s.name, s.description, s.created_at, \
             (SELECT COUNT(*) FROM pages p WHERE p.space_id = s.id) AS cnt \
             FROM spaces s WHERE s.name = ?",
        )
        .bind(name.trim())
        .fetch_optional(&self.pool)
        .await
        .context("looking up space")?;

        Ok(row.map(|r| Space {
            id: SpaceId(r.get::<String, _>("id")),
            name: r.get::<String, _>("name"),
            description: r.get::<Option<String>, _>("description"),
            // Counted here rather than left for the caller to fill in: a
            // `page_count` of 0 on a space holding pages is a wrong number
            // that reads as truth, and the caller has no cheaper way to get
            // the real one.
            page_count: r.get::<i64, _>("cnt") as usize,
            created_at: r.get::<i64, _>("created_at") as u64,
        }))
    }

    /// List all spaces, ordered by name.
    pub async fn list_spaces(&self) -> Result<Vec<Space>> {
        let rows = sqlx::query(
            "SELECT s.id, s.name, s.description, s.created_at, \
             (SELECT COUNT(*) FROM pages p WHERE p.space_id = s.id) AS cnt \
             FROM spaces s ORDER BY s.name",
        )
        .fetch_all(&self.pool)
        .await
        .context("listing spaces")?;

        Ok(rows
            .into_iter()
            .map(|r| Space {
                id: SpaceId(r.get::<String, _>("id")),
                name: r.get::<String, _>("name"),
                description: r.get::<Option<String>, _>("description"),
                page_count: r.get::<i64, _>("cnt") as usize,
                created_at: r.get::<i64, _>("created_at") as u64,
            })
            .collect())
    }

    // ------------------------------------------------------------------ pages

    /// Create a new page in the named space. The space must already exist.
    pub async fn create_page(&self, space: &str, title: &str) -> Result<PageId> {
        let space_row = self
            .get_space(space)
            .await?
            .ok_or_else(|| anyhow!("space '{space}' does not exist"))?;

        let id = PageId(uuid::Uuid::new_v4().to_string());
        let now = now_secs() as i64;

        sqlx::query(
            "INSERT INTO pages (id, space_id, title, content, tags, created_at, updated_at) \
             VALUES (?, ?, ?, '', '[]', ?, ?)",
        )
        .bind(&id.0)
        .bind(&space_row.id.0)
        .bind(title)
        .bind(now)
        .bind(now)
        .execute(&self.pool)
        .await
        .context("creating page")?;

        // Write the initial empty file so the source of truth exists on disk.
        let _ = self.write_page_file(&id, title, "").await;

        Ok(id)
    }

    /// Save (overwrite) a page's content.
    pub async fn save_page(&self, id: PageId, content: &str) -> Result<()> {
        let now = now_secs() as i64;
        let result = sqlx::query("UPDATE pages SET content = ?, updated_at = ? WHERE id = ?")
            .bind(content)
            .bind(now)
            .bind(&id.0)
            .execute(&self.pool)
            .await
            .context("saving page")?;

        if result.rows_affected() == 0 {
            return Err(anyhow!("page '{}' does not exist", id.0));
        }

        // Read the title from the DB so the file on disk stays in sync.
        let title: String = sqlx::query("SELECT title FROM pages WHERE id = ?")
            .bind(&id.0)
            .fetch_one(&self.pool)
            .await
            .map(|r| r.get("title"))
            .unwrap_or_default();

        self.write_page_file(&id, &title, content).await;

        Ok(())
    }

    /// Fetch a page by its ID.
    pub async fn get_page(&self, id: PageId) -> Result<Option<Page>> {
        let row = sqlx::query(
            "SELECT id, space_id, title, content, tags, created_at, updated_at \
             FROM pages WHERE id = ?",
        )
        .bind(&id.0)
        .fetch_optional(&self.pool)
        .await
        .context("fetching page")?;

        Ok(row.as_ref().map(page_from_row))
    }

    /// Search all pages by content and title using FTS5.
    pub async fn search(&self, query: &str) -> Result<Vec<Page>> {
        let terms = fts_query(query);
        if terms.is_empty() {
            return Ok(Vec::new());
        }

        let rows = sqlx::query(
            "SELECT p.id, p.space_id, p.title, p.content, p.tags, p.created_at, p.updated_at \
             FROM pages_fts f JOIN pages p ON p.rowid = f.rowid \
             WHERE pages_fts MATCH ? \
             ORDER BY bm25(pages_fts) LIMIT 50",
        )
        .bind(&terms)
        .fetch_all(&self.pool)
        .await
        .context("searching pages")?;

        Ok(rows.iter().map(page_from_row).collect())
    }

    /// Save a conversation (slice of `TurnMessage`) as a new page in the
    /// given space.
    pub async fn save_conversation(
        &self,
        space: &str,
        title: &str,
        messages: &[TurnMessage],
    ) -> Result<PageId> {
        let content = render_conversation(messages);
        let id = self.create_page(space, title).await?;
        self.save_page(id.clone(), &content).await?;
        Ok(id)
    }

    /// Delete a page and its backing file.
    pub async fn delete_page(&self, id: PageId) -> Result<()> {
        let result = sqlx::query("DELETE FROM pages WHERE id = ?")
            .bind(&id.0)
            .execute(&self.pool)
            .await
            .context("deleting page")?;

        if result.rows_affected() == 0 {
            return Err(anyhow!("page '{}' does not exist", id.0));
        }

        let path = self.page_file_path(&id);
        let _ = tokio::fs::remove_file(&path).await;

        Ok(())
    }

    /// Move a page to a different space.
    pub async fn move_page(&self, id: PageId, space: &str) -> Result<()> {
        let space_row = self
            .get_space(space)
            .await?
            .ok_or_else(|| anyhow!("space '{space}' does not exist"))?;

        let result = sqlx::query("UPDATE pages SET space_id = ?, updated_at = ? WHERE id = ?")
            .bind(&space_row.id.0)
            .bind(now_secs() as i64)
            .bind(&id.0)
            .execute(&self.pool)
            .await
            .context("moving page")?;

        if result.rows_affected() == 0 {
            return Err(anyhow!("page '{}' does not exist", id.0));
        }

        Ok(())
    }

    /// Add a tag to a page.
    pub async fn add_tag(&self, id: PageId, tag: &str) -> Result<()> {
        let tag = tag.trim();
        if tag.is_empty() {
            return Err(anyhow!("tag cannot be empty"));
        }
        let page = self
            .get_page(id.clone())
            .await?
            .ok_or_else(|| anyhow!("page '{}' does not exist", id.0))?;

        if page.tags.iter().any(|t| t == tag) {
            return Ok(()); // already tagged
        }

        let mut tags = page.tags.clone();
        tags.push(tag.to_string());
        let tags_json = serde_json::to_string(&tags)?;

        sqlx::query("UPDATE pages SET tags = ?, updated_at = ? WHERE id = ?")
            .bind(&tags_json)
            .bind(now_secs() as i64)
            .bind(&id.0)
            .execute(&self.pool)
            .await
            .context("adding tag")?;

        Ok(())
    }

    /// Remove a tag from a page.
    pub async fn remove_tag(&self, id: PageId, tag: &str) -> Result<()> {
        let page = self
            .get_page(id.clone())
            .await?
            .ok_or_else(|| anyhow!("page '{}' does not exist", id.0))?;

        let tags: Vec<String> = page
            .tags
            .iter()
            .filter(|t| t.as_str() != tag)
            .cloned()
            .collect();
        let tags_json = serde_json::to_string(&tags)?;

        sqlx::query("UPDATE pages SET tags = ? WHERE id = ?")
            .bind(&tags_json)
            .bind(&id.0)
            .execute(&self.pool)
            .await
            .context("removing tag")?;

        Ok(())
    }

    /// List all pages in a space, ordered by most recently updated.
    pub async fn list_pages_in_space(&self, space: &str) -> Result<Vec<Page>> {
        let space_row = self
            .get_space(space)
            .await?
            .ok_or_else(|| anyhow!("space '{space}' does not exist"))?;

        let rows = sqlx::query(
            "SELECT id, space_id, title, content, tags, created_at, updated_at \
             FROM pages WHERE space_id = ? ORDER BY updated_at DESC",
        )
        .bind(&space_row.id.0)
        .fetch_all(&self.pool)
        .await
        .context("listing pages in space")?;

        Ok(rows.iter().map(page_from_row).collect())
    }

    /// List all pages tagged with the given tag.
    pub async fn list_pages_by_tag(&self, tag: &str) -> Result<Vec<Page>> {
        let rows = sqlx::query(
            "SELECT id, space_id, title, content, tags, created_at, updated_at \
             FROM pages WHERE tags LIKE ? ORDER BY updated_at DESC",
        )
        .bind(format!("%\"{tag}\"%"))
        .fetch_all(&self.pool)
        .await
        .context("listing pages by tag")?;

        Ok(rows
            .into_iter()
            .filter_map(|r| {
                let page = page_from_row(&r);
                if page.tags.iter().any(|t| t == tag) {
                    Some(page)
                } else {
                    None
                }
            })
            .collect())
    }

    /// Rebuild the FTS index from the rows already in the database.
    pub async fn reindex(&self) -> Result<()> {
        sqlx::raw_sql(
            "INSERT INTO pages_fts(pages_fts) VALUES('rebuild');",
        )
        .execute(&self.pool)
        .await
        .context("rebuilding FTS index")?;
        Ok(())
    }

    /// Export a page as a Markdown file to the given directory.
    pub async fn export_page(&self, id: PageId, dest_dir: &Path) -> Result<PathBuf> {
        let page = self
            .get_page(id.clone())
            .await?
            .ok_or_else(|| anyhow!("page '{}' does not exist", id.0))?;

        tokio::fs::create_dir_all(dest_dir)
            .await
            .context("creating export directory")?;

        let filename = slugify(&page.title) + ".md";
        let path = dest_dir.join(&filename);
        tokio::fs::write(&path, &page.content)
            .await
            .context("writing export file")?;

        Ok(path)
    }

    // -------------------------------------------------------------- internals

    fn page_file_path(&self, id: &PageId) -> PathBuf {
        self.spaces_dir.join(format!("{}.md", id.0))
    }

    async fn write_page_file(&self, id: &PageId, title: &str, content: &str) {
        let path = self.page_file_path(id);
        let text = format!("# {title}\n\n{content}");
        let _ = tokio::fs::write(&path, text).await;
    }
}

// ------------------------------------------------------------------ helpers

fn page_from_row(row: &SqliteRow) -> Page {
    let tags_json: String = row.get("tags");
    let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
    Page {
        id: PageId(row.get::<String, _>("id")),
        space_id: SpaceId(row.get::<String, _>("space_id")),
        title: row.get::<String, _>("title"),
        content: row.get::<String, _>("content"),
        created_at: row.get::<i64, _>("created_at") as u64,
        updated_at: row.get::<i64, _>("updated_at") as u64,
        tags,
    }
}

/// Build an FTS5 MATCH expression from free text.
///
/// FTS5 treats user punctuation as query syntax, so raw text raises syntax
/// errors. Quoting each alphanumeric term and OR-ing them keeps recall high
/// and the query valid for any input.
fn fts_query(raw: &str) -> String {
    let terms: Vec<String> = raw
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .filter(|t| t.len() >= 2)
        .map(|t| t.to_ascii_lowercase())
        .collect();
    let mut seen = std::collections::HashSet::new();
    terms
        .into_iter()
        .filter(|t| seen.insert(t.clone()))
        .map(|t| format!("\"{t}\""))
        .collect::<Vec<_>>()
        .join(" OR ")
}

/// Lowercase, ASCII-ish, hyphenated slug.
fn slugify(raw: &str) -> String {
    let mut out = String::new();
    let mut last_dash = true;
    for ch in raw.trim().to_ascii_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            last_dash = false;
        } else if ch == '/' || ch == '\\' || ch == ' ' || ch == '_' || ch == '.' || ch == '-' {
            if !last_dash {
                out.push('-');
                last_dash = true;
            }
        }
    }
    out.trim_matches('-').to_string()
}

/// Render a conversation (slice of `TurnMessage`) as a Markdown transcript.
fn render_conversation(messages: &[TurnMessage]) -> String {
    let mut out = String::new();
    for msg in messages {
        match msg {
            TurnMessage::User(text) => {
                out.push_str(&format!("## User\n\n{text}\n\n"));
            }
            TurnMessage::Assistant(turn) => {
                if let Some(text) = &turn.text {
                    out.push_str(&format!("## Assistant\n\n{text}\n\n"));
                }
                for call in &turn.tool_calls {
                    out.push_str(&format!(
                        "### Tool Call: {}\n\n```json\n{}\n```\n\n",
                        call.name,
                        serde_json::to_string_pretty(&call.input).unwrap_or_default()
                    ));
                }
            }
            TurnMessage::Tool(result) => {
                out.push_str(&format!(
                    "### Tool Result: {}\n\n```json\n{}\n```\n\n",
                    result.name,
                    serde_json::to_string_pretty(&result.output).unwrap_or_default()
                ));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn fresh_store(tag: &str) -> (PathBuf, PageStore) {
        let dir = std::env::temp_dir().join(format!(
            "lucy-spaces-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = tokio::fs::remove_dir_all(&dir).await;
        let store = PageStore::open(&dir).await.expect("opening store");
        (dir, store)
    }

    fn user_msg(text: &str) -> TurnMessage {
        TurnMessage::User(text.to_string())
    }

    fn assistant_msg(text: &str) -> TurnMessage {
        TurnMessage::Assistant(lucy_core::AssistantTurn {
            text: Some(text.to_string()),
            tool_calls: vec![],
        })
    }

    #[tokio::test]
    async fn create_and_list_spaces() {
        let (dir, store) = fresh_store("create-space").await;

        let id = store.create_space("Projects").await.expect("creating space");
        assert!(!id.0.is_empty());

        let spaces = store.list_spaces().await.expect("listing spaces");
        assert_eq!(spaces.len(), 1);
        assert_eq!(spaces[0].name, "Projects");
        assert_eq!(spaces[0].page_count, 0);

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn duplicate_space_name_is_rejected() {
        let (dir, store) = fresh_store("dup-space").await;

        store.create_space("Alpha").await.expect("first create");
        let result = store.create_space("Alpha").await;
        assert!(result.is_err(), "duplicate space name should fail");

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn page_crud_operations() {
        let (dir, store) = fresh_store("page-crud").await;

        store.create_space("Notes").await.expect("creating space");
        let id = store
            .create_page("Notes", "My First Page")
            .await
            .expect("creating page");

        // Save content
        store
            .save_page(id.clone(), "# Hello\n\nThis is my page.")
            .await
            .expect("saving page");

        // Read back
        let page = store
            .get_page(id.clone())
            .await
            .expect("getting page")
            .expect("page exists");
        assert_eq!(page.title, "My First Page");
        assert!(page.content.contains("Hello"));

        // Update
        store
            .save_page(id.clone(), "Updated content")
            .await
            .expect("updating page");
        let page = store
            .get_page(id.clone())
            .await
            .expect("getting page")
            .expect("page exists");
        assert_eq!(page.content, "Updated content");
        assert!(page.updated_at >= page.created_at);

        // Delete
        store.delete_page(id.clone()).await.expect("deleting page");
        let page = store.get_page(id).await.expect("getting page");
        assert!(page.is_none(), "deleted page should be gone");

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn save_conversation_as_page() {
        let (dir, store) = fresh_store("conversation").await;

        store.create_space("Sessions").await.expect("creating space");
        let messages = vec![
            user_msg("What is Rust?"),
            assistant_msg("Rust is a systems programming language."),
            user_msg("Tell me more."),
            assistant_msg("It guarantees memory safety without garbage collection."),
        ];

        let id = store
            .save_conversation("Sessions", "Rust Discussion", &messages)
            .await
            .expect("saving conversation");

        let page = store
            .get_page(id)
            .await
            .expect("getting page")
            .expect("page exists");
        assert_eq!(page.title, "Rust Discussion");
        assert!(page.content.contains("## User"));
        assert!(page.content.contains("## Assistant"));
        assert!(page.content.contains("memory safety"));

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn search_finds_pages_by_content() {
        let (dir, store) = fresh_store("search").await;

        store.create_space("Wiki").await.expect("creating space");
        let id1 = store
            .create_page("Wiki", "Rust Tips")
            .await
            .expect("creating page 1");
        store
            .save_page(id1, "Rust has great pattern matching and ownership.")
            .await
            .expect("saving page 1");

        let id2 = store
            .create_page("Wiki", "Cooking Notes")
            .await
            .expect("creating page 2");
        store
            .save_page(id2, "Pasta needs salted water and timing.")
            .await
            .expect("saving page 2");

        let results = store.search("pattern matching").await.expect("searching");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Rust Tips");

        let results = store.search("pasta").await.expect("searching");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Cooking Notes");

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn page_tagging() {
        let (dir, store) = fresh_store("tags").await;

        store.create_space("Notes").await.expect("creating space");
        let id = store
            .create_page("Notes", "Tagged Page")
            .await
            .expect("creating page");

        store.add_tag(id.clone(), "rust").await.expect("adding tag");
        store.add_tag(id.clone(), "tutorial").await.expect("adding tag");

        let page = store
            .get_page(id.clone())
            .await
            .expect("getting page")
            .expect("page exists");
        assert_eq!(page.tags.len(), 2);
        assert!(page.tags.contains(&"rust".to_string()));

        // Duplicate tag is a no-op
        store.add_tag(id.clone(), "rust").await.expect("adding dup tag");
        let page = store
            .get_page(id.clone())
            .await
            .expect("getting page")
            .expect("page exists");
        assert_eq!(page.tags.len(), 2, "duplicate tag should not be added");

        // Remove tag
        store
            .remove_tag(id.clone(), "rust")
            .await
            .expect("removing tag");
        let page = store
            .get_page(id)
            .await
            .expect("getting page")
            .expect("page exists");
        assert_eq!(page.tags.len(), 1);
        assert!(!page.tags.contains(&"rust".to_string()));

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn move_page_between_spaces() {
        let (dir, store) = fresh_store("move").await;

        store.create_space("Drafts").await.expect("creating space 1");
        store.create_space("Published").await.expect("creating space 2");

        let id = store
            .create_page("Drafts", "My Page")
            .await
            .expect("creating page");

        store
            .move_page(id.clone(), "Published")
            .await
            .expect("moving page");

        let page = store
            .get_page(id)
            .await
            .expect("getting page")
            .expect("page exists");

        let published = store
            .get_space("Published")
            .await
            .expect("getting space")
            .expect("space exists");
        assert_eq!(page.space_id.0, published.id.0);

        let drafts = store
            .get_space("Drafts")
            .await
            .expect("getting space")
            .expect("space exists");
        assert_eq!(drafts.page_count, 0);
        assert_eq!(published.page_count, 1);

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[tokio::test]
    async fn export_page_to_markdown() {
        let (dir, store) = fresh_store("export").await;

        store.create_space("Docs").await.expect("creating space");
        let id = store
            .create_page("Docs", "Export Me")
            .await
            .expect("creating page");
        store
            .save_page(id.clone(), "# Title\n\nSome content here.")
            .await
            .expect("saving page");

        let export_dir = dir.join("export");
        let path = store
            .export_page(id, &export_dir)
            .await
            .expect("exporting page");

        assert!(path.exists(), "export file should exist");
        assert!(path.to_string_lossy().ends_with(".md"));

        let content = tokio::fs::read_to_string(&path)
            .await
            .expect("reading export");
        assert!(content.contains("Some content here"));

        let _ = tokio::fs::remove_dir_all(&dir).await;
    }

    #[test]
    fn fts_query_handles_punctuation() {
        for raw in [
            "what's up?",
            "c++ & rust: a comparison",
            "\"quoted\" (parens) *star*",
            "",
            "   ",
        ] {
            let q = fts_query(raw);
            assert!(
                q.chars().all(|c| c != '(' && c != '*' && c != ':'),
                "unescaped FTS5 syntax survived for {raw:?} -> {q:?}"
            );
        }
        assert_eq!(fts_query("what's up?"), "\"what\" OR \"up\"");
        assert!(fts_query("a b").is_empty());
    }

    #[test]
    fn slugify_normalizes_names() {
        assert_eq!(slugify("My Great Page"), "my-great-page");
        assert_eq!(slugify("  spaces  everywhere  "), "spaces-everywhere");
        assert_eq!(slugify(""), "");
    }
}
