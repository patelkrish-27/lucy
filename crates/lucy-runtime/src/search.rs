//! Cross-session search: a deterministic FTS5 index over the event log.
//!
//! Every turn Lucy persists (`save_user_message` / `save_assistant_text`) is
//! also indexed here, so a past conversation is findable by its words from
//! any later session. The index is derived state: it can be deleted and
//! rebuilt from the ADK event log, and recall is FTS5/BM25 only — the same
//! keyword-deterministic floor the knowledge base relies on, with no model
//! in the retrieval path.
//!
//! Nothing here branches on task words. The index stores whatever text flowed
//! through a session; ranking comes from FTS5, not from rules about what the
//! user asked.

use anyhow::{Context as _, Result};
use lucy_core::TurnMessage;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use sqlx::{Row, sqlite::SqliteJournalMode};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    str::FromStr as _,
    time::Duration,
};

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS session_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT    NOT NULL,
    author     TEXT    NOT NULL DEFAULT '',
    text       TEXT    NOT NULL,
    created_at INTEGER NOT NULL DEFAULT 0,
    -- Re-indexing the same turn (a fork replays events, a resume re-saves)
    -- must not double-count it, so the natural key is unique and ignored.
    UNIQUE (session_id, author, text, created_at)
);
CREATE INDEX IF NOT EXISTS session_events_session ON session_events(session_id);
CREATE VIRTUAL TABLE IF NOT EXISTS session_events_fts USING fts5(
    text,
    content='session_events', content_rowid='id',
    tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS session_events_fts_ai AFTER INSERT ON session_events BEGIN
    INSERT INTO session_events_fts(rowid, text)
    VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS session_events_fts_ad AFTER DELETE ON session_events BEGIN
    INSERT INTO session_events_fts(session_events_fts, rowid, text)
    VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS session_events_fts_au AFTER UPDATE ON session_events BEGIN
    INSERT INTO session_events_fts(session_events_fts, rowid, text)
    VALUES ('delete', old.id, old.text);
    INSERT INTO session_events_fts(rowid, text)
    VALUES (new.id, new.text);
END;
"#;

/// One indexable piece of conversation. This is the currency both
/// [`SessionSearch::index_event`] and [`crate::user_model::UserModel::update`]
/// consume; constructors map the ADK event type and the runtime's turn type
/// into it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionEvent {
    pub session_id: String,
    pub author: String,
    pub text: String,
    pub created_at: u64,
}

impl SessionEvent {
    pub fn new(
        session_id: impl Into<String>,
        author: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            author: author.into(),
            text: text.into(),
            created_at: crate::sessions::now(),
        }
    }

    pub fn with_created_at(mut self, created_at: u64) -> Self {
        self.created_at = created_at;
        self
    }

    /// Map an ADK event into the index currency. `None` when the event carries
    /// no text at all (a pure state-delta event is not findable content).
    pub fn from_adk(session_id: &str, event: &adk_core::Event) -> Option<Self> {
        let content = event.llm_response.content.as_ref()?;
        let text = content
            .parts
            .iter()
            .filter_map(adk_core::Part::text)
            .collect::<Vec<_>>()
            .join(" ");
        let text = text.trim();
        if text.is_empty() {
            return None;
        }
        Some(Self {
            session_id: session_id.to_owned(),
            author: if event.author.is_empty() {
                "unknown".to_owned()
            } else {
                event.author.clone()
            },
            text: text.to_owned(),
            created_at: event.timestamp.timestamp().max(0) as u64,
        })
    }

    pub fn from_turn(session_id: &str, message: &TurnMessage) -> Self {
        let (author, text) = match message {
            TurnMessage::User(t) => ("user", t.clone()),
            TurnMessage::Assistant(t) => ("lucy", t.text.clone().unwrap_or_default()),
            TurnMessage::Tool(t) => ("lucy", format!("{}: {}", t.name, t.output)),
        };
        Self::new(session_id, author, text)
    }
}

/// One session's place in the result set: how much of it matched and the
/// strongest hit, so the caller can decide which session to reopen.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionSummary {
    pub session_id: String,
    /// Matching events in this session.
    pub hits: usize,
    /// Best (lowest) BM25 rank across the session's hits. Lower ranks higher.
    pub score: f64,
    /// A short excerpt of the strongest matching event.
    pub snippet: String,
    /// Most recent matching event, seconds since the epoch.
    pub last_seen: u64,
}

/// FTS5 search over every indexed session event.
pub struct SessionSearch {
    pool: SqlitePool,
    #[allow(dead_code)]
    path: PathBuf,
}

impl std::fmt::Debug for SessionSearch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionSearch")
            .field("path", &self.path)
            .finish()
    }
}

impl SessionSearch {
    /// Open (creating if absent) the index database at `db`. `":memory:"`
    /// gives a throwaway index with the same shape — used as the fallback
    /// when the state directory is not writable.
    pub async fn new(db: impl AsRef<Path>) -> Result<Self> {
        let path = db.as_ref().to_path_buf();
        let (url, max_connections) = if path == Path::new(":memory:") {
            // A pooled in-memory database is one database per connection;
            // a single connection keeps the schema and the rows together.
            ("sqlite::memory:".to_owned(), 1)
        } else {
            if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                tokio::fs::create_dir_all(parent)
                    .await
                    .context("creating session search directory")?;
            }
            (format!("sqlite://{}", path.display()), 4)
        };
        let options = SqliteConnectOptions::from_str(&url)
            .with_context(|| format!("parsing session search db url {url}"))?
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .busy_timeout(Duration::from_secs(5));
        let pool = SqlitePoolOptions::new()
            .max_connections(max_connections)
            .connect_with(options)
            .await
            .context("opening session search index")?;
        sqlx::raw_sql(SCHEMA)
            .execute(&pool)
            .await
            .context("migrating session search index")?;
        Ok(Self { pool, path })
    }

    /// Add one event to the index. Duplicate events (same session, author,
    /// text and second) are ignored rather than double-counted.
    pub async fn index_event(&self, event: &SessionEvent) -> Result<()> {
        if event.text.trim().is_empty() {
            return Ok(());
        }
        sqlx::query(
            "INSERT OR IGNORE INTO session_events (session_id, author, text, created_at) \
             VALUES (?, ?, ?, ?)",
        )
        .bind(&event.session_id)
        .bind(&event.author)
        .bind(&event.text)
        .bind(event.created_at as i64)
        .execute(&self.pool)
        .await
        .context("indexing session event")?;
        Ok(())
    }

    /// BM25 search over every session's events, folded to one summary per
    /// session and ranked by the session's strongest hit.
    pub async fn search(&self, query: &str, limit: usize) -> Vec<SessionSummary> {
        let terms = lucy_knowledge::store::fts_query(query);
        if terms.is_empty() {
            return Vec::new();
        }
        // Over-fetch rows so a session is represented even when several of
        // its events rank below other sessions'.
        let row_cap = (limit.saturating_mul(16)).clamp(16, 512) as i64;
        let rows = sqlx::query(
            "SELECT c.session_id AS session_id, c.text AS text, c.created_at AS created_at, \
                    bm25(session_events_fts) AS score \
             FROM session_events_fts \
             JOIN session_events c ON c.id = session_events_fts.rowid \
             WHERE session_events_fts MATCH ? \
             ORDER BY score LIMIT ?",
        )
        .bind(&terms)
        .bind(row_cap)
        .fetch_all(&self.pool)
        .await
        .unwrap_or_default();
        let mut by_session: HashMap<String, SessionSummary> = HashMap::new();
        for row in rows {
            let session_id: String = row.get("session_id");
            let text: String = row.get("text");
            let created_at: i64 = row.get("created_at");
            let score: f64 = row.get("score");
            by_session
                .entry(session_id.clone())
                .and_modify(|s| {
                    s.hits += 1;
                    s.last_seen = s.last_seen.max(created_at.max(0) as u64);
                    if score < s.score {
                        s.score = score;
                        s.snippet = excerpt(&text);
                    }
                })
                .or_insert(SessionSummary {
                    session_id,
                    hits: 1,
                    score,
                    snippet: excerpt(&text),
                    last_seen: created_at.max(0) as u64,
                });
        }
        let mut out: Vec<SessionSummary> = by_session.into_values().collect();
        out.sort_by(|a, b| a.score.total_cmp(&b.score));
        out.truncate(limit.max(1));
        out
    }

    /// A short human-readable digest of a result set, for a `/search` reply
    /// or a planner context line.
    pub fn summarize_results(results: &[SessionSummary]) -> String {
        if results.is_empty() {
            return "No past sessions matched.".to_owned();
        }
        let mut out = format!(
            "{} past session{} matched:",
            results.len(),
            if results.len() == 1 { "" } else { "s" }
        );
        for (i, r) in results.iter().enumerate() {
            out.push_str(&format!(
                "\n{}. {} — {} hit{}; latest hit: \"{}\"",
                i + 1,
                r.session_id,
                r.hits,
                if r.hits == 1 { "" } else { "s" },
                r.snippet
            ));
        }
        out
    }
}

fn excerpt(text: &str) -> String {
    let collapsed: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    const MAX: usize = 140;
    if collapsed.chars().count() <= MAX {
        collapsed
    } else {
        let head: String = collapsed.chars().take(MAX).collect();
        format!("{head}…")
    }
}
