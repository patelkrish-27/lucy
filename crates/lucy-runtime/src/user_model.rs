//! User model: durable preferences, facts, decisions and instructions,
//! extracted deterministically from the event stream.
//!
//! Extraction is a pure function of the events ([`UserModel::extract`]) and
//! the stored profile accumulates with [`UserModel::update`]. No task routing
//! and no site knowledge lives here: the markers describe *who the user is*
//! ("I prefer", "I use", "from now on"), and what we cannot parse we simply
//! do not store. Lucy reads the profile through [`UserModel::context_for`],
//! which ranks its facts against the current query by plain keyword overlap.

use crate::search::SessionEvent;
use anyhow::{Context as _, Result};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use sqlx::{Row, sqlite::SqliteJournalMode};
use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    str::FromStr as _,
    time::Duration,
};

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS user_facts (
    kind       TEXT    NOT NULL,
    text       TEXT    NOT NULL,
    seen       INTEGER NOT NULL DEFAULT 1,
    updated_at INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (kind, text)
);
"#;

/// Durable things the user has said about themselves, bucketed by kind.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UserProfile {
    pub preferences: Vec<String>,
    pub facts: Vec<String>,
    pub decisions: Vec<String>,
    pub instructions: Vec<String>,
    pub projects: Vec<String>,
}

impl UserProfile {
    pub fn is_empty(&self) -> bool {
        self.preferences.is_empty()
            && self.facts.is_empty()
            && self.decisions.is_empty()
            && self.instructions.is_empty()
            && self.projects.is_empty()
    }

    /// Every stored fact as `(kind, text)` pairs, for ranking and rendering.
    fn pairs(&self) -> Vec<(&'static str, &String)> {
        let mut out = Vec::new();
        for t in &self.preferences {
            out.push(("preference", t));
        }
        for t in &self.facts {
            out.push(("fact", t));
        }
        for t in &self.decisions {
            out.push(("decision", t));
        }
        for t in &self.instructions {
            out.push(("instruction", t));
        }
        for t in &self.projects {
            out.push(("project", t));
        }
        out
    }

    fn push(&mut self, kind: &str, text: String) {
        let bucket = match kind {
            "preference" => &mut self.preferences,
            "decision" => &mut self.decisions,
            "instruction" => &mut self.instructions,
            "project" => &mut self.projects,
            _ => &mut self.facts,
        };
        // First occurrence wins for ordering; dedupe is by exact text.
        if !bucket.iter().any(|t| t == &text) {
            bucket.push(text);
        }
    }
}

/// The persistent user model over a small SQLite store.
pub struct UserModel {
    pool: SqlitePool,
    #[allow(dead_code)]
    path: PathBuf,
}

impl std::fmt::Debug for UserModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UserModel")
            .field("path", &self.path)
            .finish()
    }
}

impl UserModel {
    /// Open (creating if absent) the fact store at `db`. `":memory:"` gives
    /// a throwaway store with the same shape.
    pub async fn new(store: impl AsRef<Path>) -> Result<Self> {
        let path = store.as_ref().to_path_buf();
        let (url, max_connections) = if path == Path::new(":memory:") {
            ("sqlite::memory:".to_owned(), 1)
        } else {
            if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                tokio::fs::create_dir_all(parent)
                    .await
                    .context("creating user model directory")?;
            }
            (format!("sqlite://{}", path.display()), 4)
        };
        let options = SqliteConnectOptions::from_str(&url)
            .with_context(|| format!("parsing user model db url {url}"))?
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .busy_timeout(Duration::from_secs(5));
        let pool = SqlitePoolOptions::new()
            .max_connections(max_connections)
            .connect_with(options)
            .await
            .context("opening user model store")?;
        sqlx::raw_sql(SCHEMA)
            .execute(&pool)
            .await
            .context("migrating user model store")?;
        Ok(Self { pool, path })
    }

    /// Build a profile from a slice of events without persisting it. Pure:
    /// same events in, same profile out, store untouched.
    pub fn extract(events: &[SessionEvent]) -> UserProfile {
        let mut profile = UserProfile::default();
        let mut seen: HashSet<(String, String)> = HashSet::new();
        for event in events {
            if event.author != "user" {
                continue;
            }
            for (kind, text) in extract_facts(&event.text) {
                if seen.insert((kind.to_owned(), text.clone())) {
                    profile.push(kind, text);
                }
            }
        }
        profile
    }

    /// Fold one event into the stored profile. Seen counts are cumulative:
    /// hearing the same preference twice is evidence, not a duplicate.
    pub async fn update(&self, event: &SessionEvent) -> Result<()> {
        if event.author != "user" {
            return Ok(());
        }
        for (kind, text) in extract_facts(&event.text) {
            sqlx::query(
                "INSERT INTO user_facts (kind, text, seen, updated_at) VALUES (?, ?, 1, ?) \
                 ON CONFLICT(kind, text) DO UPDATE SET seen = seen + 1, updated_at = excluded.updated_at",
            )
            .bind(kind)
            .bind(&text)
            .bind(event.created_at as i64)
            .execute(&self.pool)
            .await
            .context("updating user model")?;
        }
        Ok(())
    }

    /// The current stored profile, most-repeated first inside each bucket.
    pub async fn profile(&self) -> UserProfile {
        let rows = sqlx::query(
            "SELECT kind, text FROM user_facts ORDER BY seen DESC, updated_at DESC",
        )
        .fetch_all(&self.pool)
        .await
        .unwrap_or_default();
        let mut profile = UserProfile::default();
        for row in rows {
            let kind: String = row.get("kind");
            let text: String = row.get("text");
            profile.push(&kind, text);
        }
        profile
    }

    /// The slice of the profile worth putting in front of the model for this
    /// query: facts sharing a keyword with it, else the most-repeated facts.
    /// Deterministic — no model call, no task-specific rules.
    pub async fn context_for(&self, query: &str) -> String {
        let profile = self.profile().await;
        if profile.is_empty() {
            return String::new();
        }
        let terms: HashSet<String> = query
            .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .map(|t| t.to_ascii_lowercase())
            .filter(|t| t.len() >= 2)
            .collect();
        let mut scored: Vec<(usize, (&str, &String))> = profile
            .pairs()
            .into_iter()
            .map(|pair| {
                let hay = pair.1.to_ascii_lowercase();
                let hits = terms
                    .iter()
                    .filter(|t| hay.split_whitespace().any(|w| w.trim_matches(|c: char| !c.is_ascii_alphanumeric()) == t.as_str()))
                    .count();
                (hits, pair)
            })
            .collect();
        // Relevant facts first; on no keyword overlap fall back to the whole
        // profile (it is small, and a vague query still deserves context).
        scored.sort_by(|a, b| b.0.cmp(&a.0));
        const MAX_FACTS: usize = 8;
        let mut out = String::from("## About the user\n");
        for (_, (kind, text)) in scored.into_iter().take(MAX_FACTS) {
            out.push_str(&format!("- [{kind}] {text}\n"));
        }
        out
    }
}

/// Markers that a user message carries a durable fact about the user. These
/// describe who the user is — never which task to run or which site to open.
const PREFERENCE_MARKERS: &[&str] = &[
    "i prefer ",
    "i'd prefer ",
    "i would prefer ",
    "i like ",
    "i dislike ",
    "i love ",
    "i hate ",
    "i don't like ",
    "my preference ",
];
const DECISION_MARKERS: &[&str] = &[
    "i decided ",
    "we decided ",
    "let's use ",
    "we'll use ",
    "i chose ",
    "i choose ",
    "the plan is ",
];
const INSTRUCTION_MARKERS: &[&str] = &[
    "remember that ",
    "remember this ",
    "don't forget ",
    "do not forget ",
    "from now on ",
];
const PROJECT_MARKERS: &[&str] = &[
    "i'm working on ",
    "i am working on ",
    "my project ",
    "my main project ",
];
const FACT_MARKERS: &[&str] = &[
    "i use ",
    "i'm using ",
    "i am using ",
    "my name is ",
    "call me ",
    "i live in ",
    "i work in ",
    "i study ",
];

/// One-shot transient chatter is not durable evidence about the user.
fn is_transient(lower: &str) -> bool {
    [
        "what is ",
        "what's ",
        "who is ",
        "how do i ",
        "how can i ",
        "can you ",
        "could you ",
        "please ",
        "thanks",
        "thank you",
        "hello",
        "hey ",
        "hi ",
    ]
    .iter()
    .any(|needle| lower.starts_with(needle) || lower.contains(needle))
}

fn extract_facts(text: &str) -> Vec<(&'static str, String)> {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.is_empty() || normalized.chars().count() > 400 {
        return Vec::new();
    }
    let lower = normalized.to_ascii_lowercase();
    if is_transient(&lower) {
        return Vec::new();
    }
    // "always " is anchored to the start so an occasional "always" mid-sentence
    // does not turn every observation into an instruction.
    if INSTRUCTION_MARKERS
        .iter()
        .any(|m| lower.contains(m))
        || lower.starts_with("always ")
    {
        return vec![("instruction", normalized)];
    }
    for (kind, markers) in [
        ("preference", PREFERENCE_MARKERS),
        ("decision", DECISION_MARKERS),
        ("project", PROJECT_MARKERS),
        ("fact", FACT_MARKERS),
    ] {
        if markers.iter().any(|m| lower.contains(m)) {
            return vec![(kind, normalized)];
        }
    }
    Vec::new()
}
