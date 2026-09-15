use crate::{SessionId, TurnMessage};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionData {
    pub session_id: SessionId,
    #[serde(default)]
    pub title: String,
    pub history: Vec<TurnMessage>,
    pub created_at: u64,
    pub updated_at: u64,
}

impl SessionData {
    pub fn new(session_id: SessionId) -> Self {
        let now = now_secs();
        Self {
            session_id,
            title: "untitled".into(),
            history: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        let t = title.into().trim().to_owned();
        if !t.is_empty() {
            self.title = t;
        }
        self
    }

    /// Opencode-style auto title: first user message, truncated to ~48 chars.
    pub fn autotitle_from(text: &str) -> String {
        let one_line = text.split_whitespace().collect::<Vec<_>>().join(" ");
        let mut t: String = one_line.chars().take(48).collect();
        if one_line.chars().count() > 48 {
            t.push('…');
        }
        if t.trim().is_empty() {
            "untitled".into()
        } else {
            t
        }
    }

    pub fn touch(&mut self) {
        self.updated_at = now_secs();
    }

    pub fn preview(&self) -> String {
        for m in self.history.iter().rev() {
            match m {
                TurnMessage::User(t) => return Self::autotitle_from(t),
                TurnMessage::Assistant(turn) => {
                    if let Some(t) = turn.text.as_deref() {
                        if !t.trim().is_empty() {
                            return Self::autotitle_from(t);
                        }
                    }
                }
                TurnMessage::Tool(_) => continue,
            }
        }
        self.title.clone()
    }

    pub async fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path.as_ref())
            .await
            .context("failed to read session file")?;
        let mut session: SessionData =
            serde_json::from_str(&content).context("failed to parse session data")?;
        // Back-compat: old files lack `title`.
        if session.title.trim().is_empty() {
            session.title = session.preview();
            if session.title.trim().is_empty() {
                session.title = "untitled".into();
            }
        }
        Ok(session)
    }

    pub async fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        if let Some(parent) = path.as_ref().parent() {
            fs::create_dir_all(parent)
                .await
                .context("failed to create session directory")?;
        }
        let content =
            serde_json::to_string_pretty(self).context("failed to serialize session data")?;
        fs::write(path, content)
            .await
            .context("failed to write session file")?;
        Ok(())
    }
}

pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Lightweight row for session pickers — mirrors opencode's session list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    pub id: SessionId,
    pub title: String,
    pub message_count: usize,
    pub preview: String,
    pub created_at: u64,
    pub updated_at: u64,
}

impl From<&SessionData> for SessionMeta {
    fn from(s: &SessionData) -> Self {
        Self {
            id: s.session_id.clone(),
            title: s.title.clone(),
            message_count: s.history.len(),
            preview: s.preview(),
            created_at: s.created_at,
            updated_at: s.updated_at,
        }
    }
}

/// Opencode-style multi-session store: one JSON file per session.
#[derive(Debug, Clone)]
pub struct SessionStore {
    pub dir: PathBuf,
}

impl SessionStore {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }

    pub fn default_dir() -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        PathBuf::from(home).join(".local/state/lucy/sessions")
    }

    pub fn session_path(&self, id: &SessionId) -> PathBuf {
        self.dir.join(format!("{}.json", id.0))
    }

    pub async fn ensure_dir(&self) -> Result<()> {
        fs::create_dir_all(&self.dir)
            .await
            .context("failed to create sessions dir")?;
        Ok(())
    }

    pub async fn save(&self, session: &SessionData) -> Result<()> {
        self.ensure_dir().await?;
        session
            .save_to_file(self.session_path(&session.session_id))
            .await
    }

    pub async fn load(&self, id: &SessionId) -> Result<SessionData> {
        SessionData::load_from_file(self.session_path(id)).await
    }

    pub async fn delete(&self, id: &SessionId) -> Result<()> {
        let p = self.session_path(id);
        if p.exists() {
            fs::remove_file(&p)
                .await
                .context("failed to delete session")?;
        }
        Ok(())
    }

    /// Newest-first list. Corrupt files are skipped (renamed to .corrupt-TS.bak).
    pub async fn list(&self) -> Result<Vec<SessionMeta>> {
        self.ensure_dir().await?;
        let mut out = Vec::new();
        let mut entries = fs::read_dir(&self.dir)
            .await
            .context("failed to read sessions dir")?;
        while let Some(entry) = entries
            .next_entry()
            .await
            .context("failed to read session entry")?
        {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            match SessionData::load_from_file(&path).await {
                Ok(s) => out.push(SessionMeta::from(&s)),
                Err(_) => {
                    let ts = now_secs();
                    let mut bak = path.clone().into_os_string();
                    bak.push(format!(".corrupt-{ts}.bak"));
                    let _ = std::fs::rename(&path, std::path::PathBuf::from(bak));
                }
            }
        }
        out.sort_by(|a, b| {
            b.updated_at
                .cmp(&a.updated_at)
                .then(b.created_at.cmp(&a.created_at))
        });
        Ok(out)
    }

    pub async fn create(&self, title: Option<String>) -> Result<SessionData> {
        let mut s = SessionData::new(SessionId::default());
        if let Some(t) = title {
            s = s.with_title(t);
        }
        self.save(&s).await?;
        Ok(s)
    }

    pub async fn fork(&self, src: &SessionData, title_suffix: &str) -> Result<SessionData> {
        let mut forked = SessionData::new(SessionId::default());
        forked.history = src.history.clone();
        let base = if src.title.trim().is_empty() {
            "untitled".to_owned()
        } else {
            src.title.clone()
        };
        forked.title = format!("{base} {title_suffix}").trim().to_owned();
        self.save(&forked).await?;
        Ok(forked)
    }

    /// Migrate a legacy single `session.json` into the store (opencode parity:
    /// never lose the previous conversation on upgrade).
    pub async fn migrate_legacy_file<P: AsRef<Path>>(
        &self,
        legacy: P,
    ) -> Result<Option<SessionData>> {
        let legacy = legacy.as_ref();
        if !legacy.exists() {
            return Ok(None);
        }
        // Don't migrate twice.
        if !self.list().await.unwrap_or_default().is_empty() {
            return Ok(None);
        }
        match SessionData::load_from_file(legacy).await {
            Ok(mut s) => {
                if s.title.trim().is_empty() || s.title == "untitled" {
                    let p = s.preview();
                    if !p.trim().is_empty() {
                        s.title = p;
                    }
                }
                self.save(&s).await?;
                Ok(Some(s))
            }
            Err(_) => {
                let ts = now_secs();
                let mut bak = legacy.as_os_str().to_owned();
                bak.push(format!(".corrupt-{ts}.bak"));
                let _ = std::fs::rename(legacy, std::path::PathBuf::from(bak));
                Ok(None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autotitle_truncates() {
        let t = SessionData::autotitle_from(
            "  hello   world  this is a very long message that should be truncated at some point yes",
        );
        assert!(t.chars().count() <= 49, "got {t:?}");
        assert!(t.starts_with("hello world"));
    }

    #[tokio::test]
    async fn store_roundtrip_and_list() {
        let dir = std::env::temp_dir().join(format!("lucy-sessions-test-{}", now_secs()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = SessionStore::new(dir.clone());
        let mut s = store.create(Some("demo".into())).await.unwrap();
        s.history.push(TurnMessage::User("hi".into()));
        s.touch();
        store.save(&s).await.unwrap();
        let list = store.list().await.unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].title, "demo");
        let loaded = store.load(&s.session_id).await.unwrap();
        assert_eq!(loaded.history.len(), 1);
        store.delete(&s.session_id).await.unwrap();
        assert!(store.list().await.unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
