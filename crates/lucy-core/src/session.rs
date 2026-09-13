use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::fs;
use anyhow::{Context, Result};
use crate::{SessionId, TurnMessage};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionData {
    pub session_id: SessionId,
    pub history: Vec<TurnMessage>,
    pub created_at: u64,
    pub updated_at: u64,
}

impl SessionData {
    pub fn new(session_id: SessionId) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            session_id,
            history: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }

    pub async fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path).await.context("failed to read session file")?;
        let session: SessionData = serde_json::from_str(&content).context("failed to parse session data")?;
        Ok(session)
    }

    pub async fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        if let Some(parent) = path.as_ref().parent() {
            fs::create_dir_all(parent).await.context("failed to create session directory")?;
        }
        let content = serde_json::to_string_pretty(self).context("failed to serialize session data")?;
        fs::write(path, content).await.context("failed to write session file")?;
        Ok(())
    }
}
