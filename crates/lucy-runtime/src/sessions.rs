//! Session management: opencode-style multi-session CRUD, history compaction
//! and trimming. Methods live here as `impl LucyRuntime` blocks so `lib.rs`
//! stays focused on construction and the submit/execute loop.

use anyhow::Result;

use lucy_core::{InterruptSignal, SessionId, SessionMeta, TurnMessage};

use super::LucyRuntime;

pub(crate) fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Drop old turns while keeping the newest `max_history`, preserving the
/// initial user prompt so the task goal is never lost.
pub fn trim_history(history: &mut Vec<TurnMessage>, max_history: usize) {
    if history.len() <= max_history {
        return;
    }
    let drop_n = history.len() - max_history;
    if history
        .first()
        .map_or(false, |m| matches!(m, TurnMessage::User(_)))
        && history.len() > 1
    {
        let drain_count = drop_n.min(history.len() - 1);
        history.drain(1..1 + drain_count);
    } else {
        history.drain(0..drop_n);
    }
}

impl LucyRuntime {
    pub async fn clear_session(&self) -> anyhow::Result<()> {
        let mut s = self.session.lock().await;
        s.session_id = SessionId::default();
        s.history.clear();
        s.updated_at = now();
        self.store.save(&s).await?;
        Ok(())
    }
    pub async fn compact(&self) -> anyhow::Result<String> {
        let max = self.config.sessions.max_history;
        let (before, recent, dropped) = {
            let s = self.session.lock().await;
            if s.history.len() <= max {
                return Ok("nothing to compact".to_string());
            }
            let total = s.history.len();
            let split = total.saturating_sub(20);
            (
                total,
                s.history[split..].to_vec(),
                s.history[..split].to_vec(),
            )
        };
        let full = format!("{:?}", dropped);
        let transcript: String = full.chars().take(12000).collect();
        let model = self.provider.model();
        let summary=self.provider.complete_json(&model,"Summarize this agent conversation prefix in English into 10 dense bullets: decisions, file changes, tool results, open tasks. Plain text only, no markdown.",&transcript,InterruptSignal::new()).await?;
        let summary_text = match summary {
            serde_json::Value::String(s) => s,
            other => other.to_string(),
        };
        let mut s = self.session.lock().await;
        let mut history = Vec::with_capacity(recent.len() + 1);
        history.push(TurnMessage::User(format!(
            "Conversation summary so far:\n{summary_text}"
        )));
        history.extend(recent);
        s.history = history;
        s.updated_at = now();
        self.store.save(&s).await?;
        Ok(format!(
            "compacted {before} -> {} messages",
            s.history.len()
        ))
    }
    pub fn sessions_dir(&self) -> &std::path::PathBuf {
        &self.store.dir
    }
    // ---- opencode-style session API ----
    pub async fn current_meta(&self) -> SessionMeta {
        SessionMeta::from(&*self.session.lock().await)
    }
    pub async fn current_id(&self) -> SessionId {
        self.session.lock().await.session_id.clone()
    }
    pub async fn list_sessions(&self) -> Result<Vec<SessionMeta>> {
        self.store.list().await
    }
    pub async fn new_session(&self, title: Option<String>) -> Result<SessionMeta> {
        self.interrupt.reset();
        let s = self.store.create(title).await?;
        let meta = SessionMeta::from(&s);
        *self.session.lock().await = s;
        Ok(meta)
    }
    pub async fn switch_session(&self, id: &SessionId) -> Result<SessionMeta> {
        self.interrupt.reset();
        let s = self.store.load(id).await?;
        let meta = SessionMeta::from(&s);
        *self.session.lock().await = s;
        Ok(meta)
    }
    pub async fn rename_current(&self, title: String) -> Result<SessionMeta> {
        let mut s = self.session.lock().await;
        let t = title.trim().to_owned();
        if !t.is_empty() {
            s.title = t;
        }
        s.updated_at = now();
        self.store.save(&s).await?;
        Ok(SessionMeta::from(&*s))
    }
    pub async fn delete_session(&self, id: &SessionId) -> Result<bool> {
        let current = self.session.lock().await.session_id.clone();
        let was_current = current == *id;
        self.store.delete(id).await?;
        if was_current {
            let remaining = self.store.list().await.unwrap_or_default();
            let next = if let Some(m) = remaining.first() {
                self.store.load(&m.id).await?
            } else {
                self.store.create(None).await?
            };
            *self.session.lock().await = next;
        }
        Ok(was_current)
    }
    pub async fn fork_current(&self) -> Result<SessionMeta> {
        let src = self.session.lock().await.clone();
        let forked = self.store.fork(&src, "(fork)").await?;
        let meta = SessionMeta::from(&forked);
        *self.session.lock().await = forked;
        Ok(meta)
    }
    pub async fn clear_current(&self) -> Result<SessionMeta> {
        let mut s = self.session.lock().await;
        s.history.clear();
        s.updated_at = now();
        self.store.save(&s).await?;
        Ok(SessionMeta::from(&*s))
    }
    /// Compact history to the last `keep` messages (opencode `/compact` analogue).
    /// Returns (before, after).
    pub async fn compact_current(&self, keep: usize) -> Result<(usize, usize)> {
        let mut s = self.session.lock().await;
        let before = s.history.len();
        let keep = keep.max(1);
        if before > keep {
            s.history.drain(0..before - keep);
        }
        s.updated_at = now();
        self.store.save(&s).await?;
        Ok((before, s.history.len()))
    }
    pub async fn export_current(&self, path: std::path::PathBuf) -> Result<()> {
        let s = self.session.lock().await;
        s.save_to_file(&path).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn trim_history_preserves_initial_user_prompt() {
        let mut hist = vec![
            TurnMessage::User("initial task".into()),
            TurnMessage::User("intermediate 1".into()),
            TurnMessage::User("intermediate 2".into()),
            TurnMessage::User("intermediate 3".into()),
            TurnMessage::User("latest turn".into()),
        ];
        trim_history(&mut hist, 3);
        assert_eq!(hist.len(), 3);
        assert!(matches!(&hist[0], TurnMessage::User(t) if t == "initial task"));
        assert!(matches!(&hist[1], TurnMessage::User(t) if t == "intermediate 3"));
        assert!(matches!(&hist[2], TurnMessage::User(t) if t == "latest turn"));
    }
}
