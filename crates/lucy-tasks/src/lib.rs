//! Task management and web page watching for Lucy.
//!
//! This crate provides:
//! - [`TaskStore`] — persistent SQLite-backed to-do list
//! - [`TaskNudge`] — periodic nudge scheduler for overdue/due-soon tasks
//! - [`WebPageWatcher`] — poll web pages for content changes

mod nudge;
mod watch;

pub use nudge::{Nudge, NudgeType, TaskNudge};
pub use watch::{ChangeType, PageChange, PageDiff, PageSnapshot, WatchConfig, WebPageWatcher};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use sqlx::{Row, sqlite::SqliteJournalMode};
use std::{path::PathBuf, time::Duration};
use uuid::Uuid;

/// Unique identifier for a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(pub Uuid);

impl TaskId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for TaskId {
    fn default() -> Self {
        Self::new()
    }
}

/// Unique identifier for a watch configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WatchId(pub Uuid);

impl WatchId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for WatchId {
    fn default() -> Self {
        Self::new()
    }
}

/// Current status of a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Cancelled,
}

impl TaskStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Cancelled => "cancelled",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "pending" => Some(Self::Pending),
            "in_progress" => Some(Self::InProgress),
            "completed" => Some(Self::Completed),
            "cancelled" => Some(Self::Cancelled),
            _ => None,
        }
    }
}

/// Priority level of a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TaskPriority {
    Low,
    Medium,
    High,
    Urgent,
}

impl TaskPriority {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Urgent => "urgent",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            "urgent" => Some(Self::Urgent),
            _ => None,
        }
    }
}

/// A single task in the to-do list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Task {
    pub id: TaskId,
    pub title: String,
    pub description: Option<String>,
    pub status: TaskStatus,
    pub priority: TaskPriority,
    pub due_at: Option<u64>,
    pub created_at: u64,
    pub completed_at: Option<u64>,
    pub tags: Vec<String>,
}

/// A task with only the one field a caller always has to supply.
///
/// The other eight fields are defaults a caller almost never sets, and
/// without this every construction site has to name all nine — which is how a
/// `due_at` ends up as `Some(0)` (silently overdue) in one caller and `None`
/// (never nudged) in another.
impl Task {
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            id: TaskId::new(),
            title: title.into(),
            description: None,
            status: TaskStatus::Pending,
            priority: TaskPriority::Medium,
            due_at: None,
            // `chrono::Utc` rather than `SystemTime`, because that is what the
            // store's own queries compare against — a timestamp from a
            // different clock base would order the task list at random.
            created_at: chrono::Utc::now().timestamp() as u64,
            completed_at: None,
            tags: Vec::new(),
        }
    }
}

/// SQLite-backed persistent task store.
pub struct TaskStore {
    pool: SqlitePool,
}

impl TaskStore {
    /// Open (or create) a task store at the given path.
    pub async fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("creating task store directory {}", parent.display()))?;
        }

        let options = SqliteConnectOptions::new()
            .filename(&path)
            .journal_mode(SqliteJournalMode::Wal)
            .create_if_missing(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await
            .with_context(|| format!("opening task store at {}", path.display()))?;

        let store = Self { pool };
        store.migrate().await?;
        Ok(store)
    }

    async fn migrate(&self) -> Result<()> {
        sqlx::raw_sql(
            r#"
            CREATE TABLE IF NOT EXISTS tasks (
                id            TEXT    NOT NULL PRIMARY KEY,
                title         TEXT    NOT NULL,
                description   TEXT,
                status        TEXT    NOT NULL DEFAULT 'pending',
                priority      TEXT    NOT NULL DEFAULT 'medium',
                due_at        INTEGER,
                created_at    INTEGER NOT NULL,
                completed_at  INTEGER,
                tags          TEXT    NOT NULL DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS tasks_status ON tasks(status);
            CREATE INDEX IF NOT EXISTS tasks_due_at ON tasks(due_at);
            CREATE INDEX IF NOT EXISTS tasks_priority ON tasks(priority);
            "#,
        )
        .execute(&self.pool)
        .await
        .context("running task store migrations")?;
        Ok(())
    }

    /// Add a new task. Returns the stored task with its assigned ID.
    pub async fn add(&self, task: Task) -> Result<TaskId> {
        let id = task.id;
        let tags_json = serde_json::to_string(&task.tags).context("serializing task tags")?;

        sqlx::query(
            r#"
            INSERT INTO tasks (id, title, description, status, priority, due_at, created_at, completed_at, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(id.0.to_string())
        .bind(&task.title)
        .bind(&task.description)
        .bind(task.status.as_str())
        .bind(task.priority.as_str())
        .bind(task.due_at.map(|v| v as i64))
        .bind(task.created_at as i64)
        .bind(task.completed_at.map(|v| v as i64))
        .bind(tags_json)
        .execute(&self.pool)
        .await
        .context("inserting task")?;

        Ok(id)
    }

    /// Mark a task as completed.
    pub async fn complete(&self, id: TaskId) -> Result<()> {
        let now = chrono::Utc::now().timestamp() as u64;
        let result = sqlx::query(
            r#"
            UPDATE tasks
            SET status = 'completed', completed_at = ?
            WHERE id = ? AND status != 'completed'
            "#,
        )
        .bind(now as i64)
        .bind(id.0.to_string())
        .execute(&self.pool)
        .await
        .context("completing task")?;

        if result.rows_affected() == 0 {
            anyhow::bail!("task {} not found or already completed", id.0);
        }
        Ok(())
    }

    /// Delete a task permanently.
    pub async fn delete(&self, id: TaskId) -> Result<()> {
        let result = sqlx::query("DELETE FROM tasks WHERE id = ?")
            .bind(id.0.to_string())
            .execute(&self.pool)
            .await
            .context("deleting task")?;

        if result.rows_affected() == 0 {
            anyhow::bail!("task {} not found", id.0);
        }
        Ok(())
    }

    /// List all tasks, ordered by creation time descending.
    pub async fn list(&self) -> Result<Vec<Task>> {
        let rows = sqlx::query(
            r#"
            SELECT id, title, description, status, priority, due_at, created_at, completed_at, tags
            FROM tasks
            ORDER BY created_at DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await
        .context("listing tasks")?;

        rows.into_iter().map(row_to_task).collect()
    }

    /// Get all pending/in-progress tasks that are overdue.
    pub async fn overdue(&self) -> Result<Vec<Task>> {
        let now = chrono::Utc::now().timestamp() as u64;
        let rows = sqlx::query(
            r#"
            SELECT id, title, description, status, priority, due_at, created_at, completed_at, tags
            FROM tasks
            WHERE status IN ('pending', 'in_progress')
              AND due_at IS NOT NULL
              AND due_at < ?
            ORDER BY due_at ASC
            "#,
        )
        .bind(now as i64)
        .fetch_all(&self.pool)
        .await
        .context("fetching overdue tasks")?;

        rows.into_iter().map(row_to_task).collect()
    }

    /// Get pending/in-progress tasks due within the given duration.
    pub async fn due_soon(&self, within: Duration) -> Result<Vec<Task>> {
        let now = chrono::Utc::now().timestamp() as u64;
        let horizon = now + within.as_secs() as u64;
        let rows = sqlx::query(
            r#"
            SELECT id, title, description, status, priority, due_at, created_at, completed_at, tags
            FROM tasks
            WHERE status IN ('pending', 'in_progress')
              AND due_at IS NOT NULL
              AND due_at >= ?
              AND due_at <= ?
            ORDER BY due_at ASC
            "#,
        )
        .bind(now as i64)
        .bind(horizon as i64)
        .fetch_all(&self.pool)
        .await
        .context("fetching due-soon tasks")?;

        rows.into_iter().map(row_to_task).collect()
    }

    /// Get a single task by ID.
    pub async fn get(&self, id: TaskId) -> Result<Option<Task>> {
        let row = sqlx::query(
            r#"
            SELECT id, title, description, status, priority, due_at, created_at, completed_at, tags
            FROM tasks
            WHERE id = ?
            "#,
        )
        .bind(id.0.to_string())
        .fetch_optional(&self.pool)
        .await
        .context("fetching task")?;

        row.map(row_to_task).transpose()
    }

    /// Update task status.
    pub async fn set_status(&self, id: TaskId, status: TaskStatus) -> Result<()> {
        let result = sqlx::query("UPDATE tasks SET status = ? WHERE id = ?")
            .bind(status.as_str())
            .bind(id.0.to_string())
            .execute(&self.pool)
            .await
            .context("updating task status")?;

        if result.rows_affected() == 0 {
            anyhow::bail!("task {} not found", id.0);
        }
        Ok(())
    }
}

fn row_to_task(row: sqlx::sqlite::SqliteRow) -> Result<Task> {
    let id_str: String = row.try_get("id")?;
    let id = TaskId(
        Uuid::parse_str(&id_str).with_context(|| format!("parsing task id {}", id_str))?,
    );
    let title: String = row.try_get("title")?;
    let description: Option<String> = row.try_get("description")?;
    let status_str: String = row.try_get("status")?;
    let status = TaskStatus::from_str(&status_str)
        .with_context(|| format!("unknown task status: {}", status_str))?;
    let priority_str: String = row.try_get("priority")?;
    let priority = TaskPriority::from_str(&priority_str)
        .with_context(|| format!("unknown task priority: {}", priority_str))?;
    let due_at: Option<i64> = row.try_get("due_at")?;
    let created_at: i64 = row.try_get("created_at")?;
    let completed_at: Option<i64> = row.try_get("completed_at")?;
    let tags_json: String = row.try_get("tags")?;
    let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();

    Ok(Task {
        id,
        title,
        description,
        status,
        priority,
        due_at: due_at.map(|v| v as u64),
        created_at: created_at as u64,
        completed_at: completed_at.map(|v| v as u64),
        tags,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_status_roundtrip() {
        for status in [
            TaskStatus::Pending,
            TaskStatus::InProgress,
            TaskStatus::Completed,
            TaskStatus::Cancelled,
        ] {
            assert_eq!(TaskStatus::from_str(status.as_str()), Some(status));
        }
    }

    #[test]
    fn task_priority_roundtrip() {
        for priority in [
            TaskPriority::Low,
            TaskPriority::Medium,
            TaskPriority::High,
            TaskPriority::Urgent,
        ] {
            assert_eq!(TaskPriority::from_str(priority.as_str()), Some(priority));
        }
    }

    #[test]
    fn task_id_is_unique() {
        let a = TaskId::new();
        let b = TaskId::new();
        assert_ne!(a, b);
    }
}
