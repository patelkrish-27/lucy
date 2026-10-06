//! Shared setup for the task integration tests.
//!
//! Per-test directories, because these tests write to SQLite and would
//! otherwise contend on one database file when run concurrently.

use lucy_tasks::TaskStore;
use std::path::PathBuf;

/// A throwaway store plus the directory backing it.
pub async fn fresh_store(tag: &str) -> (PathBuf, TaskStore) {
    let dir = std::env::temp_dir().join(format!(
        "lucy-tasks-it-{}-{tag}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = tokio::fs::remove_dir_all(&dir).await;
    tokio::fs::create_dir_all(&dir)
        .await
        .expect("creating fixture dir");
    // `TaskStore::open` takes the database *file*, not a directory — note the
    // asymmetry with `PageStore::open`, which takes a directory and names its
    // own files inside it.
    let db = dir.join("tasks.db");
    let store = TaskStore::open(&db).await.expect("opening store");
    (dir, store)
}
