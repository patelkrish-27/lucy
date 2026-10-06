//! Shared setup for the integration tests.
//!
//! Each test gets its own directory so they can run concurrently without
//! sharing a database file. The directory is removed by the test itself.

use lucy_spaces::PageStore;
use std::path::PathBuf;

/// A throwaway store plus the directory backing it.
///
/// The caller removes the directory at the end of the test — deliberately not
/// a `Drop`, so a failing assertion leaves the fixture on disk to inspect.
pub async fn fresh_store(tag: &str) -> (PathBuf, PageStore) {
    let dir = std::env::temp_dir().join(format!(
        "lucy-spaces-it-{}-{tag}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    // A leftover from a previous run with the same pid would make
    // `create_space` fail on a duplicate name.
    let _ = tokio::fs::remove_dir_all(&dir).await;
    tokio::fs::create_dir_all(&dir)
        .await
        .expect("creating fixture dir");
    let store = PageStore::open(&dir).await.expect("opening store");
    (dir, store)
}
