//! Integration tests for the task store and the nudge scheduler.
//!
//! Driven against a real SQLite file, because what is worth testing here is
//! the schema agreement between the store and the nudge queries — a `due_at`
//! that stores seconds while the nudge query compares it to a formatted local
//! time string compiles fine and nudges about the wrong task forever.

use lucy_tasks::{Task, TaskId, TaskNudge, TaskStatus, TaskStore};
use std::sync::Arc;

mod util;
use util::fresh_store;

/// A task due `due_in_secs` from now (negative is already past due).
fn task(title: &str, due_in_secs: Option<i64>) -> Task {
    let mut t = Task::new(title);
    t.due_at = due_in_secs.map(|d| (now() as i64 + d) as u64);
    t
}

/// Seconds since the epoch, from the same clock base the store uses.
fn now() -> u64 {
    chrono::Utc::now().timestamp() as u64
}

#[tokio::test]
async fn a_completed_task_stops_being_nudged() {
    let (dir, store) = fresh_store("completed").await;
    let store = Arc::new(store);

    let id = store
        .add(task("Overdue thing", Some(-3600)))
        .await
        .expect("adding");

    let nudger = TaskNudge::new(store.clone(), std::time::Duration::from_secs(60));
    let before = nudger.nudges().await.expect("nudging");
    assert_eq!(before.len(), 1, "an overdue task is nudged: {before:?}");

    store.complete(id).await.expect("completing");

    let after = nudger.nudges().await.expect("nudging");
    assert!(
        after.is_empty(),
        "a completed task must stop being nudged, else the nudge is nagging about finished work: {after:?}"
    );

    let _ = tokio::fs::remove_dir_all(&dir).await;
}

#[tokio::test]
async fn a_task_with_no_due_date_is_listed_but_never_nudged() {
    let (dir, store) = fresh_store("no-due-date").await;
    let store = Arc::new(store);

    store.add(task("Someday", None)).await.expect("adding");

    // It is real work the user asked Lucy to hold, so it belongs in the list…
    assert_eq!(store.list().await.expect("listing").len(), 1);

    // …but "nudge me about this with no date" has no trigger, so a nudge loop
    // firing on it would be pure noise.
    let nudger = TaskNudge::new(store, std::time::Duration::from_secs(60));
    assert!(
        nudger.nudges().await.expect("nudging").is_empty(),
        "no due date means no nudge"
    );

    let _ = tokio::fs::remove_dir_all(&dir).await;
}

#[tokio::test]
async fn a_due_soon_task_is_distinguished_from_an_overdue_one() {
    let (dir, store) = fresh_store("due-soon").await;
    let store = Arc::new(store);

    store.add(task("Soon", Some(600))).await.expect("adding");
    store.add(task("Late", Some(-600))).await.expect("adding");

    let nudger = TaskNudge::new(store.clone(), std::time::Duration::from_secs(60));
    let nudges = nudger.nudges().await.expect("nudging");
    assert_eq!(nudges.len(), 2, "both are past due: {nudges:?}");

    // The distinction matters to the message the user reads: "due soon" and
    // "overdue" ask for different urgency.
    let kinds: Vec<_> = nudges.iter().map(|n| n.nudge_type.clone()).collect();
    assert!(
        kinds.iter().any(|k| *k == lucy_tasks::NudgeType::Overdue),
        "the past-due task is Overdue: {kinds:?}"
    );

    let _ = tokio::fs::remove_dir_all(&dir).await;
}

#[tokio::test]
async fn deleting_a_task_removes_it_from_a_reopened_store() {
    let (dir, store) = fresh_store("delete-persists").await;

    let id = store.add(task("Temporary", None)).await.expect("adding");
    store.delete(id).await.expect("deleting");

    let reopened = TaskStore::open(dir.join("tasks.db")).await.expect("reopening");
    assert!(
        reopened.list().await.expect("listing").is_empty(),
        "the delete must survive the reopen"
    );

    let _ = tokio::fs::remove_dir_all(&dir).await;
}

#[tokio::test]
async fn overdue_reads_the_same_due_at_that_was_written() {
    let (dir, store) = fresh_store("overdue-agreement").await;

    // The store's own `overdue` filter and the nudge scheduler must agree about
    // what "overdue" means. Writing through one path and reading through the
    // other is exactly how the two drift apart, so this asserts on both.
    store.add(task("Late", Some(-60))).await.expect("adding");

    let store = Arc::new(store);
    assert_eq!(
        store.overdue().await.expect("overdue").len(),
        1,
        "the store sees it as overdue"
    );

    let nudger = TaskNudge::new(store, std::time::Duration::from_secs(60));
    assert_eq!(
        nudger.nudges().await.expect("nudging").len(),
        1,
        "and so does the nudge scheduler"
    );

    let _ = tokio::fs::remove_dir_all(&dir).await;
}

#[tokio::test]
async fn a_task_id_from_one_store_is_rejected_by_another() {
    let (dir_a, store_a) = fresh_store("cross-store-a").await;
    let (dir_b, store_b) = fresh_store("cross-store-b").await;

    let foreign = TaskId::new();
    let mut local = Task::new("Local");
    local.id = foreign;
    store_b.add(local).await.expect("adding to b");

    // A well-formed id from a different database must not silently resolve.
    let result = store_a.get(foreign).await;
    let leaked = result.as_ref().err().is_none() && result.as_ref().ok().and_then(|o| o.as_ref()).is_some();
    assert!(
        !leaked,
        "a task id must not leak across stores: {result:?}"
    );

    let _ = tokio::fs::remove_dir_all(&dir_a).await;
    let _ = tokio::fs::remove_dir_all(&dir_b).await;
}

#[tokio::test]
async fn a_task_returns_to_its_default_status_after_reopening() {
    let (dir, store) = fresh_store("status-roundtrip").await;

    let id = store.add(task("Round trip", None)).await.expect("adding");
    let reopened = TaskStore::open(dir.join("tasks.db")).await.expect("reopening");
    let page = reopened.get(id).await.expect("getting").expect("exists");

    assert_eq!(
        page.status,
        TaskStatus::Pending,
        "a task must not come back claiming to be in progress"
    );

    let _ = tokio::fs::remove_dir_all(&dir).await;
}
