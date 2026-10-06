//! Integration tests for the Scheduler.

use std::sync::Arc;
use std::time::Duration;

use lucy_scheduler::{
    DeliveryConfig, JobStatus, Scheduler, SchedulerConfig, ScheduledTask,
};

fn test_scheduler() -> Scheduler {
    Scheduler::new(SchedulerConfig::default())
}

#[tokio::test]
async fn schedule_and_list_tasks() {
    let scheduler = test_scheduler();
    let task = ScheduledTask::new(
        "standup",
        "0 9 * * 1-5",
        "prepare standup notes",
        DeliveryConfig::tui(),
    )
    .unwrap();
    let id = scheduler.schedule(task);

    let tasks = scheduler.list();
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].id, id);
    assert_eq!(tasks[0].name, "standup");
    assert!(tasks[0].enabled);
}

#[tokio::test]
async fn cancel_disables_task() {
    let scheduler = test_scheduler();
    let task = ScheduledTask::new("report", "@daily", "write report", DeliveryConfig::tui())
        .unwrap();
    let id = scheduler.schedule(task);

    assert!(scheduler.cancel(id));
    let tasks = scheduler.list();
    assert!(!tasks[0].enabled);
    assert_eq!(tasks[0].next_run, None);

    assert!(!scheduler.cancel(lucy_scheduler::TaskId::new()));
}

#[tokio::test]
async fn remove_deletes_task() {
    let scheduler = test_scheduler();
    let task = ScheduledTask::new("once", "@hourly", "sweep temp", DeliveryConfig::tui())
        .unwrap();
    let id = scheduler.schedule(task);

    assert!(scheduler.remove(id));
    assert!(scheduler.list().is_empty());
}

#[tokio::test]
async fn run_pending_executes_due_tasks_and_rearms() {
    let scheduler = test_scheduler();
    let mut task = ScheduledTask::new(
        "due",
        "* * * * *",
        "collect metrics",
        DeliveryConfig::tui(),
    )
    .unwrap();
    // Force the task to be due right now.
    task.next_run = Some(0);
    let id = scheduler.schedule(task);

    let results = scheduler.run_pending().await;
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].task_id, id);
    assert!(results[0].success);
    assert_eq!(results[0].goal, "collect metrics");

    // The task has been re-armed and records its last run.
    let tasks = scheduler.list();
    assert!(tasks[0].last_run.is_some());
    assert!(tasks[0].next_run.unwrap() > 0);
}

#[tokio::test]
async fn run_pending_skips_future_tasks() {
    let scheduler = test_scheduler();
    let mut task = ScheduledTask::new(
        "not-due",
        "0 0 1 1 *",
        "yearly",
        DeliveryConfig::tui(),
    )
    .unwrap();
    task.next_run = Some(u64::MAX / 2);
    scheduler.schedule(task);

    let results = scheduler.run_pending().await;
    assert!(results.is_empty());
}

#[tokio::test]
async fn run_pending_uses_real_runner() {
    let scheduler = test_scheduler().with_runner(Arc::new(|goal: String| {
        Box::pin(async move { Ok(format!("ran: {goal}")) })
            as std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<String, String>> + Send>,
            >
    }));
    let mut task = ScheduledTask::new("g", "* * * * *", "do thing", DeliveryConfig::tui())
        .unwrap();
    task.next_run = Some(0);
    scheduler.schedule(task);

    let results = scheduler.run_pending().await;
    assert_eq!(results[0].output, "ran: do thing");
}

#[tokio::test]
async fn background_job_completes() {
    let scheduler = test_scheduler();
    let id = scheduler.spawn_background("index files").await;

    for _ in 0..100 {
        let status = scheduler.job_status(id).await;
        if status.is_terminal() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    match scheduler.job_status(id).await {
        JobStatus::Completed(out) => assert_eq!(out, "completed: index files"),
        other => panic!("expected completed, got {other:?}"),
    }
}

#[tokio::test]
async fn background_job_unknown_id_is_failed() {
    let scheduler = test_scheduler();
    let status = scheduler.job_status(lucy_scheduler::JobId::new()).await;
    assert!(matches!(status, JobStatus::Failed(_)));
}

#[tokio::test]
async fn schedule_rejects_invalid_cron() {
    let result = ScheduledTask::new("bad", "not a cron", "goal", DeliveryConfig::tui());
    assert!(result.is_err());
}
