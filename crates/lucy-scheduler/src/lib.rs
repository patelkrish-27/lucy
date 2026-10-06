//! Scheduled automations and background work for Lucy.
//!
//! This crate provides:
//! - [`Scheduler`] — cron-driven task scheduling and one-off task runs
//! - [`ScheduledTask`] — a task bound to a [`CronExpression`]
//! - [`BackgroundJob`] — a long-running background goal
//! - [`cron`] — cron expression parsing (`*`, `*/5`, `1-5`, `1,3,5`, `@hourly`, …)
//! - [`executor`] — bounded-concurrency background execution
//! - [`delivery`] — result delivery to TUI, Gateway, and Webhook channels

pub mod cron;
pub mod delivery;
pub mod executor;

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub use cron::{CronError, CronExpression, CronField};
pub use delivery::{
    DeliveryChannel, DeliveryConfig, DeliveryError, DeliveryHandler, DeliveryMessage,
    DeliveryRouter, GatewayDelivery, TuiDelivery, WebhookDelivery,
};
pub use executor::{
    BackgroundJob, Executor, ExecutorError, JobId, JobStatus, now_secs,
};

/// Unique identifier for a scheduled task.
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

impl std::fmt::Display for TaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A task bound to a cron schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledTask {
    pub id: TaskId,
    pub name: String,
    /// The raw schedule string, e.g. `"0 8 * * 1-5"` or `"@hourly"`.
    pub schedule: String,
    pub cron: CronExpression,
    pub goal: String,
    pub delivery: DeliveryConfig,
    pub created_at: u64,
    /// Unix seconds of the next scheduled run; `None` when disabled.
    pub next_run: Option<u64>,
    pub last_run: Option<u64>,
    pub enabled: bool,
}

impl ScheduledTask {
    /// Create a new scheduled task, computing its first run time.
    pub fn new(
        name: impl Into<String>,
        schedule: impl Into<String>,
        goal: impl Into<String>,
        delivery: DeliveryConfig,
    ) -> Result<Self, CronError> {
        let schedule = schedule.into();
        let cron = CronExpression::parse(&schedule)?;
        let next_run = cron
            .next_after(&Utc::now())
            .map(|dt| dt.timestamp().max(0) as u64);
        Ok(Self {
            id: TaskId::new(),
            name: name.into(),
            schedule,
            cron,
            goal: goal.into(),
            delivery,
            created_at: now_secs(),
            next_run,
            last_run: None,
            enabled: true,
        })
    }
}

/// The outcome of running a scheduled task once.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: TaskId,
    pub goal: String,
    pub success: bool,
    pub output: String,
    pub timestamp: u64,
}

/// Configuration for a [`Scheduler`].
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum number of background jobs running at once.
    pub max_concurrent_jobs: usize,
    /// Default delivery config for tasks that don't specify one.
    pub default_delivery: DeliveryConfig,
    /// Router that sends task output to delivery channels.
    pub delivery_router: Arc<DeliveryRouter>,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_jobs: 4,
            default_delivery: DeliveryConfig::tui(),
            delivery_router: Arc::new(DeliveryRouter::new()),
        }
    }
}

/// Async goal runner: goal in, output or error string out.
pub type GoalRunner = Arc<
    dyn Fn(
            String,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<String, String>> + Send>,
        > + Send
        + Sync,
>;

/// The scheduler: owns scheduled tasks, background execution, and delivery.
pub struct Scheduler {
    config: SchedulerConfig,
    tasks: Arc<RwLock<HashMap<TaskId, ScheduledTask>>>,
    executor: Arc<Executor>,
    runner: GoalRunner,
}

impl Scheduler {
    /// Create a new scheduler with the given config.
    pub fn new(config: SchedulerConfig) -> Self {
        let executor = Arc::new(Executor::new(config.max_concurrent_jobs));
        Self {
            config,
            tasks: Arc::new(RwLock::new(HashMap::new())),
            executor,
            runner: default_runner(),
        }
    }

    /// Replace the goal runner used to execute scheduled goals and
    /// background jobs. The default runner echoes the goal.
    pub fn with_runner(mut self, runner: GoalRunner) -> Self {
        self.runner = runner;
        self
    }

    /// Schedule a new task. Returns its id.
    pub fn schedule(&self, task: ScheduledTask) -> TaskId {
        let id = task.id;
        let mut tasks = self.tasks.write().expect("tasks lock poisoned");
        tasks.insert(id, task);
        tracing::info!(task_id = %id, "task scheduled");
        id
    }

    /// Cancel a scheduled task by id. Returns true if it existed.
    pub fn cancel(&self, id: TaskId) -> bool {
        let mut tasks = self.tasks.write().expect("tasks lock poisoned");
        match tasks.get_mut(&id) {
            Some(task) => {
                task.enabled = false;
                task.next_run = None;
                tracing::info!(task_id = %id, "task cancelled");
                true
            }
            None => false,
        }
    }

    /// Remove a task from the registry entirely. Returns true if it existed.
    pub fn remove(&self, id: TaskId) -> bool {
        let mut tasks = self.tasks.write().expect("tasks lock poisoned");
        tasks.remove(&id).is_some()
    }

    /// List all scheduled tasks.
    pub fn list(&self) -> Vec<ScheduledTask> {
        let tasks = self.tasks.read().expect("tasks lock poisoned");
        let mut all: Vec<ScheduledTask> = tasks.values().cloned().collect();
        all.sort_by_key(|t| t.created_at);
        all
    }

    /// Run every task whose `next_run` is due, deliver the results, and
    /// recompute each task's next run time. Returns one result per task run.
    pub async fn run_pending(&self) -> Vec<TaskResult> {
        let now = now_secs();
        let due: Vec<ScheduledTask> = {
            let tasks = self.tasks.read().expect("tasks lock poisoned");
            tasks
                .values()
                .filter(|t| t.enabled && t.next_run.is_some_and(|n| n <= now))
                .cloned()
                .collect()
        };

        let mut results = Vec::new();
        for task in due {
            let executed = (self.runner)(task.goal.clone()).await;
            let timestamp = now_secs();
            let (success, output) = match executed {
                Ok(out) => (true, out),
                Err(err) => (false, err),
            };

            let message = DeliveryMessage::new(
                task.id.to_string(),
                output.clone(),
                success,
                timestamp,
                task.delivery.destination.clone(),
            );
            if let Err(err) = self.config.delivery_router.deliver(&task.delivery, &message).await
            {
                tracing::warn!(task_id = %task.id, error = %err, "delivery failed");
            }

            // Re-arm the schedule
            let next_run = DateTime::<Utc>::from_timestamp(timestamp as i64, 0)
                .and_then(|dt| task.cron.next_after(&dt))
                .map(|dt| dt.timestamp().max(0) as u64);
            {
                let mut tasks = self.tasks.write().expect("tasks lock poisoned");
                if let Some(stored) = tasks.get_mut(&task.id) {
                    stored.last_run = Some(timestamp);
                    stored.next_run = next_run;
                }
            }

            results.push(TaskResult {
                task_id: task.id,
                goal: task.goal.clone(),
                success,
                output,
                timestamp,
            });
        }
        results
    }

    /// Spawn a background job with the given goal. Returns its job id.
    pub async fn spawn_background(&self, goal: impl Into<String>) -> JobId {
        let goal = goal.into();
        let runner = self.runner.clone();
        let goal_for_job = goal.clone();
        self.executor
            .spawn(goal, move || runner(goal_for_job.clone()))
            .await
            .expect("executor rejected spawn")
    }

    /// Get the current status of a background job.
    pub async fn job_status(&self, id: JobId) -> JobStatus {
        match self.executor.status(id).await {
            Ok(status) => status,
            Err(_) => JobStatus::Failed(format!("job not found: {id}")),
        }
    }

    /// List background jobs, optionally only active ones.
    pub async fn jobs(&self) -> Vec<BackgroundJob> {
        self.executor.list().await
    }

    /// Cancel a background job.
    pub async fn cancel_job(&self, id: JobId) -> Result<JobStatus, ExecutorError> {
        self.executor.cancel(id).await
    }
}

fn default_runner() -> GoalRunner {
    Arc::new(|goal: String| {
        Box::pin(async move { Ok(format!("completed: {goal}")) })
            as std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<String, String>> + Send>,
            >
    })
}
