//! Background task execution engine.
//!
//! Manages a pool of background jobs with concurrency limits, status tracking,
//! and cancellation support.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tokio::sync::{RwLock, Semaphore};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Unique identifier for a background job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JobId(pub Uuid);

impl JobId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for JobId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for JobId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Current status of a background job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus {
    /// Waiting for a slot in the concurrency pool.
    Pending,
    /// Currently executing.
    Running,
    /// Finished successfully with output.
    Completed(String),
    /// Failed with an error message.
    Failed(String),
    /// Cancelled before completion.
    Cancelled,
}

impl JobStatus {
    /// Returns true if the job is no longer active.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            JobStatus::Completed(_) | JobStatus::Failed(_) | JobStatus::Cancelled
        )
    }

    /// Returns true if the job is still active (pending or running).
    pub fn is_active(&self) -> bool {
        matches!(self, JobStatus::Pending | JobStatus::Running)
    }
}

/// A background job with its metadata and current state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundJob {
    pub id: JobId,
    pub goal: String,
    pub status: JobStatus,
    pub result: Option<String>,
    pub created_at: u64,
    pub completed_at: Option<u64>,
}

impl BackgroundJob {
    /// Create a new background job with Pending status.
    pub fn new(goal: impl Into<String>) -> Self {
        Self {
            id: JobId::new(),
            goal: goal.into(),
            status: JobStatus::Pending,
            result: None,
            created_at: now_secs(),
            completed_at: None,
        }
    }
}

/// Errors that can occur in the executor.
#[derive(Debug, Error)]
pub enum ExecutorError {
    #[error("max concurrent jobs ({0}) reached")]
    MaxConcurrentJobs(usize),
    #[error("job not found: {0}")]
    JobNotFound(JobId),
    #[error("job already cancelled: {0}")]
    AlreadyCancelled(JobId),
    #[error("executor is shut down")]
    ShutDown,
}

/// Inner state of a running job, not serialized.
struct JobInner {
    job: BackgroundJob,
    handle: Option<JoinHandle<()>>,
    cancelled: bool,
}

/// Executes background jobs with a concurrency limit.
pub struct Executor {
    max_concurrent: usize,
    semaphore: Arc<Semaphore>,
    jobs: Arc<RwLock<HashMap<JobId, JobInner>>>,
    shutdown: Arc<RwLock<bool>>,
}

impl Executor {
    /// Create a new executor with the given concurrency limit.
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            max_concurrent,
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            jobs: Arc::new(RwLock::new(HashMap::new())),
            shutdown: Arc::new(RwLock::new(false)),
        }
    }

    /// Spawn a new background job.
    ///
    /// The `goal` is a user-provided description of what to do. The `runner`
    /// is the async closure that performs the actual work. The executor
    /// manages concurrency, status tracking, and cancellation.
    pub async fn spawn<F, Fut>(
        &self,
        goal: impl Into<String>,
        runner: F,
    ) -> Result<JobId, ExecutorError>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = Result<String, String>> + Send + 'static,
    {
        if *self.shutdown.read().await {
            return Err(ExecutorError::ShutDown);
        }

        let job = BackgroundJob::new(goal);
        let id = job.id;
        let goal_clone = job.goal.clone();

        let jobs = self.jobs.clone();

        // Acquire a permit — this enforces the concurrency limit
        let permit = self
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| ExecutorError::ShutDown)?;

        let handle = tokio::spawn(async move {
            let _permit = permit; // Hold permit until job completes

            // Mark as running
            {
                let mut jobs_guard = jobs.write().await;
                if let Some(inner) = jobs_guard.get_mut(&id) {
                    if inner.cancelled {
                        return;
                    }
                    inner.job.status = JobStatus::Running;
                }
            }

            info!(job_id = %id, goal = %goal_clone, "background job started");

            // Execute the job
            let result = runner().await;

            // Update final status
            {
                let mut jobs_guard = jobs.write().await;
                if let Some(inner) = jobs_guard.get_mut(&id) {
                    if inner.cancelled {
                        return;
                    }
                    let now = now_secs();
                    match result {
                        Ok(output) => {
                            inner.job.status = JobStatus::Completed(output.clone());
                            inner.job.result = Some(output);
                            inner.job.completed_at = Some(now);
                            info!(job_id = %id, "background job completed");
                        }
                        Err(err) => {
                            inner.job.status = JobStatus::Failed(err.clone());
                            inner.job.completed_at = Some(now);
                            warn!(job_id = %id, error = %err, "background job failed");
                        }
                    }
                }
            }

            debug!(job_id = %id, "background job finished");
        });

        // Store the job
        {
            let mut jobs_guard = self.jobs.write().await;
            jobs_guard.insert(
                id,
                JobInner {
                    job,
                    handle: Some(handle),
                    cancelled: false,
                },
            );
        }

        Ok(id)
    }

    /// Get the current status of a job.
    pub async fn status(&self, id: JobId) -> Result<JobStatus, ExecutorError> {
        let jobs = self.jobs.read().await;
        jobs.get(&id)
            .map(|inner| inner.job.status.clone())
            .ok_or(ExecutorError::JobNotFound(id))
    }

    /// Get the full job record.
    pub async fn job(&self, id: JobId) -> Result<BackgroundJob, ExecutorError> {
        let jobs = self.jobs.read().await;
        jobs.get(&id)
            .map(|inner| inner.job.clone())
            .ok_or(ExecutorError::JobNotFound(id))
    }

    /// List all jobs.
    pub async fn list(&self) -> Vec<BackgroundJob> {
        let jobs = self.jobs.read().await;
        jobs.values().map(|inner| inner.job.clone()).collect()
    }

    /// List only active (non-terminal) jobs.
    pub async fn list_active(&self) -> Vec<BackgroundJob> {
        let jobs = self.jobs.read().await;
        jobs.values()
            .filter(|inner| inner.job.status.is_active())
            .map(|inner| inner.job.clone())
            .collect()
    }

    /// Cancel a job. If it's running, the cancellation flag is set and the
    /// task will stop at its next opportunity.
    pub async fn cancel(&self, id: JobId) -> Result<JobStatus, ExecutorError> {
        let mut jobs = self.jobs.write().await;
        let inner = jobs
            .get_mut(&id)
            .ok_or(ExecutorError::JobNotFound(id))?;

        if inner.job.status.is_terminal() {
            return Err(ExecutorError::AlreadyCancelled(id));
        }

        inner.cancelled = true;
        inner.job.status = JobStatus::Cancelled;
        inner.job.completed_at = Some(now_secs());

        if let Some(handle) = inner.handle.take() {
            handle.abort();
        }

        info!(job_id = %id, "background job cancelled");
        Ok(inner.job.status.clone())
    }

    /// Clean up completed/failed/cancelled jobs from the internal map.
    /// Returns the number of jobs removed.
    pub async fn cleanup(&self) -> usize {
        let mut jobs = self.jobs.write().await;
        let before = jobs.len();
        jobs.retain(|_, inner| inner.job.status.is_active());
        before - jobs.len()
    }

    /// Shut down the executor, cancelling all active jobs.
    pub async fn shutdown(&self) {
        let mut shutdown = self.shutdown.write().await;
        *shutdown = true;
        drop(shutdown);

        let active: Vec<JobId> = {
            let jobs = self.jobs.read().await;
            jobs.values()
                .filter(|inner| inner.job.status.is_active())
                .map(|inner| inner.job.id)
                .collect()
        };

        for id in active {
            let _ = self.cancel(id).await;
        }

        info!("executor shut down");
    }

    /// Returns the number of currently active jobs.
    pub async fn active_count(&self) -> usize {
        let jobs = self.jobs.read().await;
        jobs.values()
            .filter(|inner| inner.job.status.is_active())
            .count()
    }

    /// Returns the configured maximum concurrency.
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }
}

impl Default for Executor {
    fn default() -> Self {
        Self::new(4)
    }
}

/// Returns the current time in seconds since the UNIX epoch.
pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn spawn_and_complete() {
        let exec = Executor::new(2);
        let id = exec
            .spawn("test goal", || async {
                tokio::time::sleep(Duration::from_millis(10)).await;
                Ok("done".to_string())
            })
            .await
            .unwrap();

        // Wait for completion
        for _ in 0..100 {
            let status = exec.status(id).await.unwrap();
            if status.is_terminal() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let status = exec.status(id).await.unwrap();
        assert_eq!(status, JobStatus::Completed("done".to_string()));
    }

    #[tokio::test]
    async fn spawn_and_fail() {
        let exec = Executor::new(2);
        let id = exec
            .spawn("failing goal", || async {
                tokio::time::sleep(Duration::from_millis(10)).await;
                Err("something went wrong".to_string())
            })
            .await
            .unwrap();

        for _ in 0..100 {
            let status = exec.status(id).await.unwrap();
            if status.is_terminal() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let status = exec.status(id).await.unwrap();
        assert_eq!(status, JobStatus::Failed("something went wrong".to_string()));
    }

    #[tokio::test]
    async fn cancel_pending_job() {
        let exec = Executor::new(1);
        // Fill the concurrency slot
        let _blocker = exec
            .spawn("blocker", || async {
                tokio::time::sleep(Duration::from_secs(10)).await;
                Ok("done".to_string())
            })
            .await
            .unwrap();

        // This one will be pending
        let id = exec
            .spawn("to cancel", || async {
                tokio::time::sleep(Duration::from_millis(50)).await;
                Ok("done".to_string())
            })
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_millis(20)).await;
        let status = exec.cancel(id).await.unwrap();
        assert_eq!(status, JobStatus::Cancelled);
    }

    #[tokio::test]
    async fn max_concurrent_limit() {
        let exec = Executor::new(2);
        let mut ids = Vec::new();

        for i in 0..5 {
            let id = exec
                .spawn(format!("job {i}"), || async {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    Ok("done".to_string())
                })
                .await
                .unwrap();
            ids.push(id);
        }

        tokio::time::sleep(Duration::from_millis(20)).await;
        let active = exec.active_count().await;
        assert!(active <= 2, "active count {active} exceeds limit 2");
    }

    #[tokio::test]
    async fn job_not_found() {
        let exec = Executor::new(2);
        let id = JobId::new();
        assert!(matches!(
            exec.status(id).await,
            Err(ExecutorError::JobNotFound(_))
        ));
    }

    #[tokio::test]
    async fn cleanup_removes_terminal() {
        let exec = Executor::new(4);
        let id = exec
            .spawn("quick", || async { Ok("done".to_string()) })
            .await
            .unwrap();

        for _ in 0..100 {
            if exec.status(id).await.unwrap().is_terminal() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let removed = exec.cleanup().await;
        assert_eq!(removed, 1);
        assert!(exec.list().await.is_empty());
    }

    #[tokio::test]
    async fn shutdown_cancels_active() {
        let exec = Executor::new(4);
        let id = exec
            .spawn("long running", || async {
                tokio::time::sleep(Duration::from_secs(30)).await;
                Ok("done".to_string())
            })
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_millis(20)).await;
        exec.shutdown().await;

        let status = exec.status(id).await.unwrap();
        assert_eq!(status, JobStatus::Cancelled);
    }

    #[tokio::test]
    async fn list_returns_all_jobs() {
        let exec = Executor::new(4);
        let id1 = exec
            .spawn("job 1", || async { Ok("done".to_string()) })
            .await
            .unwrap();
        let id2 = exec
            .spawn("job 2", || async { Ok("done".to_string()) })
            .await
            .unwrap();

        let jobs = exec.list().await;
        assert_eq!(jobs.len(), 2);
        let ids: Vec<_> = jobs.iter().map(|j| j.id).collect();
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
    }

    #[tokio::test]
    async fn job_metadata_is_correct() {
        let exec = Executor::new(4);
        let id = exec
            .spawn("my goal", || async { Ok("result".to_string()) })
            .await
            .unwrap();

        let job = exec.job(id).await.unwrap();
        assert_eq!(job.goal, "my goal");
        assert_eq!(job.id, id);
        assert!(job.created_at > 0);
    }
}
