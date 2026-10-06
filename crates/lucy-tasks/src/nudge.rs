//! Nudge scheduler: periodically checks for tasks that need attention.

use crate::{Task, TaskStore};
use anyhow::Result;
use std::{sync::Arc, time::Duration};
use tokio::sync::mpsc::UnboundedSender;

/// The type of nudge to deliver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NudgeType {
    /// Task is due within a short window.
    DueSoon,
    /// Task is past its due date.
    Overdue,
    /// Task has been in-progress for too long without completion.
    Stalled,
}

/// A nudge: a task that needs attention, with a human-readable message.
#[derive(Debug, Clone)]
pub struct Nudge {
    pub task: Task,
    pub nudge_type: NudgeType,
    pub message: String,
}

/// Schedules and generates nudges from a [`TaskStore`].
pub struct TaskNudge {
    store: Arc<TaskStore>,
    interval: Duration,
}

impl TaskNudge {
    /// Create a new nudge scheduler.
    pub fn new(store: Arc<TaskStore>, interval: Duration) -> Self {
        Self { store, interval }
    }

    /// Get the nudge interval.
    pub fn interval(&self) -> Duration {
        self.interval
    }

    /// Get all tasks that currently need nudging.
    pub async fn nudges(&self) -> Result<Vec<Nudge>> {
        let mut nudges = Vec::new();
        let now = chrono::Utc::now().timestamp() as u64;

        // Overdue tasks
        let overdue = self.store.overdue().await?;
        for task in overdue {
            let message = if let Some(due_at) = task.due_at {
                let overdue_by = now.saturating_sub(due_at);
                let hours = overdue_by / 3600;
                if hours > 0 {
                    format!(
                        "Task \"{}\" is overdue by {} hour{}",
                        task.title,
                        hours,
                        if hours == 1 { "" } else { "s" }
                    )
                } else {
                    let mins = overdue_by / 60;
                    format!(
                        "Task \"{}\" is overdue by {} minute{}",
                        task.title,
                        mins.max(1),
                        if mins <= 1 { "" } else { "s" }
                    )
                }
            } else {
                format!("Task \"{}\" is overdue", task.title)
            };
            nudges.push(Nudge {
                task,
                nudge_type: NudgeType::Overdue,
                message,
            });
        }

        // Due soon (within 1 hour)
        let due_soon = self.store.due_soon(Duration::from_secs(3600)).await?;
        for task in due_soon {
            // Skip if already overdue (don't double-nudge)
            if task.due_at.is_some_and(|d| d < now) {
                continue;
            }
            let message = if let Some(due_at) = task.due_at {
                let remaining = due_at.saturating_sub(now);
                let mins = remaining / 60;
                if mins > 0 {
                    format!(
                        "Task \"{}\" is due in {} minute{}",
                        task.title,
                        mins,
                        if mins == 1 { "" } else { "s" }
                    )
                } else {
                    format!("Task \"{}\" is due very soon", task.title)
                }
            } else {
                format!("Task \"{}\" is due soon", task.title)
            };
            nudges.push(Nudge {
                task,
                nudge_type: NudgeType::DueSoon,
                message,
            });
        }

        // Stalled tasks (in progress for more than 24 hours)
        let all = self.store.list().await?;
        for task in all {
            if task.status != crate::TaskStatus::InProgress {
                continue;
            }
            let created_at = task.created_at;
            let stalled_for = now.saturating_sub(created_at);
            if stalled_for > 86400 {
                let days = stalled_for / 86400;
                let message = format!(
                    "Task \"{}\" has been in progress for {} day{}",
                    task.title,
                    days,
                    if days == 1 { "" } else { "s" }
                );
                nudges.push(Nudge {
                    task,
                    nudge_type: NudgeType::Stalled,
                    message,
                });
            }
        }

        Ok(nudges)
    }

    /// Run the nudge loop, sending nudges to the given channel.
    /// Checks immediately, then every `interval`.
    pub async fn run(&self, tx: UnboundedSender<Nudge>) {
        let mut ticker = tokio::time::interval(self.interval);
        // Skip the first immediate tick — we check manually below
        ticker.tick().await;

        loop {
            match self.nudges().await {
                Ok(nudges) => {
                    for nudge in nudges {
                        if tx.send(nudge).is_err() {
                            tracing::debug!("nudge channel closed; stopping nudge loop");
                            return;
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "failed to compute nudges");
                }
            }
            ticker.tick().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nudge_type_is_distinct() {
        let a = NudgeType::DueSoon;
        let b = NudgeType::Overdue;
        let c = NudgeType::Stalled;
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }
}
