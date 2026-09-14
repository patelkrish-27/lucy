//! Lucy runtime: builds the agent, tools and session store, then runs every
//! user command through the hierarchical loop in [`planner`].
//!
//! Module layout:
//! - `planner` — main-model triage, cheap-model command compiler, closed-loop
//!   execution with verification and recovery.
//! - `prompts` — embedded, reviewable model behavior contracts.
//! - `sessions` — opencode-style multi-session CRUD, compaction, trimming.
//! - `execution` — dependency-safe execution waves (scheduler foundation).

mod execution;
mod planner;
mod prompts;
mod sessions;

pub use execution::{build_waves, has_dependency_cycle, parallel_candidate, ExecutionWave};
pub use planner::SubTask;
pub use sessions::trim_history;

use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::PathBuf,
};
