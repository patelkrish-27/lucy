//! The fast lane must leave a `classification` record per call, so the
//! two-speed claim is measurable from `model-calls.jsonl` after the fact rather
//! than only asserted.
//!
//! This lives in its own test binary because `LUCY_MODEL_LOG` is process-global:
//! sharing one file with other tests would mix their records in and make the
//! per-call count meaningless.

use async_trait::async_trait;
use lucy_core::{AgentEvent, InterruptSignal, Tool, ToolContext};
use lucy_runtime::agent_loop::{AgentBudget, AgentDeps, run_agent_loop_with};
use lucy_tools::ToolRegistry;
use serde_json::{Value, json};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// One scripted screen, held forever: every snapshot reports the same labels and
/// the same satisfied set, so a run is fully determined.
struct Held {
    seen: Mutex<Vec<String>>,
    advances: AtomicUsize,
}

impl Held {
    fn snapshot(&self) -> Value {
        self.seen
            .lock()
            .expect("held poisoned")
            .push("hint_snapshot".into());
        self.advances.fetch_add(1, Ordering::SeqCst);
        json!({
            "count": 2,
            "via": "decider",
            "hints": [{"label": "Search"}, {"label": "Queue"}],
        })
    }
    fn record(&self, tool: &str) {
        self.seen.lock().expect("held poisoned").push(tool.into());
    }
    fn calls(&self, tool: &str) -> usize {
        self.seen
            .lock()
            .expect("held poisoned")
            .iter()
            .filter(|t| t.as_str() == tool)
            .count()
    }
}

enum Kind {
    Snapshot(Arc<Held>),
    Satisfied,
    Act,
}

struct Fake {
    name: &'static str,
    kind: Kind,
    held: Option<Arc<Held>>,
}

#[async_trait]
impl Tool for Fake {
    fn name(&self) -> &str {
        self.name
    }
    fn description(&self) -> &str {
        "fake"
    }
    fn parameters_schema(&self) -> Value {
        json!({})
    }
    fn requires_approval(&self) -> bool {
        false
    }
    async fn execute(&self, _input: Value, _ctx: ToolContext) -> anyhow::Result<Value> {
        Ok(match &self.kind {
            Kind::Snapshot(held) => {
                held.record(self.name);
                held.snapshot()
            }
            Kind::Satisfied => {
                self.held.as_ref().map(|h| h.record(self.name));
                json!({"success": true, "result": "success"})
            }
            Kind::Act => {
                self.held.as_ref().map(|h| h.record(self.name));
                json!({"success": true, "tier": "decider", "label": "target"})
            }
        })
    }
}

fn registry(held: Arc<Held>) -> ToolRegistry {
    let mut r = ToolRegistry::new();
    r.register_arc(Arc::new(Fake {
        name: "hint_snapshot",
        kind: Kind::Snapshot(held.clone()),
        held: Some(held.clone()),
    }));
    r.register_arc(Arc::new(Fake {
        name: "verify",
        kind: Kind::Satisfied,
        held: Some(held.clone()),
    }));
    r.register_arc(Arc::new(Fake {
        name: "wait_until",
        kind: Kind::Satisfied,
        held: Some(held.clone()),
    }));
    r.register_arc(Arc::new(Fake {
        name: "hint_act",
        kind: Kind::Act,
        held: Some(held.clone()),
    }));
    r
}

fn default_config() -> &'static lucy_config::LucyConfig {
    static CONFIG: OnceLock<lucy_config::LucyConfig> = OnceLock::new();
    CONFIG.get_or_init(lucy_config::LucyConfig::default)
}

#[tokio::test]
async fn every_fast_lane_call_is_logged_as_a_tool_record() {
    let path = std::env::temp_dir().join(format!("lucy-fast-lane-{}.jsonl", std::process::id()));
    let _ = std::fs::remove_file(&path);
    unsafe { std::env::set_var("LUCY_MODEL_LOG", &path) };

    let held = Arc::new(Held {
        seen: Mutex::new(Vec::new()),
        advances: AtomicUsize::new(0),
    });
    let registry = registry(held.clone());
    let stub = lucy_agent::testing::StubProvider::new().push_json(json!({"objectives":[
        {"description":"One","success_check":"one is done","suggested_action":"click one"},
        {"description":"Two","success_check":"two is done","suggested_action":"click two"}
    ]}));
    let deps = AgentDeps {
        knowledge: None,
        page_observations: Default::default(),
        provider: stub.as_ref(),
        registry: &registry,
        budget: AgentBudget::default(),
        interrupt: InterruptSignal::new(),
        events: Some(tokio::sync::mpsc::unbounded_channel::<AgentEvent>().0),
        model_key: "stub/model".into(),
        approval: None,
        cdp_probe: Some(lucy_runtime::fast_perception::probe_fn(|| async { true })),
        destructive_tools: Default::default(),
        config: default_config(),
    };

    let out = run_agent_loop_with(&deps, "log the fast lane")
        .await
        .expect("loop runs");
    assert!(out.complete, "{out}");

    unsafe { std::env::remove_var("LUCY_MODEL_LOG") };
    let text = std::fs::read_to_string(&path).expect("model-call log written");
    let records: Vec<Value> = text
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();

    assert_eq!(
        records.len(),
        out.stats.fast_calls as usize,
        "exactly one record per fast call (the slow plan call is logged by the \
         provider, not here, so this file holds only the fast lane)"
    );
    assert!(out.stats.fast_calls >= 4, "{out}");
    for rec in &records {
        assert_eq!(
            rec["kind"], "tool",
            "a fast-lane call is a tool execution, not a model call — the log has \
             to keep the two apart to be readable: {rec}"
        );
        assert_eq!(rec["purpose"], "fast_lane", "{rec}");
        assert!(rec["operation"].is_string(), "{rec}");
        assert!(rec["model"].is_string(), "{rec}");
    }
    let ops: Vec<&str> = records
        .iter()
        .filter_map(|r| r["operation"].as_str())
        .collect();
    for tool in ["hint_snapshot", "hint_act", "verify"] {
        assert!(ops.contains(&tool), "{tool} missing from the log: {ops:?}");
    }
    assert_eq!(
        held.calls("hint_act"),
        2,
        "two objectives, two interactions"
    );
    assert_eq!(
        held.advances.load(Ordering::SeqCst) as usize,
        ops.iter().filter(|o| **o == "hint_snapshot").count(),
        "every snapshot is logged"
    );
    let _ = std::fs::remove_file(&path);
}
