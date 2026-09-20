/// v1 pipeline prompts — superseded by the v2 harness below, retained for the
/// legacy `planner` module's tests. See `docs/lucy-harness-architecture.md`.
#[allow(dead_code)]
pub const TRIAGE: &str = include_str!("../../../prompts/triage.md");
#[allow(dead_code)]
pub const CONTROLLER: &str = include_str!("../../../prompts/controller.md");
#[allow(dead_code)]
pub const TOOL_SELECTOR: &str = include_str!("../../../prompts/tool_selector.md");
// v2 harness (§11): merged Router+Planner, scoped Recovery, terminal Verifier.
pub const ROUTER_PLANNER: &str = include_str!("../../../prompts/router_planner.md");
pub const RECOVERY: &str = include_str!("../../../prompts/recovery.md");
pub const VERIFIER: &str = include_str!("../../../prompts/verifier.md");
