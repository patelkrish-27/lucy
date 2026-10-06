use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{HashMap, HashSet};

use crate::browser_cdp::{BrowserAction, BrowserPageSnapshot};
use crate::client::SystemOneClient;
use crate::types::Question;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalRequirement {
    pub what: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalPlan {
    pub requirements: Vec<GoalRequirement>,
    pub open: Option<String>,
    pub finish: String,
}

#[derive(Debug, Clone)]
pub struct ObservedElement {
    pub node: i64,
    pub index: String,
    pub role: String,
    pub label: String,
    pub value: String,
    pub current: String,
    pub checked: Option<String>,
    pub expanded: Option<String>,
    pub hint: String,
    pub actions: HashMap<String, BrowserAction>,
    pub options: Vec<BrowserAction>,
}

#[derive(Debug, Clone)]
pub struct PolicyStep {
    pub choice: String,
    pub kind: String,
    pub req: Option<usize>,
    pub node: Option<i64>,
    pub url: String,
    pub before: HashSet<i64>,
    pub label: Option<String>,
}

#[derive(Debug, Clone)]
pub enum PolicyOutcome {
    Action {
        action: BrowserAction,
        text_to_type: Option<String>,
        description: String,
    },
    Done {
        message: String,
    },
    Blocked {
        reason: String,
    },
}

pub struct BrowserPolicy {
    pub goal: String,
    pub plan: Option<GoalPlan>,
    pub fields: HashMap<usize, i64>,
    pub met: HashSet<usize>,
    pub skipped: HashSet<usize>,
    pub attempts: HashMap<usize, usize>,
    pub last: Option<PolicyStep>,
    pub edit_url: Option<String>,
    pub submitted: bool,
    pub waits: usize,
    pub tried: HashMap<String, usize>,
    pub typed: bool,
    pub acted: Option<String>,
    pub failed: HashMap<i64, usize>,
    pub search_added: bool,
}

impl BrowserPolicy {
    pub fn new(goal: impl Into<String>) -> Self {
        Self {
            goal: goal.into(),
            plan: None,
            fields: HashMap::new(),
            met: HashSet::new(),
            skipped: HashSet::new(),
            attempts: HashMap::new(),
            last: None,
            edit_url: None,
            submitted: false,
            waits: 0,
            tried: HashMap::new(),
            typed: false,
            acted: None,
            failed: HashMap::new(),
            search_added: false,
        }
    }

    pub fn set_plan(&mut self, plan: GoalPlan) {
        self.plan = Some(plan);
    }

    /// Convert page actions into indexed ObservedElements.
    pub fn observed(&self, page: &BrowserPageSnapshot) -> Vec<ObservedElement> {
        let mut elements: Vec<ObservedElement> = Vec::new();
        let mut node_map: HashMap<i64, usize> = HashMap::new();

        for action in &page.actions {
            if !["click", "fill", "select"].contains(&action.kind.as_str()) {
                continue;
            }
            let node = match action.node {
                Some(n) => n,
                None => continue,
            };

            let entry_idx = match node_map.get(&node) {
                Some(&i) => i,
                None => {
                    let idx_str = (elements.len() + 1).to_string();
                    let role = action.role.clone().unwrap_or_default();
                    let label = action
                        .label
                        .split(" → ")
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    let val = action
                        .current_value
                        .clone()
                        .or_else(|| action.value.clone())
                        .unwrap_or_default();
                    let hint = action.hint.clone().unwrap_or_default();
                    let checked = action.checked.clone();
                    let expanded = action.expanded.clone();

                    let el = ObservedElement {
                        node,
                        index: idx_str,
                        role,
                        label,
                        value: val.clone(),
                        current: val,
                        checked,
                        expanded,
                        hint,
                        actions: HashMap::new(),
                        options: Vec::new(),
                    };
                    let i = elements.len();
                    elements.push(el);
                    node_map.insert(node, i);
                    i
                }
            };

            let el = &mut elements[entry_idx];
            if action.kind == "select" {
                el.options.push(action.clone());
            } else {
                el.actions
                    .entry(action.kind.clone())
                    .or_insert_with(|| action.clone());
            }
        }

        // Set `.current` semantic values
        for el in &mut elements {
            if is_toggle(&el.role) {
                el.current = if el.checked.as_deref() == Some("true") {
                    "checked".into()
                } else {
                    "unchecked".into()
                };
            } else if !el.value.is_empty() || (is_field(el) && el.role != "button") {
                el.current = el.value.clone();
            } else {
                el.current = el.label.clone();
            }
        }

        elements
    }

    /// Inventory of scroll actions retained from snapshot (first-class).
    /// `observed()` drops scroll/wait kinds; this keeps scroll BrowserActions
    /// so `step()` can emit `scroll_down` when a below-fold target is needed.
    pub fn scroll_inventory(&self, page: &BrowserPageSnapshot) -> Vec<BrowserAction> {
        page.actions
            .iter()
            .filter(|a| a.kind == "scroll")
            .cloned()
            .collect()
    }

    /// Convenience: scroll_down action if available.
    pub fn scroll_down(&self, page: &BrowserPageSnapshot) -> Option<BrowserAction> {
        self.scroll_inventory(page)
            .into_iter()
            .find(|a| a.delta.unwrap_or(0) > 0)
    }

    /// Execute one decision step via Laya System-1 — speculative fan-out.
    ///
    /// Sends ONE `predict()` with multiple Question heads (`operation` +
    /// `click_target` + `type_target` + `select_target` (+ `kind` for finish)).
    /// Only the operation-relevant target head is used; unused heads are
    /// ignored. Keeps `head_max_len` via 28-char truncation and `shortlist`
    /// limit 15 (single chunk). Fallback paths preserve original `pick_element`
    /// / `classify_page_state` semantics but are not taken in the fused fast
    /// path. Guarantees ≤2 predicts per step (1 fused + optional classify
    /// collapsed into same call when possible).
    pub async fn step(
        &mut self,
        page: &BrowserPageSnapshot,
        client: &SystemOneClient,
    ) -> Result<PolicyOutcome> {
        let elements = self.observed(page);
        let scroll_down = self.scroll_down(page);
        let plan = self
            .plan
            .as_ref()
            .ok_or_else(|| anyhow!("goal plan not initialized"))?;
        let reqs = plan.requirements.clone();
        let item = plan.open.clone();
        let finish = plan.finish.clone();

        let clickable: Vec<ObservedElement> = elements
            .iter()
            .filter(|e| {
                e.actions.contains_key("click") && *self.failed.get(&e.node).unwrap_or(&0) < 2
            })
            .cloned()
            .collect();

        let by_node: HashMap<i64, ObservedElement> =
            elements.iter().map(|e| (e.node, e.clone())).collect();

        // 1. Autocomplete suggestion fan-out: keep single fused check before
        // requirement loop. Still one predict, but now via speculative helper
        // when possible (relevance filter already narrows).
        if let Some(last) = &self.last {
            if (last.kind == "fill" || last.kind == "open") && last.req.is_some() {
                let req_idx = last.req.unwrap();
                if req_idx < reqs.len() {
                    let r = &reqs[req_idx];
                    let new_items: Vec<ObservedElement> = clickable
                        .iter()
                        .filter(|e| !last.before.contains(&e.node) && relevance(e, &r.value) > 0)
                        .cloned()
                        .collect();

                    if !new_items.is_empty() {
                        // Date exact-match bypass (no model call) preserved.
                        let date_opt = month_day(&r.value);
                        let mut exact = None;
                        if let Some(d) = &date_opt {
                            let dated: Vec<&ObservedElement> = new_items
                                .iter()
                                .filter(|e| month_day(&e.label) == Some(d.clone()))
                                .collect();
                            if dated.len() == 1 {
                                exact = Some(dated[0].clone());
                            }
                        }
                        if let Some(p) = exact {
                            if let Some(act) = p.actions.get("click") {
                                self.record_pending(
                                    "pick",
                                    req_idx,
                                    Some(p.node),
                                    &page.url,
                                    &elements,
                                    Some(p.label.clone()),
                                );
                                return Ok(PolicyOutcome::Action {
                                    action: act.clone(),
                                    text_to_type: None,
                                    description: format!(
                                        "Picked suggestion [{}] {}",
                                        p.index, p.label
                                    ),
                                });
                            }
                        } else {
                            // Single speculative predict for option pick (fan-out: operation is implicit click)
                            let picked = self
                                .pick_element(
                                    client,
                                    "option",
                                    &new_items,
                                    &format!("Requirement: {} = {}", r.what, r.value),
                                    &format!("Which option sets {} to {}?", r.what, r.value),
                                    &r.value,
                                    true,
                                )
                                .await?;

                            if let Some(p) = picked {
                                if let Some(act) = p.actions.get("click") {
                                    self.record_pending(
                                        "pick",
                                        req_idx,
                                        Some(p.node),
                                        &page.url,
                                        &elements,
                                        Some(p.label.clone()),
                                    );
                                    return Ok(PolicyOutcome::Action {
                                        action: act.clone(),
                                        text_to_type: None,
                                        description: format!(
                                            "Picked suggestion [{}] {}",
                                            p.index, p.label
                                        ),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. Iterate requirements in goal order — speculative fan-out for the
        // first pending requirement. All target shortlists are prepared and
        // scored in ONE forward pass.
        for (i, r) in reqs.iter().enumerate() {
            if self.met.contains(&i) || self.skipped.contains(&i) {
                continue;
            }
            if *self.attempts.get(&i).unwrap_or(&0) >= 3 {
                self.skipped.insert(i);
                continue;
            }

            let field_node = self.fields.get(&i).copied();
            let elem_opt = field_node.and_then(|node| by_node.get(&node));

            if elem_opt.is_none() {
                // Check if already displayed on page (deterministic, no model)
                let already_satisfied = elements.iter().any(|c| {
                    is_field(c)
                        && !is_toggle(&c.role)
                        && month_day(&c.label).is_none()
                        && !words(&c.label).is_disjoint(&words(&r.what))
                        && !words(&c.label).is_disjoint(&words(&r.value))
                });
                if already_satisfied {
                    self.met.insert(i);
                    continue;
                }

                // Speculative fan-out for this requirement: one predict with
                // operation + click/type/select heads.
                let about = format!("{} {}", r.what, r.value);
                let candidates: Vec<ObservedElement> = clickable
                    .iter()
                    .filter(|c| !is_toggle(&c.role) || !words(&c.label).is_disjoint(&words(&about)))
                    .cloned()
                    .collect();

                // Fast path: shortlist and operation dispatch in one call.
                let outcome = self
                    .speculative_requirement_step(
                        client,
                        page,
                        &elements,
                        &clickable,
                        &candidates,
                        i,
                        r,
                        &about,
                        &finish,
                        scroll_down.clone(),
                    )
                    .await?;
                match outcome {
                    Some(o) => return Ok(o),
                    None => continue, // picked none -> try next requirement via wait fallback inside helper
                }
            } else if let Some(e) = elem_opt {
                if let Some(fill_act) = e.actions.get("fill") {
                    self.record_pending(
                        "fill",
                        i,
                        Some(e.node),
                        &page.url,
                        &elements,
                        Some(e.label.clone()),
                    );
                    self.typed = true;
                    self.met.insert(i);
                    return Ok(PolicyOutcome::Action {
                        action: fill_act.clone(),
                        text_to_type: Some(r.value.clone()),
                        description: format!(
                            "Filling '{}' into field '{}' [{}]",
                            r.value, e.label, e.index
                        ),
                    });
                }
                if !e.options.is_empty() {
                    let matching_opt = e
                        .options
                        .iter()
                        .find(|o| fold(&o.label).contains(&fold(&r.value)))
                        .or_else(|| e.options.first());
                    if let Some(opt) = matching_opt {
                        self.record_pending(
                            "select",
                            i,
                            Some(e.node),
                            &page.url,
                            &elements,
                            Some(e.label.clone()),
                        );
                        self.met.insert(i);
                        return Ok(PolicyOutcome::Action {
                            action: opt.clone(),
                            text_to_type: None,
                            description: format!(
                                "Selected option '{}' in [{}]",
                                opt.label, e.index
                            ),
                        });
                    }
                }
                if let Some(click_act) = e.actions.get("click") {
                    self.record_pending(
                        "click",
                        i,
                        Some(e.node),
                        &page.url,
                        &elements,
                        Some(e.label.clone()),
                    );
                    self.met.insert(i);
                    return Ok(PolicyOutcome::Action {
                        action: click_act.clone(),
                        text_to_type: None,
                        description: format!("Clicked [{}] {}", e.index, e.label),
                    });
                }
            }
        }

        // 3. All requirements satisfied -> speculative post-requirements fan-out:
        // one predict with operation + click_target + kind (classify) heads.
        let navigated = self
            .edit_url
            .as_ref()
            .map(|u| location(u) != location(&page.url))
            .unwrap_or(false);

        if let Some(item_name) = &item {
            if titled(&page.title, item_name) {
                return Ok(PolicyOutcome::Done {
                    message: format!("Opened item '{}' ({})", item_name, page.title),
                });
            }
        }

        // Speculative post-requirements dispatch (finish / submit / open / next)
        // This collapses classify_page_state + pick_element (submit/item/next)
        // into a single forward pass.
        if let Some(outcome) = self
            .speculative_post_step(
                client,
                page,
                &elements,
                &clickable,
                scroll_down.clone(),
                navigated,
            )
            .await?
        {
            return Ok(outcome);
        }

        Ok(PolicyOutcome::Blocked {
            reason: "No further interactive elements on page advance the task".into(),
        })
    }

    /// Speculative fan-out for a single pending requirement.
    ///
    /// Builds ONE `predict` with heads: `operation` (click|fill|select|scroll|wait|done|blocked)
    /// + `click_target` + `type_target` + `select_target`. Only the operation-relevant
    /// head is used. Returns `None` only when caller should try next requirement;
    /// otherwise returns `Some(PolicyOutcome)` (including Wait/Scroll fallback).
    async fn speculative_requirement_step(
        &mut self,
        client: &SystemOneClient,
        page: &BrowserPageSnapshot,
        elements: &[ObservedElement],
        _clickable: &[ObservedElement],
        candidates: &[ObservedElement],
        req_idx: usize,
        req: &GoalRequirement,
        about: &str,
        finish: &str,
        scroll_down: Option<BrowserAction>,
    ) -> Result<Option<PolicyOutcome>> {
        if candidates.is_empty() {
            if let Some(scroll) = scroll_down.clone() {
                if page.omitted_actions > 0 || needs_scroll_for_target(&req.value, elements, page) {
                    return Ok(Some(PolicyOutcome::Action {
                        action: scroll,
                        text_to_type: None,
                        description: format!(
                            "Scrolling to find field '{}' = '{}' (omitted={})",
                            req.what, req.value, page.omitted_actions
                        ),
                    }));
                }
            }
            *self.attempts.entry(req_idx).or_default() += 1;
            return Ok(Some(PolicyOutcome::Action {
                action: BrowserAction {
                    id: "wait".into(),
                    node: None,
                    kind: "wait".into(),
                    role: None,
                    label: "Wait for page elements to settle".into(),
                    value: None,
                    current_value: None,
                    checked: None,
                    expanded: None,
                    hint: None,
                    rect: None,
                    delta: None,
                },
                text_to_type: None,
                description: format!("Waiting for field '{}' to appear", req.what),
            }));
        }

        // Date exact-match bypass (no model call) — preserve pick_element optimization
        let date_opt = month_day(about);
        if let Some(d) = date_opt {
            let dated: Vec<&ObservedElement> = candidates
                .iter()
                .filter(|e| month_day(&e.label) == Some(d.clone()))
                .collect();
            if dated.len() == 1 {
                let p = dated[0].clone();
                self.fields.insert(req_idx, p.node);
                if let Some(fill_act) = p.actions.get("fill") {
                    self.record_pending(
                        "fill",
                        req_idx,
                        Some(p.node),
                        &page.url,
                        elements,
                        Some(p.label.clone()),
                    );
                    self.typed = true;
                    self.met.insert(req_idx);
                    return Ok(Some(PolicyOutcome::Action {
                        action: fill_act.clone(),
                        text_to_type: Some(req.value.clone()),
                        description: format!(
                            "Filling '{}' into field '{}' [{}]",
                            req.value, p.label, p.index
                        ),
                    }));
                }
                if let Some(click_act) = p.actions.get("click") {
                    self.record_pending(
                        "open",
                        req_idx,
                        Some(p.node),
                        &page.url,
                        elements,
                        Some(p.label.clone()),
                    );
                    return Ok(Some(PolicyOutcome::Action {
                        action: click_act.clone(),
                        text_to_type: None,
                        description: format!("Opening control '{}' [{}]", p.label, p.index),
                    }));
                }
            }
        }

        // Build shortlists for each operation-relevant head (single chunk, limit 15, 28-char trunc)
        let fillable: Vec<ObservedElement> = elements
            .iter()
            .filter(|e| e.actions.contains_key("fill"))
            .cloned()
            .collect();
        let selectable: Vec<ObservedElement> = elements
            .iter()
            .filter(|e| !e.options.is_empty())
            .cloned()
            .collect();

        let click_short = shortlist(candidates, about, 15);
        let fill_short = shortlist(&fillable, about, 15);
        let select_short = shortlist(&selectable, about, 15);

        // If single candidate and no ambiguity, avoid model call for that head — but
        // operation choice still needs a call. So we keep one fused call.
        // Truncation to 28 chars respects head_max_len=192.
        let state = json!({
            "goal": self.goal,
            "finish": finish,
            "requirement": format!("{} = {}", req.what, req.value),
            "page_title": page.title,
            "page_text": summary(&page.text, finish, 200),
        });

        let mut op_criteria = HashMap::new();
        op_criteria.insert(
            "fill".to_string(),
            json!("Fill a text field with the required value"),
        );
        op_criteria.insert(
            "select".to_string(),
            json!("Select an option in a dropdown"),
        );
        op_criteria.insert(
            "click".to_string(),
            json!("Click an element to open picker or button"),
        );
        op_criteria.insert(
            "scroll".to_string(),
            json!("Scroll down to reveal hidden fields"),
        );
        op_criteria.insert("wait".to_string(), json!("Wait for page to load"));
        op_criteria.insert("done".to_string(), json!("Task is complete"));
        op_criteria.insert("blocked".to_string(), json!("Cannot proceed"));

        let mut questions = HashMap::new();
        questions.insert(
            "operation".to_string(),
            Question::choice(
                "choose operation: click|fill|select|scroll|wait|done|blocked",
                op_criteria,
            ),
        );

        if !click_short.is_empty() {
            let mut crit = HashMap::new();
            for e in &click_short {
                let trunc: String = describe(e).chars().take(28).collect();
                crit.insert(e.node.to_string(), json!(trunc));
            }
            crit.insert("none".to_string(), json!("none of these"));
            questions.insert(
                "click_target".to_string(),
                Question::choice("Which element should be clicked?", crit),
            );
        }
        if !fill_short.is_empty() {
            let mut crit = HashMap::new();
            for e in &fill_short {
                let trunc: String = describe(e).chars().take(28).collect();
                crit.insert(e.node.to_string(), json!(trunc));
            }
            crit.insert("none".to_string(), json!("none of these"));
            questions.insert(
                "type_target".to_string(),
                Question::choice("Which field should receive typed text?", crit),
            );
        }
        if !select_short.is_empty() {
            let mut crit = HashMap::new();
            for e in &select_short {
                let trunc: String = describe(e).chars().take(28).collect();
                crit.insert(e.node.to_string(), json!(trunc));
            }
            crit.insert("none".to_string(), json!("none of these"));
            questions.insert(
                "select_target".to_string(),
                Question::choice("Which select should be used?", crit),
            );
        }

        // Single speculative forward pass
        let resp = client.predict_speculative(&state, questions).await?;
        let op = resp
            .answers
            .get("operation")
            .and_then(|a| a.as_choice())
            .unwrap_or("wait");

        match op {
            "fill" => {
                let choice = resp
                    .answers
                    .get("type_target")
                    .and_then(|a| a.as_choice())
                    .unwrap_or("none");
                if choice != "none" {
                    if let Some(p) = fill_short
                        .into_iter()
                        .find(|e| e.node.to_string() == choice)
                    {
                        self.fields.insert(req_idx, p.node);
                        if let Some(fill_act) = p.actions.get("fill") {
                            self.record_pending(
                                "fill",
                                req_idx,
                                Some(p.node),
                                &page.url,
                                elements,
                                Some(p.label.clone()),
                            );
                            self.typed = true;
                            self.met.insert(req_idx);
                            return Ok(Some(PolicyOutcome::Action {
                                action: fill_act.clone(),
                                text_to_type: Some(req.value.clone()),
                                description: format!(
                                    "Filling '{}' into field '{}' [{}]",
                                    req.value, p.label, p.index
                                ),
                            }));
                        }
                    }
                }
                // Fallback to click_target if fill not available or none
                let cchoice = resp
                    .answers
                    .get("click_target")
                    .and_then(|a| a.as_choice())
                    .unwrap_or("none");
                if cchoice != "none" {
                    if let Some(p) = click_short
                        .into_iter()
                        .find(|e| e.node.to_string() == cchoice)
                    {
                        self.fields.insert(req_idx, p.node);
                        if let Some(act) = p.actions.get("click") {
                            self.record_pending(
                                "open",
                                req_idx,
                                Some(p.node),
                                &page.url,
                                elements,
                                Some(p.label.clone()),
                            );
                            return Ok(Some(PolicyOutcome::Action {
                                action: act.clone(),
                                text_to_type: None,
                                description: format!("Opening control '{}' [{}]", p.label, p.index),
                            }));
                        }
                        if let Some(act) = p.actions.get("fill") {
                            self.record_pending(
                                "fill",
                                req_idx,
                                Some(p.node),
                                &page.url,
                                elements,
                                Some(p.label.clone()),
                            );
                            self.typed = true;
                            self.met.insert(req_idx);
                            return Ok(Some(PolicyOutcome::Action {
                                action: act.clone(),
                                text_to_type: Some(req.value.clone()),
                                description: format!(
                                    "Filling '{}' into field '{}' [{}]",
                                    req.value, p.label, p.index
                                ),
                            }));
                        }
                    }
                }
                // Scroll fallback
                if let Some(scroll) = scroll_down {
                    if page.omitted_actions > 0
                        || needs_scroll_for_target(&req.value, elements, page)
                    {
                        return Ok(Some(PolicyOutcome::Action {
                            action: scroll,
                            text_to_type: None,
                            description: format!(
                                "Scrolling to find field '{}' = '{}' (omitted={})",
                                req.what, req.value, page.omitted_actions
                            ),
                        }));
                    }
                }
                *self.attempts.entry(req_idx).or_default() += 1;
                Ok(Some(PolicyOutcome::Action {
                    action: BrowserAction {
                        id: "wait".into(),
                        node: None,
                        kind: "wait".into(),
                        role: None,
                        label: "Wait for page elements to settle".into(),
                        value: None,
                        current_value: None,
                        checked: None,
                        expanded: None,
                        hint: None,
                        rect: None,
                        delta: None,
                    },
                    text_to_type: None,
                    description: format!("Waiting for field '{}' to appear", req.what),
                }))
            }
            "select" => {
                let choice = resp
                    .answers
                    .get("select_target")
                    .and_then(|a| a.as_choice())
                    .unwrap_or("none");
                if choice != "none" {
                    if let Some(p) = select_short
                        .into_iter()
                        .find(|e| e.node.to_string() == choice)
                    {
                        // Find matching option for the requirement value
                        let matching_opt = p
                            .options
                            .iter()
                            .find(|o| fold(&o.label).contains(&fold(&req.value)))
                            .or_else(|| p.options.first());
                        if let Some(opt) = matching_opt.cloned() {
                            self.fields.insert(req_idx, p.node);
                            self.record_pending(
                                "select",
                                req_idx,
                                Some(p.node),
                                &page.url,
                                elements,
                                Some(p.label.clone()),
                            );
                            self.met.insert(req_idx);
                            return Ok(Some(PolicyOutcome::Action {
                                action: opt.clone(),
                                text_to_type: None,
                                description: format!(
                                    "Selected option '{}' in [{}]",
                                    opt.label, p.index
                                ),
                            }));
                        }
                    }
                }
                // Fallback to wait
                *self.attempts.entry(req_idx).or_default() += 1;
                Ok(Some(PolicyOutcome::Action {
                    action: BrowserAction {
                        id: "wait".into(),
                        node: None,
                        kind: "wait".into(),
                        role: None,
                        label: "Wait for page elements to settle".into(),
                        value: None,
                        current_value: None,
                        checked: None,
                        expanded: None,
                        hint: None,
                        rect: None,
                        delta: None,
                    },
                    text_to_type: None,
                    description: format!("Waiting for field '{}' to appear", req.what),
                }))
            }
            "click" => {
                let choice = resp
                    .answers
                    .get("click_target")
                    .and_then(|a| a.as_choice())
                    .unwrap_or("none");
                if choice != "none" {
                    if let Some(p) = click_short
                        .into_iter()
                        .find(|e| e.node.to_string() == choice)
                    {
                        self.fields.insert(req_idx, p.node);
                        if let Some(act) = p.actions.get("click") {
                            self.record_pending(
                                "open",
                                req_idx,
                                Some(p.node),
                                &page.url,
                                elements,
                                Some(p.label.clone()),
                            );
                            return Ok(Some(PolicyOutcome::Action {
                                action: act.clone(),
                                text_to_type: None,
                                description: format!("Opening control '{}' [{}]", p.label, p.index),
                            }));
                        }
                        if let Some(act) = p.actions.get("fill") {
                            self.record_pending(
                                "fill",
                                req_idx,
                                Some(p.node),
                                &page.url,
                                elements,
                                Some(p.label.clone()),
                            );
                            self.typed = true;
                            self.met.insert(req_idx);
                            return Ok(Some(PolicyOutcome::Action {
                                action: act.clone(),
                                text_to_type: Some(req.value.clone()),
                                description: format!(
                                    "Filling '{}' into field '{}' [{}]",
                                    req.value, p.label, p.index
                                ),
                            }));
                        }
                    }
                }
                if let Some(scroll) = scroll_down {
                    if page.omitted_actions > 0
                        || needs_scroll_for_target(&req.value, elements, page)
                    {
                        return Ok(Some(PolicyOutcome::Action {
                            action: scroll,
                            text_to_type: None,
                            description: format!(
                                "Scrolling to find field '{}' = '{}' (omitted={})",
                                req.what, req.value, page.omitted_actions
                            ),
                        }));
                    }
                }
                *self.attempts.entry(req_idx).or_default() += 1;
                Ok(Some(PolicyOutcome::Action {
                    action: BrowserAction {
                        id: "wait".into(),
                        node: None,
                        kind: "wait".into(),
                        role: None,
                        label: "Wait for page elements to settle".into(),
                        value: None,
                        current_value: None,
                        checked: None,
                        expanded: None,
                        hint: None,
                        rect: None,
                        delta: None,
                    },
                    text_to_type: None,
                    description: format!("Waiting for field '{}' to appear", req.what),
                }))
            }
            "scroll" => {
                if let Some(scroll) = scroll_down {
                    return Ok(Some(PolicyOutcome::Action {
                        action: scroll,
                        text_to_type: None,
                        description: format!(
                            "Scrolling to find field '{}' = '{}' (omitted={})",
                            req.what, req.value, page.omitted_actions
                        ),
                    }));
                }
                Ok(Some(PolicyOutcome::Action {
                    action: BrowserAction {
                        id: "wait".into(),
                        node: None,
                        kind: "wait".into(),
                        role: None,
                        label: "Wait for page elements to settle".into(),
                        value: None,
                        current_value: None,
                        checked: None,
                        expanded: None,
                        hint: None,
                        rect: None,
                        delta: None,
                    },
                    text_to_type: None,
                    description: format!("Waiting for field '{}' to appear", req.what),
                }))
            }
            "wait" => {
                *self.attempts.entry(req_idx).or_default() += 1;
                Ok(Some(PolicyOutcome::Action {
                    action: BrowserAction {
                        id: "wait".into(),
                        node: None,
                        kind: "wait".into(),
                        role: None,
                        label: "Wait for page elements to settle".into(),
                        value: None,
                        current_value: None,
                        checked: None,
                        expanded: None,
                        hint: None,
                        rect: None,
                        delta: None,
                    },
                    text_to_type: None,
                    description: format!("Waiting for field '{}' to appear", req.what),
                }))
            }
            "done" => {
                // Laya sometimes predicts "done" for an unfilled field (out-of-dist).
                // Only honor it if the field is deterministically already filled
                // or the type_target head also says "none". Otherwise fall back
                // to filling the field.
                let type_choice = resp
                    .answers
                    .get("type_target")
                    .and_then(|a| a.as_choice())
                    .unwrap_or("none");
                if type_choice != "none" {
                    if let Some(p) = fill_short
                        .iter()
                        .find(|e| e.node.to_string() == type_choice)
                    {
                        if let Some(act) = p.actions.get("fill") {
                            self.record_pending(
                                "fill",
                                req_idx,
                                Some(p.node),
                                &page.url,
                                elements,
                                Some(p.label.clone()),
                            );
                            self.typed = true;
                            self.met.insert(req_idx);
                            return Ok(Some(PolicyOutcome::Action {
                                action: act.clone(),
                                text_to_type: Some(req.value.clone()),
                                description: format!(
                                    "Filling '{}' into field '{}' [{}] (overriding done)",
                                    req.value, p.label, p.index
                                ),
                            }));
                        }
                    }
                }
                // Field truly already satisfied? Check deterministic.
                let already = elements.iter().any(|c| {
                    is_field(c)
                        && !is_toggle(&c.role)
                        && month_day(&c.label).is_none()
                        && !words(&c.label).is_disjoint(&words(&req.what))
                        && !words(&c.label).is_disjoint(&words(&req.value))
                });
                if already {
                    self.met.insert(req_idx);
                    Ok(None)
                } else {
                    // Treat spurious done as wait so we retry with fresh snapshot.
                    *self.attempts.entry(req_idx).or_default() += 1;
                    Ok(Some(PolicyOutcome::Action {
                        action: BrowserAction {
                            id: "wait".into(),
                            node: None,
                            kind: "wait".into(),
                            role: None,
                            label: "Wait for page elements to settle".into(),
                            value: None,
                            current_value: None,
                            checked: None,
                            expanded: None,
                            hint: None,
                            rect: None,
                            delta: None,
                        },
                        text_to_type: None,
                        description: format!(
                            "Waiting for field '{}' to appear (done overridden)",
                            req.what
                        ),
                    }))
                }
            }
            "blocked" => {
                *self.attempts.entry(req_idx).or_default() += 1;
                self.skipped.insert(req_idx);
                Ok(None)
            }
            _ => {
                *self.attempts.entry(req_idx).or_default() += 1;
                Ok(Some(PolicyOutcome::Action {
                    action: BrowserAction {
                        id: "wait".into(),
                        node: None,
                        kind: "wait".into(),
                        role: None,
                        label: "Wait for page elements to settle".into(),
                        value: None,
                        current_value: None,
                        checked: None,
                        expanded: None,
                        hint: None,
                        rect: None,
                        delta: None,
                    },
                    text_to_type: None,
                    description: format!("Waiting for field '{}' to appear", req.what),
                }))
            }
        }
    }

    /// Speculative post-requirements fan-out: collapses `classify_page_state` +
    /// submit/item/next picks into ONE forward pass.
    ///
    /// Heads: `operation` (click|scroll|wait|done|blocked), `click_target` (pooled),
    /// `kind` (finish/form/results/other). Only operation-relevant head is used.
    async fn speculative_post_step(
        &mut self,
        client: &SystemOneClient,
        page: &BrowserPageSnapshot,
        elements: &[ObservedElement],
        clickable: &[ObservedElement],
        scroll_down: Option<BrowserAction>,
        navigated: bool,
    ) -> Result<Option<PolicyOutcome>> {
        let plan = self
            .plan
            .as_ref()
            .ok_or_else(|| anyhow!("goal plan not initialized"))?;
        let finish = plan.finish.clone();
        let item = plan.open.clone();

        // Check submit/item/next presence to decide whether a model call is needed.
        // If no candidates, we may still need classify for done detection.
        let submit_candidates: Vec<ObservedElement> = clickable
            .iter()
            .filter(|e| {
                e.role == "button"
                    || words(&e.label)
                        .iter()
                        .any(|w| SUBMIT_WORDS.contains(&w.as_str()))
            })
            .cloned()
            .collect();
        let item_candidates: Vec<ObservedElement> = if let Some(name) = &item {
            clickable
                .iter()
                .filter(|e| relevance(e, name) > 0 && !is_field(e))
                .cloned()
                .collect()
        } else {
            Vec::new()
        };
        let next_candidates: Vec<ObservedElement> = clickable
            .iter()
            .filter(|e| *self.tried.get(&e.label).unwrap_or(&0) < 2)
            .cloned()
            .collect();

        // Scroll-early for below-fold target (e.g., feed item 260) — must be before any Laya call
        // Mirrors original pre-pick logic: if target not visible and needs scroll, emit scroll directly (0 predicts)
        if let Some(name) = &item {
            let target_needs_scroll = needs_scroll_for_target(name, elements, page)
                || page.omitted_actions > 0 && !is_target_visible(name, elements);
            if target_needs_scroll {
                if let Some(scroll) = scroll_down.clone() {
                    if !is_target_visible(name, elements) {
                        return Ok(Some(PolicyOutcome::Action {
                            action: scroll,
                            text_to_type: None,
                            description: format!(
                                "Scrolling to find item '{}' (omitted={}, visible={})",
                                name,
                                page.omitted_actions,
                                elements.len()
                            ),
                        }));
                    }
                }
            }
        }

        let needs_classify = self.submitted || navigated || plan.requirements.is_empty();
        let needs_action = !submit_candidates.is_empty()
            || !item_candidates.is_empty()
            || !next_candidates.is_empty()
            || needs_classify;

        if !needs_action {
            return Ok(None);
        }

        // Build pooled click candidates (deduplicated) for the speculative click_target head
        let mut pooled: Vec<ObservedElement> = Vec::new();
        let mut seen: HashSet<i64> = HashSet::new();
        for e in submit_candidates
            .iter()
            .chain(item_candidates.iter())
            .chain(next_candidates.iter())
        {
            if seen.insert(e.node) {
                pooled.push(e.clone());
            }
        }
        let click_short = shortlist(&pooled, &format!("{} {}", self.goal, finish), 15);

        // Quick deterministic done check via host/titled already handled above; still need
        // model-based classification for finish vs results vs form.
        let state = json!({
            "goal": self.goal,
            "finish": finish,
            "page_title": page.title,
            "page_text": summary(&page.text, &finish, 200),
            "submitted": self.submitted,
            "navigated": navigated,
        });

        let mut op_criteria = HashMap::new();
        op_criteria.insert(
            "click".to_string(),
            json!("Click an element to advance toward the goal"),
        );
        op_criteria.insert(
            "scroll".to_string(),
            json!("Scroll down to reveal more content"),
        );
        op_criteria.insert("wait".to_string(), json!("Wait for page to settle"));
        op_criteria.insert("done".to_string(), json!("Goal is complete"));
        op_criteria.insert("blocked".to_string(), json!("Cannot proceed"));

        let mut questions = HashMap::new();
        questions.insert(
            "operation".to_string(),
            Question::choice(
                "choose operation: click|scroll|wait|done|blocked",
                op_criteria,
            ),
        );

        if !click_short.is_empty() {
            let mut crit = HashMap::new();
            for e in &click_short {
                let trunc: String = describe(e).chars().take(28).collect();
                crit.insert(e.node.to_string(), json!(trunc));
            }
            // Click head allows none for blocked case
            crit.insert("none".to_string(), json!("none of these"));
            questions.insert(
                "click_target".to_string(),
                Question::choice("Which element should be clicked next?", crit),
            );
        }

        // Always include classify head when submission/navigation may have finished the task —
        // collapsed into same forward pass (fan-out).
        if needs_classify {
            let mut kind_criteria = HashMap::new();
            kind_criteria.insert("finish".to_string(), json!(finish.clone()));
            kind_criteria.insert(
                "form".to_string(),
                json!("a search or input form that still has to be submitted"),
            );
            kind_criteria.insert(
                "results".to_string(),
                json!("a list of search results or products"),
            );
            kind_criteria.insert("other".to_string(), json!("some other intermediate page"));
            questions.insert(
                "kind".to_string(),
                Question::choice("Which best describes the current page?", kind_criteria),
            );
        }

        let resp = client.predict_speculative(&state, questions).await?;

        // Handle classify first — independent Done/Wait transitions have priority over click
        if needs_classify {
            if let Some(kind_ans) = resp.answers.get("kind").and_then(|a| a.as_choice()) {
                match kind_ans {
                    "finish" => {
                        return Ok(Some(PolicyOutcome::Done {
                            message: format!("Goal completed: {}", finish),
                        }));
                    }
                    "results" if self.waits >= 2 => {
                        return Ok(Some(PolicyOutcome::Done {
                            message: format!("Goal completed: {}", finish),
                        }));
                    }
                    "results" if self.waits < 4 => {
                        self.waits += 1;
                        return Ok(Some(PolicyOutcome::Action {
                            action: BrowserAction {
                                id: "wait".into(),
                                node: None,
                                kind: "wait".into(),
                                role: None,
                                label: "Wait for results to finish rendering".into(),
                                value: None,
                                current_value: None,
                                checked: None,
                                expanded: None,
                                hint: None,
                                rect: None,
                                delta: None,
                            },
                            text_to_type: None,
                            description: "Waiting for results to load".into(),
                        }));
                    }
                    _ => {}
                }
            }
        }

        let op = resp
            .answers
            .get("operation")
            .and_then(|a| a.as_choice())
            .unwrap_or("wait");
        match op {
            "done" => {
                return Ok(Some(PolicyOutcome::Done {
                    message: format!("Goal completed: {}", finish),
                }));
            }
            "scroll" => {
                if let Some(scroll) = scroll_down {
                    return Ok(Some(PolicyOutcome::Action {
                        action: scroll,
                        text_to_type: None,
                        description: "Scrolling down".into(),
                    }));
                }
                return Ok(Some(PolicyOutcome::Action {
                    action: BrowserAction {
                        id: "wait".into(),
                        node: None,
                        kind: "wait".into(),
                        role: None,
                        label: "Wait for page elements to settle".into(),
                        value: None,
                        current_value: None,
                        checked: None,
                        expanded: None,
                        hint: None,
                        rect: None,
                        delta: None,
                    },
                    text_to_type: None,
                    description: "Waiting for page to settle".into(),
                }));
            }
            "blocked" => return Ok(None),
            "click" => {
                let choice = resp
                    .answers
                    .get("click_target")
                    .and_then(|a| a.as_choice())
                    .unwrap_or("none");
                if choice == "none" {
                    // No valid click target — maybe need scroll for item target
                    if let Some(name) = &item {
                        if (page.omitted_actions > 0
                            || needs_scroll_for_target(name, elements, page))
                            && !is_target_visible(name, elements)
                        {
                            if let Some(scroll) = scroll_down {
                                return Ok(Some(PolicyOutcome::Action {
                                    action: scroll,
                                    text_to_type: None,
                                    description: format!(
                                        "Scrolling to find item '{}' after Laya 'none' (omitted={})",
                                        name, page.omitted_actions
                                    ),
                                }));
                            }
                        }
                    }
                    return Ok(None);
                }
                if let Some(p) = click_short
                    .into_iter()
                    .find(|e| e.node.to_string() == choice)
                {
                    if let Some(act) = p.actions.get("click") {
                        // Distinguish submit vs item vs next via candidate membership for metrics/label
                        let kind = if submit_candidates.iter().any(|e| e.node == p.node) {
                            self.typed = false;
                            self.submitted = true;
                            "submit"
                        } else if item_candidates.iter().any(|e| e.node == p.node) {
                            "item"
                        } else {
                            *self.tried.entry(p.label.clone()).or_default() += 1;
                            "next"
                        };
                        self.record_pending(
                            kind,
                            0,
                            Some(p.node),
                            &page.url,
                            elements,
                            Some(p.label.clone()),
                        );
                        let desc = match kind {
                            "submit" => format!("Submitting form via [{}] {}", p.index, p.label),
                            "item" => format!("Opening item [{}] {}", p.index, p.label),
                            _ => format!("Clicking [{}] {} to advance goal", p.index, p.label),
                        };
                        return Ok(Some(PolicyOutcome::Action {
                            action: act.clone(),
                            text_to_type: None,
                            description: desc,
                        }));
                    }
                }
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    fn record_pending(
        &mut self,
        kind: &str,
        req: usize,
        node: Option<i64>,
        url: &str,
        elements: &[ObservedElement],
        label: Option<String>,
    ) {
        let before: HashSet<i64> = elements.iter().map(|e| e.node).collect();
        self.last = Some(PolicyStep {
            choice: label.clone().unwrap_or_default(),
            kind: kind.to_string(),
            req: Some(req),
            node,
            url: url.to_string(),
            before,
            label,
        });
        self.edit_url = Some(url.to_string());
    }

    async fn pick_element(
        &self,
        client: &SystemOneClient,
        qid: &str,
        candidates: &[ObservedElement],
        state_text: &str,
        instruction: &str,
        query: &str,
        allow_none: bool,
    ) -> Result<Option<ObservedElement>> {
        if candidates.is_empty() {
            return Ok(None);
        }

        // Exact date match needs no model call
        let date_opt = month_day(query);
        if let Some(d) = date_opt {
            let dated: Vec<&ObservedElement> = candidates
                .iter()
                .filter(|e| month_day(&e.label) == Some(d.clone()))
                .collect();
            if dated.len() == 1 {
                return Ok(Some(dated[0].clone()));
            }
        }

        let ranked = shortlist(candidates, query, 15);
        if ranked.is_empty() {
            return Ok(None);
        }
        if ranked.len() == 1 && !allow_none {
            return Ok(Some(ranked[0].clone()));
        }

        let mut criteria = HashMap::new();
        for e in &ranked {
            criteria.insert(e.node.to_string(), json!(describe(e)));
        }
        if allow_none {
            criteria.insert("none".to_string(), json!("none of these"));
        }

        let state = json!({ "state": state_text });
        let q = Question::choice(instruction, criteria);
        let mut questions = HashMap::new();
        questions.insert(qid.to_string(), q);

        let resp = client.predict(&state, questions).await?;
        let ans = resp
            .answers
            .get(qid)
            .ok_or_else(|| anyhow!("missing '{qid}' answer from System-1"))?;
        let choice = ans.as_choice().unwrap_or("none");

        if choice == "none" {
            return Ok(None);
        }

        Ok(ranked.into_iter().find(|e| e.node.to_string() == choice))
    }

    #[allow(dead_code)]
    async fn classify_page_state(
        &self,
        client: &SystemOneClient,
        title: &str,
        page_summary: &str,
        finish: &str,
    ) -> Result<String> {
        let state = json!({
            "title": title,
            "page_text": page_summary,
        });

        let mut criteria = HashMap::new();
        criteria.insert("finish".to_string(), json!(finish));
        criteria.insert(
            "form".to_string(),
            json!("a search or input form that still has to be submitted"),
        );
        criteria.insert(
            "results".to_string(),
            json!("a list of search results or products"),
        );
        criteria.insert("other".to_string(), json!("some other intermediate page"));

        let q = Question::choice("Which best describes the current page?", criteria);
        let mut questions = HashMap::new();
        questions.insert("kind".to_string(), q);

        let resp = client.predict(&state, questions).await?;
        let ans = resp
            .answers
            .get("kind")
            .ok_or_else(|| anyhow!("missing 'kind' answer from System-1"))?;
        let choice = ans.as_choice().unwrap_or("other").to_string();
        Ok(choice)
    }
}

// ---- Pure Heuristics & Deterministic Helpers ----

const SUBMIT_WORDS: &[&str] = &[
    "search", "submit", "find", "go", "apply", "done", "continue", "next", "confirm", "show",
];
const TOGGLES: &[&str] = &["checkbox", "radio", "switch"];
const STOP_WORDS: &[&str] = &[
    "the", "and", "for", "with", "from", "find", "open", "stop", "when", "page", "are", "this",
];

pub fn fold(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn words(text: &str) -> HashSet<String> {
    fold(text)
        .split_whitespace()
        .filter(|w| {
            (w.len() > 2 || w.chars().all(|c| c.is_ascii_digit())) && !STOP_WORDS.contains(&w)
        })
        .map(|w| w.chars().take(5).collect())
        .collect()
}

pub fn month_day(text: &str) -> Option<(String, u32)> {
    let lower = text.to_lowercase();
    let months = [
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    ];
    let tokens: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .collect();

    for (i, &t) in tokens.iter().enumerate() {
        for (m_idx, &m) in months.iter().enumerate() {
            if t.starts_with(m) {
                // Check if next token is a day: e.g. "October 15"
                if i + 1 < tokens.len() {
                    if let Ok(day) = tokens[i + 1].parse::<u32>() {
                        if (1..=31).contains(&day) {
                            return Some((months[m_idx].to_string(), day));
                        }
                    }
                }
                // Check if previous token is a day: e.g. "15 October"
                if i > 0 {
                    if let Ok(day) = tokens[i - 1].parse::<u32>() {
                        if (1..=31).contains(&day) {
                            return Some((months[m_idx].to_string(), day));
                        }
                    }
                }
            }
        }
    }
    None
}

pub fn is_toggle(role: &str) -> bool {
    TOGGLES.contains(&role)
}

pub fn is_field(e: &ObservedElement) -> bool {
    [
        "combobox",
        "textbox",
        "searchbox",
        "spinbutton",
        "checkbox",
        "radio",
        "switch",
    ]
    .contains(&e.role.as_str())
        || !e.options.is_empty()
}

pub fn describe(e: &ObservedElement) -> String {
    let mut text = format!(
        "{} {}",
        e.role,
        e.label.chars().take(50).collect::<String>()
    );
    if !e.hint.is_empty() {
        text.push_str(&format!(
            " ({})",
            e.hint.chars().take(40).collect::<String>()
        ));
    }
    if is_field(e) && e.current != e.label && !e.current.is_empty() {
        text.push_str(&format!(
            " = {}",
            e.current.chars().take(30).collect::<String>()
        ));
    }
    text
}

pub fn relevance(e: &ObservedElement, text: &str) -> usize {
    let e_words = words(&format!("{} {} {}", e.label, e.hint, e.current));
    let t_words = words(text);
    let mut score = e_words.intersection(&t_words).count();
    if let Some(date) = month_day(text) {
        if month_day(&e.label) == Some(date) {
            score += 5;
        }
    }
    score
}

pub fn shortlist(candidates: &[ObservedElement], text: &str, limit: usize) -> Vec<ObservedElement> {
    let mut scored: Vec<(usize, &ObservedElement)> =
        candidates.iter().map(|e| (relevance(e, text), e)).collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0));
    scored
        .into_iter()
        .take(limit)
        .map(|(_, e)| e.clone())
        .collect()
}

pub fn summary(text: &str, about: &str, limit_chars: usize) -> String {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let target = words(about);
    let mut scored: Vec<(usize, &str)> = lines
        .iter()
        .map(|&l| (words(l).intersection(&target).count(), l))
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0));
    scored
        .into_iter()
        .take(5)
        .map(|(_, l)| l)
        .collect::<Vec<_>>()
        .join(" | ")
        .chars()
        .take(limit_chars)
        .collect()
}

pub fn titled(title: &str, name: &str) -> bool {
    let wanted = words(name);
    let shown = words(title);
    let shared = wanted.intersection(&shown).count();
    !wanted.is_empty() && shared >= 1 && (shared * 10 >= wanted.len() * 6)
}

pub fn is_target_visible(target: &str, elements: &[ObservedElement]) -> bool {
    let t = fold(target);
    if t.is_empty() {
        return false;
    }
    elements.iter().any(|e| {
        let combined = fold(&format!("{} {} {}", e.label, e.hint, e.current));
        combined.contains(&t)
    })
}

pub fn needs_scroll_for_target(
    target: &str,
    elements: &[ObservedElement],
    _page: &BrowserPageSnapshot,
) -> bool {
    let tw = words(target);
    if tw.is_empty() {
        return false;
    }
    let max_rel = elements
        .iter()
        .map(|e| relevance(e, target))
        .max()
        .unwrap_or(0);
    // Not all target words are matched in visible set -> below-fold target needs scroll
    max_rel < tw.len()
}

fn location(url: &str) -> String {
    if let Ok(parsed) = reqwest::Url::parse(url) {
        format!("{}{}", parsed.host_str().unwrap_or(""), parsed.path())
    } else {
        url.to_string()
    }
}

/// P1-4 independent DONE verifier helper.
/// Returns true if the page state independently confirms the goal:
/// - `plan.open` via `titled(title, open)` OR `plan.finish` via
///   `summary(text, finish, 200)` containing finish keywords,
/// - and URL host matches expected (extracted from goal).
/// Pure and unit-testable; called by automation's `verify_done_independently`.
pub fn verify_done(
    title: &str,
    text: &str,
    url: &str,
    plan: &GoalPlan,
    expected_url: Option<&str>,
) -> bool {
    // Host check (strip www. for tolerance)
    let host_of = |u: &str| -> Option<String> {
        if let Ok(p) = reqwest::Url::parse(u) {
            p.host_str().map(|h| h.to_lowercase())
        } else {
            let after = u.split("://").nth(1).unwrap_or(u);
            let h = after
                .split('/')
                .next()
                .unwrap_or("")
                .split('?')
                .next()
                .unwrap_or("");
            if h.is_empty() {
                None
            } else {
                Some(h.to_lowercase())
            }
        }
    };
    let norm = |h: String| h.trim_start_matches("www.").to_string();
    // The host check only applies when the goal actually named a destination.
    // `expected_url` used to be a *guessed* URL — an unrecognised goal fell
    // through to google.com — so the check silently meant "are we on Google?",
    // which was false for every task that was not a Google task. Now `None`
    // means "the goal named no site", and the plan's own evidence below is the
    // whole test rather than being overruled by a guess.
    if let Some(expected) = expected_url {
        let hosts_match = match (host_of(url), host_of(expected)) {
            (Some(a), Some(e)) => norm(a) == norm(e),
            _ => false,
        };
        if !hosts_match {
            return false;
        }
    }
    if let Some(open) = &plan.open {
        if !open.trim().is_empty() {
            return titled(title, open);
        }
    }
    // finish via summary word overlap
    let summ = summary(text, &plan.finish, 200);
    if summ.trim().is_empty() || plan.finish.trim().is_empty() {
        return false;
    }
    let summ_words = words(&summ);
    let finish_words = words(&plan.finish);
    if finish_words.is_empty() || summ_words.is_empty() {
        return false;
    }
    let shared = summ_words.intersection(&finish_words).count();
    shared >= 1 && shared * 10 >= finish_words.len() * 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fold() {
        assert_eq!(
            fold("Where to? (London Heathrow)"),
            "where to london heathrow"
        );
        assert_eq!(fold("Tue, Oct 20"), "tue oct 20");
    }

    #[test]
    fn test_words() {
        let w = words("Where to? London Heathrow Flights");
        assert!(w.contains("where"));
        assert!(w.contains("londo"));
        assert!(w.contains("heath"));
        assert!(w.contains("fligh"));
        // Stop words removed
        assert!(
            !words("the and for with page")
                .iter()
                .any(|s| ["the", "and", "for"].contains(&s.as_str()))
        );
    }

    #[test]
    fn test_month_day() {
        assert_eq!(month_day("October 15, 2026"), Some(("oct".to_string(), 15)));
        assert_eq!(month_day("15 Oct 2026"), Some(("oct".to_string(), 15)));
        assert_eq!(month_day("Tue, Mar 3"), Some(("mar".to_string(), 3)));
        assert_eq!(month_day("random non-date text"), None);
    }

    #[test]
    fn test_titled() {
        assert!(titled("London - Wikipedia", "London"));
        assert!(!titled("London Search Results", "Tokyo"));
    }

    #[test]
    fn test_goal_plan_parsing() {
        let raw = r#"{
            "requirements": [
                {"what": "destination", "value": "Tokyo"},
                {"what": "departure date", "value": "October 15, 2026"}
            ],
            "open": null,
            "finish": "Flights to Tokyo for October 15, 2026 are displayed."
        }"#;

        let plan: GoalPlan = serde_json::from_str(raw).expect("valid plan");
        assert_eq!(plan.requirements.len(), 2);
        assert_eq!(plan.requirements[0].what, "destination");
        assert_eq!(plan.requirements[0].value, "Tokyo");
        assert_eq!(
            plan.finish,
            "Flights to Tokyo for October 15, 2026 are displayed."
        );
    }

    #[test]
    fn test_verify_done_helper_open() {
        let plan = GoalPlan {
            requirements: vec![],
            open: Some("Tokyo".into()),
            finish: "done".into(),
        };
        assert!(verify_done(
            "Tokyo - Wikipedia",
            "some text",
            "https://www.wikipedia.org/wiki/Tokyo",
            &plan,
            Some("https://www.wikipedia.org/")
        ));
        assert!(!verify_done(
            "Berlin - Wikipedia",
            "some text",
            "https://www.wikipedia.org/wiki/Tokyo",
            &plan,
            Some("https://www.wikipedia.org/")
        ));
        // host mismatch fails even if title ok
        assert!(!verify_done(
            "Tokyo - Wikipedia",
            "some text",
            "https://www.youtube.com/watch?v=1",
            &plan,
            Some("https://www.wikipedia.org/")
        ));
    }

    #[test]
    fn test_verify_done_helper_finish() {
        let plan = GoalPlan {
            requirements: vec![],
            open: None,
            finish: "Flights to Tokyo are displayed.".into(),
        };
        assert!(verify_done(
            "Google Flights",
            "Flights to Tokyo are displayed. Price $500",
            "https://www.google.com/travel/flights",
            &plan,
            Some("https://www.google.com/travel/flights")
        ));
        assert!(!verify_done(
            "Google Flights",
            "Unrelated cooking content",
            "https://www.google.com/travel/flights",
            &plan,
            Some("https://www.google.com/travel/flights")
        ));
    }

    #[test]
    fn test_scroll_inventory_retains_scroll() {
        use crate::browser_cdp::{BrowserAction, BrowserPageSnapshot};
        let policy = BrowserPolicy::new("test");
        let snap = BrowserPageSnapshot {
            url: "http://example.com/feed".into(),
            title: "Result Feed".into(),
            w: 1280,
            h: 800,
            text: "feed".into(),
            actions: vec![
                BrowserAction {
                    id: "e1".into(),
                    node: Some(1),
                    kind: "click".into(),
                    role: Some("button".into()),
                    label: "R1".into(),
                    value: None,
                    current_value: None,
                    checked: None,
                    expanded: None,
                    hint: Some("Result item 1".into()),
                    rect: None,
                    delta: None,
                },
                BrowserAction {
                    id: "scroll_down".into(),
                    node: None,
                    kind: "scroll".into(),
                    role: None,
                    label: "Scroll down".into(),
                    value: None,
                    current_value: None,
                    checked: None,
                    expanded: None,
                    hint: None,
                    rect: Some(crate::browser_cdp::Rect {
                        x: 640.0,
                        y: 400.0,
                        w: 0.0,
                        h: 0.0,
                    }),
                    delta: Some(560),
                },
                BrowserAction {
                    id: "wait".into(),
                    node: None,
                    kind: "wait".into(),
                    role: None,
                    label: "Wait for the page to update".into(),
                    value: None,
                    current_value: None,
                    checked: None,
                    expanded: None,
                    hint: None,
                    rect: None,
                    delta: None,
                },
            ],
            marker: serde_json::Value::Null,
            page_key: serde_json::Value::Null,
            guards: Default::default(),
            omitted_actions: 50,
        };
        // observed() still drops scroll/wait
        let observed = policy.observed(&snap);
        assert_eq!(observed.len(), 1, "observed should keep click only");
        assert_eq!(observed[0].label, "R1");
        // scroll_inventory retains scroll as first-class
        let scrolls = policy.scroll_inventory(&snap);
        assert_eq!(scrolls.len(), 1);
        assert_eq!(scrolls[0].kind, "scroll");
        assert_eq!(scrolls[0].delta, Some(560));
        assert!(
            scrolls[0].rect.is_some(),
            "scroll should include rect delta"
        );
        let down = policy.scroll_down(&snap).expect("scroll_down present");
        assert_eq!(down.id, "scroll_down");
        assert_eq!(down.label, "Scroll down");
    }

    #[tokio::test]
    async fn test_step_emits_scroll_when_below_fold() {
        use crate::browser_cdp::{BrowserAction, BrowserPageSnapshot};
        use crate::client::SystemOneClient;
        // Snapshot with 250 visible items (R1..R250) + scroll_down, omitted 50 -> target 260 below-fold
        let mut actions = Vec::new();
        for i in 1..=10 {
            actions.push(BrowserAction {
                id: format!("e{i}"),
                node: Some(i as i64),
                kind: "click".into(),
                role: Some("button".into()),
                label: format!("R{i}"),
                value: None,
                current_value: None,
                checked: None,
                expanded: None,
                hint: Some(format!("Result item {i}")),
                rect: None,
                delta: None,
            });
        }
        actions.push(BrowserAction {
            id: "scroll_down".into(),
            node: None,
            kind: "scroll".into(),
            role: None,
            label: "Scroll down".into(),
            value: None,
            current_value: None,
            checked: None,
            expanded: None,
            hint: None,
            rect: Some(crate::browser_cdp::Rect {
                x: 640.0,
                y: 400.0,
                w: 0.0,
                h: 0.0,
            }),
            delta: Some(560),
        });
        let snap = BrowserPageSnapshot {
            url: "http://example.com/feed".into(),
            title: "Result Feed".into(),
            w: 1280,
            h: 800,
            text: "feed".into(),
            actions,
            marker: serde_json::Value::Null,
            page_key: serde_json::Value::Null,
            guards: Default::default(),
            omitted_actions: 50,
        };
        let cfg = lucy_config::SystemOneConfig {
            enabled: true,
            provider: "decider".into(),
            base_url: "http://127.0.0.1:9".into(),
            api_key: None,
            model: "Mapika/decider-2b-vision".into(),
            direct_python: None,
            confidence_threshold: 0.1,
            timeout_ms: 1000,
            auto_start: false,
            server_command: None,
            server_model: None,
        };
        let client = SystemOneClient::new(cfg, "python3".into());
        let mut policy = BrowserPolicy::new("Open Result item 260");
        policy.set_plan(GoalPlan {
            requirements: vec![],
            open: Some("Result item 260".into()),
            finish: "Result item 260 is open".into(),
        });
        let outcome = policy.step(&snap, &client).await.expect("step");
        match outcome {
            crate::browser_policy::PolicyOutcome::Action {
                action,
                description,
                ..
            } => {
                assert_eq!(
                    action.kind, "scroll",
                    "below-fold 260 with omitted>0 should emit scroll, got {action:?} desc={description}"
                );
                assert_eq!(action.delta, Some(560));
            }
            other => panic!("expected scroll action, got {other:?}"),
        }
    }

    #[test]
    fn test_needs_scroll_for_target_and_visibility() {
        use crate::browser_cdp::BrowserAction;
        let policy = BrowserPolicy::new("test");
        let mut actions = Vec::new();
        for i in 1..=5 {
            actions.push(BrowserAction {
                id: format!("e{i}"),
                node: Some(i as i64),
                kind: "click".into(),
                role: Some("button".into()),
                label: format!("R{i}"),
                value: None,
                current_value: None,
                checked: None,
                expanded: None,
                hint: Some(format!("Result item {i}")),
                rect: None,
                delta: None,
            });
        }
        let snap = crate::browser_cdp::BrowserPageSnapshot {
            url: "http://example.com".into(),
            title: "feed".into(),
            w: 1280,
            h: 800,
            text: "".into(),
            actions,
            marker: serde_json::Value::Null,
            page_key: serde_json::Value::Null,
            guards: Default::default(),
            omitted_actions: 0,
        };
        let observed = policy.observed(&snap);
        // R260 not visible
        assert!(!is_target_visible("Result item 260", &observed));
        assert!(needs_scroll_for_target("Result item 260", &observed, &snap));
        // R3 is visible
        assert!(is_target_visible("Result item 3", &observed));
        assert!(!needs_scroll_for_target("Result item 3", &observed, &snap));
    }

    #[tokio::test]
    async fn test_speculative_fused_single_predict_requirement() {
        use crate::browser_cdp::{BrowserAction, BrowserPageSnapshot};
        use crate::client::SystemOneClient;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        // Mock Laya server: health + predict with operation=fill fan-out
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind mock");
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    break;
                };
                let mut buf = vec![0u8; 16384];
                let Ok(n) = stream.read(&mut buf).await else {
                    continue;
                };
                let req = String::from_utf8_lossy(&buf[..n]).to_string();
                if req.starts_with("GET /health") {
                    let resp = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok";
                    let _ = stream.write_all(resp.as_bytes()).await;
                } else if req.contains("POST /predict") {
                    // Requirement fan-out: operation=fill, type_target=10
                    let answers = r#"{"operation":{"choice":"fill","confidence":0.99,"probabilities":{"fill":0.99,"click":0.01}},"type_target":{"choice":"10","confidence":0.9,"probabilities":{"10":0.9,"none":0.1}},"click_target":{"choice":"20","confidence":0.8,"probabilities":{"20":0.8}},"select_target":{"choice":"none","confidence":0.5,"probabilities":{"none":0.5}}}"#;
                    let body = format!(r#"{{"answers":{}}}"#, answers);
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = stream.write_all(resp.as_bytes()).await;
                } else {
                    let resp = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
                    let _ = stream.write_all(resp.as_bytes()).await;
                }
            }
        });

        let cfg = lucy_config::SystemOneConfig {
            enabled: true,
            provider: "decider".into(),
            base_url: format!("http://{}", addr),
            api_key: None,
            model: "Mapika/decider-2b-vision".into(),
            direct_python: None,
            confidence_threshold: 0.1,
            timeout_ms: 2000,
            auto_start: false,
            server_command: None,
            server_model: None,
        };
        let client = SystemOneClient::new(cfg, "python3".into());
        let mut policy = BrowserPolicy::new("Fill name Ada");
        policy.set_plan(GoalPlan {
            requirements: vec![GoalRequirement {
                what: "Full name".into(),
                value: "Ada Lovelace".into(),
            }],
            open: None,
            finish: "Form submitted".into(),
        });

        // Snapshot: node 10 is fillable "Full name" field, node 20 is clickable fallback
        let snap = BrowserPageSnapshot {
            url: "http://example.com/form".into(),
            title: "Form".into(),
            w: 1280,
            h: 800,
            text: "Fill the form".into(),
            actions: vec![
                BrowserAction {
                    id: "fill_fullname".into(),
                    node: Some(10),
                    kind: "fill".into(),
                    role: Some("textbox".into()),
                    label: "Full name".into(),
                    value: None,
                    current_value: Some("".into()),
                    checked: None,
                    expanded: None,
                    hint: None,
                    rect: None,
                    delta: None,
                },
                BrowserAction {
                    id: "click_submit".into(),
                    node: Some(20),
                    kind: "click".into(),
                    role: Some("button".into()),
                    label: "Submit".into(),
                    value: None,
                    current_value: None,
                    checked: None,
                    expanded: None,
                    hint: None,
                    rect: None,
                    delta: None,
                },
            ],
            marker: serde_json::Value::Null,
            page_key: serde_json::Value::Null,
            guards: Default::default(),
            omitted_actions: 0,
        };

        let before = client.predict_calls();
        let outcome = policy.step(&snap, &client).await.expect("step");
        let after = client.predict_calls();
        // Speculative fan-out: exactly one forward pass for requirement (operation + targets)
        assert_eq!(
            after - before,
            1,
            "requirement step must use exactly 1 predict (fused fan-out), got {}",
            after - before
        );
        match outcome {
            crate::browser_policy::PolicyOutcome::Action {
                action,
                text_to_type,
                ..
            } => {
                assert_eq!(
                    action.kind, "fill",
                    "expected fill action from fused fan-out"
                );
                assert_eq!(action.node, Some(10));
                assert_eq!(text_to_type, Some("Ada Lovelace".into()));
            }
            other => panic!("expected fill action, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_speculative_fused_single_predict_post() {
        use crate::browser_cdp::{BrowserAction, BrowserPageSnapshot};
        use crate::client::SystemOneClient;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind mock");
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    break;
                };
                let mut buf = vec![0u8; 16384];
                let Ok(n) = stream.read(&mut buf).await else {
                    continue;
                };
                let req = String::from_utf8_lossy(&buf[..n]).to_string();
                if req.starts_with("GET /health") {
                    let resp = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok";
                    let _ = stream.write_all(resp.as_bytes()).await;
                } else if req.contains("POST /predict") {
                    // Post-requirements fan-out: operation=click, kind=other, click_target=30
                    let answers = r#"{"operation":{"choice":"click","confidence":0.95,"probabilities":{"click":0.95}},"click_target":{"choice":"30","confidence":0.9,"probabilities":{"30":0.9}},"kind":{"choice":"other","confidence":0.6,"probabilities":{"other":0.6}}}"#;
                    let body = format!(r#"{{"answers":{}}}"#, answers);
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = stream.write_all(resp.as_bytes()).await;
                } else {
                    let resp = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
                    let _ = stream.write_all(resp.as_bytes()).await;
                }
            }
        });

        let cfg = lucy_config::SystemOneConfig {
            enabled: true,
            provider: "decider".into(),
            base_url: format!("http://{}", addr),
            api_key: None,
            model: "Mapika/decider-2b-vision".into(),
            direct_python: None,
            confidence_threshold: 0.1,
            timeout_ms: 2000,
            auto_start: false,
            server_command: None,
            server_model: None,
        };
        let client = SystemOneClient::new(cfg, "python3".into());
        let mut policy = BrowserPolicy::new("Submit form");
        policy.set_plan(GoalPlan {
            requirements: vec![],
            open: None,
            finish: "Form submitted".into(),
        });
        // Pretend a form was typed so submit is expected
        policy.typed = true;

        let snap = BrowserPageSnapshot {
            url: "http://example.com/form".into(),
            title: "Form".into(),
            w: 1280,
            h: 800,
            text: "Form ready".into(),
            actions: vec![BrowserAction {
                id: "click_submit".into(),
                node: Some(30),
                kind: "click".into(),
                role: Some("button".into()),
                label: "Submit".into(),
                value: None,
                current_value: None,
                checked: None,
                expanded: None,
                hint: None,
                rect: None,
                delta: None,
            }],
            marker: serde_json::Value::Null,
            page_key: serde_json::Value::Null,
            guards: Default::default(),
            omitted_actions: 0,
        };

        let before = client.predict_calls();
        let outcome = policy.step(&snap, &client).await.expect("step");
        let after = client.predict_calls();
        // Post-requirements fan-out collapses classify + pick into one call
        assert_eq!(
            after - before,
            1,
            "post step must use exactly 1 predict (fused), got {}",
            after - before
        );
        match outcome {
            crate::browser_policy::PolicyOutcome::Action { action, .. } => {
                assert_eq!(action.kind, "click");
                assert_eq!(action.node, Some(30));
            }
            other => panic!("expected click action, got {other:?}"),
        }
    }

    #[test]
    fn test_speculative_heads_truncation_and_limits() {
        // Verify head_max_len handling: describe truncated to 28, shortlist limit 15
        let mut actions = Vec::new();
        for i in 1..=20 {
            actions.push(crate::browser_cdp::BrowserAction {
                id: format!("e{i}"),
                node: Some(i as i64),
                kind: "click".into(),
                role: Some("button".into()),
                label: format!(
                    "Very long label that exceeds limit and should be truncated {}",
                    i
                ),
                value: None,
                current_value: None,
                checked: None,
                expanded: None,
                hint: Some(
                    "hint with extra words that also should be truncated for head_max_len".into(),
                ),
                rect: None,
                delta: None,
            });
        }
        let snap = crate::browser_cdp::BrowserPageSnapshot {
            url: "http://example.com".into(),
            title: "t".into(),
            w: 1280,
            h: 800,
            text: "".into(),
            actions,
            marker: serde_json::Value::Null,
            page_key: serde_json::Value::Null,
            guards: Default::default(),
            omitted_actions: 0,
        };
        let policy = BrowserPolicy::new("test");
        let observed = policy.observed(&snap);
        // shortlist must cap at 15 (single chunk)
        let short = shortlist(&observed, "label", 15);
        assert_eq!(
            short.len(),
            15,
            "shortlist must respect limit 15 for single-chunk fan-out"
        );
        for e in &short {
            let desc = describe(e);
            let trunc: String = desc.chars().take(28).collect();
            assert!(
                trunc.chars().count() <= 28,
                "truncated describe must be <=28 for head_max_len"
            );
        }
    }
}
