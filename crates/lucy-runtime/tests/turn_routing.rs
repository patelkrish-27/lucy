//! Routing-pipeline checks that do not need a live `decider-serve` or LLM:
//! the tier→model mapping rules, the Manual-mode short circuit, the
//! classifier-down fallback, and the plan parsing the sequential executor
//! consumes.

use lucy_config::{ChatMode, LucyConfig, ProviderConfig, ProviderType, ReasoningLevel};
use lucy_runtime::{
    TurnBranch, describe_model, heuristic_turn_classification, level_model_map, parse_plan,
    render_plan_prompt, text_model_keys,
};
use serde_json::json;

fn config_with_tiers() -> LucyConfig {
    let mut cfg = LucyConfig {
        providers: vec![
            ProviderConfig {
                id: "groq".into(),
                name: "Groq".into(),
                api_url: "https://api.groq.test".into(),
                api_key: "sk-x".into(),
                provider_type: ProviderType::Text,
                available_models: vec!["flash-lite".into(), "flash".into(), "pro".into()],
                deprecated_models: Vec::new(),
            },
            ProviderConfig {
                id: "eleven".into(),
                name: "ElevenLabs".into(),
                api_url: "https://api.eleven.test".into(),
                api_key: "sk-y".into(),
                provider_type: ProviderType::Voice,
                available_models: vec!["tts".into()],
                deprecated_models: Vec::new(),
            },
        ],
        ..Default::default()
    };
    cfg.chat
        .reasoning_levels
        .set(ReasoningLevel::L1, "groq/flash-lite".into());
    cfg.chat
        .reasoning_levels
        .set(ReasoningLevel::L2, "groq/flash".into());
    cfg.chat
        .reasoning_levels
        .set(ReasoningLevel::L3, "groq/pro".into());
    cfg.set_voice_model("eleven/tts");
    cfg
}

#[test]
fn three_distinct_tiers_map_to_distinct_models() {
    let cfg = config_with_tiers();
    let map = level_model_map(&cfg);
    assert_eq!(map[&ReasoningLevel::L1], "groq/flash-lite");
    assert_eq!(map[&ReasoningLevel::L2], "groq/flash");
    assert_eq!(map[&ReasoningLevel::L3], "groq/pro");
    let mut unique: Vec<&String> = map.values().collect();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), 3, "each tier needs its own model");
}

#[test]
fn an_unbound_tier_degrades_to_the_next_cheaper_tier() {
    let mut cfg = config_with_tiers();
    // Clearing Level 3 (the planner tier) skips *down* to L2, never up, so
    // actions are always planned by a model and never escalated.
    cfg.chat
        .reasoning_levels
        .set(ReasoningLevel::L3, String::new());
    assert_eq!(cfg.resolve_level_model(ReasoningLevel::L3), "groq/flash");
    assert_eq!(
        cfg.level_fallback_note(ReasoningLevel::L3).as_deref(),
        Some("no model bound to L3 — using L2")
    );

    // Clearing L2 as well leaves L1 as the deepest bound tier, so L3 lands there.
    cfg.chat
        .reasoning_levels
        .set(ReasoningLevel::L2, String::new());
    assert_eq!(
        cfg.resolve_level_model(ReasoningLevel::L3),
        "groq/flash-lite"
    );

    // L1 is the cheapest tier: with nothing cheaper bound it falls to the
    // compiled-in default rather than climbing to a deeper model.
    cfg.chat
        .reasoning_levels
        .set(ReasoningLevel::L1, String::new());
    assert_eq!(
        cfg.resolve_level_model(ReasoningLevel::L1),
        lucy_config::DEFAULT_TEXT_MODEL
    );
    assert!(
        cfg.level_fallback_note(ReasoningLevel::L1)
            .is_some_and(|n| n.contains("default")),
        "an unbound L1 must report that it left the configured providers"
    );
}

#[test]
fn a_legacy_config_with_only_a_main_model_still_answers() {
    // An old config.toml, before the tiers existed. Loading it must leave Lucy
    // with a working model rather than an empty one.
    let raw = r#"
[chat]
main_model = "groq/flash"
"#;
    let mut cfg: LucyConfig = toml::from_str(raw).expect("legacy config parses");
    let legacy = cfg.chat.legacy_main_model.trim().to_owned();
    assert_eq!(legacy, "groq/flash");
    // The migration runs at load; mimic that for a hand-built config.
    cfg.set_anchor_model(&legacy);
    assert_eq!(cfg.resolve_level_model(ReasoningLevel::L3), "groq/flash");
    // With no tier bound, every tier resolves to the compiled-in default
    // instead of erroring.
    let empty = LucyConfig::default();
    for level in ReasoningLevel::ALL {
        assert_eq!(
            empty.resolve_level_model(level),
            lucy_config::DEFAULT_TEXT_MODEL
        );
    }
}

#[test]
fn every_tier_may_use_any_text_model_from_any_text_provider() {
    let mut cfg = config_with_tiers();
    let mut extra = cfg.providers[0].clone();
    extra.id = "openai".into();
    extra.name = "OpenAI".into();
    extra.available_models = vec!["gpt-5".into()];
    cfg.providers.push(extra);

    for level in ReasoningLevel::ALL {
        cfg.chat.reasoning_levels.set(level, "openai/gpt-5".into());
        assert_eq!(cfg.resolve_level_model(level), "openai/gpt-5");
    }
    // A voice model is never offered as a reasoning tier.
    let text = text_model_keys(&cfg);
    assert!(!text.iter().any(|k| k.contains("eleven")), "{text:?}");
}

#[test]
fn manual_chat_mode_is_a_real_setting() {
    let mut cfg = config_with_tiers();
    assert_eq!(cfg.chat_mode(), ChatMode::Auto);
    cfg.set_chat_mode(ChatMode::Manual);
    assert_eq!(cfg.chat_mode(), ChatMode::Manual);
    // `Manual` pins every turn to the Level 3 anchor.
    assert_eq!(cfg.resolve_level_model(ReasoningLevel::L3), "groq/pro");
    assert_eq!(cfg.voice_model(), "eleven/tts");
}

#[test]
fn auto_compact_is_a_persisted_toggle() {
    let mut cfg = config_with_tiers();
    assert!(cfg.auto_compact());
    cfg.set_auto_compact(false);
    assert!(!cfg.auto_compact());
    assert!(
        !cfg.general.compact_after_command,
        "the legacy mirror agrees"
    );
    let round = toml::from_str::<LucyConfig>(&toml::to_string_pretty(&cfg).expect("toml"))
        .expect("config roundtrip");
    assert!(!round.auto_compact());
}

#[test]
fn removing_a_provider_drops_it_from_every_tier_and_dropdown() {
    let mut cfg = config_with_tiers();
    assert_eq!(cfg.text_model_options().len(), 3);
    assert!(cfg.remove_provider("groq"));
    assert!(cfg.text_model_options().is_empty());
    for level in ReasoningLevel::ALL {
        assert!(
            !cfg.level_model(level).contains("groq/"),
            "{} still points at the removed provider",
            level.short()
        );
    }
    // The voice provider is untouched.
    assert_eq!(cfg.voice_model(), "eleven/tts");
}

#[test]
fn describe_model_labels_every_selection() {
    let cfg = config_with_tiers();
    assert_eq!(describe_model(&cfg, "groq/pro"), "Groq · pro");
    assert_eq!(describe_model(&cfg, "eleven/tts"), "ElevenLabs · tts");
    assert_eq!(describe_model(&cfg, ""), "(no model selected)");
}

#[test]
fn classifier_down_falls_back_instead_of_failing() {
    // Heuristic routing keeps the two branches distinguishable offline.
    for cmd in [
        "open yt & play this song",
        "launch the browser",
        "click the play button",
        "run the test suite",
    ] {
        let c = heuristic_turn_classification(cmd);
        assert_eq!(c.branch, TurnBranch::RequiresActions, "{cmd}");
        assert_eq!(c.reasoning_level, ReasoningLevel::L3, "{cmd}");
        assert_eq!(c.confidence, 0.0, "heuristics claim no confidence");
    }
    for cmd in [
        "what is a monad?",
        "explain rust ownership",
        "why is the sky blue?",
    ] {
        let c = heuristic_turn_classification(cmd);
        assert_eq!(c.branch, TurnBranch::RequiresOnlyResponse, "{cmd}");
        assert_ne!(c.reasoning_level, ReasoningLevel::L3, "{cmd}");
    }
}

#[test]
fn a_plan_drives_sequential_hyprfast_execution() {
    // The exact shape the Level 3 planner is asked for.
    let plan = parse_plan(&json!({"commands": [
        {"tool": "browser_open", "input": {"url": "https://youtube.com"}, "description": "Open YouTube"},
        {"tool": "browser_navigate", "input": {"url": "https://youtube.com/results?search_query=despacito"}, "description": "Search for the song"},
        {"tool": "hint_act", "input": {"instruction": "click the first result"}},
        {"tool": "hint_snapshot", "input": {}}
    ]}));
    assert_eq!(plan.len(), 4);
    // Order is preserved: the executor runs these one by one.
    assert_eq!(plan[0].tool, "browser_open");
    assert_eq!(
        plan[1].input["url"],
        "https://youtube.com/results?search_query=despacito"
    );
    assert_eq!(plan[2].tool, "hint_act");
    assert_eq!(plan[3].tool, "hint_snapshot");
    // Indices are 1-based and dense so the progress line reads "Step 3/4".
    for (i, step) in plan.iter().enumerate() {
        assert_eq!(step.index, i + 1);
    }
    assert!(lucy_runtime::format_plan(&plan).contains("1. Open YouTube (browser_open)"));
}

#[test]
fn planner_prompt_names_the_catalog_and_the_goal() {
    let prompt = render_plan_prompt(
        "browser_navigate — Navigate the browser\nhint_act — Natural-language action",
        "open yt & play this song",
        "(none)",
    );
    assert!(prompt.contains("browser_navigate"));
    assert!(prompt.contains("hint_act"));
    assert!(prompt.contains("open yt & play this song"));
    // The JSON contract is spelled out so the reply is parseable.
    assert!(prompt.contains(r#"{"commands""#), "{prompt}");
    assert!(prompt.contains("strictly ordered"));
}

#[test]
fn a_planner_reply_with_nothing_usable_degrades_to_the_subtask_loop() {
    // Every shape the caller must survive: no list, empty list, and entries
    // with no tool name.
    for value in [
        json!({}),
        json!({"commands": []}),
        json!({"commands": [{"description": "no tool here"}]}),
        json!({"commands": [{"tool": "   "}]}),
        json!("plain prose with no json at all"),
    ] {
        assert!(
            parse_plan(&value).is_empty(),
            "expected an empty plan for {value}"
        );
    }
}
