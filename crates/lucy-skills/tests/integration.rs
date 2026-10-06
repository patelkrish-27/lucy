//! Integration tests for the self-improving skills system.

use lucy_skills::creator::{create_skills_from_patterns, PatternDetector};
use lucy_skills::improver::SkillImprover;
use lucy_skills::{SkillChange, SkillInfo, SkillStore, TaskPattern};
use std::path::PathBuf;

/// Unique temp directory per test.
fn temp_root() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("lucy-skills-test-{}", uuid::Uuid::new_v4()));
    dir
}

async fn open_temp_store() -> (SkillStore, PathBuf) {
    let root = temp_root();
    let store = SkillStore::open(&root).await.expect("open store");
    (store, root)
}

fn pattern(description: &str, successes: u32, failures: u32) -> TaskPattern {
    TaskPattern {
        description: description.to_string(),
        steps: vec!["prepare it".to_string(), "do it".to_string(), "verify it".to_string()],
        success_count: successes,
        failure_count: failures,
    }
}

#[tokio::test]
async fn open_creates_skills_directory() {
    let (store, root) = open_temp_store().await;
    assert!(store.skills_dir().is_dir());
    assert!(store.db_path().ends_with("lucy-skills.db"));
    let _ = tokio::fs::remove_dir_all(&root).await;
}

#[tokio::test]
async fn open_loads_existing_state_from_disk() {
    let root = temp_root();
    {
        let store = SkillStore::open(&root).await.unwrap();
        store.record_outcome("some-skill", true, "ctx").await.unwrap();
        store
            .record_outcome("some-skill", false, "ctx")
            .await
            .unwrap();
    }
    // Reopen — outcomes should persist.
    let store = SkillStore::open(&root).await.unwrap();
    let skills = store.list_skills().await;
    assert!(skills.is_empty()); // no skill files yet, but outcomes file exists
    let outcomes_path = root.join("lucy-skills.outcomes.json");
    assert!(outcomes_path.exists());
    let _ = tokio::fs::remove_dir_all(&root).await;
}

#[tokio::test]
async fn record_outcome_updates_skill_stats() {
    let (store, root) = open_temp_store().await;

    // Put a skill on disk so list_skills picks it up.
    let skill_path = store.skills_dir().join("demo.md");
    tokio::fs::write(
        &skill_path,
        "---\nname: demo\ndescription: A demo skill\n---\n\n# Demo\n",
    )
    .await
    .unwrap();

    store.record_outcome("demo", true, "a").await.unwrap();
    store.record_outcome("demo", true, "b").await.unwrap();
    store.record_outcome("demo", false, "c").await.unwrap();

    let skills = store.list_skills().await;
    assert_eq!(skills.len(), 1);
    assert_eq!(skills[0].usage_count, 3);
    assert!((skills[0].success_rate - 2.0 / 3.0).abs() < 1e-9);

    let got = store.get_skill("demo").await.unwrap();
    assert_eq!(got.description, "A demo skill");
    assert!(store.get_skill("missing").await.is_none());

    let _ = tokio::fs::remove_dir_all(&root).await;
}

#[tokio::test]
async fn maybe_create_skill_requires_three_occurrences() {
    let (store, root) = open_temp_store().await;

    let too_few = pattern("Summarize a document", 1, 1);
    assert!(store.maybe_create_skill(&too_few).await.unwrap().is_none());

    let enough = pattern("Summarize a document", 2, 1);
    let created = store.maybe_create_skill(&enough).await.unwrap();
    assert!(created.is_some());
    let info = created.unwrap();
    assert_eq!(info.name, "summarize-a-document");
    assert!(info.usage_count == 3);

    // File must exist on disk and be re-discoverable.
    let on_disk = store.skills_dir().join("summarize-a-document.md");
    assert!(on_disk.exists());
    let content = tokio::fs::read_to_string(&on_disk).await.unwrap();
    assert!(content.starts_with("---\n"));
    assert!(content.contains("description:"));

    let _ = tokio::fs::remove_dir_all(&root).await;
}

#[tokio::test]
async fn maybe_create_skill_skips_duplicates() {
    let (store, root) = open_temp_store().await;

    let p = pattern("Archive old invoices", 3, 0);
    assert!(store.maybe_create_skill(&p).await.unwrap().is_some());

    // Same pattern again → no second skill.
    let p2 = pattern("Archive old invoices", 4, 1);
    assert!(store.maybe_create_skill(&p2).await.unwrap().is_none());

    let skills = store.list_skills().await;
    assert_eq!(skills.len(), 1);

    let _ = tokio::fs::remove_dir_all(&root).await;
}

#[tokio::test]
async fn creator_builds_skills_from_detected_patterns() {
    let (store, root) = open_temp_store().await;
    let detector = PatternDetector::new(0.3);

    let tasks = vec![
        ("rename a file safely".to_string(), true),
        ("rename a file safely".to_string(), true),
        ("rename a file safely".to_string(), false),
        ("one off thing".to_string(), true),
    ];

    let patterns = detector.detect_patterns(&tasks);
    assert_eq!(patterns.len(), 1);

    let created = create_skills_from_patterns(&store, &patterns).await.unwrap();
    assert_eq!(created.len(), 1);
    assert_eq!(created[0].usage_count, 3);

    let _ = tokio::fs::remove_dir_all(&root).await;
}

#[tokio::test]
async fn improve_skills_flags_low_success_rate() {
    let (store, root) = open_temp_store().await;

    let p = pattern("Transcribe a meeting", 1, 5);
    let info = store.maybe_create_skill(&p).await.unwrap().unwrap();

    // Seed matching outcome history.
    for _ in 0..1 {
        store.record_outcome(&info.name, true, "ok").await.unwrap();
    }
    for _ in 0..5 {
        store
            .record_outcome(&info.name, false, "failed")
            .await
            .unwrap();
    }

    let improvements = store.improve_skills().await.unwrap();
    assert_eq!(improvements.len(), 1);
    assert_eq!(improvements[0].skill_name, info.name);
    assert!(matches!(improvements[0].change, SkillChange::AddedStep(_)));
    assert!(improvements[0].reason.contains("Low success rate"));

    let _ = tokio::fs::remove_dir_all(&root).await;
}

#[tokio::test]
async fn improve_skills_ignores_healthy_skills() {
    let (store, root) = open_temp_store().await;

    let p = pattern("Compress images", 6, 0);
    let info = store.maybe_create_skill(&p).await.unwrap().unwrap();
    for _ in 0..6 {
        store.record_outcome(&info.name, true, "ok").await.unwrap();
    }

    let improvements = store.improve_skills().await.unwrap();
    assert!(improvements.is_empty());

    let _ = tokio::fs::remove_dir_all(&root).await;
}

#[tokio::test]
async fn improver_analyze_matches_store_outcome_math() {
    let (store, root) = open_temp_store().await;

    let p = pattern("Fetch a webpage", 2, 8);
    let info = store.maybe_create_skill(&p).await.unwrap().unwrap();
    for _ in 0..2 {
        store.record_outcome(&info.name, true, "ok").await.unwrap();
    }
    for _ in 0..8 {
        store
            .record_outcome(&info.name, false, "err")
            .await
            .unwrap();
    }

    let improver = SkillImprover::new();
    let improvements = improver.analyze(&store).await.unwrap();
    assert_eq!(improvements.len(), 1);
    match &improvements[0].change {
        SkillChange::AddedStep(step) => assert!(step.contains("fail")),
        other => panic!("expected AddedStep, got {:?}", other),
    }

    let _ = tokio::fs::remove_dir_all(&root).await;
}

#[tokio::test]
async fn apply_improvement_updates_skill_file() {
    let (store, root) = open_temp_store().await;

    let p = pattern("Index a directory", 1, 5);
    let info = store.maybe_create_skill(&p).await.unwrap().unwrap();
    for _ in 0..1 {
        store.record_outcome(&info.name, true, "ok").await.unwrap();
    }
    for _ in 0..5 {
        store
            .record_outcome(&info.name, false, "err")
            .await
            .unwrap();
    }

    let improver = SkillImprover::new();
    let improvements = improver.analyze(&store).await.unwrap();
    improver
        .apply_improvement(&store, &improvements[0])
        .await
        .unwrap();

    let path = store.skills_dir().join(format!("{}.md", info.name));
    let content = tokio::fs::read_to_string(&path).await.unwrap();
    assert!(content.contains("alternative approach"));

    let _ = tokio::fs::remove_dir_all(&root).await;
}

#[tokio::test]
async fn outcome_contexts_are_persisted_capped_and_reloadable() {
    let root = temp_root();
    {
        let store = SkillStore::open(&root).await.unwrap();
        for i in 0..120 {
            store
                .record_outcome("ctx-skill", true, &format!("ctx-{i}"))
                .await
                .unwrap();
        }
    }
    // The outcomes file should exist and parse back as valid JSON.
    let data = tokio::fs::read_to_string(root.join("lucy-skills.outcomes.json"))
        .await
        .unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&data).unwrap();
    let contexts = parsed["ctx-skill"]["contexts"].as_array().unwrap();
    assert_eq!(contexts.len(), 100); // capped at last 100
    let _ = tokio::fs::remove_dir_all(&root).await;
}

#[test]
fn skill_info_serializes_roundtrip() {
    let info = SkillInfo {
        name: "x".into(),
        description: "y".into(),
        body: "z".into(),
        success_rate: 0.5,
        usage_count: 4,
    };
    let json = serde_json::to_string(&info).unwrap();
    let back: SkillInfo = serde_json::from_str(&json).unwrap();
    assert_eq!(back.name, info.name);
    assert_eq!(back.usage_count, 4);
}
