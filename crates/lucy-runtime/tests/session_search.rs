//! Cross-session search and the user model, end to end over their SQLite
//! stores. Sessions are indexed events; the profile is deterministic
//! extraction over them. Names in these tests are the shape of the fixture,
//! per AGENTS.md — no task or site vocabulary.

use lucy_runtime::{SessionEvent, SessionSearch, UserModel};

fn event(session: &str, author: &str, text: &str, created_at: u64) -> SessionEvent {
    SessionEvent::new(session, author, text).with_created_at(created_at)
}

async fn seeded_search() -> SessionSearch {
    let dir = std::env::temp_dir().join(format!(
        "lucy-search-test-{}-{}",
        std::process::id(),
        uuid_like()
    ));
    let _ = tokio::fs::create_dir_all(&dir).await;
    let search = SessionSearch::new(dir.join("search.db"))
        .await
        .expect("open search db");
    let fixtures = [
        ("sess-alpha", "user", "how do I center a window in Hyprland", 100),
        ("sess-alpha", "lucy", "use the window dispatch center command", 101),
        ("sess-beta", "user", "playlist of ambient focus music", 200),
        ("sess-beta", "lucy", "queued the ambient focus playlist", 201),
        ("sess-gamma", "user", "which sqlite driver does the knowledge store use", 300),
    ];
    for (session, author, text, ts) in fixtures {
        search
            .index_event(&event(session, author, text, ts))
            .await
            .expect("index");
    }
    search
}

fn uuid_like() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

#[tokio::test]
async fn search_returns_sessions_that_match_the_query() {
    let search = seeded_search().await;
    let results = search.search("sqlite", 10).await;
    assert_eq!(results.len(), 1, "only the sqlite question matches: {results:?}");
    assert_eq!(results[0].session_id, "sess-gamma");
    assert!(results[0].snippet.contains("sqlite"), "snippet comes from the hit");
}

#[tokio::test]
async fn search_groups_hits_per_session() {
    let search = seeded_search().await;
    let results = search.search("ambient", 10).await;
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].session_id, "sess-beta");
    assert_eq!(results[0].hits, 2, "both the user turn and the reply count");
    assert_eq!(results[0].last_seen, 201);
}

#[tokio::test]
async fn search_respects_the_limit() {
    let search = seeded_search().await;
    // All three fixtures mention one of "user/lucy" themes; use a broad term.
    let results = search.search("the", 1).await;
    assert_eq!(results.len(), 1, "limit clamps session count");
}

#[tokio::test]
async fn reindexing_the_same_event_does_not_double_count() {
    let search = seeded_search().await;
    let e = event("sess-alpha", "user", "how do I center a window in Hyprland", 100);
    search.index_event(&e).await.expect("reindex");
    let results = search.search("center", 10).await;
    let alpha = results
        .iter()
        .find(|r| r.session_id == "sess-alpha")
        .expect("alpha still matches");
    assert_eq!(alpha.hits, 2, "user + reply hit, not user + reply + duplicate");
}

#[tokio::test]
async fn punctuation_in_a_query_never_breaks_search() {
    let search = seeded_search().await;
    // Raw FTS5 syntax characters would raise without escaping.
    let results = search.search("sqlite (driver) AND: OR \"quoted\"", 10).await;
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].session_id, "sess-gamma");
}

#[test]
fn summarize_results_names_each_match_or_reports_none() {
    let none = SessionSearch::summarize_results(&[]);
    assert_eq!(none, "No past sessions matched.");
}

#[tokio::test]
async fn summarize_results_lists_every_match() {
    let search = seeded_search().await;
    let results = search.search("ambient", 10).await;
    let summary = SessionSearch::summarize_results(&results);
    assert!(summary.contains("sess-beta"), "{summary}");
    assert!(summary.contains("2 hits"), "{summary}");
}

#[tokio::test]
async fn extract_builds_a_profile_without_persisting() {
    let events = vec![
        event("s1", "user", "I prefer concise answers in the terminal", 1),
        event("s1", "lucy", "I prefer concise answers in the terminal", 2),
        event("s1", "user", "I'm working on a Hyprland dotfiles repo", 3),
        event("s1", "user", "Remember that tests run before merges", 4),
    ];
    let profile = UserModel::extract(&events);
    assert_eq!(profile.preferences, vec!["I prefer concise answers in the terminal"]);
    assert_eq!(profile.projects, vec!["I'm working on a Hyprland dotfiles repo"]);
    assert_eq!(profile.instructions, vec!["Remember that tests run before merges"]);
    assert!(profile.facts.is_empty());
}

#[tokio::test]
async fn transient_questions_are_not_extracted() {
    let events = vec![event("s1", "user", "what is the capital of France?", 1)];
    assert!(UserModel::extract(&events).is_empty());
}

#[tokio::test]
async fn update_accumulates_and_profile_roundtrips() {
    let dir = std::env::temp_dir().join(format!(
        "lucy-usermodel-test-{}-{}",
        std::process::id(),
        uuid_like()
    ));
    let model = UserModel::new(dir.join("user.db")).await.expect("open store");
    model
        .update(&event("s1", "user", "I prefer Rust for systems code", 10))
        .await
        .expect("update");
    model
        .update(&event("s1", "user", "I prefer Rust for systems code", 20))
        .await
        .expect("update again");
    model
        .update(&event("s2", "user", "I use nvim with fish", 30))
        .await
        .expect("update");
    let profile = model.profile().await;
    assert_eq!(profile.preferences, vec!["I prefer Rust for systems code"]);
    assert_eq!(profile.facts, vec!["I use nvim with fish"]);
}

#[tokio::test]
async fn context_for_ranks_query_relevant_facts() {
    let dir = std::env::temp_dir().join(format!(
        "lucy-usermodel-ctx-{}-{}",
        std::process::id(),
        uuid_like()
    ));
    let model = UserModel::new(dir.join("user.db")).await.expect("open store");
    model
        .update(&event("s1", "user", "I prefer Rust for systems code", 10))
        .await
        .expect("update");
    model
        .update(&event("s1", "user", "I live in Europe", 20))
        .await
        .expect("update");
    let ctx = model.context_for("which language do I prefer").await;
    assert!(ctx.contains("## About the user"), "{ctx}");
    let rust = ctx.find("Rust").expect("rust fact present");
    let europe = ctx.find("Europe").expect("europe fact present");
    assert!(rust < europe, "the relevant fact ranks first:\n{ctx}");
}

#[tokio::test]
async fn empty_profile_renders_no_context() {
    let model = UserModel::new(":memory:").await.expect("open store");
    assert_eq!(model.context_for("anything").await, "");
}
