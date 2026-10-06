//! Integration tests for the Spaces library.
//!
//! These drive the public surface end to end against a real SQLite file in a
//! throwaway directory, because the parts most likely to break are the parts a
//! unit test cannot see: the FTS5 triggers keeping `pages_fts` in step with
//! `pages`, and the `space_id` bookkeeping on a move.

use lucy_spaces::{PageSearch, PageStore};

mod util;
use util::fresh_store;

#[tokio::test]
async fn a_page_saved_in_one_store_is_visible_to_a_search_opened_on_it() {
    let (dir, store) = fresh_store("search-visibility").await;
    let search = PageSearch::new(&store);

    store
        .create_space("Docs")
        .await
        .expect("creating space");
    let id = store
        .create_page("Docs", "Rust Concurrency")
        .await
        .expect("creating page");
    store
        .save_page(id.clone(), "Borrow checkers and ownership")
        .await
        .expect("saving");

    // The index must reflect the *saved* content, not the empty page created
    // moments earlier: a stale FTS row here is a page that saved successfully
    // and can then never be found again.
    let hits = search.search("ownership").await.expect("searching");
    assert_eq!(hits.len(), 1, "the saved body must be searchable");
    assert_eq!(hits[0].page.id.0, id.0);

    // And a term that is nowhere in the page must not match. Multi-word
    // queries are OR'd by FTS5, so this has to be a genuinely absent word:
    // searching "borrow checkers" here would match, correctly.
    let stale = search.search("helicopter").await.expect("searching");
    assert!(stale.is_empty(), "unrelated terms must not match");

    let _ = tokio::fs::remove_dir_all(&dir).await;
}

#[tokio::test]
async fn a_title_match_is_ranked_above_a_body_match() {
    let (dir, store) = fresh_store("ranking").await;
    let search = PageSearch::new(&store);

    store.create_space("Docs").await.expect("creating space");
    // Body-only hit, created first so insertion order cannot explain the order.
    let body = store
        .create_page("Docs", "Untitled One")
        .await
        .expect("creating page");
    store
        .save_page(body.clone(), "a mention of migration in the prose")
        .await
        .expect("saving");
    // Title hit.
    let title = store
        .create_page("Docs", "Migration Notes")
        .await
        .expect("creating page");
    store.save_page(title.clone(), "nothing relevant").await.expect("saving");

    let hits = search.search("migration").await.expect("searching");
    assert_eq!(hits.len(), 2, "both pages match: {hits:?}");
    assert_eq!(
        hits[0].page.id.0, title.0,
        "a title match outranks a body match"
    );
    assert!(hits[0].relevance > hits[1].relevance);

    let _ = tokio::fs::remove_dir_all(&dir).await;
}

#[tokio::test]
async fn moving_a_page_updates_both_spaces_page_counts() {
    let (dir, store) = fresh_store("move-counts").await;

    for name in ["Drafts", "Published"] {
        store.create_space(name).await.expect("creating space");
    }
    let id = store
        .create_page("Drafts", "Essay")
        .await
        .expect("creating page");

    let before = store.get_space("Drafts").await.expect("get").expect("space");
    assert_eq!(before.page_count, 1, "the page is counted where it was made");

    store
        .move_page(id.clone(), "Published")
        .await
        .expect("moving");

    let drafts = store
        .get_space("Drafts")
        .await
        .expect("get")
        .expect("space");
    let published = store
        .get_space("Published")
        .await
        .expect("get")
        .expect("space");
    assert_eq!(drafts.page_count, 0, "the old space no longer holds it");
    assert_eq!(published.page_count, 1, "the new space holds it now");

    let _ = tokio::fs::remove_dir_all(&dir).await;
}

#[tokio::test]
async fn a_reopened_store_sees_the_pages_the_previous_one_wrote() {
    let (dir, store) = fresh_store("persistence").await;

    store.create_space("Notes").await.expect("creating space");
    let id = store
        .create_page("Notes", "Persisted")
        .await
        .expect("creating page");
    store
        .save_page(id.clone(), "written before the process ended")
        .await
        .expect("saving");

    // Drop the first handle and open a second on the same directory: this is
    // what a restart looks like.
    let reopened = PageStore::open(&dir).await.expect("reopening");
    let page = reopened
        .get_page(id)
        .await
        .expect("getting")
        .expect("page survived the reopen");
    assert_eq!(page.title, "Persisted");
    assert_eq!(page.content, "written before the process ended");

    let _ = tokio::fs::remove_dir_all(&dir).await;
}

#[tokio::test]
async fn tag_and_recent_filters_return_only_what_they_name() {
    let (dir, store) = fresh_store("filters").await;
    let search = PageSearch::new(&store);

    store.create_space("Docs").await.expect("creating space");
    let a = store
        .create_page("Docs", "Tagged One")
        .await
        .expect("creating page");
    for tag in ["rust", "concurrency"] {
        store
            .add_tag(a.clone(), tag)
            .await
            .expect("tagging");
    }
    let b = store
        .create_page("Docs", "Untagged")
        .await
        .expect("creating page");
    store.save_page(b.clone(), "plain").await.expect("saving");

    let tagged = search.search_by_tag("rust").await.expect("searching");
    assert_eq!(tagged.len(), 1, "only the tagged page: {tagged:?}");
    assert_eq!(tagged[0].id.0, a.0);

    let recent = search.recent(10).await.expect("recents");
    assert_eq!(recent.len(), 2, "both pages are recent");

    let _ = tokio::fs::remove_dir_all(&dir).await;
}
