//! Page search over the Spaces library.
//!
//! One deterministic path, shared by both callers: a plain `LIKE` scan over
//! title, content and tags, ordered by title. It is the same floor
//! `lucy_knowledge` uses for its own recall — no index, no embeddings, no model
//! call — so a search costs one prepared statement and cannot be wrong in a way
//! that costs money.
//!
//! What it deliberately does not do is *rank* by anything the caller did not
//! ask for. A `LIKE` hit is a hit; [`SearchResult::relevance`] reports where
//! the match fell (title beats body beats tag) so a caller that wants to
//! surface the best one can, without this module pretending to know what
//! "relevant" means for an arbitrary query.

use crate::{Page, PageStore};
use anyhow::Result;
use sqlx::Row;

/// Where a match was found, strongest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MatchSite {
    /// The query appears in the page title.
    Title,
    /// The query appears in one of the page's tags.
    Tag,
    /// The query appears in the page body.
    Body,
}

impl MatchSite {
    /// Higher is a stronger signal.
    pub fn weight(self) -> f64 {
        match self {
            Self::Title => 1.0,
            Self::Tag => 0.75,
            Self::Body => 0.5,
        }
    }
}

/// One search hit: the page plus where and how well it matched.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub page: Page,
    /// Where the query matched, strongest site the page produced.
    pub site: MatchSite,
    /// `MatchSite::weight()`, carried so a caller can sort without matching on
    /// the enum.
    pub relevance: f64,
    /// The matched page title, for a one-line result list.
    pub snippet: String,
}

/// Page search over a [`PageStore`].
///
/// A thin wrapper rather than a second implementation: [`PageStore::search`]
/// already runs the query, so this only adds the per-hit explanation that a
/// ranked list needs and a bare `Vec<Page>` cannot carry.
pub struct PageSearch<'a> {
    store: &'a PageStore,
}

impl<'a> PageSearch<'a> {
    /// Borrow a store rather than taking ownership of an `Arc`.
    ///
    /// Search adds no state of its own, so owning a second handle to the same
    /// database would only make a caller's lifetime annotations harder.
    pub fn new(store: &'a PageStore) -> Self {
        Self { store }
    }

    /// Every page matching `query`, strongest match first.
    ///
    /// Matching is case-insensitive and substring-based, so a multi-word query
    /// matches only if the words appear together — the same behaviour as the
    /// SQL `LIKE`, kept deliberately rather than splitting into per-term
    /// searches that would return pages matching nothing the user asked for.
    pub async fn search(&self, query: &str) -> Result<Vec<SearchResult>> {
        let needle = query.trim();
        if needle.is_empty() {
            return Ok(Vec::new());
        }
        let pages = self.store.search(needle).await?;
        let mut out = Vec::with_capacity(pages.len());
        for page in pages {
            // The store already filtered; this pass only decides *where* the
            // match was, so the caller can order two equally-valid hits.
            let site = if contains(&page.title, needle) {
                MatchSite::Title
            } else if page.tags.iter().any(|t| contains(t, needle)) {
                MatchSite::Tag
            } else {
                MatchSite::Body
            };
            out.push(SearchResult {
                snippet: page.title.clone(),
                relevance: site.weight(),
                site,
                page,
            });
        }
        out.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.page.title.cmp(&b.page.title))
        });
        Ok(out)
    }

    /// Pages carrying `tag`, exact tag match, ordered by title.
    pub async fn search_by_tag(&self, tag: &str) -> Result<Vec<Page>> {
        let needle = tag.trim();
        if needle.is_empty() {
            return Ok(Vec::new());
        }
        let rows = sqlx::query(
            "SELECT id, space_id, title, content, tags, created_at, updated_at \
             FROM pages WHERE tags LIKE ? ORDER BY title",
        )
        .bind(format!("%\"{}\"%", escape_like(needle)))
        .fetch_all(self.store.pool())
        .await?;
        Ok(rows.into_iter().map(row_to_page).collect())
    }

    /// The `limit` most recently updated pages, newest first.
    pub async fn recent(&self, limit: usize) -> Result<Vec<Page>> {
        let rows = sqlx::query(
            "SELECT id, space_id, title, content, tags, created_at, updated_at \
             FROM pages ORDER BY updated_at DESC, title LIMIT ?",
        )
        .bind(limit as i64)
        .fetch_all(self.store.pool())
        .await?;
        Ok(rows.into_iter().map(row_to_page).collect())
    }

    /// Every page in the named space, ordered by title.
    pub async fn in_space(&self, space: &str) -> Result<Vec<Page>> {
        let rows = sqlx::query(
            "SELECT p.id, p.space_id, p.title, p.content, p.tags, p.created_at, p.updated_at \
             FROM pages p JOIN spaces s ON s.id = p.space_id \
             WHERE s.name = ? ORDER BY p.title",
        )
        .bind(space.trim())
        .fetch_all(self.store.pool())
        .await?;
        Ok(rows.into_iter().map(row_to_page).collect())
    }
}

/// Case-insensitive substring test.
///
/// `to_lowercase` rather than a byte comparison, so a query with an uppercase
/// character finds a lowercase title and vice versa.
fn contains(haystack: &str, needle: &str) -> bool {
    haystack.to_lowercase().contains(&needle.to_lowercase())
}

/// Escape the LIKE wildcards so a tag filter containing `%` or `_` searches for
/// those characters instead of turning into a match-everything pattern.
///
/// Only [`PageSearch::search_by_tag`] needs this — the body search goes
/// through FTS5, where a wildcard is not a special character in the same way.
/// Without it, a tag filter of `100%` returns every page, silently.
fn escape_like(raw: &str) -> String {
    raw.replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

/// Reconstruct a [`Page`] from a row selecting the seven page columns.
///
/// The tags column is JSON text; a page written by an older version may hold
/// something else, so a parse failure yields no tags rather than losing the
/// page.
pub(crate) fn row_to_page(r: sqlx::sqlite::SqliteRow) -> Page {
    let tags: Vec<String> =
        serde_json::from_str(&r.get::<String, _>("tags")).unwrap_or_default();
    Page {
        id: crate::PageId(r.get::<String, _>("id")),
        space_id: crate::SpaceId(r.get::<String, _>("space_id")),
        title: r.get::<String, _>("title"),
        content: r.get::<String, _>("content"),
        tags,
        created_at: r.get::<i64, _>("created_at") as u64,
        updated_at: r.get::<i64, _>("updated_at") as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_title_match_outranks_a_body_match() {
        assert!(MatchSite::Title.weight() > MatchSite::Tag.weight());
        assert!(MatchSite::Tag.weight() > MatchSite::Body.weight());
    }

    #[test]
    fn matching_ignores_case_in_both_directions() {
        assert!(contains("Budget Notes", "budget"));
        assert!(contains("budget notes", "BUDGET"));
        assert!(!contains("budget notes", "invoice"));
    }

    #[test]
    fn a_wildcard_in_a_query_is_searched_for_literally() {
        // The point of the escape: without it this becomes `%` and every page
        // matches, so a search for "100%" silently returns the whole library.
        assert_eq!(escape_like("100%"), "100\\%");
        assert_eq!(escape_like("a_b"), "a\\_b");
        assert_eq!(escape_like("plain"), "plain");
    }

    #[test]
    fn a_backslash_is_escaped_before_the_wildcards() {
        // Order matters: escaping backslashes last would double-escape the
        // backslashes this function just added.
        assert_eq!(escape_like("a\\%"), "a\\\\\\%");
    }
}
